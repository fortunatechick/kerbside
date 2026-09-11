"""The pipeline: run the stages in order over a clip.

The orchestration is deliberately thin. Each stage reads what it needs from the
frame context and attaches what it produced, so running a prefix of the stages,
or re-running from stage *k* forward with a changed parameter, is a matter of
slicing a list - which is what the console's live recompute depends on.

Stage timings are collected per frame and summarised, because "which stage costs
the frame budget" is a question the operations console answers continuously and
a question the throughput target depends on.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kerbside.core.types import Calibration, StageResult, UInt8Array
from kerbside.stages.base import FrameContext, Stage, StageError
from kerbside.stages.s1_ingest import IngestStage
from kerbside.stages.s2_stabilise import StabiliseStage
from kerbside.stages.s3_background import BackgroundStage
from kerbside.stages.s4_foreground import ForegroundStage
from kerbside.stages.s5_shadow import ShadowStage
from kerbside.stages.s6_blobs import BlobStage
from kerbside.stages.s7_track import TrackStage
from kerbside.stages.s8_speed import SpeedStage
from kerbside.stages.s9_classify import ClassifyStage

__all__ = ["Pipeline", "FrameOutput", "PipelineTimings"]


@dataclass(slots=True)
class FrameOutput:
    """What the pipeline produced for one frame."""

    frame_index: int
    time_s: float
    context: FrameContext
    elapsed_ms: float

    @property
    def blob_count(self) -> int:
        return len(self.context.blobs)

    @property
    def stage_ms(self) -> dict[str, float]:
        return {sid: r.elapsed_ms for sid, r in self.context.results.items()}


@dataclass(slots=True)
class PipelineTimings:
    """Rolling per-stage timings and the throughput they imply."""

    per_stage_ms: dict[str, list[float]] = field(default_factory=dict)
    total_ms: list[float] = field(default_factory=list)

    def record(self, output: FrameOutput) -> None:
        for stage_id, ms in output.stage_ms.items():
            self.per_stage_ms.setdefault(stage_id, []).append(ms)
        self.total_ms.append(output.elapsed_ms)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for stage_id, values in self.per_stage_ms.items():
            out[f"{stage_id}_ms"] = float(np.median(values)) if values else 0.0
        if self.total_ms:
            median = float(np.median(self.total_ms))
            out["frame_ms"] = median
            # Median rather than mean: one slow frame while a codec seeks should
            # not be reported as the throughput a user will experience.
            out["fps"] = 1000.0 / median if median > 0 else 0.0
        return out

    def slowest_stage(self) -> tuple[str, float] | None:
        if not self.per_stage_ms:
            return None
        medians = {k: float(np.median(v)) for k, v in self.per_stage_ms.items() if v}
        if not medians:
            return None
        stage_id = max(medians, key=lambda k: medians[k])
        return (stage_id, medians[stage_id])


class Pipeline:
    """S1 to S9 over a sequence of frames.

    ``calibration`` is optional but changes what the pipeline can do rather than
    merely how well: without it the perspective size filter and every real-world
    measurement are unavailable, and the stages say so in their results instead
    of quietly producing pixel-space numbers that look like metres.
    """

    def __init__(
        self,
        frame_interval_s: float = 1.0 / 25.0,
        calibration: Calibration | None = None,
        stages: Iterable[Stage] | None = None,
    ) -> None:
        if stages is None:
            self.ingest = IngestStage(frame_interval_s=frame_interval_s)
            self.stabilise = StabiliseStage()
            self.background = BackgroundStage()
            self.foreground = ForegroundStage()
            self.shadow = ShadowStage()
            self.blobs = BlobStage()
            self.track = TrackStage()
            self.speed = SpeedStage()
            self.classify = ClassifyStage()
            self._stages: list[Stage] = [
                self.ingest, self.stabilise, self.background,
                self.foreground, self.shadow, self.blobs,
                self.track, self.speed, self.classify,
            ]
        else:
            self._stages = list(stages)
            for stage in self._stages:
                setattr(self, stage.stage_id.split("_", 1)[1], stage)
        self.calibration = calibration
        self.timings = PipelineTimings()
        self._frame_index = 0
        self._previous_mask = None

    # -- configuration -----------------------------------------------------

    @property
    def stages(self) -> tuple[Stage, ...]:
        return tuple(self._stages)

    def stage(self, stage_id: str) -> Stage:
        for stage in self._stages:
            if stage.stage_id == stage_id:
                return stage
        raise KeyError(
            f"unknown stage {stage_id!r}; the pipeline has "
            f"{', '.join(s.stage_id for s in self._stages)}"
        )

    def configure(self, stage_id: str, **params: Any) -> None:
        """Change a stage's parameters, resetting it and everything after it.

        Downstream state is reset too, because a background model built under
        one set of parameters does not describe the scene under another, and a
        metric delta measured against stale state describes neither setting.
        """
        target = self.stage(stage_id)
        target.set_params(**params)
        seen = False
        for stage in self._stages:
            if stage is target:
                seen = True
                continue
            if seen:
                stage.reset()
        self.timings = PipelineTimings()
        self._frame_index = 0

    def reset(self) -> None:
        for stage in self._stages:
            stage.reset()
        self.timings = PipelineTimings()
        self._frame_index = 0
        self._previous_mask = None

    def set_frame_interval(self, interval_s: float) -> None:
        for stage in self._stages:
            if isinstance(stage, IngestStage):
                stage.set_frame_interval(interval_s)

    def describe(self) -> list[dict[str, Any]]:
        return [stage.describe() for stage in self._stages]

    # -- execution ---------------------------------------------------------

    def process_frame(
        self, bgr: UInt8Array, frame_index: int | None = None, time_s: float | None = None
    ) -> FrameOutput:
        """Run every stage over one frame."""
        index = self._frame_index if frame_index is None else frame_index
        interval = getattr(self.ingest, "frame_interval_s", 1.0 / 25.0)
        ctx = FrameContext(
            frame_index=index,
            time_s=index * interval if time_s is None else time_s,
            bgr=np.asarray(bgr, dtype=np.uint8),
            calibration=self.calibration,
        )
        ctx.previous_mask = self._previous_mask
        started = time.perf_counter()
        for stage in self._stages:
            stage(ctx)
        elapsed = (time.perf_counter() - started) * 1000.0
        self._frame_index = index + 1

        # Carry the settled mask forward: S3's selective update needs the
        # previous frame's answer, and it runs before this frame produces one.
        self._previous_mask = (
            ctx.foreground_no_shadow
            if ctx.foreground_no_shadow is not None
            else ctx.foreground
        )
        background = getattr(self, "background", None)
        if background is not None and hasattr(background, "note_mask"):
            background.note_mask(self._previous_mask)

        output = FrameOutput(
            frame_index=index, time_s=ctx.time_s, context=ctx, elapsed_ms=elapsed
        )
        self.timings.record(output)
        return output

    def run(self, frames: Iterable[UInt8Array]) -> Iterator[FrameOutput]:
        """Run the pipeline over a sequence of frames."""
        self.reset()
        for index, frame in enumerate(frames):
            yield self.process_frame(frame, frame_index=index)

    def run_from(
        self, ctx: FrameContext, stage_id: str
    ) -> dict[str, StageResult]:
        """Re-run one frame from ``stage_id`` forward, reusing what came before.

        This is what makes the pipeline view feel immediate: changing a shadow
        threshold re-runs S5 and S6 on the current frame, not the whole clip
        from S1. It requires the caller to have kept the context, which the
        stage cache does.
        """
        ids = [s.stage_id for s in self._stages]
        if stage_id not in ids:
            raise StageError(
                f"cannot resume from unknown stage {stage_id!r}; the pipeline has "
                f"{', '.join(ids)}"
            )
        out: dict[str, StageResult] = {}
        for stage in self._stages[ids.index(stage_id) :]:
            out[stage.stage_id] = stage(ctx)
        return out

    @property
    def is_warm(self) -> bool:
        """Whether the background model has seen enough frames to be trusted."""
        background = getattr(self, "background", None)
        return bool(getattr(background, "is_warm", True))
