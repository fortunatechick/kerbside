"""The analysis session: one source, one pipeline, one calibration.

Everything the API serves comes from here. There is exactly one session per
process because there is exactly one operator in front of the console - the
brief rules out multi-tenancy, and pretending otherwise would add a layer of
indirection to every route for no benefit.

Frames are held in memory. A synthetic scenario is rendered once, up front,
rather than generated on demand: rendering costs about a third of a second a
frame at supersampled quality, which is fine as a one-off and hopeless inside a
frame handler. A video file is decoded once for the same reason - it makes
scrubbing, stepping and the pipeline inspector free, which is what the console
is built around.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kerbside.core.cache import StageCache, cache_key
from kerbside.core.types import Calibration, GroundTruth, GTFrame, UInt8Array
from kerbside.data.renderer import TrafficRenderer
from kerbside.data.scenarios import build_scene, scenario_info, scenario_names
from kerbside.stages.base import FrameContext, StageError
from kerbside.stages.pipeline import FrameOutput, Pipeline
from kerbside.stages.s1_ingest import VideoInfo, VideoSource

__all__ = ["Session", "SessionState", "get_session"]


@dataclass(slots=True)
class SessionState:
    """What the status bar shows."""

    source_name: str = "none"
    source_kind: str = "none"
    width_px: int = 0
    height_px: int = 0
    frame_count: int = 0
    container_fps: float = 0.0
    measured_fps: float = 0.0
    codec: str = ""
    bit_depth: int = 8
    calibrated: bool = False
    has_ground_truth: bool = False
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_name": self.source_name,
            "source_kind": self.source_kind,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "frame_count": self.frame_count,
            "container_fps": self.container_fps,
            "measured_fps": self.measured_fps,
            "codec": self.codec,
            "bit_depth": self.bit_depth,
            "calibrated": self.calibrated,
            "has_ground_truth": self.has_ground_truth,
            "warnings": list(self.warnings),
        }


class Session:
    """One loaded clip, its pipeline, and everything computed from it."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.frames: list[UInt8Array] = []
        self.ground_truth: GroundTruth | None = None
        self.calibration: Calibration | None = None
        self.pipeline = Pipeline()
        self.cache = StageCache()
        self.state = SessionState()
        self.playing = False
        self.play_speed = 1.0
        self.cursor = 0
        self._outputs: dict[int, FrameOutput] = {}
        self._telemetry: list[dict[str, Any]] = []
        self._log: list[dict[str, Any]] = []
        self._analysis_done = False

    # -- sources -----------------------------------------------------------

    def load_scenario(
        self,
        name: str = "free_flow",
        duration_s: float = 12.0,
        seed: int = 4242,
        density: float | None = None,
        speed_kmph_range: tuple[float, float] | None = None,
        shadow_strength: float | None = None,
        width_px: int | None = None,
        height_px: int | None = None,
        supersample: int = 1,
    ) -> SessionState:
        """Render a synthetic scenario and adopt its exact calibration.

        The renderer's own homography is adopted as the starting calibration, so
        the console is usable immediately. The user can still calibrate by
        clicking, and the calibration view compares their result against this
        one - which is the whole point of having a known answer available.
        """
        if name not in scenario_names():
            raise StageError(
                f"unknown scenario {name!r}; available: {', '.join(scenario_names())}"
            )
        scene = build_scene(
            name,
            seed=seed,
            duration_s=duration_s,
            density=density,
            speed_kmph_range=speed_kmph_range,
            shadow_strength=shadow_strength,
            width_px=width_px,
            height_px=height_px,
            supersample=supersample,
        )
        renderer = TrafficRenderer(scene)
        frames: list[UInt8Array] = []
        gts: list[GTFrame] = []
        for frame, gt in renderer.render_clip():
            frames.append(frame)
            gts.append(gt)

        with self._lock:
            self._reset_analysis()
            self.frames = frames
            self.ground_truth = renderer.ground_truth(gts)
            self.calibration = renderer.calibration()
            self.pipeline = Pipeline(
                frame_interval_s=1.0 / scene.fps, calibration=self.calibration
            )
            self.state = SessionState(
                source_name=f"{name} (synthetic, seed {seed})",
                source_kind="synthetic",
                width_px=scene.width_px,
                height_px=scene.height_px,
                frame_count=len(frames),
                container_fps=scene.fps,
                measured_fps=scene.fps,
                codec="raw",
                calibrated=True,
                has_ground_truth=True,
                warnings=[],
            )
            self.cursor = 0
            self._note(
                "source",
                f"loaded scenario {name}: {len(frames)} frames at {scene.fps:g} fps",
            )
        return self.state

    def load_video(self, path: str | Path, max_frames: int = 1200) -> SessionState:
        """Decode a clip from disk.

        Bounded by ``max_frames`` because the frames are held in memory: 1200
        frames of 720p is about 3 GB, which is already generous for a console
        that exists to inspect a stretch of road rather than to archive it.
        """
        source = VideoSource(path)
        frames: list[UInt8Array] = []
        try:
            while len(frames) < max_frames:
                frame = source.read()
                if frame is None:
                    break
                frames.append(frame)
            info: VideoInfo = source.info
        finally:
            source.release()

        if not frames:
            raise StageError(
                f"{path}: opened but produced no frames. The container may be "
                f"empty, or its codec may not be supported by this OpenCV build."
            )

        with self._lock:
            self._reset_analysis()
            self.frames = frames
            self.ground_truth = None
            # No calibration: a real clip has no known geometry until the user
            # supplies one, and inventing a default would produce speeds that
            # look plausible and mean nothing.
            self.calibration = None
            self.pipeline = Pipeline(
                frame_interval_s=info.frame_interval_s, calibration=None
            )
            self.state = SessionState(
                source_name=Path(path).name,
                source_kind="file",
                width_px=info.width_px,
                height_px=info.height_px,
                frame_count=len(frames),
                container_fps=info.container_fps,
                measured_fps=info.measured_fps,
                codec=info.codec,
                bit_depth=info.bit_depth,
                calibrated=False,
                has_ground_truth=False,
                warnings=list(info.warnings())
                + [
                    "no calibration is loaded, so speeds and real-world sizes are "
                    "unavailable until the camera is calibrated"
                ],
            )
            self.cursor = 0
            self._note("source", f"loaded {Path(path).name}: {len(frames)} frames")
        return self.state

    # -- calibration -------------------------------------------------------

    def set_calibration(self, calibration: Calibration) -> None:
        with self._lock:
            self.calibration = calibration
            self.pipeline.calibration = calibration
            self.state.calibrated = calibration.is_valid
            self.state.warnings = [
                w
                for w in self.state.warnings
                if "no calibration is loaded" not in w
            ]
            self._reset_analysis()
            self._note("calibration", f"calibration set from {calibration.source}")

    # -- analysis ----------------------------------------------------------

    #: How many frames keep their full intermediates after an analysis. Each is
    #: about 35 MB at 720p, so this is the difference between 350 MB and several
    #: gigabytes. Playback is sequential, so a small window covers it; the stage
    #: inspector on an older frame triggers a replay instead.
    RETAINED_CONTEXTS = 10

    def _reset_analysis(self) -> None:
        self._outputs = {}
        self._telemetry = []
        self._analysis_done = False
        self._final_count = 0
        self._final_counts_by_lane = {}
        self.cache.clear()

    @property
    def analysed(self) -> bool:
        return self._analysis_done

    def frame(self, index: int) -> UInt8Array:
        if not self.frames:
            raise StageError("no source is loaded; pick a scenario or load a clip")
        return self.frames[int(np.clip(index, 0, len(self.frames) - 1))]

    def analyse(self, force: bool = False) -> dict[str, Any]:
        """Run the whole pipeline over the clip and keep the per-frame results.

        Done as one pass rather than lazily per frame because the stages are
        stateful: the background model, the tracker and the classifier all
        depend on having seen every previous frame, so a random-access API over
        a partially-run clip would return results that depend on what the user
        happened to click first.
        """
        with self._lock:
            if self._analysis_done and not force:
                return self.summary()
            if not self.frames:
                raise StageError("no source is loaded; pick a scenario or load a clip")

            self.pipeline.reset()
            self._outputs = {}
            self._telemetry = []
            started = time.perf_counter()
            recent: deque[int] = deque()
            for index, frame in enumerate(self.frames):
                output = self.pipeline.process_frame(frame, frame_index=index)
                self._outputs[index] = output
                self._telemetry.append(self._telemetry_for(output))
                # Only the most recent few frames keep their intermediates. The
                # telemetry above is already extracted, and it is what every
                # summary, count and speed is computed from; the arrays are only
                # needed by the stage inspector, which looks at one frame.
                recent.append(index)
                while len(recent) > self.RETAINED_CONTEXTS:
                    self._outputs[recent.popleft()].context.release_intermediates()
            elapsed = time.perf_counter() - started
            self._analysis_done = True
            # The tracker's live counters move if the pipeline is ever replayed
            # to recover a released frame's detail. The totals are a property of
            # the completed analysis, so they are frozen here rather than read
            # back from a stage whose state can legitimately be rewound.
            self._final_count = self.pipeline.track.total_count
            self._final_counts_by_lane = dict(self.pipeline.track.counts_by_lane())
            self._note(
                "analyse",
                f"processed {len(self.frames)} frames in {elapsed:.1f} s "
                f"({len(self.frames) / max(elapsed, 1e-6):.1f} fps)",
            )
            return self.summary()

    def output(self, index: int) -> FrameOutput | None:
        return self._outputs.get(int(index))

    def context(self, index: int, *, intermediates: bool = False) -> FrameContext:
        """The frame context for one frame, computing it if necessary.

        The stages are stateful - the background model, the tracker and the
        classifier all depend on having seen every previous frame - so a single
        frame cannot be computed in isolation. When the requested frame has not
        been processed, the whole clip is.

        ``intermediates=True`` additionally guarantees the large per-frame
        arrays are present. Analysis releases them for all but the most recent
        few frames, so the stage inspector asks for them explicitly and the
        clip is replayed up to that frame when they have gone. Callers that only
        need the overlay - boxes, ground points, trails - must leave this False,
        because they would otherwise trigger a replay on every scrub.
        """
        output = self.output(index)
        if output is None:
            self.analyse()
            output = self.output(index)
        if output is None:
            raise StageError(f"frame {index} is outside this clip")
        if intermediates and not output.context.has_intermediates:
            return self._replay_to(index)
        return output.context

    def _replay_to(self, index: int) -> FrameContext:
        """Re-run the pipeline from frame 0 to ``index``, keeping that context.

        Deterministic, so the result is identical to the one the full analysis
        produced - this recovers detail that was released, it does not change
        any number. The cost is linear in ``index``, which is why it happens
        only when a user opens the stage inspector on an old frame.
        """
        with self._lock:
            self.pipeline.reset()
            context: FrameContext | None = None
            for i in range(index + 1):
                output = self.pipeline.process_frame(self.frames[i], frame_index=i)
                context = output.context
                if i < index:
                    context.release_intermediates()
            if context is None:
                raise StageError(f"frame {index} is outside this clip")
            self._outputs[index] = FrameOutput(
                frame_index=index,
                time_s=context.time_s,
                elapsed_ms=self._telemetry[index]["frame_ms"] if index < len(self._telemetry) else 0.0,
                context=context,
            )
            self._note("replay", f"re-ran frames 0-{index} to recover stage detail")
            return context

    def telemetry(self, index: int) -> dict[str, Any]:
        if 0 <= index < len(self._telemetry):
            return self._telemetry[index]
        return {"frame_index": index, "tracks": [], "counts": {}, "stages": {}}

    def _telemetry_for(self, output: FrameOutput) -> dict[str, Any]:
        ctx = output.context
        return {
            "frame_index": output.frame_index,
            "time_s": round(output.time_s, 4),
            "frame_ms": round(output.elapsed_ms, 2),
            "frame_type": ctx.frame_type.value,
            "blobs": len(ctx.blobs),
            "tracks": [snapshot.to_dict() for snapshot in ctx.track_snapshots],
            "counts": dict(self.pipeline.track.counts_by_lane()),
            "total_count": self.pipeline.track.total_count,
            "shake_px": [round(ctx.shake_dx_px, 3), round(ctx.shake_dy_px, 3)],
            "stages": {
                stage_id: {
                    "ms": round(result.elapsed_ms, 2),
                    "metrics": {k: round(float(v), 4) for k, v in result.metrics.items()},
                }
                for stage_id, result in ctx.results.items()
            },
        }

    def summary(self) -> dict[str, Any]:
        timings = self.pipeline.timings.summary()
        return {
            "state": self.state.to_dict(),
            "frames": len(self.frames),
            "analysed": self._analysis_done,
            "total_count": (
                self._final_count if self._analysis_done else self.pipeline.track.total_count
            ),
            "counts_by_lane": (
                self._final_counts_by_lane
                if self._analysis_done
                else self.pipeline.track.counts_by_lane()
            ),
            "timings": {k: round(v, 3) for k, v in timings.items()},
            "slowest_stage": self.pipeline.timings.slowest_stage(),
            "cache": self.cache.stats(),
        }

    # -- console log -------------------------------------------------------

    def _note(self, source: str, message: str, **extra: Any) -> None:
        self._log.append(
            {
                "t": time.time(),
                "source": source,
                "message": message,
                **extra,
            }
        )
        del self._log[:-400]

    def note(self, source: str, message: str, **extra: Any) -> None:
        with self._lock:
            self._note(source, message, **extra)

    def log(self, limit: int = 200) -> list[dict[str, Any]]:
        return self._log[-limit:]

    # -- caching helper ----------------------------------------------------

    def stage_cache_key(self, frame_index: int, stage_id: str) -> str:
        order = [s.stage_id for s in self.pipeline.stages]
        upto = order.index(stage_id) + 1 if stage_id in order else len(order)
        return cache_key(
            self.state.source_name,
            frame_index,
            [self.pipeline.stages[i].params for i in range(upto)],
        )


_SESSION: Session | None = None
_SESSION_LOCK = threading.Lock()


def get_session() -> Session:
    """The process-wide session, created on first use."""
    global _SESSION
    with _SESSION_LOCK:
        if _SESSION is None:
            _SESSION = Session()
        return _SESSION
