"""S1 - ingest: decode, measure, convert.

Three jobs, and the middle one matters more than it looks.

**Decode and describe.** Resolution, channel count, bit depth, codec and
container frame rate are Unit 1 information about the signal, and the console
shows them because a user debugging a bad result needs to know what they fed in.

**Measure the true frame interval.** Every speed in this system is a distance
divided by a time, and that time comes from here. Container metadata is often
wrong - a file written by one tool, re-wrapped by another, or produced by a
recorder that dropped frames under load will happily claim 30 fps while
delivering 25. A 20% error in the frame interval is a 20% error in every speed,
applied uniformly, with nothing in the output that looks wrong. So the interval
is measured from presentation timestamps and compared against what the container
claims, and a disagreement is reported rather than resolved silently.

**Convert.** Greyscale by ITU-R BT.601 with the weights exposed, because the
luma weights are a modelling choice about how the sensor's channels contribute
and a user comparing colour spaces should be able to change them. HSV is
computed here too, once, since S3 and S5 both need it and converting twice per
frame is a measurable share of the frame budget at 720p.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import FloatArray, StageResult, UInt8Array
from kerbside.stages.base import FrameContext, Stage, StageError

__all__ = ["VideoInfo", "IngestStage", "VideoSource", "BT601_WEIGHTS"]

BT601_WEIGHTS = (0.114, 0.587, 0.299)
"""Luma weights in BGR order: 0.299 R + 0.587 G + 0.114 B.

Green dominates because the eye is most sensitive to it. Using flat 1/3 weights
instead measurably changes which vehicles survive thresholding - a red car
against grey asphalt has a much larger BT.601 contrast than an equal-weight one -
so the weights are a parameter rather than a constant."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """What the container says about a clip, and what it actually delivers.

    The two fps figures are separate fields on purpose. Reconciling them into
    one number would throw away the only evidence that something is wrong.
    """

    path: str
    width_px: int
    height_px: int
    channels: int
    bit_depth: int
    frame_count: int
    container_fps: float
    measured_fps: float
    codec: str
    duration_s: float

    @property
    def fps_disagreement(self) -> float:
        """Relative difference between the claimed and the measured frame rate."""
        if self.container_fps <= 0:
            return float("inf")
        return abs(self.measured_fps - self.container_fps) / self.container_fps

    @property
    def fps_is_trustworthy(self) -> bool:
        """Whether the container's claim can be used without comment.

        Two percent is the line. Below it the difference is timestamp
        quantisation; above it the container is describing a different clip from
        the one it contains, and a speed computed from the claimed rate carries
        that error in full.
        """
        return self.fps_disagreement <= 0.02

    @property
    def frame_interval_s(self) -> float:
        """The interval every speed will be divided by."""
        fps = self.measured_fps if self.measured_fps > 0 else self.container_fps
        if fps <= 0:
            raise StageError(
                f"{self.path}: neither the container nor the timestamps give a "
                f"usable frame rate, so no speed can be computed from this clip"
            )
        return 1.0 / fps

    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.fps_is_trustworthy:
            out.append(
                f"the container claims {self.container_fps:.3f} fps but the "
                f"timestamps measure {self.measured_fps:.3f} fps, a "
                f"{self.fps_disagreement * 100:.1f}% disagreement. Speeds use the "
                f"measured rate; if that is wrong, every speed is wrong by the "
                f"same factor."
            )
        if self.frame_count <= 1:
            out.append("this clip has one frame or fewer - nothing can be tracked")
        if min(self.width_px, self.height_px) < 240:
            out.append(
                f"at {self.width_px}x{self.height_px} a distant vehicle covers "
                f"only a handful of pixels; expect counting to degrade"
            )
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "frame_count": self.frame_count,
            "container_fps": self.container_fps,
            "measured_fps": self.measured_fps,
            "codec": self.codec,
            "duration_s": self.duration_s,
            "fps_disagreement": self.fps_disagreement,
            "fps_is_trustworthy": self.fps_is_trustworthy,
            "warnings": list(self.warnings()),
        }


class VideoSource:
    """A decoded clip, described honestly.

    Opening the file also measures its real frame interval, which needs a short
    pass over the presentation timestamps. That pass is done once, at open, over
    a bounded number of frames - enough to average out quantisation, cheap
    enough not to delay the first frame.
    """

    def __init__(self, path: str | Path, probe_frames: int = 90) -> None:
        self.path = str(path)
        self._capture = cv2.VideoCapture(self.path)
        if not self._capture.isOpened():
            raise StageError(
                f"could not open {self.path} for decoding. Check the file exists "
                f"and that OpenCV was built with a decoder for its codec."
            )
        self.info = self._probe(probe_frames)
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _probe(self, probe_frames: int) -> VideoInfo:
        cap = self._capture
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        container_fps = float(cap.get(cv2.CAP_PROP_FPS))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = (
            "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4)).strip()
            if fourcc
            else "unknown"
        )

        # Measure the interval from presentation timestamps. The median of the
        # per-frame differences rather than the mean, because a single long gap
        # - a dropped frame, or a keyframe the decoder took longer over - would
        # drag a mean and would then be applied to every speed in the clip.
        timestamps_ms: list[float] = []
        for _ in range(max(2, probe_frames)):
            ok, _frame = cap.read()
            if not ok:
                break
            timestamps_ms.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))

        measured_fps = 0.0
        if len(timestamps_ms) >= 3:
            deltas = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
            deltas = deltas[deltas > 1e-6]
            if deltas.size:
                measured_fps = float(1000.0 / np.median(deltas))
        if measured_fps <= 0:
            # Some containers report no timestamps at all. Falling back to the
            # container rate is the only option, and the caller is told the
            # measurement was unavailable rather than being given a fabricated
            # agreement.
            measured_fps = container_fps

        ok, frame = cap.read() if frame_count else (False, None)
        if frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        channels = int(frame.shape[2]) if ok and frame is not None and frame.ndim == 3 else 1
        bit_depth = 8 if ok and frame is not None and frame.dtype == np.uint8 else 16

        duration = frame_count / measured_fps if measured_fps > 0 else 0.0
        return VideoInfo(
            path=self.path,
            width_px=width,
            height_px=height,
            channels=channels,
            bit_depth=bit_depth,
            frame_count=frame_count,
            container_fps=container_fps,
            measured_fps=measured_fps,
            codec=codec,
            duration_s=duration,
        )

    def read(self) -> UInt8Array | None:
        ok, frame = self._capture.read()
        return np.asarray(frame, dtype=np.uint8) if ok else None

    def seek(self, frame_index: int) -> None:
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> VideoSource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


class IngestStage(Stage):
    """S1: colour conversion and the frame interval every speed divides by."""

    stage_id = "s1_ingest"
    name = "Ingest"
    units = (1, 2)
    rationale = (
        "Converts each frame to greyscale by ITU-R BT.601 and to HSV once for "
        "the stages that need it, and supplies the frame interval. The interval "
        "is measured from presentation timestamps rather than taken from the "
        "container, because every speed in the system is divided by it and a "
        "container that lies produces confidently wrong speeds with nothing in "
        "the output that looks wrong."
    )
    param_specs = (
        ParamSpec(
            "weight_b", "Blue weight", "float", BT601_WEIGHTS[0], 0.0, 1.0, 0.001,
            description="BT.601 luma weight for the blue channel.",
        ),
        ParamSpec("weight_g", "Green weight", "float", BT601_WEIGHTS[1], 0.0, 1.0, 0.001),
        ParamSpec("weight_r", "Red weight", "float", BT601_WEIGHTS[2], 0.0, 1.0, 0.001),
        ParamSpec(
            "normalise_weights", "Normalise weights", "bool", True,
            description=(
                "Rescale the three weights to sum to one, so changing one does "
                "not also change overall brightness."
            ),
        ),
    )

    def __init__(self, frame_interval_s: float = 1.0 / 25.0, **params: Any) -> None:
        self._frame_interval_s = frame_interval_s
        super().__init__(**params)

    @property
    def frame_interval_s(self) -> float:
        return self._frame_interval_s

    def set_frame_interval(self, interval_s: float) -> None:
        if interval_s <= 0:
            raise ValueError(f"frame interval must be positive, got {interval_s} s")
        self._frame_interval_s = interval_s

    def to_grey(self, bgr: UInt8Array) -> FloatArray:
        """Weighted luma, as a float array in 0-255."""
        w = np.array(
            [self.param("weight_b"), self.param("weight_g"), self.param("weight_r")],
            dtype=np.float32,
        )
        total = float(w.sum())
        if total <= 1e-9:
            raise StageError(
                "the three luma weights sum to zero, which would make every "
                "frame black; set at least one above zero"
            )
        if self.param("normalise_weights"):
            w = w / total
        # float32 throughout: the input is eight bits, and on a 720p frame the
        # narrower type halves the memory traffic for every stage downstream.
        return np.asarray(np.tensordot(bgr.astype(np.float32), w, axes=([2], [0])))

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            if ctx.bgr.ndim != 3 or ctx.bgr.shape[2] != 3:
                raise StageError(
                    f"S1 expects a 3-channel BGR frame, got shape {ctx.bgr.shape}"
                )
            ctx.grey = self.to_grey(ctx.bgr)
            hsv = cv2.cvtColor(ctx.bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] /= 255.0
            hsv[..., 2] /= 255.0
            ctx.hsv = hsv
            ctx.dt_s = self._frame_interval_s

            result.image = np.asarray(np.clip(ctx.grey, 0, 255), dtype=np.uint8)
            result.metrics = {
                "width_px": float(ctx.bgr.shape[1]),
                "height_px": float(ctx.bgr.shape[0]),
                "frame_interval_ms": self._frame_interval_s * 1000.0,
                "mean_luma": float(ctx.grey.mean()),
                "luma_p1": float(np.percentile(ctx.grey, 1)),
                "luma_p99": float(np.percentile(ctx.grey, 99)),
            }
            # Usable dynamic range, which is what decides whether the difference
            # image downstream will have anything for Otsu to separate.
            span = result.metrics["luma_p99"] - result.metrics["luma_p1"]
            result.metrics["dynamic_range"] = span
            if span < 40.0:
                result.extras["hint"] = (
                    f"only {span:.0f} grey levels of usable range; consider "
                    f"contrast stretching or histogram equalisation before "
                    f"background differencing"
                )
        return result
