"""S3 - background modelling, written by hand.

The scene is mostly road. If you knew what the empty road looked like, anything
differing from it would be a vehicle. Building that reference from the video
itself is background modelling, and this module implements three ways to do it
with NumPy - which is the point of the exercise. OpenCV's MOG2 and KNN appear
only in the comparison study, clearly labelled, never on the production path.

**Running average** is the whole of Unit 2 applied along time instead of space:

    B <- (1 - alpha) B + alpha I

is a first-order recursive low-pass filter on each pixel's intensity over time.
Its cutoff is set by ``alpha``, its impulse response is exponential with time
constant ``1/alpha`` frames, and calling it a temporal low-pass filter is not an
analogy - it is the same filter, on the same footing as a smoothing kernel, with
time as the axis. Anything that changes faster than the cutoff (a passing
vehicle) is rejected by the filter and therefore appears in the difference;
anything slower (the sun moving, a cloud) is tracked and disappears.

**Temporal median** takes the per-pixel median over a sliding window. More
robust than the mean, because a vehicle occupying a pixel for a minority of the
window does not move the median at all, where it does move an average. The cost
is memory: N frames held per pixel instead of one number.

**Three-frame difference** uses no model at all, intersecting the differences
between consecutive frames. Immune to illumination drift and to ghosting
entirely, and hopeless on slow or stopped vehicles, which by definition differ
little between adjacent frames.

**The difference is taken in colour, not in luma**, and that is not a detail.
BT.601 weights red at 0.299, so a red car on grey asphalt can have almost
exactly the road's luminance while differing from it enormously in colour.
Measured on this project's own vehicle palette against its own road surface:

    colour   luma difference   max channel difference
    red                  1.2                     94.0
    blue                 4.2                     68.0
    brown                4.6                     30.0
    green                9.1                     34.0

Four of the ten body colours are invisible to a greyscale background
subtraction. Not hard to detect - *invisible*, differing from the road by about
one grey level. On a clip with a red vehicle the mask contains its shadow and
not the vehicle, which looks like a shadow problem and is actually a colour
problem introduced two stages earlier. So the default difference is the maximum
absolute difference across the three channels, and the greyscale difference is
retained as the comparison baseline it deserves to be.

**Selective update** addresses trap 5. A vehicle stopped at a signal is, to a
running average, simply what the road looks like now, so it is absorbed into the
model - and when it moves off it leaves a vehicle-shaped hole that the pipeline
detects as a phantom vehicle in the wrong place, travelling at zero. Freezing
the update inside pixels currently labelled foreground prevents that, at the
cost of a model that never learns about a genuinely changed scene until the
foreground clears. Both behaviours are worth being able to show, so it is a
toggle rather than a decision made here.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Literal

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import BoolArray, FloatArray, FrameType, StageResult
from kerbside.stages.base import FrameContext, Stage, StageError

__all__ = ["BackgroundStage", "BackgroundModel"]

BackgroundModel = Literal["running_average", "temporal_median", "three_frame", "reference"]
DifferenceMode = Literal["max_channel", "euclidean", "grey"]


class BackgroundStage(Stage):
    """S3: maintain a model of the empty road and difference against it."""

    stage_id = "s3_background"
    name = "Background"
    units = (2, 4)
    rationale = (
        "Maintains a model of the empty road and subtracts it. The running "
        "average is literally a temporal low-pass filter - the Unit 2 smoothing "
        "operator with time as its axis - so a vehicle is whatever changes "
        "faster than its cutoff. The comparison is made in colour rather than "
        "luma, because a red vehicle has almost exactly the luminance of "
        "asphalt and is invisible to a greyscale difference. Selective update "
        "freezes the model inside "
        "foreground pixels so a stopped vehicle is not absorbed and does not "
        "leave a phantom behind when it moves off."
    )
    param_specs = (
        ParamSpec(
            "model", "Model", "enum", "running_average",
            choices=("running_average", "temporal_median", "three_frame", "reference"),
        ),
        ParamSpec(
            "difference_mode", "Difference", "enum", "max_channel",
            choices=("max_channel", "euclidean", "grey"),
            description=(
                "How frame and model are compared. Greyscale is the textbook "
                "choice and misses red, blue, green and brown vehicles entirely, "
                "because their luminance matches asphalt; the colour modes do "
                "not."
            ),
        ),
        ParamSpec(
            "alpha", "Learning rate", "float", 0.02, 0.0005, 0.5, 0.0005,
            description=(
                "Running-average adaptation rate. The model's time constant is "
                "1/alpha frames, so 0.02 at 25 fps is about two seconds."
            ),
        ),
        ParamSpec(
            "median_window", "Median window", "int", 25, 3, 121, 2, unit="frames",
            description="How many frames the temporal median holds per pixel.",
        ),
        ParamSpec(
            "median_stride", "Median stride", "int", 2, 1, 20, 1, unit="frames",
            description=(
                "Sample every nth frame into the window, so a given memory "
                "budget can span a longer stretch of time."
            ),
        ),
        ParamSpec(
            "selective_update", "Selective update", "bool", True,
            description=(
                "Freeze the model inside current foreground. Prevents a stopped "
                "vehicle being absorbed and leaving a ghost (trap 5)."
            ),
        ),
        ParamSpec(
            "selective_dilate_px", "Foreground guard", "odd_int", 5, 1, 31, 2, unit="px",
            description=(
                "Widen the frozen region, so the halo just outside a vehicle - "
                "which is partly vehicle - is not learned either."
            ),
        ),
        ParamSpec(
            "warmup_frames", "Warm-up", "int", 12, 0, 300, 1, unit="frames",
            description=(
                "Frames to build the model before reporting detections. Without "
                "it the first frame is entirely foreground."
            ),
        ),
        ParamSpec(
            "frame_type_threshold", "I-frame threshold", "float", 2.0, 0.05, 60.0, 0.05,
            description=(
                "Mean absolute frame difference above which the frame is "
                "classified I, for the motion-triggered recording statistic."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._grey: FloatArray | None = None
        self._bgr: FloatArray | None = None
        self._window: deque[FloatArray] = deque()
        self._window_bgr: deque[FloatArray] = deque()
        self._history: deque[FloatArray] = deque(maxlen=3)
        self._previous_grey: FloatArray | None = None
        self._last_mask: BoolArray | None = None
        self._frames_seen = 0
        super().__init__(**params)

    def reset(self) -> None:
        self._grey = None
        self._bgr = None
        self._window = deque()
        self._window_bgr = deque()
        self._history = deque(maxlen=3)
        self._previous_grey = None
        self._last_mask = None
        self._frames_seen = 0

    def note_mask(self, mask: BoolArray | None) -> None:
        """Record the mask the pipeline settled on for the frame just finished.

        Selective update has to freeze the model where the *previous* frame
        found a vehicle, and S3 runs before S4 does, so the information has to
        arrive from outside rather than from this frame's context.
        """
        self._last_mask = None if mask is None else np.asarray(mask, dtype=bool)

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    @property
    def is_warm(self) -> bool:
        """Whether the model has seen enough frames to be trusted."""
        return self._frames_seen >= int(self.param("warmup_frames"))

    # -- the three models --------------------------------------------------

    def _update_running_average(
        self, grey: FloatArray, bgr: FloatArray, frozen: BoolArray | None
    ) -> None:
        alpha = float(self.param("alpha"))
        if self._grey is None or self._bgr is None:
            self._grey = grey.copy()
            self._bgr = bgr.copy()
            return
        if frozen is None:
            self._grey += alpha * (grey - self._grey)
            self._bgr += alpha * (bgr - self._bgr)
            return
        # Selective update: a per-pixel learning rate that is alpha where the
        # pixel is believed to be road and zero where it is believed to be a
        # vehicle. Expressed as a mask rather than a branch so it stays one
        # vectorised pass over the frame.
        rate = np.where(frozen, np.float32(0.0), np.float32(alpha))
        self._grey += rate * (grey - self._grey)
        self._bgr += rate[:, :, None] * (bgr - self._bgr)

    def _update_median(self, grey: FloatArray, bgr: FloatArray) -> None:
        stride = int(self.param("median_stride"))
        if self._frames_seen % stride:
            return
        window = int(self.param("median_window"))
        self._window.append(grey.copy())
        self._window_bgr.append(bgr.copy())
        while len(self._window) > window:
            self._window.popleft()
            self._window_bgr.popleft()
        if self._window:
            self._grey = np.median(np.stack(self._window), axis=0)
            self._bgr = np.median(np.stack(self._window_bgr), axis=0)

    def _difference(self, frame: FloatArray, model: FloatArray, grey: FloatArray) -> FloatArray:
        """Compare a frame against the model, in colour by default.

        ``max_channel`` takes the largest absolute per-channel difference, which
        is the most sensitive of the three and the cheapest. ``euclidean`` is the
        RGB distance scaled back onto a 0-255 range so its threshold is
        comparable. ``grey`` is the textbook luma difference, kept because the
        ablation needs it - it is the baseline that demonstrates the problem
        rather than a setting anyone should run.
        """
        mode = self.param("difference_mode")
        if mode == "grey":
            model_grey = model.mean(axis=2) if model.ndim == 3 else model
            return np.asarray(np.abs(grey - model_grey), dtype=np.float32)
        delta = np.abs(frame - model)
        if mode == "max_channel":
            return np.asarray(delta.max(axis=2), dtype=np.float32)
        if mode == "euclidean":
            return np.asarray(
                np.sqrt((delta**2).sum(axis=2)) / np.sqrt(3.0), dtype=np.float32
            )
        raise StageError(f"unknown difference mode {mode!r}")

    def _three_frame_difference(self, grey: FloatArray) -> FloatArray:
        """Intersection of the two adjacent frame differences.

        ``|I_t - I_{t-1}| and |I_{t-1} - I_{t-2}|`` keeps only pixels that
        changed on both steps, which suppresses the trailing "ghost" edge a
        single frame difference leaves where the object *was*. It needs no model
        at all, so nothing can be absorbed into one - and it sees nothing that
        is not moving between adjacent frames.
        """
        self._history.append(grey.copy())
        if len(self._history) < 3:
            return np.zeros(grey.shape, dtype=np.float64)
        a, b, c = self._history[0], self._history[1], self._history[2]
        return np.minimum(np.abs(c - b), np.abs(b - a))

    # -- execution ---------------------------------------------------------

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            grey = ctx.require_grey()
            bgr = ctx.bgr.astype(np.float32)
            model: BackgroundModel = self.param("model")

            frozen: BoolArray | None = None
            previous = ctx.previous_mask if ctx.previous_mask is not None else self._last_mask
            if self.param("selective_update") and previous is not None:
                guard = int(self.param("selective_dilate_px"))
                mask = previous.astype(np.uint8)
                if guard > 1:
                    mask = cv2.dilate(mask, np.ones((guard, guard), np.uint8))
                frozen = np.asarray(mask > 0, dtype=bool)

            if model == "three_frame":
                difference = self._three_frame_difference(grey)
                # No model exists, so downstream stages that want a background -
                # S5 needs one in HSV - are given the previous frame, which is
                # the closest thing this model has to one.
                self._grey = self._previous_grey if self._previous_grey is not None else grey
                self._bgr = bgr if self._bgr is None else self._bgr
            elif model == "temporal_median":
                self._update_median(grey, bgr)
                if self._grey is None or self._bgr is None:
                    self._grey, self._bgr = grey.copy(), bgr.copy()
                difference = self._difference(bgr, self._bgr, grey)
            elif model == "running_average":
                self._update_running_average(grey, bgr, frozen)
                assert self._grey is not None and self._bgr is not None
                difference = self._difference(bgr, self._bgr, grey)
            elif model == "reference":
                # A fixed first frame. Only correct when the clip genuinely
                # opens on an empty road, which is why it is not the default -
                # but it is the cleanest possible background when it holds, and
                # a useful upper bound for the comparison study.
                if self._grey is None or self._bgr is None:
                    self._grey, self._bgr = grey.copy(), bgr.copy()
                difference = self._difference(bgr, self._bgr, grey)
            else:
                raise StageError(f"unknown background model {model!r}")

            self._previous_grey = grey.copy()
            self._frames_seen += 1

            ctx.background_grey = self._grey
            ctx.background_bgr = self._bgr
            ctx.difference = difference

            energy = (
                float(np.abs(grey - self._previous_grey).mean())
                if self._previous_grey is not None
                else float("inf")
            )
            ctx.frame_type = (
                FrameType.I
                if energy >= float(self.param("frame_type_threshold"))
                else FrameType.P
            )

            result.image = np.asarray(np.clip(difference, 0, 255), dtype=np.uint8)
            result.metrics = {
                "frames_seen": float(self._frames_seen),
                "warm": 1.0 if self.is_warm else 0.0,
                "difference_mean": float(difference.mean()),
                "difference_p99": float(np.percentile(difference, 99)),
                "frozen_fraction": float(frozen.mean()) if frozen is not None else 0.0,
                "model_memory_frames": float(len(self._window)) if self._window else 1.0,
            }
            if not self.is_warm:
                result.extras["hint"] = (
                    f"the background model has seen {self._frames_seen} of "
                    f"{int(self.param('warmup_frames'))} warm-up frames; "
                    f"detections before it is warm are not meaningful"
                )
        return result

    def background_hsv(self) -> FloatArray:
        """The background in HSV, which S5 needs to test for shadow.

        Converted from the stored BGR model rather than maintained separately:
        averaging hue directly would be wrong, because hue is an angle and the
        mean of 179 and 1 is 90 - a colour neither frame contained.
        """
        if self._bgr is None:
            raise StageError("no background model has been built yet")
        hsv = cv2.cvtColor(
            np.clip(self._bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV
        ).astype(np.float64)
        hsv[..., 1] /= 255.0
        hsv[..., 2] /= 255.0
        return hsv

    def background_bgr_u8(self) -> np.ndarray:
        if self._bgr is None:
            raise StageError("no background model has been built yet")
        return np.clip(self._bgr, 0, 255).astype(np.uint8)
