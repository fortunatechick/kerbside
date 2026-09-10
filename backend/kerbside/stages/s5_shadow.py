"""S5 - shadow suppression. The headline result.

A cast shadow is part of the foreground by every criterion S4 applies: it
differs from the background, it moves with the vehicle, it is the right size. It
is also not a vehicle, and treating it as one does something worse than
inflating an area. When the sun is low, a vehicle's shadow reaches across a lane
boundary and touches the vehicle beside it, and the two become **one connected
component**. Two vehicles are counted as one. The count is silently low.

That failure presents as a tracking problem - a track that swallowed its
neighbour - and it is diagnosed for hours in S7 by people who have not realised
it is a colour-processing problem that was decided three stages earlier. That is
trap 2, and this stage is where it is fixed.

The test is Unit 7 in three lines. A shadow does not replace the road, it
changes how the road is *lit*, so the surface's own colour survives:

* **Value ratio** ``V_frame / V_background`` in ``[alpha, beta]`` with
  ``0 < alpha < beta < 1``. Darker, but not black - a black car is not a shadow,
  and unchanged road is not one either.
* **Saturation difference** ``S_frame - S_background <= tau_S``. A shadowed
  surface does not become more colourful. One-sided on purpose: shadows commonly
  desaturate, so a large negative difference is still a shadow.
* **Hue difference** within ``tau_H``, **computed circularly**. Hue wraps at
  0/179, so a naive absolute difference calls two shades of red maximally
  different - and red is exactly where the wrap happens (trap 6).

The hue test carries a condition the textbook statement omits, and omitting it
breaks the stage on exactly the surface it matters most for. **Hue is undefined
when saturation is near zero**, and asphalt is near-neutral: measured on the
reference scene its background saturation is about 0.075, and the hue difference
across foreground pixels has a median of 10 and a 95th percentile of 67 - which
is not a measurement of colour, it is the arctangent of two small noisy
differences. Applying a ten-degree hue tolerance there rejects half the shadow
outright. So the hue condition is only enforced where both the frame and the
background are chromatic enough for hue to mean something, and passes
automatically below that. The value and saturation conditions still apply.

Removing shadow pixels from the mask is what splits the bridged blob back into
two vehicles. Everything the ablation reports about this stage - false-merge
rate, count error, speed MAE - follows from that one operation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import BoolArray, FloatArray, StageResult
from kerbside.operators.colour import HUE_MAX, hue_difference
from kerbside.operators.morphology import fill_small_holes
from kerbside.operators.segment import label_components
from kerbside.stages.base import FrameContext, Stage, StageError, as_uint8_mask

__all__ = ["ShadowStage", "ShadowDecision"]


@dataclass(frozen=True, slots=True)
class ShadowDecision:
    """The three decision maps, kept separately for the Shadow Lab.

    Showing which of the three conditions rejected a pixel is far more useful
    than showing the conjunction: a user tuning the thresholds needs to know
    whether their shadow is failing on brightness, on saturation or on hue, and
    the combined mask cannot tell them.
    """

    value_ratio: FloatArray
    saturation_delta: FloatArray
    hue_delta: FloatArray
    passes_value: BoolArray
    passes_saturation: BoolArray
    passes_hue: BoolArray
    shadow: BoolArray

    def rejection_counts(self, foreground: BoolArray) -> dict[str, int]:
        """How many foreground pixels each condition alone rejected."""
        fg = np.asarray(foreground, dtype=bool)
        return {
            "value": int((fg & ~self.passes_value).sum()),
            "saturation": int((fg & ~self.passes_saturation).sum()),
            "hue": int((fg & ~self.passes_hue).sum()),
        }


class ShadowStage(Stage):
    """S5: remove cast shadows from the foreground mask."""

    stage_id = "s5_shadow"
    name = "Shadow"
    units = (7,)
    rationale = (
        "Classifies a foreground pixel as shadow when it is darker than the "
        "background but not black, no more saturated, and close to it in hue "
        "measured circularly. A shadow changes how the road is lit, not what it "
        "is made of, so hue and saturation survive while value drops. Removing "
        "those pixels is what splits two vehicles that a shadow had bridged into "
        "one blob - a miscount that looks like a tracking failure but was "
        "decided here."
    )
    param_specs = (
        ParamSpec(
            "enabled", "Enabled", "bool", True,
            description="Turn off to measure what shadow suppression is worth.",
        ),
        ParamSpec(
            "value_ratio_low", "Value ratio, low", "float", 0.40, 0.01, 0.95, 0.01,
            description="Below this the pixel is too dark to be a shadow - it is a dark object.",
        ),
        ParamSpec(
            "value_ratio_high", "Value ratio, high", "float", 0.85, 0.05, 0.999, 0.01,
            description="Above this the pixel is barely darker than the background.",
        ),
        ParamSpec(
            "saturation_delta", "Saturation tolerance", "float", 0.18, 0.0, 1.0, 0.01,
            description=(
                "How much *more* saturated than the background a shadow may be. "
                "Larger than the classical 0.1 because saturation is a ratio, "
                "and on a near-neutral surface a small absolute channel change "
                "moves it a long way."
            ),
        ),
        ParamSpec(
            "hue_delta", "Hue tolerance", "float", 10.0, 0.0, 90.0, 0.5,
            description="Circular hue tolerance on OpenCV's 0-179 scale.",
        ),
        ParamSpec(
            "min_background_value", "Background floor", "float", 0.06, 0.0, 0.5, 0.01,
            description=(
                "Skip pixels whose background is nearly black: there is no "
                "meaningful ratio there, and calling them shadow would erase "
                "every dark corner of the frame."
            ),
        ),
        ParamSpec(
            "open_px", "Clean shadow mask", "odd_int", 3, 1, 15, 2, unit="px",
            description=(
                "Open the shadow mask before subtracting it, so scattered "
                "single pixels inside a vehicle are not punched out of it."
            ),
        ),
        ParamSpec(
            "hue_saturation_floor", "Hue test applies above", "float", 0.15, 0.0, 1.0, 0.01,
            description=(
                "Minimum saturation for the hue condition to be enforced. Hue is "
                "undefined on a neutral surface, and asphalt is neutral, so "
                "below this the hue test would be testing noise."
            ),
        ),
        ParamSpec(
            "restore_hole_px", "Restore holes up to", "float", 500.0, 0.0, 20000.0, 50.0,
            unit="px",
            description=(
                "Fill holes this small that removing shadow punched inside a "
                "blob - a mid-grey roof whose colour happens to satisfy the "
                "test. Bounded by area on purpose: an unbounded fill would put "
                "the shadow itself straight back, because subtracting it from "
                "inside a larger region leaves an enclosed hole."
            ),
        ),
        ParamSpec(
            "open_after_px", "Sliver removal", "odd_int", 3, 1, 15, 2, unit="px",
            description=(
                "Open the mask after subtracting shadow. The boundary between a "
                "vehicle and its own shadow is not crisp, so subtraction leaves "
                "thin slivers along it - and a sliver has a plausible area even "
                "though it has no plausible shape."
            ),
        ),
        ParamSpec(
            "close_after_px", "Tidy after removal", "odd_int", 5, 1, 15, 2, unit="px",
            description=(
                "Close the mask after subtracting shadow, so a ragged shadow "
                "boundary does not shatter a vehicle into fragments."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._last: ShadowDecision | None = None
        super().__init__(**params)

    @property
    def last_decision(self) -> ShadowDecision | None:
        """The most recent decision maps, for the Shadow Lab's heatmaps."""
        return self._last

    def decide(
        self, hsv_frame: FloatArray, hsv_background: FloatArray, foreground: BoolArray
    ) -> ShadowDecision:
        """Evaluate the three conditions and combine them."""
        if hsv_frame.shape != hsv_background.shape:
            raise StageError(
                f"the frame is {hsv_frame.shape} in HSV but the background model "
                f"is {hsv_background.shape}; they must match"
            )
        hsv_frame = np.asarray(hsv_frame, dtype=np.float32)
        hsv_background = np.asarray(hsv_background, dtype=np.float32)
        v_frame = hsv_frame[..., 2]
        v_background = hsv_background[..., 2]
        floor = float(self.param("min_background_value"))
        usable = v_background > floor

        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(usable, v_frame / np.maximum(v_background, 1e-6), 0.0)

        saturation_delta = hsv_frame[..., 1] - hsv_background[..., 1]
        hue_delta = hue_difference(hsv_frame[..., 0], hsv_background[..., 0], HUE_MAX)

        passes_value = (
            usable
            & (ratio >= float(self.param("value_ratio_low")))
            & (ratio <= float(self.param("value_ratio_high")))
        )
        passes_saturation = saturation_delta <= float(self.param("saturation_delta"))
        # Hue is only meaningful where there is enough saturation to define it.
        # On near-neutral asphalt the hue difference is the arctangent of two
        # small noisy quantities, so enforcing a tolerance on it rejects genuine
        # shadow; above the floor it is a real and discriminating measurement.
        floor_s = float(self.param("hue_saturation_floor"))
        chromatic = (hsv_frame[..., 1] >= floor_s) & (hsv_background[..., 1] >= floor_s)
        passes_hue = (hue_delta <= float(self.param("hue_delta"))) | ~chromatic

        shadow = np.asarray(
            passes_value & passes_saturation & passes_hue & np.asarray(foreground, dtype=bool),
            dtype=bool,
        )
        return ShadowDecision(
            value_ratio=ratio,
            saturation_delta=saturation_delta,
            hue_delta=hue_delta,
            passes_value=passes_value,
            passes_saturation=passes_saturation,
            passes_hue=passes_hue,
            shadow=shadow,
        )

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            foreground = ctx.foreground
            if foreground is None:
                raise StageError("S4 has not run: there is no foreground mask to refine")

            if not self.param("enabled"):
                ctx.shadow = np.zeros(foreground.shape, dtype=bool)
                ctx.foreground_no_shadow = foreground
                result.mask = foreground
                result.image = as_uint8_mask(foreground)
                result.metrics = {
                    "shadow_px": 0.0,
                    "components_before": 0.0,
                    "components_after": 0.0,
                }
                result.extras["disabled"] = True
                return result

            if ctx.background_bgr is None:
                raise StageError(
                    "shadow suppression needs a colour background model; S3 has "
                    "not produced one"
                )
            hsv_background = _to_hsv(ctx.background_bgr)
            decision = self.decide(ctx.require_hsv(), hsv_background, foreground)
            shadow = decision.shadow

            open_px = int(self.param("open_px"))
            if open_px > 1:
                shadow = np.asarray(
                    cv2.morphologyEx(
                        as_uint8_mask(shadow), cv2.MORPH_OPEN,
                        np.ones((open_px, open_px), np.uint8),
                    )
                    > 0,
                    dtype=bool,
                )

            before = label_components(foreground, 8, with_stats=False).count
            cleaned = np.asarray(foreground & ~shadow, dtype=bool)

            open_px = int(self.param("open_after_px"))
            if open_px > 1:
                cleaned = np.asarray(
                    cv2.morphologyEx(
                        as_uint8_mask(cleaned), cv2.MORPH_OPEN,
                        np.ones((open_px, open_px), np.uint8),
                    )
                    > 0,
                    dtype=bool,
                )

            close_px = int(self.param("close_after_px"))
            if close_px > 1:
                # The boundary between a vehicle and its own shadow is not
                # crisp, so subtracting the shadow leaves a ragged edge and a
                # scatter of isolated survivors. Closing tidies both without
                # moving the vehicle outline appreciably.
                cleaned = np.asarray(
                    cv2.morphologyEx(
                        as_uint8_mask(cleaned), cv2.MORPH_CLOSE,
                        np.ones((close_px, close_px), np.uint8),
                    )
                    > 0,
                    dtype=bool,
                )

            hole_px = float(self.param("restore_hole_px"))
            if hole_px > 0:
                cleaned = fill_small_holes(cleaned, int(hole_px))
            shadow = np.asarray(foreground & ~cleaned, dtype=bool)

            after = label_components(cleaned, 8, with_stats=False).count

            ctx.shadow = shadow
            ctx.foreground_no_shadow = cleaned
            self._last = decision

            result.mask = cleaned
            result.image = as_uint8_mask(cleaned)
            result.metrics = {
                "shadow_px": float(shadow.sum()),
                "shadow_fraction_of_foreground": (
                    float(shadow.sum() / foreground.sum()) if foreground.any() else 0.0
                ),
                "foreground_before_px": float(foreground.sum()),
                "foreground_after_px": float(cleaned.sum()),
                "components_before": float(before),
                "components_after": float(after),
                "components_gained": float(after - before),
            }
            result.extras["rejections"] = decision.rejection_counts(foreground)
            if after > before:
                # The headline event: removing shadow separated components that
                # were bridged. Recorded per frame so the Shadow Lab can find
                # and jump to the frames where it mattered.
                result.extras["split_event"] = True
        return result


def _to_hsv(bgr: FloatArray) -> FloatArray:
    """Background model to HSV, in float32.

    Float32 throughout the mask stages rather than float64: the input is eight
    bits, none of these comparisons need fifteen significant figures, and
    halving the memory traffic halves the time on arrays this size.
    """
    hsv = cv2.cvtColor(np.clip(bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(
        np.float32
    )
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv
