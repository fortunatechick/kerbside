"""S2 - stabilisation: undo camera shake before anything differences the frame.

A pole-mounted camera vibrates. Wind, passing lorries and the pole's own
resonance move the image by a few pixels, and a few pixels is catastrophic for
everything downstream: background subtraction compares a frame against a model
built when the camera was pointing somewhere slightly different, so every edge
in the scene - every lane marking, every kerb, the horizon - appears in the
difference image. The mask lights up everywhere and the count goes to nonsense.
That is trap 8, and its signature is a first night of real-footage testing that
looks like total failure.

The fix is Unit 3 doing what it is best at. Shifting an image multiplies its
Fourier transform by a linear phase ramp and changes nothing else, so the
normalised cross-power spectrum of two frames is that ramp, and its inverse
transform is an impulse at the shift. Because the magnitudes are divided out,
the estimate is immune to the illumination changes that would defeat a
correlation on the images themselves.

Two details make it work on real footage rather than on a test pattern:

* the estimate is taken over a **static region** the user nominates - verge,
  buildings, the far kerb - because a frame full of moving traffic contains a
  second, competing displacement, and phase correlation will happily lock onto
  it when the traffic covers enough of the frame;
* a **confidence test** on the correlation peak, so that a featureless or
  fog-bound frame declines to report a displacement instead of injecting the
  very motion the stage exists to remove.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import FloatArray, StageResult
from kerbside.operators.frequency import phase_correlation
from kerbside.stages.base import FrameContext, Stage, StageError

__all__ = ["StabiliseStage"]


class StabiliseStage(Stage):
    """S2: estimate global frame-to-frame translation and compensate for it."""

    stage_id = "s2_stabilise"
    name = "Stabilise"
    units = (3,)
    rationale = (
        "Estimates the frame-to-frame translation by phase correlation on the "
        "DFT and warps it out. A pole-mounted camera moves by a few pixels in "
        "the wind, and without compensation the background difference contains "
        "every edge in the scene rather than the vehicles - the mask lights up "
        "everywhere and counting fails completely."
    )
    param_specs = (
        ParamSpec(
            "enabled", "Enabled", "bool", True,
            description=(
                "Turn off to see what shake does to the foreground mask. On a "
                "shaking clip the answer is that the whole frame becomes "
                "foreground."
            ),
        ),
        ParamSpec(
            "reference_mode", "Reference", "enum", "first",
            choices=("first", "previous"),
            description=(
                "Align to the first frame, which stops drift accumulating, or "
                "to the previous frame, which follows a slow pan."
            ),
        ),
        ParamSpec(
            "region_top", "Static region top", "float", 0.0, 0.0, 1.0, 0.01,
            description="Fraction of the frame height where the static region starts.",
        ),
        ParamSpec(
            "region_bottom", "Static region bottom", "float", 1.0, 0.0, 1.0, 0.01,
            description=(
                "Where it ends. Narrowing this to a band of verge or skyline "
                "keeps moving traffic out of the estimate."
            ),
        ),
        ParamSpec("upsample_factor", "Sub-pixel resolution", "int", 20, 1, 100, 1),
        ParamSpec(
            "downsample", "Correlate at 1/n scale", "int", 2, 1, 8, 1,
            description=(
                "Estimate the shift on a decimated copy. Camera shake is a "
                "low-frequency, whole-frame motion, so halving the resolution "
                "costs almost nothing in accuracy and a factor of four in time; "
                "the sub-pixel refinement then works in the decimated frame and "
                "the answer is scaled back up."
            ),
        ),
        ParamSpec(
            "measure_residual", "Measure residual", "bool", False,
            description=(
                "Re-correlate after warping to report what shake is left. "
                "Doubles the cost of the stage, so it is off outside the "
                "stabilisation study."
            ),
        ),
        ParamSpec(
            "max_shift_px", "Maximum plausible shift", "float", 24.0, 1.0, 200.0, 1.0,
            description=(
                "Estimates larger than this are rejected as a mis-lock rather "
                "than applied."
            ),
        ),
        ParamSpec(
            "min_peak_ratio", "Minimum peak ratio", "float", 1.6, 1.0, 10.0, 0.05,
            description=(
                "How far the correlation peak must stand above the rest of the "
                "surface before the estimate is believed."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._reference: FloatArray | None = None
        self._previous: FloatArray | None = None
        self._cumulative = np.zeros(2, dtype=np.float64)
        super().__init__(**params)

    def reset(self) -> None:
        self._reference = None
        self._previous = None
        self._cumulative = np.zeros(2, dtype=np.float64)

    def _region(self, grey: FloatArray) -> FloatArray:
        top = float(self.param("region_top"))
        bottom = float(self.param("region_bottom"))
        if top >= bottom:
            raise StageError(
                f"the static region is empty: top {top:.2f} is not above bottom "
                f"{bottom:.2f}"
            )
        h = grey.shape[0]
        y0 = int(np.clip(round(top * h), 0, h - 2))
        y1 = int(np.clip(round(bottom * h), y0 + 2, h))
        band = np.asarray(grey[y0:y1], dtype=np.float32)
        step = int(self.param("downsample"))
        if step > 1:
            band = cv2.resize(
                band, (band.shape[1] // step, band.shape[0] // step),
                interpolation=cv2.INTER_AREA,
            )
        return np.asarray(band, dtype=np.float64)

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            grey = ctx.require_grey()
            region = self._region(grey)

            if not self.param("enabled"):
                ctx.stabilised = False
                ctx.shake_dx_px = 0.0
                ctx.shake_dy_px = 0.0
                self._previous = region
                if self._reference is None:
                    self._reference = region
                result.metrics = {"dx_px": 0.0, "dy_px": 0.0, "residual_px": 0.0}
                result.extras["disabled"] = True
                return result

            if self._reference is None:
                # The first frame defines the reference; there is nothing to
                # align it to, and reporting a zero displacement is the truth.
                self._reference = region
                self._previous = region
                ctx.stabilised = True
                result.metrics = {"dx_px": 0.0, "dy_px": 0.0, "residual_px": 0.0, "peak_ratio": 0.0}
                return result

            anchor = (
                self._reference
                if self.param("reference_mode") == "first"
                else (self._previous if self._previous is not None else self._reference)
            )
            correlation = phase_correlation(
                anchor, region, window=True,
                upsample_factor=int(self.param("upsample_factor")),
            )

            step = float(self.param("downsample"))
            dx, dy = correlation.dx_px * step, correlation.dy_px * step
            confident = (
                correlation.peak_ratio >= float(self.param("min_peak_ratio"))
                and float(np.hypot(dx, dy)) <= float(self.param("max_shift_px"))
            )
            if self.param("reference_mode") == "previous" and confident:
                # Per-frame displacements accumulate into an absolute offset;
                # applying only the incremental one would leave the frame
                # drifting away from the background model a pixel at a time.
                self._cumulative += np.array([dx, dy])
                dx, dy = float(self._cumulative[0]), float(self._cumulative[1])

            if not confident:
                # Declining is the point of the confidence test. A featureless
                # frame produces a broad, flat correlation surface whose maximum
                # is arbitrary, and warping by it would inject exactly the
                # motion this stage exists to remove.
                dx = dy = 0.0
                result.extras["hint"] = (
                    f"the correlation peak stood only {correlation.peak_ratio:.2f}x "
                    f"above the surrounding surface, or the estimate exceeded the "
                    f"plausible shift; this frame was left unwarped rather than "
                    f"warped by an unreliable estimate"
                )

            ctx.shake_dx_px = dx
            ctx.shake_dy_px = dy
            ctx.stabilisation_confident = confident
            ctx.stabilised = True

            if abs(dx) > 1e-3 or abs(dy) > 1e-3:
                # Warp by the negative of the measured content displacement: if
                # the scene moved right, the frame must move left to put it back.
                matrix = np.array([[1.0, 0.0, -dx], [0.0, 1.0, -dy]], dtype=np.float64)
                ctx.bgr = np.asarray(
                    cv2.warpAffine(
                        ctx.bgr, matrix, (ctx.bgr.shape[1], ctx.bgr.shape[0]),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                    ),
                    dtype=np.uint8,
                )
                ctx.grey = np.asarray(
                    cv2.warpAffine(
                        grey, matrix, (grey.shape[1], grey.shape[0]),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                    ),
                    dtype=np.float64,
                )
                hsv = ctx.require_hsv()
                ctx.hsv = np.asarray(
                    cv2.warpAffine(
                        hsv, matrix, (hsv.shape[1], hsv.shape[0]),
                        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE,
                    ),
                    dtype=np.float64,
                )

            self._previous = self._region(ctx.require_grey())
            residual = self._residual_px(ctx) if self.param("measure_residual") else 0.0
            result.image = np.asarray(np.clip(ctx.grey, 0, 255), dtype=np.uint8)
            result.metrics = {
                "dx_px": dx,
                "dy_px": dy,
                "shift_px": float(np.hypot(dx, dy)),
                "peak_ratio": correlation.peak_ratio,
                "residual_px": residual,
                "confident": 1.0 if confident else 0.0,
            }
        return result

    def _residual_px(self, ctx: FrameContext) -> float:
        """How much shake is left after warping.

        Reported rather than assumed to be zero: interpolation, a
        non-translational component such as a small rotation, and the
        confidence rejection above all leave a residue, and the honest number is
        what a user needs when deciding whether stabilisation is working.
        """
        if self._reference is None:
            return 0.0
        region = self._region(ctx.require_grey())
        if region.shape != self._reference.shape:
            return float("nan")
        check = phase_correlation(self._reference, region, window=True, upsample_factor=10)
        return float(check.magnitude_px) * float(self.param("downsample"))


def rms_shake_px(offsets: list[tuple[float, float]]) -> float:
    """RMS magnitude of a sequence of per-frame offsets.

    The headline figure for the stabilisation study: the renderer records the
    shake it applied, so the residual after stabilisation can be compared
    against the amplitude that went in rather than merely looking smaller.
    """
    if not offsets:
        return 0.0
    arr = np.asarray(offsets, dtype=np.float64)
    return float(np.sqrt(np.mean(arr[:, 0] ** 2 + arr[:, 1] ** 2)))
