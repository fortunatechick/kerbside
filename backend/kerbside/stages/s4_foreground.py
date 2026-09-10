"""S4 - foreground extraction: turn a difference image into a clean mask.

The difference from S3 is a continuous-valued image. Turning it into a decision
takes a threshold, and then the decision needs cleaning up, because a raw
threshold of a real difference image is speckled with noise and hollow wherever
the vehicle happened to match the road behind it - windscreens especially, since
a windscreen reflecting a grey sky is a very good match for grey asphalt.

The sequence is threshold, median, open, close, fill, and each step is there for
a specific defect:

* **Otsu** picks the level from the histogram rather than from a person, so it
  tracks changing light through the clip. Its separability is reported, because
  on a frame containing no vehicles Otsu still returns a threshold and only that
  number says the answer means nothing.
* **Median filter** removes isolated speckle without moving the vehicle
  boundary, which a mean filter would.
* **Opening** removes what is smaller than its element; **closing** fills what
  is smaller than *its* element. Both elements are perspective-scaled, because a
  half-metre object is forty pixels across in the near field and three at the
  horizon, and no single pixel size is right for both (trap 4).

  The closing element's *height* is scaled by the across-road scale rather than
  the along-road one, because it is bridging a gap in a vehicle's silhouette and
  a silhouette extends upward by the vehicle's height. A metre of height
  projects at nearly the across-road scale; a metre along the road is
  foreshortened five or six times more. Sized the other way the element comes
  out a couple of pixels tall, the window band survives as a gap, and every
  vehicle is reported as two blobs - a roof and a bumper - which then presents
  as a tracking problem rather than as a structuring element that was too short.
* **Hole filling** recovers interiors the closing was too small to bridge,
  without the outward growth a larger closing would cause.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import BoolArray, StageResult
from kerbside.geometry.perspective import PerspectiveModel
from kerbside.operators.morphology import banded_morphology, fill_holes, structuring_element
from kerbside.operators.segment import otsu_threshold
from kerbside.stages.base import FrameContext, Stage, StageError, as_uint8_mask

__all__ = ["ForegroundStage"]


class ForegroundStage(Stage):
    """S4: threshold the difference image and clean up the mask."""

    stage_id = "s4_foreground"
    name = "Foreground"
    units = (2, 4, 6)
    rationale = (
        "Thresholds the background difference with Otsu, then cleans the mask "
        "with a median filter and perspective-scaled opening and closing. "
        "Opening kills the speckle a difference image always carries; closing "
        "fills vehicle interiors, where a windscreen reflecting the sky matches "
        "the road better than the bodywork does."
    )
    param_specs = (
        ParamSpec(
            "threshold_mode", "Threshold", "enum", "otsu",
            choices=("otsu", "fixed", "adaptive"),
            description=(
                "Otsu adapts to changing light; fixed is the comparison "
                "baseline; adaptive uses a local neighbourhood mean."
            ),
        ),
        ParamSpec("fixed_level", "Fixed level", "float", 28.0, 1.0, 200.0, 1.0),
        ParamSpec(
            "otsu_offset", "Otsu offset", "float", 0.0, -40.0, 40.0, 1.0,
            description="Bias the computed level towards or against foreground.",
        ),
        ParamSpec(
            "min_threshold", "Threshold floor", "float", 12.0, 0.0, 120.0, 1.0,
            description=(
                "A floor under whatever the threshold rule returns. On a frame "
                "with no vehicles Otsu splits the noise distribution in half and "
                "reports a level near zero; the floor stops that becoming a "
                "frame full of detections."
            ),
        ),
        ParamSpec("median_px", "Median filter", "odd_int", 5, 1, 15, 2, unit="px"),
        ParamSpec(
            "open_m", "Opening size", "float", 0.15, 0.0, 2.0, 0.05, unit="m",
            description="Real-world size of the opening element. Zero disables it.",
        ),
        ParamSpec(
            "close_m", "Closing size", "float", 0.9, 0.0, 4.0, 0.05, unit="m",
            description=(
                "Real-world size of the closing element. Too large and adjacent "
                "lanes merge; too small and vehicle interiors stay hollow."
            ),
        ),
        ParamSpec(
            "close_vertical", "Closing height scale", "enum", "height",
            choices=("height", "along"),
            description=(
                "What the closing element's height measures. 'height' scales it "
                "by the across-road scale, which is what a vehicle silhouette "
                "needs; 'along' scales it by the along-road scale, which is "
                "correct only for something lying flat on the road."
            ),
        ),
        ParamSpec(
            "perspective_scaled", "Perspective-scaled elements", "bool", True,
            description=(
                "Size the elements from the homography per image band. Turn off "
                "to see trap 4: one fixed size cannot serve both the near field "
                "and the horizon."
            ),
        ),
        ParamSpec("fixed_open_px", "Fixed opening size", "odd_int", 3, 1, 41, 2, unit="px"),
        ParamSpec("fixed_close_px", "Fixed closing size", "odd_int", 11, 1, 61, 2, unit="px"),
        ParamSpec("bands", "Perspective bands", "int", 12, 1, 60, 1),
        ParamSpec("fill_holes", "Fill holes", "bool", True),
        ParamSpec(
            "roi_top", "Ignore above row", "float", 0.0, 0.0, 1.0, 0.01,
            description=(
                "Fraction of frame height to exclude from the top. The sky is "
                "never foreground, and including it drags the Otsu histogram."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._model: PerspectiveModel | None = None
        self._model_key: tuple[int, int, float] | None = None
        super().__init__(**params)

    def _perspective(self, ctx: FrameContext) -> PerspectiveModel | None:
        """The perspective model, rebuilt only when the calibration changes.

        Building it walks every image row through the homography, which is far
        too expensive to repeat per frame; keying the cache on the calibration
        means a recalibration is picked up without a stale model surviving it.
        """
        if ctx.calibration is None or not ctx.calibration.is_valid:
            return None
        h = np.asarray(ctx.calibration.h_world_to_img, dtype=np.float64)
        key = (ctx.calibration.width_px, ctx.calibration.height_px, float(h.sum()))
        if self._model is None or self._model_key != key:
            self._model = PerspectiveModel(
                h, ctx.calibration.width_px, ctx.calibration.height_px
            )
            self._model_key = key
        return self._model

    def _threshold(self, ctx: FrameContext) -> tuple[BoolArray, float, float]:
        if ctx.difference is None:
            raise StageError("S3 has not run: there is no difference image to threshold")
        difference = np.clip(ctx.difference, 0, 255).astype(np.uint8)

        roi = np.ones(difference.shape, dtype=bool)
        top = float(self.param("roi_top"))
        if top > 0:
            roi[: int(round(top * difference.shape[0]))] = False

        mode = self.param("threshold_mode")
        separability = 0.0
        if mode == "otsu":
            stats = otsu_threshold(difference, mask=roi)
            level = stats.threshold + float(self.param("otsu_offset"))
            separability = stats.separability
        elif mode == "fixed":
            level = float(self.param("fixed_level"))
        elif mode == "adaptive":
            block = 31
            local = cv2.adaptiveThreshold(
                difference, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                block, -float(self.param("fixed_level")),
            )
            return (np.asarray(local > 0, dtype=bool) & roi, float(self.param("fixed_level")), 0.0)
        else:
            raise StageError(f"unknown threshold mode {mode!r}")

        level = max(level, float(self.param("min_threshold")))
        return (np.asarray(difference > level, dtype=bool) & roi, level, separability)

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            mask, level, separability = self._threshold(ctx)
            raw_area = int(mask.sum())

            median_px = int(self.param("median_px"))
            if median_px > 1:
                mask = np.asarray(
                    cv2.medianBlur(as_uint8_mask(mask), median_px) > 0, dtype=bool
                )
            after_median = int(mask.sum())

            model = self._perspective(ctx) if self.param("perspective_scaled") else None
            bands = int(self.param("bands"))

            open_size = float(self.param("open_m"))
            if open_size > 0:
                if model is not None:
                    mask = np.asarray(
                        banded_morphology(
                            as_uint8_mask(mask), model, open_size, op="open", bands=bands
                        )
                        > 0,
                        dtype=bool,
                    )
                else:
                    element = structuring_element(int(self.param("fixed_open_px")), shape="ellipse")
                    mask = np.asarray(
                        cv2.morphologyEx(as_uint8_mask(mask), cv2.MORPH_OPEN, element) > 0,
                        dtype=bool,
                    )
            after_open = int(mask.sum())

            close_size = float(self.param("close_m"))
            if close_size > 0:
                if model is not None:
                    mask = np.asarray(
                        banded_morphology(
                            as_uint8_mask(mask), model, close_size, op="close",
                            bands=bands, vertical=str(self.param("close_vertical")),
                        )
                        > 0,
                        dtype=bool,
                    )
                else:
                    element = structuring_element(
                        int(self.param("fixed_close_px")), shape="ellipse"
                    )
                    mask = np.asarray(
                        cv2.morphologyEx(as_uint8_mask(mask), cv2.MORPH_CLOSE, element) > 0,
                        dtype=bool,
                    )
            after_close = int(mask.sum())

            if self.param("fill_holes"):
                mask = fill_holes(mask)

            ctx.foreground = mask
            ctx.threshold_level = level
            ctx.foreground_no_shadow = None  # S5 has not run on this frame yet

            result.mask = mask
            result.image = as_uint8_mask(mask)
            result.metrics = {
                "threshold": level,
                "separability": separability,
                "raw_area_px": float(raw_area),
                "after_median_px": float(after_median),
                "after_open_px": float(after_open),
                "after_close_px": float(after_close),
                "final_area_px": float(mask.sum()),
                "coverage": float(mask.mean()),
                "speckle_removed_px": float(raw_area - after_open),
                "interior_filled_px": float(int(mask.sum()) - after_open),
                "perspective_scaled": 1.0 if model is not None else 0.0,
            }
            if model is None and self.param("perspective_scaled"):
                result.extras["hint"] = (
                    "perspective-scaled elements were requested but no valid "
                    "calibration is loaded, so a single fixed element size was "
                    "used instead; near-field and distant vehicles will be "
                    "cleaned unequally"
                )
            if mask.mean() > 0.45:
                result.extras["warning"] = (
                    f"{mask.mean() * 100:.0f}% of the frame is foreground. That "
                    f"usually means the background model is not warm, the camera "
                    f"moved without stabilisation, or the threshold is far too "
                    f"low - not that the road is full of vehicles."
                )
        return result
