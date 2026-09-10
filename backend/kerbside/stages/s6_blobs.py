"""S6 - blobs: turn a mask into individual vehicles.

Connected-component labelling is what individuates. Up to here the pipeline has
a set of foreground pixels; from here it has *things*, each with a position, a
size and - through the homography - a real-world footprint.

Two decisions carry the stage.

**Which components are vehicles.** Not by a global minimum area: under
perspective the same car covers over a thousand times more pixels in the near
field than at the horizon, so any single threshold either admits near-field
noise or discards every distant vehicle (trap 3). Instead the expected area is
computed *at the blob's own road position* by projecting a reference footprint
through the homography, and the blob is accepted if its area falls within a
tolerance band of that.

Area alone is not enough, though, and the failure is instructive. Subtracting a
shadow leaves thin slivers along the boundary where the vehicle met it, and a
sliver six metres long and twenty centimetres wide has exactly the area of a
small car. It is only implausible once you look at its *dimensions*. So the
filter also converts the blob's width and length into metres and requires both
to be within the range a road vehicle occupies - which is the same homography
doing more of the work it is already there for.

**Which components are more than one vehicle.** A blob whose real-world
footprint is implausibly large is not a bus; it is two vehicles that a shadow
bridge or a short following distance merged. A distance-transform watershed
splits it at the neck. Splits are reported rather than performed quietly,
because how often they happen is a measurement of how well S5 did its job.

Every blob's geometry is taken at its **ground point** - the bottom-centre of
its bounding box - because that is the point that lies on the road plane. The
centroid does not, and using it produces distances that are wrong in a way that
grows with vehicle height and looks entirely plausible (trap 1).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import AnyArray, BBox, Blob, Footprint, Lane, Point2, StageResult
from kerbside.geometry.perspective import PerspectiveModel, footprint_quad_px, polygon_area_px
from kerbside.operators.segment import Connectivity, label_components, watershed_split
from kerbside.stages.base import FrameContext, Stage

__all__ = ["BlobStage", "point_in_lane"]


def point_in_lane(point: Point2, lane: Lane) -> bool:
    """Whether a ground point lies inside a lane polygon."""
    polygon = np.asarray(lane.polygon_px, dtype=np.float32)
    if polygon.shape[0] < 3:
        return False
    return cv2.pointPolygonTest(polygon, (float(point.x), float(point.y)), False) >= 0


class BlobStage(Stage):
    """S6: label components, filter them by perspective size, split merges."""

    stage_id = "s6_blobs"
    name = "Blobs"
    units = (1, 6)
    rationale = (
        "Labels connected components and accepts each one only if its pixel "
        "area matches what a vehicle would cover at that place on the road, "
        "projected through the homography. A single global area threshold "
        "cannot work under perspective: set it to reject near-field noise and "
        "every distant vehicle disappears with it."
    )
    param_specs = (
        ParamSpec(
            "connectivity", "Connectivity", "enum", "8", choices=("4", "8", "m"),
            description=(
                "4 fragments a vehicle whose mask narrows to a diagonal; 8 and "
                "m give identical components, and m additionally makes the path "
                "between two pixels unique."
            ),
        ),
        ParamSpec(
            "area_lower", "Area band, low", "float", 0.22, 0.02, 1.0, 0.01,
            description=(
                "Smallest acceptable area as a multiple of a reference car's "
                "projected footprint. A motorcycle is about a quarter."
            ),
        ),
        ParamSpec(
            "area_upper", "Area band, high", "float", 6.0, 1.0, 30.0, 0.1,
            description="Largest acceptable area. A bus is about six.",
        ),
        ParamSpec(
            "min_expected_area_px", "Measurable footprint floor", "float", 200.0, 0.0, 4000.0,
            10.0, unit="px",
            description=(
                "Reject blobs at image rows where a reference car would project "
                "to less than this. It is a statement about the geometry rather "
                "than about the blob: beyond that row a vehicle is a handful of "
                "pixels, a one-pixel error in its ground point is several metres "
                "of road, and nothing measured there can be trusted. Real "
                "counting cameras are aimed at a working range for the same "
                "reason."
            ),
        ),
        ParamSpec(
            "min_width_m", "Minimum width", "float", 0.5, 0.05, 3.0, 0.05, unit="m",
            description=(
                "Narrower than this is not a vehicle. Rejects the slivers that "
                "shadow subtraction leaves, which have a plausible area and no "
                "plausible shape."
            ),
        ),
        ParamSpec("max_width_m", "Maximum width", "float", 3.6, 1.0, 12.0, 0.1, unit="m"),
        ParamSpec("min_length_m", "Minimum length", "float", 0.8, 0.1, 6.0, 0.1, unit="m"),
        ParamSpec("max_length_m", "Maximum length", "float", 26.0, 2.0, 60.0, 0.5, unit="m"),
        ParamSpec(
            "min_area_px", "Absolute floor", "float", 24.0, 1.0, 4000.0, 1.0, unit="px",
            description=(
                "A hard floor regardless of perspective. Below a couple of dozen "
                "pixels there is no shape to measure, whatever the geometry says."
            ),
        ),
        ParamSpec(
            "split_threshold", "Split above", "float", 2.6, 1.2, 12.0, 0.1,
            description=(
                "Attempt a split when a blob's footprint exceeds this multiple "
                "of the reference. Set high enough not to split genuine lorries."
            ),
        ),
        ParamSpec("split_enabled", "Split merged blobs", "bool", True),
        ParamSpec("split_marker_ratio", "Split marker depth", "float", 0.45, 0.05, 0.95, 0.05),
        ParamSpec(
            "require_lane", "Only inside lanes", "bool", False,
            description=(
                "Discard blobs whose ground point is outside every lane "
                "polygon. Useful on a scene with a busy verge; harmful if the "
                "lane polygons are drawn tightly."
            ),
        ),
        ParamSpec(
            "max_blobs", "Maximum blobs", "int", 120, 1, 2000, 1,
            description=(
                "A guard, not a tuning knob. A pathological mask can produce "
                "thousands of components, and the frame budget matters more "
                "than reporting all of them."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._model: PerspectiveModel | None = None
        self._model_key: tuple[int, int, float] | None = None
        self._split_events = 0
        super().__init__(**params)

    def reset(self) -> None:
        self._split_events = 0

    @property
    def split_events(self) -> int:
        return self._split_events

    def _perspective(self, ctx: FrameContext) -> PerspectiveModel | None:
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

    def _footprint_m(
        self, model: PerspectiveModel, bbox: BBox, area_px: float
    ) -> tuple[Footprint | None, Point2 | None, float]:
        """Real-world footprint implied by a blob, and its road position.

        Width comes from the box width at the ground row and length from the
        blob's *area* rather than from its box height, because the box height is
        contaminated by the vehicle's own height - a tall van and a long lorry
        can project to the same box. Area over width is the length of a
        rectangle covering the same road, which is much closer to what the class
        bands describe.
        """
        ground = bbox.ground_point_px
        world = model.image_to_world_m(ground.x, ground.y)
        if world is None:
            return (None, None, 0.0)
        across, along = model.scale_px_per_m(ground.y)
        if across <= 1e-6 or along <= 1e-6:
            return (None, None, 0.0)

        width_m = bbox.w_px / across
        expected = model.expected_area_px(ground.y)
        # Convert the blob's pixel area into road area through the same local
        # scale the expectation used, so the two are comparable by construction.
        area_m2 = area_px / max(across * along, 1e-9)
        length_m = area_m2 / max(width_m, 1e-3)
        return (
            Footprint(length_m=float(length_m), width_m=float(width_m)),
            Point2(float(world[0]), float(world[1])),
            expected,
        )

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            mask = ctx.mask
            model = self._perspective(ctx)
            connectivity: Connectivity = (
                "m" if self.param("connectivity") == "m" else int(self.param("connectivity"))  # type: ignore[assignment]
            )

            labelled = label_components(mask, connectivity)
            candidates = list(labelled.stats)
            rejected_small = 0
            rejected_large = 0
            rejected_shape = 0
            rejected_far = 0
            rejected_lane = 0
            split_here = 0

            blobs: list[Blob] = []
            next_id = 1
            lanes = ctx.calibration.lanes if ctx.calibration is not None else ()

            for stats in candidates:
                if len(blobs) >= int(self.param("max_blobs")):
                    break
                bbox = BBox(
                    float(stats.x_px), float(stats.y_px),
                    float(stats.w_px), float(stats.h_px),
                )
                area = float(stats.area_px)
                if area < float(self.param("min_area_px")):
                    rejected_small += 1
                    continue

                pieces: list[tuple[BBox, float, AnyArray]] = []
                # Extract the component inside its own bounding box. Comparing
                # the full label image against a label once per component costs
                # the frame area times the component count, which on a shattered
                # mask is seconds per frame.
                y0, x0 = int(stats.y_px), int(stats.x_px)
                y1, x1 = y0 + int(stats.h_px), x0 + int(stats.w_px)
                component = np.asarray(
                    labelled.labels[y0:y1, x0:x1] == stats.label, dtype=bool
                )
                origin = (x0, y0)

                if model is not None and self.param("split_enabled"):
                    footprint, _world, _expected = self._footprint_m(model, bbox, area)
                    too_big = (
                        footprint is not None
                        and footprint.area_m2
                        > float(self.param("split_threshold")) * 4.0 * 1.8
                    )
                    if too_big:
                        pieces = self._split(component, origin)
                        if len(pieces) > 1:
                            split_here += len(pieces) - 1

                if not pieces:
                    pieces = [(bbox, area, component)]

                for piece_bbox, piece_area, piece_mask in pieces:
                    blob = self._build_blob(
                        next_id, piece_bbox, piece_area, piece_mask, origin, model, lanes, ctx
                    )
                    if blob is None:
                        rejected_small += 1
                        continue
                    if model is not None:
                        floor = float(self.param("min_expected_area_px"))
                        if model.expected_area_px(blob.ground_point_px.y) < floor:
                            rejected_far += 1
                            continue
                        lo, hi = model.area_band_px(
                            blob.ground_point_px.y,
                            float(self.param("area_lower")),
                            float(self.param("area_upper")),
                        )
                        if hi <= 0 or piece_area < lo:
                            rejected_small += 1
                            continue
                        if piece_area > hi:
                            rejected_large += 1
                            continue
                        if not self._plausible_shape(blob):
                            rejected_shape += 1
                            continue
                    if self.param("require_lane") and lanes and blob.lane_id is None:
                        rejected_lane += 1
                        continue
                    blobs.append(blob)
                    next_id += 1

            self._split_events += split_here
            ctx.blobs = tuple(blobs)
            ctx.labels = labelled.labels

            result.mask = mask
            result.metrics = {
                "components": float(labelled.count),
                "blobs": float(len(blobs)),
                "rejected_small": float(rejected_small),
                "rejected_large": float(rejected_large),
                "rejected_shape": float(rejected_shape),
                "rejected_beyond_range": float(rejected_far),
                "rejected_outside_lane": float(rejected_lane),
                "splits_this_frame": float(split_here),
                "splits_total": float(self._split_events),
                "calibrated": 1.0 if model is not None else 0.0,
            }
            if model is None:
                result.extras["hint"] = (
                    "no valid calibration, so blobs were accepted on the "
                    "absolute pixel floor alone. Under perspective that either "
                    "admits near-field noise or discards distant vehicles - "
                    "calibrate to make the size filter meaningful."
                )
        return result

    def _plausible_shape(self, blob: Blob) -> bool:
        """Whether a blob's real-world dimensions are those of a road vehicle.

        Checked in metres rather than pixels, so one rule covers the whole
        frame. A shadow sliver and a small car can share a pixel area; they
        cannot share a width.
        """
        footprint = blob.footprint_m
        if footprint is None:
            return True
        return (
            float(self.param("min_width_m"))
            <= footprint.width_m
            <= float(self.param("max_width_m"))
            and float(self.param("min_length_m"))
            <= footprint.length_m
            <= float(self.param("max_length_m"))
        )

    def _split(
        self, component: AnyArray, origin: tuple[int, int]
    ) -> list[tuple[BBox, float, AnyArray]]:
        """Attempt to separate a component that is too large to be one vehicle.

        ``component`` is in its own bounding-box frame; ``origin`` puts the
        resulting boxes back into image coordinates.
        """
        pieces = watershed_split(
            component, min_distance_ratio=float(self.param("split_marker_ratio"))
        )
        ox, oy = origin
        out: list[tuple[BBox, float, AnyArray]] = []
        for value in np.unique(pieces):
            if value <= 0:
                continue
            piece = pieces == value
            ys, xs = np.nonzero(piece)
            if xs.size == 0:
                continue
            bbox = BBox(
                float(xs.min() + ox), float(ys.min() + oy),
                float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1),
            )
            out.append((bbox, float(piece.sum()), piece))
        return out if len(out) > 1 else []

    def _build_blob(
        self,
        blob_id: int,
        bbox: BBox,
        area_px: float,
        component: AnyArray,
        origin: tuple[int, int],
        model: PerspectiveModel | None,
        lanes: tuple[Lane, ...],
        ctx: FrameContext,
    ) -> Blob | None:
        ground = bbox.ground_point_px
        ys, xs = np.nonzero(component)
        if xs.size == 0:
            return None
        centroid = Point2(float(xs.mean() + origin[0]), float(ys.mean() + origin[1]))

        world: Point2 | None = None
        footprint: Footprint | None = None
        expected = 0.0
        if model is not None:
            footprint, world, expected = self._footprint_m(model, bbox, area_px)
            if world is None:
                # Above the horizon: this cannot be a point on the road in front
                # of the camera, so it is not a vehicle wherever it came from.
                return None

        lane_id: int | None = None
        for lane in lanes:
            if point_in_lane(ground, lane):
                lane_id = lane.lane_id
                break

        shadow_px = 0
        if ctx.shadow is not None:
            y0, y1 = int(bbox.y_px), int(bbox.y2_px) + 1
            x0, x1 = int(bbox.x_px), int(bbox.x2_px) + 1
            shadow_px = int(ctx.shadow[y0:y1, x0:x1].sum())

        return Blob(
            blob_id=blob_id,
            bbox_px=bbox,
            area_px=area_px,
            centroid_px=centroid,
            ground_point_px=ground,
            world_pos_m=world,
            footprint_m=footprint,
            expected_area_px=expected,
            lane_id=lane_id,
            fill_ratio=float(area_px / bbox.area_px) if bbox.area_px > 0 else 0.0,
            was_split=False,
            shadow_px=shadow_px,
        )


def expected_footprint_area_px(model: PerspectiveModel, ground_y_px: float) -> float:
    """Projected area of the reference footprint at an image row.

    Exposed for the inspector's pixel probe, which reports what area a vehicle
    *would* have at the point under the cursor - the quickest way to see why a
    blob was accepted or rejected.
    """
    return model.expected_area_px(ground_y_px)


def reference_quad_px(model: PerspectiveModel, x_m: float, y_m: float) -> np.ndarray:
    """The reference footprint projected at a road position, for the overlay."""
    return footprint_quad_px(model.h_world_to_img, x_m, y_m, model.footprint)


def quad_area_px(quad: np.ndarray) -> float:
    return polygon_area_px(quad)
