"""Core data types for the Kerbside pipeline.

Unit convention (see README, Engineering standards): every physical quantity
carries its unit as a suffix in the identifier.

    ``_px``   pixels (image plane, origin top-left, +y downwards)
    ``_m``    metres (road plane, Z = 0)
    ``_s``    seconds
    ``_kmph`` kilometres per hour
    ``_deg``  degrees
    ``_ms``   milliseconds (wall-clock timings)

Mixing pixels and metres silently is the single most likely source of a wrong
answer in this project, so the suffix is mandatory, not cosmetic.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
UInt8Array = NDArray[np.uint8]
BoolArray = NDArray[np.bool_]
Int32Array = NDArray[np.int32]

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class VehicleClass(StrEnum):
    """Footprint-based vehicle classes (S9).

    Classification is by real-world footprint recovered through the
    homography, never by appearance, so the class bands are documented
    constants rather than learned parameters. See :data:`CLASS_BANDS_M`.
    """

    TWO_WHEELER = "two_wheeler"
    CAR = "car"
    LCV = "lcv"
    TRUCK_BUS = "truck_bus"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClassBand:
    """Documented length/width band for one vehicle class.

    ``length_m`` and ``width_m`` are ranges on the real-world footprint.
    ``aspect_max`` is the length/width ratio above which the band is rejected,
    used as a tie-breaker between adjacent bands. These numbers are nominal
    design values for this project, reproduced verbatim in the report; they are
    not a legal or standards-body classification.
    """

    vehicle_class: VehicleClass
    length_m: tuple[float, float]
    width_m: tuple[float, float]
    aspect_max: float


CLASS_BANDS_M: tuple[ClassBand, ...] = (
    ClassBand(VehicleClass.TWO_WHEELER, (0.0, 2.6), (0.0, 1.15), aspect_max=4.0),
    ClassBand(VehicleClass.CAR, (2.6, 5.4), (1.15, 2.15), aspect_max=3.6),
    ClassBand(VehicleClass.LCV, (5.4, 7.6), (1.6, 2.6), aspect_max=4.2),
    ClassBand(VehicleClass.TRUCK_BUS, (7.6, 25.0), (1.9, 3.2), aspect_max=9.0),
)
"""Class bands used by S9. Stated here so the classification is falsifiable."""


class Direction(StrEnum):
    """Travel direction along the road axis (world +Y runs away from camera)."""

    AWAY = "away"
    TOWARD = "toward"
    UNKNOWN = "unknown"


class TrackState(StrEnum):
    """Track lifecycle (S7): tentative to confirmed to coasting to terminated."""

    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    TERMINATED = "terminated"


class DistanceMetric(StrEnum):
    """Unit 1 distance measures, exposed as a tracking parameter so they can be
    compared for real rather than merely mentioned."""

    EUCLIDEAN = "euclidean"
    CITY_BLOCK = "city_block"
    CHESSBOARD = "chessboard"


class FrameType(StrEnum):
    """Unit 8 frame classification driving motion-triggered recording."""

    I = "I"  # noqa: E741 - the letter is the standard name
    P = "P"


# --------------------------------------------------------------------------
# Geometry primitives
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point2:
    """A 2-D point. The suffix on the owning field states which frame it is in."""

    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_array(self) -> FloatArray:
        return np.array([self.x, self.y], dtype=np.float64)

    def __iter__(self) -> Iterator[float]:
        yield self.x
        yield self.y


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in pixels, origin top-left."""

    x_px: float
    y_px: float
    w_px: float
    h_px: float

    @property
    def x2_px(self) -> float:
        return self.x_px + self.w_px

    @property
    def y2_px(self) -> float:
        return self.y_px + self.h_px

    @property
    def area_px(self) -> float:
        return max(0.0, self.w_px) * max(0.0, self.h_px)

    @property
    def centroid_px(self) -> Point2:
        """Blob centroid. NOT on the road plane - see trap 1."""
        return Point2(self.x_px + self.w_px / 2.0, self.y_px + self.h_px / 2.0)

    @property
    def ground_point_px(self) -> Point2:
        """Bottom-centre of the box: the point taken to lie on the road plane.

        Every image-to-world conversion in the pipeline uses this point, never
        the centroid, because the centroid sits above the road plane and does
        not obey the road-plane homography (trap 1).
        """
        return Point2(self.x_px + self.w_px / 2.0, self.y_px + self.h_px)

    @property
    def aspect(self) -> float:
        return self.w_px / self.h_px if self.h_px > 0 else 0.0

    def clipped(self, width_px: int, height_px: int) -> BBox:
        x1 = max(0.0, self.x_px)
        y1 = max(0.0, self.y_px)
        x2 = min(float(width_px), self.x2_px)
        y2 = min(float(height_px), self.y2_px)
        return BBox(x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1))

    def iou(self, other: BBox) -> float:
        x1 = max(self.x_px, other.x_px)
        y1 = max(self.y_px, other.y_px)
        x2 = min(self.x2_px, other.x2_px)
        y2 = min(self.y2_px, other.y2_px)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        union = self.area_px + other.area_px - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def from_points(points_px: FloatArray) -> BBox:
        """Axis-aligned box enclosing an (N, 2) array of image points."""
        if points_px.size == 0:
            return BBox(0.0, 0.0, 0.0, 0.0)
        xs = points_px[:, 0]
        ys = points_px[:, 1]
        return BBox(
            float(xs.min()),
            float(ys.min()),
            float(xs.max() - xs.min()),
            float(ys.max() - ys.min()),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x_px, self.y_px, self.w_px, self.h_px)


@dataclass(frozen=True, slots=True)
class Footprint:
    """Real-world extent of a vehicle on the road plane."""

    length_m: float
    width_m: float
    height_m: float = 0.0

    @property
    def area_m2(self) -> float:
        return self.length_m * self.width_m

    @property
    def aspect(self) -> float:
        return self.length_m / self.width_m if self.width_m > 0 else 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.length_m, self.width_m, self.height_m)


# --------------------------------------------------------------------------
# Camera and calibration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CameraModel:
    """Pinhole camera looking at a road plane.

    World frame: +X across the road (to the right when looking downstream),
    +Y along the road away from the camera, +Z up. The road surface is the
    plane Z = 0. The camera sits at ``(x_m, y_m, height_m)``, tilted
    ``tilt_deg`` below the horizon, yawed ``yaw_deg`` about the vertical and
    rolled ``roll_deg`` about its optical axis.

    This is the *rendering* camera in the synthetic pipeline, so its road-plane
    homography is known in closed form and serves as the ground truth against
    which the calibration recovered from four clicked points is judged.
    """

    width_px: int
    height_px: int
    height_m: float
    tilt_deg: float
    yaw_deg: float = 0.0
    roll_deg: float = 0.0
    fov_deg: float = 55.0
    x_m: float = 0.0
    y_m: float = 0.0

    @property
    def focal_px(self) -> float:
        """Focal length in pixels, from the horizontal field of view."""
        return (self.width_px / 2.0) / float(np.tan(np.deg2rad(self.fov_deg) / 2.0))

    @property
    def principal_point_px(self) -> Point2:
        return Point2(self.width_px / 2.0, self.height_px / 2.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> CameraModel:
        allowed = set(CameraModel.__dataclass_fields__)
        return CameraModel(**{k: v for k, v in d.items() if k in allowed})


@dataclass(frozen=True, slots=True)
class Lane:
    """One lane: an image-space polygon plus a counting line.

    ``polygon_px`` is closed implicitly and is used for point-in-polygon lane
    assignment of a blob's *ground point*. ``counting_line_px`` is the pair of
    endpoints a track must cross, travelling in ``direction``, to be counted.
    """

    lane_id: int
    name: str
    polygon_px: tuple[tuple[float, float], ...]
    counting_line_px: tuple[tuple[float, float], tuple[float, float]]
    direction: Direction = Direction.AWAY
    speed_limit_kmph: float = 60.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane_id": self.lane_id,
            "name": self.name,
            "polygon_px": [list(p) for p in self.polygon_px],
            "counting_line_px": [
                list(self.counting_line_px[0]),
                list(self.counting_line_px[1]),
            ],
            "direction": self.direction.value,
            "speed_limit_kmph": self.speed_limit_kmph,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Lane:
        cl = d["counting_line_px"]
        return Lane(
            lane_id=int(d["lane_id"]),
            name=str(d["name"]),
            polygon_px=tuple((float(p[0]), float(p[1])) for p in d["polygon_px"]),
            counting_line_px=(
                (float(cl[0][0]), float(cl[0][1])),
                (float(cl[1][0]), float(cl[1][1])),
            ),
            direction=Direction(d.get("direction", "away")),
            speed_limit_kmph=float(d.get("speed_limit_kmph", 60.0)),
        )


@dataclass(frozen=True, slots=True)
class Calibration:
    """Road-plane calibration: the bridge between pixels and metres.

    ``h_world_to_img`` maps homogeneous road-plane points ``(x_m, y_m, 1)`` to
    homogeneous image points ``(x_px, y_px, 1)``. It is stored normalised so
    that ``h[2, 2] == 1``.
    """

    h_world_to_img: FloatArray
    width_px: int
    height_px: int
    source: str = "unknown"
    image_points_px: FloatArray | None = None
    world_points_m: FloatArray | None = None
    reprojection_rmse_px: float = 0.0
    reprojection_max_px: float = 0.0
    rect_width_m: float = 0.0
    rect_length_m: float = 0.0
    lanes: tuple[Lane, ...] = ()
    notes: str = ""

    @property
    def h_img_to_world(self) -> FloatArray:
        h_inv = np.linalg.inv(np.asarray(self.h_world_to_img, dtype=np.float64))
        return np.asarray(h_inv / h_inv[2, 2], dtype=np.float64)

    @property
    def is_valid(self) -> bool:
        h = np.asarray(self.h_world_to_img, dtype=np.float64)
        return bool(np.all(np.isfinite(h)) and abs(np.linalg.det(h)) > 1e-12)

    @property
    def horizon_row_px(self) -> float:
        """Image row of the road plane's horizon.

        Rows at or above this are not road surface in front of the camera. The
        inverse homography still returns a finite number there - a point just
        above the horizon maps to somewhere behind the camera or kilometres
        away - so anything converting pixels to metres has to test this
        explicitly rather than relying on a non-finite result to signal it.
        """
        h_inv = self.h_img_to_world
        a, b, c = h_inv[2, 0], h_inv[2, 1], h_inv[2, 2]
        if abs(b) < 1e-12:
            return float("-inf")
        return float(-(a * (self.width_px / 2.0) + c) / b)

    def is_on_road_plane(self, x_px: float, y_px: float, margin_px: float = 1.0) -> bool:
        """Whether an image point can be a point on the road in front of the camera."""
        return bool(np.isfinite(y_px) and y_px > self.horizon_row_px + margin_px)

    @property
    def quality_verdict(self) -> str:
        """Plain-language verdict on the calibration residual (interface 6.3)."""
        if not self.is_valid:
            return "invalid - the four points are degenerate (collinear or repeated)"
        r = self.reprojection_rmse_px
        if r < 0.5:
            return "excellent - residual below half a pixel"
        if r < 1.5:
            return "good - residual around one pixel"
        if r < 4.0:
            return "marginal - check the clicked points lie on the road surface"
        return "poor - re-click the quadrilateral; speeds will be unreliable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "h_world_to_img": np.asarray(self.h_world_to_img).tolist(),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "source": self.source,
            "image_points_px": (
                None
                if self.image_points_px is None
                else np.asarray(self.image_points_px).tolist()
            ),
            "world_points_m": (
                None
                if self.world_points_m is None
                else np.asarray(self.world_points_m).tolist()
            ),
            "reprojection_rmse_px": self.reprojection_rmse_px,
            "reprojection_max_px": self.reprojection_max_px,
            "rect_width_m": self.rect_width_m,
            "rect_length_m": self.rect_length_m,
            "lanes": [lane.to_dict() for lane in self.lanes],
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Calibration:
        return Calibration(
            h_world_to_img=np.asarray(d["h_world_to_img"], dtype=np.float64),
            width_px=int(d["width_px"]),
            height_px=int(d["height_px"]),
            source=str(d.get("source", "unknown")),
            image_points_px=(
                None
                if d.get("image_points_px") is None
                else np.asarray(d["image_points_px"], dtype=np.float64)
            ),
            world_points_m=(
                None
                if d.get("world_points_m") is None
                else np.asarray(d["world_points_m"], dtype=np.float64)
            ),
            reprojection_rmse_px=float(d.get("reprojection_rmse_px", 0.0)),
            reprojection_max_px=float(d.get("reprojection_max_px", 0.0)),
            rect_width_m=float(d.get("rect_width_m", 0.0)),
            rect_length_m=float(d.get("rect_length_m", 0.0)),
            lanes=tuple(Lane.from_dict(x) for x in d.get("lanes", [])),
            notes=str(d.get("notes", "")),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> Calibration:
        return Calibration.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def with_lanes(self, lanes: tuple[Lane, ...]) -> Calibration:
        return replace(self, lanes=lanes)


# --------------------------------------------------------------------------
# Pipeline products
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Blob:
    """A connected component accepted by S6 as a candidate vehicle."""

    blob_id: int
    bbox_px: BBox
    area_px: float
    centroid_px: Point2
    ground_point_px: Point2
    world_pos_m: Point2 | None = None
    footprint_m: Footprint | None = None
    expected_area_px: float = 0.0
    lane_id: int | None = None
    fill_ratio: float = 0.0
    was_split: bool = False
    shadow_px: int = 0


@dataclass(frozen=True, slots=True)
class Detection:
    """A blob promoted to a detection for the tracker."""

    frame_index: int
    time_s: float
    blob: Blob


@dataclass(slots=True)
class Track:
    """A tracked vehicle (S7) with both speed routes (S8) attached."""

    track_id: int
    state: TrackState = TrackState.TENTATIVE
    hits: int = 0
    misses: int = 0
    age_frames: int = 0
    first_frame: int = 0
    last_frame: int = 0
    lane_id: int | None = None
    direction: Direction = Direction.UNKNOWN
    counted: bool = False
    counted_frame: int | None = None
    vehicle_class: VehicleClass = VehicleClass.UNKNOWN
    footprint_m: Footprint | None = None
    colour_name: str = "unknown"
    colour_hsv: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bbox_px: BBox | None = None
    ground_point_px: Point2 | None = None
    world_pos_m: Point2 | None = None
    velocity_mps: tuple[float, float] = (0.0, 0.0)
    speed_kmph_centroid: float = 0.0
    speed_kmph_blockmatch: float = 0.0
    mean_speed_kmph_centroid: float = 0.0
    mean_speed_kmph_blockmatch: float = 0.0
    history_px: list[tuple[float, float]] = field(default_factory=list)
    history_world_m: list[tuple[float, float]] = field(default_factory=list)
    history_time_s: list[float] = field(default_factory=list)
    speed_history_kmph: list[float] = field(default_factory=list)
    gt_vehicle_id: int | None = None


@dataclass(slots=True)
class StageResult:
    """The output of one pipeline stage, with the metadata the UI needs.

    ``units`` are the syllabus units the stage exercises and ``rationale`` is
    the one-sentence justification surfaced in the inspector. Carrying both on
    the result rather than in a distant lookup table keeps the explanation next
    to the thing it explains.
    """

    stage_id: str
    name: str
    units: tuple[int, ...]
    rationale: str
    image: UInt8Array | None = None
    mask: BoolArray | None = None
    elapsed_ms: float = 0.0
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


# --------------------------------------------------------------------------
# Ground truth (synthetic renderer)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GTVehicleFrame:
    """Exact ground truth for one vehicle in one frame.

    Two ground points are recorded deliberately:

    ``ground_point_px``
        the projection of ``world_pos_m`` through the rendering homography -
        the point that genuinely lies on the road plane, and the value the
        Phase 1 gate checks.
    ``bbox_ground_point_px``
        bottom-centre of the image bounding box - the observable the pipeline
        can actually measure from a silhouette.

    Their difference, ``ground_point_bias_px``, is trap 1 and trap 11 made
    measurable: the systematic error a real estimator inherits from vehicle
    height, and later from motion blur stretching the silhouette.
    """

    vehicle_id: int
    world_pos_m: tuple[float, float]
    ground_point_px: tuple[float, float]
    bbox_px: tuple[float, float, float, float]
    bbox_ground_point_px: tuple[float, float]
    ground_point_bias_px: tuple[float, float]
    speed_kmph: float
    heading_deg: float
    lane_id: int
    direction: Direction
    vehicle_class: VehicleClass
    footprint_m: tuple[float, float, float]
    colour_name: str
    body_rgb: tuple[int, int, int]
    occlusion_frac: float
    visible: bool
    visible_area_px: int = 0
    shadow_area_px: int = 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["vehicle_class"] = self.vehicle_class.value
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GTVehicleFrame:
        return GTVehicleFrame(
            vehicle_id=int(d["vehicle_id"]),
            world_pos_m=(float(d["world_pos_m"][0]), float(d["world_pos_m"][1])),
            ground_point_px=(
                float(d["ground_point_px"][0]),
                float(d["ground_point_px"][1]),
            ),
            bbox_px=(
                float(d["bbox_px"][0]),
                float(d["bbox_px"][1]),
                float(d["bbox_px"][2]),
                float(d["bbox_px"][3]),
            ),
            bbox_ground_point_px=(
                float(d["bbox_ground_point_px"][0]),
                float(d["bbox_ground_point_px"][1]),
            ),
            ground_point_bias_px=(
                float(d["ground_point_bias_px"][0]),
                float(d["ground_point_bias_px"][1]),
            ),
            speed_kmph=float(d["speed_kmph"]),
            heading_deg=float(d["heading_deg"]),
            lane_id=int(d["lane_id"]),
            direction=Direction(d["direction"]),
            vehicle_class=VehicleClass(d["vehicle_class"]),
            footprint_m=(
                float(d["footprint_m"][0]),
                float(d["footprint_m"][1]),
                float(d["footprint_m"][2]),
            ),
            colour_name=str(d["colour_name"]),
            body_rgb=(
                int(d["body_rgb"][0]),
                int(d["body_rgb"][1]),
                int(d["body_rgb"][2]),
            ),
            occlusion_frac=float(d["occlusion_frac"]),
            visible=bool(d["visible"]),
            visible_area_px=int(d.get("visible_area_px", 0)),
            shadow_area_px=int(d.get("shadow_area_px", 0)),
        )


@dataclass(frozen=True, slots=True)
class GTFrame:
    """Ground truth for one rendered frame."""

    frame_index: int
    time_s: float
    vehicles: tuple[GTVehicleFrame, ...]
    shake_offset_px: tuple[float, float] = (0.0, 0.0)
    illumination: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "time_s": self.time_s,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "shake_offset_px": list(self.shake_offset_px),
            "illumination": self.illumination,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GTFrame:
        so = d.get("shake_offset_px", [0.0, 0.0])
        return GTFrame(
            frame_index=int(d["frame_index"]),
            time_s=float(d["time_s"]),
            vehicles=tuple(GTVehicleFrame.from_dict(v) for v in d["vehicles"]),
            shake_offset_px=(float(so[0]), float(so[1])),
            illumination=float(d.get("illumination", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class GroundTruth:
    """Complete ground truth for a rendered clip.

    ``h_world_to_img`` is the exact rendering homography. The Phase 1 gate
    recomputes it from ``camera`` and checks it reproduces every recorded
    ground point to within a pixel, and that a from-scratch DLT fit to four
    road-plane correspondences recovers it to within 2%.
    """

    scenario: str
    width_px: int
    height_px: int
    fps: float
    seed: int
    camera: CameraModel
    h_world_to_img: FloatArray
    lanes: tuple[Lane, ...]
    frames: tuple[GTFrame, ...]
    lane_marking_pitch_m: float = 9.0
    road_width_m: float = 14.0
    notes: str = ""

    @property
    def duration_s(self) -> float:
        return len(self.frames) / self.fps if self.fps > 0 else 0.0

    def counts_by_lane(self) -> dict[int, int]:
        """Unique vehicle ids ever visible, per lane - the counting oracle."""
        seen: dict[int, set[int]] = {}
        for fr in self.frames:
            for v in fr.vehicles:
                if v.visible:
                    seen.setdefault(v.lane_id, set()).add(v.vehicle_id)
        return {k: len(s) for k, s in sorted(seen.items())}

    @property
    def total_vehicles(self) -> int:
        ids = {v.vehicle_id for fr in self.frames for v in fr.vehicles if v.visible}
        return len(ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "width_px": self.width_px,
            "height_px": self.height_px,
            "fps": self.fps,
            "seed": self.seed,
            "camera": self.camera.to_dict(),
            "h_world_to_img": np.asarray(self.h_world_to_img).tolist(),
            "lanes": [lane.to_dict() for lane in self.lanes],
            "lane_marking_pitch_m": self.lane_marking_pitch_m,
            "road_width_m": self.road_width_m,
            "notes": self.notes,
            "frames": [f.to_dict() for f in self.frames],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> GroundTruth:
        return GroundTruth(
            scenario=str(d["scenario"]),
            width_px=int(d["width_px"]),
            height_px=int(d["height_px"]),
            fps=float(d["fps"]),
            seed=int(d["seed"]),
            camera=CameraModel.from_dict(d["camera"]),
            h_world_to_img=np.asarray(d["h_world_to_img"], dtype=np.float64),
            lanes=tuple(Lane.from_dict(x) for x in d["lanes"]),
            frames=tuple(GTFrame.from_dict(x) for x in d["frames"]),
            lane_marking_pitch_m=float(d.get("lane_marking_pitch_m", 9.0)),
            road_width_m=float(d.get("road_width_m", 14.0)),
            notes=str(d.get("notes", "")),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> GroundTruth:
        return GroundTruth.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


RESOLUTIONS_PX: dict[str, tuple[int, int]] = {
    "1080p": (1920, 1080),
    "720p": (1280, 720),
    "480p": (854, 480),
    "360p": (640, 360),
}
"""Standard resolutions for the Unit 1 resolution study."""
