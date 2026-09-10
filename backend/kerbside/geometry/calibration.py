"""Road-plane calibration from four clicked points.

Speed is meaningless without geometry. This module turns four points the user
clicks on the road surface, plus the two real dimensions of the rectangle they
bound, into the homography every metric quantity downstream depends on - and,
just as importantly, into an honest account of how much that homography can be
trusted.

Assist, do not assume. :func:`reference_distance_suggestions` offers common road
dimensions as *starting points the user must verify against their own road*.
Lane widths and dashed-marking pitches vary by country, by road class and by the
age of the markings; a tape measure is the ground truth, not this table.

Syllabus context: Unit 1 (coordinates and distance measures) and Unit 5 (the
Canny + Hough lane-marking detection that proposes lane polygons).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kerbside.core.types import Calibration, Direction, FloatArray, Lane
from kerbside.geometry.homography import (
    apply_h,
    apply_h_single,
    dlt_homography,
    normalise_h,
    reprojection_stats,
)
from kerbside.geometry.perspective import (
    estimate_focal_px_from_homography,
    vertical_model_from_homography,
)

__all__ = [
    "ReferenceDistance",
    "REFERENCE_DISTANCE_SUGGESTIONS",
    "CameraPlausibility",
    "CalibrationReport",
    "order_quad_points",
    "calibrate_from_rectangle",
    "calibrate_from_correspondences",
    "measure_distance_m",
    "metre_grid_segments_px",
    "birdseye_homography",
    "decompose_camera_from_homography",
    "assess_calibration",
    "lanes_from_world_bands",
    "reference_distance_suggestions",
    "calibration_path_for",
]


# --------------------------------------------------------------------------
# Reference distances - suggestions, never truth
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReferenceDistance:
    """A commonly encountered road dimension, offered as a starting point."""

    name: str
    nominal_m: float
    plausible_range_m: tuple[float, float]
    caution: str


REFERENCE_DISTANCE_SUGGESTIONS: tuple[ReferenceDistance, ...] = (
    ReferenceDistance(
        name="Single carriageway lane width",
        nominal_m=3.5,
        plausible_range_m=(2.5, 3.75),
        caution="Varies by road class and country; urban lanes are often narrower.",
    ),
    ReferenceDistance(
        name="Dashed centre-line marking length",
        nominal_m=3.0,
        plausible_range_m=(1.0, 5.0),
        caution="Marking length and gap differ between standards and wear down.",
    ),
    ReferenceDistance(
        name="Dashed marking pitch (mark plus gap)",
        nominal_m=9.0,
        plausible_range_m=(4.0, 18.0),
        caution="Frequently changes with speed limit; count several to average.",
    ),
    ReferenceDistance(
        name="Passenger car length",
        nominal_m=4.4,
        plausible_range_m=(3.4, 5.2),
        caution="Only for a rough sanity check - never as a calibration input.",
    ),
)


def reference_distance_suggestions() -> tuple[ReferenceDistance, ...]:
    """Suggestions the user must verify against their own road.

    Surfaced in the calibration view as prompts, with the caution text attached,
    so nothing here can be mistaken for a measured value.
    """
    return REFERENCE_DISTANCE_SUGGESTIONS


# --------------------------------------------------------------------------
# Building a calibration
# --------------------------------------------------------------------------


def order_quad_points(points_px: FloatArray) -> FloatArray:
    """Put four clicked points into the canonical order.

    Canonical order is near-left, near-right, far-right, far-left, where 'near'
    means larger image row (closer to the camera on a downward-looking view).
    Accepting clicks in any order removes a whole class of silent calibration
    errors, where a transposed pair produces a homography that is a valid matrix
    but folds the road over itself.
    """
    pts = np.asarray(points_px, dtype=np.float64)
    if pts.shape != (4, 2):
        raise ValueError(f"expected exactly 4 points, got {pts.shape}")
    by_row = pts[np.argsort(pts[:, 1])]
    far = by_row[:2][np.argsort(by_row[:2, 0])]  # smaller row = further away
    near = by_row[2:][np.argsort(by_row[2:, 0])]
    return np.array([near[0], near[1], far[1], far[0]], dtype=np.float64)


def calibrate_from_correspondences(
    image_points_px: FloatArray,
    world_points_m: FloatArray,
    width_px: int,
    height_px: int,
    source: str = "manual",
    notes: str = "",
) -> Calibration:
    """Fit the world-to-image homography to explicit correspondences.

    The fit is the from-scratch normalised DLT; the residuals are computed in
    pixels, which is the unit the user can judge against what they clicked.
    """
    img = np.asarray(image_points_px, dtype=np.float64)
    world = np.asarray(world_points_m, dtype=np.float64)
    if img.shape != world.shape or img.ndim != 2 or img.shape[1] != 2:
        raise ValueError(
            f"need matching (N, 2) point sets, got {img.shape} and {world.shape}"
        )
    if img.shape[0] < 4:
        raise ValueError(
            f"a road-plane calibration needs at least 4 points, got {img.shape[0]}"
        )

    h = dlt_homography(world, img)
    stats = reprojection_stats(h, world, img)
    return Calibration(
        h_world_to_img=h,
        width_px=width_px,
        height_px=height_px,
        source=source,
        image_points_px=img,
        world_points_m=world,
        reprojection_rmse_px=stats.rmse_px,
        reprojection_max_px=stats.max_px,
        notes=notes,
    )


def calibrate_from_rectangle(
    image_points_px: FloatArray,
    rect_width_m: float,
    rect_length_m: float,
    width_px: int,
    height_px: int,
    origin_m: tuple[float, float] = (0.0, 0.0),
    source: str = "manual",
    reorder: bool = True,
    notes: str = "",
) -> Calibration:
    """Calibrate from four points bounding a rectangle of known real size.

    ``rect_width_m`` is measured across the road (world X) and ``rect_length_m``
    along it (world Y). ``origin_m`` places the near-left corner in world
    coordinates, which lets a user calibrate on a patch of road that is not at
    the world origin without disturbing the lane definitions.

    A four-point fit has exactly enough correspondences to determine a
    homography, so its reprojection residual is zero by construction and says
    nothing about accuracy. Judge the result with :func:`measure_distance_m`
    against a tape measure, and with :func:`assess_calibration`, which checks
    whether the implied camera is physically plausible.
    """
    if rect_width_m <= 0 or rect_length_m <= 0:
        raise ValueError(
            f"rectangle dimensions must be positive, got "
            f"width {rect_width_m} m and length {rect_length_m} m"
        )
    img = np.asarray(image_points_px, dtype=np.float64)
    if img.shape != (4, 2):
        raise ValueError(f"expected exactly 4 clicked points, got {img.shape}")
    if reorder:
        img = order_quad_points(img)

    ox, oy = origin_m
    world = np.array(
        [
            [ox, oy],
            [ox + rect_width_m, oy],
            [ox + rect_width_m, oy + rect_length_m],
            [ox, oy + rect_length_m],
        ],
        dtype=np.float64,
    )
    calib = calibrate_from_correspondences(
        img, world, width_px, height_px, source=source, notes=notes
    )
    return Calibration(
        h_world_to_img=calib.h_world_to_img,
        width_px=width_px,
        height_px=height_px,
        source=source,
        image_points_px=img,
        world_points_m=world,
        reprojection_rmse_px=calib.reprojection_rmse_px,
        reprojection_max_px=calib.reprojection_max_px,
        rect_width_m=rect_width_m,
        rect_length_m=rect_length_m,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Validation tools
# --------------------------------------------------------------------------


def measure_distance_m(
    calib: Calibration, p1_px: tuple[float, float], p2_px: tuple[float, float]
) -> float | None:
    """Real-world distance between two clicked image points on the road plane.

    This is the validation tool of interface 6.3: the user clicks two points a
    known distance apart and compares. It is the only check that catches a wrong
    *scale* - a homography fitted to a quadrilateral whose real dimensions were
    entered incorrectly is perfectly self-consistent and will still produce
    confidently wrong speeds.

    Returns ``None`` if either point is above the horizon, where the road plane
    is not imaged at all. That check has to be explicit: the inverse homography
    returns a perfectly finite number for a point above the horizon, so without
    it the tool would answer a question it cannot answer, with confidence.
    """
    if not (
        calib.is_on_road_plane(p1_px[0], p1_px[1])
        and calib.is_on_road_plane(p2_px[0], p2_px[1])
    ):
        return None
    h_inv = calib.h_img_to_world
    w1 = apply_h_single(h_inv, p1_px[0], p1_px[1])
    w2 = apply_h_single(h_inv, p2_px[0], p2_px[1])
    if not all(np.isfinite(v) for v in (*w1, *w2)):
        return None
    return float(np.hypot(w2[0] - w1[0], w2[1] - w1[1]))


def metre_grid_segments_px(
    calib: Calibration,
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    step_m: float = 1.0,
    samples_per_line: int = 24,
) -> list[list[tuple[float, float]]]:
    """Project a metre grid on the road surface, as polylines in image space.

    Each grid line is sampled rather than drawn as a straight segment because
    although a straight world line does image to a straight image line, sampling
    lets the caller clip it against the horizon cleanly and keeps the drawing
    code free of projective special cases.

    Points that fall behind the camera or above the horizon are dropped, which
    splits a line into several polylines - hence the list-of-polylines return.
    """
    if step_m <= 0:
        raise ValueError(f"grid step must be positive, got {step_m} m")
    h = np.asarray(calib.h_world_to_img, dtype=np.float64)
    polylines: list[list[tuple[float, float]]] = []

    def emit(world_pts: FloatArray) -> None:
        img = apply_h(h, world_pts)
        run: list[tuple[float, float]] = []
        for (u, v) in img:
            inside = (
                np.isfinite(u)
                and np.isfinite(v)
                and -calib.width_px <= u <= 2 * calib.width_px
                and -calib.height_px <= v <= 2 * calib.height_px
            )
            if inside:
                run.append((float(u), float(v)))
            elif len(run) >= 2:
                polylines.append(run)
                run = []
            else:
                run = []
        if len(run) >= 2:
            polylines.append(run)

    xs = np.arange(x_range_m[0], x_range_m[1] + step_m * 0.5, step_m)
    ys = np.arange(y_range_m[0], y_range_m[1] + step_m * 0.5, step_m)
    along = np.linspace(y_range_m[0], y_range_m[1], samples_per_line)
    across = np.linspace(x_range_m[0], x_range_m[1], samples_per_line)
    for x in xs:
        emit(np.column_stack([np.full(samples_per_line, x), along]))
    for y in ys:
        emit(np.column_stack([across, np.full(samples_per_line, y)]))
    return polylines


def birdseye_homography(
    calib: Calibration,
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    px_per_m: float = 8.0,
) -> tuple[FloatArray, int, int]:
    """Image-to-birdseye warp for the rectified view, with its output size.

    The birdseye image has world +X to the right and world +Y *upwards*, so that
    traffic moving away from the camera moves up the rectified image, which is
    how a plan view is normally read.
    """
    if px_per_m <= 0:
        raise ValueError(f"px_per_m must be positive, got {px_per_m}")
    width_m = x_range_m[1] - x_range_m[0]
    length_m = y_range_m[1] - y_range_m[0]
    if width_m <= 0 or length_m <= 0:
        raise ValueError("birdseye ranges must be increasing")
    out_w = max(1, int(round(width_m * px_per_m)))
    out_h = max(1, int(round(length_m * px_per_m)))
    world_to_bev = np.array(
        [
            [px_per_m, 0.0, -x_range_m[0] * px_per_m],
            [0.0, -px_per_m, y_range_m[1] * px_per_m],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    img_to_bev = world_to_bev @ np.asarray(calib.h_img_to_world, dtype=np.float64)
    return normalise_h(img_to_bev), out_w, out_h


@dataclass(frozen=True, slots=True)
class CameraPlausibility:
    """The camera implied by a calibration, for a physical sanity check.

    A homography fitted to four badly clicked points is still a valid matrix.
    What it is not is a matrix produced by a camera on a pole: it implies an
    absurd focal length, a negative mounting height, or a tilt that would point
    the lens at the sky. Decomposing it and showing the numbers lets the user
    compare against the pole they can see.
    """

    focal_px: float | None
    height_m: float | None
    tilt_deg: float | None
    is_plausible: bool
    reasons: tuple[str, ...]


def decompose_camera_from_homography(
    h: FloatArray, width_px: int, height_px: int
) -> CameraPlausibility:
    """Recover approximate focal length, mounting height and tilt from ``H``.

    Assumes square pixels and a principal point at the image centre, which is a
    good approximation for the vast majority of fixed CCTV installations and is
    stated as an assumption rather than hidden.
    """
    reasons: list[str] = []
    h = normalise_h(np.asarray(h, dtype=np.float64))
    focal = estimate_focal_px_from_homography(h, width_px, height_px)
    if focal is None:
        return CameraPlausibility(
            None, None, None, False,
            ("focal length is not recoverable - the view is too close to "
             "fronto-parallel, or the clicked quadrilateral is degenerate",),
        )

    vm = vertical_model_from_homography(h, width_px, height_px, focal)
    if vm is None:
        return CameraPlausibility(
            focal, None, None, False,
            ("the vertical direction is not recoverable from this homography",),
        )

    cx, cy = width_px / 2.0, height_px / 2.0
    k_inv = np.array(
        [[1.0 / focal, 0.0, -cx / focal], [0.0, 1.0 / focal, -cy / focal], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    m = k_inv @ h
    mu = vm.scale_s
    r1 = m[:, 0] / mu
    r2 = m[:, 1] / mu
    r3 = np.cross(r1, r2)
    t = m[:, 2] / mu
    r = np.column_stack([r1, r2, r3])
    centre = -r.T @ t
    height_m = float(centre[2])
    optical_axis_world = r.T @ np.array([0.0, 0.0, 1.0])
    tilt_deg = float(np.rad2deg(np.arcsin(np.clip(-optical_axis_world[2], -1.0, 1.0))))

    if not (0.2 * width_px <= focal <= 12.0 * width_px):
        reasons.append(
            f"implied focal length {focal:.0f} px is outside the range a real "
            f"lens would give on a {width_px} px wide sensor"
        )
    if not (1.5 <= height_m <= 25.0):
        reasons.append(
            f"implied mounting height {height_m:.1f} m is not a plausible pole height"
        )
    if not (1.0 <= tilt_deg <= 60.0):
        reasons.append(
            f"implied tilt {tilt_deg:.1f} degrees below horizontal is not a "
            f"plausible roadside camera angle"
        )
    return CameraPlausibility(
        focal_px=float(focal),
        height_m=height_m,
        tilt_deg=tilt_deg,
        is_plausible=not reasons,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Everything the calibration view needs to tell the user where it stands."""

    reprojection_rmse_px: float
    reprojection_max_px: float
    verdict: str
    exactly_determined: bool
    camera: CameraPlausibility
    horizon_row_px: float
    covered_y_range_m: tuple[float, float] | None
    warnings: tuple[str, ...]


def assess_calibration(calib: Calibration) -> CalibrationReport:
    """Assemble the quality verdict, plausibility check and coverage extent."""
    warnings: list[str] = []
    n_points = (
        0
        if calib.image_points_px is None
        else int(np.asarray(calib.image_points_px).shape[0])
    )
    exactly_determined = n_points == 4
    if exactly_determined:
        warnings.append(
            "Four points determine a homography exactly, so the reprojection "
            "residual is zero by construction and is not evidence of accuracy. "
            "Verify the scale with the two-point distance tool."
        )

    camera = decompose_camera_from_homography(
        calib.h_world_to_img, calib.width_px, calib.height_px
    )
    warnings.extend(camera.reasons)

    h_inv = calib.h_img_to_world
    u = calib.width_px / 2.0
    horizon = calib.horizon_row_px

    covered: tuple[float, float] | None = None
    near = apply_h_single(h_inv, u, float(calib.height_px - 1))
    far = apply_h_single(h_inv, u, float(min(calib.height_px - 1, horizon + 12.0)))
    if np.isfinite(near[1]) and np.isfinite(far[1]):
        covered = (float(min(near[1], far[1])), float(max(near[1], far[1])))
        if covered[1] - covered[0] < 5.0:
            warnings.append(
                f"the calibrated road plane only spans {covered[1] - covered[0]:.1f} m "
                f"of view - speeds will be estimated from very short tracks"
            )

    return CalibrationReport(
        reprojection_rmse_px=calib.reprojection_rmse_px,
        reprojection_max_px=calib.reprojection_max_px,
        verdict=calib.quality_verdict,
        exactly_determined=exactly_determined,
        camera=camera,
        horizon_row_px=horizon,
        covered_y_range_m=covered,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------
# Lanes
# --------------------------------------------------------------------------


def lanes_from_world_bands(
    calib: Calibration,
    bands_m: list[tuple[float, float]],
    y_range_m: tuple[float, float],
    directions: list[Direction] | None = None,
    counting_line_y_m: float | None = None,
    speed_limit_kmph: float = 60.0,
    names: list[str] | None = None,
) -> tuple[Lane, ...]:
    """Build image-space lane polygons from lane edges given in metres.

    Defining lanes in world coordinates and projecting them - rather than
    clicking polygons directly in the image - keeps the lane edges parallel on
    the road, which is what they actually are, and makes the counting lines
    exactly perpendicular to the direction of travel.
    """
    h = np.asarray(calib.h_world_to_img, dtype=np.float64)
    y0, y1 = y_range_m
    if counting_line_y_m is None:
        counting_line_y_m = y0 + 0.45 * (y1 - y0)

    lanes: list[Lane] = []
    for idx, (x0, x1) in enumerate(bands_m):
        edge_samples = 8
        ys = np.linspace(y0, y1, edge_samples)
        left = np.column_stack([np.full(edge_samples, x0), ys])
        right = np.column_stack([np.full(edge_samples, x1), ys[::-1]])
        poly_world = np.vstack([left, right])
        poly_img = apply_h(h, poly_world)
        poly = tuple(
            (float(p[0]), float(p[1])) for p in poly_img if np.isfinite(p).all()
        )
        if len(poly) < 3:
            continue
        cl = apply_h(
            h,
            np.array([[x0, counting_line_y_m], [x1, counting_line_y_m]], dtype=np.float64),
        )
        direction = (
            directions[idx] if directions and idx < len(directions) else Direction.AWAY
        )
        name = names[idx] if names and idx < len(names) else f"Lane {idx + 1}"
        lanes.append(
            Lane(
                lane_id=idx,
                name=name,
                polygon_px=poly,
                counting_line_px=(
                    (float(cl[0, 0]), float(cl[0, 1])),
                    (float(cl[1, 0]), float(cl[1, 1])),
                ),
                direction=direction,
                speed_limit_kmph=speed_limit_kmph,
            )
        )
    return tuple(lanes)


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def calibration_path_for(key: str, root: str | Path = "data/calib") -> Path:
    """Path of the saved calibration for a camera or clip key.

    Calibrations are keyed by source rather than stored in one blob so that
    swapping clips cannot silently apply the wrong geometry - which would be
    invisible in the UI and catastrophic for every speed on screen.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in key).strip("_")
    if not safe:
        raise ValueError(f"calibration key {key!r} has no usable characters")
    return Path(root) / f"{safe}.json"
