"""Planar homography: from-scratch normalised DLT, plus the closed-form
road-plane homography of a pinhole camera.

Syllabus context: Unit 1 (image as a matrix, coordinate systems) and the
geometric foundation every metric quantity in the project rests on. Speed,
footprint-based class and the perspective-scaled structuring elements are all
downstream of this file, so it is written and tested first.

Conventions
-----------
World frame  +X across the road, +Y along the road away from the camera,
             +Z up. The road surface is the plane Z = 0.
Image frame  x to the right, y downwards, origin at the top-left pixel centre.
Homography   ``h_world_to_img`` maps ``(x_m, y_m, 1)`` to ``(x_px, y_px, 1)``
             up to scale, and is stored normalised so ``h[2, 2] == 1``.

``cv2.findHomography`` appears once, in :func:`cv2_homography`, purely as a
test oracle for the from-scratch implementation. It is never on the production
path - the point of the exercise is that the DLT was written here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kerbside.core.types import CameraModel, FloatArray

__all__ = [
    "ReprojectionStats",
    "HomographyComparison",
    "apply_h",
    "apply_h_single",
    "invert_h",
    "normalise_h",
    "dlt_homography",
    "normalise_points",
    "reprojection_stats",
    "rotation_world_to_camera",
    "camera_matrix",
    "camera_projection_matrix",
    "homography_from_camera",
    "project_world_points",
    "horizon_row_px",
    "compare_homographies",
    "cv2_homography",
]


# --------------------------------------------------------------------------
# Point mapping
# --------------------------------------------------------------------------


def normalise_h(h: FloatArray) -> FloatArray:
    """Scale a homography so that ``h[2, 2] == 1``.

    A homography is defined only up to scale, so two matrices that differ by a
    constant factor are the same transform. Normalising makes them comparable
    element-wise and makes stored JSON stable.
    """
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (3, 3):
        raise ValueError(f"homography must be 3x3, got {h.shape}")
    denom = h[2, 2]
    if abs(denom) < 1e-15:
        # Degenerate normalisation target; fall back to unit Frobenius norm so
        # the caller still gets a finite matrix rather than an inf/nan cascade.
        norm = float(np.linalg.norm(h))
        if norm < 1e-15:
            raise ValueError("homography is the zero matrix")
        return np.asarray(h / norm, dtype=np.float64)
    return np.asarray(h / denom, dtype=np.float64)


def apply_h(h: FloatArray, points: FloatArray) -> FloatArray:
    """Map an ``(N, 2)`` array of points through a homography.

    Points whose mapped homogeneous weight is ~0 lie on the horizon line of the
    transform (infinitely far away on the plane) and are returned as ``inf``
    rather than silently wrapping to a huge finite coordinate.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {pts.shape}")
    h = np.asarray(h, dtype=np.float64)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    mapped = homo @ h.T
    w = mapped[:, 2]
    out = np.full((pts.shape[0], 2), np.inf, dtype=np.float64)
    finite = np.abs(w) > 1e-12
    out[finite] = mapped[finite, :2] / w[finite, None]
    return out


def apply_h_single(h: FloatArray, x: float, y: float) -> tuple[float, float]:
    """Map one point. Convenience wrapper over :func:`apply_h`."""
    mapped = apply_h(h, np.array([[x, y]], dtype=np.float64))
    return (float(mapped[0, 0]), float(mapped[0, 1]))


def invert_h(h: FloatArray) -> FloatArray:
    """Invert a homography, returning the result normalised."""
    h = np.asarray(h, dtype=np.float64)
    if abs(float(np.linalg.det(h))) < 1e-15:
        raise ValueError("homography is singular and cannot be inverted")
    return normalise_h(np.asarray(np.linalg.inv(h), dtype=np.float64))


# --------------------------------------------------------------------------
# Normalised DLT
# --------------------------------------------------------------------------


def normalise_points(points: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Hartley normalisation: translate to the centroid, scale to mean distance
    ``sqrt(2)`` from the origin.

    Returns ``(T, normalised_points)`` where ``T`` is the ``3 x 3`` similarity
    transform such that ``normalised = T @ homogeneous(points)``.

    Without this step the DLT design matrix mixes columns of wildly different
    magnitude - pixel coordinates near 1000 against a homogeneous 1 - and the
    smallest singular vector is dominated by round-off. Normalising is what
    makes the from-scratch DLT numerically equal to the library one.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {pts.shape}")
    centroid = pts.mean(axis=0)
    centred = pts - centroid
    mean_dist = float(np.sqrt((centred**2).sum(axis=1)).mean())
    scale = float(np.sqrt(2.0) / mean_dist) if mean_dist > 1e-12 else 1.0
    t = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalised = centred * scale
    return t, normalised


def dlt_homography(src_points: FloatArray, dst_points: FloatArray) -> FloatArray:
    """Estimate the homography mapping ``src`` to ``dst`` by normalised DLT.

    Mathematical definition. For a correspondence ``x <-> x'`` the homography
    satisfies ``x' = H x`` up to scale, so ``x' x (H x) = 0``. Writing ``H`` as
    the 9-vector ``h`` and expanding the cross product gives two independent
    linear equations per correspondence:

        [ -x, -y, -1,  0,  0,  0, x'x, x'y, x' ] h = 0
        [  0,  0,  0, -x, -y, -1, y'x, y'y, y' ] h = 0

    Stacking ``N >= 4`` correspondences gives ``A h = 0``; the least-squares
    solution under ``||h|| = 1`` is the right singular vector of ``A`` with the
    smallest singular value.

    Parameters
    ----------
    src_points, dst_points
        ``(N, 2)`` arrays with ``N >= 4`` matching points, in any consistent
        units. No three source points may be collinear.
    """
    src = np.asarray(src_points, dtype=np.float64)
    dst = np.asarray(dst_points, dtype=np.float64)
    if src.shape != dst.shape:
        raise ValueError(f"point sets must match in shape: {src.shape} vs {dst.shape}")
    if src.ndim != 2 or src.shape[1] != 2:
        raise ValueError(f"points must be (N, 2), got {src.shape}")
    n = src.shape[0]
    if n < 4:
        raise ValueError(f"a homography needs at least 4 correspondences, got {n}")

    t_src, src_n = normalise_points(src)
    t_dst, dst_n = normalise_points(dst)

    a = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y = src_n[i]
        u, v = dst_n[i]
        a[2 * i] = [-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]
        a[2 * i + 1] = [0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]

    # The null vector of A is the right singular vector with the smallest
    # singular value. Points not in general position - three of them collinear -
    # leave the null space more than one dimensional, and the degeneracy check
    # after denormalisation catches the arbitrary vector that results.
    _, _singular, vt = np.linalg.svd(a)
    h_norm = vt[-1].reshape(3, 3)

    h = np.linalg.inv(t_dst) @ h_norm @ t_src
    if abs(float(np.linalg.det(h))) < 1e-14:
        raise ValueError(
            "degenerate correspondences: the recovered homography is singular "
            "(are three of the points collinear, or two coincident?)"
        )
    return normalise_h(h)


@dataclass(frozen=True, slots=True)
class ReprojectionStats:
    """Per-correspondence reprojection residuals for a fitted homography."""

    residuals_px: FloatArray
    rmse_px: float
    max_px: float
    mean_px: float


def reprojection_stats(
    h: FloatArray, src_points: FloatArray, dst_points: FloatArray
) -> ReprojectionStats:
    """Residuals of ``h`` mapping ``src`` onto ``dst``, in destination units.

    Reported in the calibration view: a sub-pixel RMSE means the four clicked
    points are mutually consistent, and a large one means at least one of them
    is not on the road surface.
    """
    mapped = apply_h(h, src_points)
    dst = np.asarray(dst_points, dtype=np.float64)
    residuals = np.sqrt(((mapped - dst) ** 2).sum(axis=1))
    finite = residuals[np.isfinite(residuals)]
    if finite.size == 0:
        return ReprojectionStats(residuals, float("inf"), float("inf"), float("inf"))
    return ReprojectionStats(
        residuals_px=residuals,
        rmse_px=float(np.sqrt((finite**2).mean())),
        max_px=float(finite.max()),
        mean_px=float(finite.mean()),
    )


# --------------------------------------------------------------------------
# Pinhole camera - the closed-form road-plane homography
# --------------------------------------------------------------------------


def _rot_x(angle_rad: float) -> FloatArray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _rot_z(angle_rad: float) -> FloatArray:
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_world_to_camera(camera: CameraModel) -> FloatArray:
    """Rotation taking world direction vectors into the camera frame.

    The camera frame is the usual computer-vision one: ``x`` right, ``y`` down,
    ``z`` along the optical axis. The base orientation ``R0`` looks along world
    +Y with world +X to the right and world -Z downwards; tilt then rotates the
    optical axis down towards the road, yaw swings it about the vertical, and
    roll spins it about the optical axis.

    A quick sanity check the tests rely on: with zero yaw and roll, the third
    row of ``R`` is ``(0, cos tilt, -sin tilt)``, i.e. the optical axis points
    forwards and downwards, as a pole-mounted camera does.
    """
    r0 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    tilt = np.deg2rad(camera.tilt_deg)
    yaw = np.deg2rad(camera.yaw_deg)
    roll = np.deg2rad(camera.roll_deg)
    return _rot_z(roll) @ _rot_x(tilt) @ r0 @ _rot_z(-yaw)


def camera_matrix(camera: CameraModel) -> FloatArray:
    """Intrinsic matrix ``K`` with square pixels and centred principal point."""
    f = camera.focal_px
    pp = camera.principal_point_px
    return np.array(
        [[f, 0.0, pp.x], [0.0, f, pp.y], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def camera_projection_matrix(camera: CameraModel) -> FloatArray:
    """Full ``3 x 4`` projection matrix ``P = K [R | -R C]``.

    Needed for points off the road plane - the roof corners of a vehicle - which
    is exactly why a single homography is not enough to render a scene but is
    enough to measure one.
    """
    k = camera_matrix(camera)
    r = rotation_world_to_camera(camera)
    c = np.array([camera.x_m, camera.y_m, camera.height_m], dtype=np.float64)
    t = -r @ c
    return k @ np.hstack([r, t.reshape(3, 1)])


def homography_from_camera(camera: CameraModel) -> FloatArray:
    """Exact road-plane (Z = 0) homography of a pinhole camera.

    Dropping the third column of ``P`` - the column that multiplies the world
    ``Z`` coordinate - leaves the ``3 x 3`` map from ``(x_m, y_m, 1)`` to
    homogeneous image coordinates. Because it is derived rather than fitted,
    this matrix is the ground truth against which the four-point DLT
    calibration is judged.
    """
    p = camera_projection_matrix(camera)
    h = p[:, [0, 1, 3]]
    return normalise_h(h)


def project_world_points(camera: CameraModel, points_m: FloatArray) -> FloatArray:
    """Project ``(N, 3)`` world points in metres to ``(N, 2)`` image pixels.

    Points at or behind the camera plane are returned as ``inf`` so that a
    caller clipping geometry cannot mistake a mirrored projection for a real
    one.
    """
    pts = np.asarray(points_m, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"world points must be (N, 3), got {pts.shape}")
    p = camera_projection_matrix(camera)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=np.float64)])
    mapped = homo @ p.T
    depth = mapped[:, 2]
    out = np.full((pts.shape[0], 2), np.inf, dtype=np.float64)
    front = depth > 1e-9
    out[front] = mapped[front, :2] / depth[front, None]
    return out


def horizon_row_px(camera: CameraModel) -> float:
    """Image row of the horizon line.

    Everything below this row can be a point on the road plane; everything above
    it cannot. The perspective-scaled structuring elements (trap 4) and the
    perspective size filter (trap 3) both need it as a guard.
    """
    r = rotation_world_to_camera(camera)
    k = camera_matrix(camera)
    # The horizon is the image of the plane's line at infinity: directions
    # (dx, dy, 0) with the homogeneous weight going to zero.
    h_inf = k @ r[:, :2]
    # Row of the vanishing point of the road direction (0, 1).
    v = h_inf @ np.array([0.0, 1.0])
    if abs(v[2]) < 1e-12:
        return float("-inf")
    return float(v[1] / v[2])


# --------------------------------------------------------------------------
# Comparison and cross-check
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HomographyComparison:
    """How far apart two homographies are, in units a human can judge.

    Comparing matrices element-wise is meaningless because they are defined up
    to scale and their entries have mixed units. Instead both are used to map
    the same sample of road-plane points and the disagreement is reported in
    pixels, and in metres after mapping back.
    """

    max_image_error_px: float
    rmse_image_error_px: float
    max_world_error_m: float
    rmse_world_error_m: float
    relative_error: float
    sample_extent_m: tuple[float, float]

    def within(self, relative_tolerance: float) -> bool:
        return self.relative_error <= relative_tolerance


def compare_homographies(
    h_a: FloatArray,
    h_b: FloatArray,
    x_range_m: tuple[float, float],
    y_range_m: tuple[float, float],
    samples: int = 15,
) -> HomographyComparison:
    """Compare two world-to-image homographies over a patch of road.

    ``relative_error`` is the worst world-space disagreement expressed as a
    fraction of the diagonal of the sampled road patch, which is the number the
    Phase 1 gate and the 2% calibration-accuracy criterion are stated against.
    """
    xs = np.linspace(x_range_m[0], x_range_m[1], samples)
    ys = np.linspace(y_range_m[0], y_range_m[1], samples)
    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float64)

    img_a = apply_h(h_a, grid)
    img_b = apply_h(h_b, grid)
    valid = np.isfinite(img_a).all(axis=1) & np.isfinite(img_b).all(axis=1)
    if not valid.any():
        raise ValueError("no sampled point projects finitely under both homographies")
    img_err = np.sqrt(((img_a[valid] - img_b[valid]) ** 2).sum(axis=1))

    # Map the first homography's image points back through the second: the
    # disagreement in metres is what actually matters for speed.
    world_back = apply_h(invert_h(h_b), img_a[valid])
    world_valid = np.isfinite(world_back).all(axis=1)
    world_err = np.sqrt(((world_back[world_valid] - grid[valid][world_valid]) ** 2).sum(axis=1))
    if world_err.size == 0:
        world_err = np.array([np.inf])

    diag_m = float(
        np.hypot(x_range_m[1] - x_range_m[0], y_range_m[1] - y_range_m[0])
    )
    return HomographyComparison(
        max_image_error_px=float(img_err.max()),
        rmse_image_error_px=float(np.sqrt((img_err**2).mean())),
        max_world_error_m=float(world_err.max()),
        rmse_world_error_m=float(np.sqrt((world_err**2).mean())),
        relative_error=float(world_err.max() / diag_m) if diag_m > 0 else float("inf"),
        sample_extent_m=(x_range_m[1] - x_range_m[0], y_range_m[1] - y_range_m[0]),
    )


def cv2_homography(src_points: FloatArray, dst_points: FloatArray) -> FloatArray:
    """``cv2.findHomography`` wrapper used **only** as a test oracle.

    The production calibration path uses :func:`dlt_homography`. This exists so
    the tests can assert the hand-written DLT agrees with a mature independent
    implementation, which is the honest way to claim it is correct.
    """
    import cv2

    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 1, 2)
    h, _ = cv2.findHomography(src, dst, method=0)
    if h is None:
        raise ValueError("cv2.findHomography failed on these correspondences")
    return normalise_h(np.asarray(h, dtype=np.float64))
