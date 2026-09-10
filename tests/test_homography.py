"""Tests for the from-scratch homography and the perspective model."""

from __future__ import annotations

import numpy as np
import pytest

from kerbside.core.types import CameraModel
from kerbside.geometry.homography import (
    apply_h,
    apply_h_single,
    camera_projection_matrix,
    compare_homographies,
    cv2_homography,
    dlt_homography,
    homography_from_camera,
    horizon_row_px,
    invert_h,
    normalise_h,
    normalise_points,
    project_world_points,
    reprojection_stats,
    rotation_world_to_camera,
)
from kerbside.geometry.perspective import (
    PerspectiveModel,
    estimate_focal_px_from_homography,
    footprint_area_px,
    homography_jacobian,
    pixels_per_metre,
    vertical_model_from_homography,
)

CAMERA = CameraModel(width_px=1280, height_px=720, height_m=8.0, tilt_deg=12.0, fov_deg=55.0)


@pytest.fixture()
def h() -> np.ndarray:
    return homography_from_camera(CAMERA)


# --------------------------------------------------------------------------
# Known-answer tests
# --------------------------------------------------------------------------


def test_identity_homography_is_a_no_op() -> None:
    pts = np.array([[0.0, 0.0], [3.5, -2.0], [100.0, 40.0]])
    np.testing.assert_allclose(apply_h(np.eye(3), pts), pts, atol=1e-12)


def test_pure_translation_is_recovered_exactly() -> None:
    """A hand-worked case: four points shifted by a constant vector."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    dst = src + np.array([7.0, -3.0])
    h = dlt_homography(src, dst)
    expected = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, -3.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(h, expected, atol=1e-10)


def test_known_projective_map_is_recovered() -> None:
    """The unit square onto an arbitrary quadrilateral, then checked by mapping
    a fifth point that took no part in the fit."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    truth = normalise_h(
        np.array([[2.0, 0.3, 5.0], [-0.4, 1.7, 9.0], [0.002, 0.004, 1.0]])
    )
    dst = apply_h(truth, src)
    h = dlt_homography(src, dst)
    np.testing.assert_allclose(h, truth, atol=1e-10)

    held_out = np.array([[0.37, 0.62]])
    np.testing.assert_allclose(apply_h(h, held_out), apply_h(truth, held_out), atol=1e-9)


def test_dlt_matches_opencv_on_a_road_quadrilateral(h: np.ndarray) -> None:
    """The hand-written DLT is checked against an independent implementation.

    Agreement to well under a pixel over the whole road patch is the honest
    basis for claiming the from-scratch version is correct.
    """
    world = np.array([[-5.0, 20.0], [5.0, 20.0], [5.0, 45.0], [-5.0, 45.0]])
    img = apply_h(h, world)
    ours = dlt_homography(world, img)
    theirs = cv2_homography(world, img)
    comparison = compare_homographies(ours, theirs, (-7.0, 7.0), (16.0, 90.0))
    assert comparison.max_image_error_px < 1e-3


def test_normalise_points_centres_and_scales() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(500.0, 200.0, size=(12, 2))
    t, normalised = normalise_points(pts)
    np.testing.assert_allclose(normalised.mean(axis=0), [0.0, 0.0], atol=1e-9)
    mean_dist = np.sqrt((normalised**2).sum(axis=1)).mean()
    assert mean_dist == pytest.approx(np.sqrt(2.0), abs=1e-9)

    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    np.testing.assert_allclose((homo @ t.T)[:, :2], normalised, atol=1e-9)


def test_inverse_round_trips(h: np.ndarray) -> None:
    world = np.array([[-3.0, 25.0], [4.0, 60.0], [0.0, 100.0]])
    back = apply_h(invert_h(h), apply_h(h, world))
    np.testing.assert_allclose(back, world, atol=1e-8)


# --------------------------------------------------------------------------
# Error handling - a wrong answer must be an exception, never a silent number
# --------------------------------------------------------------------------


def test_too_few_points_is_an_error() -> None:
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="at least 4"):
        dlt_homography(pts, pts)


def test_collinear_points_are_rejected() -> None:
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="degenerate|singular"):
        dlt_homography(src, dst)


def test_mismatched_point_sets_are_rejected() -> None:
    with pytest.raises(ValueError, match="match in shape"):
        dlt_homography(np.zeros((4, 2)), np.zeros((5, 2)))


def test_singular_homography_cannot_be_inverted() -> None:
    with pytest.raises(ValueError, match="singular"):
        invert_h(np.ones((3, 3)))


def test_points_on_the_horizon_map_to_infinity_not_to_a_huge_number(h: np.ndarray) -> None:
    """A point a few pixels above the horizon is kilometres away on the road
    plane; returning inf rather than a finite number is what stops a track
    there from producing a plausible-looking four-digit speed."""
    horizon = horizon_row_px(CAMERA)
    h_inv = invert_h(h)
    above = apply_h(h_inv, np.array([[640.0, horizon - 3.0]]))
    assert above[0, 1] < 0 or not np.isfinite(above[0, 1])


# --------------------------------------------------------------------------
# Camera geometry
# --------------------------------------------------------------------------


def test_rotation_is_orthonormal_and_points_down_the_road() -> None:
    r = rotation_world_to_camera(CAMERA)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-12)
    assert float(np.linalg.det(r)) == pytest.approx(1.0, abs=1e-12)
    tilt = np.deg2rad(CAMERA.tilt_deg)
    np.testing.assert_allclose(r[2], [0.0, np.cos(tilt), -np.sin(tilt)], atol=1e-12)


def test_horizon_sits_above_image_centre_by_focal_times_tan_tilt() -> None:
    expected = CAMERA.height_px / 2.0 - CAMERA.focal_px * np.tan(np.deg2rad(CAMERA.tilt_deg))
    assert horizon_row_px(CAMERA) == pytest.approx(expected, abs=1e-6)


def test_plane_homography_agrees_with_the_full_projection_matrix(h: np.ndarray) -> None:
    world = np.array([[-4.0, 22.0, 0.0], [3.0, 55.0, 0.0], [0.0, 90.0, 0.0]])
    np.testing.assert_allclose(
        project_world_points(CAMERA, world), apply_h(h, world[:, :2]), atol=1e-9
    )


def test_points_behind_the_camera_are_not_silently_mirrored() -> None:
    behind = np.array([[0.0, -40.0, 0.0]])
    assert not np.isfinite(project_world_points(CAMERA, behind)).all()


# --------------------------------------------------------------------------
# Perspective model
# --------------------------------------------------------------------------


def test_focal_length_is_recovered_from_the_plane_homography(h: np.ndarray) -> None:
    recovered = estimate_focal_px_from_homography(h, CAMERA.width_px, CAMERA.height_px)
    assert recovered is not None
    assert recovered == pytest.approx(CAMERA.focal_px, rel=1e-9)


def test_out_of_plane_points_match_the_full_projection(h: np.ndarray) -> None:
    """The height model reproduces the true projection matrix using only ``H``.

    This is what lets the pipeline reason about a vehicle's silhouette - which
    includes its roof - from a calibration that only ever saw the road surface.
    """
    model = PerspectiveModel(h, CAMERA.width_px, CAMERA.height_px)
    for x_m, y_m, z_m in [(0.0, 25.0, 1.5), (-4.0, 60.0, 3.2), (3.5, 18.0, 0.9)]:
        exact = project_world_points(CAMERA, np.array([[x_m, y_m, z_m]]))[0]
        estimated = model.image_point_at_height_px(x_m, y_m, z_m)
        assert estimated is not None
        np.testing.assert_allclose(estimated, exact, atol=1e-6)


def test_vertical_vanishing_point_is_below_the_image_for_a_downward_camera(
    h: np.ndarray,
) -> None:
    model = vertical_model_from_homography(h, CAMERA.width_px, CAMERA.height_px)
    assert model is not None
    vp = model.vanishing_point_px
    assert vp is not None
    assert vp[1] > CAMERA.height_px
    assert vp[0] == pytest.approx(CAMERA.width_px / 2.0, abs=1e-6)


def test_jacobian_matches_a_finite_difference(h: np.ndarray) -> None:
    x_m, y_m, eps = 1.5, 40.0, 1e-4
    j = homography_jacobian(h, x_m, y_m)
    dx = (
        np.array(apply_h_single(h, x_m + eps, y_m))
        - np.array(apply_h_single(h, x_m - eps, y_m))
    ) / (2 * eps)
    dy = (
        np.array(apply_h_single(h, x_m, y_m + eps))
        - np.array(apply_h_single(h, x_m, y_m - eps))
    ) / (2 * eps)
    np.testing.assert_allclose(j[:, 0], dx, rtol=1e-5)
    np.testing.assert_allclose(j[:, 1], dy, rtol=1e-5)


def test_expected_area_falls_monotonically_with_distance(h: np.ndarray) -> None:
    """Trap 3 in one assertion: the same car covers vastly different pixel
    counts depending on where it is, so one global area threshold cannot work."""
    model = PerspectiveModel(h, CAMERA.width_px, CAMERA.height_px)
    rows = [700, 600, 500, 400, 300, 200]
    areas = [model.expected_area_px(r) for r in rows]
    assert all(a > b > 0 for a, b in zip(areas[:-1], areas[1:], strict=True))
    assert areas[0] / areas[-1] > 50.0


def test_structuring_elements_shrink_with_distance_and_stay_anisotropic(
    h: np.ndarray,
) -> None:
    """Trap 4: the element must scale with perspective and stay wider than tall
    on an oblique view, or it merges lanes near the camera and hollows out
    vehicles far from it."""
    model = PerspectiveModel(h, CAMERA.width_px, CAMERA.height_px)
    near_w, near_h = model.se_size_px(700.0, 0.5)
    far_w, far_h = model.se_size_px(300.0, 0.5)
    assert near_w > far_w
    assert near_h >= far_h
    assert near_w > near_h
    assert all(v % 2 == 1 for v in (near_w, near_h, far_w, far_h))


def test_image_to_world_refuses_points_above_the_horizon(h: np.ndarray) -> None:
    model = PerspectiveModel(h, CAMERA.width_px, CAMERA.height_px)
    assert model.image_to_world_m(640.0, model.horizon_row_px - 5.0) is None
    assert model.image_to_world_m(640.0, 700.0) is not None


def test_footprint_area_matches_local_scale_in_the_small(h: np.ndarray) -> None:
    """Far from the horizon the projection is locally affine, so the projected
    footprint area should agree with the product of the two local scales."""
    x_m, y_m = 0.0, 30.0
    across, along = pixels_per_metre(h, x_m, y_m)
    exact = footprint_area_px(h, x_m, y_m)
    assert exact == pytest.approx(4.0 * 1.8 * across * along, rel=0.05)


def test_projection_matrix_scales_with_resolution() -> None:
    """Doubling the sampling at a fixed field of view scales the projection by
    exactly two - which is what lets the renderer supersample and still report
    ground truth at the output resolution."""
    big = CameraModel(
        width_px=CAMERA.width_px * 2,
        height_px=CAMERA.height_px * 2,
        height_m=CAMERA.height_m,
        tilt_deg=CAMERA.tilt_deg,
        fov_deg=CAMERA.fov_deg,
    )
    scale = np.diag([2.0, 2.0, 1.0])
    np.testing.assert_allclose(
        camera_projection_matrix(big), scale @ camera_projection_matrix(CAMERA), atol=1e-9
    )


def test_reprojection_stats_are_zero_for_an_exact_fit(h: np.ndarray) -> None:
    world = np.array([[-5.0, 20.0], [5.0, 20.0], [5.0, 45.0], [-5.0, 45.0]])
    stats = reprojection_stats(h, world, apply_h(h, world))
    assert stats.rmse_px < 1e-9
    assert stats.max_px < 1e-9
