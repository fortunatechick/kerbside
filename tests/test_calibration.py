"""Calibration recovered from four clicked points.

The acceptance criterion is that a homography fitted to four points a user
clicks on the road lands within 2% of the known rendering homography. That is
checked three ways: with mathematically exact clicks, which isolates the
estimator from the input; with sub-pixel clicks over the stretch of road a
vehicle is actually tracked across, which is the realistic working case; and as
a characterisation of how the error grows with click noise and with range,
because past about 60 m a single pixel is worth metres and no estimator can
recover what the sampling threw away.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kerbside.core.types import Calibration, Direction
from kerbside.data.renderer import TrafficRenderer
from kerbside.data.scenarios import DEFAULT_CAMERA, build_scene
from kerbside.geometry.calibration import (
    assess_calibration,
    birdseye_homography,
    calibrate_from_correspondences,
    calibrate_from_rectangle,
    calibration_path_for,
    decompose_camera_from_homography,
    lanes_from_world_bands,
    measure_distance_m,
    metre_grid_segments_px,
    order_quad_points,
    reference_distance_suggestions,
)
from kerbside.geometry.homography import (
    apply_h,
    compare_homographies,
    homography_from_camera,
)

CAMERA = DEFAULT_CAMERA
TRUE_H = homography_from_camera(CAMERA)
ROAD_PATCH = ((-7.0, 7.0), (18.0, 80.0))
"""The full stretch of visible road the geometry is compared over."""

TRACKING_PATCH = ((-7.0, 7.0), (18.0, 60.0))
"""The stretch a vehicle is actually tracked and timed across, which is where
the 2% criterion has to hold. Beyond it a single pixel is worth metres, so the
geometry degrades whatever the estimator does - a fact the tests below
characterise rather than paper over."""

RECT_WIDTH_M = 10.5
RECT_LENGTH_M = 25.0
RECT_ORIGIN_M = (-5.25, 22.0)


def _rect_world() -> np.ndarray:
    ox, oy = RECT_ORIGIN_M
    return np.array(
        [
            [ox, oy],
            [ox + RECT_WIDTH_M, oy],
            [ox + RECT_WIDTH_M, oy + RECT_LENGTH_M],
            [ox, oy + RECT_LENGTH_M],
        ],
        dtype=np.float64,
    )


def _clicked_points(noise_px: float = 0.0, seed: int = 0) -> np.ndarray:
    pts = apply_h(TRUE_H, _rect_world())
    if noise_px > 0:
        rng = np.random.default_rng(seed)
        pts = pts + rng.normal(0.0, noise_px, size=pts.shape)
    return pts


# --------------------------------------------------------------------------
# The 2% acceptance criterion
# --------------------------------------------------------------------------


def test_exact_clicks_recover_the_rendering_homography() -> None:
    calib = calibrate_from_rectangle(
        _clicked_points(),
        RECT_WIDTH_M,
        RECT_LENGTH_M,
        CAMERA.width_px,
        CAMERA.height_px,
        origin_m=RECT_ORIGIN_M,
    )
    comparison = compare_homographies(TRUE_H, calib.h_world_to_img, *ROAD_PATCH)
    assert comparison.relative_error < 1e-9
    assert comparison.max_image_error_px < 1e-6


def _relative_errors(noise_px: float, patch, draws: int = 40) -> np.ndarray:
    """Relative homography error over a road patch, across independent draws."""
    base = apply_h(TRUE_H, _rect_world())
    errors = []
    for seed in range(draws):
        clicks = base
        if noise_px > 0:
            rng = np.random.default_rng(seed)
            clicks = np.round(base + rng.normal(0.0, noise_px, base.shape))
        calib = calibrate_from_rectangle(
            clicks, RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px, CAMERA.height_px,
            origin_m=RECT_ORIGIN_M,
        )
        errors.append(
            compare_homographies(TRUE_H, calib.h_world_to_img, *patch).relative_error
        )
    return np.array(errors)


def test_sub_pixel_clicking_stays_within_two_percent_over_the_tracked_road() -> None:
    """The acceptance criterion under realistic clicking.

    Measured over ``TRACKING_PATCH`` - the stretch of road a vehicle is actually
    tracked across, rather than out to the horizon where a pixel is worth
    several metres. Half-pixel click accuracy is achievable because the
    calibration view offers zoom; see the characterisation test below for what
    happens without it.
    """
    errors = _relative_errors(0.5, TRACKING_PATCH)
    assert float(np.median(errors)) < 0.02
    assert float(np.percentile(errors, 95)) < 0.02


def test_calibration_error_grows_with_click_noise_and_with_range() -> None:
    """Characterise the degradation rather than hide it.

    Two facts fall out, and both are reported in the calibration view rather
    than being left for the user to discover:

    * error scales with click precision, so the view must offer zoom and must
      say that a careless click costs accuracy;
    * error grows with how far down the road the geometry is extrapolated,
      because a pixel near the horizon is worth far more metres than a pixel in
      the near field. At whole-pixel clicking the median is still near one
      percent over the tracked road, but the tail exceeds two percent, and by
      100 m it exceeds it outright.
    """
    exact = _relative_errors(0.0, TRACKING_PATCH)
    assert float(exact.max()) < 1e-9

    half_px = _relative_errors(0.5, TRACKING_PATCH)
    one_px = _relative_errors(1.0, TRACKING_PATCH)
    two_px = _relative_errors(2.0, TRACKING_PATCH)
    assert float(np.median(half_px)) < float(np.median(one_px)) < float(np.median(two_px))

    near = _relative_errors(1.0, ((-7.0, 7.0), (18.0, 60.0)))
    far = _relative_errors(1.0, ((-7.0, 7.0), (18.0, 100.0)))
    assert float(np.median(near)) < float(np.median(far))
    # The honest headline: whole-pixel clicking is good in the median and
    # marginal in the tail, which is why the view asks the user to zoom.
    assert float(np.median(one_px)) < 0.02
    assert float(np.percentile(one_px, 95)) > 0.02


def test_a_larger_calibration_rectangle_is_more_robust_to_click_noise() -> None:
    """Why the UI should tell the user to click a big quadrilateral.

    The same click error subtends a smaller angle at the far corners of a large
    rectangle, so the recovered geometry is better conditioned. Demonstrating it
    is what justifies the instruction rather than merely asserting it.
    """

    def worst_error(width_m: float, length_m: float) -> float:
        world = np.array(
            [
                [-width_m / 2, 25.0],
                [width_m / 2, 25.0],
                [width_m / 2, 25.0 + length_m],
                [-width_m / 2, 25.0 + length_m],
            ]
        )
        errors = []
        for seed in range(12):
            rng = np.random.default_rng(seed)
            clicks = np.round(apply_h(TRUE_H, world) + rng.normal(0.0, 1.0, (4, 2)))
            calib = calibrate_from_rectangle(
                clicks,
                width_m,
                length_m,
                CAMERA.width_px,
                CAMERA.height_px,
                origin_m=(-width_m / 2, 25.0),
            )
            errors.append(
                compare_homographies(TRUE_H, calib.h_world_to_img, *ROAD_PATCH).relative_error
            )
        return float(np.mean(errors))

    assert worst_error(10.5, 30.0) < worst_error(3.5, 8.0)


# --------------------------------------------------------------------------
# Point ordering
# --------------------------------------------------------------------------


def test_clicks_in_any_order_give_the_same_calibration() -> None:
    """A transposed pair produces a valid matrix that folds the road over
    itself, so canonical ordering is a correctness requirement, not a
    convenience."""
    canonical = _clicked_points()
    reference = calibrate_from_rectangle(
        canonical, RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px, CAMERA.height_px,
        origin_m=RECT_ORIGIN_M,
    )
    rng = np.random.default_rng(3)
    for _ in range(6):
        shuffled = canonical[rng.permutation(4)]
        calib = calibrate_from_rectangle(
            shuffled, RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px, CAMERA.height_px,
            origin_m=RECT_ORIGIN_M,
        )
        np.testing.assert_allclose(
            calib.h_world_to_img, reference.h_world_to_img, atol=1e-9
        )


def test_order_quad_points_puts_the_near_edge_first() -> None:
    ordered = order_quad_points(_clicked_points()[[2, 0, 3, 1]])
    assert ordered[0][1] > ordered[3][1]  # near-left below far-left
    assert ordered[0][0] < ordered[1][0]  # near-left left of near-right


def test_order_quad_points_rejects_the_wrong_number_of_points() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        order_quad_points(np.zeros((3, 2)))


# --------------------------------------------------------------------------
# Validation tools
# --------------------------------------------------------------------------


def test_distance_tool_recovers_a_known_road_distance() -> None:
    """The only check that catches a wrong scale.

    A homography fitted to a quadrilateral whose real dimensions were entered
    incorrectly is perfectly self-consistent and has a zero residual - it just
    reports every distance in the wrong units. Measuring a known separation is
    what exposes that.
    """
    calib = calibrate_from_rectangle(
        _clicked_points(), RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px,
        CAMERA.height_px, origin_m=RECT_ORIGIN_M,
    )
    a_m, b_m = (-3.0, 30.0), (-3.0, 50.0)
    p1, p2 = apply_h(TRUE_H, np.array([a_m, b_m]))
    measured = measure_distance_m(calib, tuple(p1), tuple(p2))
    assert measured is not None
    assert measured == pytest.approx(20.0, abs=0.01)


def test_distance_tool_exposes_a_mis_entered_rectangle_size() -> None:
    """The failure the tool exists to catch: the geometry looks perfect and
    every measured distance is wrong by the same factor."""
    calib = calibrate_from_rectangle(
        _clicked_points(),
        RECT_WIDTH_M,
        RECT_LENGTH_M * 2.0,  # the user typed the length twice too large
        CAMERA.width_px,
        CAMERA.height_px,
        origin_m=RECT_ORIGIN_M,
    )
    assert calib.reprojection_rmse_px < 1e-6  # residual says nothing is wrong
    p1, p2 = apply_h(TRUE_H, np.array([(-3.0, 30.0), (-3.0, 50.0)]))
    measured = measure_distance_m(calib, tuple(p1), tuple(p2))
    assert measured is not None
    assert measured > 30.0  # but the tape measure would say 20 m


def test_distance_tool_refuses_points_above_the_horizon() -> None:
    calib = calibrate_from_rectangle(
        _clicked_points(), RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px,
        CAMERA.height_px, origin_m=RECT_ORIGIN_M,
    )
    assert measure_distance_m(calib, (640.0, 10.0), (640.0, 12.0)) is None


def test_camera_decomposition_recovers_the_mounting() -> None:
    """The physical sanity check behind the calibration verdict: the recovered
    pole height and tilt are things the user can compare against the pole."""
    report = decompose_camera_from_homography(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    assert report.is_plausible, report.reasons
    assert report.focal_px == pytest.approx(CAMERA.focal_px, rel=1e-6)
    assert report.height_m == pytest.approx(CAMERA.height_m, rel=1e-6)
    assert report.tilt_deg == pytest.approx(CAMERA.tilt_deg, rel=1e-6)


def test_an_implausible_camera_is_reported_as_such() -> None:
    """A badly clicked quadrilateral yields a valid matrix that implies an
    impossible camera; saying so is more useful than a zero residual."""
    world = _rect_world()
    clicks = apply_h(TRUE_H, world)
    clicks[2] = clicks[2] + np.array([0.0, 140.0])  # one corner dragged off the road
    calib = calibrate_from_rectangle(
        clicks, RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px, CAMERA.height_px,
        origin_m=RECT_ORIGIN_M, reorder=False,
    )
    report = assess_calibration(calib)
    assert not report.camera.is_plausible
    assert report.warnings


def test_a_four_point_fit_warns_that_its_residual_proves_nothing() -> None:
    calib = calibrate_from_rectangle(
        _clicked_points(), RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px,
        CAMERA.height_px, origin_m=RECT_ORIGIN_M,
    )
    report = assess_calibration(calib)
    assert report.exactly_determined
    assert any("not evidence of accuracy" in w for w in report.warnings)
    assert calib.quality_verdict.startswith("excellent")


def test_overdetermined_fit_reports_a_real_residual() -> None:
    """With more than four correspondences the residual becomes informative,
    because the system is no longer exactly determined."""
    rng = np.random.default_rng(11)
    world = np.array([[x, y] for x in (-6.0, -2.0, 2.0, 6.0) for y in (20.0, 40.0, 65.0)])
    img = apply_h(TRUE_H, world) + rng.normal(0.0, 0.8, (world.shape[0], 2))
    calib = calibrate_from_correspondences(
        img, world, CAMERA.width_px, CAMERA.height_px
    )
    assert 0.1 < calib.reprojection_rmse_px < 3.0
    assert calib.reprojection_max_px >= calib.reprojection_rmse_px


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------


def test_metre_grid_projects_onto_the_road() -> None:
    calib = Calibration(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    lines = metre_grid_segments_px(calib, (-7.0, 7.0), (20.0, 45.0), step_m=1.0)
    assert len(lines) >= 15 + 26 - 4
    for polyline in lines:
        assert len(polyline) >= 2
        for x, y in polyline:
            assert np.isfinite(x) and np.isfinite(y)


def test_metre_grid_spacing_shrinks_with_distance() -> None:
    """If the grid does not visibly compress towards the horizon the
    calibration is wrong, which is exactly what the side-by-side view in the
    calibration screen is asking the user to look for."""
    calib = Calibration(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    rows = [apply_h(TRUE_H, np.array([[0.0, y]]))[0][1] for y in (20.0, 21.0, 60.0, 61.0)]
    near_gap = abs(rows[1] - rows[0])
    far_gap = abs(rows[3] - rows[2])
    assert near_gap > 4.0 * far_gap
    assert metre_grid_segments_px(calib, (-3.0, 3.0), (20.0, 25.0), step_m=1.0)


def test_metre_grid_rejects_a_non_positive_step() -> None:
    calib = Calibration(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    with pytest.raises(ValueError, match="step must be positive"):
        metre_grid_segments_px(calib, (-3.0, 3.0), (20.0, 25.0), step_m=0.0)


def test_birdseye_rectifies_the_road_plane() -> None:
    """A rectangle on the road must come out a rectangle in the plan view, with
    traffic moving away from the camera moving up the image."""
    calib = Calibration(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    bev, out_w, out_h = birdseye_homography(calib, (-7.0, 7.0), (18.0, 80.0), px_per_m=8.0)
    assert (out_w, out_h) == (112, 496)

    corners_world = np.array([[-7.0, 18.0], [7.0, 18.0], [7.0, 80.0], [-7.0, 80.0]])
    in_image = apply_h(TRUE_H, corners_world)
    in_bev = apply_h(bev, in_image)
    np.testing.assert_allclose(
        in_bev,
        np.array([[0.0, out_h], [out_w, out_h], [out_w, 0.0], [0.0, 0.0]]),
        atol=1e-6,
    )


def test_birdseye_rejects_an_inverted_range() -> None:
    calib = Calibration(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    with pytest.raises(ValueError, match="increasing"):
        birdseye_homography(calib, (7.0, -7.0), (18.0, 80.0))


# --------------------------------------------------------------------------
# Lanes and persistence
# --------------------------------------------------------------------------


def test_lane_polygons_contain_their_own_lane_centre() -> None:
    import cv2

    calib = Calibration(TRUE_H, CAMERA.width_px, CAMERA.height_px)
    bands = [(-7.0, -3.5), (-3.5, 0.0), (0.0, 3.5), (3.5, 7.0)]
    lanes = lanes_from_world_bands(
        calib, bands, (20.0, 70.0), directions=[Direction.TOWARD] * 2 + [Direction.AWAY] * 2
    )
    assert len(lanes) == 4
    for lane, (x0, x1) in zip(lanes, bands, strict=True):
        centre = apply_h(TRUE_H, np.array([[(x0 + x1) / 2.0, 45.0]]))[0]
        poly = np.array(lane.polygon_px, dtype=np.float32)
        assert cv2.pointPolygonTest(poly, (float(centre[0]), float(centre[1])), False) > 0
        # And it must not contain the neighbouring lane's centre.
        other = apply_h(TRUE_H, np.array([[x1 + 1.75, 45.0]]))[0]
        if 0 <= other[0] < CAMERA.width_px:
            assert cv2.pointPolygonTest(poly, (float(other[0]), float(other[1])), False) < 0


def test_calibration_round_trips_through_json(tmp_path) -> None:
    calib = calibrate_from_rectangle(
        _clicked_points(), RECT_WIDTH_M, RECT_LENGTH_M, CAMERA.width_px,
        CAMERA.height_px, origin_m=RECT_ORIGIN_M, source="test",
    )
    calib = calib.with_lanes(
        lanes_from_world_bands(calib, [(-3.5, 0.0), (0.0, 3.5)], (20.0, 70.0))
    )
    path = tmp_path / "calib.json"
    calib.save(path)
    loaded = Calibration.load(path)

    np.testing.assert_allclose(loaded.h_world_to_img, calib.h_world_to_img, atol=1e-12)
    assert loaded.source == "test"
    assert len(loaded.lanes) == 2
    assert loaded.lanes[0].polygon_px == calib.lanes[0].polygon_px
    assert loaded.rect_width_m == RECT_WIDTH_M


def test_calibration_key_is_sanitised_into_a_filename() -> None:
    path = calibration_path_for("clip/free flow:01.mp4", root="data/calib")
    assert path.name == "clip_free_flow_01.mp4.json"
    with pytest.raises(ValueError, match="no usable characters"):
        calibration_path_for("///")


def test_reference_distances_are_offered_with_a_caution() -> None:
    """Assist, do not assume: every suggested dimension must carry the reason
    the user has to check it against their own road."""
    suggestions = reference_distance_suggestions()
    assert suggestions
    for s in suggestions:
        assert s.caution
        lo, hi = s.plausible_range_m
        assert lo <= s.nominal_m <= hi


# --------------------------------------------------------------------------
# Against a rendered clip, which is what the UI workflow actually does
# --------------------------------------------------------------------------


def test_calibrating_a_rendered_clip_matches_the_rendering_geometry() -> None:
    """End to end: render a clip, click four points on its road surface, and
    check the recovered geometry against the one the clip was rendered with."""
    scene = replace(build_scene("free_flow", seed=21, duration_s=0.2), supersample=1)
    renderer = TrafficRenderer(scene)

    # Corners of a patch bounded by the outer lane edges and two marking pitches.
    road = scene.road
    x0, x1 = -road.half_width_m, road.half_width_m
    y0, y1 = 24.0, 24.0 + road.marking_pitch_m * 4
    world = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    clicks = np.round(apply_h(renderer.h_world_to_img, world))

    calib = calibrate_from_rectangle(
        clicks, x1 - x0, y1 - y0, scene.width_px, scene.height_px,
        origin_m=(x0, y0), source=f"synthetic:{scene.name}",
    )
    comparison = compare_homographies(
        renderer.h_world_to_img, calib.h_world_to_img, *ROAD_PATCH
    )
    assert comparison.relative_error < 0.02

    report = assess_calibration(calib)
    assert report.camera.is_plausible, report.camera.reasons
    assert report.camera.height_m == pytest.approx(scene.camera.height_m, rel=0.05)
    assert report.camera.tilt_deg == pytest.approx(scene.camera.tilt_deg, rel=0.05)
