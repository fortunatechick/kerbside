"""The Phase 1 gate: the renderer's ground-truth contract.

Every accuracy number this project reports is measured against this renderer, so
these tests are not about whether the picture looks nice. They check that the
recorded ground truth is *exactly* what it claims to be:

* the recorded image ground point is the projection of the recorded world
  position through the recorded homography;
* the cast shadows really do scale HSV value while leaving hue and saturation
  close to unchanged, which is the premise the whole of S5 rests on;
* occlusion is measured from the frame that was written;
* the same seed gives the same pixels.
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from kerbside.core.types import GroundTruth
from kerbside.data.renderer import SceneConfig, TrafficRenderer, write_video
from kerbside.data.scenarios import build_scene
from kerbside.geometry.homography import apply_h, homography_from_camera
from kerbside.geometry.perspective import PerspectiveModel

GATE_TOLERANCE_PX = 1.0
"""The Phase 1 gate tolerance. The construction is exact, so the measured error
is many orders of magnitude below this; the loose number is what the brief
asks for and the assertion below reports the actual figure when it fails."""


def _fully_visible(vehicle, config, min_area_px: int = 400) -> bool:
    """Only vehicles wholly inside the frame carry a meaningful box bottom.

    A vehicle running off the bottom of the image has a bounding box that is
    clipped by the sensor rather than by the object, so its 'ground point' is a
    property of the crop, not of the geometry. Excluding those keeps the bias
    measurement about perspective rather than about framing.
    """
    x, y, w, h = vehicle.bbox_px
    return (
        vehicle.visible_area_px >= min_area_px
        and vehicle.occlusion_frac < 0.02
        and x >= 0
        and y >= 0
        and x + w <= config.width_px
        and y + h <= config.height_px
    )


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------


def test_ground_point_is_the_projection_of_the_recorded_world_position(
    free_flow_clip,
) -> None:
    """The gate: world position, homography and ground point are consistent.

    The homography is recomputed from the recorded camera parameters rather than
    reused from the renderer, so this checks the whole chain - camera model,
    projection matrix, plane homography and the recorded values - rather than
    comparing a number against itself.
    """
    renderer, _frames, gts = free_flow_clip
    h = homography_from_camera(renderer.camera)

    worst_px = 0.0
    checked = 0
    for gt in gts:
        if not gt.vehicles:
            continue
        world = np.array([v.world_pos_m for v in gt.vehicles], dtype=np.float64)
        recorded = np.array([v.ground_point_px for v in gt.vehicles], dtype=np.float64)
        projected = apply_h(h, world) + np.array(gt.shake_offset_px)
        finite = np.isfinite(projected).all(axis=1) & np.isfinite(recorded).all(axis=1)
        if not finite.any():
            continue
        err = np.sqrt(((projected[finite] - recorded[finite]) ** 2).sum(axis=1))
        worst_px = max(worst_px, float(err.max()))
        checked += int(finite.sum())

    assert checked > 100, f"only {checked} vehicle-frames checked - clip is too sparse"
    assert worst_px < GATE_TOLERANCE_PX, (
        f"recorded ground point disagrees with the projection of the recorded "
        f"world position by up to {worst_px:.4f} px over {checked} vehicle-frames"
    )
    # The construction is exact, so anything above float noise is a real bug.
    assert worst_px < 1e-6


def test_the_gate_holds_under_camera_shake(shaken_clip) -> None:
    """Shake translates the delivered frame, so the ground truth must translate
    with it - otherwise every stabilisation experiment would be scored against
    coordinates that describe a frame nobody was shown."""
    renderer, _frames, gts = shaken_clip
    h = homography_from_camera(renderer.camera)
    shaken = [gt for gt in gts if gt.shake_offset_px != (0.0, 0.0)]
    assert shaken, "the camera_shake scenario produced no shake"

    worst_px = 0.0
    for gt in shaken:
        for v in gt.vehicles:
            projected = apply_h(h, np.array([v.world_pos_m]))[0] + np.array(
                gt.shake_offset_px
            )
            if not np.isfinite(projected).all():
                continue
            worst_px = max(
                worst_px, float(np.hypot(*(projected - np.array(v.ground_point_px))))
            )
    assert worst_px < 1e-6

    amplitudes = [abs(gt.shake_offset_px[0]) for gt in gts]
    assert max(amplitudes) > 1.0, "shake amplitude is too small to be a useful test"


def test_bounding_box_ground_point_is_biased_below_the_true_one(free_flow_clip) -> None:
    """Trap 1 and trap 11, quantified rather than asserted.

    The bottom-centre of the silhouette is not the projection of the footprint
    centre: the nearest footprint corner is closer to the camera than the centre
    is, so the box bottom sits lower in the image. The bias is systematic, grows
    with vehicle size, and is exactly the kind of error that looks like a
    plausible speed rather than like a bug - which is why the renderer records
    it as a field instead of leaving it to be discovered later.
    """
    renderer, _frames, gts = free_flow_clip
    by_class: dict[str, list[float]] = {}
    for gt in gts:
        for v in gt.vehicles:
            if _fully_visible(v, renderer.config):
                by_class.setdefault(v.vehicle_class.value, []).append(
                    v.ground_point_bias_px[1]
                )

    assert by_class, "no vehicles large enough to measure the bias on"
    for name, biases in by_class.items():
        assert float(np.mean(biases)) > 0.0, (
            f"{name}: bounding-box ground point should sit below the true ground "
            f"point, mean bias was {np.mean(biases):.2f} px"
        )

    if "car" in by_class and "truck_bus" in by_class:
        assert float(np.mean(by_class["truck_bus"])) > float(np.mean(by_class["car"]))


def test_ground_point_maps_to_the_vehicles_near_extremity(free_flow_clip) -> None:
    """What the bounding-box ground point actually measures.

    It is not the centre of the footprint. The lowest row of the silhouette is
    the image of the *near* edge of the vehicle, so mapping it back through the
    homography lands half a vehicle length short of the centre - and lands there
    exactly, not approximately, because with no roll or yaw the image row of a
    road point depends only on distance along the road.

    This matters twice over. Absolute position carries a bias of ``length / 2``,
    which shifts where a counting line is deemed to have been crossed. But the
    bias is *constant for a given vehicle*, so it cancels when consecutive
    positions are differenced - which is why S8 can build an unbiased speed on a
    biased position estimate.
    """
    renderer, _frames, gts = free_flow_clip
    model = PerspectiveModel(
        renderer.h_world_to_img, renderer.config.width_px, renderer.config.height_px
    )

    along_residual_m: list[float] = []
    across_residual_m: list[float] = []
    for gt in gts:
        for v in gt.vehicles:
            if not _fully_visible(v, renderer.config):
                continue
            x, y, w, hgt = v.bbox_px
            recovered = model.image_to_world_m(x + w / 2.0, y + hgt)
            if recovered is None:
                continue
            half_length_m = v.footprint_m[0] / 2.0
            along_residual_m.append(recovered[1] + half_length_m - v.world_pos_m[1])
            across_residual_m.append(recovered[0] - v.world_pos_m[0])

    assert len(along_residual_m) > 50, f"only {len(along_residual_m)} usable samples"
    assert float(np.max(np.abs(along_residual_m))) < 0.05, (
        "the ground point should map to the near edge of the footprint to within "
        "a few centimetres"
    )
    assert abs(float(np.median(across_residual_m))) < 0.15


def test_centroid_is_a_much_worse_ground_estimate_than_the_box_bottom(
    free_flow_clip,
) -> None:
    """Trap 1, measured in the units that matter.

    In pixels the two candidates are not dramatically different - a few pixels
    against a few tens. The gap only becomes obvious after mapping back through
    the road-plane homography, because a pixel near the horizon is worth far more
    metres than a pixel in the near field. The centroid sits half a vehicle
    height above the road plane, so it maps to a point further down the road than
    the vehicle actually is; the error grows with distance and with vehicle
    height, which is what makes it look like a plausible reading rather than a
    bug, and is why trucks and cars would read differently.
    """
    renderer, _frames, gts = free_flow_clip
    model = PerspectiveModel(
        renderer.h_world_to_img, renderer.config.width_px, renderer.config.height_px
    )

    box_err_m: list[float] = []
    centroid_err_m: list[float] = []
    heights_m: list[float] = []
    distances_m: list[float] = []
    for gt in gts:
        for v in gt.vehicles:
            if not _fully_visible(v, renderer.config):
                continue
            x, y, w, hgt = v.bbox_px
            from_box = model.image_to_world_m(x + w / 2.0, y + hgt)
            from_centroid = model.image_to_world_m(x + w / 2.0, y + hgt / 2.0)
            if from_box is None or from_centroid is None:
                continue
            # Compare like with like: correct the box estimate for the known
            # half-length offset established above, so this measures estimator
            # quality rather than a definitional difference.
            truth_y = v.world_pos_m[1]
            box_err_m.append(abs(from_box[1] + v.footprint_m[0] / 2.0 - truth_y))
            centroid_err_m.append(abs(from_centroid[1] - truth_y))
            heights_m.append(v.footprint_m[2])
            distances_m.append(truth_y)

    assert len(box_err_m) > 50
    box_median = float(np.median(box_err_m))
    centroid_median = float(np.median(centroid_err_m))
    assert centroid_median > 20.0 * box_median, (
        f"ground point error {box_median:.3f} m vs centroid error "
        f"{centroid_median:.3f} m - the centroid should be far worse"
    )

    # The signature that makes this error dangerous rather than obvious: it
    # scales with how tall the vehicle is and with how far away it is.
    assert float(np.corrcoef(centroid_err_m, heights_m)[0, 1]) > 0.5
    assert float(np.corrcoef(centroid_err_m, distances_m)[0, 1]) > 0.5


# --------------------------------------------------------------------------
# Shadows - the premise behind S5
# --------------------------------------------------------------------------


def test_cast_shadows_scale_value_and_broadly_preserve_hue() -> None:
    """The renderer's shadow model is what S5 will be tested against, so it has
    to be physically defensible rather than convenient.

    Two clips identical except for ``cast_shadows`` isolate the shadow pixels
    exactly. Inside them the HSV value ratio must sit inside the band S5 looks
    for, the saturation change must be small, and the hue must be close to
    unchanged - with the small deviations that the ambient sky tint introduces,
    so that S5's tolerances are genuinely exercised rather than trivially met.
    """
    base = replace(
        build_scene("long_shadows", seed=31, duration_s=0.4),
        supersample=1,
    )
    base = replace(base, atmosphere=replace(base.atmosphere, sensor_noise_sigma=0.0))
    lit = TrafficRenderer(base)
    unlit = TrafficRenderer(
        replace(base, lighting=replace(base.lighting, cast_shadows=False))
    )

    frame_index = lit.frame_count // 2
    with_shadow, gt = lit.render_frame(frame_index)
    without_shadow, _ = unlit.render_frame(frame_index)

    # Shadow pixels are those that changed but are not covered by a vehicle.
    changed = np.abs(
        with_shadow.astype(np.int16) - without_shadow.astype(np.int16)
    ).max(axis=2) > 6
    vehicles = np.zeros(changed.shape, dtype=bool)
    for v in gt.vehicles:
        x, y, w, h = (int(round(c)) for c in v.bbox_px)
        vehicles[max(0, y - 2) : y + h + 2, max(0, x - 2) : x + w + 2] = True
    shadow = changed & ~vehicles
    assert shadow.sum() > 2000, f"only {shadow.sum()} shadow pixels - nothing to test"

    hsv_lit = cv2.cvtColor(with_shadow, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_bg = cv2.cvtColor(without_shadow, cv2.COLOR_BGR2HSV).astype(np.float32)

    v_ratio = hsv_lit[..., 2][shadow] / np.maximum(hsv_bg[..., 2][shadow], 1.0)
    assert 0.35 < float(np.median(v_ratio)) < 0.85, (
        f"shadow value ratio median {np.median(v_ratio):.3f} is outside the band "
        f"S5 assumes - darker than the background but not black"
    )

    s_diff = (hsv_lit[..., 1][shadow] - hsv_bg[..., 1][shadow]) / 255.0
    assert abs(float(np.median(s_diff))) < 0.12

    hue_diff = np.abs(hsv_lit[..., 0][shadow] - hsv_bg[..., 0][shadow])
    hue_diff = np.minimum(hue_diff, 180.0 - hue_diff)  # circular, as in S5
    assert float(np.median(hue_diff)) < 20.0


def test_shadows_are_absent_when_disabled() -> None:
    scene = replace(build_scene("free_flow", seed=5, duration_s=0.2), supersample=1)
    scene = replace(scene, atmosphere=replace(scene.atmosphere, sensor_noise_sigma=0.0))
    lit = TrafficRenderer(scene)
    unlit = TrafficRenderer(
        replace(scene, lighting=replace(scene.lighting, cast_shadows=False))
    )
    a, _ = lit.render_frame(0)
    b, _ = unlit.render_frame(0)
    assert np.abs(a.astype(np.int16) - b.astype(np.int16)).max() > 10


# --------------------------------------------------------------------------
# Occlusion, determinism, layout
# --------------------------------------------------------------------------


def test_occlusion_is_measured_from_the_rendered_frame(occluded_clip) -> None:
    _renderer, _frames, gts = occluded_clip
    occlusions = [v.occlusion_frac for gt in gts for v in gt.vehicles]
    assert occlusions
    assert max(occlusions) > 0.25, (
        "the heavy-occlusion scenario produced no substantially occluded vehicle"
    )
    assert all(0.0 <= o <= 1.0 for o in occlusions)
    # A fully hidden vehicle must be reported as not visible, not as a vehicle
    # with zero area that later metrics would count as a miss.
    for gt in gts:
        for v in gt.vehicles:
            assert v.visible == (v.visible_area_px > 0)


def test_rendering_is_deterministic() -> None:
    scene = replace(build_scene("free_flow", seed=1234, duration_s=0.5), supersample=1)
    a = [f for f, _ in TrafficRenderer(scene).render_clip()]
    b = [f for f, _ in TrafficRenderer(scene).render_clip()]
    assert len(a) == len(b)
    for i, (fa, fb) in enumerate(zip(a, b, strict=True)):
        assert np.array_equal(fa, fb), f"frame {i} differs between identical runs"


def test_a_different_seed_gives_different_traffic() -> None:
    one = replace(build_scene("free_flow", seed=1, duration_s=0.5), supersample=1)
    two = replace(one, seed=2)
    fa, _ = TrafficRenderer(one).render_frame(5)
    fb, _ = TrafficRenderer(two).render_frame(5)
    assert not np.array_equal(fa, fb)


def test_lane_markings_land_at_the_configured_real_world_pitch() -> None:
    """The road plan is built in metres and warped through the known homography,
    so a marking sampled at a multiple of the pitch must be bright and the gap
    between markings dark. This is what makes the clip usable as a calibration
    exercise: the pitch on screen is the pitch that was configured.
    """
    scene = replace(build_scene("free_flow", seed=3, duration_s=0.1), supersample=2)
    scene = replace(scene, atmosphere=replace(scene.atmosphere, sensor_noise_sigma=0.0))
    renderer = TrafficRenderer(scene)
    frame, _ = renderer.render_frame(0)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    road = scene.road
    dash_x_m = road.lane_bands_m()[2][1]  # a dashed lane edge
    pitch = road.marking_pitch_m
    on_marking, in_gap = [], []
    for k in range(2, 7):
        for y_m, bucket in (
            (k * pitch + road.marking_length_m * 0.5, on_marking),
            (k * pitch + road.marking_length_m + (pitch - road.marking_length_m) * 0.5, in_gap),
        ):
            u, v = apply_h(renderer.h_world_to_img, np.array([[dash_x_m, y_m]]))[0]
            if 0 <= u < scene.width_px and 0 <= v < scene.height_px:
                bucket.append(float(grey[int(v), int(u)]))

    assert len(on_marking) >= 3 and len(in_gap) >= 3
    assert float(np.mean(on_marking)) > float(np.mean(in_gap)) + 40.0


def test_lane_polygons_and_counting_lines_are_inside_the_frame() -> None:
    renderer = TrafficRenderer(
        replace(build_scene("free_flow", duration_s=0.1), supersample=1)
    )
    assert len(renderer.lanes) == len(renderer.config.road.lane_centres_m)
    for lane in renderer.lanes:
        assert len(lane.polygon_px) >= 3
        for x, y in lane.counting_line_px:
            assert -50 <= x <= renderer.config.width_px + 50
            assert 0 <= y <= renderer.config.height_px


def test_resolution_variants_project_consistently() -> None:
    """Halving the resolution at a fixed field of view halves every image
    coordinate, which is what makes the Unit 1 resolution study a study of
    sampling rather than of two unrelated scenes."""
    big = replace(build_scene("free_flow", seed=8, duration_s=0.3), supersample=1)
    small = big.with_resolution(640, 360)
    _, gt_big = TrafficRenderer(big).render_frame(3)
    _, gt_small = TrafficRenderer(small).render_frame(3)

    by_id = {v.vehicle_id: v for v in gt_small.vehicles}
    compared = 0
    for v in gt_big.vehicles:
        other = by_id.get(v.vehicle_id)
        if other is None:
            continue
        np.testing.assert_allclose(v.world_pos_m, other.world_pos_m, atol=1e-9)
        np.testing.assert_allclose(
            np.array(v.ground_point_px) / 2.0, other.ground_point_px, atol=1e-6
        )
        compared += 1
    assert compared > 0


# --------------------------------------------------------------------------
# Serialisation and video output
# --------------------------------------------------------------------------


def test_ground_truth_round_trips_through_json(tmp_path, free_flow_clip) -> None:
    renderer, _frames, gts = free_flow_clip
    gt = renderer.ground_truth(gts)
    path = tmp_path / "gt.json"
    gt.save(path)
    loaded = GroundTruth.load(path)

    assert loaded.scenario == gt.scenario
    assert loaded.total_vehicles == gt.total_vehicles
    assert loaded.counts_by_lane() == gt.counts_by_lane()
    np.testing.assert_allclose(loaded.h_world_to_img, gt.h_world_to_img, atol=1e-12)
    assert len(loaded.frames) == len(gt.frames)
    first_a, first_b = gt.frames[0], loaded.frames[0]
    assert len(first_a.vehicles) == len(first_b.vehicles)
    if first_a.vehicles:
        np.testing.assert_allclose(
            first_a.vehicles[0].ground_point_px,
            first_b.vehicles[0].ground_point_px,
            atol=1e-9,
        )


def test_ground_truth_counts_are_per_lane_unique_vehicles(free_flow_clip) -> None:
    renderer, _frames, gts = free_flow_clip
    gt = renderer.ground_truth(gts)
    assert gt.total_vehicles == sum(gt.counts_by_lane().values())
    assert gt.total_vehicles > 0
    assert gt.duration_s == pytest.approx(len(gts) / gt.fps)


def test_write_video_round_trips(tmp_path) -> None:
    scene = replace(build_scene("free_flow", seed=11, duration_s=0.4), supersample=1)
    renderer = TrafficRenderer(scene)
    frames = [f for f, _ in renderer.render_clip()]
    path = tmp_path / "clip.avi"
    written = write_video(
        path, iter(frames), scene.fps, (scene.width_px, scene.height_px), codec="MJPG"
    )
    assert written == len(frames)

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == written
        ok, frame = cap.read()
        assert ok and frame.shape == frames[0].shape
    finally:
        cap.release()


def test_write_video_can_declare_a_frame_rate_it_does_not_have(tmp_path) -> None:
    """Trap 7 needs to be reproducible, so the writer can manufacture a
    container whose metadata disagrees with the true frame interval."""
    scene = replace(build_scene("free_flow", seed=12, duration_s=0.3), supersample=1)
    renderer = TrafficRenderer(scene)
    path = tmp_path / "lying.avi"
    write_video(
        path,
        (f for f, _ in renderer.render_clip()),
        scene.fps,
        (scene.width_px, scene.height_px),
        codec="MJPG",
        declared_fps=30.0,
    )
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0, abs=0.51)
    finally:
        cap.release()
    assert scene.fps != 30.0


def test_write_video_rejects_a_wrong_sized_frame(tmp_path) -> None:
    bad = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="expected"):
        write_video(tmp_path / "bad.avi", iter([bad]), 25.0, (64, 64), codec="MJPG")


def test_renderer_rejects_an_out_of_range_frame() -> None:
    renderer = TrafficRenderer(
        replace(SceneConfig(duration_s=0.2), supersample=1)
    )
    with pytest.raises(IndexError, match="out of range"):
        renderer.render_frame(renderer.frame_count)


# --------------------------------------------------------------------------
# Shutter integration
# --------------------------------------------------------------------------


def _blur_scene(shutter_s: float):
    scene = replace(build_scene("motion_blur", seed=64, duration_s=0.4), supersample=1)
    return replace(
        scene,
        atmosphere=replace(
            scene.atmosphere, shutter_s=shutter_s, sensor_noise_sigma=0.0
        ),
    )


def test_shutter_integration_smears_the_silhouette() -> None:
    """Trap 11 present in the data: a finite exposure stretches the silhouette
    along the direction of travel, which moves the bounding box and therefore
    the ground point that every speed is built on."""
    sharp = TrafficRenderer(_blur_scene(0.0))
    blurred = TrafficRenderer(_blur_scene(0.020))
    index = sharp.frame_count // 2
    a, gt = sharp.render_frame(index)
    b, _ = blurred.render_frame(index)
    assert not np.array_equal(a, b)

    moving = [v for v in gt.vehicles if v.visible_area_px > 1500 and v.speed_kmph > 40]
    assert moving, "no fast vehicle large enough to measure the smear on"
    v = max(moving, key=lambda v: v.visible_area_px)
    x, y, w, h = (int(round(c)) for c in v.bbox_px)

    def edge_softness(img: np.ndarray) -> float:
        """Fraction of the box's bottom band holding intermediate values.

        A sharp silhouette has a step edge - background or body, little in
        between. Integrating the exposure fills that gap in.
        """
        band = cv2.cvtColor(
            img[max(0, y + h - 12) : y + h + 12, max(0, x) : x + w], cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        lo, hi = float(band.min()), float(band.max())
        if hi - lo < 20:
            return 0.0
        mid = (band - lo) / (hi - lo)
        return float(((mid > 0.25) & (mid < 0.75)).mean())

    assert edge_softness(b) > edge_softness(a) * 1.3


def test_shutter_integration_only_touches_the_vehicles() -> None:
    """The integration runs inside each vehicle's swept bounding box for speed.
    That optimisation is only sound if it leaves the rest of the frame exactly
    as it was, so the road surface far from any vehicle must be untouched."""
    sharp = TrafficRenderer(_blur_scene(0.0))
    blurred = TrafficRenderer(_blur_scene(0.020))
    index = sharp.frame_count // 2
    a, gt = sharp.render_frame(index)
    b, _ = blurred.render_frame(index)

    near_vehicles = np.zeros(a.shape[:2], dtype=bool)
    for v in gt.vehicles:
        x, y, w, h = (int(round(c)) for c in v.bbox_px)
        speed_pad = 40
        near_vehicles[
            max(0, y - speed_pad) : y + h + speed_pad,
            max(0, x - speed_pad) : x + w + speed_pad,
        ] = True
    changed = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2) > 2
    assert changed.any()
    assert not (changed & ~near_vehicles).any(), (
        "shutter integration altered pixels far from every vehicle"
    )


def test_shutter_integration_does_not_dominate_the_frame_budget() -> None:
    """A guard on the cost, not on the wall clock.

    Averaging whole frames per vehicle per sub-pose - the obvious
    implementation - costs frame area times sample count times vehicle count,
    which made this scenario tens of times slower than any other. The ratio is
    generously bounded so the test is about complexity rather than about the
    machine it runs on.
    """
    import time

    def seconds_per_frame(shutter_s: float) -> float:
        renderer = TrafficRenderer(_blur_scene(shutter_s))
        indices = range(min(6, renderer.frame_count))
        started = time.perf_counter()
        for i in indices:
            renderer.render_frame(i)
        return (time.perf_counter() - started) / len(list(indices))

    sharp = seconds_per_frame(0.0)
    blurred = seconds_per_frame(0.020)
    assert blurred < 4.0 * sharp + 0.05, (
        f"shutter integration cost {blurred:.3f} s/frame against {sharp:.3f} s/frame "
        f"sharp - the per-vehicle region restriction has regressed"
    )
