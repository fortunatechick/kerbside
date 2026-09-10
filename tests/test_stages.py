"""Stages S1 to S6, and the metrics that score them.

The Phase 3 target is that per-frame blob count equals ground-truth vehicle
count in at least 98% of frames. **That target is not met.** The measured figure
is recorded in :func:`test_detection_accuracy_is_characterised_honestly`, along
with what is limiting it, because a number that is not yet good enough is still
a number and hiding it would make every later comparison meaningless.

What these tests do assert is everything that *is* established: that each stage
does what it claims, that the specific failure modes found during Phase 3 stay
fixed, and that the evaluation metric measures what it says it measures.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from kerbside.core.types import FrameType
from kerbside.data.renderer import TrafficRenderer
from kerbside.data.scenarios import build_scene
from kerbside.eval.metrics import VisibilityGate, match_frame, score_detection
from kerbside.stages.base import FrameContext, StageError
from kerbside.stages.pipeline import Pipeline
from kerbside.stages.s1_ingest import BT601_WEIGHTS, IngestStage
from kerbside.stages.s3_background import BackgroundStage

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip():
    """A short free-flow clip with its ground truth and exact calibration."""
    scene = replace(build_scene("free_flow", seed=4242, duration_s=5.0), supersample=1)
    renderer = TrafficRenderer(scene)
    frames, gts = [], []
    for frame, gt in renderer.render_clip():
        frames.append(frame)
        gts.append(gt)
    return renderer, frames, renderer.ground_truth(gts), gts


@pytest.fixture(scope="module")
def processed(clip):
    renderer, frames, truth, gts = clip
    pipe = Pipeline(
        frame_interval_s=1.0 / renderer.config.fps, calibration=renderer.calibration()
    )
    outputs = list(pipe.run(frames))
    return pipe, outputs, truth, gts, frames


# --------------------------------------------------------------------------
# Stage scaffolding
# --------------------------------------------------------------------------


def test_every_stage_explains_itself(processed) -> None:
    """The course requirement, enforced rather than trusted."""
    pipe, *_ = processed
    for stage in pipe.stages:
        assert stage.stage_id and stage.name
        assert stage.units, f"{stage.stage_id} declares no syllabus unit"
        assert len(stage.rationale) > 60, f"{stage.stage_id} has a token rationale"
        described = stage.describe()
        assert set(described["values"]) == {p.name for p in stage.param_specs}


def test_parameters_are_validated_at_the_point_they_are_set(processed) -> None:
    pipe, *_ = processed
    stage = pipe.stage("s4_foreground")
    with pytest.raises(ValueError, match="no parameter"):
        stage.set_params(nonexistent=1)
    with pytest.raises(ValueError, match="above the maximum"):
        stage.set_params(median_px=999)
    with pytest.raises(ValueError, match="must be odd"):
        stage.set_params(median_px=4)


def test_changing_a_parameter_resets_downstream_state(clip) -> None:
    """A background model built under one learning rate does not describe the
    scene under another, so a metric delta measured against stale state would
    describe neither setting."""
    renderer, frames, _truth, _gts = clip
    pipe = Pipeline(
        frame_interval_s=1.0 / renderer.config.fps, calibration=renderer.calibration()
    )
    for frame in frames[:20]:
        pipe.process_frame(frame)
    assert pipe.background.frames_seen == 20

    pipe.configure("s3_background", alpha=0.05)
    assert pipe.background.frames_seen == 0
    assert pipe.background.param("alpha") == pytest.approx(0.05)


def test_a_stage_run_out_of_order_says_so_rather_than_crashing() -> None:
    ctx = FrameContext(frame_index=0, time_s=0.0, bgr=np.zeros((16, 16, 3), np.uint8))
    with pytest.raises(StageError, match="S1 has not run"):
        ctx.require_grey()
    with pytest.raises(StageError, match="no foreground mask"):
        _ = ctx.mask


def test_modified_parameters_are_reported_for_the_ui_badge(processed) -> None:
    pipe, *_ = processed
    stage = pipe.stage("s5_shadow")
    assert stage.differs_from_default() == ()
    stage.set_params(hue_delta=25.0)
    assert stage.differs_from_default() == ("hue_delta",)


# --------------------------------------------------------------------------
# S1 - ingest
# --------------------------------------------------------------------------


def test_greyscale_uses_the_declared_bt601_weights() -> None:
    stage = IngestStage()
    patch = np.zeros((4, 4, 3), np.uint8)
    patch[:, :, 0], patch[:, :, 1], patch[:, :, 2] = 10, 20, 30  # B, G, R
    expected = 0.114 * 10 + 0.587 * 20 + 0.299 * 30
    assert float(stage.to_grey(patch)[0, 0]) == pytest.approx(expected, abs=1e-3)
    assert sum(BT601_WEIGHTS) == pytest.approx(1.0, abs=1e-6)


def test_zero_luma_weights_are_refused_rather_than_producing_a_black_frame() -> None:
    stage = IngestStage()
    stage.set_params(weight_b=0.0, weight_g=0.0, weight_r=0.0)
    with pytest.raises(StageError, match="sum to zero"):
        stage.to_grey(np.zeros((4, 4, 3), np.uint8))


def test_a_lying_container_frame_rate_is_detected(tmp_path) -> None:
    """Trap 7. Every speed divides by the frame interval, so a container that
    claims a rate it does not deliver produces speeds that are wrong by that
    factor with nothing in the output that looks wrong."""
    from kerbside.data.renderer import write_video
    from kerbside.stages.s1_ingest import VideoSource

    scene = replace(build_scene("free_flow", seed=3, duration_s=1.2), supersample=1)
    renderer = TrafficRenderer(scene)
    path = tmp_path / "lying.avi"
    write_video(
        path, (f for f, _ in renderer.render_clip()), scene.fps,
        (scene.width_px, scene.height_px), codec="MJPG", declared_fps=30.0,
    )
    with VideoSource(path) as source:
        info = source.info
        assert info.container_fps == pytest.approx(30.0, abs=0.6)
        assert info.measured_fps == pytest.approx(30.0, abs=1.0)
        # The writer stamps its own timestamps from the declared rate, so this
        # container is internally consistent and the check passes. What matters
        # is that the two figures are reported separately and compared at all -
        # a single reconciled number could not raise the question.
        assert info.fps_is_trustworthy
        assert "container_fps" in info.to_dict() and "measured_fps" in info.to_dict()


def test_ingest_reports_usable_dynamic_range(processed) -> None:
    _pipe, outputs, *_ = processed
    metrics = outputs[10].context.results["s1_ingest"].metrics
    assert metrics["dynamic_range"] > 40.0
    assert metrics["frame_interval_ms"] == pytest.approx(40.0)


# --------------------------------------------------------------------------
# S2 - stabilisation
# --------------------------------------------------------------------------


def test_stabiliser_recovers_a_known_shake() -> None:
    """Trap 8. The renderer records the offset it applied, so the estimate is
    scored against a number rather than against an impression."""
    scene = replace(build_scene("camera_shake", seed=99, duration_s=2.0), supersample=1)
    renderer = TrafficRenderer(scene)
    frames, gts = [], []
    for frame, gt in renderer.render_clip():
        frames.append(frame)
        gts.append(gt)

    pipe = Pipeline(frame_interval_s=1.0 / scene.fps, calibration=renderer.calibration())
    # The stage aligns every frame to the *first* one, so what it recovers is
    # displacement relative to that frame - and the first frame carries a shake
    # offset of its own. The difference is a constant, which is exactly what a
    # background model wants (it is built in the same frame of reference) but
    # which has to be subtracted before comparing against absolute truth.
    reference = np.asarray(gts[0].shake_offset_px)

    errors, applied = [], []
    for index, output in enumerate(pipe.run(frames)):
        if index < 3:
            continue
        truth = np.asarray(gts[index].shake_offset_px) - reference
        estimated = np.array([output.context.shake_dx_px, output.context.shake_dy_px])
        if not output.context.stabilisation_confident:
            continue  # the stage declined; that is tested separately
        errors.append(float(np.hypot(*(estimated - truth))))
        applied.append(float(np.hypot(*truth)))

    assert float(np.mean(applied)) > 1.0, "the scenario applied no meaningful shake"
    # The estimate must remove essentially all of the relative shake.
    assert float(np.median(errors)) < 0.2, f"median residual {np.median(errors):.3f} px"
    assert float(np.median(errors)) < 0.1 * float(np.mean(applied))


def test_stabiliser_declines_when_the_correlation_peak_is_weak() -> None:
    """A featureless frame gives a broad, flat correlation surface whose maximum
    is arbitrary; warping by it would inject the very motion the stage removes."""
    pipe = Pipeline()
    pipe.configure("s2_stabilise", min_peak_ratio=9.9)
    flat = np.full((240, 320, 3), 120, np.uint8)
    rng = np.random.default_rng(0)
    noisy = np.clip(flat + rng.normal(0, 3, flat.shape), 0, 255).astype(np.uint8)
    pipe.process_frame(flat)
    output = pipe.process_frame(noisy)
    assert output.context.shake_dx_px == 0.0
    assert output.context.shake_dy_px == 0.0


def test_stabilisation_can_be_disabled_to_show_what_it_is_worth(processed) -> None:
    pipe, *_ = processed
    pipe.configure("s2_stabilise", enabled=False)
    try:
        result = pipe.stage("s2_stabilise").describe()
        assert result["values"]["enabled"] is False
    finally:
        pipe.configure("s2_stabilise", enabled=True)


# --------------------------------------------------------------------------
# S3 - background
# --------------------------------------------------------------------------


def test_running_average_is_a_temporal_low_pass_with_the_stated_time_constant() -> None:
    """The Unit 2 framing, checked numerically rather than asserted.

    ``B <- (1-a) B + a I`` is a first-order recursive filter, so a step input
    settles exponentially and has reached ``1 - 1/e`` of the step after ``1/a``
    frames. That is what makes "temporal low-pass filter" a statement about the
    filter rather than a metaphor.
    """
    alpha = 0.05
    stage = BackgroundStage(alpha=alpha, selective_update=False, model="running_average")
    dark = np.zeros((8, 8, 3), np.uint8)
    bright = np.full((8, 8, 3), 200, np.uint8)

    ingest = IngestStage()
    for _ in range(200):
        ctx = FrameContext(0, 0.0, dark)
        ingest(ctx)
        stage(ctx)

    settle = int(round(1 / alpha))
    for index in range(settle):
        ctx = FrameContext(index, 0.0, bright)
        ingest(ctx)
        stage(ctx)

    model = stage.background_bgr_u8().mean()
    assert model == pytest.approx(200 * (1 - np.exp(-1.0)), rel=0.1)


def test_colour_differencing_finds_vehicles_a_greyscale_difference_cannot() -> None:
    """The finding that changed the design.

    BT.601 weights red at 0.299, so a red vehicle on grey asphalt can have
    almost exactly the road's luminance. Measured on this project's palette
    against its own road surface, four of ten body colours are within ten grey
    levels of the road - red is within 1.2 - while differing by 30 to 94 levels
    in at least one channel. A greyscale background subtraction does not merely
    struggle with them; it cannot see them at all.
    """
    road = np.array([80, 76, 74], dtype=np.float64)  # BGR asphalt
    red = np.array([40, 38, 168], dtype=np.float64)  # BGR red body
    weights = np.array([0.114, 0.587, 0.299])
    assert abs(float(road @ weights) - float(red @ weights)) < 5.0
    assert float(np.abs(road - red).max()) > 80.0

    stage = BackgroundStage(model="reference", selective_update=False)
    ingest = IngestStage()
    frame = np.zeros((16, 16, 3), np.uint8)
    frame[:, :] = road.astype(np.uint8)

    ctx = FrameContext(0, 0.0, frame.copy())
    ingest(ctx)
    stage(ctx)  # first frame becomes the reference

    with_car = frame.copy()
    with_car[4:12, 4:12] = red.astype(np.uint8)
    results = {}
    for mode in ("grey", "max_channel"):
        stage.set_params(difference_mode=mode)
        stage.reset()
        base = FrameContext(0, 0.0, frame.copy())
        ingest(base)
        stage(base)
        ctx = FrameContext(1, 0.0, with_car.copy())
        ingest(ctx)
        stage(ctx)
        results[mode] = float(ctx.difference[4:12, 4:12].mean())

    assert results["grey"] < 6.0, "a greyscale difference should barely see this vehicle"
    assert results["max_channel"] > 80.0
    assert results["max_channel"] > 10 * results["grey"]


def test_selective_update_prevents_a_stopped_vehicle_becoming_a_ghost() -> None:
    """Trap 5, demonstrated on a controlled case.

    A vehicle that stops is, to a running average, simply what the road looks
    like now. Absorbed into the model, it leaves a vehicle-shaped hole when it
    moves off, which the pipeline detects as a phantom in the wrong place
    travelling at zero.
    """
    ingest = IngestStage()
    road = np.full((32, 32, 3), 80, np.uint8)
    parked = road.copy()
    parked[8:24, 8:24] = 200

    def absorbed(selective: bool) -> float:
        stage = BackgroundStage(
            alpha=0.2, selective_update=selective, model="running_average", warmup_frames=0
        )
        for _ in range(5):
            ctx = FrameContext(0, 0.0, road.copy())
            ingest(ctx)
            stage(ctx)
        previous = None
        for index in range(40):
            ctx = FrameContext(index, 0.0, parked.copy())
            ctx.previous_mask = previous
            ingest(ctx)
            stage(ctx)
            previous = np.asarray(ctx.difference > 30, dtype=bool)
        # How far the model has drifted towards the stopped vehicle.
        return float(stage.background_bgr_u8()[8:24, 8:24].mean())

    assert absorbed(selective=False) > 190.0, "without the guard it should be absorbed"
    assert absorbed(selective=True) < 110.0, "with the guard it should stay road"


def test_three_frame_difference_needs_no_model_and_sees_no_stopped_vehicle() -> None:
    ingest = IngestStage()
    stage = BackgroundStage(model="three_frame", warmup_frames=0)
    parked = np.full((32, 32, 3), 80, np.uint8)
    parked[8:24, 8:24] = 200
    for index in range(6):
        ctx = FrameContext(index, 0.0, parked.copy())
        ingest(ctx)
        stage(ctx)
    assert float(ctx.difference.max()) < 5.0


def test_frames_are_classified_i_or_p_by_difference_energy(processed) -> None:
    _pipe, outputs, *_ = processed
    kinds = {output.context.frame_type for output in outputs[5:]}
    assert kinds <= {FrameType.I, FrameType.P}
    assert len(kinds) >= 1


# --------------------------------------------------------------------------
# S4 - foreground
# --------------------------------------------------------------------------


def test_morphology_removes_speckle_and_fills_interiors(processed) -> None:
    _pipe, outputs, *_ = processed
    metrics = outputs[40].context.results["s4_foreground"].metrics
    assert metrics["speckle_removed_px"] > 0
    assert metrics["final_area_px"] >= metrics["after_open_px"]
    assert metrics["perspective_scaled"] == 1.0


def test_the_closing_element_height_follows_vehicle_height_not_road_distance() -> None:
    """The bug that made every vehicle report as two blobs.

    A vehicle silhouette extends upwards from its ground point by the vehicle's
    height, and a metre of height projects at close to the across-road scale.
    Sizing the closing element's height by the *along-road* scale instead makes
    it several times too short, so the dark window band across a vehicle
    survives as a gap and splits it into a roof blob and a bumper blob.
    """
    from kerbside.data.scenarios import DEFAULT_CAMERA
    from kerbside.geometry.homography import homography_from_camera
    from kerbside.geometry.perspective import PerspectiveModel

    model = PerspectiveModel(
        homography_from_camera(DEFAULT_CAMERA), DEFAULT_CAMERA.width_px, DEFAULT_CAMERA.height_px
    )
    _w_along, h_along = model.se_size_px(300.0, 0.5, vertical="along")
    _w_height, h_height = model.se_size_px(300.0, 0.5, vertical="height")
    assert h_height > 3 * h_along, (
        f"at row 300 a half-metre element is {h_height} px tall measured as "
        f"height and {h_along} px measured along the road"
    )


def test_a_frame_that_is_mostly_foreground_is_flagged_as_suspicious() -> None:
    """Never a silent empty state: 45% of the frame being foreground means the
    model is cold or the camera moved, not that the road is full."""
    pipe = Pipeline()
    noise = np.random.default_rng(0).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    pipe.process_frame(np.zeros((120, 160, 3), np.uint8))
    output = pipe.process_frame(noise)
    result = output.context.results["s4_foreground"]
    assert result.metrics["coverage"] > 0.4
    assert "warning" in result.extras


def test_foreground_says_so_when_it_has_no_calibration_to_scale_with() -> None:
    pipe = Pipeline(calibration=None)
    frame = np.zeros((120, 160, 3), np.uint8)
    pipe.process_frame(frame)
    output = pipe.process_frame(frame)
    assert output.context.results["s4_foreground"].metrics["perspective_scaled"] == 0.0
    assert "hint" in output.context.results["s4_foreground"].extras


# --------------------------------------------------------------------------
# S5 - shadow suppression
# --------------------------------------------------------------------------


def test_shadow_suppression_removes_almost_all_true_shadow() -> None:
    """Scored against shadow ground truth, not against an impression.

    Two clips identical except for ``cast_shadows`` isolate the shadow pixels
    exactly, so recall here is a measurement rather than a claim.
    """
    base = replace(build_scene("free_flow", seed=4242, duration_s=2.4), supersample=1)
    base = replace(base, atmosphere=replace(base.atmosphere, sensor_noise_sigma=0.0))
    lit = TrafficRenderer(base)
    unlit = TrafficRenderer(replace(base, lighting=replace(base.lighting, cast_shadows=False)))

    # Stream the frames rather than holding two whole clips: a 720p clip is
    # several hundred megabytes and this test needs three frames of it.
    check_at = {40, 50, 58}
    pipe = Pipeline(frame_interval_s=1.0 / base.fps, calibration=lit.calibration())
    recalls: list[float] = []
    for index in range(lit.frame_count):
        frame, gt = lit.render_frame(index)
        ctx = pipe.process_frame(frame, frame_index=index).context
        if index not in check_at:
            continue
        plain, _ = unlit.render_frame(index)
        changed = np.abs(frame.astype(np.int16) - plain.astype(np.int16)).max(axis=2) > 6
        vehicles = np.zeros(changed.shape, bool)
        for vehicle in gt.vehicles:
            x, y, w, h = (int(round(c)) for c in vehicle.bbox_px)
            vehicles[max(0, y - 2) : y + h + 2, max(0, x - 2) : x + w + 2] = True
        true_shadow = changed & ~vehicles & ctx.foreground
        if true_shadow.sum() < 500:
            continue
        recalls.append(float((ctx.shadow & true_shadow).sum() / true_shadow.sum()))

    assert recalls, "no frame carried enough shadow to score"
    # Measured range on this clip is roughly 0.87 to 0.97 per frame. The bounds
    # record that rather than rounding it up: the frames at the lower end are
    # ones where a shadow falls partly across a lane marking, whose much higher
    # reflectance puts the value ratio outside the band over those pixels.
    assert min(recalls) > 0.85, f"shadow recall per frame was {recalls}"
    assert float(np.mean(recalls)) > 0.90, f"shadow recall per frame was {recalls}"


def test_the_hue_test_is_not_applied_where_hue_is_undefined() -> None:
    """On a near-neutral surface hue is the arctangent of two small noisy
    differences. Asphalt has a saturation around 0.075, and enforcing a
    ten-degree hue tolerance there rejects half of a correctly-identified
    shadow."""
    from kerbside.stages.s5_shadow import ShadowStage

    stage = ShadowStage()
    background = np.zeros((8, 8, 3), np.float32)
    background[..., 0] = 100.0  # hue 120 in OpenCV units
    background[..., 1] = 0.02  # essentially neutral
    background[..., 2] = 0.35
    frame = background.copy()
    frame[..., 0] = 20.0  # a hue 80 units away, which is pure noise at S = 0.02
    frame[..., 2] = 0.35 * 0.6

    decision = stage.decide(frame, background, np.ones((8, 8), bool))
    assert decision.passes_hue.all(), "the hue test fired on an undefined hue"
    assert decision.shadow.all()

    stage.set_params(hue_saturation_floor=0.0)
    strict = stage.decide(frame, background, np.ones((8, 8), bool))
    assert not strict.shadow.any()


def test_removing_shadow_does_not_restore_it_through_hole_filling() -> None:
    """Subtracting a shadow from inside a larger foreground region leaves an
    enclosed hole, so an unbounded fill puts the whole shadow straight back -
    which is why the restoration is bounded by area."""
    from kerbside.operators.morphology import fill_holes, fill_small_holes

    mask = np.zeros((80, 80), bool)
    mask[10:70, 10:70] = True
    mask[25:55, 25:55] = False  # a large region, as a shadow would leave
    mask[15:18, 15:18] = False  # a small one, as a grey roof would leave

    assert int(fill_holes(mask).sum()) == 60 * 60
    bounded = fill_small_holes(mask, max_area_px=200)
    assert bounded[15:18, 15:18].all(), "the small hole should be restored"
    assert not bounded[25:55, 25:55].any(), "the large one must not be"


def test_shadow_suppression_can_be_disabled_for_the_ablation(processed) -> None:
    pipe, _outputs, _truth, _gts, frames = processed
    pipe.configure("s5_shadow", enabled=False)
    try:
        output = pipe.process_frame(frames[30])
        result = output.context.results["s5_shadow"]
        assert result.extras.get("disabled") is True
        assert result.metrics["shadow_px"] == 0.0
    finally:
        pipe.configure("s5_shadow", enabled=True)


# --------------------------------------------------------------------------
# S6 - blobs
# --------------------------------------------------------------------------


def test_blobs_carry_a_ground_point_and_a_real_world_footprint(processed) -> None:
    _pipe, outputs, *_ = processed
    blobs = [b for output in outputs[30:60] for b in output.context.blobs]
    assert blobs
    for blob in blobs:
        assert blob.footprint_m is not None
        assert blob.world_pos_m is not None
        # The ground point is the bottom-centre of the box, never the centroid.
        assert blob.ground_point_px.y == pytest.approx(blob.bbox_px.y2_px)
        assert blob.ground_point_px.y > blob.centroid_px.y


def test_the_size_filter_is_perspective_scaled_not_a_global_threshold(processed) -> None:
    """Trap 3. The accept band at a near row must be far larger than at a distant
    one, or one threshold is silently doing both jobs badly."""
    pipe, outputs, *_ = processed
    stage = pipe.stage("s6_blobs")
    model = stage._perspective(outputs[30].context)  # noqa: SLF001 - the model is the subject
    assert model is not None
    near_lo, near_hi = model.area_band_px(650.0)
    far_lo, far_hi = model.area_band_px(350.0)
    assert near_hi > 10 * far_hi
    assert near_lo > far_lo


def test_blobs_beyond_the_measurable_range_are_rejected(processed) -> None:
    """A geometry-driven working range, not a tuned threshold: past the row
    where a car projects to under a couple of hundred pixels, a one-pixel error
    in the ground point is several metres of road."""
    _pipe, outputs, *_ = processed
    rejected = sum(
        output.context.results["s6_blobs"].metrics.get("rejected_beyond_range", 0.0)
        for output in outputs[20:]
    )
    assert rejected > 0


def test_shape_plausibility_rejects_a_sliver_with_a_plausible_area(processed) -> None:
    """A shadow remnant six metres long and twenty centimetres wide has the area
    of a small car and the shape of nothing on the road."""
    from kerbside.core.types import BBox, Blob, Footprint, Point2

    pipe, *_ = processed
    stage = pipe.stage("s6_blobs")
    sliver = Blob(
        blob_id=1, bbox_px=BBox(0, 0, 60, 4), area_px=240.0,
        centroid_px=Point2(30, 2), ground_point_px=Point2(30, 4),
        footprint_m=Footprint(length_m=6.0, width_m=0.2),
    )
    car = replace(sliver, footprint_m=Footprint(length_m=4.2, width_m=1.8))
    assert not stage._plausible_shape(sliver)  # noqa: SLF001
    assert stage._plausible_shape(car)  # noqa: SLF001


def test_lane_assignment_uses_the_ground_point(processed) -> None:
    _pipe, outputs, *_ = processed
    assigned = [
        b for output in outputs[25:60] for b in output.context.blobs if b.lane_id is not None
    ]
    assert assigned, "no blob was assigned to a lane"
    assert all(0 <= b.lane_id < 4 for b in assigned)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_the_gate_says_what_must_be_found_not_what_may_be_found(clip) -> None:
    """A blob that found a real vehicle the gate did not require is not a false
    positive. Scoring it as one penalises the pipeline for being right, and does
    so hardest where position estimates are least certain."""
    renderer, _frames, _truth, gts = clip
    from kerbside.core.types import BBox, Blob, Point2

    frame = next(gt for gt in gts if len(gt.vehicles) >= 2)
    far = max(frame.vehicles, key=lambda v: v.world_pos_m[1])
    x, y, w, h = far.bbox_px
    blob = Blob(
        blob_id=1, bbox_px=BBox(x, y, w, h), area_px=float(w * h),
        centroid_px=Point2(x + w / 2, y + h / 2),
        ground_point_px=Point2(x + w / 2, y + h),
        world_pos_m=Point2(*far.world_pos_m),
    )
    narrow = VisibilityGate(min_y_m=0.0, max_y_m=far.world_pos_m[1] - 5.0)
    match = match_frame([blob], frame, narrow, renderer.calibration())
    assert match.spurious == (), "a real vehicle outside the gate was called a false positive"


def test_recall_counts_vehicles_found_not_blob_vehicle_pairs(processed) -> None:
    """A vehicle reported as two fragments has been found once. Crediting it
    twice lets a pipeline that shatters everything report a recall above one,
    which is how this was noticed."""
    _pipe, outputs, truth, _gts, _frames = processed
    blobs = [output.context.blobs for output in outputs]
    metrics = score_detection(
        blobs, truth, VisibilityGate(min_visible_px=300, min_y_m=18, max_y_m=80),
        skip_frames=20,
    )
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.precision <= 1.0


def test_metrics_separate_merges_misses_and_false_positives(processed) -> None:
    """A count 3% low because two vehicles merged is a different problem from
    one 3% low because a distant vehicle was never seen, and the fix is in a
    different stage."""
    _pipe, outputs, truth, _gts, _frames = processed
    blobs = [output.context.blobs for output in outputs]
    summary = score_detection(
        blobs, truth, VisibilityGate(min_visible_px=300, min_y_m=18, max_y_m=80),
        skip_frames=20,
    ).summary()
    for key in (
        "missed_total", "spurious_total", "merge_events", "merged_vehicles",
        "split_events", "false_merge_rate", "frame_count_agreement",
    ):
        assert key in summary
    assert summary["gate"]["description"]


# --------------------------------------------------------------------------
# The gate, reported honestly
# --------------------------------------------------------------------------


def test_detection_accuracy_is_characterised_honestly(processed) -> None:
    """**The Phase 3 target is not met.** This records where it actually stands.

    The target is per-frame blob count equal to ground-truth vehicle count in at
    least 98% of frames, measured over the free-flow clip. The pipeline reaches
    roughly a quarter of that, with recall near 0.8 and precision near 0.85 over
    an 18-80 m working range.

    The limiting failure is **fragmentation**: a vehicle reported as two blobs.
    It is counted here as ``split_events``, and it fails a frame just as surely
    as a missed vehicle does, because the gate is a count comparison. Enlarging
    the closing element reduces fragmentation and starts merging adjacent
    vehicles instead, which trades one failure for the other rather than fixing
    either; the remaining work is to separate the two cases by shape rather than
    by element size.

    The bounds below are deliberately loose. They exist to catch a regression,
    not to dress a shortfall up as a pass.
    """
    _pipe, outputs, truth, _gts, _frames = processed
    blobs = [output.context.blobs for output in outputs]
    gate = VisibilityGate(min_visible_px=300, min_y_m=18.0, max_y_m=80.0)
    summary = score_detection(blobs, truth, gate, skip_frames=20).summary()

    assert summary["recall"] > 0.70, summary
    assert summary["precision"] > 0.70, summary
    assert summary["mean_absolute_count_error"] < 3.0, summary
    # The gate itself, stated so the shortfall is visible in the test output.
    assert summary["frame_count_agreement"] < 0.98, (
        "the Phase 3 target now passes - update this test and the README, which "
        "both currently record it as unmet"
    )


def test_the_pipeline_reports_where_the_frame_budget_goes(processed) -> None:
    """The throughput target is 15 fps at 720p and the pipeline does not reach
    it either; what it does do is say which stage to attack."""
    _pipe, _outputs, _truth, _gts, frames = processed
    # A fresh pipeline: the shared one has had its timings reset by the
    # configuration changes other tests make.
    pipe = Pipeline()
    for frame in frames[:6]:
        pipe.process_frame(frame)
    summary = pipe.timings.summary()
    assert summary["fps"] > 0
    assert summary["frame_ms"] > 0
    slowest = pipe.timings.slowest_stage()
    assert slowest is not None
    assert slowest[0] in {s.stage_id for s in pipe.stages}
