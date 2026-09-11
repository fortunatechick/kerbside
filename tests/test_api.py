"""The API surface, exercised end to end.

Phase 5's gate is that every endpoint is exercised end to end, and until this
file existed that was only ever true of a session someone drove by hand. Three
of the defects found while wiring the console would have been caught here: a
route whose return annotation FastAPI cannot build a response model from, an
analysis that exhausted memory on a clip of ordinary length, and a registry that
is empty unless something else imported the operator modules first.

The session is a process-wide singleton, so these tests run against one small
clip rendered once. That is deliberate: they are checking that the plumbing
carries the right values, not re-checking the pipeline, which the stage tests
already do at length.
"""

from __future__ import annotations

import base64

import pytest
from fastapi.testclient import TestClient

from kerbside.api import session as session_module
from kerbside.core.registry import REGISTRY

# ``serve`` is what production runs; importing the router alone would test a
# different application than the one that ships.
from kerbside.serve import create_app

SCENARIO = {
    "name": "free_flow",
    "duration_s": 1.2,
    "seed": 4242,
    "width_px": 320,
    "height_px": 180,
    "supersample": 1,
}


@pytest.fixture(scope="module")
def client():
    """One client over one loaded clip, small enough to analyse in seconds."""
    session_module._SESSION = None  # a fresh session, whatever ran before
    app = create_app()
    with TestClient(app) as c:
        response = c.post("/api/source/scenario", json=SCENARIO)
        assert response.status_code == 200, response.text
        yield c
    session_module._SESSION = None


# --------------------------------------------------------------------------
# Source and status
# --------------------------------------------------------------------------


def test_scenarios_are_listed_with_what_each_one_exercises(client):
    body = client.get("/api/scenarios").json()
    names = {s["name"] for s in body["scenarios"]}
    assert {"free_flow", "night", "long_shadows", "camera_shake"} <= names
    # A scenario the user cannot tell apart from another is not a choice, so
    # every one carries the failure mode it isolates.
    for scenario in body["scenarios"]:
        assert scenario["description"].strip()
        assert scenario["exercises"].strip()


def test_loading_a_scenario_reports_the_measured_frame_rate(client):
    state = client.get("/api/status").json()["state"]
    assert state["width_px"] == 320
    assert state["height_px"] == 180
    assert state["frame_count"] > 10
    assert state["has_ground_truth"] is True
    # Speed divides by the frame interval, so the measured rate is the one that
    # matters and it is reported separately from the declared one.
    assert state["measured_fps"] == pytest.approx(state["container_fps"], rel=1e-3)


def test_an_unknown_scenario_says_so_rather_than_failing_obscurely(client):
    response = client.post("/api/source/scenario", json={**SCENARIO, "name": "nope"})
    assert response.status_code == 400
    assert "nope" in response.json()["detail"]


def test_a_missing_video_path_is_a_message_not_a_stack_trace(client):
    response = client.post("/api/source/video", json={"path": "does/not/exist.mp4"})
    assert response.status_code == 400
    assert response.json()["detail"]


# --------------------------------------------------------------------------
# Pipeline and the operator registry
# --------------------------------------------------------------------------


def test_the_pipeline_describes_all_nine_stages_with_units_and_rationale(client):
    stages = client.get("/api/pipeline").json()["stages"]
    assert [s["stage_id"][:2] for s in stages] == [f"s{i}" for i in range(1, 10)]
    for stage in stages:
        assert stage["units"], f"{stage['stage_id']} claims no syllabus unit"
        assert stage["rationale"].strip(), f"{stage['stage_id']} has no rationale"
        for spec in stage["params"]:
            assert spec["label"].strip()
            assert spec["name"] in stage["values"]


def test_the_operator_registry_is_populated_and_explains_itself(client):
    """The registry fills by import side effect, so its size is pinned here.

    ``@register`` runs when the operator module is imported. Anything that
    reaches ``REGISTRY`` without importing ``kerbside.operators`` sees an empty
    registry and no error - which would silently empty the part of the UI that
    exists to satisfy "every operator must be explainable".
    """
    operators = client.get("/api/operators").json()["operators"]
    assert len(operators) == 35
    assert len(operators) == len(REGISTRY.to_dict()["operators"])
    assert {u for op in operators for u in op["units"]} <= set(range(1, 9))
    for op in operators:
        assert op["rationale"].strip(), f"{op['operator_id']} has no rationale"
        assert op["units"], f"{op['operator_id']} claims no syllabus unit"


def test_setting_a_parameter_reports_it_as_modified(client):
    body = client.post(
        "/api/pipeline/params",
        json={"stage_id": "s4_foreground", "params": {"threshold_mode": "fixed"}},
    ).json()
    assert body["stage"]["values"]["threshold_mode"] == "fixed"
    assert "threshold_mode" in body["stage"]["modified"]

    stages = client.post("/api/pipeline/reset").json()["stages"]
    s4 = next(s for s in stages if s["stage_id"] == "s4_foreground")
    assert s4["modified"] == []


def test_an_out_of_range_parameter_is_refused_at_the_point_it_is_set(client):
    response = client.post(
        "/api/pipeline/params",
        json={"stage_id": "s4_foreground", "params": {"threshold_mode": "banana"}},
    )
    assert response.status_code == 400
    assert "banana" in response.json()["detail"]


def test_an_unknown_stage_is_named_in_the_error(client):
    response = client.post(
        "/api/pipeline/params", json={"stage_id": "s99_nope", "params": {}}
    )
    assert response.status_code == 400
    assert "s99_nope" in response.json()["detail"]


# --------------------------------------------------------------------------
# Analysis, frames and telemetry
# --------------------------------------------------------------------------


def test_analysis_runs_the_clip_and_reports_where_the_time_went(client):
    body = client.post("/api/analyse").json()
    assert body["analysed"] is True
    assert body["frames"] > 10
    slowest, ms = body["slowest_stage"]
    assert slowest.startswith("s")
    assert ms > 0
    assert set(body["timings"]) >= {f"s{i}" for i in range(1, 10)} or any(
        k.startswith("s8") for k in body["timings"]
    )


def test_a_frame_is_returned_as_jpeg_with_the_overlay_applied(client):
    response = client.get("/api/frame/5.jpg?boxes=1&ground_points=1&labels=1")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content[:2] == b"\xff\xd8"  # JPEG SOI


def test_the_mask_layers_still_render_after_intermediates_are_released(client):
    """Analysis releases the big arrays; the mask layers need them back.

    This is the path that ``Session.context(intermediates=True)`` exists for.
    Frame 0 is well outside the retained window once the clip has been analysed,
    so asking for the mask here forces the replay rather than exercising the
    happy path.
    """
    response = client.get("/api/frame/0.jpg?mask=1&shadow=1&blobs=1")
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8"


def test_the_stage_inspector_returns_every_stage_with_its_metrics(client):
    body = client.get("/api/frame/5/stages?include_images=true").json()
    assert body["frame_index"] == 5
    assert len(body["stages"]) == 9
    imaged = [s for s in body["stages"] if s.get("image_png")]
    assert imaged, "no stage produced an image for the inspector"
    for entry in imaged:
        assert base64.b64decode(entry["image_png"])[:8] == b"\x89PNG\r\n\x1a\n"
    speed = next(s for s in body["stages"] if s["stage_id"] == "s8_speed")
    assert speed["units"] == [8] or 8 in speed["units"]


def test_the_probe_reports_the_decision_not_just_the_numbers(client):
    body = client.get("/api/frame/5/probe?x=100&y=140").json()
    assert body["rgb"] and body["background_rgb"] and body["hsv"]
    # A probe that gives a difference and a threshold and leaves the comparison
    # to the reader is half a probe.
    assert body["is_foreground"] == (body["difference"] >= body["threshold"])


def test_probing_outside_the_frame_says_the_frame_size(client):
    response = client.get("/api/frame/5/probe?x=99999&y=0")
    assert response.status_code == 400
    assert "320" in response.json()["detail"]


def test_transport_clamps_the_cursor_to_the_clip(client):
    frames = client.get("/api/status").json()["frames"]
    body = client.post("/api/transport", json={"cursor": 10_000}).json()
    assert body["cursor"] == frames - 1
    body = client.post("/api/transport", json={"cursor": -5}).json()
    assert body["cursor"] == 0
    body = client.post("/api/transport", json={"step": 3}).json()
    assert body["cursor"] == 3
    body = client.post("/api/transport", json={"playing": True, "speed": 2.0}).json()
    assert body["playing"] is True and body["speed"] == 2.0
    client.post("/api/transport", json={"playing": False, "speed": 1.0})


def test_telemetry_carries_both_speed_routes_for_every_track(client):
    """The comparison is the finding, so neither route may go missing.

    The field names matter as much as the values: these were once named for a
    centroid the stage no longer uses, and a reader would have concluded Route A
    was the very estimator the design argues against.
    """
    with client.websocket_connect("/ws/telemetry") as socket:
        payload = socket.receive_json()
    assert "frame_index" in payload and "total_count" in payload
    for track in payload["tracks"]:
        assert "speed_kmph_track" in track
        assert "speed_kmph_blockmatch" in track
        assert "speed_kmph_centroid" not in track


# --------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------


def test_calibration_reports_the_implied_camera_and_a_verdict(client):
    body = client.get("/api/calibration").json()
    assert body["calibrated"] is True
    assert body["verdict"].strip()
    camera = body["camera"]
    assert camera["height_m"] > 0
    assert camera["focal_px"] > 0
    assert isinstance(camera["plausible"], bool)
    # The suggestions exist to stop a nominal figure being used without checking
    # the local standard, so each must carry its caution.
    for suggestion in body["suggestions"]:
        assert suggestion["caution"].strip()


def test_four_clicks_calibrate_and_the_residual_is_flagged_as_uninformative(client):
    state = client.get("/api/status").json()["state"]
    w, h = state["width_px"], state["height_px"]
    body = client.post(
        "/api/calibration",
        json={
            "points_px": [[w * 0.30, h * 0.90], [w * 0.70, h * 0.90],
                          [w * 0.58, h * 0.55], [w * 0.42, h * 0.55]],
            "rect_width_m": 7.0,
            "rect_length_m": 20.0,
            "lane_count": 4,
            "lane_width_m": 3.5,
        },
    ).json()
    assert body["calibrated"] is True
    # Four points determine a homography exactly, so the residual is
    # structurally zero and proves nothing. Saying so is the point.
    assert body["exactly_determined"] is True


def test_a_degenerate_quadrilateral_is_refused(client):
    response = client.post(
        "/api/calibration",
        json={
            "points_px": [[10, 10], [10, 10], [10, 10], [10, 10]],
            "rect_width_m": 3.5,
            "rect_length_m": 12.0,
        },
    )
    assert response.status_code == 400


def test_the_distance_tool_measures_between_two_road_points(client):
    state = client.get("/api/status").json()["state"]
    w, h = state["width_px"], state["height_px"]
    body = client.post(
        "/api/calibration/measure",
        json={"a_px": [w * 0.35, h * 0.88], "b_px": [w * 0.65, h * 0.88]},
    ).json()
    assert body["distance_m"] is not None
    assert body["distance_m"] > 0
    # The note is the whole value of the tool: a calibration can be perfectly
    # self-consistent and still wrong by a constant factor.
    assert "tape measure" in body["note"]


def test_a_point_above_the_horizon_has_no_road_position(client):
    body = client.post(
        "/api/calibration/measure", json={"a_px": [10, 0], "b_px": [20, 0]}
    ).json()
    assert body["distance_m"] is None


def test_the_metre_grid_comes_back_as_polylines(client):
    body = client.get("/api/calibration/grid?step_m=2.0").json()
    assert body["step_m"] == 2.0
    assert body["polylines"]
    assert all(len(point) == 2 for line in body["polylines"] for point in line)


def test_the_birdseye_rectification_renders(client):
    response = client.get("/api/calibration/birdseye/5.jpg?px_per_m=4")
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8"


def test_a_calibration_round_trips_through_disk(client, tmp_path, monkeypatch):
    from kerbside.api import routes

    monkeypatch.setattr(routes, "calibration_path_for", lambda key: tmp_path / f"{key}.json")
    saved = client.post("/api/calibration/save", json={"key": "unit-test"})
    assert saved.status_code == 200
    loaded = client.post("/api/calibration/load", json={"key": "unit-test"})
    assert loaded.status_code == 200
    assert loaded.json()["calibrated"] is True


# --------------------------------------------------------------------------
# Analytics, ablations, report
# --------------------------------------------------------------------------


def test_analytics_never_calls_anything_a_violation(client):
    body = client.get("/api/analytics").json()
    assert "flagged for review" in body["disclaimer"].lower()
    text = str(body).lower()
    assert "violation" not in text
    assert "offence" not in text
    assert body["speed"]["p85_kmph"] >= 0
    assert len(body["speed_histogram"]["counts"]) > 0


def test_analytics_csv_is_a_download_with_a_header_row(client):
    response = client.get("/api/analytics/csv")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    header = response.text.splitlines()[0]
    # Units in the column names, for the same reason they are in the identifiers.
    assert "speed_kmph" in header


def test_every_ablation_states_its_question_before_it_runs(client):
    ablations = client.get("/api/ablations").json()["ablations"]
    assert len(ablations) == 9
    for ablation in ablations:
        assert "?" in ablation["question"], ablation["key"]


def test_an_unknown_ablation_is_refused(client):
    assert client.post("/api/ablations/not-a-study?duration_s=1").status_code == 400


def test_the_report_embeds_the_parameters_that_produced_it(client):
    body = client.get("/api/report").json()
    assert body["generated_utc"]
    assert len(body["stages"]) == 9
    # A result without the settings that produced it cannot be reproduced or
    # argued with, which is the entire purpose of the document.
    assert all("values" in stage for stage in body["stages"])
    assert body["calibration"] is not None


def test_the_html_report_is_self_contained(client):
    response = client.get("/api/report.html")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    html = response.text
    assert html.lstrip().startswith("<!doctype html>")
    assert "<style>" in html, "styles must be inline, not a linked file"
    assert 'src="http' not in html, "a figure is being fetched rather than embedded"
    assert "not an enforcement device" in html.lower()


def test_the_log_records_what_the_session_did(client):
    entries = client.get("/api/log?limit=10").json()["entries"]
    assert entries
    for entry in entries:
        assert entry["source"] and entry["message"]


# --------------------------------------------------------------------------
# Static hosting
# --------------------------------------------------------------------------


def test_the_root_route_answers_whether_or_not_the_frontend_is_built(client):
    """A 404 at the root looks like a broken install; this must not 404.

    Whichever branch ``create_app`` took, the root returns a document - the
    built console, or the page explaining how to build it. This is also the
    route whose ``FileResponse | HTMLResponse`` annotation FastAPI could not
    build a response model from, which failed at import time and only once
    ``frontend/dist`` existed.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.text.lstrip()[:15].lower().startswith("<!doctype html")


def test_the_openapi_schema_builds(client):
    """Proves every route's annotations are ones FastAPI can model."""
    schema = client.get("/openapi.json").json()
    assert "/api/status" in schema["paths"]
    assert "/api/analytics" in schema["paths"]


# --------------------------------------------------------------------------
# Phase 6's gate, measured through the API the console actually uses
# --------------------------------------------------------------------------


def test_calibrating_through_the_api_lands_within_two_percent(client):
    """Phase 6's gate: calibrate through the UI to within 2% of the truth.

    Phase 1 established that the *estimator* recovers the rendering homography
    essentially exactly, and that click precision rather than the estimator is
    what bounds the answer. This is the same question asked of the path a user
    actually takes: project four known road-plane corners to pixels, perturb
    them by half a pixel the way a careful click would, POST them to
    ``/api/calibration``, and then measure distances back out through
    ``/api/calibration/measure``.

    Half a pixel is the right perturbation. Exact clicks would measure the
    estimator, which Phase 1 already did; whole-pixel clicks are known to have a
    tail past 2% and would be measuring the mouse.
    """
    import numpy as np

    from kerbside.api.session import get_session
    from kerbside.geometry.homography import apply_h_single

    # At 720p, because that is the frame a user clicks on and click error is
    # what bounds this answer. Half a pixel on the module's 320x180 fixture
    # would be two pixels here, which measures a shakier hand than the gate
    # describes. This test runs last in the file for that reason: it replaces
    # the session's source.
    assert client.post(
        "/api/source/scenario",
        json={**SCENARIO, "duration_s": 0.4, "width_px": 1280, "height_px": 720},
    ).status_code == 200

    session = get_session()
    truth = session.calibration
    assert truth is not None, "the synthetic scenario should arrive calibrated"

    def to_px(x_m: float, y_m: float) -> tuple[float, float]:
        return apply_h_single(truth.h_world_to_img, x_m, y_m)

    # A large quadrilateral, because a small one amplifies click error - which is
    # the advice the calibration view gives the user.
    corners_m = [(-3.5, 22.0), (3.5, 22.0), (3.5, 46.0), (-3.5, 46.0)]
    rng = np.random.default_rng(20240517)
    clicked = [
        [c + float(rng.uniform(-0.5, 0.5)) for c in to_px(x, y)] for x, y in corners_m
    ]

    response = client.post(
        "/api/calibration",
        json={
            "points_px": clicked,
            "rect_width_m": 7.0,
            "rect_length_m": 24.0,
            "origin_x_m": -3.5,
            "origin_y_m": 22.0,
            "lane_count": 4,
            "lane_width_m": 3.5,
        },
    )
    assert response.status_code == 200, response.text

    # Now measure back out, over the working range the pipeline actually uses.
    errors: list[float] = []
    for y_start, y_end in [(20.0, 30.0), (25.0, 45.0), (30.0, 60.0), (22.0, 58.0)]:
        a = to_px(0.0, y_start)
        b = to_px(0.0, y_end)
        body = client.post(
            "/api/calibration/measure", json={"a_px": list(a), "b_px": list(b)}
        ).json()
        assert body["distance_m"] is not None, (y_start, y_end)
        expected = y_end - y_start
        errors.append(abs(body["distance_m"] - expected) / expected)

    worst = max(errors)
    assert worst < 0.02, (
        f"Phase 6's gate is 2%; the worst span is off by {worst:.2%}. "
        f"Per-span errors: {[f'{e:.2%}' for e in errors]}"
    )
