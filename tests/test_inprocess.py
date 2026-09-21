"""The in-process dispatcher that runs the console in a browser.

The web build hands every request to :func:`kerbside.api.inprocess.dispatch`
instead of an HTTP server, under Pyodide. Nothing about that path can be
exercised in a browser from pytest, but the dispatcher itself is ordinary Python
- so these tests hold it to the one property that matters: **it answers exactly
as the server does.** Same status, same body, same errors, for the same request.
If it drifts, the deployed site shows something the local console would not.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from kerbside.api import session as session_module
from kerbside.api.inprocess import dispatch, dispatch_text, set_progress
from kerbside.serve import create_app

CLIP = {"name": "free_flow", "duration_s": 1.2, "seed": 4242, "width_px": 320, "height_px": 180}


def call(method: str, path: str, query: dict | None = None, body=None):
    return asyncio.run(dispatch(method, path, query, body))


def as_json(result):
    status, content_type, body = result
    assert content_type.startswith("application/json"), content_type
    return status, json.loads(body)


@pytest.fixture(scope="module")
def loaded():
    session_module._SESSION = None
    status, _ = as_json(call("POST", "/api/source/scenario", body=CLIP))
    assert status == 200
    status, body = as_json(call("POST", "/api/analyse"))
    assert status == 200 and body["analysed"] is True
    yield
    set_progress(None)
    session_module._SESSION = None


def test_json_routes_answer_as_the_server_does(loaded) -> None:
    """Parity, checked against the real HTTP stack on the same session."""
    with TestClient(create_app()) as client:
        for path in ("/api/status", "/api/pipeline", "/api/operators", "/api/calibration",
                     "/api/analytics", "/api/telemetry", "/api/ablations"):
            status, body = as_json(call("GET", path))
            reference = client.get(path)
            assert status == reference.status_code, path
            assert body == reference.json(), f"{path} differs between in-process and HTTP"


def test_path_and_query_parameters_are_converted_like_fastapi_does(loaded) -> None:
    status, body = as_json(call("GET", "/api/frame/5/stages", {"include_images": "false"}))
    assert status == 200
    assert body["frame_index"] == 5  # "5" in the path arrives as the integer 5
    assert all("image_png" not in s for s in body["stages"])  # "false" arrives as False


def test_images_come_back_as_bytes_with_their_type(loaded) -> None:
    status, content_type, body = call("GET", "/api/frame/5.jpg", {"boxes": "1", "rev": "7"})
    assert status == 200
    assert content_type == "image/jpeg"
    assert body[:2] == b"\xff\xd8"


def test_downloads_come_back_whole(loaded) -> None:
    status, content_type, body = call("GET", "/api/report.html")
    assert status == 200 and content_type.startswith("text/html")
    assert b"not an enforcement device" in body.lower()
    status, content_type, body = call("GET", "/api/analytics/csv")
    assert status == 200 and b"speed_kmph" in body.splitlines()[0]


def test_errors_keep_the_servers_status_and_message(loaded) -> None:
    bad_stage = {"stage_id": "s99", "params": {}}
    status, body = as_json(call("POST", "/api/pipeline/params", body=bad_stage))
    assert status == 400 and "s99" in body["detail"]

    status, body = as_json(call("POST", "/api/source/scenario", body={"duration_s": -1}))
    assert status == 422  # the same Pydantic model refuses it

    status, body = as_json(call("GET", "/api/frame/5/probe", {"x": "99999", "y": "0"}))
    assert status == 400 and "320" in body["detail"]

    status, body = as_json(call("GET", "/api/frame/5/probe", {"x": "10"}))
    assert status == 422 and "y" in body["detail"]  # a required query parameter is missing

    status, body = as_json(call("GET", "/api/no/such/route"))
    assert status == 404


def test_the_stream_says_why_it_is_unavailable_rather_than_hanging(loaded) -> None:
    """The MJPEG stream never ends. Called in-process it would never return."""
    status, body = as_json(call("GET", "/api/stream.mjpg"))
    assert status == 404
    assert "live connection" in body["detail"]


def test_the_string_entry_point_matches_the_structured_one(loaded) -> None:
    """The worker calls dispatch_text across the JavaScript boundary."""
    direct = call("GET", "/api/frame/3/probe", {"x": "20", "y": "150"})
    via_text = asyncio.run(
        dispatch_text("GET", "/api/frame/3/probe", '{"x": "20", "y": "150"}', None)
    )
    assert direct == via_text

    body = json.dumps({"stage_id": "s4_foreground", "params": {"threshold_mode": "otsu"}})
    status, _, _ = asyncio.run(dispatch_text("POST", "/api/pipeline/params", "{}", body))
    assert status == 200


def test_long_operations_report_progress(loaded) -> None:
    """In a browser a minute of silence is indistinguishable from a crash."""
    events: list[tuple[str, int, int]] = []
    set_progress(lambda phase, done, total: events.append((phase, done, total)))
    try:
        call("POST", "/api/source/scenario", body={**CLIP, "duration_s": 0.6})
        call("POST", "/api/analyse")
    finally:
        set_progress(None)
    phases = {phase for phase, _, _ in events}
    assert {"render", "analyse"} <= phases
    analyse = [(d, t) for p, d, t in events if p == "analyse"]
    assert analyse[0][0] == 0 and analyse[-1][0] == analyse[-1][1] - 1


def test_ablations_accept_the_browser_resolution() -> None:
    """The browser build runs studies at a reduced size; the route must honour it."""
    session_module._SESSION = None
    status, body = as_json(
        call("POST", "/api/ablations/difference",
             {"duration_s": "0.8", "width_px": "320", "height_px": "180"})
    )
    assert status == 200, body
    assert body["rows"], "the study returned no rows"
    session_module._SESSION = None
