"""The HTTP and WebSocket surface.

Split by transport rather than by feature, because the transports have genuinely
different characteristics and the split is where the interesting decisions live:

* **REST** for anything request-shaped - loading a source, reading and setting
  parameters, calibrating, inspecting one stage on one frame, running an
  ablation.
* **MJPEG over plain HTTP** for the annotated video. The browser decodes it
  natively in an ``<img>``; there is no player to ship and nothing to buffer.
* **WebSocket** for per-frame telemetry. Small JSON at frame rate.

Raw frames deliberately do not go over the WebSocket. A 720p frame is 2.7 MB
uncompressed, and at 25 fps that saturates the socket and starves the telemetry
the console needs to stay responsive.

Every failure that a user can act on is raised as a :class:`StageError` and
returned as a 400 with the message intact. A bare 500 with a stack trace tells
the user nothing about the parameter set they just chose.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from kerbside.api.overlay import OverlayOptions, draw_overlay, encode_jpeg
from kerbside.api.session import Session, get_session
from kerbside.core.registry import REGISTRY
from kerbside.core.types import Calibration
from kerbside.data.scenarios import scenario_info, scenario_names
from kerbside.eval.ablation import ABLATIONS, run_ablation
from kerbside.geometry.calibration import (
    assess_calibration,
    birdseye_homography,
    calibrate_from_rectangle,
    calibration_path_for,
    lanes_from_world_bands,
    measure_distance_m,
    metre_grid_segments_px,
    reference_distance_suggestions,
)
from kerbside.stages.base import StageError

router = APIRouter()


def _fail(error: Exception) -> HTTPException:
    """Turn a stage error into a 400 the interface can display verbatim."""
    return HTTPException(status_code=400, detail=str(error))


# ==========================================================================
# Source and status
# ==========================================================================


class ScenarioRequest(BaseModel):
    name: str = "free_flow"
    duration_s: float = Field(12.0, gt=0.2, le=120.0)
    seed: int = 4242
    density: float | None = Field(None, ge=0.0, le=6.0)
    speed_min_kmph: float | None = Field(None, gt=0)
    speed_max_kmph: float | None = Field(None, gt=0)
    shadow_strength: float | None = Field(None, ge=0.0, le=1.0)
    width_px: int | None = Field(None, ge=160, le=1920)
    height_px: int | None = Field(None, ge=120, le=1080)
    supersample: int = Field(1, ge=1, le=3)


@router.get("/api/scenarios")
def list_scenarios() -> dict[str, Any]:
    return {
        "scenarios": [
            {
                "name": name,
                "label": scenario_info(name).label,
                "description": scenario_info(name).description,
                "exercises": scenario_info(name).exercises,
            }
            for name in scenario_names()
        ]
    }


@router.post("/api/source/scenario")
def load_scenario(request: ScenarioRequest) -> dict[str, Any]:
    session = get_session()
    speed_range = None
    if request.speed_min_kmph and request.speed_max_kmph:
        speed_range = (request.speed_min_kmph, request.speed_max_kmph)
    try:
        state = session.load_scenario(
            name=request.name,
            duration_s=request.duration_s,
            seed=request.seed,
            density=request.density,
            speed_kmph_range=speed_range,
            shadow_strength=request.shadow_strength,
            width_px=request.width_px,
            height_px=request.height_px,
            supersample=request.supersample,
        )
    except (StageError, ValueError, KeyError) as error:
        raise _fail(error) from error
    return state.to_dict()


class VideoRequest(BaseModel):
    path: str


@router.post("/api/source/video")
def load_video(request: VideoRequest) -> dict[str, Any]:
    session = get_session()
    try:
        return session.load_video(request.path).to_dict()
    except (StageError, ValueError, OSError) as error:
        raise _fail(error) from error


@router.get("/api/status")
def status() -> dict[str, Any]:
    return get_session().summary()


@router.post("/api/analyse")
def analyse(force: bool = False) -> dict[str, Any]:
    try:
        return get_session().analyse(force=force)
    except StageError as error:
        raise _fail(error) from error


@router.get("/api/log")
def log(limit: int = 200) -> dict[str, Any]:
    return {"entries": get_session().log(limit)}


# ==========================================================================
# Pipeline and operators
# ==========================================================================


@router.get("/api/pipeline")
def pipeline() -> dict[str, Any]:
    session = get_session()
    return {"stages": session.pipeline.describe()}


@router.get("/api/operators")
def operators() -> dict[str, Any]:
    return REGISTRY.to_dict()


class ParamRequest(BaseModel):
    stage_id: str
    params: dict[str, Any]


@router.post("/api/pipeline/params")
def set_params(request: ParamRequest) -> dict[str, Any]:
    session = get_session()
    before = dict(session.pipeline.stage(request.stage_id).params)
    started = time.perf_counter()
    try:
        session.pipeline.configure(request.stage_id, **request.params)
    except (KeyError, ValueError) as error:
        raise _fail(error) from error
    session._reset_analysis()  # noqa: SLF001 - the session owns this state
    changed = {
        k: v for k, v in session.pipeline.stage(request.stage_id).params.items()
        if before.get(k) != v
    }
    session.note(
        request.stage_id,
        "parameters changed: "
        + ", ".join(f"{k}={v}" for k, v in changed.items()) if changed else "no change",
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        changed=changed,
    )
    return {"stage": session.pipeline.stage(request.stage_id).describe()}


@router.post("/api/pipeline/reset")
def reset_params(stage_id: str | None = None) -> dict[str, Any]:
    session = get_session()
    stages = (
        [session.pipeline.stage(stage_id)] if stage_id else list(session.pipeline.stages)
    )
    for stage in stages:
        stage.set_params(**stage.defaults())
    session._reset_analysis()  # noqa: SLF001
    session.note("pipeline", f"reset {'one stage' if stage_id else 'all stages'} to defaults")
    return {"stages": session.pipeline.describe()}


def _encode_image(image: np.ndarray) -> str:
    """A stage output as a base64 PNG, for single-frame inspection.

    PNG rather than JPEG here: the inspector shows binary masks and difference
    images, where JPEG's block artefacts would be indistinguishable from the
    thing being inspected.
    """
    array = np.asarray(image)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    ok, buffer = cv2.imencode(".png", array)
    if not ok:
        raise StageError("failed to encode a stage image")
    return base64.b64encode(buffer).decode("ascii")


@router.get("/api/frame/{index}/stages")
def frame_stages(index: int, include_images: bool = True) -> dict[str, Any]:
    """Every stage's output for one frame - the pipeline view's data."""
    session = get_session()
    try:
        ctx = session.context(index, intermediates=True)
    except StageError as error:
        raise _fail(error) from error

    stages = []
    for stage in session.pipeline.stages:
        result = ctx.results.get(stage.stage_id)
        if result is None:
            continue
        entry: dict[str, Any] = {
            "stage_id": result.stage_id,
            "name": result.name,
            "units": list(result.units),
            "rationale": result.rationale,
            "elapsed_ms": round(result.elapsed_ms, 3),
            "metrics": {k: float(v) for k, v in result.metrics.items()},
            "extras": {
                k: v for k, v in result.extras.items() if isinstance(v, (str, int, float, dict, list, bool))
            },
            "modified": list(stage.differs_from_default()),
        }
        if include_images:
            image = result.image if result.image is not None else result.mask
            if image is not None:
                entry["image_png"] = _encode_image(image)
        stages.append(entry)
    return {"frame_index": index, "stages": stages}


@router.get("/api/frame/{index}/probe")
def probe(index: int, x: int, y: int) -> dict[str, Any]:
    """The pixel probe: everything the pipeline knows about one pixel.

    Reports the raw values, the background model's values, the difference, and
    - crucially - *why* the pixel passed or failed the current threshold. A
    probe that reports numbers without the decision leaves the user to redo the
    comparison in their head.
    """
    session = get_session()
    try:
        ctx = session.context(index, intermediates=True)
    except StageError as error:
        raise _fail(error) from error

    height, width = ctx.bgr.shape[:2]
    if not (0 <= x < width and 0 <= y < height):
        raise HTTPException(400, f"({x}, {y}) is outside a {width}x{height} frame")

    bgr = ctx.bgr[y, x]
    hsv = ctx.hsv[y, x] if ctx.hsv is not None else np.zeros(3)
    background = ctx.background_bgr[y, x] if ctx.background_bgr is not None else np.zeros(3)
    difference = float(ctx.difference[y, x]) if ctx.difference is not None else 0.0
    threshold = ctx.threshold_level

    world = None
    if session.calibration is not None and session.calibration.is_valid:
        if session.calibration.is_on_road_plane(float(x), float(y)):
            from kerbside.geometry.homography import apply_h_single

            wx, wy = apply_h_single(session.calibration.h_img_to_world, float(x), float(y))
            world = [round(wx, 3), round(wy, 3)]

    return {
        "x": x,
        "y": y,
        "rgb": [int(bgr[2]), int(bgr[1]), int(bgr[0])],
        "hsv": [round(float(hsv[0]), 1), round(float(hsv[1]), 4), round(float(hsv[2]), 4)],
        "background_rgb": [int(background[2]), int(background[1]), int(background[0])],
        "difference": round(difference, 3),
        "threshold": round(threshold, 3),
        "is_foreground": bool(ctx.foreground[y, x]) if ctx.foreground is not None else False,
        "is_shadow": bool(ctx.shadow[y, x]) if ctx.shadow is not None else False,
        "world_m": world,
        "verdict": (
            f"difference {difference:.1f} "
            f"{'exceeds' if difference > threshold else 'is below'} the threshold "
            f"{threshold:.1f}"
        ),
    }


# ==========================================================================
# Stream and telemetry
# ==========================================================================


def _trail_map(session: Session, upto: int, length: int = 24) -> dict[int, list[tuple[float, float]]]:
    trails: dict[int, list[tuple[float, float]]] = {}
    for index in range(max(0, upto - length), upto + 1):
        for snapshot in session.telemetry(index).get("tracks", []):
            point = snapshot.get("ground_point_px")
            if point:
                trails.setdefault(snapshot["track_id"], []).append((point[0], point[1]))
    return trails


@router.get("/api/frame/{index}.jpg")
def frame_jpeg(index: int, request: Request) -> StreamingResponse:
    """One annotated frame, for stepping and scrubbing."""
    session = get_session()
    options = OverlayOptions.from_query(dict(request.query_params))
    try:
        # Boxes, ground points and trails come from the frozen track snapshots,
        # which survive an analysis. Only the mask layers read the arrays that
        # analysis releases, so only they are worth a replay.
        ctx = session.context(index, intermediates=options.needs_intermediates)
    except StageError as error:
        raise _fail(error) from error
    frame = draw_overlay(
        ctx.bgr, ctx, options, session.calibration, _trail_map(session, index)
    )
    return StreamingResponse(iter([encode_jpeg(frame)]), media_type="image/jpeg")


@router.get("/api/stream.mjpg")
def stream(request: Request, fps: float = Query(25.0, gt=0.1, le=120.0)) -> StreamingResponse:
    """The annotated video as MJPEG.

    Paced to the requested rate rather than sent as fast as possible: an
    unpaced stream would deliver a twelve-second clip in under a second and the
    console would show a blur.
    """
    session = get_session()
    options = OverlayOptions.from_query(dict(request.query_params))

    def frames() -> Any:
        interval = 1.0 / fps
        while True:
            if not session.frames:
                time.sleep(0.2)
                continue
            index = session.cursor
            try:
                # Never with intermediates: a replay here would stall the stream
                # for seconds on a single frame. A mask layer during playback
                # shows only for frames whose detail is still retained.
                ctx = session.context(index)
            except StageError:
                time.sleep(0.2)
                continue
            annotated = draw_overlay(
                ctx.bgr, ctx, options, session.calibration, _trail_map(session, index)
            )
            payload = encode_jpeg(annotated)
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\n\r\n"
                + payload
                + b"\r\n"
            )
            if session.playing:
                session.cursor = (index + 1) % max(1, len(session.frames))
            time.sleep(interval / max(session.play_speed, 0.05))

    return StreamingResponse(
        frames(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


class TransportRequest(BaseModel):
    playing: bool | None = None
    cursor: int | None = None
    speed: float | None = Field(None, ge=0.05, le=8.0)
    step: int | None = None


@router.post("/api/transport")
def transport(request: TransportRequest) -> dict[str, Any]:
    session = get_session()
    if request.playing is not None:
        session.playing = request.playing
    if request.speed is not None:
        session.play_speed = request.speed
    if request.cursor is not None and session.frames:
        session.cursor = int(np.clip(request.cursor, 0, len(session.frames) - 1))
    if request.step is not None and session.frames:
        session.cursor = int(
            np.clip(session.cursor + request.step, 0, len(session.frames) - 1)
        )
    return {
        "playing": session.playing,
        "cursor": session.cursor,
        "speed": session.play_speed,
        "frames": len(session.frames),
    }


@router.websocket("/ws/telemetry")
async def telemetry_socket(socket: WebSocket) -> None:
    """Per-frame telemetry, paced and never queued.

    If the client falls behind, the newest frame is sent and the missed ones are
    dropped rather than buffered. A console showing stale telemetry beside a
    live video is worse than one that skips: the numbers would not describe the
    frame on screen.
    """
    await socket.accept()
    session = get_session()
    last_sent = -1
    try:
        while True:
            index = session.cursor
            if index != last_sent and session.analysed:
                await socket.send_text(json.dumps(session.telemetry(index)))
                last_sent = index
            await asyncio.sleep(0.03)
    except (WebSocketDisconnect, RuntimeError):
        return


# ==========================================================================
# Calibration
# ==========================================================================


class CalibrationRequest(BaseModel):
    points_px: list[list[float]] = Field(..., min_length=4, max_length=4)
    rect_width_m: float = Field(..., gt=0.1, le=100.0)
    rect_length_m: float = Field(..., gt=0.1, le=400.0)
    origin_x_m: float = 0.0
    origin_y_m: float = 20.0
    lane_count: int = Field(4, ge=1, le=12)
    lane_width_m: float = Field(3.5, gt=1.0, le=6.0)


@router.get("/api/calibration")
def calibration() -> dict[str, Any]:
    session = get_session()
    if session.calibration is None:
        return {"calibrated": False, "suggestions": _suggestions()}
    report = assess_calibration(session.calibration)
    return {
        "calibrated": session.calibration.is_valid,
        "homography": np.asarray(session.calibration.h_world_to_img).tolist(),
        "rmse_px": report.reprojection_rmse_px,
        "verdict": report.verdict,
        "exactly_determined": report.exactly_determined,
        "warnings": list(report.warnings),
        "camera": {
            "focal_px": report.camera.focal_px,
            "height_m": report.camera.height_m,
            "tilt_deg": report.camera.tilt_deg,
            "plausible": report.camera.is_plausible,
            "reasons": list(report.camera.reasons),
        },
        "horizon_row_px": report.horizon_row_px,
        "covered_y_range_m": report.covered_y_range_m,
        "lanes": [lane.to_dict() for lane in session.calibration.lanes],
        "suggestions": _suggestions(),
    }


def _suggestions() -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "nominal_m": s.nominal_m,
            "range_m": list(s.plausible_range_m),
            "caution": s.caution,
        }
        for s in reference_distance_suggestions()
    ]


@router.post("/api/calibration")
def calibrate(request: CalibrationRequest) -> dict[str, Any]:
    session = get_session()
    if not session.frames:
        raise HTTPException(400, "load a source before calibrating")
    try:
        calib = calibrate_from_rectangle(
            np.asarray(request.points_px, dtype=np.float64),
            request.rect_width_m,
            request.rect_length_m,
            session.state.width_px,
            session.state.height_px,
            origin_m=(request.origin_x_m, request.origin_y_m),
            source="clicked",
        )
        half = request.lane_count * request.lane_width_m / 2.0
        bands = [
            (-half + i * request.lane_width_m, -half + (i + 1) * request.lane_width_m)
            for i in range(request.lane_count)
        ]
        from kerbside.core.types import Direction

        directions = [
            Direction.TOWARD if band[1] <= 0 else Direction.AWAY for band in bands
        ]
        calib = calib.with_lanes(
            lanes_from_world_bands(
                calib, bands, (request.origin_y_m, request.origin_y_m + 45.0),
                directions=directions,
            )
        )
    except (ValueError, StageError) as error:
        raise _fail(error) from error

    session.set_calibration(calib)
    return calibration()


class MeasureRequest(BaseModel):
    a_px: list[float] = Field(..., min_length=2, max_length=2)
    b_px: list[float] = Field(..., min_length=2, max_length=2)


@router.post("/api/calibration/measure")
def measure(request: MeasureRequest) -> dict[str, Any]:
    """The validation tool: the only check that catches a wrong scale."""
    session = get_session()
    if session.calibration is None:
        raise HTTPException(400, "calibrate before measuring")
    distance = measure_distance_m(
        session.calibration, (request.a_px[0], request.a_px[1]), (request.b_px[0], request.b_px[1])
    )
    if distance is None:
        return {
            "distance_m": None,
            "note": "at least one point is above the horizon, where the road plane "
            "is not imaged and no distance can be computed",
        }
    return {
        "distance_m": round(distance, 3),
        "note": "compare this against a tape measure. A calibration can be "
        "perfectly self-consistent and still wrong by a constant factor, and "
        "this is the only check that catches it.",
    }


@router.get("/api/calibration/grid")
def grid(step_m: float = 1.0, x_min: float = -8.0, x_max: float = 8.0,
         y_min: float = 15.0, y_max: float = 70.0) -> dict[str, Any]:
    session = get_session()
    if session.calibration is None:
        raise HTTPException(400, "calibrate first")
    try:
        lines = metre_grid_segments_px(
            session.calibration, (x_min, x_max), (y_min, y_max), step_m=step_m
        )
    except ValueError as error:
        raise _fail(error) from error
    return {"polylines": lines, "step_m": step_m}


@router.get("/api/calibration/birdseye/{index}.jpg")
def birdseye(index: int, px_per_m: float = 8.0) -> StreamingResponse:
    session = get_session()
    if session.calibration is None:
        raise HTTPException(400, "calibrate first")
    matrix, width, height = birdseye_homography(
        session.calibration, (-8.0, 8.0), (15.0, 70.0), px_per_m=px_per_m
    )
    frame = session.frame(index)
    warped = cv2.warpPerspective(frame, matrix, (width, height))
    return StreamingResponse(iter([encode_jpeg(warped)]), media_type="image/jpeg")


class SaveCalibrationRequest(BaseModel):
    key: str


@router.post("/api/calibration/save")
def save_calibration(request: SaveCalibrationRequest) -> dict[str, Any]:
    session = get_session()
    if session.calibration is None:
        raise HTTPException(400, "nothing to save")
    path = calibration_path_for(request.key)
    session.calibration.save(path)
    return {"saved": str(path)}


@router.post("/api/calibration/load")
def load_calibration(request: SaveCalibrationRequest) -> dict[str, Any]:
    path = calibration_path_for(request.key)
    if not Path(path).exists():
        raise HTTPException(404, f"no saved calibration named {request.key!r}")
    session = get_session()
    session.set_calibration(Calibration.load(path))
    return calibration()


# ==========================================================================
# Analytics and ablation
# ==========================================================================


@router.get("/api/analytics")
def analytics() -> dict[str, Any]:
    from kerbside.eval.analytics import build_analytics

    session = get_session()
    try:
        session.analyse()
    except StageError as error:
        raise _fail(error) from error
    return build_analytics(session)


@router.get("/api/analytics/csv")
def analytics_csv() -> StreamingResponse:
    from kerbside.eval.analytics import analytics_csv as build_csv

    session = get_session()
    session.analyse()
    payload = build_csv(session)
    return StreamingResponse(
        iter([payload]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="kerbside-vehicles.csv"'},
    )


@router.get("/api/ablations")
def list_ablations() -> dict[str, Any]:
    return {
        "ablations": [
            {"key": key, "label": preset.label, "question": preset.question}
            for key, preset in ABLATIONS.items()
        ]
    }


@router.post("/api/ablations/{key}")
def run_one_ablation(key: str, duration_s: float = 6.0) -> dict[str, Any]:
    try:
        return run_ablation(key, duration_s=duration_s)
    except (KeyError, ValueError, StageError) as error:
        raise _fail(error) from error


@router.get("/api/report")
def report() -> JSONResponse:
    from kerbside.eval.report import build_report

    session = get_session()
    session.analyse()
    return JSONResponse(build_report(session))


@router.get("/api/report.html")
def report_html() -> HTMLResponse:
    """The same report as one self-contained HTML file.

    Served as a download so the console's Export button hands the user a file
    they can email or print, rather than a JSON blob they would have to render
    themselves.
    """
    from kerbside.eval.report import build_report, render_report_html

    session = get_session()
    session.analyse()
    return HTMLResponse(
        render_report_html(build_report(session)),
        headers={"Content-Disposition": 'attachment; filename="kerbside-report.html"'},
    )
