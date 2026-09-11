"""Drawing the annotated frame.

The overlay is where this project is judged, so it is treated as a designed
surface rather than as debug output. The colours are the interface tokens, the
labels are monospace on a translucent chip so they stay legible over any frame,
and every layer can be turned off independently.

One element matters more than the rest: the **ground point**, drawn as a filled
dot at the bottom-centre of each box. That is the point that actually lies on
the road plane and the point every speed is computed from, and it has to be
visible for the overlay to be evidence of anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kerbside.core.types import Calibration, TrackSnapshot, TrackState, UInt8Array
from kerbside.stages.base import FrameContext

__all__ = ["OverlayOptions", "draw_overlay"]

# Interface tokens, in BGR for OpenCV.
ACCENT = (214, 167, 63)
FLOW_FREE = (113, 191, 47)
FLOW_SLOW = (49, 185, 232)
FLOW_STOP = (77, 72, 229)
SHADOW_HL = (250, 139, 167)
MOTION_HL = (98, 115, 249)
TEXT = (244, 238, 232)
MUTED = (127, 113, 99)


@dataclass(slots=True)
class OverlayOptions:
    """Which layers to draw. Persisted per user by the console."""

    boxes: bool = True
    ground_points: bool = True
    trails: bool = True
    labels: bool = True
    lanes: bool = True
    counting_lines: bool = True
    mask: bool = False
    shadow: bool = False
    blobs: bool = False
    trail_frames: int = 12
    speed_limit_kmph: float = 60.0

    @property
    def needs_intermediates(self) -> bool:
        """Whether these layers read arrays that a completed analysis releases.

        The boxes, labels, trails and lane geometry all come from the frozen
        track snapshots and the calibration, which are kept for every frame. The
        three mask layers read the per-frame arrays, which are not.
        """
        return self.mask or self.shadow or self.blobs

    @staticmethod
    def from_query(values: dict[str, str]) -> OverlayOptions:
        def flag(name: str, default: bool) -> bool:
            raw = values.get(name)
            return default if raw is None else raw.lower() in ("1", "true", "yes", "on")

        options = OverlayOptions(
            boxes=flag("boxes", True),
            ground_points=flag("ground_points", True),
            trails=flag("trails", True),
            labels=flag("labels", True),
            lanes=flag("lanes", True),
            counting_lines=flag("counting_lines", True),
            mask=flag("mask", False),
            shadow=flag("shadow", False),
            blobs=flag("blobs", False),
        )
        try:
            options.trail_frames = int(values.get("trail_frames", 12))
            options.speed_limit_kmph = float(values.get("speed_limit_kmph", 60.0))
        except ValueError:
            pass
        return options


def _chip(
    canvas: UInt8Array, text: str, x: int, y: int, colour: tuple[int, int, int]
) -> None:
    """A label on a translucent chip, so it stays readable over any frame.

    Drawing text straight onto video is unreadable roughly half the time - white
    on a white van, black on asphalt in shadow. The chip costs a blend and
    removes the problem entirely.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.4, 1
    (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x0, y0 = x, max(0, y - h - 6)
    x1, y1 = x + w + 10, y + baseline - 2
    if x1 > canvas.shape[1]:
        x0, x1 = canvas.shape[1] - (w + 10), canvas.shape[1]
    patch = canvas[max(0, y0) : y1, max(0, x0) : x1]
    if patch.size:
        # 70% opacity, per the overlay rules.
        patch[:] = (patch.astype(np.float32) * 0.3 + np.float32(18)).astype(np.uint8)
    cv2.putText(
        canvas, text, (max(0, x0) + 5, y - 4), font, scale, colour, thickness, cv2.LINE_AA
    )


def draw_overlay(
    frame: UInt8Array,
    ctx: FrameContext,
    options: OverlayOptions,
    calibration: Calibration | None = None,
    trails: dict[int, list[tuple[float, float]]] | None = None,
) -> UInt8Array:
    """Compose the annotated frame."""
    canvas = np.ascontiguousarray(frame.copy())

    if options.mask and ctx.foreground_no_shadow is not None:
        tint = np.zeros_like(canvas)
        tint[ctx.foreground_no_shadow] = ACCENT
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.35, 0)

    if options.shadow and ctx.shadow is not None:
        tint = np.zeros_like(canvas)
        tint[ctx.shadow] = SHADOW_HL
        canvas = cv2.addWeighted(canvas, 1.0, tint, 0.45, 0)

    if options.lanes and calibration is not None and calibration.lanes:
        # 8% fill, so lane polygons never fight the video.
        layer = canvas.copy()
        for lane in calibration.lanes:
            polygon = np.round(np.asarray(lane.polygon_px)).astype(np.int32)
            if polygon.shape[0] >= 3:
                cv2.fillPoly(layer, [polygon], (60, 48, 36))
        canvas = cv2.addWeighted(canvas, 0.92, layer, 0.08, 0)

    if options.counting_lines and calibration is not None:
        for lane in calibration.lanes:
            (x0, y0), (x1, y1) = lane.counting_line_px
            _dashed_line(canvas, (x0, y0), (x1, y1), ACCENT)

    if options.blobs:
        for blob in ctx.blobs:
            box = blob.bbox_px
            cv2.rectangle(
                canvas,
                (int(box.x_px), int(box.y_px)),
                (int(box.x2_px), int(box.y2_px)),
                MUTED, 1,
            )

    for snapshot in ctx.track_snapshots:
        if snapshot.state is TrackState.TENTATIVE:
            continue
        _draw_track(canvas, snapshot, options, trails)

    return canvas


def _draw_track(
    canvas: UInt8Array,
    track: TrackSnapshot,
    options: OverlayOptions,
    trails: dict[int, list[tuple[float, float]]] | None,
) -> None:
    if track.bbox_px is None:
        return
    x, y, w, h = (int(round(v)) for v in track.bbox_px)
    speed = track.speed_kmph_track

    colour = ACCENT
    if track.counted:
        colour = FLOW_FREE
    if speed > options.speed_limit_kmph:
        colour = FLOW_STOP

    if options.trails and trails is not None:
        history = trails.get(track.track_id, [])[-options.trail_frames :]
        for i in range(1, len(history)):
            # Fade the trail towards its tail so the direction of travel reads
            # without needing an arrowhead.
            alpha = i / max(1, len(history))
            faded = tuple(int(c * (0.25 + 0.75 * alpha)) for c in colour)
            cv2.line(
                canvas,
                (int(history[i - 1][0]), int(history[i - 1][1])),
                (int(history[i][0]), int(history[i][1])),
                faded, 1, cv2.LINE_AA,
            )

    if options.boxes:
        cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)

    if options.ground_points and track.ground_point_px is not None:
        gx, gy = (int(round(v)) for v in track.ground_point_px)
        cv2.circle(canvas, (gx, gy), 4, colour, -1, cv2.LINE_AA)
        cv2.circle(canvas, (gx, gy), 4, (20, 16, 12), 1, cv2.LINE_AA)

    if options.labels:
        parts = [f"#{track.track_id}"]
        if speed > 0.5:
            parts.append(f"{speed:.0f} km/h")
        if track.vehicle_class.value != "unknown":
            parts.append(track.vehicle_class.value.replace("_", " "))
        if track.colour_name != "unknown":
            parts.append(track.colour_name)
        _chip(canvas, "  ".join(parts), x, max(14, y - 2), TEXT)


def _dashed_line(
    canvas: UInt8Array,
    start: tuple[float, float],
    end: tuple[float, float],
    colour: tuple[int, int, int],
    dash: int = 12,
) -> None:
    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(end, dtype=np.float64)
    length = float(np.hypot(*(p1 - p0)))
    if length < 1:
        return
    steps = max(2, int(length / dash))
    for i in range(0, steps, 2):
        a = p0 + (p1 - p0) * (i / steps)
        b = p0 + (p1 - p0) * (min(i + 1, steps) / steps)
        cv2.line(
            canvas, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colour, 2, cv2.LINE_AA
        )


def encode_jpeg(frame: UInt8Array, quality: int = 82) -> bytes:
    """Encode a frame for the MJPEG stream.

    Quality 82 rather than 95: the difference is invisible at streaming sizes
    and roughly halves the bytes, which matters because the stream and the
    telemetry share a connection budget.
    """
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("failed to encode a frame as JPEG")
    return bytes(buffer)
