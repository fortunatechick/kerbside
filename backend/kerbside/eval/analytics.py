"""The numbers a traffic engineer would actually read.

Counts and speeds are the raw output; these are the statistics the profession
uses to interpret them, and a couple of them are chosen deliberately over more
obvious alternatives.

**The 85th percentile, not the mean.** Traffic engineering uses it because it
describes the speed most drivers consider reasonable and is far less sensitive
to a handful of very slow vehicles - a queue behind a turning lorry drags a mean
badly and barely moves the 85th percentile.

**Headway, not just flow.** The time gap between successive vehicles says
something flow cannot: two roads carrying the same vehicles per hour behave
completely differently if one delivers them evenly and the other in platoons.

**The fundamental diagram.** Flow against density is the classic plot of the
field, and it is the one that shows whether a road is free-flowing or
congested - the two states sit on opposite branches of the same curve at the
same flow.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np

from kerbside.core.types import TrackState

if TYPE_CHECKING:
    from kerbside.api.session import Session

__all__ = ["build_analytics", "analytics_csv", "percentile_85", "headways_s"]


def percentile_85(speeds: list[float]) -> float:
    """The standard traffic-engineering speed statistic."""
    if not speeds:
        return 0.0
    return float(np.percentile(np.asarray(speeds, dtype=np.float64), 85))


def headways_s(times_s: list[float]) -> list[float]:
    """Time gaps between successive vehicles crossing a line."""
    if len(times_s) < 2:
        return []
    ordered = sorted(times_s)
    return [round(b - a, 3) for a, b in zip(ordered[:-1], ordered[1:], strict=True)]


def _vehicle_rows(session: Session) -> list[dict[str, Any]]:
    """One row per counted vehicle, which is what the table and CSV show."""
    speeds_by_track: dict[int, list[float]] = {}
    class_by_track: dict[int, str] = {}
    colour_by_track: dict[int, str] = {}
    footprint_by_track: dict[int, list[float] | None] = {}

    for index in range(len(session.frames)):
        for snapshot in session.telemetry(index).get("tracks", []):
            track_id = snapshot["track_id"]
            speed = snapshot.get("speed_kmph_track", 0.0)
            if speed > 0.5:
                speeds_by_track.setdefault(track_id, []).append(speed)
            if snapshot.get("vehicle_class") != "unknown":
                class_by_track[track_id] = snapshot["vehicle_class"]
            if snapshot.get("colour_name") != "unknown":
                colour_by_track[track_id] = snapshot["colour_name"]
            if snapshot.get("footprint_m"):
                footprint_by_track[track_id] = snapshot["footprint_m"]

    rows: list[dict[str, Any]] = []
    for event in session.pipeline.track.events:
        speeds = speeds_by_track.get(event.track_id, [event.speed_kmph])
        footprint = footprint_by_track.get(event.track_id)
        rows.append(
            {
                "track_id": event.track_id,
                "lane_id": event.lane_id,
                "direction": event.direction.value,
                "time_s": round(event.time_s, 3),
                "frame_index": event.frame_index,
                "speed_kmph": round(float(np.median(speeds)), 2),
                "speed_at_line_kmph": round(event.speed_kmph, 2),
                "vehicle_class": class_by_track.get(event.track_id, "unknown"),
                "colour": colour_by_track.get(event.track_id, "unknown"),
                "length_m": round(footprint[0], 2) if footprint else None,
                "width_m": round(footprint[1], 2) if footprint else None,
            }
        )
    return rows


def build_analytics(session: Session, interval_s: float = 5.0) -> dict[str, Any]:
    """Everything the analytics view renders."""
    rows = _vehicle_rows(session)
    speeds = [r["speed_kmph"] for r in rows if r["speed_kmph"] > 0]
    duration_s = len(session.frames) / max(session.state.measured_fps, 1e-6)

    limit = 60.0
    if session.calibration is not None and session.calibration.lanes:
        limit = session.calibration.lanes[0].speed_limit_kmph

    # Per-interval counts and flow.
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(row["time_s"] // interval_s), []).append(row)
    intervals = []
    for bucket in range(int(duration_s // interval_s) + 1):
        entries = buckets.get(bucket, [])
        bucket_speeds = [e["speed_kmph"] for e in entries if e["speed_kmph"] > 0]
        intervals.append(
            {
                "t_s": round(bucket * interval_s, 2),
                "count": len(entries),
                "flow_veh_per_hour": round(len(entries) * 3600.0 / interval_s, 1),
                "mean_speed_kmph": round(float(np.mean(bucket_speeds)), 2)
                if bucket_speeds
                else 0.0,
            }
        )

    # The fundamental diagram: density is flow over speed, which is the
    # relationship the diagram exists to show rather than an extra measurement.
    fundamental = []
    for entry in intervals:
        if entry["mean_speed_kmph"] > 1.0 and entry["count"] > 0:
            fundamental.append(
                {
                    "flow_veh_per_hour": entry["flow_veh_per_hour"],
                    "density_veh_per_km": round(
                        entry["flow_veh_per_hour"] / entry["mean_speed_kmph"], 2
                    ),
                    "mean_speed_kmph": entry["mean_speed_kmph"],
                }
            )

    histogram_counts, histogram_edges = (
        np.histogram(speeds, bins=12, range=(0, max(80.0, max(speeds) if speeds else 80.0)))
        if speeds
        else (np.zeros(12, dtype=int), np.linspace(0, 80, 13))
    )

    flagged = [r for r in rows if r["speed_kmph"] > limit]

    by_class: dict[str, int] = {}
    by_lane: dict[str, int] = {}
    for row in rows:
        by_class[row["vehicle_class"]] = by_class.get(row["vehicle_class"], 0) + 1
        by_lane[str(row["lane_id"])] = by_lane.get(str(row["lane_id"]), 0) + 1

    return {
        "total_count": len(rows),
        "duration_s": round(duration_s, 2),
        "counts_by_lane": by_lane,
        "counts_by_class": by_class,
        "speed": {
            "mean_kmph": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "median_kmph": round(float(np.median(speeds)), 2) if speeds else 0.0,
            "p85_kmph": round(percentile_85(speeds), 2),
            "limit_kmph": limit,
        },
        "speed_histogram": {
            "counts": [int(c) for c in histogram_counts],
            "edges": [round(float(e), 1) for e in histogram_edges],
        },
        "intervals": intervals,
        "fundamental_diagram": fundamental,
        "headways_s": headways_s([r["time_s"] for r in rows]),
        "vehicles": rows,
        "flagged": flagged,
        "disclaimer": (
            "Vehicles above the configured limit are flagged for review only. "
            "This system is not a speed enforcement device: a legally admissible "
            "measurement requires type approval and calibration certification "
            "that this project does not have."
        ),
    }


def analytics_csv(session: Session) -> str:
    """One row per counted vehicle, for export."""
    rows = _vehicle_rows(session)
    buffer = io.StringIO()
    columns = [
        "track_id", "lane_id", "direction", "time_s", "frame_index",
        "speed_kmph", "speed_at_line_kmph", "vehicle_class", "colour",
        "length_m", "width_m",
    ]
    buffer.write(",".join(columns) + "\n")
    for row in rows:
        buffer.write(
            ",".join("" if row.get(c) is None else str(row.get(c)) for c in columns) + "\n"
        )
    return buffer.getvalue()


def live_summary(session: Session, index: int) -> dict[str, Any]:
    """The live panel beside the video: counts and speeds as of one frame."""
    telemetry = session.telemetry(index)
    speeds = [
        t["speed_kmph_track"]
        for t in telemetry.get("tracks", [])
        if t.get("state") != TrackState.TENTATIVE.value and t.get("speed_kmph_track", 0) > 0.5
    ]
    return {
        "frame_index": index,
        "total_count": telemetry.get("total_count", 0),
        "counts_by_lane": telemetry.get("counts", {}),
        "mean_speed_kmph": round(float(np.mean(speeds)), 1) if speeds else 0.0,
        "p85_kmph": round(percentile_85(speeds), 1),
        "active_tracks": len(telemetry.get("tracks", [])),
    }
