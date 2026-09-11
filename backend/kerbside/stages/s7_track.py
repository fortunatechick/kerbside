"""S7 - tracking and counting.

Up to here every frame has been considered on its own. Tracking is where the
pipeline starts to have a memory, and that memory is what makes counting robust:
a vehicle that fragments into two blobs on one frame out of twenty is still one
track, and a speck of noise that appears for two frames never becomes one.

Three decisions carry the stage.

**Association is gated in metres, not pixels.** A twenty-pixel gate is about a
metre in the near field and eight metres near the horizon, so a pixel gate
quietly applies a different standard to each end of the road - loose enough to
swap two distant vehicles, tight enough to drop a near one that moved quickly.
Converting the blob's ground point to road-plane coordinates first makes one
number mean the same thing everywhere.

**The search is centred on a prediction, not on the last position.** A vehicle
at 60 km/h moves 0.67 m per frame at 25 fps, so a gate large enough to catch it
without prediction is large enough to catch its neighbour too. A constant
velocity predictor costs nothing and shrinks the gate to the size of the actual
uncertainty.

**Counting is a directed crossing of a line, with a debounce.** A track hovering
on the line - which happens whenever traffic is queuing across it - would
otherwise count once per frame. The direction test also stops a reversing or
mis-associated track from counting a second time on its way back.

The distance metric itself is a parameter, because Unit 1 offers three and the
choice is worth measuring rather than assuming: Euclidean, city-block and
chessboard differ by up to 41% on a diagonal displacement, and a gate calibrated
for one is a different gate under another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import (
    Blob,
    Direction,
    DistanceMetric,
    Lane,
    Point2,
    StageResult,
    Track,
    TrackState,
)
from kerbside.geometry.homography import apply_h_single, invert_h
from kerbside.stages.base import FrameContext, Stage

__all__ = ["TrackStage", "CountEvent", "world_distance"]


def world_distance(
    a: tuple[float, float], b: tuple[float, float], metric: DistanceMetric
) -> float:
    """Distance between two road-plane points under one of the Unit 1 metrics.

    ``euclidean``
        the straight-line distance, and the only one that is rotation
        invariant.
    ``city_block``
        ``|dx| + |dy]``. Never smaller than Euclidean, and up to 41% larger on a
        perfect diagonal, so a gate set for one is effectively tighter under the
        other.
    ``chessboard``
        ``max(|dx|, |dy|)``. Never larger than Euclidean, and up to 29% smaller.

    On a road the displacement between consecutive frames is almost entirely
    along the road, where all three agree closely; the difference shows up in
    the association gate for lane changes and for the lateral jitter of a
    fragmenting blob.
    """
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    if metric is DistanceMetric.CITY_BLOCK:
        return dx + dy
    if metric is DistanceMetric.CHESSBOARD:
        return max(dx, dy)
    return float(np.hypot(dx, dy))


@dataclass(frozen=True, slots=True)
class CountEvent:
    """A vehicle counted crossing a lane's line."""

    track_id: int
    lane_id: int
    frame_index: int
    time_s: float
    direction: Direction
    speed_kmph: float
    world_y_m: float


@dataclass(slots=True)
class _LaneLine:
    """A lane's counting line, expressed where the crossing test happens."""

    lane_id: int
    y_m: float
    direction: Direction
    speed_limit_kmph: float


class TrackStage(Stage):
    """S7: associate blobs into tracks, and count line crossings."""

    stage_id = "s7_track"
    name = "Track"
    units = (1,)
    rationale = (
        "Associates each frame's blobs with existing tracks by nearest "
        "neighbour, gated by a maximum distance measured in metres on the road "
        "plane rather than in pixels - a pixel gate means a metre in the near "
        "field and eight metres at the horizon. A constant-velocity prediction "
        "centres the search, and a track is counted when its ground point "
        "crosses its lane's line in the expected direction, debounced so a "
        "vehicle queuing on the line cannot count twice."
    )
    param_specs = (
        ParamSpec(
            "max_association_m", "Association gate", "float", 4.0, 0.5, 30.0, 0.25, unit="m",
            description=(
                "Largest road-plane distance between a prediction and a blob "
                "that still counts as the same vehicle."
            ),
        ),
        ParamSpec(
            "metric", "Distance metric", "enum", "euclidean",
            choices=("euclidean", "city_block", "chessboard"),
            description="The Unit 1 distance measures, exposed so they can be compared.",
        ),
        ParamSpec(
            "confirm_hits", "Confirm after", "int", 3, 1, 20, 1, unit="frames",
            description=(
                "Detections before a track is believed. This is what stops a "
                "two-frame speck of noise ever being counted."
            ),
        ),
        ParamSpec(
            "coast_frames", "Coast for", "int", 6, 0, 60, 1, unit="frames",
            description=(
                "Frames a confirmed track survives without a detection. Long "
                "enough to ride out an occlusion, short enough not to drift into "
                "the next vehicle."
            ),
        ),
        ParamSpec(
            "velocity_smoothing", "Velocity smoothing", "float", 0.4, 0.05, 1.0, 0.05,
            description=(
                "Blend of the newly measured velocity into the running estimate. "
                "1.0 uses the latest frame alone, which is very noisy at 25 fps."
            ),
        ),
        ParamSpec(
            "count_debounce_m", "Counting debounce", "float", 1.5, 0.0, 20.0, 0.25, unit="m",
            description=(
                "How far past the line a track must travel before it could count "
                "again. Stops a vehicle stopped on the line counting every frame."
            ),
        ),
        ParamSpec(
            "min_count_speed_kmph", "Minimum speed to count", "float", 2.0, 0.0, 40.0, 0.5,
            description=(
                "Below this a track is not moving through the line, it is "
                "sitting on it."
            ),
        ),
        ParamSpec(
            "max_speed_kmph", "Implausible speed", "float", 200.0, 20.0, 400.0, 5.0,
            description=(
                "Velocity estimates above this are rejected as an association "
                "error rather than recorded as a reading."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self._counts: dict[int, int] = {}
        self._events: list[CountEvent] = []
        self._lines: tuple[_LaneLine, ...] = ()
        self._lines_key: int | None = None
        self._crossed: dict[int, float] = {}
        super().__init__(**params)

    def reset(self) -> None:
        self._tracks = {}
        self._next_id = 1
        self._counts = {}
        self._events = []
        self._lines = ()
        self._lines_key = None
        self._crossed = {}

    # -- accessors ---------------------------------------------------------

    @property
    def tracks(self) -> tuple[Track, ...]:
        return tuple(self._tracks.values())

    @property
    def confirmed_tracks(self) -> tuple[Track, ...]:
        return tuple(
            t for t in self._tracks.values() if t.state is not TrackState.TENTATIVE
        )

    @property
    def events(self) -> tuple[CountEvent, ...]:
        return tuple(self._events)

    @property
    def total_count(self) -> int:
        return len(self._events)

    def counts_by_lane(self) -> dict[int, int]:
        return dict(sorted(self._counts.items()))

    # -- lane lines --------------------------------------------------------

    def _lane_lines(self, ctx: FrameContext) -> tuple[_LaneLine, ...]:
        """Counting lines as road-plane distances, built once per calibration.

        The line is stored in the calibration as a pair of image points, but the
        crossing test belongs in world coordinates: in the image a line at a
        fixed distance is not horizontal unless the camera has no yaw, and a
        track's progress along the road is exactly the quantity the test is
        about.
        """
        calibration = ctx.calibration
        if calibration is None or not calibration.lanes:
            return ()
        key = id(calibration)
        if self._lines_key == key and self._lines:
            return self._lines

        h_inv = invert_h(calibration.h_world_to_img)
        lines: list[_LaneLine] = []
        for lane in calibration.lanes:
            (x0, y0), (x1, y1) = lane.counting_line_px
            wx, wy = apply_h_single(h_inv, (x0 + x1) / 2.0, (y0 + y1) / 2.0)
            if not np.isfinite(wy):
                continue
            lines.append(
                _LaneLine(
                    lane_id=lane.lane_id,
                    y_m=float(wy),
                    direction=lane.direction,
                    speed_limit_kmph=lane.speed_limit_kmph,
                )
            )
        self._lines = tuple(lines)
        self._lines_key = key
        return self._lines

    # -- association -------------------------------------------------------

    def _predict(self, track: Track, dt_s: float) -> tuple[float, float]:
        if track.world_pos_m is None:
            return (0.0, 0.0)
        return (
            track.world_pos_m.x + track.velocity_mps[0] * dt_s,
            track.world_pos_m.y + track.velocity_mps[1] * dt_s,
        )

    def _associate(
        self, blobs: tuple[Blob, ...], dt_s: float
    ) -> tuple[dict[int, int], set[int]]:
        """Greedy nearest-neighbour association, closest pair first.

        Greedy rather than optimal (Hungarian) on purpose: with a gate this
        tight the assignment is almost always unambiguous, and taking the
        globally closest pair first gives the same answer as the optimal
        assignment in every case where the answer is obvious - while being far
        easier to reason about when it is not. Where they would differ, the
        situation is one where the gate should have rejected both.
        """
        metric = DistanceMetric(self.param("metric"))
        gate = float(self.param("max_association_m"))

        usable = [
            (index, blob)
            for index, blob in enumerate(blobs)
            if blob.world_pos_m is not None
        ]
        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            if track.world_pos_m is None:
                continue
            predicted = self._predict(track, dt_s)
            for index, blob in usable:
                assert blob.world_pos_m is not None
                distance = world_distance(
                    predicted, (blob.world_pos_m.x, blob.world_pos_m.y), metric
                )
                if distance <= gate:
                    pairs.append((distance, track_id, index))

        pairs.sort()
        assigned: dict[int, int] = {}
        used_blobs: set[int] = set()
        for _distance, track_id, index in pairs:
            if track_id in assigned or index in used_blobs:
                continue
            assigned[track_id] = index
            used_blobs.add(index)
        return assigned, used_blobs

    # -- execution ---------------------------------------------------------

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            dt_s = ctx.dt_s if ctx.dt_s > 0 else 1.0 / 25.0
            lines = self._lane_lines(ctx)
            assigned, used = self._associate(ctx.blobs, dt_s)

            for track_id, track in list(self._tracks.items()):
                track.age_frames += 1
                index = assigned.get(track_id)
                if index is None:
                    track.misses += 1
                    if track.state is TrackState.CONFIRMED:
                        track.state = TrackState.COASTING
                    if track.misses > int(self.param("coast_frames")):
                        track.state = TrackState.TERMINATED
                        del self._tracks[track_id]
                        continue
                    # Coast on the last velocity, so an occluded vehicle is
                    # still roughly in the right place when it reappears.
                    if track.world_pos_m is not None:
                        predicted = self._predict(track, dt_s)
                        track.world_pos_m = Point2(*predicted)
                    continue

                blob = ctx.blobs[index]
                self._update(track, blob, ctx, dt_s)

            for index, blob in enumerate(ctx.blobs):
                if index in used or blob.world_pos_m is None:
                    continue
                self._spawn(blob, ctx)

            counted_now = self._count(ctx, lines)
            ctx.tracks = self.tracks
            ctx.counted_this_frame = counted_now
            result.metrics = {
                "tracks": float(len(self._tracks)),
                "confirmed": float(len(self.confirmed_tracks)),
                "associated": float(len(assigned)),
                "spawned": float(len(ctx.blobs) - len(used)),
                "counted_this_frame": float(counted_now),
                "total_count": float(self.total_count),
            }
            result.extras["counts_by_lane"] = self.counts_by_lane()
            if not lines:
                result.extras["hint"] = (
                    "no lane counting lines are defined, so tracks are "
                    "maintained but nothing is counted; define lanes in the "
                    "calibration view"
                )
        return result

    def _update(self, track: Track, blob: Blob, ctx: FrameContext, dt_s: float) -> None:
        assert blob.world_pos_m is not None
        previous = track.world_pos_m
        track.hits += 1
        track.misses = 0
        track.last_frame = ctx.frame_index
        track.bbox_px = blob.bbox_px
        track.ground_point_px = blob.ground_point_px
        track.world_pos_m = blob.world_pos_m
        track.footprint_m = blob.footprint_m
        if blob.lane_id is not None:
            track.lane_id = blob.lane_id
        # A track becomes confirmed either by accumulating enough hits, or by
        # being re-associated after coasting - a track that survived an
        # occlusion has already proved itself. Written out rather than left to
        # `and`/`or` precedence, because this is a state machine and a reader
        # should not have to recall the binding order to check it.
        confirmed = (
            track.state is TrackState.TENTATIVE
            and track.hits >= int(self.param("confirm_hits"))
        ) or track.state is TrackState.COASTING
        if confirmed:
            track.state = TrackState.CONFIRMED

        if previous is not None and dt_s > 0:
            measured = (
                (blob.world_pos_m.x - previous.x) / dt_s,
                (blob.world_pos_m.y - previous.y) / dt_s,
            )
            speed_kmph = float(np.hypot(*measured)) * 3.6
            if speed_kmph <= float(self.param("max_speed_kmph")):
                # A blended update rather than the raw frame-to-frame velocity.
                # At 25 fps a vehicle moves under a metre a frame, so a
                # half-pixel wobble in the ground point is several km/h of noise
                # on the instantaneous estimate.
                alpha = float(self.param("velocity_smoothing"))
                track.velocity_mps = (
                    (1 - alpha) * track.velocity_mps[0] + alpha * measured[0],
                    (1 - alpha) * track.velocity_mps[1] + alpha * measured[1],
                )
                track.direction = (
                    Direction.AWAY if track.velocity_mps[1] > 0 else Direction.TOWARD
                )

        track.history_px.append(blob.ground_point_px.as_tuple())
        track.history_world_m.append((blob.world_pos_m.x, blob.world_pos_m.y))
        track.history_time_s.append(ctx.time_s)
        if len(track.history_px) > 120:
            del track.history_px[0]
            del track.history_world_m[0]
            del track.history_time_s[0]

    def _spawn(self, blob: Blob, ctx: FrameContext) -> None:
        assert blob.world_pos_m is not None
        track = Track(
            track_id=self._next_id,
            state=TrackState.TENTATIVE,
            hits=1,
            first_frame=ctx.frame_index,
            last_frame=ctx.frame_index,
            lane_id=blob.lane_id,
            bbox_px=blob.bbox_px,
            ground_point_px=blob.ground_point_px,
            world_pos_m=blob.world_pos_m,
            footprint_m=blob.footprint_m,
        )
        track.history_px.append(blob.ground_point_px.as_tuple())
        track.history_world_m.append((blob.world_pos_m.x, blob.world_pos_m.y))
        track.history_time_s.append(ctx.time_s)
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _count(self, ctx: FrameContext, lines: tuple[_LaneLine, ...]) -> int:
        """Count confirmed tracks crossing their lane's line in its direction."""
        if not lines:
            return 0
        by_lane = {line.lane_id: line for line in lines}
        debounce = float(self.param("count_debounce_m"))
        min_speed = float(self.param("min_count_speed_kmph"))
        counted = 0

        for track in self._tracks.values():
            if track.state is TrackState.TENTATIVE or track.counted:
                continue
            if track.lane_id is None or track.world_pos_m is None:
                continue
            line = by_lane.get(track.lane_id)
            if line is None or len(track.history_world_m) < 2:
                continue

            previous_y = track.history_world_m[-2][1]
            current_y = track.world_pos_m.y
            speed_kmph = float(np.hypot(*track.velocity_mps)) * 3.6
            if speed_kmph < min_speed:
                continue

            if line.direction is Direction.AWAY:
                crossed = previous_y <= line.y_m < current_y
                past = current_y - line.y_m
            else:
                crossed = previous_y >= line.y_m > current_y
                past = line.y_m - current_y
            if not crossed or past < 0:
                continue
            if self._crossed.get(track.track_id, -1e9) > line.y_m - debounce:
                continue

            track.counted = True
            track.counted_frame = ctx.frame_index
            self._crossed[track.track_id] = line.y_m
            self._counts[line.lane_id] = self._counts.get(line.lane_id, 0) + 1
            self._events.append(
                CountEvent(
                    track_id=track.track_id,
                    lane_id=line.lane_id,
                    frame_index=ctx.frame_index,
                    time_s=ctx.time_s,
                    direction=line.direction,
                    speed_kmph=speed_kmph,
                    world_y_m=current_y,
                )
            )
            counted += 1
        return counted


def lane_for_point(point: Point2, lanes: tuple[Lane, ...]) -> int | None:
    """Lane containing an image point, or None."""
    import cv2

    for lane in lanes:
        polygon = np.asarray(lane.polygon_px, dtype=np.float32)
        if polygon.shape[0] >= 3 and cv2.pointPolygonTest(
            polygon, (float(point.x), float(point.y)), False
        ) >= 0:
            return lane.lane_id
    return None
