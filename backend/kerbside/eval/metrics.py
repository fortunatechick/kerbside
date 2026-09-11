"""Scoring the pipeline against the renderer's exact ground truth.

Two principles run through this module.

**Score against what was detectable.** A vehicle three pixels wide at the
horizon, or one hidden almost entirely behind a lorry, is not a failure of the
pipeline in any useful sense - counting it as a miss makes every number worse
without making any of them more informative. So every metric takes an explicit
visibility gate and *reports what it used*, rather than quietly applying one. A
figure quoted without its gate is not a figure.

The gate says what the pipeline is **required** to find. It does not say what it
is **allowed** to find, and conflating the two is a mistake that costs the
pipeline dearly for being right. A vehicle at 110 m is real; if the gate stops
at 90 m and the blob for that vehicle is scored against the gated vehicle list,
it matches nothing and is recorded as a false positive - so the pipeline is
penalised precisely for detecting a vehicle that is there. Worse, the effect is
strongest exactly where position estimates are least certain: a one-pixel error
in a ground point at 100 m is about eight metres of world position, so which
side of the gate a distant blob falls on is close to arbitrary.

Matching therefore runs against *every* visible vehicle, and the gate is applied
only afterwards: a required vehicle nobody found is a miss, and a blob that
matches no vehicle at all - gated or not - is a false positive.

**Report the failure modes, not just the score.** A count that is 3% low
because two vehicles merged is a different problem from one that is 3% low
because a distant vehicle was never detected, and the fix is in a different
stage. :class:`DetectionMetrics` separates merges, splits, misses and false
positives for exactly that reason.

Matching is by ground point in *metres on the road plane* wherever a calibration
is available, not in pixels. A twenty-pixel gate is a metre in the near field
and eight metres at the horizon, so a pixel gate silently applies a different
standard to each end of the road.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kerbside.core.types import Blob, Calibration, GroundTruth, GTFrame, GTVehicleFrame
from kerbside.geometry.homography import apply_h_single, invert_h

__all__ = [
    "VisibilityGate",
    "FrameMatch",
    "DetectionMetrics",
    "CountAccuracy",
    "visible_vehicles",
    "match_frame",
    "score_detection",
    "count_accuracy",
    "false_merge_rate",
    "SpeedComparison",
    "match_tracks_to_vehicles",
    "score_speed",
    "score_lane_and_class",
]


@dataclass(frozen=True, slots=True)
class VisibilityGate:
    """What counts as a vehicle the pipeline was supposed to find.

    Quoted alongside every metric that uses it. The defaults are deliberately
    permissive - a vehicle covering 150 pixels is small but genuinely there -
    and the ablation sweeps them, because how a score moves with the gate says
    as much as the score does.
    """

    min_visible_px: int = 150
    max_occlusion: float = 0.75
    min_y_m: float = 0.0
    max_y_m: float = 1e9

    def admits(self, vehicle: GTVehicleFrame) -> bool:
        return (
            vehicle.visible
            and vehicle.visible_area_px >= self.min_visible_px
            and vehicle.occlusion_frac <= self.max_occlusion
            and self.min_y_m <= vehicle.world_pos_m[1] <= self.max_y_m
        )

    def describe(self) -> str:
        parts = [
            f"at least {self.min_visible_px} visible px",
            f"at most {self.max_occlusion:.0%} occluded",
        ]
        if self.min_y_m > 0 or self.max_y_m < 1e9:
            parts.append(f"between {self.min_y_m:.0f} and {self.max_y_m:.0f} m")
        return ", ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_visible_px": self.min_visible_px,
            "max_occlusion": self.max_occlusion,
            "min_y_m": self.min_y_m,
            "max_y_m": self.max_y_m,
            "description": self.describe(),
        }


def visible_vehicles(frame: GTFrame, gate: VisibilityGate) -> tuple[GTVehicleFrame, ...]:
    """The vehicles in a frame the pipeline was expected to detect."""
    return tuple(v for v in frame.vehicles if gate.admits(v))


@dataclass(frozen=True, slots=True)
class FrameMatch:
    """How one frame's blobs lined up against its ground-truth vehicles."""

    frame_index: int
    matched: tuple[tuple[int, int], ...]
    missed: tuple[int, ...]
    spurious: tuple[int, ...]
    merged: tuple[tuple[int, tuple[int, ...]], ...]
    split: tuple[tuple[int, tuple[int, ...]], ...]
    expected: int
    detected: int
    distances_m: tuple[float, ...] = ()

    @property
    def merge_count(self) -> int:
        """Vehicles lost to a merge: a blob covering n vehicles loses n-1."""
        return sum(len(ids) - 1 for _blob, ids in self.merged)

    @property
    def split_count(self) -> int:
        return sum(len(blobs) - 1 for _vehicle, blobs in self.split)


def match_frame(
    blobs: Sequence[Blob],
    frame: GTFrame,
    gate: VisibilityGate,
    calibration: Calibration | None = None,
    max_distance_m: float = 3.5,
    max_distance_px: float = 60.0,
    present_min_px: int = 40,
) -> FrameMatch:
    """Associate detected blobs with ground-truth vehicles for one frame.

    A blob claims a vehicle when the vehicle's true ground point lies inside the
    blob's bounding box, or within the gate distance of the blob's ground point.
    Containment is checked first because it is the relation that actually
    matters for a merge: a blob that spans two vehicles contains both of their
    ground points, and no distance threshold expresses that as cleanly.

    Blobs are matched against every vehicle with at least ``present_min_px``
    visible, not only against the gated ones. See the module docstring: the gate
    decides what must be found, and a blob that found something real is not a
    false positive just because the gate did not ask for it.
    """
    required = visible_vehicles(frame, gate)
    required_ids = {v.vehicle_id for v in required}
    present = tuple(
        v for v in frame.vehicles if v.visible and v.visible_area_px >= present_min_px
    )
    h_inv = invert_h(calibration.h_world_to_img) if calibration is not None else None

    claims: dict[int, list[int]] = {i: [] for i in range(len(blobs))}
    vehicle_claims: dict[int, list[int]] = {v.vehicle_id: [] for v in present}
    distances: list[float] = []

    for vehicle in present:
        vx, vy = vehicle.ground_point_px
        vx0, vy0, vw, vh = vehicle.bbox_px
        vx1, vy1 = vx0 + vw, vy0 + vh
        for index, blob in enumerate(blobs):
            box = blob.bbox_px
            # A blob claims a vehicle if it holds the vehicle's ground point, or
            # if it lies substantially inside the vehicle's own box.
            #
            # The second clause is what makes fragmentation legible. When a
            # vehicle splits into a roof blob and a bumper blob, only the bumper
            # contains the ground point; the roof contains nothing and its own
            # ground point sits metres away on the road plane, because it is the
            # bottom edge of the roof rather than of the vehicle. Matching on the
            # ground point alone reports that roof as a *false positive* - an
            # object detected where there is none - when what actually happened
            # is that one object was reported twice. The two call for completely
            # different fixes, so the matcher has to tell them apart.
            inside = (
                box.x_px - 2 <= vx <= box.x2_px + 2 and box.y_px - 2 <= vy <= box.y2_px + 2
            )
            if not inside:
                ox = max(0.0, min(box.x2_px, vx1) - max(box.x_px, vx0))
                oy = max(0.0, min(box.y2_px, vy1) - max(box.y_px, vy0))
                overlap = ox * oy
                inside = box.area_px > 0 and overlap / box.area_px >= 0.5
            if inside:
                claims[index].append(vehicle.vehicle_id)
                vehicle_claims[vehicle.vehicle_id].append(index)
                continue

            if h_inv is not None and blob.world_pos_m is not None:
                wx, wy = apply_h_single(h_inv, vx, vy)
                if np.isfinite(wx) and np.isfinite(wy):
                    d = float(np.hypot(blob.world_pos_m.x - wx, blob.world_pos_m.y - wy))
                    if d <= max_distance_m:
                        claims[index].append(vehicle.vehicle_id)
                        vehicle_claims[vehicle.vehicle_id].append(index)
                        distances.append(d)
                    continue
            d_px = float(np.hypot(blob.ground_point_px.x - vx, blob.ground_point_px.y - vy))
            if d_px <= max_distance_px:
                claims[index].append(vehicle.vehicle_id)
                vehicle_claims[vehicle.vehicle_id].append(index)

    matched: list[tuple[int, int]] = []
    merged: list[tuple[int, tuple[int, ...]]] = []
    split: list[tuple[int, tuple[int, ...]]] = []

    for index, ids in claims.items():
        if len(ids) == 1 and ids[0] in required_ids:
            matched.append((index, ids[0]))
        elif len(ids) > 1 and required_ids.intersection(ids):
            merged.append((index, tuple(sorted(ids))))
    for vehicle_id, indices in vehicle_claims.items():
        if len(indices) > 1 and vehicle_id in required_ids:
            split.append((vehicle_id, tuple(sorted(indices))))

    claimed_vehicles = {vid for ids in claims.values() for vid in ids}
    missed = tuple(
        sorted(v.vehicle_id for v in required if v.vehicle_id not in claimed_vehicles)
    )
    spurious = tuple(sorted(i for i, ids in claims.items() if not ids))

    # The count comparison is between what the gate required and what the
    # pipeline reported *for those vehicles*. Blobs matching a real but
    # ungated vehicle are neither credited nor charged - they are outside the
    # question being asked.
    detected_for_required = len(matched) + len(merged) + len(spurious)
    for vehicle_id, indices in vehicle_claims.items():
        if vehicle_id in required_ids and len(indices) > 1:
            detected_for_required += len(indices) - 1

    return FrameMatch(
        frame_index=frame.frame_index,
        matched=tuple(matched),
        missed=missed,
        spurious=spurious,
        merged=tuple(merged),
        split=tuple(split),
        expected=len(required),
        detected=detected_for_required,
        distances_m=tuple(distances),
    )


@dataclass(slots=True)
class DetectionMetrics:
    """Per-frame detection quality, with the failure modes kept apart."""

    gate: VisibilityGate
    frames: int = 0
    frames_scored: int = 0
    expected_total: int = 0
    detected_total: int = 0
    matched_total: int = 0
    missed_total: int = 0
    spurious_total: int = 0
    merge_events: int = 0
    merged_vehicles: int = 0
    split_events: int = 0
    exact_count_frames: int = 0
    per_frame_error: list[int] = field(default_factory=list)
    matches: list[FrameMatch] = field(default_factory=list)

    def add(self, match: FrameMatch) -> None:
        self.frames += 1
        self.frames_scored += 1
        self.expected_total += match.expected
        self.detected_total += match.detected
        self.matched_total += len(match.matched)
        self.missed_total += len(match.missed)
        self.spurious_total += len(match.spurious)
        self.merge_events += len(match.merged)
        self.merged_vehicles += match.merge_count
        self.split_events += len(match.split)
        error = match.detected - match.expected
        self.per_frame_error.append(error)
        if error == 0:
            self.exact_count_frames += 1
        self.matches.append(match)

    @property
    def frame_count_agreement(self) -> float:
        """Fraction of frames where the blob count equalled the vehicle count.

        This is the Phase 3 gate. It is a strict measure - one extra blob from a
        speck of noise fails the frame just as surely as a missed vehicle - and
        that strictness is the point: it is sensitive to exactly the mask defects
        S4 and S5 exist to remove.
        """
        return self.exact_count_frames / self.frames_scored if self.frames_scored else 0.0

    @property
    def recall(self) -> float:
        """Fraction of required vehicles that at least one blob found.

        Counted per *vehicle*, not per blob-vehicle pair. A vehicle reported as
        two fragments has been found once, not twice - crediting it twice would
        let a pipeline that shatters everything report a recall above one, which
        is how this was noticed.
        """
        return (
            (self.expected_total - self.missed_total) / self.expected_total
            if self.expected_total
            else 0.0
        )

    @property
    def precision(self) -> float:
        denominator = self.matched_total + self.spurious_total
        return self.matched_total / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def false_merge_rate(self) -> float:
        """Vehicles lost to merges, as a fraction of those expected.

        The headline number for the shadow ablation, because a shadow bridge
        does not produce a missed detection - it produces a blob in the right
        place that happens to contain two vehicles.
        """
        return self.merged_vehicles / self.expected_total if self.expected_total else 0.0

    @property
    def mean_absolute_count_error(self) -> float:
        return float(np.mean(np.abs(self.per_frame_error))) if self.per_frame_error else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "gate": self.gate.to_dict(),
            "frames_scored": self.frames_scored,
            "expected_total": self.expected_total,
            "detected_total": self.detected_total,
            "frame_count_agreement": self.frame_count_agreement,
            "recall": self.recall,
            "precision": self.precision,
            "f1": self.f1,
            "missed_total": self.missed_total,
            "spurious_total": self.spurious_total,
            "merge_events": self.merge_events,
            "merged_vehicles": self.merged_vehicles,
            "false_merge_rate": self.false_merge_rate,
            "split_events": self.split_events,
            "mean_absolute_count_error": self.mean_absolute_count_error,
        }


def score_detection(
    per_frame_blobs: Sequence[Sequence[Blob]],
    ground_truth: GroundTruth,
    gate: VisibilityGate | None = None,
    calibration: Calibration | None = None,
    skip_frames: int = 0,
) -> DetectionMetrics:
    """Score a clip's blobs against ground truth.

    ``skip_frames`` excludes the background model's warm-up, during which the
    pipeline has not yet seen enough of the scene to know what the road looks
    like. Including those frames would measure the warm-up rather than the
    stages, and the number of frames skipped is reported so the exclusion is
    visible rather than assumed.
    """
    gate = gate or VisibilityGate()
    metrics = DetectionMetrics(gate=gate)
    frames = ground_truth.frames
    for index, blobs in enumerate(per_frame_blobs):
        if index < skip_frames or index >= len(frames):
            continue
        metrics.add(match_frame(blobs, frames[index], gate, calibration))
    return metrics


@dataclass(frozen=True, slots=True)
class CountAccuracy:
    """Clip-level counting accuracy: unique vehicles, not per-frame blobs."""

    counted: int
    expected: int
    per_lane_counted: dict[int, int]
    per_lane_expected: dict[int, int]

    @property
    def accuracy(self) -> float:
        """One minus the relative error, floored at zero.

        Expressed as an error rather than a ratio because over-counting and
        under-counting are equally wrong: counting 12 where 10 passed is 80%
        accurate here, not 120%.
        """
        if self.expected == 0:
            return 1.0 if self.counted == 0 else 0.0
        return max(0.0, 1.0 - abs(self.counted - self.expected) / self.expected)

    @property
    def error(self) -> int:
        return self.counted - self.expected

    def summary(self) -> dict[str, Any]:
        return {
            "counted": self.counted,
            "expected": self.expected,
            "error": self.error,
            "accuracy": self.accuracy,
            "per_lane_counted": dict(sorted(self.per_lane_counted.items())),
            "per_lane_expected": dict(sorted(self.per_lane_expected.items())),
        }


def count_accuracy(
    counted: int,
    ground_truth: GroundTruth,
    gate: VisibilityGate | None = None,
    per_lane_counted: dict[int, int] | None = None,
) -> CountAccuracy:
    """Compare a clip-level count against the ground-truth vehicle count."""
    gate = gate or VisibilityGate()
    seen: dict[int, set[int]] = {}
    for frame in ground_truth.frames:
        for vehicle in visible_vehicles(frame, gate):
            seen.setdefault(vehicle.lane_id, set()).add(vehicle.vehicle_id)
    expected_by_lane = {k: len(v) for k, v in seen.items()}
    expected = len({vid for ids in seen.values() for vid in ids})
    return CountAccuracy(
        counted=counted,
        expected=expected,
        per_lane_counted=per_lane_counted or {},
        per_lane_expected=expected_by_lane,
    )


def false_merge_rate(metrics: DetectionMetrics) -> float:
    """Convenience accessor, so the ablation reads as what it measures."""
    return metrics.false_merge_rate


# --------------------------------------------------------------------------
# Phase 4: tracking, counting, speed, lane and class
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpeedComparison:
    """Both speed routes scored against ground truth, side by side.

    The comparison is the point. Route A measures where a silhouette's edge is
    and Route B where texture went, so they fail in independent ways; reporting
    only the better one would hide which conditions each suits.
    """

    samples: int
    track_mae_kmph: float
    track_bias_kmph: float
    blockmatch_mae_kmph: float
    blockmatch_bias_kmph: float
    blockmatch_samples: int
    agreement_kmph: float

    def summary(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "track_mae_kmph": self.track_mae_kmph,
            "track_bias_kmph": self.track_bias_kmph,
            "blockmatch_samples": self.blockmatch_samples,
            "blockmatch_mae_kmph": self.blockmatch_mae_kmph,
            "blockmatch_bias_kmph": self.blockmatch_bias_kmph,
            "route_agreement_kmph": self.agreement_kmph,
        }


def match_tracks_to_vehicles(
    per_frame_tracks: Sequence[Sequence[Any]],
    ground_truth: GroundTruth,
    max_distance_m: float = 4.0,
    skip_frames: int = 0,
) -> dict[int, int]:
    """Assign each track the ground-truth vehicle it overlapped most.

    By majority over the track's whole life rather than by its first frame: a
    track that starts on a shadow and settles onto a vehicle belongs to the
    vehicle. The comparison is against the vehicle's *near extremity*, because
    that is what a ground point measures - see the Phase 1 findings.
    """
    tally: dict[int, dict[int, int]] = {}
    for index, tracks in enumerate(per_frame_tracks):
        if index < skip_frames or index >= len(ground_truth.frames):
            continue
        vehicles = ground_truth.frames[index].vehicles
        for track in tracks:
            position = getattr(track, "world_pos_m", None)
            if position is None:
                continue
            px, py = (position if isinstance(position, tuple) else (position.x, position.y))
            best_id, best_d = None, max_distance_m
            for vehicle in vehicles:
                if not vehicle.visible:
                    continue
                near_y = vehicle.world_pos_m[1] - vehicle.footprint_m[0] / 2.0
                d = float(np.hypot(px - vehicle.world_pos_m[0], py - near_y))
                if d < best_d:
                    best_id, best_d = vehicle.vehicle_id, d
            if best_id is not None:
                tally.setdefault(track.track_id, {})
                tally[track.track_id][best_id] = tally[track.track_id].get(best_id, 0) + 1

    return {
        track_id: max(votes, key=lambda k: votes[k])
        for track_id, votes in tally.items()
        if votes
    }


def score_speed(
    per_frame_tracks: Sequence[Sequence[Any]],
    per_frame_estimates: Sequence[dict[int, Any]],
    ground_truth: GroundTruth,
    assignment: dict[int, int],
    min_speed_kmph: float = 5.0,
    skip_frames: int = 0,
) -> SpeedComparison:
    """Score both speed routes against the recorded per-frame speeds.

    Near-stationary vehicles are excluded: a relative error is meaningless
    there and an absolute one measures the tracker's velocity filter settling
    rather than either route.
    """
    track_errors: list[float] = []
    block_errors: list[float] = []
    agreements: list[float] = []

    for index, tracks in enumerate(per_frame_tracks):
        if index < skip_frames or index >= len(ground_truth.frames):
            continue
        truth_by_id = {v.vehicle_id: v for v in ground_truth.frames[index].vehicles}
        estimates = per_frame_estimates[index] if index < len(per_frame_estimates) else {}
        for track in tracks:
            vehicle_id = assignment.get(track.track_id)
            if vehicle_id is None:
                continue
            vehicle = truth_by_id.get(vehicle_id)
            if vehicle is None or vehicle.speed_kmph < min_speed_kmph:
                continue
            estimate = estimates.get(track.track_id)
            if estimate is None:
                continue
            if estimate.track_kmph > 0:
                track_errors.append(estimate.track_kmph - vehicle.speed_kmph)
            if estimate.blockmatch_usable and estimate.blockmatch_kmph > 0:
                block_errors.append(estimate.blockmatch_kmph - vehicle.speed_kmph)
                if estimate.track_kmph > 0:
                    agreements.append(estimate.agreement_kmph)

    def mae(values: list[float]) -> float:
        return float(np.mean(np.abs(values))) if values else float("nan")

    def bias(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return SpeedComparison(
        samples=len(track_errors),
        track_mae_kmph=mae(track_errors),
        track_bias_kmph=bias(track_errors),
        blockmatch_samples=len(block_errors),
        blockmatch_mae_kmph=mae(block_errors),
        blockmatch_bias_kmph=bias(block_errors),
        agreement_kmph=float(np.median(agreements)) if agreements else float("nan"),
    )


def score_lane_and_class(
    per_frame_tracks: Sequence[Sequence[Any]],
    ground_truth: GroundTruth,
    assignment: dict[int, int],
    skip_frames: int = 0,
) -> dict[str, Any]:
    """Lane-assignment and class accuracy over matched tracks.

    Lane is scored per frame - a track can drift across a boundary and the
    question is how often the assignment is right. Class and colour are scored
    once per track, because they are properties of the vehicle and S9
    accumulates evidence over the whole track.
    """
    lane_correct = lane_total = 0
    class_by_track: dict[int, tuple[str, str]] = {}
    colour_by_track: dict[int, tuple[str, str]] = {}

    for index, tracks in enumerate(per_frame_tracks):
        if index < skip_frames or index >= len(ground_truth.frames):
            continue
        truth_by_id = {v.vehicle_id: v for v in ground_truth.frames[index].vehicles}
        for track in tracks:
            vehicle_id = assignment.get(track.track_id)
            vehicle = truth_by_id.get(vehicle_id) if vehicle_id is not None else None
            if vehicle is None:
                continue
            if track.lane_id is not None:
                lane_total += 1
                lane_correct += int(track.lane_id == vehicle.lane_id)
            class_by_track[track.track_id] = (
                track.vehicle_class.value,
                vehicle.vehicle_class.value,
            )
            colour_by_track[track.track_id] = (track.colour_name, vehicle.colour_name)

    classed = [(p, t) for p, t in class_by_track.values() if p != "unknown"]
    coloured = [(p, t) for p, t in colour_by_track.values() if p != "unknown"]
    return {
        "lane_accuracy": lane_correct / lane_total if lane_total else 0.0,
        "lane_samples": lane_total,
        "class_accuracy": (
            sum(p == t for p, t in classed) / len(classed) if classed else 0.0
        ),
        "class_samples": len(classed),
        "class_named_fraction": (
            len(classed) / len(class_by_track) if class_by_track else 0.0
        ),
        "colour_accuracy": (
            sum(p == t for p, t in coloured) / len(coloured) if coloured else 0.0
        ),
        "colour_samples": len(coloured),
        "class_confusion": _confusion(classed),
    }


def _confusion(pairs: list[tuple[str, str]]) -> dict[str, dict[str, int]]:
    """Which classes are confused for which - the informative part of a score."""
    out: dict[str, dict[str, int]] = {}
    for predicted, truth in pairs:
        out.setdefault(truth, {})
        out[truth][predicted] = out[truth].get(predicted, 0) + 1
    return out
