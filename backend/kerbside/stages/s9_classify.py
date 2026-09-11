"""S9 - classification and colour.

**Class by footprint, never by appearance.** The vehicle's extent on the road
plane is recovered through the homography and bucketed into two-wheeler, car,
LCV or truck-or-bus using bands written down as named constants in
:data:`kerbside.core.types.CLASS_BANDS_M`. That is a deliberate constraint and
also the honest one: the bands are reproduced verbatim in the report, so anyone
can disagree with a particular boundary and see exactly what changing it would
do. A classifier whose decision rule cannot be inspected cannot be argued with.

The estimate is accumulated over a track's whole life rather than taken from one
frame. A single frame's footprint depends on how the mask happened to come out
that frame - a shadow at the edge, a fragment reattached - whereas a median over
thirty frames is a property of the vehicle. The class is also weighted towards
frames where the vehicle was *closest*, because that is where the pixels-per-metre
scale is highest and a metre is measured most precisely.

**Colour from an HSV histogram of body pixels**, with the darkest and brightest
deciles discarded. Windows are much darker than the paint and specular
highlights much brighter, and both have hues unrelated to the body; trimming the
extremes of the value distribution removes them without needing to know where on
the vehicle they are. Achromatic bodies - white, silver, grey, black, which
between them are most of the traffic - have no usable hue at all and are named
by value instead.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import (
    CLASS_BANDS_M,
    Footprint,
    StageResult,
    Track,
    TrackState,
    VehicleClass,
)
from kerbside.operators.colour import name_colour
from kerbside.stages.base import FrameContext, Stage

__all__ = ["ClassifyStage", "classify_footprint", "ClassVote"]


def classify_footprint(footprint: Footprint) -> tuple[VehicleClass, str]:
    """Bucket a real-world footprint into a class, with the reason.

    The bands are checked in order and the first whose length *and* width both
    admit the footprint wins. Aspect ratio is the tie-breaker: a blob 5 m long
    and 0.9 m wide has a car's length and a motorcycle's width, and an aspect
    ratio that says it is neither - most likely a shadow sliver or two
    two-wheelers in line.

    Returns the class and a short human-readable justification, because a
    classification the interface cannot explain is one a user cannot check.
    """
    length, width = footprint.length_m, footprint.width_m
    if length <= 0 or width <= 0:
        return (VehicleClass.UNKNOWN, "no measurable footprint")

    for band in CLASS_BANDS_M:
        if not (band.length_m[0] <= length < band.length_m[1]):
            continue
        if not (band.width_m[0] <= width < band.width_m[1]):
            continue
        if footprint.aspect > band.aspect_max:
            return (
                VehicleClass.UNKNOWN,
                f"{length:.1f} x {width:.1f} m fits {band.vehicle_class.value} by "
                f"size but its aspect ratio {footprint.aspect:.1f} exceeds "
                f"{band.aspect_max:.1f}",
            )
        return (
            band.vehicle_class,
            f"{length:.1f} x {width:.1f} m falls in the {band.vehicle_class.value} "
            f"band ({band.length_m[0]:.1f}-{band.length_m[1]:.1f} m long, "
            f"{band.width_m[0]:.2f}-{band.width_m[1]:.2f} m wide)",
        )

    # Nothing matched on both dimensions. Fall back to length alone, which is
    # the more reliable of the two - width is a single box dimension at the
    # ground row, length comes from area and is less sensitive to one bad edge.
    for band in CLASS_BANDS_M:
        if band.length_m[0] <= length < band.length_m[1]:
            return (
                band.vehicle_class,
                f"{length:.1f} m long matches {band.vehicle_class.value}, though "
                f"{width:.2f} m is outside that band's width range",
            )
    return (
        VehicleClass.UNKNOWN,
        f"{length:.1f} x {width:.1f} m is outside every documented class band",
    )


@dataclass(slots=True)
class ClassVote:
    """Accumulated class and colour evidence for one track."""

    lengths_m: list[float] = field(default_factory=list)
    widths_m: list[float] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)
    colours: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    hsv: list[tuple[float, float, float]] = field(default_factory=list)

    def footprint(self) -> Footprint | None:
        """Weighted median footprint over the track's life.

        Median rather than mean: one frame where the mask picked up a shadow
        doubles the length, and a mean carries that forever where a median
        ignores it.
        """
        if not self.lengths_m:
            return None
        return Footprint(
            length_m=_weighted_median(self.lengths_m, self.weights),
            width_m=_weighted_median(self.widths_m, self.weights),
        )

    def colour(self) -> tuple[str, float]:
        if not self.colours:
            return ("unknown", 0.0)
        total = sum(self.colours.values())
        name = max(self.colours, key=lambda k: self.colours[k])
        return (name, self.colours[name] / total if total else 0.0)

    def median_hsv(self) -> tuple[float, float, float]:
        if not self.hsv:
            return (0.0, 0.0, 0.0)
        arr = np.asarray(self.hsv, dtype=np.float64)
        return (
            float(np.median(arr[:, 0])),
            float(np.median(arr[:, 1])),
            float(np.median(arr[:, 2])),
        )


def _weighted_median(values: list[float], weights: list[float]) -> float:
    if not values:
        return 0.0
    order = np.argsort(np.asarray(values))
    v = np.asarray(values)[order]
    w = np.asarray(weights if weights else [1.0] * len(values))[order]
    cumulative = np.cumsum(w)
    if cumulative[-1] <= 0:
        return float(np.median(v))
    return float(v[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])


class ClassifyStage(Stage):
    """S9: name each track's class from its footprint and its colour from HSV."""

    stage_id = "s9_classify"
    name = "Classify"
    units = (1, 7)
    rationale = (
        "Buckets each track by the real-world footprint recovered through the "
        "homography, using documented length and width bands rather than "
        "appearance - so the rule can be inspected and disagreed with. Colour "
        "comes from an HSV histogram of body pixels with the darkest and "
        "brightest deciles discarded, because those are windows and highlights "
        "rather than paint, and achromatic bodies are named by value since they "
        "have no usable hue."
    )
    param_specs = (
        ParamSpec(
            "min_samples", "Samples before naming", "int", 3, 1, 60, 1, unit="frames",
            description=(
                "Frames of evidence before a class is reported. One frame's "
                "footprint is a property of that frame's mask, not of the vehicle."
            ),
        ),
        ParamSpec(
            "near_weighting", "Weight nearby frames", "bool", True,
            description=(
                "Weight each frame's footprint by the local pixels-per-metre "
                "scale, so measurements taken where a metre spans more pixels "
                "count for more."
            ),
        ),
        ParamSpec("colour_bins", "Hue bins", "int", 36, 6, 180, 6),
        ParamSpec(
            "colour_trim", "Value trim", "float", 0.10, 0.0, 0.4, 0.01,
            description="Fraction discarded from each end of the value distribution.",
        ),
        ParamSpec(
            "achromatic_saturation", "Achromatic below", "float", 0.18, 0.0, 1.0, 0.01,
        ),
        ParamSpec(
            "colour_min_area_px", "Minimum body pixels", "float", 200.0, 20.0, 20000.0, 10.0,
            description=(
                "Below this there are too few body pixels for a hue histogram to "
                "have a meaningful peak."
            ),
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._votes: dict[int, ClassVote] = {}
        self._reasons: dict[int, str] = {}
        super().__init__(**params)

    def reset(self) -> None:
        self._votes = {}
        self._reasons = {}

    @property
    def reasons(self) -> dict[int, str]:
        """Why each track was given its class, for the selected-vehicle card."""
        return dict(self._reasons)

    def vote_for(self, track_id: int) -> ClassVote | None:
        return self._votes.get(track_id)

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            named = 0
            for track in ctx.tracks:
                if track.state is TrackState.TENTATIVE:
                    continue
                vote = self._votes.setdefault(track.track_id, ClassVote())
                self._accumulate(vote, track, ctx)
                if len(vote.lengths_m) < int(self.param("min_samples")):
                    continue

                footprint = vote.footprint()
                if footprint is not None:
                    vehicle_class, reason = classify_footprint(footprint)
                    track.vehicle_class = vehicle_class
                    track.footprint_m = footprint
                    self._reasons[track.track_id] = reason
                    named += 1

                colour_name, _confidence = vote.colour()
                if colour_name != "unknown":
                    track.colour_name = colour_name
                    track.colour_hsv = vote.median_hsv()

            ctx.track_snapshots = tuple(
                track.snapshot(ctx.frame_index, ctx.time_s) for track in ctx.tracks
            )

            counts: dict[str, int] = defaultdict(int)
            for track in ctx.tracks:
                if track.state is not TrackState.TENTATIVE:
                    counts[track.vehicle_class.value] += 1
            result.metrics = {"tracks_named": float(named)}
            result.extras["class_counts"] = dict(counts)
        return result

    def _accumulate(self, vote: ClassVote, track: Track, ctx: FrameContext) -> None:
        if track.footprint_m is not None and track.footprint_m.length_m > 0:
            weight = 1.0
            if self.param("near_weighting") and track.ground_point_px is not None:
                # A metre is measured most precisely where it spans the most
                # pixels, which is nearest the camera.
                weight = max(1.0, float(track.ground_point_px.y) / 100.0)
            vote.lengths_m.append(track.footprint_m.length_m)
            vote.widths_m.append(track.footprint_m.width_m)
            vote.weights.append(weight)

        if track.bbox_px is None or ctx.foreground_no_shadow is None:
            return
        box = track.bbox_px.clipped(ctx.bgr.shape[1], ctx.bgr.shape[0])
        if box.area_px < float(self.param("colour_min_area_px")):
            return
        y0, y1 = int(box.y_px), int(box.y2_px)
        x0, x1 = int(box.x_px), int(box.x2_px)
        if y1 - y0 < 4 or x1 - x0 < 4:
            return

        # Body pixels only: the shadow-suppressed mask inside the box. Naming a
        # colour from the bounding box would average in whatever road happens to
        # sit in the box's corners.
        patch = ctx.bgr[y0:y1, x0:x1]
        mask = ctx.foreground_no_shadow[y0:y1, x0:x1]
        if int(mask.sum()) < float(self.param("colour_min_area_px")):
            return

        name, hsv, confidence = name_colour(
            patch,
            mask,
            bins=int(self.param("colour_bins")),
            trim_decile=float(self.param("colour_trim")),
            achromatic_saturation=float(self.param("achromatic_saturation")),
        )
        if name != "unknown":
            vote.colours[name] += max(confidence, 0.05)
            vote.hsv.append(hsv)
