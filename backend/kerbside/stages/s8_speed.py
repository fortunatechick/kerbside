"""S8 - speed, by two independent routes.

**Route A, track-based.** Convert the ground point to road-plane coordinates for
two frames separated by a known interval and divide the distance by the time.
Simple, and it inherits every error in the silhouette: a shadow at the edge, a
partial occlusion, a bounding box smeared by motion blur all move the ground
point, and all of them move it in a way that persists for as long as the cause
does.

There is a subtlety that makes this work better than it has any right to. The
ground point does not map to the centre of the vehicle - it maps to the vehicle's
*near extremity*, half a length short, exactly and measurably so. That is a real
bias in position. But it is a **constant** for a given vehicle, so it cancels the
moment two positions are differenced. An unbiased speed on a biased position.

**Route B, block matching (Unit 8).** Tile the vehicle's box into blocks, discard
the ones with too little texture to say anything, match the rest against the
previous frame, take the *median* displacement, refine it to sub-pixel by phase
correlation, and map it to the road. It measures where texture went rather than
where a silhouette's edge is, so its errors are independent of Route A's.

Route B has a subtlety of its own, and getting it wrong biases every reading.
Block matching measures the image displacement of texture *on the vehicle body*,
a metre or so above the road. Mapping that displacement through the road-plane
homography attributes it to the wrong plane, and the resulting error scales with
the height the blocks locked onto - so it does not average away. The homography
of the plane at that height is one added term (see
:meth:`PerspectiveModel.homography_at_height`), and using it removes the bias
entirely.

Both routes are computed for every track and reported side by side, because the
comparison is the finding. Neither is presented as the answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import FloatArray, StageResult, Track, TrackState
from kerbside.geometry.homography import apply_h_single, invert_h
from kerbside.geometry.perspective import PerspectiveModel
from kerbside.operators.frequency import phase_correlation
from kerbside.operators.motion import aggregate_vector, estimate_motion_field
from kerbside.stages.base import FrameContext, Stage

__all__ = ["SpeedStage", "SpeedEstimate"]

MPS_TO_KMPH = 3.6


@dataclass(frozen=True, slots=True)
class SpeedEstimate:
    """One track's speed by both routes, with what each was based on."""

    track_id: int
    track_kmph: float
    blockmatch_kmph: float
    track_samples: int
    textured_blocks: int
    total_blocks: int
    refined_px: float

    @property
    def agreement_kmph(self) -> float:
        return abs(self.track_kmph - self.blockmatch_kmph)

    @property
    def blockmatch_usable(self) -> bool:
        """Whether enough textured blocks survived for the median to mean anything.

        Trap 9. A plain-sided white truck has no texture, every candidate
        position matches equally well, and the vectors are arbitrary. Reporting a
        speed from three surviving blocks would look exactly like a measurement.
        """
        return self.textured_blocks >= 4


class SpeedStage(Stage):
    """S8: estimate each track's speed by both routes and compare them."""

    stage_id = "s8_speed"
    name = "Speed"
    units = (3, 8)
    rationale = (
        "Estimates speed twice. The track route differences the ground point's "
        "road-plane position over a known interval - biased in absolute "
        "position by half a vehicle length, but the bias is constant and "
        "cancels under differencing. The block-matching route measures where "
        "body texture went, mapped through the homography of the plane at that "
        "texture's own height rather than the road's, so the two routes fail in "
        "independent ways and the comparison means something."
    )
    param_specs = (
        ParamSpec(
            "window_frames", "Smoothing window", "int", 7, 2, 60, 1, unit="frames",
            description=(
                "Positions fitted for the track-route speed. A straight-line fit "
                "over several frames rejects the per-frame jitter that "
                "differencing two frames amplifies."
            ),
        ),
        ParamSpec("blockmatch_enabled", "Block matching", "bool", True),
        ParamSpec("block_size", "Block size", "int", 16, 4, 64, 4, unit="px"),
        ParamSpec("search_range", "Search range", "int", 16, 2, 64, 1, unit="px"),
        ParamSpec(
            "search", "Search strategy", "enum", "exhaustive",
            choices=("exhaustive", "two_d_log", "three_step"),
        ),
        ParamSpec(
            "min_variance", "Texture gate", "float", 25.0, 0.0, 2000.0, 1.0,
            description=(
                "Blocks below this intensity variance say nothing about "
                "displacement, and a median over too few survivors is not a "
                "measurement (trap 9)."
            ),
        ),
        ParamSpec(
            "subpixel_refine", "Phase-correlation refinement", "bool", True,
            description=(
                "Refine the integer block vector to sub-pixel on the vehicle's "
                "own box. At 25 fps a pixel is several km/h."
            ),
        ),
        ParamSpec(
            "texture_height_m", "Texture height", "float", 1.0, 0.0, 4.0, 0.1, unit="m",
            description=(
                "Assumed height above the road of the body texture the blocks "
                "lock onto. The displacement is mapped through the homography of "
                "the plane at this height; using the road plane instead biases "
                "every block-matching reading."
            ),
        ),
        ParamSpec(
            "max_speed_kmph", "Implausible speed", "float", 200.0, 20.0, 400.0, 5.0,
        ),
    )

    def __init__(self, **params: Any) -> None:
        self._previous_grey: FloatArray | None = None
        self._model: PerspectiveModel | None = None
        self._model_key: tuple[int, int, float] | None = None
        self._estimates: dict[int, SpeedEstimate] = {}
        super().__init__(**params)

    def reset(self) -> None:
        self._previous_grey = None
        self._estimates = {}

    @property
    def estimates(self) -> dict[int, SpeedEstimate]:
        return dict(self._estimates)

    def _perspective(self, ctx: FrameContext) -> PerspectiveModel | None:
        if ctx.calibration is None or not ctx.calibration.is_valid:
            return None
        h = np.asarray(ctx.calibration.h_world_to_img, dtype=np.float64)
        key = (ctx.calibration.width_px, ctx.calibration.height_px, float(h.sum()))
        if self._model is None or self._model_key != key:
            self._model = PerspectiveModel(
                h, ctx.calibration.width_px, ctx.calibration.height_px
            )
            self._model_key = key
        return self._model

    # -- Route A -----------------------------------------------------------

    def _track_speed_kmph(self, track: Track) -> tuple[float, int]:
        """Speed from a straight-line fit through recent road-plane positions.

        A least-squares fit rather than a difference of the two most recent
        positions. Differencing two frames divides a sub-pixel position error by
        a 40 ms interval, which turns a half-pixel wobble into several km/h; the
        fit uses every sample in the window and its slope is the velocity.
        """
        window = int(self.param("window_frames"))
        positions = track.history_world_m[-window:]
        times = track.history_time_s[-window:]
        if len(positions) < 2:
            return (0.0, len(positions))

        t = np.asarray(times, dtype=np.float64)
        span = float(t[-1] - t[0])
        if span <= 1e-6:
            return (0.0, len(positions))
        t = t - t[0]
        xy = np.asarray(positions, dtype=np.float64)
        # Slope of x and y against time, by least squares.
        design = np.vstack([t, np.ones_like(t)]).T
        try:
            solution, *_ = np.linalg.lstsq(design, xy, rcond=None)
        except np.linalg.LinAlgError:
            return (0.0, len(positions))
        vx, vy = float(solution[0, 0]), float(solution[0, 1])
        return (float(np.hypot(vx, vy)) * MPS_TO_KMPH, len(positions))

    # -- Route B -----------------------------------------------------------

    def _blockmatch_speed_kmph(
        self, track: Track, ctx: FrameContext, model: PerspectiveModel, dt_s: float
    ) -> tuple[float, int, int, float]:
        """Speed from the median block displacement over the vehicle's box."""
        if self._previous_grey is None or track.bbox_px is None:
            return (0.0, 0, 0, 0.0)
        grey = ctx.require_grey()
        box = track.bbox_px.clipped(grey.shape[1], grey.shape[0])
        block = int(self.param("block_size"))
        if box.w_px < block or box.h_px < block:
            return (0.0, 0, 0, 0.0)

        field = estimate_motion_field(
            grey,
            self._previous_grey,
            block_size=block,
            search_range=int(self.param("search_range")),
            search=str(self.param("search")),  # type: ignore[arg-type]
            min_variance=float(self.param("min_variance")),
            region=(int(box.x_px), int(box.y_px), int(box.w_px), int(box.h_px)),
        )
        total = field.rows * field.cols
        textured = int(field.textured.sum())
        if total == 0:
            return (0.0, 0, 0, 0.0)

        dx, dy = aggregate_vector(field, textured_only=textured >= 4)

        refined = 0.0
        if self.param("subpixel_refine"):
            y0, y1 = int(box.y_px), int(box.y2_px)
            x0, x1 = int(box.x_px), int(box.x2_px)
            current = grey[y0:y1, x0:x1]
            # Shift the reference window by the integer estimate first, so the
            # correlation only has to resolve the fractional remainder - which
            # is what it is good at and what the integer search cannot give.
            ry0 = int(np.clip(y0 - round(dy), 0, grey.shape[0] - (y1 - y0)))
            rx0 = int(np.clip(x0 - round(dx), 0, grey.shape[1] - (x1 - x0)))
            reference = self._previous_grey[ry0 : ry0 + (y1 - y0), rx0 : rx0 + (x1 - x0)]
            if reference.shape == current.shape and min(reference.shape) >= 8:
                correlation = phase_correlation(reference, current, upsample_factor=20)
                if correlation.is_confident and correlation.magnitude_px < 3.0:
                    dx += correlation.dx_px
                    dy += correlation.dy_px
                    refined = correlation.magnitude_px

        # Map the displacement through the plane the texture actually sits on.
        height_m = float(self.param("texture_height_m"))
        h_plane = model.homography_at_height(height_m)
        if h_plane is None:
            h_plane = model.h_world_to_img
        h_inv = invert_h(h_plane)

        centre_x = box.x_px + box.w_px / 2.0
        centre_y = box.y_px + box.h_px / 2.0
        start = apply_h_single(h_inv, centre_x - dx, centre_y - dy)
        end = apply_h_single(h_inv, centre_x, centre_y)
        if not all(np.isfinite(v) for v in (*start, *end)) or dt_s <= 0:
            return (0.0, textured, total, refined)

        speed = float(np.hypot(end[0] - start[0], end[1] - start[1])) / dt_s * MPS_TO_KMPH
        if speed > float(self.param("max_speed_kmph")):
            return (0.0, textured, total, refined)
        return (speed, textured, total, refined)

    # -- execution ---------------------------------------------------------

    def process(self, ctx: FrameContext) -> StageResult:
        with self._timed() as result:
            model = self._perspective(ctx)
            dt_s = ctx.dt_s if ctx.dt_s > 0 else 1.0 / 25.0
            tracks: tuple[Track, ...] = ctx.tracks

            estimates: dict[int, SpeedEstimate] = {}
            agreements: list[float] = []
            for track in tracks:
                if track.state is TrackState.TENTATIVE:
                    continue
                track_kmph, samples = self._track_speed_kmph(track)
                block_kmph, textured, total, refined = (0.0, 0, 0, 0.0)
                if model is not None and self.param("blockmatch_enabled"):
                    block_kmph, textured, total, refined = self._blockmatch_speed_kmph(
                        track, ctx, model, dt_s
                    )

                track.speed_kmph_track = track_kmph
                track.speed_kmph_blockmatch = block_kmph
                track.speed_history_kmph.append(track_kmph)
                if len(track.speed_history_kmph) > 120:
                    del track.speed_history_kmph[0]
                track.mean_speed_kmph_track = float(
                    np.mean(track.speed_history_kmph[-30:])
                )

                estimate = SpeedEstimate(
                    track_id=track.track_id,
                    track_kmph=track_kmph,
                    blockmatch_kmph=block_kmph,
                    track_samples=samples,
                    textured_blocks=textured,
                    total_blocks=total,
                    refined_px=refined,
                )
                estimates[track.track_id] = estimate
                if estimate.blockmatch_usable and block_kmph > 0:
                    agreements.append(estimate.agreement_kmph)

            self._estimates = estimates
            self._previous_grey = ctx.require_grey().copy()

            speeds = [e.track_kmph for e in estimates.values() if e.track_kmph > 0]
            result.metrics = {
                "tracks_measured": float(len(estimates)),
                "mean_speed_kmph": float(np.mean(speeds)) if speeds else 0.0,
                "route_agreement_kmph": float(np.median(agreements)) if agreements else 0.0,
                "blockmatch_usable": float(
                    sum(1 for e in estimates.values() if e.blockmatch_usable)
                ),
            }
            if model is None:
                result.extras["hint"] = (
                    "no valid calibration, so no speed can be computed at all - "
                    "speed is a distance in metres divided by a time, and "
                    "without geometry there are no metres"
                )
        return result


def percentile_85(speeds: list[float]) -> float:
    """The 85th-percentile speed, the standard traffic-engineering statistic.

    Traffic engineering uses it rather than the mean because it describes the
    speed most drivers consider reasonable, and is far less sensitive to a
    handful of very slow vehicles than an average is.
    """
    if not speeds:
        return 0.0
    return float(np.percentile(np.asarray(speeds, dtype=np.float64), 85))
