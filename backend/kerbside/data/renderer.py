"""Synthetic fixed-camera traffic renderer with exact ground truth.

Every accuracy number in this project is measured against this file, so it is
built first and built carefully. The renderer projects simple cuboid vehicles
through a *known* pinhole camera onto a road plane and records, per frame and
per vehicle, the world position, image bounding box, ground point, lane, speed,
class, colour and occlusion. Nothing is hand-annotated and nothing is estimated,
so the ground truth is exact by construction rather than by agreement.

What makes the output a fair test rather than a rigged one:

* **Cast shadows are physically constructed.** Each cuboid vertex is projected
  along the sun direction onto the road plane and the convex hull of those
  points is darkened. Because darkening is a multiplication of all three
  channels, hue and saturation are preserved exactly and value is scaled - which
  is precisely the model S5 tests for. A small ambient sky tint is added on top
  by default so the saturation and hue tolerances in S5 are genuinely exercised
  rather than trivially satisfied.
* **Occlusion is measured, not assumed.** Silhouettes are rasterised in
  painter's order into an id buffer and each vehicle's visible pixel count is
  compared against its unoccluded count, so ``occlusion_frac`` is a real
  measurement of the frame that was written.
* **Texture is fixed to the vehicle, not to the image.** Windows, lights, wheels
  and livery are geometry in the vehicle's own frame, so block matching (Unit 8)
  measures genuine texture displacement. Vehicles with ``texture_amp`` at zero
  are deliberately plain, which is how the aperture problem of trap 9 appears in
  the data rather than having to be argued for.
* **Stop-and-go emerges from car following.** Speeds come from an intelligent
  driver model, not a scripted sine wave, so congested clips contain real
  queue formation and full stops - the input that trap 5 (background ghosting)
  needs.

Syllabus context: this module is the source of the data, not a demonstration of
an operator. It exercises Unit 1 (image formation and resolution) and supplies
the Unit 7 colour content that S5 and S9 work on.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from kerbside.core.types import (
    BBox,
    Calibration,
    CameraModel,
    Direction,
    FloatArray,
    GroundTruth,
    GTFrame,
    GTVehicleFrame,
    Int32Array,
    Lane,
    UInt8Array,
    VehicleClass,
)
from kerbside.geometry.calibration import lanes_from_world_bands
from kerbside.geometry.homography import (
    camera_projection_matrix,
    homography_from_camera,
    horizon_row_px,
    normalise_h,
)

__all__ = [
    "VehicleSpec",
    "VEHICLE_CATALOGUE",
    "COLOUR_CATALOGUE",
    "RoadSpec",
    "TrafficConfig",
    "LightingConfig",
    "AtmosphereConfig",
    "ShakeConfig",
    "SceneConfig",
    "TrafficRenderer",
    "write_video",
]


# --------------------------------------------------------------------------
# Catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VehicleSpec:
    """A vehicle body: real dimensions and how much surface detail it carries.

    ``texture_amp`` is the number of extra livery panels drawn on the sides and
    roof. Zero means a plain-sided body, which is the case block matching finds
    hardest (trap 9) and which therefore has to be present in the data.
    """

    name: str
    vehicle_class: VehicleClass
    length_m: float
    width_m: float
    height_m: float
    cabin_frac: float = 0.55
    texture_amp: int = 2
    weight: float = 1.0


VEHICLE_CATALOGUE: tuple[VehicleSpec, ...] = (
    VehicleSpec("motorcycle", VehicleClass.TWO_WHEELER, 2.10, 0.82, 1.38, 0.70, 1, 0.10),
    VehicleSpec("hatchback", VehicleClass.CAR, 3.95, 1.72, 1.48, 0.58, 3, 0.26),
    VehicleSpec("saloon", VehicleClass.CAR, 4.55, 1.80, 1.45, 0.52, 2, 0.24),
    VehicleSpec("suv", VehicleClass.CAR, 4.75, 1.95, 1.76, 0.60, 3, 0.16),
    VehicleSpec("panel_van", VehicleClass.LCV, 5.90, 2.05, 2.45, 0.30, 1, 0.11),
    VehicleSpec("box_truck", VehicleClass.TRUCK_BUS, 9.20, 2.45, 3.35, 0.22, 0, 0.07),
    VehicleSpec("bus", VehicleClass.TRUCK_BUS, 11.50, 2.55, 3.15, 0.90, 4, 0.06),
)
"""Body catalogue. ``weight`` is the sampling probability in free-flow traffic;
the mix is deliberately car-heavy so class accuracy is not inflated by an
unrealistically even distribution across the four buckets."""


COLOUR_CATALOGUE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("white", (232, 233, 235)),
    ("silver", (168, 172, 177)),
    ("grey", (108, 112, 118)),
    ("black", (28, 30, 34)),
    ("red", (168, 38, 40)),
    ("blue", (38, 74, 148)),
    ("green", (40, 110, 74)),
    ("yellow", (214, 176, 46)),
    ("orange", (206, 108, 38)),
    ("brown", (104, 74, 52)),
)
"""Body colours as RGB. White, silver, grey and black dominate real vehicle
parcs, so the sampler below weights them accordingly - which matters because the
achromatic cases are exactly the ones S9's colour naming finds hardest."""

_COLOUR_WEIGHTS: tuple[float, ...] = (0.24, 0.18, 0.16, 0.14, 0.08, 0.08, 0.04, 0.03, 0.03, 0.02)


# --------------------------------------------------------------------------
# Scene configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoadSpec:
    """Road layout in world metres.

    Lanes are defined by their centre lines and a common width, both in metres,
    and projected into the image by the calibration - so lane edges stay parallel
    on the road, which is what they physically are.
    """

    lane_centres_m: tuple[float, ...] = (-5.25, -1.75, 1.75, 5.25)
    lane_width_m: float = 3.5
    directions: tuple[Direction, ...] = (
        Direction.TOWARD,
        Direction.TOWARD,
        Direction.AWAY,
        Direction.AWAY,
    )
    marking_pitch_m: float = 9.0
    marking_length_m: float = 3.0
    shoulder_m: float = 2.0
    y_near_m: float = 8.0
    y_far_m: float = 150.0
    asphalt_rgb: tuple[int, int, int] = (74, 76, 80)
    verge_rgb: tuple[int, int, int] = (58, 68, 48)
    marking_rgb: tuple[int, int, int] = (196, 196, 190)

    @property
    def lane_count(self) -> int:
        return len(self.lane_centres_m)

    @property
    def half_width_m(self) -> float:
        return max(abs(c) for c in self.lane_centres_m) + self.lane_width_m / 2.0

    def lane_bands_m(self) -> list[tuple[float, float]]:
        return [
            (c - self.lane_width_m / 2.0, c + self.lane_width_m / 2.0)
            for c in self.lane_centres_m
        ]


@dataclass(frozen=True, slots=True)
class TrafficConfig:
    """Traffic demand and driver behaviour.

    The speed and acceleration parameters feed an intelligent driver model, so
    congestion produces genuine stop-and-go waves instead of a scripted
    slowdown. That matters because a vehicle that actually stops is what makes
    background ghosting (trap 5) appear.
    """

    flow_veh_per_hour_per_lane: float = 900.0
    desired_speed_kmph: tuple[float, float] = (48.0, 72.0)
    lateral_jitter_m: float = 0.22
    min_spawn_gap_m: float = 12.0
    accel_max_mps2: float = 1.4
    decel_comfort_mps2: float = 2.0
    time_headway_s: float = 1.4
    jam_gap_m: float = 2.5
    speed_limit_kmph: float = 60.0
    class_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LightingConfig:
    """Illumination and the cast-shadow model.

    ``sun_azimuth_deg`` is the compass-style direction the shadow is thrown
    towards, measured from world +X towards world +Y; ``sun_elevation_deg`` is
    the sun's height above the horizon. Together they give the horizontal
    displacement per metre of object height, which is all a flat-ground shadow
    needs.

    ``shadow_value_scale`` multiplies all three channels, which preserves hue and
    saturation exactly and scales value - the ideal case S5 assumes.
    ``shadow_channel_gain_rgb`` then applies a small *per-channel* correction on
    top, because a shadowed surface is still lit, just by the sky rather than by
    the sun, and skylight is bluer. Radiometrically that is a change of
    illuminant, which multiplies the surface reflectance per channel; it is not
    an additive glow, and modelling it additively would inflate the saturation
    of near-neutral asphalt far beyond anything a camera records.

    The default gains are mild, which is what a camera with auto white balance
    actually delivers. Setting them all to 1.0 makes the shadow a pure value
    scaling and S5's job artificially easy, so they are non-unity by default and
    S5's hue and saturation tolerances are genuinely exercised.
    """

    ambient: float = 1.0
    sun_azimuth_deg: float = 196.0
    sun_elevation_deg: float = 23.0
    shadow_value_scale: float = 0.55
    shadow_channel_gain_rgb: tuple[float, float, float] = (0.95, 1.00, 1.06)
    shadow_softness_px: int = 3
    cast_shadows: bool = True
    night: bool = False
    headlights: bool = False
    headlight_gain: float = 0.85
    headlight_reach_m: float = 16.0


@dataclass(frozen=True, slots=True)
class AtmosphereConfig:
    """Weather and sensor effects applied to the finished frame."""

    rain_intensity: float = 0.0
    rain_streak_len_px: int = 22
    contrast_loss: float = 0.0
    sensor_noise_sigma: float = 2.0
    shutter_s: float = 0.0
    shutter_samples: int = 5


@dataclass(frozen=True, slots=True)
class ShakeConfig:
    """Sinusoidal camera shake of known amplitude.

    A pole-mounted camera vibrates. The offset is recorded per frame in the
    ground truth so S2's phase-correlation stabiliser can be scored against a
    number rather than against an impression.
    """

    amplitude_px: float = 0.0
    frequency_hz: float = 1.7
    vertical_ratio: float = 0.6
    phase: float = 0.0


@dataclass(frozen=True, slots=True)
class SceneConfig:
    """Everything needed to render a clip deterministically."""

    name: str = "free_flow"
    width_px: int = 1280
    height_px: int = 720
    fps: float = 25.0
    duration_s: float = 20.0
    seed: int = 20240501
    camera: CameraModel = field(
        default_factory=lambda: CameraModel(
            width_px=1280, height_px=720, height_m=8.0, tilt_deg=12.0, fov_deg=55.0
        )
    )
    road: RoadSpec = field(default_factory=RoadSpec)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    lighting: LightingConfig = field(default_factory=LightingConfig)
    atmosphere: AtmosphereConfig = field(default_factory=AtmosphereConfig)
    shake: ShakeConfig = field(default_factory=ShakeConfig)
    supersample: int = 2
    plan_px_per_m: float = 10.0
    notes: str = ""

    @property
    def frame_count(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))

    def with_resolution(self, width_px: int, height_px: int) -> SceneConfig:
        """Same scene at a different resolution, for the Unit 1 resolution study.

        The field of view is held fixed so the camera sees the same road; only
        the sampling changes, which is exactly the variable the study is about.
        """
        cam = replace(self.camera, width_px=width_px, height_px=height_px)
        return replace(self, width_px=width_px, height_px=height_px, camera=cam)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Vehicle:
    """One simulated vehicle. ``s_m`` is distance travelled along its own
    direction of travel, which makes car following identical for both
    carriageways."""

    vehicle_id: int
    lane_id: int
    spec: VehicleSpec
    colour_name: str
    body_rgb: tuple[int, int, int]
    direction: Direction
    lane_centre_m: float
    x_offset_m: float
    s_m: float
    speed_mps: float
    desired_speed_mps: float
    livery_seed: int
    entered_frame: int = 0

    @property
    def heading_deg(self) -> float:
        return 0.0 if self.direction is Direction.AWAY else 180.0

    def world_xy_m(self, s_m: float | None = None) -> tuple[float, float]:
        s = self.s_m if s_m is None else s_m
        x = self.lane_centre_m + self.x_offset_m
        y = s if self.direction is Direction.AWAY else -s
        return (x, y)


class _Simulation:
    """Deterministic traffic simulation, run to completion before rendering.

    Separating the simulation from the drawing means the ground truth exists
    independently of how the scene is painted, and it makes seeking to a frame
    free - which the console's scrub bar depends on.
    """

    def __init__(self, config: SceneConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self._next_id = 1
        weights = np.array(
            [s.weight for s in VEHICLE_CATALOGUE], dtype=np.float64
        )
        if config.traffic.class_weights:
            for i, spec in enumerate(VEHICLE_CATALOGUE):
                weights[i] = config.traffic.class_weights.get(spec.name, weights[i])
        self._spec_p = weights / weights.sum()
        self._colour_p = np.array(_COLOUR_WEIGHTS, dtype=np.float64)
        self._colour_p = self._colour_p / self._colour_p.sum()

    # -- world extent ------------------------------------------------------

    def _travel_span_m(self, direction: Direction) -> tuple[float, float]:
        """Range of ``s_m`` over which a vehicle exists, per direction.

        ``s_m`` is distance along the vehicle's own direction of travel and
        always increases, so car following is identical on both carriageways.
        For the AWAY carriageway ``y = +s``; for TOWARD ``y = -s``, which puts
        the span at negative ``s`` but still running low to high.
        """
        road = self.config.road
        margin = 16.0
        near, far = road.y_near_m - margin, road.y_far_m + margin
        if direction is Direction.AWAY:
            return (near, far)
        return (-far, -near)

    def run(self) -> list[list[_Vehicle]]:
        """Simulate every frame, returning per-frame snapshots of the vehicles."""
        cfg = self.config
        dt_s = 1.0 / cfg.fps
        active: list[_Vehicle] = []
        snapshots: list[list[_Vehicle]] = []

        spawn_prob = [
            cfg.traffic.flow_veh_per_hour_per_lane / 3600.0 * dt_s
            for _ in cfg.road.lane_centres_m
        ]

        # Warm the road up before frame zero so the clip does not open on an
        # empty carriageway that then fills - which would make the first seconds
        # unrepresentative of the configured flow.
        warmup_frames = int(round(cfg.fps * 8.0))
        for step in range(-warmup_frames, cfg.frame_count):
            self._spawn(active, spawn_prob, step)
            self._advance(active, dt_s)
            active = [v for v in active if self._in_span(v)]
            if step >= 0:
                snapshots.append([replace(v) for v in active])
        return snapshots

    def _in_span(self, v: _Vehicle) -> bool:
        lo, hi = self._travel_span_m(v.direction)
        return lo <= v.s_m <= hi

    def _spawn(self, active: list[_Vehicle], spawn_prob: list[float], frame: int) -> None:
        cfg = self.config
        for lane_id, centre in enumerate(cfg.road.lane_centres_m):
            if self.rng.random() >= spawn_prob[lane_id]:
                continue
            direction = cfg.road.directions[lane_id]
            entry_s = self._entry_s_m(direction)
            behind = [v.s_m for v in active if v.lane_id == lane_id]
            if behind and min(behind) - entry_s < cfg.traffic.min_spawn_gap_m:
                continue

            spec = VEHICLE_CATALOGUE[int(self.rng.choice(len(VEHICLE_CATALOGUE), p=self._spec_p))]
            colour_idx = int(self.rng.choice(len(COLOUR_CATALOGUE), p=self._colour_p))
            colour_name, body_rgb = COLOUR_CATALOGUE[colour_idx]
            v_lo, v_hi = cfg.traffic.desired_speed_kmph
            desired_kmph = float(self.rng.uniform(v_lo, v_hi))
            active.append(
                _Vehicle(
                    vehicle_id=self._next_id,
                    lane_id=lane_id,
                    spec=spec,
                    colour_name=colour_name,
                    body_rgb=body_rgb,
                    direction=direction,
                    lane_centre_m=centre,
                    x_offset_m=float(
                        self.rng.normal(0.0, cfg.traffic.lateral_jitter_m)
                    ),
                    s_m=entry_s,
                    speed_mps=desired_kmph / 3.6,
                    desired_speed_mps=desired_kmph / 3.6,
                    livery_seed=int(self.rng.integers(0, 2**31 - 1)),
                    entered_frame=frame,
                )
            )
            self._next_id += 1

    def _entry_s_m(self, direction: Direction) -> float:
        """Vehicles always enter at the low end of their own travel span."""
        return self._travel_span_m(direction)[0]

    def _advance(self, active: list[_Vehicle], dt_s: float) -> None:
        """One intelligent-driver-model step.

        The IDM acceleration is

            a = a_max [ 1 - (v / v0)^4 - (s* / gap)^2 ]
            s* = s0 + v T + v dv / (2 sqrt(a_max b))

        with ``gap`` the clear distance to the vehicle ahead and ``dv`` the
        closing speed. It is deterministic, produces realistic queue discharge,
        and - unlike a scripted slowdown - genuinely brings vehicles to a stop
        when demand exceeds capacity.
        """
        t = self.config.traffic
        by_lane: dict[int, list[_Vehicle]] = {}
        for v in active:
            by_lane.setdefault(v.lane_id, []).append(v)

        for vehicles in by_lane.values():
            vehicles.sort(key=lambda v: v.s_m)
            for i, v in enumerate(vehicles):
                leader = vehicles[i + 1] if i + 1 < len(vehicles) else None
                free = 1.0 - (v.speed_mps / max(v.desired_speed_mps, 0.1)) ** 4
                interaction = 0.0
                if leader is not None:
                    gap_m = (
                        leader.s_m
                        - v.s_m
                        - (leader.spec.length_m + v.spec.length_m) / 2.0
                    )
                    gap_m = max(gap_m, 0.35)
                    dv = v.speed_mps - leader.speed_mps
                    s_star = (
                        t.jam_gap_m
                        + v.speed_mps * t.time_headway_s
                        + v.speed_mps
                        * dv
                        / (2.0 * float(np.sqrt(t.accel_max_mps2 * t.decel_comfort_mps2)))
                    )
                    s_star = max(s_star, t.jam_gap_m)
                    interaction = (s_star / gap_m) ** 2
                accel = t.accel_max_mps2 * (free - interaction)
                accel = float(np.clip(accel, -6.0, t.accel_max_mps2))
                v.speed_mps = max(0.0, v.speed_mps + accel * dt_s)
                v.s_m += v.speed_mps * dt_s


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------


def _rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


class TrafficRenderer:
    """Renders a :class:`SceneConfig` to frames plus exact ground truth."""

    def __init__(self, config: SceneConfig) -> None:
        if config.supersample < 1:
            raise ValueError(f"supersample must be >= 1, got {config.supersample}")
        if config.camera.width_px != config.width_px or config.camera.height_px != config.height_px:
            config = replace(
                config,
                camera=replace(
                    config.camera, width_px=config.width_px, height_px=config.height_px
                ),
            )
        self.config = config
        self.camera = config.camera
        self.h_world_to_img = homography_from_camera(self.camera)
        self.projection = camera_projection_matrix(self.camera)
        self.horizon_row_px = horizon_row_px(self.camera)
        self._ss = config.supersample
        self._w_ss = config.width_px * self._ss
        self._h_ss = config.height_px * self._ss
        self.lanes = self._build_lanes()
        self._road_bgr = self._render_road_background()
        self._sim_frames = _Simulation(config).run()

    # -- public API --------------------------------------------------------

    @property
    def frame_count(self) -> int:
        return len(self._sim_frames)

    def calibration(self) -> Calibration:
        """The exact rendering calibration, for tests and for the 'known
        homography' baseline the UI compares a user calibration against."""
        return Calibration(
            h_world_to_img=self.h_world_to_img,
            width_px=self.config.width_px,
            height_px=self.config.height_px,
            source=f"synthetic:{self.config.name}",
            lanes=self.lanes,
            notes="exact rendering homography, not a fit",
        )

    def render_frame(self, index: int) -> tuple[UInt8Array, GTFrame]:
        """Render one frame and its ground truth."""
        if not 0 <= index < len(self._sim_frames):
            raise IndexError(
                f"frame {index} out of range for a {len(self._sim_frames)}-frame clip"
            )
        vehicles = self._sim_frames[index]
        time_s = index / self.config.fps
        shake_px = self._shake_offset_px(time_s)

        gt_vehicles = self._ground_truth(vehicles, shake_px)
        frame = self._paint(vehicles, time_s, shake_px, index)
        return frame, GTFrame(
            frame_index=index,
            time_s=time_s,
            vehicles=tuple(gt_vehicles),
            shake_offset_px=shake_px,
            illumination=self.config.lighting.ambient,
        )

    def render_clip(self) -> Iterator[tuple[UInt8Array, GTFrame]]:
        for i in range(len(self._sim_frames)):
            yield self.render_frame(i)

    def ground_truth(self, frames: list[GTFrame]) -> GroundTruth:
        return GroundTruth(
            scenario=self.config.name,
            width_px=self.config.width_px,
            height_px=self.config.height_px,
            fps=self.config.fps,
            seed=self.config.seed,
            camera=self.camera,
            h_world_to_img=self.h_world_to_img,
            lanes=self.lanes,
            frames=tuple(frames),
            lane_marking_pitch_m=self.config.road.marking_pitch_m,
            road_width_m=self.config.road.half_width_m * 2.0,
            notes=self.config.notes,
        )

    # -- geometry helpers --------------------------------------------------

    def _project(self, points_world_m: FloatArray) -> FloatArray:
        """Project ``(N, 3)`` world points to output-resolution pixels."""
        pts = np.asarray(points_world_m, dtype=np.float64)
        homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
        mapped = homo @ self.projection.T
        depth = mapped[:, 2]
        out = np.full((pts.shape[0], 2), np.inf, dtype=np.float64)
        front = depth > 1e-6
        out[front] = mapped[front, :2] / depth[front, None]
        return out

    def _cuboid_world(self, v: _Vehicle, s_m: float | None = None) -> FloatArray:
        """The eight corners of a vehicle in world metres.

        Order: the four footprint corners (z = 0) then the four roof corners,
        each running rear-left, rear-right, front-right, front-left in the
        vehicle's own frame.
        """
        half_w = v.spec.width_m / 2.0
        half_l = v.spec.length_m / 2.0
        h = v.spec.height_m
        local = np.array(
            [
                [-half_w, -half_l, 0.0],
                [half_w, -half_l, 0.0],
                [half_w, half_l, 0.0],
                [-half_w, half_l, 0.0],
                [-half_w, -half_l, h],
                [half_w, -half_l, h],
                [half_w, half_l, h],
                [-half_w, half_l, h],
            ],
            dtype=np.float64,
        )
        return self._local_to_world(local, v, s_m)

    def _local_to_world(
        self, local: FloatArray, v: _Vehicle, s_m: float | None = None
    ) -> FloatArray:
        a = np.deg2rad(v.heading_deg)
        c, s = float(np.cos(a)), float(np.sin(a))
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        x_m, y_m = v.world_xy_m(s_m)
        return local @ rot.T + np.array([x_m, y_m, 0.0], dtype=np.float64)

    def _shadow_polygon_world(self, v: _Vehicle, s_m: float | None = None) -> FloatArray:
        """Ground-plane shadow of a cuboid under a directional light.

        Each vertex at height ``z`` casts to ``(x, y) + z * d`` where ``d`` is the
        horizontal displacement per metre of height, ``cot(elevation)`` in the
        sun's azimuth. The shadow is the convex hull of the eight cast points
        together with the footprint itself, which for a convex body is exact.
        """
        lig = self.config.lighting
        elev = np.deg2rad(max(lig.sun_elevation_deg, 3.0))
        az = np.deg2rad(lig.sun_azimuth_deg)
        per_m = float(1.0 / np.tan(elev))
        d = np.array([np.cos(az) * per_m, np.sin(az) * per_m], dtype=np.float64)

        corners = self._cuboid_world(v, s_m)
        ground = corners[:, :2] + corners[:, 2:3] * d[None, :]
        pts = np.vstack([corners[:4, :2], ground])
        hull = cv2.convexHull(pts.astype(np.float32)).reshape(-1, 2)
        return np.hstack([hull.astype(np.float64), np.zeros((hull.shape[0], 1))])

    def _shake_offset_px(self, time_s: float) -> tuple[float, float]:
        sh = self.config.shake
        if sh.amplitude_px <= 0.0:
            return (0.0, 0.0)
        w = 2.0 * np.pi * sh.frequency_hz * time_s + sh.phase
        dx = sh.amplitude_px * float(np.sin(w))
        dy = sh.amplitude_px * sh.vertical_ratio * float(np.sin(1.7 * w + 1.1))
        return (dx, dy)

    def _build_lanes(self) -> tuple[Lane, ...]:
        road = self.config.road
        calib = Calibration(
            h_world_to_img=self.h_world_to_img,
            width_px=self.config.width_px,
            height_px=self.config.height_px,
            source="synthetic",
        )
        # Keep the counting line comfortably inside the field of view so a
        # vehicle is fully visible on both sides of it - a counting line at the
        # very edge of frame produces crossings that no tracker could confirm.
        y0 = road.y_near_m + 6.0
        y1 = min(road.y_far_m, 90.0)
        return lanes_from_world_bands(
            calib,
            road.lane_bands_m(),
            (y0, y1),
            directions=list(road.directions),
            counting_line_y_m=y0 + 0.35 * (y1 - y0),
            speed_limit_kmph=self.config.traffic.speed_limit_kmph,
            names=[f"L{i + 1} {d.value}" for i, d in enumerate(road.directions)],
        )

    # -- ground truth ------------------------------------------------------

    def _silhouette_px(self, v: _Vehicle) -> FloatArray | None:
        """Convex silhouette of a vehicle at output resolution, or ``None`` if
        it does not project in front of the camera."""
        corners_px = self._project(self._cuboid_world(v))
        if not np.isfinite(corners_px).all():
            return None
        hull = cv2.convexHull(corners_px.astype(np.float32)).reshape(-1, 2)
        return hull.astype(np.float64)

    def _ground_truth(
        self, vehicles: list[_Vehicle], shake_px: tuple[float, float]
    ) -> list[GTVehicleFrame]:
        w, h = self.config.width_px, self.config.height_px
        records: list[GTVehicleFrame] = []

        drawable: list[tuple[_Vehicle, FloatArray, float]] = []
        for v in vehicles:
            sil = self._silhouette_px(v)
            if sil is None:
                continue
            sil = sil + np.array(shake_px, dtype=np.float64)
            x_m, y_m = v.world_xy_m()
            depth = self._camera_depth_m(x_m, y_m)
            drawable.append((v, sil, depth))
        drawable.sort(key=lambda item: -item[2])  # far first, near overwrites

        id_buffer: Int32Array = np.zeros((h, w), dtype=np.int32)
        solo_area: dict[int, int] = {}
        scratch = np.zeros((h, w), dtype=np.uint8)
        for v, sil, _ in drawable:
            poly = np.round(sil).astype(np.int32).reshape(-1, 1, 2)
            x0, y0, bw, bh = cv2.boundingRect(poly)
            x0c, y0c = max(0, x0), max(0, y0)
            x1c, y1c = min(w, x0 + bw), min(h, y0 + bh)
            if x1c <= x0c or y1c <= y0c:
                solo_area[v.vehicle_id] = 0
                continue
            scratch[y0c:y1c, x0c:x1c] = 0
            cv2.fillConvexPoly(scratch, poly, 1)
            solo_area[v.vehicle_id] = int(scratch[y0c:y1c, x0c:x1c].sum())
            scratch[y0c:y1c, x0c:x1c] = 0
            cv2.fillConvexPoly(id_buffer, poly, int(v.vehicle_id))

        visible_counts = {
            int(k): int(c)
            for k, c in zip(*np.unique(id_buffer, return_counts=True), strict=True)
            if k != 0
        }

        for v, sil, _ in drawable:
            x_m, y_m = v.world_xy_m()
            gp = self._project(np.array([[x_m, y_m, 0.0]]))[0]
            gp = (float(gp[0] + shake_px[0]), float(gp[1] + shake_px[1]))
            bbox = BBox.from_points(sil)
            bbox_gp = bbox.ground_point_px
            solo = solo_area.get(v.vehicle_id, 0)
            vis = visible_counts.get(v.vehicle_id, 0)
            occlusion = 0.0 if solo <= 0 else float(np.clip(1.0 - vis / solo, 0.0, 1.0))
            records.append(
                GTVehicleFrame(
                    vehicle_id=v.vehicle_id,
                    world_pos_m=(x_m, y_m),
                    ground_point_px=gp,
                    bbox_px=bbox.as_tuple(),
                    bbox_ground_point_px=(bbox_gp.x, bbox_gp.y),
                    ground_point_bias_px=(bbox_gp.x - gp[0], bbox_gp.y - gp[1]),
                    speed_kmph=v.speed_mps * 3.6,
                    heading_deg=v.heading_deg,
                    lane_id=v.lane_id,
                    direction=v.direction,
                    vehicle_class=v.spec.vehicle_class,
                    footprint_m=(v.spec.length_m, v.spec.width_m, v.spec.height_m),
                    colour_name=v.colour_name,
                    body_rgb=v.body_rgb,
                    occlusion_frac=occlusion,
                    visible=vis > 0,
                    visible_area_px=vis,
                )
            )
        records.sort(key=lambda r: r.vehicle_id)
        return records

    def _camera_depth_m(self, x_m: float, y_m: float) -> float:
        """Camera-frame depth of a road point - the painter's-order key."""
        homo = np.array([x_m, y_m, 0.0, 1.0], dtype=np.float64)
        return float((self.projection @ homo)[2])

    # -- painting ----------------------------------------------------------

    def _render_road_background(self) -> UInt8Array:
        """Paint the static scene once: sky, verge, asphalt and lane markings.

        The road surface is built as a birdseye plan in metres and warped into
        the camera view through the known homography, so lane markings land at
        exactly the configured real-world pitch and the asphalt texture is
        perspective-consistent - which is what gives the phase-correlation
        stabiliser of S2 something real to lock onto.
        """
        cfg = self.config
        road = cfg.road
        ss = self._ss
        lig = cfg.lighting

        frame = np.zeros((self._h_ss, self._w_ss, 3), dtype=np.uint8)
        sky = np.array(
            _rgb_to_bgr((150, 160, 172) if not lig.night else (14, 16, 22)), dtype=np.uint8
        )
        frame[:, :] = sky

        # Verge below the horizon, then the road plan on top of it.
        horizon_ss = int(np.clip(self.horizon_row_px * ss, 0, self._h_ss))
        verge = np.array(
            _rgb_to_bgr(road.verge_rgb if not lig.night else (18, 22, 16)), dtype=np.uint8
        )
        frame[horizon_ss:, :] = verge

        plan, plan_to_world = self._road_plan_bgr()
        scale = np.array([[ss, 0, 0], [0, ss, 0], [0, 0, 1]], dtype=np.float64)
        plan_to_img = normalise_h(scale @ self.h_world_to_img @ plan_to_world)
        warped = cv2.warpPerspective(
            plan,
            plan_to_img,
            (self._w_ss, self._h_ss),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpPerspective(
            np.full(plan.shape[:2], 255, dtype=np.uint8),
            plan_to_img,
            (self._w_ss, self._h_ss),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        np.copyto(frame, warped, where=(mask > 127)[:, :, None])

        if lig.ambient != 1.0:
            frame = np.clip(frame.astype(np.float32) * lig.ambient, 0, 255).astype(np.uint8)
        return frame

    def _road_plan_bgr(self) -> tuple[UInt8Array, FloatArray]:
        """The road surface drawn in plan view, plus its plan-to-world map."""
        cfg = self.config
        road = cfg.road
        ppm = cfg.plan_px_per_m
        half_w = road.half_width_m + road.shoulder_m
        y0, y1 = road.y_near_m - 20.0, road.y_far_m + 40.0
        w = int(round(2 * half_w * ppm))
        h = int(round((y1 - y0) * ppm))

        rng = np.random.default_rng(cfg.seed ^ 0x5EED)
        base = np.array(_rgb_to_bgr(road.asphalt_rgb), dtype=np.float32)
        plan = np.repeat(np.repeat(base.reshape(1, 1, 3), h, axis=0), w, axis=1)

        # Correlated asphalt texture: white noise blurred to a few centimetres,
        # so minification at distance attenuates it rather than aliasing it into
        # false motion.
        noise = rng.normal(0.0, 1.0, size=(h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=ppm * 0.18)
        noise *= 9.0 / max(float(noise.std()), 1e-6)
        plan += noise[:, :, None]

        # Tyre-polished wheel tracks, which give each lane a subtle longitudinal
        # structure that lane-marking edge detection has to cope with.
        for centre in road.lane_centres_m:
            for side in (-0.8, 0.8):
                cx = int(round((centre + side + half_w) * ppm))
                if 0 <= cx < w:
                    band = int(max(2, 0.35 * ppm))
                    lo, hi = max(0, cx - band), min(w, cx + band)
                    plan[:, lo:hi] += 5.0

        marking = np.array(_rgb_to_bgr(road.marking_rgb), dtype=np.float32)
        mark_w_px = max(1, int(round(0.15 * ppm)))

        def draw_line_world(x_m: float, dashed: bool) -> None:
            cx = int(round((x_m + half_w) * ppm))
            if not 0 <= cx < w:
                return
            lo, hi = max(0, cx - mark_w_px), min(w, cx + mark_w_px + 1)
            if not dashed:
                plan[:, lo:hi] = marking
                return
            # Anchor the dash phase to world y = 0 rather than to the top of
            # the plan, so a mark occupies y in [k * pitch, k * pitch + length]
            # for integer k. That makes the on-screen pitch predictable from
            # the configuration, which is what lets a clip be used as a
            # calibration exercise with a known reference distance.
            k0 = int(np.floor((y0 - road.marking_length_m) / road.marking_pitch_m))
            k1 = int(np.ceil(y1 / road.marking_pitch_m))
            for k in range(k0, k1 + 1):
                mark_y0 = k * road.marking_pitch_m
                mark_y1 = mark_y0 + road.marking_length_m
                # Plan row 0 is the far end, so the larger world y is the
                # smaller row index.
                r0 = int(round((y1 - mark_y1) * ppm))
                r1 = int(round((y1 - mark_y0) * ppm))
                r0, r1 = max(0, r0), min(h, r1)
                if r1 > r0:
                    plan[r0:r1, lo:hi] = marking

        bands = road.lane_bands_m()
        edges = sorted({round(e, 4) for band in bands for e in band})
        for i, x in enumerate(edges):
            is_outer = i == 0 or i == len(edges) - 1
            is_centre = abs(x) < 1e-6
            draw_line_world(x, dashed=not (is_outer or is_centre))
        if any(abs(x) < 1e-6 for x in edges):
            # Double centre line separating the two carriageways.
            draw_line_world(-0.18, dashed=False)
            draw_line_world(0.18, dashed=False)

        plan_bgr = np.clip(plan, 0, 255).astype(np.uint8)
        plan_bgr = cv2.GaussianBlur(plan_bgr, (3, 3), 0.6)

        # Plan pixel (col, row) -> world (x_m, y_m); row 0 is the far end.
        plan_to_world = np.array(
            [[1.0 / ppm, 0.0, -half_w], [0.0, -1.0 / ppm, y1], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return plan_bgr, plan_to_world

    def _paint(
        self,
        vehicles: list[_Vehicle],
        time_s: float,
        shake_px: tuple[float, float],
        frame_index: int,
    ) -> UInt8Array:
        cfg = self.config
        ss = self._ss
        canvas = self._road_bgr.copy()

        order: list[tuple[_Vehicle, float]] = []
        for v in vehicles:
            x_m, y_m = v.world_xy_m()
            order.append((v, self._camera_depth_m(x_m, y_m)))
        order.sort(key=lambda item: -item[1])

        if cfg.lighting.cast_shadows:
            self._paint_shadows(canvas, [v for v, _ in order])
        if cfg.lighting.headlights:
            self._paint_headlights(canvas, [v for v, _ in order])

        shutter = cfg.atmosphere.shutter_s
        for v, _ in order:
            if shutter > 0.0:
                self._paint_vehicle_blurred(canvas, v, shutter)
            else:
                self._paint_vehicle(canvas, v)

        if ss > 1:
            canvas = cv2.resize(
                canvas, (cfg.width_px, cfg.height_px), interpolation=cv2.INTER_AREA
            )

        if shake_px != (0.0, 0.0):
            m = np.array(
                [[1.0, 0.0, shake_px[0]], [0.0, 1.0, shake_px[1]]], dtype=np.float64
            )
            canvas = cv2.warpAffine(
                canvas,
                m,
                (cfg.width_px, cfg.height_px),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )

        canvas = self._apply_atmosphere(canvas, frame_index)
        return canvas

    def _to_ss(
        self, pts_px: FloatArray, origin_ss: tuple[int, int] = (0, 0)
    ) -> Int32Array:
        """Output-resolution points to supersampled integer canvas coordinates.

        ``origin_ss`` shifts the result into the frame of a sub-array of the
        canvas, which is what lets the shutter integration below work inside a
        vehicle-sized region instead of over the whole frame.
        """
        pts = np.round(pts_px * self._ss).astype(np.int32)
        if origin_ss != (0, 0):
            pts = pts - np.array(origin_ss, dtype=np.int32)
        return pts

    def _paint_shadows(self, canvas: UInt8Array, vehicles: list[_Vehicle]) -> None:
        """Darken the road inside each shadow polygon.

        Multiplying all three channels scales HSV value while leaving hue and
        saturation untouched; the small additive tint then models sky
        illumination, which desaturates the shadow slightly and shifts it towards
        blue. Both effects are what S5's three thresholds are looking for.
        """
        lig = self.config.lighting
        mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        for v in vehicles:
            poly_world = self._shadow_polygon_world(v)
            poly_px = self._project(poly_world)
            if not np.isfinite(poly_px).all():
                continue
            cv2.fillConvexPoly(mask, self._to_ss(poly_px).reshape(-1, 1, 2), 255)
        if not mask.any():
            return
        if lig.shadow_softness_px > 0:
            k = lig.shadow_softness_px * self._ss
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.6, k * 0.5))

        alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
        gain_rgb = np.asarray(lig.shadow_channel_gain_rgb, dtype=np.float32)
        gain_bgr = gain_rgb[::-1] * lig.shadow_value_scale
        shadowed = canvas.astype(np.float32) * gain_bgr[None, None, :]
        blended = canvas.astype(np.float32) * (1.0 - alpha) + shadowed * alpha
        np.copyto(canvas, np.clip(blended, 0, 255).astype(np.uint8))

    def _paint_headlights(self, canvas: UInt8Array, vehicles: list[_Vehicle]) -> None:
        """Additive light pools on the road ahead of each vehicle.

        Headlight pools move like objects and saturate the value channel, which
        is exactly why night needs its own parameter preset rather than a
        re-tuning of the daytime one (trap 12). Rendering them properly is what
        makes that argument demonstrable instead of asserted.
        """
        lig = self.config.lighting
        glow = np.zeros(canvas.shape[:2], dtype=np.float32)
        for v in vehicles:
            reach = lig.headlight_reach_m
            half_w = v.spec.width_m * 0.9
            local = np.array(
                [
                    [-half_w, v.spec.length_m / 2.0, 0.0],
                    [half_w, v.spec.length_m / 2.0, 0.0],
                    [half_w * 1.9, v.spec.length_m / 2.0 + reach, 0.0],
                    [-half_w * 1.9, v.spec.length_m / 2.0 + reach, 0.0],
                ],
                dtype=np.float64,
            )
            pool_px = self._project(self._local_to_world(local, v))
            if not np.isfinite(pool_px).all():
                continue
            layer = np.zeros_like(glow)
            cv2.fillConvexPoly(layer, self._to_ss(pool_px).reshape(-1, 1, 2), 1.0)
            glow += layer
        if not glow.any():
            return
        glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=4.0 * self._ss)
        boosted = canvas.astype(np.float32) + glow[:, :, None] * 190.0 * lig.headlight_gain
        np.copyto(canvas, np.clip(boosted, 0, 255).astype(np.uint8))

    def _paint_vehicle_blurred(
        self, canvas: UInt8Array, v: _Vehicle, shutter_s: float
    ) -> None:
        """Render a vehicle as the average of sub-poses across the exposure.

        This is a real shutter integration, not a post-hoc directional blur, so
        the silhouette stretches in the direction of travel and the bounding box
        - and therefore the ground point - acquires exactly the bias trap 11
        warns about. Measuring that bias against ground truth is the point.

        The integration runs inside the swept bounding box of the vehicle rather
        than over the whole canvas. Averaging full frames would cost the frame
        area times the sample count times the vehicle count per frame, which for
        a supersampled 720p scene with a dozen vehicles is hundreds of megabytes
        of copying per frame - slow enough to make the scenario unusable.
        """
        n = max(2, self.config.atmosphere.shutter_samples)
        offsets = np.linspace(-0.5, 0.5, n) * shutter_s * v.speed_mps
        roi = self._swept_roi_ss(v, float(offsets[0]), float(offsets[-1]))
        if roi is None:
            return
        x0, y0, x1, y1 = roi

        base = canvas[y0:y1, x0:x1].copy()
        layer = np.empty_like(base)
        acc = np.zeros(base.shape, dtype=np.float32)
        for off in offsets:
            layer[:] = base
            self._paint_vehicle(layer, v, s_offset_m=float(off), origin_ss=(x0, y0))
            acc += layer
        canvas[y0:y1, x0:x1] = np.clip(acc / n, 0, 255).astype(np.uint8)

    def _swept_roi_ss(
        self, v: _Vehicle, s_min_m: float, s_max_m: float, pad_px: int = 3
    ) -> tuple[int, int, int, int] | None:
        """Supersampled canvas box covering a vehicle over the whole exposure."""
        corners = np.vstack(
            [
                self._project(self._cuboid_world(v, v.s_m + s_min_m)),
                self._project(self._cuboid_world(v, v.s_m + s_max_m)),
            ]
        )
        if not np.isfinite(corners).all():
            return None
        pts = self._to_ss(corners)
        x0 = max(0, int(pts[:, 0].min()) - pad_px)
        y0 = max(0, int(pts[:, 1].min()) - pad_px)
        x1 = min(self._w_ss, int(pts[:, 0].max()) + pad_px + 1)
        y1 = min(self._h_ss, int(pts[:, 1].max()) + pad_px + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return (x0, y0, x1, y1)

    def _paint_vehicle(
        self,
        canvas: UInt8Array,
        v: _Vehicle,
        s_offset_m: float = 0.0,
        origin_ss: tuple[int, int] = (0, 0),
    ) -> None:
        """Draw one cuboid vehicle with its visible faces and surface detail."""
        s_m = v.s_m + s_offset_m
        body = np.array(_rgb_to_bgr(v.body_rgb), dtype=np.float32)
        night = self.config.lighting.night
        exposure = 0.42 if night else 1.0

        faces = self._visible_faces(v, s_m)
        for name, quad_local, shade in faces:
            quad_px = self._project(self._local_to_world(quad_local, v, s_m))
            if not np.isfinite(quad_px).all():
                continue
            colour = np.clip(body * shade * exposure, 0, 255)
            cv2.fillConvexPoly(
                canvas,
                self._to_ss(quad_px, origin_ss).reshape(-1, 1, 2),
                (float(colour[0]), float(colour[1]), float(colour[2])),
                lineType=cv2.LINE_AA,
            )
            for patch_local, patch_bgr in self._face_details(v, name, quad_local):
                patch_px = self._project(self._local_to_world(patch_local, v, s_m))
                if not np.isfinite(patch_px).all():
                    continue
                pc = np.clip(np.array(patch_bgr, dtype=np.float32) * exposure, 0, 255)
                cv2.fillConvexPoly(
                    canvas,
                    self._to_ss(patch_px, origin_ss).reshape(-1, 1, 2),
                    (float(pc[0]), float(pc[1]), float(pc[2])),
                    lineType=cv2.LINE_AA,
                )

    def _visible_faces(
        self, v: _Vehicle, s_m: float
    ) -> list[tuple[str, FloatArray, float]]:
        """Backface-cull the cuboid and return visible faces, far face first.

        Culling by the sign of the face normal against the viewing ray is the
        textbook test and costs nothing here; drawing back-to-front within a
        vehicle then gives correct overlap without a depth buffer.
        """
        hw = v.spec.width_m / 2.0
        hl = v.spec.length_m / 2.0
        ht = v.spec.height_m
        defs: list[tuple[str, FloatArray, FloatArray, float]] = [
            (
                "roof",
                np.array([[-hw, -hl, ht], [hw, -hl, ht], [hw, hl, ht], [-hw, hl, ht]]),
                np.array([0.0, 0.0, 1.0]),
                1.00,
            ),
            (
                "front",
                np.array([[-hw, hl, 0.0], [hw, hl, 0.0], [hw, hl, ht], [-hw, hl, ht]]),
                np.array([0.0, 1.0, 0.0]),
                0.70,
            ),
            (
                "rear",
                np.array([[-hw, -hl, 0.0], [hw, -hl, 0.0], [hw, -hl, ht], [-hw, -hl, ht]]),
                np.array([0.0, -1.0, 0.0]),
                0.76,
            ),
            (
                "left",
                np.array([[-hw, -hl, 0.0], [-hw, hl, 0.0], [-hw, hl, ht], [-hw, -hl, ht]]),
                np.array([-1.0, 0.0, 0.0]),
                0.86,
            ),
            (
                "right",
                np.array([[hw, -hl, 0.0], [hw, hl, 0.0], [hw, hl, ht], [hw, -hl, ht]]),
                np.array([1.0, 0.0, 0.0]),
                0.82,
            ),
        ]
        cam_centre = np.array(
            [self.camera.x_m, self.camera.y_m, self.camera.height_m], dtype=np.float64
        )
        visible: list[tuple[str, FloatArray, float, float]] = []
        for name, quad_local, normal_local, shade in defs:
            quad_world = self._local_to_world(quad_local.astype(np.float64), v, s_m)
            normal_world = self._local_to_world(
                np.vstack([np.zeros(3), normal_local]).astype(np.float64), v, s_m
            )
            normal_dir = normal_world[1] - normal_world[0]
            centre = quad_world.mean(axis=0)
            if float(np.dot(normal_dir, cam_centre - centre)) <= 0.0:
                continue
            depth = float(np.linalg.norm(cam_centre - centre))
            visible.append((name, quad_local.astype(np.float64), shade, depth))
        visible.sort(key=lambda item: -item[3])
        return [(n, q, s) for n, q, s, _ in visible]

    def _face_details(
        self, v: _Vehicle, face: str, quad_local: FloatArray
    ) -> list[tuple[FloatArray, tuple[float, float, float]]]:
        """Windows, lights, wheels and livery, in the vehicle's own frame.

        All of it is geometry attached to the body, so it translates with the
        vehicle exactly. That is what makes block matching in S8 measure real
        texture displacement instead of matching a pattern that is nailed to the
        image.
        """
        hw = v.spec.width_m / 2.0
        hl = v.spec.length_m / 2.0
        ht = v.spec.height_m
        glass = (46.0, 42.0, 38.0)
        dark = (24.0, 24.0, 26.0)
        night = self.config.lighting.night
        lamp = (218.0, 228.0, 240.0) if night else (188.0, 196.0, 206.0)
        tail = (48.0, 48.0, 196.0)
        out: list[tuple[FloatArray, tuple[float, float, float]]] = []
        cab = float(np.clip(v.spec.cabin_frac, 0.05, 0.95))

        if face in ("front", "rear"):
            y = hl if face == "front" else -hl
            z0, z1 = ht * 0.45, ht * 0.92
            two_wheeler = v.spec.vehicle_class is VehicleClass.TWO_WHEELER
            out.append(
                (
                    np.array(
                        [
                            [-hw * 0.86, y, z0],
                            [hw * 0.86, y, z0],
                            [hw * 0.86, y, z1],
                            [-hw * 0.86, y, z1],
                        ]
                    ),
                    (52.0, 48.0, 46.0) if two_wheeler else glass,
                )
            )
            light = lamp if face == "front" else tail
            for sx in (-1.0, 1.0):
                cx = sx * hw * 0.66
                out.append(
                    (
                        np.array(
                            [
                                [cx - hw * 0.2, y, ht * 0.16],
                                [cx + hw * 0.2, y, ht * 0.16],
                                [cx + hw * 0.2, y, ht * 0.30],
                                [cx - hw * 0.2, y, ht * 0.30],
                            ]
                        ),
                        light,
                    )
                )
        elif face in ("left", "right"):
            x = -hw if face == "left" else hw
            z0, z1 = ht * 0.50, ht * 0.88
            if v.spec.vehicle_class is not VehicleClass.TWO_WHEELER:
                out.append(
                    (
                        np.array(
                            [
                                [x, -hl * cab, z0],
                                [x, hl * cab, z0],
                                [x, hl * cab, z1],
                                [x, -hl * cab, z1],
                            ]
                        ),
                        glass,
                    )
                )
            else:
                # A rider, not a cabin: a dark upper mass over the rear half.
                out.append(
                    (
                        np.array(
                            [
                                [x, -hl * 0.55, ht * 0.55],
                                [x, hl * 0.25, ht * 0.55],
                                [x, hl * 0.25, ht],
                                [x, -hl * 0.55, ht],
                            ]
                        ),
                        (52.0, 48.0, 46.0),
                    )
                )
            wheel_r = min(0.34, ht * 0.24)
            for sy in (-0.68, 0.68):
                cy = sy * hl
                out.append(
                    (
                        np.array(
                            [
                                [x, cy - wheel_r, 0.0],
                                [x, cy + wheel_r, 0.0],
                                [x, cy + wheel_r, wheel_r * 1.5],
                                [x, cy - wheel_r, wheel_r * 1.5],
                            ]
                        ),
                        dark,
                    )
                )
            rng = np.random.default_rng(v.livery_seed ^ (0 if face == "left" else 1))
            for _ in range(v.spec.texture_amp):
                y0 = float(rng.uniform(-hl * 0.85, hl * 0.55))
                y1 = y0 + float(rng.uniform(0.25, 0.9))
                zz0 = float(rng.uniform(0.12, 0.42)) * ht
                zz1 = zz0 + float(rng.uniform(0.06, 0.2)) * ht
                shade = float(rng.uniform(0.55, 1.35))
                base = np.array(_rgb_to_bgr(v.body_rgb), dtype=np.float64) * shade
                out.append(
                    (
                        np.array([[x, y0, zz0], [x, y1, zz0], [x, y1, zz1], [x, y0, zz1]]),
                        (float(base[0]), float(base[1]), float(base[2])),
                    )
                )
        elif face == "roof":
            out.append(
                (
                    np.array(
                        [
                            [-hw * 0.72, -hl * cab * 0.9, ht],
                            [hw * 0.72, -hl * cab * 0.9, ht],
                            [hw * 0.72, hl * cab * 0.9, ht],
                            [-hw * 0.72, hl * cab * 0.9, ht],
                        ]
                    ),
                    tuple(float(c) for c in np.array(_rgb_to_bgr(v.body_rgb)) * 1.06),  # type: ignore[misc]
                )
            )
        return out

    def _apply_atmosphere(self, canvas: UInt8Array, frame_index: int) -> UInt8Array:
        """Rain streaks, contrast loss and sensor noise, applied last.

        Order matters: these model the atmosphere and the sensor, both of which
        act after the scene has formed, so they are applied after downsampling
        and after shake rather than being baked into the scene.
        """
        atm = self.config.atmosphere
        out = canvas.astype(np.float32)

        if atm.rain_intensity > 0.0:
            rng = np.random.default_rng(self.config.seed * 7919 + frame_index)
            h, w = canvas.shape[:2]
            n = int(atm.rain_intensity * w * h / 4200.0)
            streaks = np.zeros((h, w), dtype=np.float32)
            xs = rng.integers(0, w, size=n)
            ys = rng.integers(0, h, size=n)
            length = atm.rain_streak_len_px
            lean = int(length * 0.28)
            for x, y in zip(xs, ys, strict=True):
                cv2.line(
                    streaks,
                    (int(x), int(y)),
                    (int(x + lean), int(min(h - 1, y + length))),
                    float(rng.uniform(0.35, 1.0)),
                    1,
                )
            streaks = cv2.GaussianBlur(streaks, (0, 0), 0.8)
            out += streaks[:, :, None] * 105.0

        if atm.contrast_loss > 0.0:
            grey_level = out.mean()
            out = out * (1.0 - atm.contrast_loss) + grey_level * atm.contrast_loss

        if atm.sensor_noise_sigma > 0.0:
            rng = np.random.default_rng(self.config.seed * 104729 + frame_index)
            out += rng.normal(0.0, atm.sensor_noise_sigma, size=out.shape)

        return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Video output
# --------------------------------------------------------------------------


def write_video(
    path: str | Path,
    frames: Iterator[UInt8Array],
    fps: float,
    size_px: tuple[int, int],
    codec: str = "mp4v",
    declared_fps: float | None = None,
) -> int:
    """Write BGR frames to a video file, returning the number written.

    ``declared_fps`` deliberately lets the container metadata disagree with the
    true frame rate. Trap 7 is that container fps often lies, and every speed in
    the system divides by the frame interval - so being able to *manufacture* a
    lying container is what turns S1's timestamp measurement from a claim into a
    tested behaviour.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
    writer = cv2.VideoWriter(
        str(p), fourcc, float(declared_fps if declared_fps else fps), size_px
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"OpenCV could not open a video writer for {p} with codec {codec!r}. "
            f"Try codec='MJPG' with a .avi extension."
        )
    count = 0
    try:
        for frame in frames:
            if frame.shape[1] != size_px[0] or frame.shape[0] != size_px[1]:
                raise ValueError(
                    f"frame {count} is {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {size_px[0]}x{size_px[1]}"
                )
            writer.write(frame)
            count += 1
    finally:
        writer.release()
    if count == 0:
        raise ValueError("no frames were written")
    return count
