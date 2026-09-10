"""Scenario presets for the synthetic renderer.

Each preset is a complete :class:`~kerbside.data.renderer.SceneConfig` chosen to
exercise a specific failure mode from the traps list, so that the ablation study
in the report is comparing meaningfully different conditions rather than the
same clip with the brightness changed:

===================  =========================================================
``free_flow``        The reference condition. Every hard accuracy gate is
                     stated against this.
``congested``        Car following at saturation: queues, full stops and short
                     headways. Produces the stopped vehicles that make
                     background ghosting (trap 5) appear.
``night``            Low ambient, headlight pools on the road surface and
                     saturated value. A different parameter regime, not a
                     broken pipeline (trap 12).
``rain``             Streaks plus contrast loss - periodic interference the
                     frequency-domain filtering of Unit 2 has something to say
                     about.
``heavy_occlusion``  Dense near-lane traffic hiding the far lane, for the
                     occlusion accuracy target and the failure catalogue.
``camera_shake``     Sinusoidal jitter of known amplitude, so the S2
                     phase-correlation stabiliser can be scored (trap 8).
``long_shadows``     A low sun throwing shadows across lane boundaries - the
                     input the S5 headline result is measured on (trap 2).
``motion_blur``      A long shutter, so the silhouette stretches along the
                     direction of travel and the ground point acquires the bias
                     trap 11 warns about.
===================  =========================================================

The console exposes vehicle density, speed range and shadow strength as
sliders; :func:`build_scene` is where those land, so a user-tweaked scenario is
still a plain :class:`SceneConfig` and still deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from kerbside.core.types import CameraModel, Direction
from kerbside.data.renderer import (
    AtmosphereConfig,
    LightingConfig,
    RoadSpec,
    SceneConfig,
    ShakeConfig,
    TrafficConfig,
)

__all__ = [
    "ScenarioInfo",
    "SCENARIOS",
    "scenario_names",
    "scenario_info",
    "build_scene",
]


DEFAULT_CAMERA = CameraModel(
    width_px=1280, height_px=720, height_m=8.0, tilt_deg=12.0, fov_deg=55.0
)
"""Reference mounting: 8 m up a pole, 12 degrees below horizontal, 55 degree
horizontal field of view. Chosen so the road fills the frame from about 15 m to
the horizon, which is a realistic counting-camera geometry."""


@dataclass(frozen=True, slots=True)
class ScenarioInfo:
    """A preset plus the human-readable reason it exists."""

    name: str
    label: str
    description: str
    exercises: str
    config: SceneConfig


def _base(name: str, **overrides: object) -> SceneConfig:
    cfg = SceneConfig(name=name, camera=DEFAULT_CAMERA)
    return replace(cfg, **overrides)  # type: ignore[arg-type]


_FREE_FLOW = _base("free_flow")

_CONGESTED = _base(
    "congested",
    traffic=TrafficConfig(
        flow_veh_per_hour_per_lane=1750.0,
        desired_speed_kmph=(26.0, 46.0),
        min_spawn_gap_m=7.0,
        accel_max_mps2=1.1,
        decel_comfort_mps2=1.9,
        time_headway_s=1.1,
        jam_gap_m=2.0,
    ),
    duration_s=24.0,
)

_NIGHT = _base(
    "night",
    lighting=LightingConfig(
        ambient=0.30,
        night=True,
        headlights=True,
        cast_shadows=False,
        headlight_gain=0.95,
        headlight_reach_m=18.0,
    ),
    atmosphere=AtmosphereConfig(sensor_noise_sigma=6.0),
    traffic=TrafficConfig(
        flow_veh_per_hour_per_lane=620.0, desired_speed_kmph=(50.0, 74.0)
    ),
)

_RAIN = _base(
    "rain",
    lighting=LightingConfig(
        ambient=0.62,
        cast_shadows=False,
        sun_elevation_deg=70.0,
    ),
    atmosphere=AtmosphereConfig(
        rain_intensity=1.0,
        rain_streak_len_px=26,
        contrast_loss=0.28,
        sensor_noise_sigma=4.0,
    ),
    traffic=TrafficConfig(
        flow_veh_per_hour_per_lane=760.0, desired_speed_kmph=(40.0, 60.0)
    ),
)

_HEAVY_OCCLUSION = _base(
    "heavy_occlusion",
    # A near carriageway running much heavier than the far one: large vehicles
    # in the near lanes hide the far lanes for extended runs, which is how
    # occlusion actually happens on a real oblique view.
    road=RoadSpec(
        lane_centres_m=(-5.25, -1.75, 1.75, 5.25),
        directions=(
            Direction.TOWARD,
            Direction.TOWARD,
            Direction.AWAY,
            Direction.AWAY,
        ),
    ),
    traffic=TrafficConfig(
        flow_veh_per_hour_per_lane=1500.0,
        desired_speed_kmph=(32.0, 52.0),
        min_spawn_gap_m=8.0,
        class_weights={
            "motorcycle": 0.06,
            "hatchback": 0.16,
            "saloon": 0.16,
            "suv": 0.14,
            "panel_van": 0.18,
            "box_truck": 0.16,
            "bus": 0.14,
        },
    ),
    duration_s=24.0,
)

_CAMERA_SHAKE = _base(
    "camera_shake",
    shake=ShakeConfig(amplitude_px=3.6, frequency_hz=1.9, vertical_ratio=0.65),
)

_LONG_SHADOWS = _base(
    "long_shadows",
    lighting=LightingConfig(
        sun_azimuth_deg=188.0,
        sun_elevation_deg=15.0,
        shadow_value_scale=0.48,
        shadow_channel_gain_rgb=(0.93, 1.00, 1.09),
    ),
    traffic=TrafficConfig(
        flow_veh_per_hour_per_lane=1250.0,
        desired_speed_kmph=(42.0, 64.0),
        min_spawn_gap_m=9.0,
    ),
    duration_s=24.0,
)

_MOTION_BLUR = _base(
    "motion_blur",
    atmosphere=AtmosphereConfig(shutter_s=0.020, shutter_samples=7, sensor_noise_sigma=2.5),
    traffic=TrafficConfig(
        flow_veh_per_hour_per_lane=700.0, desired_speed_kmph=(70.0, 95.0)
    ),
)


SCENARIOS: dict[str, ScenarioInfo] = {
    info.name: info
    for info in (
        ScenarioInfo(
            "free_flow",
            "Free flow",
            "Uncongested daytime traffic at the reference camera geometry.",
            "The baseline every hard accuracy gate is stated against.",
            _FREE_FLOW,
        ),
        ScenarioInfo(
            "congested",
            "Congested",
            "Saturated flow with queue formation, short headways and full stops.",
            "Stopped vehicles that get absorbed into a running-average "
            "background and leave a ghost when they move off (trap 5).",
            _CONGESTED,
        ),
        ScenarioInfo(
            "night",
            "Night",
            "Low ambient light with headlight pools cast on the road surface.",
            "Saturated value and moving light pools that behave like objects, "
            "which need their own parameter preset rather than a compromise "
            "one (trap 12).",
            _NIGHT,
        ),
        ScenarioInfo(
            "rain",
            "Rain",
            "Rain streaks with reduced contrast and elevated sensor noise.",
            "Near-periodic interference for frequency-domain filtering, and a "
            "low-contrast difference image for Otsu to cope with.",
            _RAIN,
        ),
        ScenarioInfo(
            "heavy_occlusion",
            "Heavy occlusion",
            "Dense mixed traffic in which large near-lane vehicles hide the far lanes.",
            "The occlusion accuracy target and the catalogue of failure modes "
            "behind it.",
            _HEAVY_OCCLUSION,
        ),
        ScenarioInfo(
            "camera_shake",
            "Camera shake",
            "Sinusoidal pole vibration of a known amplitude, recorded per frame.",
            "Phase-correlation stabilisation in S2, scored against the recorded "
            "offset; without it the whole frame reads as foreground (trap 8).",
            _CAMERA_SHAKE,
        ),
        ScenarioInfo(
            "long_shadows",
            "Long shadows",
            "A low sun throwing vehicle shadows across adjacent lane boundaries.",
            "The S5 headline result: shadows bridging two vehicles into one "
            "blob, which presents as a tracking failure but is a colour "
            "processing failure (trap 2).",
            _LONG_SHADOWS,
        ),
        ScenarioInfo(
            "motion_blur",
            "Motion blur",
            "A long shutter at high speed, integrated across the exposure.",
            "Silhouettes stretched along the direction of travel, biasing the "
            "bounding box and therefore the ground point (trap 11).",
            _MOTION_BLUR,
        ),
    )
}


def scenario_names() -> tuple[str, ...]:
    return tuple(SCENARIOS.keys())


def scenario_info(name: str) -> ScenarioInfo:
    try:
        return SCENARIOS[name]
    except KeyError:
        raise KeyError(
            f"unknown scenario {name!r}; available: {', '.join(sorted(SCENARIOS))}"
        ) from None


def build_scene(
    name: str = "free_flow",
    *,
    seed: int | None = None,
    duration_s: float | None = None,
    fps: float | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
    density: float | None = None,
    speed_kmph_range: tuple[float, float] | None = None,
    shadow_strength: float | None = None,
    supersample: int | None = None,
) -> SceneConfig:
    """Build a scene from a preset with the console's sliders applied.

    ``density`` scales the configured flow (1.0 leaves the preset alone),
    ``speed_kmph_range`` replaces the desired-speed band outright, and
    ``shadow_strength`` runs from 0 (no cast shadows) to 1 (as dark as the
    preset makes them), interpolating the value scale towards 1.0 - no
    darkening - as it falls.

    Every argument is optional and every omitted one leaves the preset
    untouched, so a scenario name alone always reproduces the published preset.
    """
    cfg = scenario_info(name).config

    if width_px is not None or height_px is not None:
        w = width_px if width_px is not None else cfg.width_px
        h = height_px if height_px is not None else cfg.height_px
        cfg = cfg.with_resolution(w, h)
    if seed is not None:
        cfg = replace(cfg, seed=seed)
    if duration_s is not None:
        if duration_s <= 0:
            raise ValueError(f"duration must be positive, got {duration_s} s")
        cfg = replace(cfg, duration_s=duration_s)
    if fps is not None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        cfg = replace(cfg, fps=fps)
    if supersample is not None:
        cfg = replace(cfg, supersample=supersample)

    if density is not None:
        if density < 0:
            raise ValueError(f"density scale must be non-negative, got {density}")
        cfg = replace(
            cfg,
            traffic=replace(
                cfg.traffic,
                flow_veh_per_hour_per_lane=cfg.traffic.flow_veh_per_hour_per_lane * density,
            ),
        )
    if speed_kmph_range is not None:
        lo, hi = speed_kmph_range
        if not 0 < lo <= hi:
            raise ValueError(
                f"speed range must satisfy 0 < low <= high, got {speed_kmph_range}"
            )
        cfg = replace(cfg, traffic=replace(cfg.traffic, desired_speed_kmph=(lo, hi)))
    if shadow_strength is not None:
        s = float(min(max(shadow_strength, 0.0), 1.0))
        base = cfg.lighting.shadow_value_scale
        cfg = replace(
            cfg,
            lighting=replace(
                cfg.lighting,
                cast_shadows=cfg.lighting.cast_shadows and s > 0.0,
                shadow_value_scale=1.0 - (1.0 - base) * s,
            ),
        )
    return cfg
