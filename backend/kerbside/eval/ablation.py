"""The ablation studies.

Each preset answers a question the project actually has, rather than sweeping a
parameter because it exists. The question is stored alongside the sweep and
shown above the chart, because a chart without its question is a chart nobody
can act on.

Every study runs the whole pipeline over a clip per setting, which is slow and
unavoidable: the stages are stateful, so a background model built under one
learning rate cannot be reused to evaluate another, and reusing it would produce
a comparison of two things that were never run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kerbside.data.renderer import TrafficRenderer
from kerbside.data.scenarios import build_scene, scenario_names
from kerbside.eval.metrics import (
    VisibilityGate,
    count_accuracy,
    match_tracks_to_vehicles,
    score_detection,
    score_lane_and_class,
    score_speed,
)
from kerbside.stages.pipeline import Pipeline

__all__ = ["AblationPreset", "ABLATIONS", "run_ablation", "evaluate_once"]

GATE = VisibilityGate(min_visible_px=300, min_y_m=18.0, max_y_m=80.0)
"""The working range every ablation is scored over, quoted with every result.

Beyond about 80 m a vehicle is a handful of pixels and a one-pixel ground-point
error is several metres of road, so a figure that includes it measures the
sampling limit rather than the parameter under study."""


@dataclass(frozen=True, slots=True)
class AblationPreset:
    """One study: an axis, the settings along it, and the question it answers."""

    key: str
    label: str
    question: str
    stage_id: str
    param: str
    values: tuple[Any, ...]
    scenario: str = "free_flow"
    extra: dict[str, dict[str, Any]] = field(default_factory=dict)


ABLATIONS: dict[str, AblationPreset] = {
    "shadow": AblationPreset(
        key="shadow",
        label="Shadow suppression on/off",
        question=(
            "What does HSV shadow suppression buy? Measured as count error, "
            "false-merge rate and speed MAE, on a clip with a low sun."
        ),
        stage_id="s5_shadow",
        param="enabled",
        values=(False, True),
        scenario="long_shadows",
    ),
    "background": AblationPreset(
        key="background",
        label="Background model",
        question=(
            "Running average, temporal median or three-frame difference - which "
            "counts most accurately, and which ghosts?"
        ),
        stage_id="s3_background",
        param="model",
        values=("running_average", "temporal_median", "three_frame", "reference"),
    ),
    "alpha": AblationPreset(
        key="alpha",
        label="Running-average learning rate",
        question=(
            "The model's time constant is 1/alpha frames. How fast can it adapt "
            "before it starts absorbing the vehicles it is meant to detect?"
        ),
        stage_id="s3_background",
        param="alpha",
        values=(0.005, 0.01, 0.02, 0.05, 0.1),
    ),
    "difference": AblationPreset(
        key="difference",
        label="Difference in colour or luma",
        question=(
            "A red vehicle differs from asphalt by about one grey level and by "
            "94 in its red channel. How much does that cost a greyscale "
            "difference in practice?"
        ),
        stage_id="s3_background",
        param="difference_mode",
        values=("grey", "max_channel", "euclidean"),
    ),
    "morphology": AblationPreset(
        key="morphology",
        label="Perspective-scaled structuring elements",
        question=(
            "A half-metre element is 37 px across in the near field and 3 px at "
            "the horizon. What does one fixed size cost?"
        ),
        stage_id="s4_foreground",
        param="perspective_scaled",
        values=(False, True),
    ),
    "connectivity": AblationPreset(
        key="connectivity",
        label="Connectivity: 4 vs 8 vs m",
        question=(
            "4-connectivity fragments a vehicle whose mask narrows to a diagonal. "
            "8 and m are provably identical in their components - so what "
            "actually differs is 4 against the other two."
        ),
        stage_id="s6_blobs",
        param="connectivity",
        values=("4", "8", "m"),
    ),
    "search": AblationPreset(
        key="search",
        label="Block-matching search algorithm",
        question=(
            "Exhaustive search is exact and 30 times dearer. Is a fast search "
            "good enough for a speed measurement, as opposed to for a codec?"
        ),
        stage_id="s8_speed",
        param="search",
        values=("exhaustive", "two_d_log", "three_step"),
    ),
    "metric": AblationPreset(
        key="metric",
        label="Tracking distance measure",
        question=(
            "Euclidean, city-block and chessboard differ by up to 41% on a "
            "diagonal. Does the choice change how many vehicles are counted?"
        ),
        stage_id="s7_track",
        param="metric",
        values=("euclidean", "city_block", "chessboard"),
    ),
    "scenario": AblationPreset(
        key="scenario",
        label="Scenario robustness",
        question=(
            "How does the whole pipeline hold up across free flow, congestion, "
            "night, rain, occlusion, shake, long shadows and motion blur?"
        ),
        stage_id="",
        param="scenario",
        values=tuple(scenario_names()),
    ),
}


def evaluate_once(
    scenario: str,
    duration_s: float,
    seed: int = 4242,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run the pipeline once over a clip and score everything against truth."""
    scene = build_scene(scenario, seed=seed, duration_s=duration_s)
    scene = scene.__class__(**{**scene.__dict__, "supersample": 1}) if False else scene
    from dataclasses import replace as _replace

    scene = _replace(scene, supersample=1)
    renderer = TrafficRenderer(scene)
    frames, gts = [], []
    for frame, gt in renderer.render_clip():
        frames.append(frame)
        gts.append(gt)
    truth = renderer.ground_truth(gts)
    calibration = renderer.calibration()

    pipe = Pipeline(frame_interval_s=1.0 / scene.fps, calibration=calibration)
    for stage_id, params in (overrides or {}).items():
        pipe.configure(stage_id, **params)

    started = time.perf_counter()
    blobs, tracks, estimates = [], [], []
    for output in pipe.run(frames):
        blobs.append(output.context.blobs)
        tracks.append(output.context.track_snapshots)
        estimates.append(dict(pipe.speed.estimates))
    elapsed = time.perf_counter() - started

    skip = 20
    detection = score_detection(blobs, truth, GATE, calibration, skip_frames=skip).summary()
    assignment = match_tracks_to_vehicles(tracks, truth, skip_frames=skip)
    speed = score_speed(tracks, estimates, truth, assignment, skip_frames=skip).summary()
    lane_class = score_lane_and_class(tracks, truth, assignment, skip_frames=skip)
    counting = count_accuracy(
        pipe.track.total_count, truth, GATE, pipe.track.counts_by_lane()
    ).summary()

    return {
        "detection": detection,
        "speed": speed,
        "lane_class": {k: v for k, v in lane_class.items() if k != "class_confusion"},
        "class_confusion": lane_class["class_confusion"],
        "counting": counting,
        "fps": round(len(frames) / max(elapsed, 1e-6), 2),
        "frames": len(frames),
    }


def run_ablation(key: str, duration_s: float = 6.0, seed: int = 4242) -> dict[str, Any]:
    """Sweep one axis and return a table the interface can chart directly."""
    if key not in ABLATIONS:
        raise KeyError(
            f"unknown ablation {key!r}; available: {', '.join(sorted(ABLATIONS))}"
        )
    preset = ABLATIONS[key]
    rows: list[dict[str, Any]] = []

    for value in preset.values:
        if preset.param == "scenario":
            result = evaluate_once(str(value), duration_s, seed)
            setting = str(value)
        else:
            result = evaluate_once(
                preset.scenario, duration_s, seed, {preset.stage_id: {preset.param: value}}
            )
            setting = str(value)
        rows.append(
            {
                "setting": setting,
                "count_accuracy": round(result["counting"]["accuracy"], 4),
                "counted": result["counting"]["counted"],
                "expected": result["counting"]["expected"],
                "recall": round(result["detection"]["recall"], 4),
                "precision": round(result["detection"]["precision"], 4),
                "false_merge_rate": round(result["detection"]["false_merge_rate"], 4),
                "frame_agreement": round(result["detection"]["frame_count_agreement"], 4),
                "speed_mae_kmph": _round(result["speed"]["track_mae_kmph"]),
                "blockmatch_mae_kmph": _round(result["speed"]["blockmatch_mae_kmph"]),
                "lane_accuracy": round(result["lane_class"]["lane_accuracy"], 4),
                "class_accuracy": round(result["lane_class"]["class_accuracy"], 4),
                "fps": result["fps"],
            }
        )

    return {
        "key": preset.key,
        "label": preset.label,
        "question": preset.question,
        "axis": preset.param,
        "scenario": preset.scenario if preset.param != "scenario" else "all",
        "gate": GATE.to_dict(),
        "duration_s": duration_s,
        "rows": rows,
    }


def _round(value: float) -> float | None:
    return None if value is None or not np.isfinite(value) else round(float(value), 3)
