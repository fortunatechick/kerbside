"""Render scenario clips and their ground truth to disk.

    python -m kerbside.data.make_clips --scenario free_flow
    python -m kerbside.data.make_clips --all --duration 20

Each run writes three things next to each other:

* ``data/clips/<name>.mp4``  the video the pipeline will be pointed at;
* ``data/gt/<name>.json``    exact per-frame ground truth for every vehicle;
* ``data/gt/<name>.png``     a mid-clip still, for the README and for a quick
  visual check that the scenario is what it claims to be.

Clips are small and cheap to regenerate, so they are treated as build output
rather than as committed assets - which also keeps the repository free of large
binaries while leaving every number in the report reproducible from a seed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from kerbside.core.types import GTFrame
from kerbside.data.renderer import TrafficRenderer, write_video
from kerbside.data.scenarios import build_scene, scenario_info, scenario_names

DEFAULT_CLIP_DIR = Path("data/clips")
DEFAULT_GT_DIR = Path("data/gt")


def render_one(
    scenario: str,
    clip_dir: Path = DEFAULT_CLIP_DIR,
    gt_dir: Path = DEFAULT_GT_DIR,
    *,
    duration_s: float | None = None,
    fps: float | None = None,
    seed: int | None = None,
    width_px: int | None = None,
    height_px: int | None = None,
    supersample: int | None = None,
    codec: str = "mp4v",
    extension: str = ".mp4",
    declared_fps: float | None = None,
    quiet: bool = False,
) -> tuple[Path, Path]:
    """Render one scenario, returning the clip and ground-truth paths."""
    scene = build_scene(
        scenario,
        seed=seed,
        duration_s=duration_s,
        fps=fps,
        width_px=width_px,
        height_px=height_px,
        supersample=supersample,
    )
    started = time.perf_counter()
    renderer = TrafficRenderer(scene)

    frames = []
    gts: list[GTFrame] = []
    for frame, gt in renderer.render_clip():
        frames.append(frame)
        gts.append(gt)

    clip_path = clip_dir / f"{scenario}{extension}"
    written = write_video(
        clip_path,
        iter(frames),
        scene.fps,
        (scene.width_px, scene.height_px),
        codec=codec,
        declared_fps=declared_fps,
    )

    ground_truth = renderer.ground_truth(gts)
    gt_path = gt_dir / f"{scenario}.json"
    ground_truth.save(gt_path)

    still_path = gt_dir / f"{scenario}.png"
    still_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(still_path), frames[len(frames) // 2])

    if not quiet:
        elapsed = time.perf_counter() - started
        counts = ground_truth.counts_by_lane()
        print(
            f"{scenario:16s} {written:4d} frames  "
            f"{scene.width_px}x{scene.height_px} @ {scene.fps:g} fps  "
            f"{ground_truth.total_vehicles:3d} vehicles  "
            f"lanes {counts}  "
            f"[{elapsed:.1f} s]  -> {clip_path}"
        )
    return clip_path, gt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kerbside.data.make_clips",
        description="Render synthetic traffic clips with exact ground truth.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=scenario_names(),
        help="scenario to render; repeatable. Defaults to free_flow.",
    )
    parser.add_argument("--all", action="store_true", help="render every scenario")
    parser.add_argument("--duration", type=float, default=None, help="clip length in seconds")
    parser.add_argument("--fps", type=float, default=None, help="true frame rate")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--width", type=int, default=None, help="frame width in pixels")
    parser.add_argument("--height", type=int, default=None, help="frame height in pixels")
    parser.add_argument(
        "--supersample",
        type=int,
        default=None,
        help="render scale for anti-aliasing (1 is fastest, 2 is the default)",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="FourCC video codec; use MJPG with --ext .avi if mp4v is unavailable",
    )
    parser.add_argument("--ext", default=".mp4", help="clip file extension")
    parser.add_argument(
        "--declared-fps",
        type=float,
        default=None,
        help=(
            "frame rate to write into the container metadata, which may "
            "deliberately differ from the true rate so that S1's timestamp "
            "measurement has something to catch (trap 7)"
        ),
    )
    parser.add_argument("--clip-dir", type=Path, default=DEFAULT_CLIP_DIR)
    parser.add_argument("--gt-dir", type=Path, default=DEFAULT_GT_DIR)
    parser.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = parser.parse_args(argv)

    if args.list:
        for name in scenario_names():
            info = scenario_info(name)
            print(f"{name:16s} {info.label}\n{'':16s} {info.description}")
            print(f"{'':16s} exercises: {info.exercises}\n")
        return 0

    if args.all:
        scenarios = list(scenario_names())
    elif args.scenario:
        scenarios = args.scenario
    else:
        scenarios = ["free_flow"]

    for name in scenarios:
        try:
            render_one(
                name,
                clip_dir=args.clip_dir,
                gt_dir=args.gt_dir,
                duration_s=args.duration,
                fps=args.fps,
                seed=args.seed,
                width_px=args.width,
                height_px=args.height,
                supersample=args.supersample,
                codec=args.codec,
                extension=args.ext,
                declared_fps=args.declared_fps,
            )
        except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not swallow
            print(f"{name}: FAILED - {exc}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
