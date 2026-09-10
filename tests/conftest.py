"""Shared fixtures.

Rendering is the expensive part of the test suite, so the reference clips are
built once per session at a small size and reused. They stay small deliberately:
the Phase 1 gate is about geometric exactness, which a short clip proves just as
well as a long one, and the accuracy gates that need volume live in the
evaluation suite rather than in unit tests.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from kerbside.core.types import GTFrame
from kerbside.data.renderer import SceneConfig, TrafficRenderer
from kerbside.data.scenarios import build_scene


def _render(config: SceneConfig) -> tuple[TrafficRenderer, list, list[GTFrame]]:
    renderer = TrafficRenderer(config)
    frames = []
    gts = []
    for frame, gt in renderer.render_clip():
        frames.append(frame)
        gts.append(gt)
    return renderer, frames, gts


@pytest.fixture(scope="session")
def free_flow_scene() -> SceneConfig:
    """A short free-flow clip at reduced supersampling, for speed."""
    return replace(
        build_scene("free_flow", seed=4242, duration_s=3.0),
        supersample=1,
    )


@pytest.fixture(scope="session")
def free_flow_clip(free_flow_scene: SceneConfig):
    return _render(free_flow_scene)


@pytest.fixture(scope="session")
def shaken_clip():
    scene = replace(
        build_scene("camera_shake", seed=99, duration_s=2.0), supersample=1
    )
    return _render(scene)


@pytest.fixture(scope="session")
def occluded_clip():
    scene = replace(
        build_scene("heavy_occlusion", seed=7, duration_s=3.0), supersample=1
    )
    return _render(scene)
