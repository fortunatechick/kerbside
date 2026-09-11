"""The two constraints nothing else enforces.

Most of this project's rules are the kind a reviewer checks by reading. These
two are not: "no machine learning anywhere in the pipeline" is a property of
every line in `backend/kerbside/`, and "same video and parameters give identical
output" is a property of a run rather than of any function. Both were asserted
in prose and by nothing else until this file existed.
"""

from __future__ import annotations

import ast
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kerbside.data.renderer import TrafficRenderer
from kerbside.data.scenarios import build_scene
from kerbside.stages.pipeline import Pipeline

PACKAGE = Path(__file__).resolve().parents[1] / "backend" / "kerbside"

# Names that would mean a learned model had entered the pipeline. cv2.dnn is
# listed as an attribute path rather than an import because OpenCV is imported
# legitimately everywhere and only that submodule is forbidden.
FORBIDDEN_IMPORTS = {
    "torch",
    "tensorflow",
    "keras",
    "sklearn",
    "scikit_learn",
    "xgboost",
    "lightgbm",
    "onnx",
    "onnxruntime",
    "ultralytics",
    "mediapipe",
}
FORBIDDEN_CALLS = (
    "cv2.dnn",
    "CascadeClassifier",
    "HOGDescriptor",
    "readNetFrom",
    "cv2.ml.",
    "createBackgroundSubtractorKNN",
)

#: The one permitted exception, and the only file allowed to use it. The brief
#: allows OpenCV's mixture-of-Gaussians subtractor as a clearly-labelled
#: reference baseline in the comparison study and nowhere else.
BASELINE_CALL = "createBackgroundSubtractorMOG2"
BASELINE_FILE = "s3_background.py"


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_package_has_files_to_check() -> None:
    """Guards the guard: a glob that matches nothing passes every test below."""
    files = _python_files()
    assert len(files) > 25, f"only found {len(files)} modules under {PACKAGE}"


def test_no_learned_model_is_imported_anywhere_in_the_package() -> None:
    """Constraint 1, enforced rather than asserted.

    Parsed rather than grepped: a comment mentioning PyTorch is fine, and an
    import of it is not, and only the AST can tell those apart.
    """
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                root = name.split(".")[0].lower()
                if root in FORBIDDEN_IMPORTS:
                    offenders.append(f"{path.name}:{node.lineno} imports {name}")
    assert not offenders, "a learned model has entered the pipeline: " + "; ".join(offenders)


def test_no_pretrained_detector_is_called_anywhere_in_the_package() -> None:
    """Haar cascades, HOG descriptors and cv2.dnn all load trained weights."""
    offenders: list[str] = []
    for path in _python_files():
        source = path.read_text(encoding="utf-8")
        # Strip comments and docstrings: this file names every one of these
        # terms in prose, and so does the documentation in the stages.
        tree = ast.parse(source, filename=str(path))
        code = ast.unparse(_strip_docstrings(tree))
        code = re.sub(r"#.*", "", code)
        for needle in FORBIDDEN_CALLS:
            if needle in code:
                offenders.append(f"{path.name} calls {needle}")
    assert not offenders, "a pretrained detector is being used: " + "; ".join(offenders)


def test_the_mog2_baseline_exists_only_where_it_is_permitted() -> None:
    """The one exception, pinned to one file and one clearly-labelled option.

    The brief permits OpenCV's mixture-of-Gaussians subtractor as a reference
    baseline in the comparison study and never on the production path. A test
    is what keeps "never the production path" true a month from now.
    """
    users = [
        path.name
        for path in _python_files()
        if BASELINE_CALL in ast.unparse(_strip_docstrings(ast.parse(path.read_text("utf-8"))))
    ]
    assert users == [BASELINE_FILE], f"the MOG2 baseline leaked into {users}"

    from kerbside.stages.s3_background import BackgroundStage

    stage = BackgroundStage()
    spec = next(p for p in stage.param_specs if p.name == "model")
    assert "mog2_baseline" in spec.choices
    assert stage.param("model") != "mog2_baseline", "the baseline must never be the default"
    # "Clearly labelled" has to mean something a reader sees, not a comment.
    assert "baseline" in spec.description.lower()
    assert "never the default" in spec.description.lower()


def _strip_docstrings(tree: ast.Module) -> ast.Module:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    return tree


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def clip():
    scene = replace(
        build_scene("free_flow", seed=4242, duration_s=2.0),
        supersample=1,
        width_px=320,
        height_px=180,
    )
    renderer = TrafficRenderer(scene)
    frames = [frame for frame, _gt in renderer.render_clip()]
    return renderer, frames


def _run(frames, calibration, fps=25.0):
    pipeline = Pipeline(frame_interval_s=1.0 / fps, calibration=calibration)
    out = []
    for index, frame in enumerate(frames):
        output = pipeline.process_frame(frame, frame_index=index)
        ctx = output.context
        out.append(
            (
                len(ctx.blobs),
                tuple(
                    sorted(
                        round(float(v), 6)
                        for v in ctx.results["s4_foreground"].metrics.values()
                    )
                ),
                tuple(
                    (
                        snapshot.track_id,
                        round(snapshot.speed_kmph_track, 9),
                        round(snapshot.speed_kmph_blockmatch, 9),
                        snapshot.lane_id,
                        snapshot.vehicle_class,
                    )
                    for snapshot in ctx.track_snapshots
                ),
            )
        )
    return out


def test_the_whole_pipeline_is_deterministic(clip) -> None:
    """Constraint 5: same video, same parameters, identical output.

    Not the renderer - that is asserted separately - but the pipeline: blob
    counts, stage metrics, and every per-track speed, lane and class. Anything
    seeded from the clock, iterated over an unordered set, or carrying state
    between runs shows up here as a difference in one of those.
    """
    _renderer, frames = clip
    first = _run(frames, _renderer.calibration())
    second = _run(frames, _renderer.calibration())
    assert len(first) == len(frames)
    for index, (a, b) in enumerate(zip(first, second, strict=True)):
        assert a == b, f"frame {index} differs between two identical runs"


def test_numpy_global_state_does_not_leak_between_runs(clip) -> None:
    """A pipeline that reads the global RNG is deterministic only by luck.

    Seeding NumPy differently either side of a run must change nothing. If it
    does, something is drawing from the global generator rather than from a
    seeded one it owns, and the run is reproducible only until someone else
    draws a random number first.
    """
    _renderer, frames = clip
    calibration = _renderer.calibration()

    np.random.seed(1)
    first = _run(frames[:12], calibration)
    np.random.seed(999_999)
    second = _run(frames[:12], calibration)
    assert first == second, "the pipeline is reading NumPy's global RNG"
