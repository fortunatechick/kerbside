"""Shared scaffolding for the pipeline stages.

A stage is a named, parameterised step that takes the frame context so far,
adds something to it, and reports how long it took and what it produced. The
uniformity is what lets the console render a spine of nine stages, time each
one, and let the user change any parameter and re-run from there forward
without the UI knowing anything about what a particular stage does.

Two things every stage carries beyond its implementation: the syllabus units it
exercises and a one-sentence rationale. Those are surfaced in the inspector, and
they are a course requirement rather than decoration.

Stages that hold state - the background model, the tracker - implement
:meth:`Stage.reset`. Anything cached across frames must be cleared there, or a
parameter change would be evaluated against a model built under the old
parameters and the metric delta shown to the user would be meaningless.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from kerbside.core.registry import ParamSpec
from kerbside.core.types import (
    AnyArray,
    Blob,
    BoolArray,
    Calibration,
    FloatArray,
    FrameType,
    StageResult,
    Track,
    TrackSnapshot,
    UInt8Array,
)

__all__ = ["FrameContext", "Stage", "StageError"]


class StageError(RuntimeError):
    """A stage could not produce a result, with a reason a user can act on.

    Distinct from a programming error on purpose: this is the class the API
    turns into a specific message in the interface. "The parameter set produced
    zero detections" is a fact about the parameters; an IndexError is a fact
    about the code, and the two should not be reported the same way.
    """


@dataclass(slots=True)
class FrameContext:
    """Everything the pipeline knows about the frame currently being processed.

    Stages read what they need and attach what they produce. Passing one mutable
    context rather than threading a dozen return values keeps the stage
    signatures uniform - which is what lets the pipeline run an arbitrary prefix
    of the stages, cache the result, and resume from any point when a parameter
    downstream changes.
    """

    frame_index: int
    time_s: float
    bgr: UInt8Array

    # S1
    grey: FloatArray | None = None
    hsv: FloatArray | None = None
    dt_s: float = 0.0

    # S2
    shake_dx_px: float = 0.0
    shake_dy_px: float = 0.0
    stabilised: bool = False
    stabilisation_confident: bool = True

    # S3
    background_grey: FloatArray | None = None
    background_bgr: FloatArray | None = None
    difference: FloatArray | None = None
    frame_type: FrameType = FrameType.P

    # S4
    foreground: BoolArray | None = None
    threshold_level: float = 0.0

    #: The previous frame's final mask, supplied by the pipeline.
    #: S3 needs it for selective update, and it cannot come from this context -
    #: the context is rebuilt per frame, so ``foreground`` is always None when
    #: S3 runs and a selective update keyed on it would silently never fire.
    previous_mask: BoolArray | None = None

    # S5
    shadow: BoolArray | None = None
    foreground_no_shadow: BoolArray | None = None

    # S6
    blobs: tuple[Blob, ...] = ()
    labels: AnyArray | None = None

    # S7 onwards. The tracker owns the track objects across frames; the context
    # carries a view of them so S8 and S9 need no reference to the stage that
    # produced them.
    tracks: tuple[Track, ...] = ()
    counted_this_frame: int = 0

    #: Frozen per-frame copies, filled in after the last stage that mutates a
    #: track. Anything that outlives the frame must use these: ``tracks`` holds
    #: live objects, so storing them per frame stores the same object each time.
    track_snapshots: tuple[TrackSnapshot, ...] = ()

    # Shared
    calibration: Calibration | None = None
    results: dict[str, StageResult] = field(default_factory=dict)

    #: Set by :meth:`release_intermediates`. Endpoints that need a stage image
    #: check this rather than probing a dozen fields for None.
    intermediates_released: bool = False

    @property
    def mask(self) -> BoolArray:
        """The best foreground mask available so far.

        Shadow-suppressed if S5 has run, raw foreground otherwise. Downstream
        stages ask for this rather than choosing, so turning S5 off in the UI
        genuinely removes it from the pipeline instead of leaving a stale mask
        in place.
        """
        if self.foreground_no_shadow is not None:
            return self.foreground_no_shadow
        if self.foreground is not None:
            return self.foreground
        raise StageError(
            "no foreground mask has been produced yet; S4 must run before any "
            "stage that consumes one"
        )

    def release_intermediates(self) -> None:
        """Drop the large per-frame arrays, keeping what outlives the frame.

        A context holds roughly 35 MB of float32 intermediates at 720p, so
        retaining one per frame costs gigabytes over a clip and fails partway
        through with an allocation error. What the console needs after the fact
        is small: the frozen track snapshots, the blob records, and each stage's
        timing and metrics. ``bgr`` is a view onto the decoded clip the session
        already holds, so it stays - releasing it would save nothing and would
        break the overlay.

        This is one-way. :meth:`Session.context` re-runs the pipeline when a
        released frame's intermediates are needed again.
        """
        self.grey = None
        self.hsv = None
        self.background_grey = None
        self.background_bgr = None
        self.difference = None
        self.foreground = None
        self.previous_mask = None
        self.shadow = None
        self.foreground_no_shadow = None
        self.labels = None
        self.tracks = ()
        for result in self.results.values():
            result.image = None
            result.mask = None
        self.intermediates_released = True

    @property
    def has_intermediates(self) -> bool:
        return not self.intermediates_released

    def require_grey(self) -> FloatArray:
        if self.grey is None:
            raise StageError("S1 has not run: no greyscale frame is available")
        return self.grey

    def require_hsv(self) -> FloatArray:
        if self.hsv is None:
            raise StageError("S1 has not run: no HSV frame is available")
        return self.hsv

    def require_calibration(self) -> Calibration:
        if self.calibration is None or not self.calibration.is_valid:
            raise StageError(
                "this stage needs a road-plane calibration; calibrate the camera "
                "first, or load a saved calibration for this clip"
            )
        return self.calibration


class Stage(ABC):
    """Base class for a pipeline stage.

    Subclasses declare ``stage_id``, ``name``, ``units``, ``rationale`` and
    ``param_specs``, then implement :meth:`process`. Parameters are validated on
    assignment against the same schema the interface renders, so an out-of-range
    value fails at the point it is set rather than somewhere inside NumPy three
    stages later.
    """

    stage_id: ClassVar[str] = ""
    name: ClassVar[str] = ""
    units: ClassVar[tuple[int, ...]] = ()
    rationale: ClassVar[str] = ""
    param_specs: ClassVar[tuple[ParamSpec, ...]] = ()

    def __init__(self, **params: Any) -> None:
        if not self.stage_id:
            raise TypeError(f"{type(self).__name__} does not declare a stage_id")
        if not self.rationale:
            raise TypeError(
                f"{type(self).__name__} does not declare a rationale; every stage "
                f"must be able to explain itself in the inspector"
            )
        self._params: dict[str, Any] = {p.name: p.default for p in self.param_specs}
        self.set_params(**params)
        self.reset()

    # -- parameters --------------------------------------------------------

    @property
    def params(self) -> dict[str, Any]:
        return dict(self._params)

    def param(self, name: str) -> Any:
        try:
            return self._params[name]
        except KeyError:
            raise KeyError(
                f"stage {self.stage_id!r} has no parameter {name!r}; it has "
                f"{', '.join(sorted(self._params)) or 'none'}"
            ) from None

    def set_params(self, **params: Any) -> None:
        """Validate and apply parameters, then clear any cached state.

        Resetting is not optional. A background model built under one learning
        rate does not describe the scene under another, and evaluating a changed
        parameter against stale state would show the user a metric delta that
        describes neither setting.
        """
        if not params:
            return
        specs = {p.name: p for p in self.param_specs}
        unknown = set(params) - set(specs)
        if unknown:
            raise ValueError(
                f"stage {self.stage_id!r} has no parameter(s) "
                f"{', '.join(sorted(unknown))}; it has "
                f"{', '.join(sorted(specs)) or 'none'}"
            )
        for key, value in params.items():
            self._params[key] = specs[key].coerce(value)
        self.reset()

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.param_specs}

    def differs_from_default(self) -> tuple[str, ...]:
        """Parameter names currently set away from their default.

        The console puts a badge on a panel when this is non-empty, so a user
        who has been experimenting can see at a glance which stage is no longer
        in its published configuration.
        """
        defaults = self.defaults()
        return tuple(k for k, v in self._params.items() if v != defaults[k])

    def reset(self) -> None:  # noqa: B027 - optional hook, not an abstract method
        """Clear per-clip state.

        Deliberately concrete and empty rather than abstract: most stages are
        pure functions of one frame and have nothing to clear, and forcing them
        to write an empty override would be noise that hides the stages where
        the hook genuinely matters.
        """

    # -- execution ---------------------------------------------------------

    @abstractmethod
    def process(self, ctx: FrameContext) -> StageResult:
        """Run the stage over one frame, mutating ``ctx`` and reporting on it."""

    def __call__(self, ctx: FrameContext) -> StageResult:
        result = self.process(ctx)
        ctx.results[self.stage_id] = result
        return result

    @contextmanager
    def _timed(self, **metrics: float) -> Iterator[StageResult]:
        """Build a StageResult and time the block that fills it in."""
        result = StageResult(
            stage_id=self.stage_id,
            name=self.name,
            units=self.units,
            rationale=self.rationale,
            params=self.params,
            metrics=dict(metrics),
        )
        started = time.perf_counter()
        try:
            yield result
        finally:
            result.elapsed_ms = (time.perf_counter() - started) * 1000.0

    def describe(self) -> dict[str, Any]:
        """Stage metadata for the API, including the parameter schema."""
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "units": list(self.units),
            "rationale": self.rationale,
            "params": [p.to_dict() for p in self.param_specs],
            "values": self.params,
            "modified": list(self.differs_from_default()),
        }


def as_uint8_mask(mask: BoolArray) -> UInt8Array:
    """Boolean mask to the 0/255 form OpenCV expects."""
    return (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
