"""The operator library: pure functions, one module per syllabus area.

Importing this package imports every module, which is what populates
:data:`kerbside.core.registry.REGISTRY`. The API layer depends on that: it can
enumerate and describe every operator without knowing any of their names in
advance.

No module here performs I/O, holds state, or reads a global. Every function
takes arrays and explicit parameters and returns arrays.
"""

from __future__ import annotations

from kerbside.operators import (
    colour,
    edges,
    frequency,
    morphology,
    motion,
    point,
    segment,
    spatial,
    transforms,
)

__all__ = [
    "colour",
    "edges",
    "frequency",
    "morphology",
    "motion",
    "point",
    "segment",
    "spatial",
    "transforms",
]
