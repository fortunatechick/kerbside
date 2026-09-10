"""The operator registry.

Every operator in the library declares four things beyond its implementation:
an id, the syllabus unit or units it belongs to, a parameter schema, and a
one-sentence rationale saying what it does and why this pipeline uses it.

That is a course requirement rather than a nicety - the interface has to be able
to explain each stage - but it also earns its keep in the code. The parameter
schema is what lets the inspector build a control for an operator it has never
heard of, what bounds the values the API will accept, and what "reset to
default" resets to. Keeping it next to the function means the three cannot drift
apart.

Nothing here executes image processing; it is metadata plus a callable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

ParamKind = Literal["float", "int", "bool", "enum", "odd_int"]

__all__ = [
    "ParamSpec",
    "OperatorSpec",
    "OperatorRegistry",
    "REGISTRY",
    "register",
    "operator",
]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One tunable parameter of an operator.

    ``kind`` drives both validation and the control the inspector renders.
    ``odd_int`` exists because structuring elements and filter kernels must have
    an odd side length to have a well-defined centre; silently accepting an even
    one produces a half-pixel shift that is very hard to see and very easy to
    blame on something else.
    """

    name: str
    label: str
    kind: ParamKind
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    unit: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if self.kind == "enum" and not self.choices:
            raise ValueError(f"parameter {self.name!r} is an enum with no choices")
        if self.kind == "enum" and self.default not in self.choices:
            raise ValueError(
                f"parameter {self.name!r} default {self.default!r} is not one of "
                f"{self.choices}"
            )
        if self.kind == "odd_int" and int(self.default) % 2 == 0:
            raise ValueError(
                f"parameter {self.name!r} must default to an odd value, got {self.default}"
            )

    def coerce(self, value: Any) -> Any:
        """Validate and convert a value from the API into what the operator wants.

        Raises rather than clamping. A parameter arriving out of range means the
        caller believes something the system does not, and silently substituting
        a different number would make the resulting image unexplainable.
        """
        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and value in (0, 1):
                return bool(value)
            raise ValueError(f"{self.name}: expected a boolean, got {value!r}")

        if self.kind == "enum":
            if value not in self.choices:
                raise ValueError(
                    f"{self.name}: {value!r} is not one of {', '.join(self.choices)}"
                )
            return value

        if self.kind in ("int", "odd_int"):
            try:
                number: float = int(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.name}: expected an integer, got {value!r}") from None
            if self.kind == "odd_int" and int(number) % 2 == 0:
                raise ValueError(
                    f"{self.name}: must be odd so the kernel has a centre pixel, "
                    f"got {int(number)}"
                )
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(f"{self.name}: expected a number, got {value!r}") from None

        if self.minimum is not None and number < self.minimum:
            raise ValueError(
                f"{self.name}: {number} is below the minimum {self.minimum}"
                f"{' ' + self.unit if self.unit else ''}"
            )
        if self.maximum is not None and number > self.maximum:
            raise ValueError(
                f"{self.name}: {number} is above the maximum {self.maximum}"
                f"{' ' + self.unit if self.unit else ''}"
            )
        return int(number) if self.kind in ("int", "odd_int") else float(number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "choices": list(self.choices),
            "unit": self.unit,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    """An operator plus everything the interface needs to explain and drive it."""

    operator_id: str
    name: str
    units: tuple[int, ...]
    rationale: str
    fn: Callable[..., Any]
    params: tuple[ParamSpec, ...] = ()
    category: str = "general"
    reference: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def param(self, name: str) -> ParamSpec:
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(
            f"operator {self.operator_id!r} has no parameter {name!r}; "
            f"it has {', '.join(p.name for p in self.params) or 'none'}"
        )

    def coerce_params(self, values: dict[str, Any], strict: bool = True) -> dict[str, Any]:
        """Fill in defaults and validate everything the caller supplied.

        With ``strict`` set, an unknown parameter name is an error rather than
        being ignored - a typo in a parameter name would otherwise present as
        "changing this slider does nothing", which is a miserable thing to debug
        through a web interface.
        """
        known = {p.name for p in self.params}
        unknown = set(values) - known
        if unknown and strict:
            raise ValueError(
                f"operator {self.operator_id!r} has no parameter(s) "
                f"{', '.join(sorted(unknown))}; it has "
                f"{', '.join(sorted(known)) or 'none'}"
            )
        out = dict(self.defaults)
        for p in self.params:
            if p.name in values:
                out[p.name] = p.coerce(values[p.name])
        return out

    def call(self, *args: Any, **params: Any) -> Any:
        """Invoke the operator with validated parameters."""
        return self.fn(*args, **self.coerce_params(params))

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator_id": self.operator_id,
            "name": self.name,
            "units": list(self.units),
            "rationale": self.rationale,
            "category": self.category,
            "reference": self.reference,
            "tags": list(self.tags),
            "params": [p.to_dict() for p in self.params],
        }


class OperatorRegistry:
    """A name-to-operator map with lookup by syllabus unit and by category."""

    def __init__(self) -> None:
        self._specs: dict[str, OperatorSpec] = {}

    def add(self, spec: OperatorSpec) -> OperatorSpec:
        if spec.operator_id in self._specs:
            raise ValueError(
                f"operator id {spec.operator_id!r} is already registered by "
                f"{self._specs[spec.operator_id].fn.__module__}"
            )
        if not spec.rationale.strip():
            raise ValueError(
                f"operator {spec.operator_id!r} has no rationale; every operator "
                f"must be able to explain itself in the inspector"
            )
        if not spec.units:
            raise ValueError(
                f"operator {spec.operator_id!r} declares no syllabus unit"
            )
        self._specs[spec.operator_id] = spec
        return spec

    def __contains__(self, operator_id: object) -> bool:
        return operator_id in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[OperatorSpec]:
        return iter(sorted(self._specs.values(), key=lambda s: (s.units[0], s.operator_id)))

    def get(self, operator_id: str) -> OperatorSpec:
        try:
            return self._specs[operator_id]
        except KeyError:
            raise KeyError(
                f"unknown operator {operator_id!r}; registered operators are "
                f"{', '.join(sorted(self._specs))}"
            ) from None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def by_unit(self, unit: int) -> tuple[OperatorSpec, ...]:
        return tuple(s for s in self if unit in s.units)

    def by_category(self, category: str) -> tuple[OperatorSpec, ...]:
        return tuple(s for s in self if s.category == category)

    def units(self) -> tuple[int, ...]:
        return tuple(sorted({u for s in self for u in s.units}))

    def to_dict(self) -> dict[str, Any]:
        return {"operators": [s.to_dict() for s in self]}


REGISTRY = OperatorRegistry()
"""The single process-wide registry. Modules under ``operators/`` populate it at
import time; :mod:`kerbside.operators` imports them all so that importing the
package is enough to see every operator."""


def register(
    operator_id: str,
    name: str,
    units: Iterable[int],
    rationale: str,
    params: Iterable[ParamSpec] = (),
    category: str = "general",
    reference: str = "",
    tags: Iterable[str] = (),
    registry: OperatorRegistry | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator registering a pure function as a named operator.

    The decorated function is returned unchanged, so it stays directly callable
    and directly testable. The registry holds a reference for the API and the
    inspector; it is not a wrapper and adds no runtime cost to normal calls.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = OperatorSpec(
            operator_id=operator_id,
            name=name,
            units=tuple(units),
            rationale=rationale,
            fn=fn,
            params=tuple(params),
            category=category,
            reference=reference,
            tags=tuple(tags),
        )
        (registry or REGISTRY).add(spec)
        return fn

    return decorate


def operator(operator_id: str, registry: OperatorRegistry | None = None) -> OperatorSpec:
    """Look up a registered operator."""
    return (registry or REGISTRY).get(operator_id)
