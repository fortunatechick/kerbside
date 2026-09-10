"""Unit 8 - motion estimation by block matching.

Block matching asks, for each square of the current frame, where that square sat
in the previous one. It is how every video codec since H.261 has exploited
temporal correlation, and it is also a speed measurement: displacement over a
known frame interval, mapped through the road-plane homography, is metres per
second.

That second use is why this module exists in a traffic system. The centroid
route in S8 measures where a *silhouette* is, so it inherits every error in the
silhouette - a shadow at the edge, a partial occlusion, a bounding box smeared
by motion blur. Block matching measures where *texture* went, which is a
different and largely independent estimate. S8 runs both and reports the
comparison honestly.

Three search strategies are implemented because the syllabus asks for the
comparison and because the comparison has a real answer: exhaustive search finds
the true minimum of the cost surface, the fast searches find a local one, and
the gap between them is visible on the SAD surface the Motion Lab draws.

Two traps are handled here rather than discovered later:

* **The aperture problem** (trap 9). A plain-sided white truck has no texture,
  so every candidate position matches equally well and the resulting vector is
  arbitrary. Blocks are gated by intensity variance, and the per-vehicle
  aggregate is a *median* rather than a mean, so a handful of arbitrary vectors
  cannot drag the estimate.
* **Tie-breaking** (trap 10). In a uniform region many candidates share the
  minimum cost, and taking whichever the loop saw first turns the vector field
  into noise that correlates with the scan order. Every search here breaks ties
  towards zero displacement, which is both the correct prior and stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, BoolArray, FloatArray, FrameType, UInt8Array

__all__ = [
    "CostName",
    "SearchName",
    "BlockMatch",
    "MotionField",
    "SearchComparison",
    "sad",
    "ssd",
    "mad",
    "block_cost",
    "exhaustive_search",
    "two_d_log_search",
    "three_step_search",
    "search_block",
    "block_variance",
    "estimate_motion_field",
    "motion_compensate",
    "compensation_psnr",
    "sad_surface",
    "compare_search_algorithms",
    "classify_frame",
    "aggregate_vector",
]

CostName = Literal["sad", "ssd", "mad"]
SearchName = Literal["exhaustive", "two_d_log", "three_step"]


# --------------------------------------------------------------------------
# Cost functions
# --------------------------------------------------------------------------


def sad(a: AnyArray, b: AnyArray) -> float:
    """Sum of absolute differences.

    The classical cost: no multiplication, so it was cheap in hardware, and it
    is more tolerant of a few badly wrong pixels than a squared cost because it
    does not square the outlier.
    """
    return float(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)).sum())


def ssd(a: AnyArray, b: AnyArray) -> float:
    """Sum of squared differences. Penalises large single-pixel errors heavily,
    which makes it more sensitive to an occluding edge crossing the block."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float((d * d).sum())


def mad(a: AnyArray, b: AnyArray) -> float:
    """Mean absolute difference: SAD divided by the block area.

    Identical rankings to SAD for a fixed block size, but comparable across
    block sizes, which the ablation needs.
    """
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    return float(np.abs(d).mean()) if d.size else float("inf")


_COSTS = {"sad": sad, "ssd": ssd, "mad": mad}


def block_cost(a: AnyArray, b: AnyArray, cost: CostName = "sad") -> float:
    try:
        return _COSTS[cost](a, b)
    except KeyError:
        raise ValueError(f"unknown cost {cost!r}; use sad, ssd or mad") from None


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BlockMatch:
    """The result of matching one block.

    **Sign convention.** ``dx_px`` and ``dy_px`` are the displacement of the
    *content*: positive ``dx_px`` means the texture in this block sat further
    left in the reference frame and has moved right. This is the opposite of the
    motion vector a video codec transmits, which points from the current block
    back to its match in the reference, and :attr:`reference_offset_px` gives
    that form.

    The choice is deliberate rather than arbitrary. S8 refines a block match
    with :func:`~kerbside.operators.frequency.phase_correlation`, which reports
    content displacement, and combines the result with a centroid-derived
    velocity, which is also content displacement. Three quantities that must be
    added together should not disagree about which way is positive - a sign
    error there produces speeds of the right magnitude pointing the wrong way,
    which looks like a tracking bug rather than an arithmetic one.
    """

    dx_px: int
    dy_px: int
    cost: float
    comparisons: int
    variance: float = 0.0
    textured: bool = True

    @property
    def magnitude_px(self) -> float:
        return float(np.hypot(self.dx_px, self.dy_px))

    @property
    def reference_offset_px(self) -> tuple[int, int]:
        """The same match as a codec-style motion vector, ``(dx, dy)``.

        This is the offset to add to the block's own position to find its match
        in the reference frame, which is what motion compensation needs.
        """
        return (-self.dx_px, -self.dy_px)


class _Evaluator:
    """Evaluates candidate displacements, counting and caching comparisons.

    The comparison count is the currency of the whole search comparison, so it
    is measured rather than derived from a formula - a fast search that revisits
    a candidate must not be charged for it twice, because a real implementation
    would not recompute it either.
    """

    __slots__ = (
        "block",
        "comparisons",
        "cost_name",
        "height",
        "reference",
        "seen",
        "width",
        "x",
        "y",
    )

    def __init__(
        self,
        reference: AnyArray,
        block: AnyArray,
        y: int,
        x: int,
        cost_name: CostName,
    ) -> None:
        self.reference = np.asarray(reference, dtype=np.float64)
        self.block = np.asarray(block, dtype=np.float64)
        self.y = y
        self.x = x
        self.cost_name = cost_name
        self.seen: dict[tuple[int, int], float] = {}
        self.comparisons = 0
        self.height, self.width = self.reference.shape[:2]

    def __call__(self, dy: int, dx: int) -> float:
        key = (dy, dx)
        cached = self.seen.get(key)
        if cached is not None:
            return cached
        bh, bw = self.block.shape[:2]
        ry, rx = self.y + dy, self.x + dx
        if ry < 0 or rx < 0 or ry + bh > self.height or rx + bw > self.width:
            # A candidate that falls off the reference frame is not a match at
            # any price. Returning infinity rather than clamping the coordinates
            # stops the search inventing a displacement that would put the block
            # outside the image it came from.
            self.seen[key] = float("inf")
            return float("inf")
        self.comparisons += 1
        value = block_cost(self.block, self.reference[ry : ry + bh, rx : rx + bw], self.cost_name)
        self.seen[key] = value
        return value

    def better(
        self, candidate: tuple[int, int], best: tuple[int, int], tolerance: float = 1e-9
    ) -> bool:
        """Whether ``candidate`` beats ``best``, breaking ties towards zero.

        Trap 10. In a flat region a great many candidates share the minimum
        cost; picking whichever the loop happened to reach first makes the
        vector field a picture of the scan order. Preferring the smaller
        displacement is both the correct prior - most blocks have not moved far -
        and deterministic.
        """
        c_cost, b_cost = self(*candidate), self(*best)
        if c_cost < b_cost - tolerance:
            return True
        if c_cost > b_cost + tolerance:
            return False
        return (candidate[0] ** 2 + candidate[1] ** 2) < (best[0] ** 2 + best[1] ** 2)


def _initial_step(search_range: int) -> int:
    """First step size for the coarse-to-fine searches.

    The classical choice is half the search range rounded to a power of two, so
    a range of 7 starts at 4 and takes the three steps 4, 2, 1 that give the
    three-step search its name; a range of 16 starts at 8. Starting at the full
    range instead would put the first probes on the boundary of the search
    window, where half the pattern falls outside it and the first decision is
    made on fewer candidates than intended.
    """
    return 1 << max(0, (int(search_range) - 1).bit_length() - 1)


def exhaustive_search(
    reference: AnyArray, block: AnyArray, y: int, x: int,
    search_range: int = 16, cost: CostName = "sad",
) -> BlockMatch:
    """Full search over every candidate in the search window.

    Evaluates ``(2r + 1)^2`` positions and therefore finds the true global
    minimum of the cost surface. That makes it the reference the fast searches
    are measured against - it cannot be beaten on accuracy, only on cost.
    """
    if search_range < 1:
        raise ValueError(f"search range must be at least 1, got {search_range}")
    ev = _Evaluator(reference, block, y, x, cost)
    best = (0, 0)
    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            if ev.better((dy, dx), best):
                best = (dy, dx)
    # ``best`` is the offset into the reference; the content displacement is
    # its negation. See BlockMatch for why that is the reported form.
    return BlockMatch(
        dx_px=-best[1], dy_px=-best[0], cost=ev(*best), comparisons=ev.comparisons
    )


def three_step_search(
    reference: AnyArray, block: AnyArray, y: int, x: int,
    search_range: int = 16, cost: CostName = "sad",
) -> BlockMatch:
    """Three-step search: a coarse-to-fine walk over nine candidates per step.

    Start with a step of about half the search range, evaluate the centre and
    its eight neighbours at that spacing, move to the winner, halve the step,
    repeat. For a range of 16 that is three steps of 8, 4, 2 and a final 1, so
    roughly 25 comparisons against the exhaustive search's 1089.

    It assumes the cost surface is unimodal. When it is not - and a periodic
    texture like a dashed lane marking guarantees it is not - the first coarse
    step can land in the wrong basin and every later step refines the wrong
    minimum. The Motion Lab draws exactly that on the SAD surface.
    """
    if search_range < 1:
        raise ValueError(f"search range must be at least 1, got {search_range}")
    ev = _Evaluator(reference, block, y, x, cost)
    best = (0, 0)
    step = _initial_step(search_range)
    while step >= 1:
        centre = best
        for dy in (-step, 0, step):
            for dx in (-step, 0, step):
                candidate = (centre[0] + dy, centre[1] + dx)
                if abs(candidate[0]) > search_range or abs(candidate[1]) > search_range:
                    continue
                if ev.better(candidate, best):
                    best = candidate
        if step == 1:
            break
        step //= 2
    # ``best`` is the offset into the reference; the content displacement is
    # its negation. See BlockMatch for why that is the reported form.
    return BlockMatch(
        dx_px=-best[1], dy_px=-best[0], cost=ev(*best), comparisons=ev.comparisons
    )


def two_d_log_search(
    reference: AnyArray, block: AnyArray, y: int, x: int,
    search_range: int = 16, cost: CostName = "sad",
) -> BlockMatch:
    """Two-dimensional logarithmic search: a cross pattern that halves on a hit.

    Evaluate the centre and four neighbours in a plus shape at the current step.
    If the centre wins, halve the step; otherwise move to the winner and keep
    the step. When the step reaches one, evaluate all eight neighbours and stop.

    Five candidates per iteration against the three-step search's nine, so it is
    usually cheaper still - but it follows a gradient, so it is even more
    committed to whichever basin it starts in.
    """
    if search_range < 1:
        raise ValueError(f"search range must be at least 1, got {search_range}")
    ev = _Evaluator(reference, block, y, x, cost)
    best = (0, 0)
    step = _initial_step(search_range)

    guard = 0
    while step > 1 and guard < 64:
        guard += 1
        centre = best
        for dy, dx in ((-step, 0), (step, 0), (0, -step), (0, step)):
            candidate = (centre[0] + dy, centre[1] + dx)
            if abs(candidate[0]) > search_range or abs(candidate[1]) > search_range:
                continue
            if ev.better(candidate, best):
                best = candidate
        if best == centre:
            step //= 2

    centre = best
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            candidate = (centre[0] + dy, centre[1] + dx)
            if abs(candidate[0]) > search_range or abs(candidate[1]) > search_range:
                continue
            if ev.better(candidate, best):
                best = candidate
    # ``best`` is the offset into the reference; the content displacement is
    # its negation. See BlockMatch for why that is the reported form.
    return BlockMatch(
        dx_px=-best[1], dy_px=-best[0], cost=ev(*best), comparisons=ev.comparisons
    )


_SEARCHES = {
    "exhaustive": exhaustive_search,
    "two_d_log": two_d_log_search,
    "three_step": three_step_search,
}


def search_block(
    reference: AnyArray, block: AnyArray, y: int, x: int,
    search_range: int = 16, cost: CostName = "sad", search: SearchName = "exhaustive",
) -> BlockMatch:
    """Match one block by the named search strategy."""
    try:
        fn = _SEARCHES[search]
    except KeyError:
        raise ValueError(
            f"unknown search {search!r}; use exhaustive, two_d_log or three_step"
        ) from None
    return fn(reference, block, y, x, search_range, cost)


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------


def block_variance(block: AnyArray) -> float:
    """Intensity variance of a block: the texture gate of trap 9.

    A block with no variance matches every candidate equally well, so its
    displacement carries no information. Reporting one anyway is worse than
    reporting none, because it looks exactly like a measurement.
    """
    return float(np.var(np.asarray(block, dtype=np.float64)))


@dataclass(slots=True)
class MotionField:
    """A grid of block matches over one frame pair."""

    block_size: int
    search_range: int
    search: SearchName
    cost: CostName
    rows: int
    cols: int
    dx_px: FloatArray
    dy_px: FloatArray
    costs: FloatArray
    variance: FloatArray
    textured: BoolArray
    comparisons: int = 0
    matches: list[BlockMatch] = field(default_factory=list)

    @property
    def textured_fraction(self) -> float:
        return float(self.textured.mean()) if self.textured.size else 0.0

    def vectors(self, textured_only: bool = True) -> FloatArray:
        """The displacement vectors as an ``(N, 2)`` array of ``(dx, dy)``."""
        mask = self.textured if textured_only else np.ones_like(self.textured, dtype=bool)
        return np.stack([self.dx_px[mask], self.dy_px[mask]], axis=1)

    def block_origin(self, row: int, col: int) -> tuple[int, int]:
        return (col * self.block_size, row * self.block_size)


@register(
    "motion.field",
    "Block-matching motion field",
    units=(8,),
    rationale=(
        "Estimates where each block of the current frame sat in the previous "
        "one. It is how codecs exploit temporal correlation, and - divided by "
        "the frame interval and mapped through the homography - it is an "
        "independent speed measurement that depends on texture rather than on "
        "the silhouette."
    ),
    params=(
        ParamSpec("block_size", "Block size", "int", 16, 4, 64, 4, unit="px"),
        ParamSpec("search_range", "Search range", "int", 16, 1, 64, 1, unit="px"),
        ParamSpec(
            "search", "Search strategy", "enum", "exhaustive",
            choices=("exhaustive", "two_d_log", "three_step"),
        ),
        ParamSpec("cost", "Cost function", "enum", "sad", choices=("sad", "ssd", "mad")),
        ParamSpec(
            "min_variance", "Texture gate", "float", 25.0, 0.0, 2000.0, 1.0,
            description="Blocks below this intensity variance are matched but marked untextured.",
        ),
    ),
    category="motion",
)
def estimate_motion_field(
    current: AnyArray,
    reference: AnyArray,
    block_size: int = 16,
    search_range: int = 16,
    search: SearchName = "exhaustive",
    cost: CostName = "sad",
    min_variance: float = 25.0,
    region: tuple[int, int, int, int] | None = None,
) -> MotionField:
    """Estimate a block motion field between two frames.

    ``region`` restricts the field to ``(x, y, w, h)``, which is how S8 uses it:
    matching only the blocks inside a vehicle's bounding box costs a fraction of
    a full-frame field and is all the vehicle's speed needs.

    Untextured blocks are matched and reported, but flagged, so the Motion Lab
    can show both the raw field and the gated one. Discarding them silently
    would hide the aperture problem rather than demonstrate it.
    """
    cur = _to_grey(current)
    ref = _to_grey(reference)
    if cur.shape != ref.shape:
        raise ValueError(
            f"frames must have the same shape, got {cur.shape} and {ref.shape}"
        )
    if block_size < 2:
        raise ValueError(f"block size must be at least 2, got {block_size}")

    if region is None:
        x0, y0, w, h = 0, 0, cur.shape[1], cur.shape[0]
    else:
        x0, y0, w, h = (int(v) for v in region)
        x0 = max(0, x0)
        y0 = max(0, y0)
        w = min(w, cur.shape[1] - x0)
        h = min(h, cur.shape[0] - y0)
    rows = max(0, h // block_size)
    cols = max(0, w // block_size)

    dx = np.zeros((rows, cols), dtype=np.float64)
    dy = np.zeros((rows, cols), dtype=np.float64)
    costs = np.zeros((rows, cols), dtype=np.float64)
    variance = np.zeros((rows, cols), dtype=np.float64)
    textured = np.zeros((rows, cols), dtype=bool)
    matches: list[BlockMatch] = []
    total_comparisons = 0

    for r in range(rows):
        for c in range(cols):
            by = y0 + r * block_size
            bx = x0 + c * block_size
            block = cur[by : by + block_size, bx : bx + block_size]
            var = block_variance(block)
            match = search_block(ref, block, by, bx, search_range, cost, search)
            match = BlockMatch(
                dx_px=match.dx_px, dy_px=match.dy_px, cost=match.cost,
                comparisons=match.comparisons, variance=var,
                textured=var >= min_variance,
            )
            dx[r, c] = match.dx_px
            dy[r, c] = match.dy_px
            costs[r, c] = match.cost
            variance[r, c] = var
            textured[r, c] = match.textured
            total_comparisons += match.comparisons
            matches.append(match)

    return MotionField(
        block_size=block_size, search_range=search_range, search=search, cost=cost,
        rows=rows, cols=cols, dx_px=dx, dy_px=dy, costs=costs, variance=variance,
        textured=textured, comparisons=total_comparisons, matches=matches,
    )


def _to_grey(image: AnyArray) -> FloatArray:
    x = np.asarray(image)
    if x.ndim == 3:
        x = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return np.asarray(x, dtype=np.float64)


def aggregate_vector(field: MotionField, textured_only: bool = True) -> tuple[float, float]:
    """Collapse a motion field to one displacement, by the median.

    The median rather than the mean, and this is trap 9 again. A vehicle with a
    plain flank contributes blocks whose vectors are arbitrary; a mean is
    dragged by every one of them, a median ignores them as long as they are the
    minority. Taking the median of each component independently is the cheap
    componentwise version, which is what a real implementation would do.
    """
    vectors = field.vectors(textured_only)
    if vectors.size == 0:
        vectors = field.vectors(textured_only=False)
    if vectors.size == 0:
        return (0.0, 0.0)
    return (float(np.median(vectors[:, 0])), float(np.median(vectors[:, 1])))


def motion_compensate(reference: AnyArray, field: MotionField) -> FloatArray:
    """Rebuild the current frame from the reference and the motion field.

    This is what a codec transmits instead of the frame itself, and the quality
    of the rebuild is the honest measure of a motion estimator: a vector field
    that looks tidy but predicts badly is not a good field.
    """
    ref = _to_grey(reference)
    out = np.zeros_like(ref)
    bs = field.block_size
    for r in range(field.rows):
        for c in range(field.cols):
            by, bx = r * bs, c * bs
            # field stores content displacement, so the match in the reference
            # lies at minus that offset.
            sy = int(by - field.dy_px[r, c])
            sx = int(bx - field.dx_px[r, c])
            sy = int(np.clip(sy, 0, ref.shape[0] - bs))
            sx = int(np.clip(sx, 0, ref.shape[1] - bs))
            out[by : by + bs, bx : bx + bs] = ref[sy : sy + bs, sx : sx + bs]
    return out


def compensation_psnr(
    current: AnyArray, reference: AnyArray, field: MotionField
) -> float:
    """PSNR of the motion-compensated prediction against the true frame."""
    cur = _to_grey(current)
    predicted = motion_compensate(reference, field)
    rows = field.rows * field.block_size
    cols = field.cols * field.block_size
    if rows == 0 or cols == 0:
        return float("nan")
    mse = float(np.mean((cur[:rows, :cols] - predicted[:rows, :cols]) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def sad_surface(
    reference: AnyArray, current: AnyArray, y: int, x: int,
    block_size: int = 16, search_range: int = 16, cost: CostName = "sad",
) -> FloatArray:
    """The full cost surface for one block, for the Motion Lab's heatmap.

    Always computed exhaustively regardless of which search is in use, because
    the point of drawing it is to show what the fast searches did *not* look at.

    Indexed ``[dy + range, dx + range]`` where ``(dy, dx)`` is the offset into
    the *reference* frame - the codec convention, not the content-displacement
    convention :class:`BlockMatch` reports - because that is what the search
    actually walks over. A match reported as ``(dx_px, dy_px)`` sits at
    ``surface[range - dy_px, range - dx_px]``.
    """
    ref = _to_grey(reference)
    cur = _to_grey(current)
    block = cur[y : y + block_size, x : x + block_size]
    size = 2 * search_range + 1
    surface = np.full((size, size), np.inf, dtype=np.float64)
    for i, dy in enumerate(range(-search_range, search_range + 1)):
        for j, dx in enumerate(range(-search_range, search_range + 1)):
            ry, rx = y + dy, x + dx
            if ry < 0 or rx < 0 or ry + block_size > ref.shape[0] or rx + block_size > ref.shape[1]:
                continue
            surface[i, j] = block_cost(
                block, ref[ry : ry + block_size, rx : rx + block_size], cost
            )
    return surface


@dataclass(frozen=True, slots=True)
class SearchComparison:
    """How one search strategy performed against exhaustive search."""

    search: SearchName
    comparisons: int
    speedup: float
    mv_mae_px: float
    exact_fraction: float
    prediction_psnr_db: float
    elapsed_ms: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "search": self.search,
            "comparisons": self.comparisons,
            "speedup": self.speedup,
            "mv_mae_px": self.mv_mae_px,
            "exact_fraction": self.exact_fraction,
            "prediction_psnr_db": self.prediction_psnr_db,
            "elapsed_ms": self.elapsed_ms,
        }


def compare_search_algorithms(
    current: AnyArray,
    reference: AnyArray,
    block_size: int = 16,
    search_range: int = 16,
    cost: CostName = "sad",
    min_variance: float = 25.0,
    textured_only: bool = True,
) -> dict[str, SearchComparison]:
    """Run all three searches on one frame pair and score them against exhaustive.

    Exhaustive search is the reference for accuracy because it is exact by
    construction. The fast searches are scored on how far their vectors land
    from it, how often they land on it exactly, how well they still predict the
    frame, and how many block comparisons each cost.

    The prediction PSNR is the important column. A fast search can be well off
    the exhaustive vector and still predict almost as well, because it found a
    different position that matches nearly as closely - which is fine for a
    codec and not fine for a speed measurement, and that distinction is the
    finding worth reporting.
    """
    import time

    fields: dict[str, MotionField] = {}
    timings: dict[str, float] = {}
    for name in ("exhaustive", "two_d_log", "three_step"):
        started = time.perf_counter()
        fields[name] = estimate_motion_field(
            current, reference, block_size, search_range, name, cost, min_variance
        )
        timings[name] = (time.perf_counter() - started) * 1000.0

    truth = fields["exhaustive"]
    gate = truth.textured if textured_only else np.ones_like(truth.textured, dtype=bool)
    if not gate.any():
        gate = np.ones_like(truth.textured, dtype=bool)

    out: dict[str, SearchComparison] = {}
    for name, fld in fields.items():
        errors = np.hypot(fld.dx_px[gate] - truth.dx_px[gate], fld.dy_px[gate] - truth.dy_px[gate])
        exact = np.isclose(fld.dx_px[gate], truth.dx_px[gate]) & np.isclose(
            fld.dy_px[gate], truth.dy_px[gate]
        )
        out[name] = SearchComparison(
            search=name,  # type: ignore[arg-type]
            comparisons=fld.comparisons,
            speedup=truth.comparisons / max(1, fld.comparisons),
            mv_mae_px=float(errors.mean()) if errors.size else 0.0,
            exact_fraction=float(exact.mean()) if exact.size else 0.0,
            prediction_psnr_db=compensation_psnr(current, reference, fld),
            elapsed_ms=timings[name],
        )
    return out


@register(
    "motion.classify_frame",
    "I/P frame classification",
    units=(8,),
    rationale=(
        "Labels a frame I when it differs enough from its predecessor to be "
        "worth storing whole, and P otherwise. This is what a digital video "
        "recorder does to avoid storing an empty road all night, and the "
        "proportion of I frames is the storage saving it achieves."
    ),
    params=(
        ParamSpec(
            "threshold", "Change threshold", "float", 2.0, 0.05, 60.0, 0.05,
            description="Mean absolute frame difference above which the frame is stored whole.",
        ),
    ),
    category="motion",
)
def classify_frame(
    current: AnyArray, reference: AnyArray | None, threshold: float = 2.0
) -> tuple[FrameType, float]:
    """Classify a frame as I or P by frame-difference energy.

    Returns the classification and the measured energy, because the threshold is
    a policy decision and the energy is the evidence. The first frame of a clip
    has no predecessor and is always an I frame.
    """
    if reference is None:
        return (FrameType.I, float("inf"))
    cur = _to_grey(current)
    ref = _to_grey(reference)
    if cur.shape != ref.shape:
        raise ValueError(f"frames must have the same shape, got {cur.shape} and {ref.shape}")
    energy = float(np.abs(cur - ref).mean())
    return (FrameType.I if energy >= threshold else FrameType.P, energy)


def draw_motion_field(
    image: AnyArray,
    field: MotionField,
    textured_only: bool = True,
    scale: float = 1.0,
    colour: tuple[int, int, int] = (249, 115, 98),
) -> UInt8Array:
    """Draw the motion vectors as arrows over a frame, for the Motion Lab."""
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bgr = (int(colour[2]), int(colour[1]), int(colour[0]))
    half = field.block_size // 2
    for r in range(field.rows):
        for c in range(field.cols):
            if textured_only and not field.textured[r, c]:
                continue
            cx = c * field.block_size + half
            cy = r * field.block_size + half
            ex = int(round(cx + field.dx_px[r, c] * scale))
            ey = int(round(cy + field.dy_px[r, c] * scale))
            cv2.arrowedLine(canvas, (cx, cy), (ex, ey), bgr, 1, cv2.LINE_AA, tipLength=0.3)
    return np.asarray(canvas, dtype=np.uint8)
