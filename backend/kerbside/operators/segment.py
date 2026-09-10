"""Unit 6 - thresholding, connectivity and region segmentation.

Two questions, in order. *Which pixels belong to an object?* - thresholding. And
*which of those pixels belong to the same object?* - connected-component
labelling. In this pipeline the first is Otsu on the background-difference
image, and the second is what individuates one vehicle from another.

Connectivity is not a detail here. 4-connectivity refuses diagonal links, so a
vehicle whose mask narrows to a diagonal thread splits in two. 8-connectivity
accepts them, so two vehicles touching only at a corner merge into one - and it
also makes a diagonal line *and* its background both connected, which cannot
both be true of a curve and its complement. m-connectivity removes that
ambiguity by admitting a diagonal link only where no 4-connected route already
exists.

One consequence is worth stating plainly, because it is easy to assume
otherwise and the ablation would be misleading if it did. **m-connectivity and
8-connectivity always produce identical components.** If a shared 4-neighbour of
a diagonal pair is foreground then a 4-path through it already joins them; if
neither is, m admits the diagonal directly. Either way the pair ends up in the
same component, so the partitions cannot differ - which is confirmed over random
masks in the tests. What m changes is the *uniqueness of the path* between two
pixels, which matters for boundary following, chain codes and thinning, not for
counting. So the connectivity study reports count differences between 4 and 8,
and reports path ambiguity for m, rather than pretending all three differ in the
same way.

Otsu is written from its definition rather than called, because the quantity it
maximises - between-class variance - is also the number worth reporting: when
the difference image has no real bimodality, Otsu still returns a threshold, and
its separability is the only thing that says so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as _csgraph_components

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, BoolArray, FloatArray, Int32Array, UInt8Array

__all__ = [
    "Connectivity",
    "OtsuResult",
    "ComponentStats",
    "LabelResult",
    "otsu_threshold",
    "otsu",
    "adaptive_threshold",
    "label_components",
    "component_stats",
    "region_growing",
    "split_and_merge",
    "watershed_split",
    "connectivity_disagreement",
]

Connectivity = Literal[4, 8, "m"]


def _to_grey_u8(image: AnyArray) -> UInt8Array:
    x = np.asarray(image)
    if x.ndim == 3:
        x = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return np.asarray(x, dtype=np.uint8)


# --------------------------------------------------------------------------
# Thresholding
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OtsuResult:
    """An Otsu threshold together with how well it actually separated anything."""

    threshold: float
    between_class_variance: float
    total_variance: float
    foreground_fraction: float
    _plateau_width: int = 1

    @property
    def separability(self) -> float:
        """Between-class variance as a fraction of the total, in ``[0, 1]``.

        Otsu always returns a threshold, even for a unimodal histogram where
        there is nothing to separate. This number is what distinguishes "the
        frame contains vehicles against a road" from "the frame contains only
        road, and the threshold just cut the noise distribution in half". S4
        uses it to refuse to report detections it cannot justify.
        """
        if self.total_variance <= 1e-12:
            return 0.0
        return float(np.clip(self.between_class_variance / self.total_variance, 0.0, 1.0))

    @property
    def is_bimodal(self) -> bool:
        """Whether the histogram genuinely had two modes to separate.

        The threshold is calibrated against a measurement, not guessed. Splitting
        a single Gaussian at its own median already yields a separability of
        0.64 - the two halves of one mode have means about 1.6 standard
        deviations apart - so any cut below that would call every image bimodal.
        Well-separated modes score above 0.95. The line is drawn at 0.85, which
        sits clear of the unimodal case with room for a noisy one.
        """
        return self.separability > 0.85

    @property
    def plateau_width(self) -> int:
        """How many levels achieved the maximal between-class variance.

        A wide plateau means the histogram has an empty gap: every level inside
        it separates the two modes identically well, so the criterion cannot
        distinguish between them. That is not a failure, but it does mean the
        exact threshold returned is arbitrary within the gap and will move under
        small changes in the tails.
        """
        return self._plateau_width


def otsu_threshold(
    image: AnyArray, mask: AnyArray | None = None, plateau_midpoint: bool = True
) -> OtsuResult:
    """Otsu's threshold, computed from the definition.

    Otsu chooses the level that maximises the between-class variance

        ``sigma_b^2(t) = w0(t) w1(t) [mu0(t) - mu1(t)]^2``

    where ``w`` are the class probabilities and ``mu`` the class means either
    side of ``t``. Maximising the separation between the classes is equivalent
    to minimising the variance within them, and it needs only the histogram, so
    the whole search over 256 levels is a pair of cumulative sums.

    ``mask`` restricts the statistics to a region. That matters in S4: computing
    the threshold over the whole frame includes the sky and the verge, which are
    never foreground and which drag the histogram towards a level that suits
    them rather than the road.

    ``plateau_midpoint`` addresses a property of the criterion that is easy to
    miss. When the two modes are separated by an empty gap, *every* level inside
    that gap produces exactly the same between-class variance, so the criterion
    is flat and a plain ``argmax`` returns the lowest level of the plateau -
    right against the upper tail of the dark mode. On a background-difference
    image, where the background is a spike near zero and the foreground a long
    tail, that admits far more noise than a level in the middle of the gap
    would. Taking the midpoint of the plateau is deterministic, is no less
    optimal by Otsu's own criterion, and is materially more stable. Set it False
    to reproduce the textbook and OpenCV behaviour exactly.
    """
    grey = _to_grey_u8(image)
    values = grey[np.asarray(mask, dtype=bool)] if mask is not None else grey.ravel()
    if values.size == 0:
        raise ValueError("cannot compute a threshold over an empty region")

    counts = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    levels = np.arange(256, dtype=np.float64)
    probability = counts / total

    w0 = np.cumsum(probability)
    w1 = 1.0 - w0
    mean_cumulative = np.cumsum(probability * levels)
    mean_total = float(mean_cumulative[-1])

    # Levels where either class is empty give no valid split; excluding them is
    # what stops the maximum landing at 0 or 255 on a one-sided histogram.
    valid = (w0 > 1e-12) & (w1 > 1e-12)
    between = np.zeros(256, dtype=np.float64)
    between[valid] = (
        (mean_total * w0[valid] - mean_cumulative[valid]) ** 2 / (w0[valid] * w1[valid])
    )

    peak = float(between.max())
    plateau = np.nonzero(between >= peak * (1.0 - 1e-9))[0] if peak > 0 else np.array([0])
    best = int(plateau[len(plateau) // 2]) if plateau_midpoint else int(plateau[0])

    variance_total = float(np.sum(probability * (levels - mean_total) ** 2))
    return OtsuResult(
        threshold=float(best),
        between_class_variance=float(between[best]),
        total_variance=variance_total,
        foreground_fraction=float(w1[best]),
        _plateau_width=int(plateau.size),
    )


@register(
    "segment.otsu",
    "Otsu threshold",
    units=(6,),
    rationale=(
        "Chooses the threshold maximising between-class variance, so the level "
        "comes from the histogram rather than from a person. On a background "
        "difference image it adapts frame by frame to changing light - and its "
        "separability says when the histogram was not bimodal enough for the "
        "answer to mean anything."
    ),
    params=(
        ParamSpec(
            "offset", "Threshold offset", "float", 0.0, -60.0, 60.0, 1.0,
            description="Added to the computed level, to bias towards or against foreground.",
        ),
        ParamSpec("invert", "Invert", "bool", False),
        ParamSpec(
            "plateau_midpoint", "Centre an empty gap", "bool", True,
            description=(
                "Take the middle of the plateau of maximal between-class "
                "variance rather than its lowest level. Off reproduces the "
                "textbook argmax exactly."
            ),
        ),
    ),
    category="segment",
)
def otsu(
    image: AnyArray,
    offset: float = 0.0,
    invert: bool = False,
    plateau_midpoint: bool = True,
) -> BoolArray:
    """Binary mask from Otsu's threshold."""
    result = otsu_threshold(image, plateau_midpoint=plateau_midpoint)
    grey = _to_grey_u8(image).astype(np.float64)
    level = result.threshold + offset
    return np.asarray(grey <= level if invert else grey > level, dtype=bool)


@register(
    "segment.adaptive_threshold",
    "Adaptive threshold",
    units=(6,),
    rationale=(
        "Thresholds each pixel against a local neighbourhood mean rather than a "
        "global level. Where illumination varies across the frame - a road half "
        "in shadow, or a pool of headlight - no single global level works, and "
        "this is the comparison that shows it."
    ),
    params=(
        ParamSpec("block_size", "Neighbourhood", "odd_int", 31, 3, 201, 2, unit="px"),
        ParamSpec("offset", "Offset", "float", 5.0, -60.0, 60.0, 1.0),
        ParamSpec("method", "Weighting", "enum", "gaussian", choices=("mean", "gaussian")),
        ParamSpec("invert", "Invert", "bool", False),
    ),
    category="segment",
)
def adaptive_threshold(
    image: AnyArray,
    block_size: int = 31,
    offset: float = 5.0,
    method: str = "gaussian",
    invert: bool = False,
) -> BoolArray:
    """Local adaptive threshold."""
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError(f"block size must be odd and at least 3, got {block_size}")
    grey = _to_grey_u8(image)
    adaptive = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    )
    kind = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    out = cv2.adaptiveThreshold(grey, 255, adaptive, kind, int(block_size), float(offset))
    return np.asarray(out > 0, dtype=bool)


# --------------------------------------------------------------------------
# Connected components, including m-connectivity
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComponentStats:
    """Geometry of one labelled component."""

    label: int
    area_px: int
    x_px: int
    y_px: int
    w_px: int
    h_px: int
    centroid_px: tuple[float, float]

    @property
    def fill_ratio(self) -> float:
        """Area over bounding-box area.

        A single compact vehicle fills most of its box; two vehicles bridged by
        a diagonal shadow fill much less of theirs, so this is one of the cheap
        signals that a blob needs splitting.
        """
        box = self.w_px * self.h_px
        return self.area_px / box if box else 0.0


@dataclass(frozen=True, slots=True)
class LabelResult:
    """A labelled image plus its component statistics."""

    labels: Int32Array
    count: int
    stats: tuple[ComponentStats, ...]
    connectivity: Connectivity

    def mask_for(self, label: int) -> BoolArray:
        return np.asarray(self.labels == label, dtype=bool)


def _m_connectivity_edges(mask: BoolArray) -> tuple[AnyArray, AnyArray]:
    """Edge list for m-adjacency over a binary mask.

    Two foreground pixels are m-adjacent when either

    * ``q`` is a 4-neighbour of ``p``, or
    * ``q`` is a diagonal neighbour of ``p`` **and** the two pixels they both
      touch in the 4-sense are background.

    The second clause is the whole point. Under 8-connectivity a diagonal pair
    is linked even when there is already a 4-connected route through a shared
    neighbour, so between two pixels there can be several distinct shortest
    paths, and a diagonal line ends up connected *and* its background ends up
    connected through the same corners. Admitting the diagonal only when no
    4-route exists removes that ambiguity.

    It does not change the components, though, and the implementation makes that
    visible: whenever the diagonal clause is refused, it is refused precisely
    because a shared 4-neighbour is foreground - which the first clause has
    already linked through. So the m-partition equals the 8-partition, always.
    """
    h, w = mask.shape
    index = np.arange(h * w, dtype=np.int64).reshape(h, w)
    rows: list[AnyArray] = []
    cols: list[AnyArray] = []

    # 4-neighbour links: right and down cover every 4-adjacency once.
    right = mask[:, :-1] & mask[:, 1:]
    rows.append(index[:, :-1][right])
    cols.append(index[:, 1:][right])
    down = mask[:-1, :] & mask[1:, :]
    rows.append(index[:-1, :][down])
    cols.append(index[1:, :][down])

    # Down-right diagonals: the shared 4-neighbours are (r, c+1) and (r+1, c).
    both = mask[:-1, :-1] & mask[1:, 1:]
    shared_clear = ~mask[:-1, 1:] & ~mask[1:, :-1]
    dr = both & shared_clear
    rows.append(index[:-1, :-1][dr])
    cols.append(index[1:, 1:][dr])

    # Down-left diagonals: the shared 4-neighbours are (r, c-1) and (r+1, c).
    both = mask[:-1, 1:] & mask[1:, :-1]
    shared_clear = ~mask[:-1, :-1] & ~mask[1:, 1:]
    dl = both & shared_clear
    rows.append(index[:-1, 1:][dl])
    cols.append(index[1:, :-1][dl])

    return np.concatenate(rows), np.concatenate(cols)


def label_components(mask: AnyArray, connectivity: Connectivity = 8) -> LabelResult:
    """Label the connected components of a binary mask.

    4 and 8 use OpenCV's two-pass labelling. m-connectivity has no library
    implementation, so it is built here: the adjacency is expressed as an edge
    list and resolved with a sparse graph traversal, which is the same
    union-find in a different coat and avoids a per-pixel Python loop.
    """
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m > 0
    if m.ndim != 2:
        raise ValueError(f"expected a 2-D mask, got shape {m.shape}")

    if connectivity in (4, 8):
        count, raw_labels = cv2.connectedComponents(
            m.astype(np.uint8), connectivity=int(connectivity)
        )
        labels: Int32Array = np.asarray(raw_labels, dtype=np.int32)
        return LabelResult(
            labels=labels,
            count=int(count) - 1,
            stats=component_stats(labels, int(count) - 1),
            connectivity=connectivity,
        )

    if connectivity != "m":
        raise ValueError(f"connectivity must be 4, 8 or 'm', got {connectivity!r}")

    h, w = m.shape
    if not m.any():
        empty = np.zeros((h, w), dtype=np.int32)
        return LabelResult(empty, 0, (), "m")

    rows, cols = _m_connectivity_edges(m)
    n = h * w
    graph = coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n)
    )
    n_components, component = _csgraph_components(graph, directed=False, return_labels=True)
    component = component.reshape(h, w)

    # Every background pixel is its own singleton component; keep only those
    # containing foreground and renumber from 1.
    used = np.unique(component[m])
    remap = np.zeros(n_components, dtype=np.int32)
    remap[used] = np.arange(1, used.size + 1, dtype=np.int32)
    m_labels: Int32Array = np.asarray(
        np.where(m, remap[component], 0), dtype=np.int32
    )
    return LabelResult(
        labels=m_labels,
        count=int(used.size),
        stats=component_stats(m_labels, int(used.size)),
        connectivity="m",
    )


def component_stats(labels: Int32Array, count: int) -> tuple[ComponentStats, ...]:
    """Area, bounding box and centroid for each label from 1 to ``count``."""
    if count <= 0:
        return ()
    flat = np.asarray(labels).ravel()
    h, w = labels.shape
    ys, xs = np.divmod(np.arange(flat.size), w)

    areas = np.bincount(flat, minlength=count + 1)
    sum_x = np.bincount(flat, weights=xs, minlength=count + 1)
    sum_y = np.bincount(flat, weights=ys, minlength=count + 1)
    min_x = np.full(count + 1, w, dtype=np.int64)
    max_x = np.full(count + 1, -1, dtype=np.int64)
    min_y = np.full(count + 1, h, dtype=np.int64)
    max_y = np.full(count + 1, -1, dtype=np.int64)
    np.minimum.at(min_x, flat, xs)
    np.maximum.at(max_x, flat, xs)
    np.minimum.at(min_y, flat, ys)
    np.maximum.at(max_y, flat, ys)

    out: list[ComponentStats] = []
    for label in range(1, count + 1):
        area = int(areas[label])
        if area == 0:
            continue
        out.append(
            ComponentStats(
                label=label,
                area_px=area,
                x_px=int(min_x[label]),
                y_px=int(min_y[label]),
                w_px=int(max_x[label] - min_x[label] + 1),
                h_px=int(max_y[label] - min_y[label] + 1),
                centroid_px=(float(sum_x[label] / area), float(sum_y[label] / area)),
            )
        )
    return tuple(out)


def connectivity_disagreement(mask: AnyArray) -> dict[str, object]:
    """Compare 4, 8 and m labelling of the same mask.

    Returns the component counts and the pixels where the three disagree about
    which component a pixel belongs to - which the ablation renders directly, so
    the difference between the connectivities is something to look at rather
    than a number to take on trust.

    Expect the 8 and m columns to agree exactly: as explained in the module
    docstring, they induce the same partition. The disagreement mask therefore
    shows where 4-connectivity fragments a component that the other two keep
    whole, which is the difference that actually costs counts.
    """
    results = {str(c): label_components(mask, c) for c in (4, 8, "m")}
    counts = {name: r.count for name, r in results.items()}

    # A pixel is a disagreement point when the partition induced by one
    # connectivity differs from another's: compare the component sizes each
    # assigns to it, which is invariant to how the labels happen to be numbered.
    sizes: dict[str, AnyArray] = {}
    for name, r in results.items():
        area_by_label = np.zeros(r.count + 1, dtype=np.int64)
        for s in r.stats:
            area_by_label[s.label] = s.area_px
        sizes[name] = area_by_label[r.labels]

    disagree = (sizes["4"] != sizes["8"]) | (sizes["8"] != sizes["m"])
    return {
        "counts": counts,
        "disagreement_mask": np.asarray(disagree & (np.asarray(mask) > 0), dtype=bool),
        "disagreement_px": int((disagree & (np.asarray(mask) > 0)).sum()),
        "results": results,
    }


# --------------------------------------------------------------------------
# Region-based segmentation
# --------------------------------------------------------------------------


@register(
    "segment.region_growing",
    "Region growing",
    units=(6,),
    rationale=(
        "Grows a region from seed pixels, admitting a neighbour when it is "
        "within a tolerance of the region's running mean. Unlike a global "
        "threshold it uses spatial continuity, so it can follow a vehicle whose "
        "brightness drifts across its length."
    ),
    params=(
        ParamSpec("tolerance", "Tolerance", "float", 12.0, 0.5, 128.0, 0.5),
        ParamSpec("connectivity", "Connectivity", "enum", "8", choices=("4", "8")),
        ParamSpec("max_iterations", "Maximum iterations", "int", 500, 1, 5000, 10),
    ),
    category="segment",
)
def region_growing(
    image: AnyArray,
    seeds: AnyArray,
    tolerance: float = 12.0,
    connectivity: str = "8",
    max_iterations: int = 500,
) -> BoolArray:
    """Grow a region from seeds by intensity similarity.

    The comparison is against the *running mean of the region*, not against the
    seed value. Comparing against the seed makes the result depend heavily on
    which pixel happened to be chosen; comparing against the mean lets the
    region track a gradual change without drifting arbitrarily far, because
    every admitted pixel pulls the mean.

    The iteration cap is a real bound, not a formality: a tolerance set too high
    grows to the whole frame, and doing that inside a frame handler should stop
    rather than stall the stream.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")
    grey = _to_grey_u8(image).astype(np.float64)
    region = np.asarray(seeds, dtype=bool).copy()
    if region.shape != grey.shape:
        raise ValueError(f"seed shape {region.shape} does not match image shape {grey.shape}")
    if not region.any():
        raise ValueError("region growing needs at least one seed pixel")

    element: AnyArray = np.ones((3, 3), np.uint8)
    if connectivity == "4":
        element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

    for _ in range(max(1, int(max_iterations))):
        mean = float(grey[region].mean())
        candidates = (
            cv2.dilate(region.astype(np.uint8), element, borderValue=0) > 0
        ) & ~region
        admitted = candidates & (np.abs(grey - mean) <= tolerance)
        if not admitted.any():
            break
        region |= admitted
    return region


def split_and_merge(
    image: AnyArray, threshold: float = 12.0, min_size: int = 8
) -> Int32Array:
    """Quadtree split-and-merge segmentation.

    Split: recursively quarter any block whose intensity range exceeds
    ``threshold``, down to ``min_size``. Merge: join adjacent leaf blocks whose
    means are within the same tolerance, using the same edge-list traversal as
    the m-connectivity labelling above.

    Returns a label image. Included for completeness of Unit 6 and as a
    comparison for the watershed; the production splitter in S6 uses the
    distance-transform watershed, which respects vehicle shape rather than the
    quadtree grid.
    """
    grey = _to_grey_u8(image).astype(np.float64)
    h, w = grey.shape
    labels = np.zeros((h, w), dtype=np.int32)
    blocks: list[tuple[int, int, int, int]] = []

    def split(y: int, x: int, bh: int, bw: int) -> None:
        block = grey[y : y + bh, x : x + bw]
        if block.size == 0:
            return
        homogeneous = float(block.max() - block.min()) <= threshold
        if homogeneous or bh <= min_size or bw <= min_size:
            blocks.append((y, x, bh, bw))
            return
        hy, hx = bh // 2, bw // 2
        split(y, x, hy, hx)
        split(y, x + hx, hy, bw - hx)
        split(y + hy, x, bh - hy, hx)
        split(y + hy, x + hx, bh - hy, bw - hx)

    split(0, 0, h, w)

    means = np.zeros(len(blocks) + 1, dtype=np.float64)
    for i, (y, x, bh, bw) in enumerate(blocks, start=1):
        labels[y : y + bh, x : x + bw] = i
        means[i] = float(grey[y : y + bh, x : x + bw].mean())

    # Merge adjacent blocks with similar means.
    rows: list[AnyArray] = []
    cols: list[AnyArray] = []
    for a, b in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        differ = a != b
        pa, pb = a[differ], b[differ]
        similar = np.abs(means[pa] - means[pb]) <= threshold
        rows.append(pa[similar])
        cols.append(pb[similar])

    n = len(blocks) + 1
    graph = coo_matrix(
        (np.ones(sum(r.size for r in rows), dtype=np.int8),
         (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, n),
    )
    _, merged = _csgraph_components(graph, directed=False, return_labels=True)
    return np.asarray(merged[labels], dtype=np.int32)


@register(
    "segment.watershed_split",
    "Watershed split",
    units=(6,),
    rationale=(
        "Separates blobs that touch by flooding the distance transform from its "
        "local maxima. In S6 this is what recovers two vehicles that a shadow "
        "bridge or a short following distance merged into one component, when "
        "the merged footprint is implausibly large for a single vehicle."
    ),
    params=(
        ParamSpec(
            "min_distance_ratio", "Marker distance", "float", 0.45, 0.05, 0.95, 0.05,
            description="Seed markers at this fraction of the peak distance transform.",
        ),
        ParamSpec("blur", "Distance smoothing", "float", 1.5, 0.0, 8.0, 0.1, unit="px"),
    ),
    category="segment",
)
def watershed_split(
    mask: AnyArray, min_distance_ratio: float = 0.45, blur: float = 1.5
) -> Int32Array:
    """Split touching blobs with a distance-transform watershed.

    The distance transform is large in the middle of each blob and small at the
    neck where two blobs touch, so thresholding it at a fraction of its peak
    isolates one marker per blob. Flooding from those markers puts the boundary
    at the neck, which is where a person would draw it.

    ``min_distance_ratio`` is the knob that matters: too high and a small
    vehicle loses its marker entirely, too low and a single vehicle with a
    slightly ragged mask gets two.
    """
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m > 0
    if not m.any():
        return np.zeros(m.shape, dtype=np.int32)

    distance = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    if blur > 0:
        distance = cv2.GaussianBlur(distance, (0, 0), sigmaX=float(blur))
    peak = float(distance.max())
    if peak <= 0:
        return np.zeros(m.shape, dtype=np.int32)

    sure_foreground = distance >= min_distance_ratio * peak
    marker_count, markers = cv2.connectedComponents(sure_foreground.astype(np.uint8))
    if marker_count <= 2:
        # One marker: nothing to split. Return the mask as a single component
        # rather than an empty result the caller would read as a lost vehicle.
        return np.asarray(m.astype(np.int32), dtype=np.int32)

    markers = markers.astype(np.int32) + 1
    unknown = m & ~sure_foreground
    markers[unknown] = 0
    markers[~m] = 1

    colour = cv2.cvtColor((m.astype(np.uint8)) * 255, cv2.COLOR_GRAY2BGR)
    cv2.watershed(colour, markers)

    out = np.where(m & (markers > 1), markers - 1, 0).astype(np.int32)
    # Watershed marks its boundaries -1; give those pixels to a neighbour so the
    # split does not carve a one-pixel gap out of every vehicle's area.
    boundary = m & (markers == -1)
    if boundary.any():
        grown = cv2.dilate(out.astype(np.float32), np.ones((3, 3), np.uint8)).astype(np.int32)
        out = np.where(boundary, grown, out)
    return np.asarray(out, dtype=np.int32)


def distance_transform(mask: AnyArray) -> FloatArray:
    """Euclidean distance from each foreground pixel to the nearest background."""
    m = np.asarray(mask)
    if m.dtype != np.bool_:
        m = m > 0
    return np.asarray(
        cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5), dtype=np.float64
    )
