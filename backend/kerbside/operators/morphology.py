"""Unit 4 - morphology.

Morphology asks a shape question of a shape: does this structuring element fit
inside the object here, and does it hit the object here. Everything else -
opening, closing, boundaries, thinning, reconstruction - is built from those two
questions.

This is the core of foreground cleanup in S4. A raw background difference is
speckled with noise and hollow where a windscreen happened to match the road
behind it. Opening with a small element removes speckle because noise is smaller
than the element; closing with a larger one fills the interior because the holes
are smaller than *that* element. Both statements are about size, which is the
problem: under perspective, "small" near the camera and "small" at the horizon
differ by more than an order of magnitude.

:func:`banded_morphology` is the answer to that. It applies the operation in
horizontal bands with the element resized per band from the homography, so a
half-metre element stays half a metre wherever it is applied. That is trap 4
handled by construction; a single global element either merges adjacent lanes in
the near field or leaves distant vehicles hollow, and no value of it does
neither.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, BoolArray, UInt8Array

if TYPE_CHECKING:
    from kerbside.geometry.perspective import PerspectiveModel

__all__ = [
    "SEShape",
    "MorphOp",
    "structuring_element",
    "erode",
    "dilate",
    "opening",
    "closing",
    "morph",
    "hit_or_miss",
    "boundary",
    "thin",
    "thicken",
    "reconstruct_by_dilation",
    "fill_holes",
    "fill_small_holes",
    "top_hat",
    "bottom_hat",
    "banded_morphology",
    "remove_small_components",
]

SEShape = Literal["rect", "ellipse", "cross"]
MorphOp = Literal["erode", "dilate", "open", "close"]

_CV_SHAPE = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


def _as_binary_u8(mask: AnyArray) -> UInt8Array:
    """Coerce a mask to 0/255 uint8, which is what OpenCV's morphology wants."""
    m = np.asarray(mask)
    if m.dtype == np.bool_:
        return (m.astype(np.uint8)) * 255
    if m.dtype != np.uint8:
        m = np.clip(m, 0, 255).astype(np.uint8)
    return np.where(m > 0, 255, 0).astype(np.uint8)


def structuring_element(
    width: int, height: int | None = None, shape: SEShape = "ellipse"
) -> UInt8Array:
    """Build a structuring element.

    ``width`` and ``height`` are separate on purpose. On an oblique road view a
    vehicle projects wider than it is long, so an isotropic element that is
    large enough to close a vehicle vertically is large enough to bridge two
    lanes horizontally. Both must be odd so the element has a centre pixel; an
    even one shifts the result by half a pixel in a way that is nearly
    impossible to see and easy to blame on something else.
    """
    height = width if height is None else height
    if width < 1 or height < 1 or width % 2 == 0 or height % 2 == 0:
        raise ValueError(
            f"structuring element sides must be positive and odd so it has a "
            f"centre pixel, got {width}x{height}"
        )
    if shape not in _CV_SHAPE:
        raise ValueError(
            f"unknown structuring element shape {shape!r}; "
            f"use one of {list(_CV_SHAPE)}"
        )
    return np.asarray(
        cv2.getStructuringElement(_CV_SHAPE[shape], (width, height)), dtype=np.uint8
    )


def erode(image: AnyArray, element: AnyArray, iterations: int = 1) -> AnyArray:
    """Erosion: keep a pixel only where the element fits entirely inside the set.

    Shrinks objects, removes anything smaller than the element, and separates
    shapes joined by a thin bridge - which is what makes it the first half of an
    opening.
    """
    return cv2.erode(
        np.asarray(image), np.asarray(element, dtype=np.uint8),
        iterations=int(iterations), borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


def dilate(image: AnyArray, element: AnyArray, iterations: int = 1) -> AnyArray:
    """Dilation: keep a pixel where the reflected element hits the set at all.

    Grows objects and fills gaps narrower than the element.
    """
    return cv2.dilate(
        np.asarray(image), np.asarray(element, dtype=np.uint8),
        iterations=int(iterations), borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


def opening(image: AnyArray, element: AnyArray) -> AnyArray:
    """Erosion then dilation. Removes small objects, keeps large ones the size
    they were - which a plain erosion would not."""
    return cv2.morphologyEx(
        np.asarray(image), cv2.MORPH_OPEN, np.asarray(element, dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


def closing(image: AnyArray, element: AnyArray) -> AnyArray:
    """Dilation then erosion. Fills small holes and gaps, keeps the outer
    boundary where it was."""
    return cv2.morphologyEx(
        np.asarray(image), cv2.MORPH_CLOSE, np.asarray(element, dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT, borderValue=0,
    )


@register(
    "morphology.morph",
    "Morphological operation",
    units=(4,),
    rationale=(
        "Erosion, dilation, opening and closing with a chosen structuring "
        "element. Opening removes the speckle a background difference leaves "
        "behind; closing fills vehicle interiors, where windows often match the "
        "road better than the bodywork does."
    ),
    params=(
        ParamSpec("op", "Operation", "enum", "open", choices=("erode", "dilate", "open", "close")),
        ParamSpec("width", "Element width", "odd_int", 3, 1, 61, 2, unit="px"),
        ParamSpec("height", "Element height", "odd_int", 3, 1, 61, 2, unit="px"),
        ParamSpec(
            "shape", "Element shape", "enum", "ellipse",
            choices=("rect", "ellipse", "cross"),
        ),
        ParamSpec("iterations", "Iterations", "int", 1, 1, 10, 1),
    ),
    category="morphology",
)
def morph(
    image: AnyArray,
    op: MorphOp = "open",
    width: int = 3,
    height: int = 3,
    shape: SEShape = "ellipse",
    iterations: int = 1,
) -> AnyArray:
    """Apply a named morphological operation.

    Works on binary masks and on greyscale alike: greyscale erosion is a local
    minimum and greyscale dilation a local maximum, which is the same fit-and-hit
    definition applied to the umbra of the surface. That is what makes greyscale
    opening a usable illumination-gradient estimator in S3.
    """
    element = structuring_element(width, height, shape)
    out = np.asarray(image)
    for _ in range(max(1, int(iterations))):
        if op == "erode":
            out = erode(out, element)
        elif op == "dilate":
            out = dilate(out, element)
        elif op == "open":
            out = opening(out, element)
        elif op == "close":
            out = closing(out, element)
        else:
            raise ValueError(f"unknown operation {op!r}; use erode, dilate, open or close")
    return out


def hit_or_miss(mask: AnyArray, foreground: AnyArray, background: AnyArray) -> BoolArray:
    """Hit-or-miss transform: match a pattern of set *and* unset pixels.

    ``A (*) B = (A erode B_fg) and (A^c erode B_bg)``

    Unlike erosion, which only asks what must be present, this also asks what
    must be absent, so it detects an exact local configuration - an isolated
    pixel, a corner, a line end. It is the basis of the thinning below.
    """
    fg = np.asarray(foreground, dtype=np.uint8)
    bg = np.asarray(background, dtype=np.uint8)
    if np.any((fg > 0) & (bg > 0)):
        raise ValueError(
            "the foreground and background parts of a hit-or-miss element "
            "overlap; a pixel cannot be required to be both set and unset"
        )
    m = _as_binary_u8(mask)
    fits = erode(m, fg)
    misses = erode(cv2.bitwise_not(m), bg)
    return np.asarray(cv2.bitwise_and(fits, misses) > 0, dtype=bool)


@register(
    "morphology.boundary",
    "Boundary extraction",
    units=(4,),
    rationale=(
        "The set minus its own erosion, which leaves a one-pixel outline. Used "
        "to draw vehicle outlines in the overlay and to measure the perimeter "
        "that distinguishes a compact blob from a ragged shadow-merged one."
    ),
    params=(
        ParamSpec("width", "Element width", "odd_int", 3, 3, 21, 2, unit="px"),
        ParamSpec("shape", "Element shape", "enum", "rect", choices=("rect", "ellipse", "cross")),
        ParamSpec(
            "inner", "Inner boundary", "bool", True,
            description="Inner keeps pixels of the object; outer keeps pixels just outside it.",
        ),
    ),
    category="morphology",
)
def boundary(
    mask: AnyArray, width: int = 3, shape: SEShape = "rect", inner: bool = True
) -> BoolArray:
    """Morphological boundary of a binary mask."""
    m = _as_binary_u8(mask)
    element = structuring_element(width, width, shape)
    if inner:
        return np.asarray(cv2.subtract(m, erode(m, element)) > 0, dtype=bool)
    return np.asarray(cv2.subtract(dilate(m, element), m) > 0, dtype=bool)


# Zhang-Suen thinning is expressed as two alternating sub-iterations. Both are
# written out below rather than tabulated, because the conditions are the whole
# content of the algorithm and a lookup table hides them completely.


def _neighbours(padded: BoolArray) -> tuple[BoolArray, ...]:
    """The eight neighbours P2..P9 in clockwise order from north."""
    return (
        padded[:-2, 1:-1],  # P2 north
        padded[:-2, 2:],  # P3 north-east
        padded[1:-1, 2:],  # P4 east
        padded[2:, 2:],  # P5 south-east
        padded[2:, 1:-1],  # P6 south
        padded[2:, :-2],  # P7 south-west
        padded[1:-1, :-2],  # P8 west
        padded[:-2, :-2],  # P9 north-west
    )


def _thin_pass(mask: BoolArray, step: int) -> BoolArray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    p2, p3, p4, p5, p6, p7, p8, p9 = _neighbours(padded)

    # B(P1): how many neighbours are set. Between 2 and 6 keeps end points and
    # rejects interior points.
    b = p2.astype(np.int8) + p3 + p4 + p5 + p6 + p7 + p8 + p9

    # A(P1): the number of 0-to-1 transitions going round the neighbourhood.
    # Exactly one is what guarantees removing the pixel cannot disconnect the
    # shape - this is the condition that makes thinning topology preserving.
    seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
    a = np.zeros(mask.shape, dtype=np.int8)
    for first, second in zip(seq[:-1], seq[1:], strict=True):
        a += (~first & second).astype(np.int8)

    if step == 0:
        c1 = ~(p2 & p4 & p6)
        c2 = ~(p4 & p6 & p8)
    else:
        c1 = ~(p2 & p4 & p8)
        c2 = ~(p2 & p6 & p8)

    removable = mask & (b >= 2) & (b <= 6) & (a == 1) & c1 & c2
    return np.asarray(mask & ~removable, dtype=bool)


@register(
    "morphology.thin",
    "Thinning",
    units=(4,),
    rationale=(
        "Zhang-Suen thinning reduces a region to a one-pixel skeleton while "
        "preserving its connectivity. On lane markings it turns a painted stripe "
        "of varying width into a centre line, which is a far better input to a "
        "Hough transform than the stripe's two parallel edges."
    ),
    params=(ParamSpec("max_iterations", "Maximum iterations", "int", 50, 1, 500, 1),),
    category="morphology",
)
def thin(mask: AnyArray, max_iterations: int = 50) -> BoolArray:
    """Zhang-Suen thinning to a one-pixel-wide skeleton.

    Iterates two complementary sub-passes until nothing changes or the iteration
    budget runs out. The budget is not a formality: a pathological input can
    converge slowly, and an unbounded loop inside a frame handler would stall
    the stream rather than produce a bad result you could see.
    """
    m = np.asarray(_as_binary_u8(mask) > 0)
    for _ in range(max(1, int(max_iterations))):
        before = m
        m = _thin_pass(m, 0)
        m = _thin_pass(m, 1)
        if np.array_equal(m, before):
            break
    return m


def thicken(mask: AnyArray, max_iterations: int = 50) -> BoolArray:
    """Thickening, as thinning of the complement.

    The duality is exact and is the reason there is no separate implementation:
    growing a set is shrinking its background.
    """
    inverted = np.asarray(_as_binary_u8(mask) == 0, dtype=bool)
    return np.asarray(~thin(inverted, max_iterations), dtype=bool)


def reconstruct_by_dilation(
    marker: AnyArray, mask: AnyArray, element: AnyArray | None = None
) -> BoolArray:
    """Geodesic reconstruction: repeatedly dilate the marker, clipped to the mask.

    The result is exactly those connected components of ``mask`` that contain at
    least one marker pixel - whole components, at their original shape. That is
    what makes it strictly better than an opening for removing objects: an
    opening erodes what it keeps, reconstruction does not.
    """
    se = (
        structuring_element(3, 3, "cross")
        if element is None
        else np.asarray(element, np.uint8)
    )
    m = _as_binary_u8(mask)
    current = cv2.bitwise_and(_as_binary_u8(marker), m)
    while True:
        grown = cv2.bitwise_and(dilate(current, se), m)
        if np.array_equal(grown, current):
            return np.asarray(grown > 0, dtype=bool)
        current = grown


@register(
    "morphology.fill_holes",
    "Hole filling",
    units=(4,),
    rationale=(
        "Fills every enclosed background region by reconstructing the "
        "background from the image border: whatever the border cannot reach is a "
        "hole. In S4 this recovers vehicle interiors that a closing was too "
        "small to bridge, without enlarging the vehicle outline the way a larger "
        "closing would."
    ),
    category="morphology",
)
def fill_holes(mask: AnyArray) -> BoolArray:
    """Fill holes in a binary mask: every background region the border cannot reach.

    Implemented as a flood fill from outside rather than as an iterated geodesic
    reconstruction. The two give identical results, but reconstruction converges
    only after as many dilations as the longest path inside the region, which on
    a 720p mask is hundreds of passes over the whole frame and dominated the
    frame budget; the flood fill is a single traversal.

    The image is padded by one pixel first, so a hole that touches the frame
    edge is still reachable from outside and is correctly left unfilled.
    """
    m = np.asarray(_as_binary_u8(mask) > 0)
    h, w = m.shape
    flood = np.zeros((h + 2, w + 2), np.uint8)
    flood[1:-1, 1:-1] = m.astype(np.uint8)
    # cv2.floodFill needs its own one-pixel border on the mask argument.
    guard = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(flood, guard, (0, 0), 2)
    outside = flood[1:-1, 1:-1] == 2
    return np.asarray(m | ~outside, dtype=bool)


def fill_small_holes(mask: AnyArray, max_area_px: int) -> BoolArray:
    """Fill enclosed background regions smaller than ``max_area_px``.

    :func:`fill_holes` fills every hole regardless of size, which is right when
    the holes are windows in a vehicle and catastrophic when one of them is a
    shadow that another stage just removed: subtracting a shadow from the middle
    of a larger foreground region leaves an enclosed hole, and an unconditional
    fill puts the whole shadow straight back.

    Bounding the fill by area separates the two cases cleanly, because they
    differ by orders of magnitude - a windscreen is hundreds of pixels and a
    cast shadow is tens of thousands.
    """
    if max_area_px < 0:
        raise ValueError(f"maximum hole area must be non-negative, got {max_area_px}")
    m = np.asarray(_as_binary_u8(mask) > 0)
    if max_area_px == 0:
        return m

    background = ~m
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        background.astype(np.uint8), connectivity=4
    )
    if count <= 1:
        return m

    # Any background component touching the border is outside the object, not a
    # hole, however small it happens to be inside the frame.
    border = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
    fill = np.zeros(count, dtype=bool)
    for label in range(1, count):
        if label in border:
            continue
        fill[label] = stats[label, cv2.CC_STAT_AREA] <= max_area_px
    return np.asarray(m | fill[labels], dtype=bool)


def top_hat(image: AnyArray, element: AnyArray) -> AnyArray:
    """Image minus its opening: bright detail smaller than the element.

    On a road surface with an illumination gradient, a greyscale opening with a
    large element estimates the gradient, so the top hat is the surface detail
    with the gradient removed. That is a background estimate built entirely from
    morphology, and S3 offers it alongside the temporal models.
    """
    return cv2.morphologyEx(
        np.asarray(image), cv2.MORPH_TOPHAT, np.asarray(element, dtype=np.uint8)
    )


def bottom_hat(image: AnyArray, element: AnyArray) -> AnyArray:
    """Closing minus the image: dark detail smaller than the element."""
    return cv2.morphologyEx(
        np.asarray(image), cv2.MORPH_BLACKHAT, np.asarray(element, dtype=np.uint8)
    )


def banded_morphology(
    mask: AnyArray,
    model: PerspectiveModel,
    size_m: float,
    op: MorphOp = "close",
    bands: int = 12,
    shape: SEShape = "ellipse",
    min_px: int = 3,
    max_px: int = 41,
    vertical: str = "along",
) -> AnyArray:
    """Apply a morphological operation with a perspective-scaled element.

    The image is split into ``bands`` horizontal strips and each is processed
    with an element sized for its own distance, so ``size_m`` stays a real-world
    size everywhere rather than a pixel count that means something different at
    the top of the frame than at the bottom.

    Strips overlap by the element's own half-height and are recombined on the
    non-overlapping interiors, so no band boundary appears in the output. That
    detail matters: a visible seam every fiftieth row would be picked up as a
    horizontal edge by S6 and would fragment blobs that straddle it.

    ``bands`` trades accuracy against cost. Twelve bands over a 720-line frame
    holds the element within a few percent of the exact per-row size, at a
    twelfth of the cost of doing it per row.

    ``vertical`` is passed through to
    :meth:`~kerbside.geometry.perspective.PerspectiveModel.se_size_px` and
    decides whether the element's height measures distance along the road or
    height above it. Use ``"height"`` whenever the target is a vehicle
    silhouette rather than something lying flat on the road.
    """
    if bands < 1:
        raise ValueError(f"need at least one band, got {bands}")
    if size_m <= 0:
        raise ValueError(f"structuring element size must be positive, got {size_m} m")

    m = np.asarray(mask)
    height = m.shape[0]
    out = m.copy()
    edges = np.linspace(0, height, bands + 1).round().astype(int)

    for i in range(bands):
        y0, y1 = int(edges[i]), int(edges[i + 1])
        if y1 <= y0:
            continue
        centre_row = (y0 + y1) / 2.0
        if centre_row <= model.horizon_row_px + 1.0:
            continue
        w, h = model.se_size_px(
            centre_row, size_m, min_px=min_px, max_px=max_px, vertical=vertical
        )
        halo = h // 2 + 1
        a0, a1 = max(0, y0 - halo), min(height, y1 + halo)
        strip = m[a0:a1]
        processed = morph(strip, op, w, h, shape)
        out[y0:y1] = processed[y0 - a0 : y1 - a0]
    return out


def remove_small_components(
    mask: AnyArray, min_area_px: int, connectivity: int = 8
) -> BoolArray:
    """Drop connected components below an area, keeping the rest untouched.

    Preferred over an opening when the aim is purely to delete small things: an
    opening also rounds the corners of everything it keeps, which changes the
    blob areas the perspective size filter is about to judge.
    """
    if connectivity not in (4, 8):
        raise ValueError(f"OpenCV labelling supports 4 or 8 connectivity, got {connectivity}")
    m = _as_binary_u8(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=connectivity)
    keep = np.zeros(count, dtype=bool)
    for label in range(1, count):
        keep[label] = stats[label, cv2.CC_STAT_AREA] >= min_area_px
    return np.asarray(keep[labels], dtype=bool)
