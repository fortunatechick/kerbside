"""Unit 5 - edge detection.

An edge is a place where intensity changes quickly, so every operator here is a
derivative. First-order operators (Roberts, Prewitt, Sobel) estimate the
gradient and mark edges where its magnitude is large. Second-order operators
(Laplacian, Laplacian of Gaussian) mark edges where the second derivative
crosses zero, which locates them more precisely and responds to noise far more
enthusiastically.

Two jobs in this pipeline. Lane-marking detection - Canny followed by a Hough
transform - recovers the lane geometry that the calibration view proposes to the
user. And edge evidence refines blob boundaries under motion blur, where the
background difference alone gives a silhouette smeared along the direction of
travel.

Canny is written out here rather than called, because it is the flagship of the
unit and because each of its four stages is a decision worth being able to point
at: smooth, differentiate, thin by non-maximum suppression, then threshold with
hysteresis so that a weak edge survives if and only if it is connected to a
strong one. The tests check it against OpenCV's implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, BoolArray, FloatArray, UInt8Array
from kerbside.operators.morphology import reconstruct_by_dilation, structuring_element
from kerbside.operators.point import clip_to_uint8
from kerbside.operators.spatial import LAPLACIAN_4, LAPLACIAN_8, correlate2d, gaussian_filter

__all__ = [
    "EdgeOperator",
    "GradientResult",
    "HoughLine",
    "ROBERTS_X",
    "ROBERTS_Y",
    "PREWITT_X",
    "PREWITT_Y",
    "SOBEL_X",
    "SOBEL_Y",
    "gradient",
    "edge_magnitude",
    "laplacian_of_gaussian_kernel",
    "log_edges",
    "zero_crossings",
    "canny",
    "edge_map",
    "compare_edge_operators",
    "hough_lines",
    "draw_hough_lines",
]

EdgeOperator = Literal["roberts", "prewitt", "sobel", "laplacian", "log", "canny"]

ROBERTS_X = np.array([[1, 0], [0, -1]], dtype=np.float64)
ROBERTS_Y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
"""Roberts cross: the smallest possible difference operator, on the diagonals.
Cheap, and with a 2x2 support it has no centre pixel, so the estimate sits
half a pixel off the grid - which is why it fell out of use."""

PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
"""Prewitt: a central difference averaged over three rows, so it smooths across
the edge as it differentiates along it."""

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
"""Sobel: Prewitt with the centre row weighted double, which is a small
triangular smoothing and makes it noticeably less noise-sensitive."""


def _to_grey(image: AnyArray) -> FloatArray:
    x = np.asarray(image)
    if x.ndim == 3:
        x = cv2.cvtColor(clip_to_uint8(x), cv2.COLOR_BGR2GRAY)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {x.shape}")
    return np.asarray(x, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GradientResult:
    """Gradient components with the derived magnitude and direction."""

    gx: FloatArray
    gy: FloatArray

    @property
    def magnitude(self) -> FloatArray:
        return np.hypot(self.gx, self.gy)

    @property
    def magnitude_l1(self) -> FloatArray:
        """City-block magnitude, ``|gx| + |gy|``.

        The classical cheap approximation - no square root - and a direct use of
        the Unit 1 distance measures. It overestimates diagonal edges by up to
        41%, which is visible in the comparison study.
        """
        return np.asarray(np.abs(self.gx) + np.abs(self.gy), dtype=np.float64)

    @property
    def direction_rad(self) -> FloatArray:
        return np.arctan2(self.gy, self.gx)

    @property
    def direction_deg(self) -> FloatArray:
        return np.rad2deg(self.direction_rad)


def gradient(image: AnyArray, operator: EdgeOperator = "sobel") -> GradientResult:
    """First-order gradient by the named operator.

    Roberts is padded to a 3x3 support so its output is aligned with the others;
    without that its half-pixel offset would put its edges one pixel from
    everyone else's in the comparison, which reads as an accuracy difference
    rather than the sampling difference it is.
    """
    grey = _to_grey(image)
    if operator == "sobel":
        kx, ky = SOBEL_X, SOBEL_Y
    elif operator == "prewitt":
        kx, ky = PREWITT_X, PREWITT_Y
    elif operator == "roberts":
        kx = np.zeros((3, 3))
        ky = np.zeros((3, 3))
        kx[1:, 1:] = ROBERTS_X
        ky[1:, 1:] = ROBERTS_Y
    else:
        raise ValueError(
            f"{operator!r} is not a first-order gradient operator; use roberts, "
            f"prewitt or sobel"
        )
    return GradientResult(gx=correlate2d(grey, kx), gy=correlate2d(grey, ky))


def edge_magnitude(image: AnyArray, operator: EdgeOperator = "sobel") -> UInt8Array:
    """Gradient magnitude, normalised for display."""
    mag = gradient(image, operator).magnitude
    peak = float(mag.max())
    if peak < 1e-9:
        return np.zeros(mag.shape, dtype=np.uint8)
    return clip_to_uint8(mag * (255.0 / peak))


def laplacian_of_gaussian_kernel(size: int, sigma: float = 1.4) -> FloatArray:
    """Laplacian of Gaussian, from its analytic form.

    ``LoG(x, y) = -1/(pi s^4) [1 - (x^2+y^2)/(2 s^2)] exp(-(x^2+y^2)/(2 s^2))``

    Smoothing and differentiating in one kernel, which is both cheaper than
    doing them separately and better defined: the scale at which edges are
    sought is the single parameter ``sigma``. The kernel is made exactly
    zero-sum so a flat region gives exactly zero response - otherwise a constant
    offset leaks through and shifts every zero crossing.
    """
    if size < 3 or size % 2 == 0:
        raise ValueError(f"LoG kernel size must be odd and at least 3, got {size}")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax, indexing="ij")
    r2 = xx**2 + yy**2
    s2 = sigma * sigma
    kernel = (r2 - 2.0 * s2) / (s2 * s2) * np.exp(-r2 / (2.0 * s2))
    return np.asarray(kernel - kernel.mean(), dtype=np.float64)


def zero_crossings(response: AnyArray, threshold: float = 0.0) -> BoolArray:
    """Mark pixels where a signed response changes sign against a neighbour.

    ``threshold`` requires a minimum slope across the crossing. Without it every
    flat region produces a dense mesh of crossings from numerical noise, which
    is the usual reason a LoG edge map looks like static.
    """
    r = np.asarray(response, dtype=np.float64)
    out = np.zeros(r.shape, dtype=bool)
    for shift_axis, shift in ((0, 1), (1, 1), (0, -1), (1, -1)):
        rolled = np.roll(r, shift, axis=shift_axis)
        crossing = (np.sign(r) != np.sign(rolled)) & (np.abs(r - rolled) > threshold)
        out |= crossing
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = False
    return out


@register(
    "edges.log",
    "Laplacian of Gaussian",
    units=(5,),
    rationale=(
        "Smooths and takes the second derivative in one kernel, then marks the "
        "zero crossings. It localises edges more precisely than a gradient "
        "threshold and produces closed contours, at the cost of responding "
        "strongly to noise - which is why sigma, not a threshold, is the "
        "parameter that matters."
    ),
    params=(
        ParamSpec("size", "Kernel size", "odd_int", 9, 3, 31, 2, unit="px"),
        ParamSpec("sigma", "Sigma", "float", 1.4, 0.4, 8.0, 0.1, unit="px"),
        ParamSpec("threshold", "Slope threshold", "float", 2.0, 0.0, 60.0, 0.5),
    ),
    category="edges",
)
def log_edges(
    image: AnyArray, size: int = 9, sigma: float = 1.4, threshold: float = 2.0
) -> BoolArray:
    """Edges as zero crossings of the Laplacian of Gaussian."""
    response = correlate2d(_to_grey(image), laplacian_of_gaussian_kernel(size, sigma))
    return zero_crossings(response, threshold)


@register(
    "edges.canny",
    "Canny",
    units=(5,),
    rationale=(
        "Smooth, differentiate, thin by non-maximum suppression, then threshold "
        "with hysteresis so a weak edge survives only where it connects to a "
        "strong one. The hysteresis is what makes it usable on lane markings: a "
        "marking that fades in the distance stays one continuous line instead of "
        "breaking into fragments a Hough transform would miss."
    ),
    params=(
        ParamSpec("low", "Low threshold", "float", 40.0, 0.0, 500.0, 1.0),
        ParamSpec("high", "High threshold", "float", 110.0, 0.0, 1000.0, 1.0),
        ParamSpec("blur_size", "Gaussian size", "odd_int", 5, 3, 21, 2, unit="px"),
        ParamSpec("blur_sigma", "Gaussian sigma", "float", 1.4, 0.1, 8.0, 0.1, unit="px"),
        ParamSpec(
            "l2_gradient", "Exact gradient magnitude", "bool", True,
            description="Euclidean magnitude rather than the cheaper city-block approximation.",
        ),
    ),
    category="edges",
)
def canny(
    image: AnyArray,
    low: float = 40.0,
    high: float = 110.0,
    blur_size: int = 5,
    blur_sigma: float = 1.4,
    l2_gradient: bool = True,
) -> BoolArray:
    """Canny edge detection, implemented from the four stages.

    1. **Smooth** with a Gaussian. Differentiation amplifies high frequencies,
       which is exactly where noise lives, so this is not optional.
    2. **Differentiate** with Sobel to get magnitude and direction.
    3. **Non-maximum suppression**: keep a pixel only if its magnitude is at
       least as large as its two neighbours *along the gradient direction*. This
       is what turns a wide ridge of response into a one-pixel line. The
       direction is quantised to the four available neighbour pairs, which is
       the classical formulation.
    4. **Hysteresis**: pixels above ``high`` are edges; pixels above ``low`` are
       edges only if connected to one. Implemented as a morphological
       reconstruction of the weak map from the strong map, which is exactly the
       "connected to" relation and needs no flood-fill loop.
    """
    if low > high:
        raise ValueError(f"low threshold {low} must not exceed high threshold {high}")
    grey = _to_grey(image)
    smoothed = np.asarray(
        gaussian_filter(grey.astype(np.float32), blur_size, blur_sigma), dtype=np.float64
    )
    grad = GradientResult(
        gx=correlate2d(smoothed, SOBEL_X), gy=correlate2d(smoothed, SOBEL_Y)
    )
    magnitude = grad.magnitude if l2_gradient else grad.magnitude_l1

    # Quantise the gradient direction to the four neighbour pairs: 0, 45, 90 and
    # 135 degrees. Angles are taken modulo 180 because an edge and its reverse
    # are the same edge.
    angle = np.rad2deg(np.arctan2(grad.gy, grad.gx)) % 180.0
    sector = np.zeros(angle.shape, dtype=np.int8)
    sector[(angle >= 22.5) & (angle < 67.5)] = 1
    sector[(angle >= 67.5) & (angle < 112.5)] = 2
    sector[(angle >= 112.5) & (angle < 157.5)] = 3

    padded = np.pad(magnitude, 1, mode="constant", constant_values=0.0)
    # Neighbours to compare against, as (row, column) offsets along the gradient.
    # Rows increase downwards, so a gradient at +45 degrees - both components
    # positive - points down and to the right, and the pair to compare is
    # (+1, +1) and (-1, -1). Getting this pair the wrong way round for the
    # diagonal sectors suppresses diagonal edges against neighbours that lie
    # *across* the gradient rather than along it, which quietly deletes them.
    neighbour_offsets = {
        0: ((0, 1), (0, -1)),  # gradient horizontal: compare left and right
        1: ((1, 1), (-1, -1)),  # gradient at 45 degrees: down-right, up-left
        2: ((-1, 0), (1, 0)),  # gradient vertical: compare up and down
        3: ((1, -1), (-1, 1)),  # gradient at 135 degrees: down-left, up-right
    }
    keep = np.zeros(magnitude.shape, dtype=bool)
    for s, ((dy1, dx1), (dy2, dx2)) in neighbour_offsets.items():
        here = sector == s
        if not here.any():
            continue
        a = padded[1 + dy1 : padded.shape[0] - 1 + dy1, 1 + dx1 : padded.shape[1] - 1 + dx1]
        b = padded[1 + dy2 : padded.shape[0] - 1 + dy2, 1 + dx2 : padded.shape[1] - 1 + dx2]
        keep |= here & (magnitude >= a) & (magnitude >= b)

    thinned = np.where(keep, magnitude, 0.0)
    strong = thinned >= high
    weak = thinned >= low
    if not strong.any():
        return np.zeros(strong.shape, dtype=bool)
    return reconstruct_by_dilation(strong, weak, structuring_element(3, 3, "rect"))


def edge_map(
    image: AnyArray, operator: EdgeOperator = "sobel", threshold: float = 0.0, **kwargs: float
) -> BoolArray:
    """Binary edge map from any of the operators, on a common footing.

    ``threshold`` is a fraction of the maximum gradient magnitude for the
    first-order operators and for the Laplacian, and is ignored by Canny and LoG
    which carry their own. Putting them all behind one call is what makes the
    five-operator comparison a comparison rather than five separate stories.
    """
    if operator == "canny":
        return np.asarray(canny(image, **kwargs), dtype=bool)
    if operator == "log":
        return np.asarray(log_edges(image, **kwargs), dtype=bool)
    if operator == "laplacian":
        kernel = LAPLACIAN_8 if kwargs.get("connectivity", 8) == 8 else LAPLACIAN_4
        response = correlate2d(_to_grey(image), kernel)
        return zero_crossings(response, float(kwargs.get("slope", 4.0)))
    mag = gradient(image, operator).magnitude
    peak = float(mag.max())
    if peak < 1e-9:
        return np.zeros(mag.shape, dtype=bool)
    cut = threshold if threshold > 0 else 0.2
    return np.asarray(mag >= cut * peak, dtype=bool)


def compare_edge_operators(
    image: AnyArray, threshold: float = 0.2
) -> dict[str, dict[str, float]]:
    """Run all five operators on one frame and summarise the differences.

    Reports edge density and mean gradient magnitude rather than an accuracy
    score, because on a real frame there is no ground-truth edge map to score
    against. The honest comparison is what each operator responds to and how
    much of the image it marks, not which one is "right".
    """
    grey = _to_grey(image)
    out: dict[str, dict[str, float]] = {}
    for name in ("roberts", "prewitt", "sobel", "laplacian", "log", "canny"):
        edges = edge_map(grey, name, threshold)
        entry = {
            "edge_density": float(edges.mean()),
            "edge_pixels": float(edges.sum()),
        }
        if name in ("roberts", "prewitt", "sobel"):
            g = gradient(grey, name)
            entry["mean_magnitude"] = float(g.magnitude.mean())
            entry["l1_over_l2"] = float(
                g.magnitude_l1.sum() / max(1e-9, g.magnitude.sum())
            )
        out[name] = entry
    return out


@dataclass(frozen=True, slots=True)
class HoughLine:
    """A line in normal form, ``x cos(theta) + y sin(theta) = rho``."""

    rho_px: float
    theta_rad: float
    votes: int

    @property
    def theta_deg(self) -> float:
        return float(np.rad2deg(self.theta_rad))

    def endpoints(
        self, width_px: int, height_px: int
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Where the line crosses the image bounds, for drawing."""
        c, s = float(np.cos(self.theta_rad)), float(np.sin(self.theta_rad))
        x0, y0 = c * self.rho_px, s * self.rho_px
        span = float(max(width_px, height_px)) * 2.0
        return (
            (x0 + span * -s, y0 + span * c),
            (x0 - span * -s, y0 - span * c),
        )


@register(
    "edges.hough_lines",
    "Hough line transform",
    units=(5,),
    rationale=(
        "Every edge pixel votes for all the lines through it, in normal form; a "
        "real line collects a peak of votes. Because each pixel votes "
        "independently, a lane marking broken into dashes still produces one "
        "peak - which is the whole reason to use it for lane geometry rather "
        "than fitting to a connected component."
    ),
    params=(
        ParamSpec("rho_step", "Rho resolution", "float", 1.0, 0.25, 8.0, 0.25, unit="px"),
        ParamSpec("theta_steps", "Theta bins", "int", 180, 45, 720, 45),
        ParamSpec("threshold", "Vote threshold", "int", 80, 5, 2000, 5),
        ParamSpec("max_lines", "Maximum lines", "int", 12, 1, 64, 1),
        ParamSpec(
            "min_separation_rho", "Peak separation (rho)", "float", 12.0, 0.0, 200.0, 1.0
        ),
        ParamSpec(
            "min_separation_theta_deg",
            "Peak separation (theta)",
            "float",
            8.0,
            0.0,
            90.0,
            1.0,
        ),
    ),
    category="edges",
)
def hough_lines(
    edges: AnyArray,
    rho_step: float = 1.0,
    theta_steps: int = 180,
    threshold: int = 80,
    max_lines: int = 12,
    min_separation_rho: float = 12.0,
    min_separation_theta_deg: float = 8.0,
) -> list[HoughLine]:
    """Standard Hough transform, accumulated directly.

    A point ``(x, y)`` in the image is a sinusoid in ``(rho, theta)`` space, and
    collinear points share the ``(rho, theta)`` where their sinusoids meet. The
    accumulator is built by evaluating ``rho = x cos t + y sin t`` for every edge
    pixel at every ``theta`` bin at once, which is one outer product rather than
    a double loop.

    Peaks are then taken greedily with a minimum separation, because a strong
    line lights up a neighbourhood of the accumulator and taking the top ``n``
    cells outright returns the same line several times over.
    """
    if rho_step <= 0:
        raise ValueError(f"rho step must be positive, got {rho_step}")
    if theta_steps < 2:
        raise ValueError(f"need at least 2 theta bins, got {theta_steps}")

    mask = np.asarray(edges)
    if mask.dtype != np.bool_:
        mask = mask > 0
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return []

    height, width = mask.shape
    thetas = np.linspace(0.0, np.pi, int(theta_steps), endpoint=False)
    diagonal = float(np.hypot(height, width))
    rho_bins = int(np.ceil(2.0 * diagonal / rho_step)) + 1

    rho = xs[:, None] * np.cos(thetas)[None, :] + ys[:, None] * np.sin(thetas)[None, :]
    rho_index = np.rint((rho + diagonal) / rho_step).astype(np.int64)
    np.clip(rho_index, 0, rho_bins - 1, out=rho_index)

    theta_index = np.tile(np.arange(theta_steps), (xs.size, 1))
    flat = np.bincount(
        (rho_index * theta_steps + theta_index).ravel(), minlength=rho_bins * theta_steps
    )
    accumulator = flat.reshape(rho_bins, theta_steps)

    theta_sep = int(round(min_separation_theta_deg / 180.0 * theta_steps))
    rho_sep = int(round(min_separation_rho / rho_step))

    work = accumulator.astype(np.int64).copy()
    lines: list[HoughLine] = []
    for _ in range(max(1, int(max_lines))):
        idx = int(np.argmax(work))
        r_i, t_i = np.unravel_index(idx, work.shape)
        votes = int(work[r_i, t_i])
        if votes < threshold:
            break
        lines.append(
            HoughLine(
                rho_px=float(r_i) * rho_step - diagonal,
                theta_rad=float(thetas[t_i]),
                votes=votes,
            )
        )
        row, col = int(r_i), int(t_i)
        r0, r1 = max(0, row - rho_sep), min(rho_bins, row + rho_sep + 1)
        t0, t1 = col - theta_sep, col + theta_sep + 1
        work[r0:r1, max(0, t0) : min(theta_steps, t1)] = 0
        # Theta wraps at pi with a sign flip in rho, so a line near either end of
        # the range has a second neighbourhood to suppress.
        if t0 < 0:
            work[rho_bins - r1 : rho_bins - r0, theta_steps + t0 :] = 0
        if t1 > theta_steps:
            work[rho_bins - r1 : rho_bins - r0, : t1 - theta_steps] = 0
    return lines


def draw_hough_lines(
    image: AnyArray, lines: list[HoughLine], colour: tuple[int, int, int] = (63, 167, 214)
) -> UInt8Array:
    """Draw lines onto a copy of an image, for the inspector."""
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(clip_to_uint8(canvas), cv2.COLOR_GRAY2BGR)
    h, w = canvas.shape[:2]
    for line in lines:
        (x1, y1), (x2, y2) = line.endpoints(w, h)
        cv2.line(
            canvas, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
            (colour[2], colour[1], colour[0]), 1, cv2.LINE_AA,
        )
    return np.asarray(canvas, dtype=np.uint8)
