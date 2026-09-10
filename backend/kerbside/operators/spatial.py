"""Unit 2 - neighbourhood processing.

A spatial filter replaces each pixel with a function of the pixels around it.
Smoothing filters suppress noise at the cost of blurring edges; sharpening
filters do the reverse. The pipeline needs both: the foreground mask is cleaned
with a median filter before morphology, and edge-based refinement of blob
boundaries under motion blur needs a sharpening step.

``convolve2d`` is written out by hand from the definition, because the point of
the exercise is that the correlation-versus-convolution distinction and the
border handling are understood rather than delegated. The production filters
call OpenCV, which is the same computation orders of magnitude faster; the tests
assert the two agree, which is the honest basis for using the fast one.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, FloatArray
from kerbside.operators.point import clip_to_uint8

__all__ = [
    "BorderMode",
    "pad",
    "convolve2d",
    "correlate2d",
    "box_filter",
    "gaussian_filter",
    "median_filter",
    "laplacian",
    "unsharp_mask",
    "high_boost",
    "gaussian_kernel",
    "box_kernel",
    "LAPLACIAN_4",
    "LAPLACIAN_8",
]

BorderMode = Literal["reflect", "replicate", "constant", "wrap"]

LAPLACIAN_4 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
"""4-neighbour Laplacian: the discrete second derivative along the axes only."""

LAPLACIAN_8 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float64)
"""8-neighbour Laplacian, which includes the diagonal second differences and is
therefore more isotropic and more noise-sensitive."""

_NUMPY_BORDER: dict[BorderMode, Literal["reflect", "edge", "constant", "wrap"]] = {
    "reflect": "reflect",
    "replicate": "edge",
    "constant": "constant",
    "wrap": "wrap",
}
"""Border modes named as NumPy names them. NumPy calls edge replication "edge"
and OpenCV calls it BORDER_REPLICATE; keeping one vocabulary in the public API
and translating at the boundary means a caller never has to know which library
performed the padding."""

_CV_BORDER = {
    "reflect": cv2.BORDER_REFLECT_101,
    "replicate": cv2.BORDER_REPLICATE,
    "constant": cv2.BORDER_CONSTANT,
    "wrap": cv2.BORDER_WRAP,
}


def _check_odd(size: int, what: str) -> int:
    if size < 1 or size % 2 == 0:
        raise ValueError(
            f"{what} must be a positive odd number so the kernel has a centre "
            f"pixel, got {size}"
        )
    return int(size)


def pad(image: AnyArray, radius: int, mode: BorderMode = "reflect") -> FloatArray:
    """Pad an image by ``radius`` on every side.

    Border handling is a real choice, not boilerplate. ``constant`` invents a
    dark frame around the image, which a difference-based pipeline would detect
    as foreground along every edge; ``reflect`` invents nothing and is the
    default here for that reason.
    """
    x = np.asarray(image, dtype=np.float64)
    if radius == 0:
        return x
    if mode not in _NUMPY_BORDER:
        raise ValueError(
            f"unknown border mode {mode!r}; use one of {list(_NUMPY_BORDER)}"
        )
    width: list[tuple[int, int]] = [(radius, radius), (radius, radius)]
    width += [(0, 0)] * (x.ndim - 2)
    return np.asarray(np.pad(x, width, mode=_NUMPY_BORDER[mode]), dtype=np.float64)


def correlate2d(
    image: AnyArray, kernel: AnyArray, mode: BorderMode = "reflect"
) -> FloatArray:
    """Two-dimensional correlation, written out from the definition.

    ``g(x, y) = sum_s sum_t w(s, t) f(x + s, y + t)``

    The sliding window is materialised with a stride trick so the sum is a
    single tensor contraction rather than a Python loop over pixels - the same
    arithmetic, at a speed that makes it usable as a reference implementation in
    tests rather than only on paper.
    """
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1]:
        raise ValueError(f"kernel must be square and 2-D, got shape {k.shape}")
    _check_odd(k.shape[0], "kernel size")
    radius = k.shape[0] // 2

    x = np.asarray(image, dtype=np.float64)
    if x.ndim == 3:
        return np.stack(
            [correlate2d(x[:, :, c], k, mode) for c in range(x.shape[2])], axis=2
        )

    padded = pad(x, radius, mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, k.shape)
    return np.asarray(np.einsum("ijst,st->ij", windows, k), dtype=np.float64)


def convolve2d(
    image: AnyArray, kernel: AnyArray, mode: BorderMode = "reflect"
) -> FloatArray:
    """Two-dimensional convolution: correlation with the kernel rotated by 180 degrees.

    The distinction only vanishes for symmetric kernels, which is why it is so
    often skipped and so occasionally the reason a gradient comes out with the
    wrong sign.
    """
    return correlate2d(image, np.asarray(kernel, dtype=np.float64)[::-1, ::-1], mode)


def box_kernel(size: int) -> FloatArray:
    """Normalised averaging kernel."""
    size = _check_odd(size, "box size")
    return np.full((size, size), 1.0 / (size * size), dtype=np.float64)


def gaussian_kernel(size: int, sigma: float = 0.0) -> FloatArray:
    """Separable Gaussian kernel, normalised to unit sum.

    A ``sigma`` of zero adopts OpenCV's convention of deriving it from the
    kernel size, so that a caller who only knows how large a neighbourhood they
    want does not have to invent a standard deviation to go with it.
    """
    size = _check_odd(size, "kernel size")
    if sigma <= 0:
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    ax = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    line = np.exp(-(ax**2) / (2.0 * sigma * sigma))
    line /= line.sum()
    return np.outer(line, line)


@register(
    "spatial.box",
    "Box filter",
    units=(2,),
    rationale=(
        "Replaces each pixel with the unweighted mean of its neighbourhood. The "
        "cheapest smoother and the most damaging to edges, which is exactly why "
        "it is here: comparing it against the Gaussian and the median is how the "
        "cost of that damage is shown rather than asserted."
    ),
    params=(ParamSpec("size", "Kernel size", "odd_int", 3, 1, 31, 2, unit="px"),),
    category="spatial",
)
def box_filter(image: AnyArray, size: int = 3) -> AnyArray:
    """Unweighted neighbourhood mean."""
    size = _check_odd(size, "kernel size")
    if size == 1:
        return np.asarray(image).copy()
    return cv2.blur(np.asarray(image), (size, size), borderType=cv2.BORDER_REFLECT_101)


@register(
    "spatial.gaussian",
    "Gaussian filter",
    units=(2,),
    rationale=(
        "Weighted neighbourhood mean with a Gaussian profile - separable, "
        "isotropic, and with no ringing. It is the smoothing used before edge "
        "detection, because differentiating a noisy signal amplifies exactly the "
        "frequencies the noise lives in."
    ),
    params=(
        ParamSpec("size", "Kernel size", "odd_int", 5, 1, 31, 2, unit="px"),
        ParamSpec(
            "sigma", "Sigma", "float", 0.0, 0.0, 10.0, 0.1, unit="px",
            description="Zero derives sigma from the kernel size.",
        ),
    ),
    category="spatial",
)
def gaussian_filter(image: AnyArray, size: int = 5, sigma: float = 0.0) -> AnyArray:
    """Gaussian smoothing."""
    size = _check_odd(size, "kernel size")
    if size == 1:
        return np.asarray(image).copy()
    return cv2.GaussianBlur(
        np.asarray(image), (size, size), sigmaX=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )


@register(
    "spatial.median",
    "Median filter",
    units=(2,),
    rationale=(
        "Replaces each pixel with the neighbourhood median - an order statistic, "
        "not an average. It removes the isolated speckle a background difference "
        "produces without blurring the vehicle boundary the way a mean filter "
        "would, which is why S4 uses it before morphology rather than after."
    ),
    params=(ParamSpec("size", "Kernel size", "odd_int", 3, 1, 15, 2, unit="px"),),
    category="spatial",
)
def median_filter(image: AnyArray, size: int = 3) -> AnyArray:
    """Neighbourhood median.

    Non-linear, so it does not distribute an outlier over its neighbours the way
    a convolution does: a single salt-and-pepper pixel is discarded outright
    rather than smeared across the kernel.
    """
    size = _check_odd(size, "kernel size")
    if size == 1:
        return np.asarray(image).copy()
    x = np.asarray(image)
    if x.dtype != np.uint8 and size > 5:
        # OpenCV only supports large apertures on 8-bit data; say so rather than
        # failing inside the library with an opaque assertion.
        raise ValueError(
            f"median aperture {size} needs 8-bit input, got {x.dtype}; convert "
            f"first or use size <= 5"
        )
    return cv2.medianBlur(x, size)


@register(
    "spatial.laplacian",
    "Laplacian",
    units=(2, 5),
    rationale=(
        "The isotropic second derivative. It responds to intensity transitions "
        "rather than to intensity itself, so adding a scaled copy back to the "
        "original sharpens it. Being second order, it is markedly more sensitive "
        "to noise than a gradient operator."
    ),
    params=(
        ParamSpec(
            "connectivity", "Neighbourhood", "enum", "8", choices=("4", "8"),
            description="4 uses the axial second differences only; 8 adds the diagonals.",
        ),
        ParamSpec("normalise", "Normalise for display", "bool", True),
    ),
    category="spatial",
)
def laplacian(
    image: AnyArray, connectivity: str = "8", normalise: bool = True
) -> AnyArray:
    """Discrete Laplacian.

    With ``normalise`` set the signed result is shifted and scaled into the
    8-bit range for display; without it the raw signed float is returned, which
    is what a sharpening step needs.
    """
    kernel = LAPLACIAN_8 if str(connectivity) == "8" else LAPLACIAN_4
    x = np.asarray(image, dtype=np.float32)
    out = cv2.filter2D(x, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT_101)
    if not normalise:
        return np.asarray(out, dtype=np.float64)
    peak = float(np.abs(out).max())
    if peak < 1e-9:
        return np.full(out.shape, 128, dtype=np.uint8)
    return clip_to_uint8(out * (127.0 / peak) + 128.0)


@register(
    "spatial.unsharp",
    "Unsharp mask",
    units=(2,),
    rationale=(
        "Subtracts a blurred copy to isolate the detail, then adds it back "
        "amplified. Used on motion-blurred frames to recover a usable blob "
        "boundary before the perspective size filter judges the blob's area."
    ),
    params=(
        ParamSpec("size", "Blur size", "odd_int", 5, 3, 31, 2, unit="px"),
        ParamSpec("sigma", "Blur sigma", "float", 0.0, 0.0, 10.0, 0.1, unit="px"),
        ParamSpec("amount", "Amount", "float", 1.0, 0.0, 5.0, 0.05),
    ),
    category="spatial",
    reference="g = f + k (f - blur(f))",
)
def unsharp_mask(
    image: AnyArray, size: int = 5, sigma: float = 0.0, amount: float = 1.0
) -> AnyArray:
    """Unsharp masking. ``amount`` of 1 is the classical definition."""
    if amount < 0:
        raise ValueError(f"amount must be non-negative, got {amount}")
    x = np.asarray(image, dtype=np.float64)
    blurred = np.asarray(gaussian_filter(x.astype(np.float32), size, sigma), dtype=np.float64)
    return clip_to_uint8(x + amount * (x - blurred))


@register(
    "spatial.high_boost",
    "High-boost filter",
    units=(2,),
    rationale=(
        "Unsharp masking generalised: g = A f - blur(f), with A of 1 giving the "
        "pure high-pass and larger A retaining more of the original. It is the "
        "operator to reach for when a frame needs sharpening without losing its "
        "low-frequency content."
    ),
    params=(
        ParamSpec("boost", "Boost A", "float", 1.5, 1.0, 5.0, 0.05),
        ParamSpec("size", "Blur size", "odd_int", 5, 3, 31, 2, unit="px"),
        ParamSpec("sigma", "Blur sigma", "float", 0.0, 0.0, 10.0, 0.1, unit="px"),
    ),
    category="spatial",
    reference="g = A f - blur(f)",
)
def high_boost(
    image: AnyArray, boost: float = 1.5, size: int = 5, sigma: float = 0.0
) -> AnyArray:
    """High-boost filtering."""
    if boost < 1.0:
        raise ValueError(f"boost must be at least 1, got {boost}")
    x = np.asarray(image, dtype=np.float64)
    blurred = np.asarray(gaussian_filter(x.astype(np.float32), size, sigma), dtype=np.float64)
    return clip_to_uint8(boost * x - blurred)
