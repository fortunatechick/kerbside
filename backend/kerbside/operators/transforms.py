"""Unit 3 - orthogonal transforms and transform coding.

Three transforms, one question. The DFT, the DCT and the Hadamard transform all
re-express an image as a weighted sum of basis images; what separates them is
how much of the image's energy ends up in how few coefficients. That property -
energy compaction - is the whole of transform coding, and it is why JPEG uses
the DCT rather than either of the others.

The question this module is here to answer is an operational one rather than a
textbook one: **how far can CCTV bitrate fall before vehicle counting
degrades?** Storage on a roadside recorder is finite and someone has already
chosen a quality factor. :func:`compression_study` measures what that choice
costs in fidelity, and the ablation in the evaluation suite measures what it
costs in counts.

The Hadamard matrix is built here by the Sylvester recursion rather than
imported, because its structure - entries of only plus and minus one, so a
transform needs no multiplications at all - is the entire reason it was ever
used, and that is much clearer from the construction than from a library call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.fft import dctn, idctn

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, FloatArray
from kerbside.operators.point import clip_to_uint8

__all__ = [
    "TransformName",
    "CompressionResult",
    "dct2",
    "idct2",
    "hadamard_matrix",
    "hadamard2",
    "ihadamard2",
    "transform2",
    "inverse_transform2",
    "blockwise",
    "keep_largest_coefficients",
    "zonal_mask",
    "compress_blocks",
    "compression_study",
    "energy_compaction",
    "psnr",
]

TransformName = Literal["dct", "dft", "hadamard"]


def psnr(reference: AnyArray, test: AnyArray, peak: float = 255.0) -> float:
    """Peak signal-to-noise ratio in dB.

    Infinite for identical inputs, which is reported as ``inf`` rather than
    being capped: a capped value would silently claim a lossy result where the
    round trip was actually exact.
    """
    a = np.asarray(reference, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} against {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10((peak * peak) / mse))


def dct2(block: AnyArray) -> FloatArray:
    """Two-dimensional DCT-II, orthonormal.

    The DCT is real, and its implicit even extension at the boundaries avoids
    the discontinuity the DFT's periodic extension creates - which is precisely
    why it compacts energy better on natural images.
    """
    return np.asarray(dctn(np.asarray(block, dtype=np.float64), type=2, norm="ortho"))


def idct2(coefficients: AnyArray) -> FloatArray:
    """Inverse of :func:`dct2`."""
    return np.asarray(
        idctn(np.asarray(coefficients, dtype=np.float64), type=2, norm="ortho")
    )


def hadamard_matrix(order: int, sequency: bool = True) -> FloatArray:
    """Hadamard matrix of size ``order``, built by the Sylvester recursion.

    ``H_1 = [1]`` and ``H_{2n} = [[H_n, H_n], [H_n, -H_n]]``, so every entry is
    plus or minus one and the forward transform needs only additions and
    subtractions. That was the point of it: on hardware without a multiplier it
    is nearly free.

    ``sequency`` reorders the rows by their number of sign changes, which is the
    Hadamard analogue of ordering by frequency. Without it the natural
    (Sylvester) order scatters the low-sequency basis vectors through the
    matrix, and a zonal mask that keeps "the first k coefficients" keeps an
    arbitrary set rather than the smooth ones.
    """
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError(f"Hadamard order must be a power of two, got {order}")
    h: AnyArray = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < order:
        h = np.block([[h, h], [h, -h]])
    if sequency:
        changes = np.count_nonzero(np.diff(np.sign(h), axis=1), axis=1)
        h = h[np.argsort(changes, kind="stable")]
    return np.asarray(h / np.sqrt(order), dtype=np.float64)


def hadamard2(block: AnyArray, sequency: bool = True) -> FloatArray:
    """Two-dimensional Hadamard transform, orthonormal."""
    x = np.asarray(block, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError(f"Hadamard transform needs a square block, got {x.shape}")
    h = hadamard_matrix(x.shape[0], sequency)
    return h @ x @ h.T


def ihadamard2(coefficients: AnyArray, sequency: bool = True) -> FloatArray:
    """Inverse Hadamard transform. The orthonormal matrix is its own inverse
    transpose, so this is the forward transform applied again."""
    x = np.asarray(coefficients, dtype=np.float64)
    h = hadamard_matrix(x.shape[0], sequency)
    return h.T @ x @ h


def transform2(block: AnyArray, name: TransformName = "dct", sequency: bool = True) -> AnyArray:
    """Forward transform by name, for the three-way comparison."""
    if name == "dct":
        return dct2(block)
    if name == "hadamard":
        return hadamard2(block, sequency)
    if name == "dft":
        return np.fft.fft2(np.asarray(block, dtype=np.float64), norm="ortho")
    raise ValueError(f"unknown transform {name!r}; use dct, dft or hadamard")


def inverse_transform2(
    coefficients: AnyArray, name: TransformName = "dct", sequency: bool = True
) -> FloatArray:
    """Inverse transform by name."""
    if name == "dct":
        return idct2(coefficients)
    if name == "hadamard":
        return ihadamard2(coefficients, sequency)
    if name == "dft":
        return np.asarray(np.real(np.fft.ifft2(coefficients, norm="ortho")), dtype=np.float64)
    raise ValueError(f"unknown transform {name!r}; use dct, dft or hadamard")


def blockwise(image: AnyArray, block_size: int = 8) -> tuple[AnyArray, tuple[int, int]]:
    """Split an image into a grid of square blocks, padding to fit.

    Returns the blocks as ``(rows, cols, block, block)`` together with the
    original shape, so :func:`compress_blocks` can crop the padding back off
    rather than leaving an image a few pixels larger than it started.
    """
    if block_size < 2:
        raise ValueError(f"block size must be at least 2, got {block_size}")
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {x.shape}")
    h, w = x.shape
    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w)), mode="edge")
    rows = x.shape[0] // block_size
    cols = x.shape[1] // block_size
    blocks = x.reshape(rows, block_size, cols, block_size).swapaxes(1, 2)
    return blocks, (h, w)


def keep_largest_coefficients(coefficients: AnyArray, keep: int) -> AnyArray:
    """Threshold coding: keep the ``keep`` largest coefficients by magnitude.

    This is the upper bound on what any transform can do at a given coefficient
    budget, because it chooses the best subset per block with hindsight. Real
    codecs use a fixed zonal mask or a quantisation table instead, so the gap
    between this and :func:`zonal_mask` is the price of not transmitting which
    coefficients were kept.
    """
    c = np.asarray(coefficients)
    flat = np.abs(c).ravel()
    if keep >= flat.size:
        return c.copy()
    if keep <= 0:
        return np.zeros_like(c)
    cutoff_index = flat.size - keep
    threshold = np.partition(flat, cutoff_index)[cutoff_index]
    return np.where(np.abs(c) >= threshold, c, 0)


def zonal_mask(block_size: int, keep: int) -> AnyArray:
    """Zonal coding mask keeping the ``keep`` lowest-frequency coefficients.

    Ordered by ``u + v``, the anti-diagonal zig-zag that JPEG follows, so the
    retained zone is the top-left triangle where natural-image energy lives.
    Unlike threshold coding this needs no side information: encoder and decoder
    agree on the zone in advance.
    """
    if keep < 0:
        raise ValueError(f"keep must be non-negative, got {keep}")
    u, v = np.meshgrid(
        np.arange(block_size), np.arange(block_size), indexing="ij"
    )
    order = np.argsort((u + v).ravel() * block_size + np.minimum(u, v).ravel(), kind="stable")
    mask = np.zeros(block_size * block_size, dtype=bool)
    mask[order[: min(keep, mask.size)]] = True
    return mask.reshape(block_size, block_size)


def compress_blocks(
    image: AnyArray,
    keep: int,
    block_size: int = 8,
    name: TransformName = "dct",
    zonal: bool = False,
) -> FloatArray:
    """Block transform coding round trip, keeping ``keep`` coefficients per block."""
    blocks, (h, w) = blockwise(image, block_size)
    rows, cols = blocks.shape[:2]
    mask = zonal_mask(block_size, keep) if zonal else None

    out = np.zeros((rows, cols, block_size, block_size), dtype=np.float64)
    for r in range(rows):
        for c in range(cols):
            coeffs = transform2(blocks[r, c], name)
            kept = coeffs * mask if mask is not None else keep_largest_coefficients(coeffs, keep)
            out[r, c] = inverse_transform2(kept, name)
    merged = out.swapaxes(1, 2).reshape(rows * block_size, cols * block_size)
    return merged[:h, :w]


@dataclass(frozen=True, slots=True)
class CompressionResult:
    """One point on a rate-distortion curve."""

    transform: TransformName
    block_size: int
    kept_coefficients: int
    total_coefficients: int
    zonal: bool
    psnr_db: float
    rmse: float

    @property
    def compression_ratio(self) -> float:
        """Coefficients discarded per coefficient kept, ignoring entropy coding.

        Deliberately not called a bitrate. A real codec also entropy-codes what
        survives, so this overstates the achievable file size; it is a fair
        comparison *between* transforms at equal coefficient budgets, which is
        what the study needs.
        """
        return self.total_coefficients / max(1, self.kept_coefficients)


@register(
    "transform.compression_study",
    "Transform compression study",
    units=(3,),
    rationale=(
        "Measures fidelity against the number of coefficients retained, for the "
        "DCT, the DFT and the Hadamard transform on the same frame. It answers "
        "the operational question behind the resolution and bitrate ablation: "
        "how far can recorded quality fall before the pipeline stops counting "
        "correctly."
    ),
    params=(
        ParamSpec("block_size", "Block size", "int", 8, 2, 32, 2, unit="px"),
        ParamSpec(
            "name", "Transform", "enum", "dct", choices=("dct", "dft", "hadamard")
        ),
        ParamSpec(
            "zonal", "Zonal coding", "bool", False,
            description="Keep a fixed low-frequency zone rather than the largest coefficients.",
        ),
    ),
    category="transform",
)
def compression_study(
    image: AnyArray,
    keeps: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    block_size: int = 8,
    name: TransformName = "dct",
    zonal: bool = False,
) -> list[CompressionResult]:
    """Rate-distortion curve for one transform at several coefficient budgets."""
    grey = np.asarray(image, dtype=np.float64)
    if grey.ndim == 3:
        import cv2

        grey = cv2.cvtColor(grey.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)

    total = block_size * block_size
    results: list[CompressionResult] = []
    for keep in keeps:
        if not 0 < keep <= total:
            raise ValueError(
                f"keep must be in 1..{total} for a {block_size}x{block_size} block, got {keep}"
            )
        restored = compress_blocks(grey, keep, block_size, name, zonal)
        results.append(
            CompressionResult(
                transform=name,
                block_size=block_size,
                kept_coefficients=keep,
                total_coefficients=total,
                zonal=zonal,
                psnr_db=psnr(grey, restored),
                rmse=float(np.sqrt(np.mean((grey - restored) ** 2))),
            )
        )
    return results


def energy_compaction(
    image: AnyArray, block_size: int = 8, name: TransformName = "dct"
) -> FloatArray:
    """Fraction of block energy captured by the largest ``k`` coefficients.

    Returns a curve of length ``block_size**2``, averaged over every block.
    Plotting the three transforms together is the most direct statement of why
    the DCT won: for natural images its curve reaches 99% of the energy in a
    handful of coefficients, where the Hadamard transform needs noticeably more.
    """
    blocks, _ = blockwise(image, block_size)
    rows, cols = blocks.shape[:2]
    total = block_size * block_size
    accumulated = np.zeros(total, dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            coeffs = np.abs(np.asarray(transform2(blocks[r, c], name))) ** 2
            ordered = np.sort(coeffs.ravel())[::-1]
            energy = ordered.sum()
            if energy > 0:
                accumulated += np.cumsum(ordered) / energy
    return accumulated / max(1, rows * cols)


@register(
    "transform.dct_preview",
    "DCT block preview",
    units=(3,),
    rationale=(
        "Shows a frame reconstructed from a limited number of DCT coefficients "
        "per block. The blocking and ringing that appear at low budgets are what "
        "a background-difference image amplifies, which is why aggressive "
        "recording quality costs counting accuracy rather than just looking bad."
    ),
    params=(
        ParamSpec("keep", "Coefficients per block", "int", 8, 1, 64, 1),
        ParamSpec("block_size", "Block size", "int", 8, 2, 32, 2, unit="px"),
        ParamSpec("zonal", "Zonal coding", "bool", False),
    ),
    category="transform",
)
def dct_preview(
    image: AnyArray, keep: int = 8, block_size: int = 8, zonal: bool = False
) -> AnyArray:
    """Reconstruct a frame from a truncated DCT, for display."""
    grey = np.asarray(image, dtype=np.float64)
    if grey.ndim == 3:
        import cv2

        grey = cv2.cvtColor(grey.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    return clip_to_uint8(compress_blocks(grey, keep, block_size, "dct", zonal))
