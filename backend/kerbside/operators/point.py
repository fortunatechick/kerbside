"""Unit 2 - point processing.

A point operation maps each pixel through a function of its own value alone:
``s = T(r)``. No neighbours, no context. That makes these operators the cheapest
things in the library and the right first response to a frame that is too dark,
too flat, or backlit - which is most of what a roadside camera delivers outside
the middle of the day.

Where they are used in the pipeline:

* contrast stretching and histogram equalisation condition night and backlit
  frames before background differencing, because a difference image built from
  a frame with 30 grey levels of usable range has nothing for Otsu to threshold;
* thresholding is the elementary form of the foreground decision in S4;
* the power-law transform is the correction for the gamma that
  :mod:`kerbside.data.degrade` applies, and the pair are inverses;
* intensity slicing and bit-plane slicing are diagnostic views in the pipeline
  inspector rather than production stages.

Every function is pure: arrays in, arrays out, no I/O, no globals.
"""

from __future__ import annotations

import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, FloatArray, UInt8Array

__all__ = [
    "negative",
    "contrast_stretch",
    "threshold",
    "intensity_slice",
    "log_transform",
    "power_law",
    "bit_plane",
    "histogram",
    "cumulative_histogram",
    "histogram_equalise",
    "match_histogram",
    "clip_to_uint8",
]


def _as_float(image: AnyArray) -> FloatArray:
    return np.asarray(image, dtype=np.float64)


def clip_to_uint8(image: AnyArray) -> UInt8Array:
    """Round and clamp to the 8-bit display range.

    Kept in one place because rounding differences between operators would show
    up as a one-level difference in the pipeline inspector's pixel probe and
    look like a bug in whichever operator was selected at the time.
    """
    return np.clip(np.rint(np.asarray(image, dtype=np.float64)), 0, 255).astype(np.uint8)


@register(
    "point.negative",
    "Negative",
    units=(2,),
    rationale=(
        "Maps s = L-1-r, inverting the intensity scale. Detail buried in the "
        "bright end of a backlit frame is easier to see - and easier to "
        "threshold - once it sits in the dark end instead."
    ),
    category="point",
    reference="s = (L-1) - r",
)
def negative(image: AnyArray, levels: int = 256) -> UInt8Array:
    """Photographic negative.

    Parameters
    ----------
    image
        Any 2-D or 3-D array of intensities.
    levels
        Number of grey levels ``L``; 256 for 8-bit data.
    """
    if levels < 2:
        raise ValueError(f"need at least 2 grey levels, got {levels}")
    return clip_to_uint8((levels - 1) - _as_float(image))


@register(
    "point.contrast_stretch",
    "Contrast stretch",
    units=(2,),
    rationale=(
        "Linearly rescales the intensity range between two percentiles onto the "
        "full scale. A night or backlit frame often occupies a fraction of the "
        "available range, and a background difference built from it has too "
        "little contrast for Otsu to find a valley."
    ),
    params=(
        ParamSpec(
            "low_percentile", "Low percentile", "float", 1.0, 0.0, 49.0, 0.5, unit="%",
            description="Input level mapped to 0. Above zero, this clips outliers.",
        ),
        ParamSpec(
            "high_percentile", "High percentile", "float", 99.0, 51.0, 100.0, 0.5, unit="%",
            description="Input level mapped to the top of the scale.",
        ),
    ),
    category="point",
    reference="s = (r - r_low) * (L-1) / (r_high - r_low)",
)
def contrast_stretch(
    image: AnyArray, low_percentile: float = 1.0, high_percentile: float = 99.0
) -> UInt8Array:
    """Percentile-based linear contrast stretch.

    Using percentiles rather than the raw minimum and maximum matters: a single
    hot pixel or one specular highlight sets the maximum, and stretching to it
    achieves nothing. Clipping a percent from each end is what makes the
    operator useful on real frames.
    """
    if low_percentile >= high_percentile:
        raise ValueError(
            f"low percentile {low_percentile} must be below high percentile "
            f"{high_percentile}"
        )
    x = _as_float(image)
    lo = float(np.percentile(x, low_percentile))
    hi = float(np.percentile(x, high_percentile))
    if hi - lo < 1e-9:
        # A uniform image has no contrast to stretch. Returning it unchanged is
        # the honest answer; scaling by 1/0 would produce a white frame that
        # looks like a working operator.
        return clip_to_uint8(x)
    return clip_to_uint8((x - lo) * (255.0 / (hi - lo)))


@register(
    "point.threshold",
    "Threshold",
    units=(2, 6),
    rationale=(
        "Splits the intensity scale at a level, the elementary segmentation. "
        "In S4 the level comes from Otsu rather than from a person, but the "
        "decision itself is this operator."
    ),
    params=(
        ParamSpec("level", "Level", "float", 128.0, 0.0, 255.0, 1.0),
        ParamSpec(
            "invert", "Invert", "bool", False,
            description="Keep pixels below the level instead of above it.",
        ),
    ),
    category="point",
    reference="s = 255 if r > T else 0",
)
def threshold(image: AnyArray, level: float = 128.0, invert: bool = False) -> UInt8Array:
    """Binary threshold, returning 0 or 255."""
    x = _as_float(image)
    mask = x <= level if invert else x > level
    return (mask.astype(np.uint8)) * 255


@register(
    "point.intensity_slice",
    "Intensity slicing",
    units=(2,),
    rationale=(
        "Highlights one band of the intensity scale, optionally preserving the "
        "rest. Used in the inspector to show exactly which pixels a candidate "
        "threshold band would select, before committing to it."
    ),
    params=(
        ParamSpec("low", "Band low", "float", 100.0, 0.0, 255.0, 1.0),
        ParamSpec("high", "Band high", "float", 180.0, 0.0, 255.0, 1.0),
        ParamSpec("value", "Highlight value", "float", 255.0, 0.0, 255.0, 1.0),
        ParamSpec(
            "keep_background", "Keep background", "bool", True,
            description="Leave pixels outside the band untouched rather than zeroing them.",
        ),
    ),
    category="point",
)
def intensity_slice(
    image: AnyArray,
    low: float = 100.0,
    high: float = 180.0,
    value: float = 255.0,
    keep_background: bool = True,
) -> UInt8Array:
    """Set pixels inside ``[low, high]`` to ``value``."""
    if low > high:
        raise ValueError(f"band low {low} must not exceed band high {high}")
    x = _as_float(image)
    inside = (x >= low) & (x <= high)
    out = x.copy() if keep_background else np.zeros_like(x)
    out[inside] = value
    return clip_to_uint8(out)


@register(
    "point.log",
    "Log transform",
    units=(2,),
    rationale=(
        "Compresses a wide dynamic range into a displayable one by expanding "
        "dark values and compressing bright ones. This is what makes a Fourier "
        "magnitude spectrum legible, where the DC term is orders of magnitude "
        "larger than everything else."
    ),
    params=(ParamSpec("scale", "Scale c", "float", 1.0, 0.01, 10.0, 0.01),),
    category="point",
    reference="s = c log(1 + r)",
)
def log_transform(image: AnyArray, scale: float = 1.0) -> UInt8Array:
    """Logarithmic intensity transform, normalised to the 8-bit range."""
    x = _as_float(image)
    if x.min() < 0:
        x = x - x.min()
    out = np.log1p(x)
    peak = float(out.max())
    if peak <= 0:
        return clip_to_uint8(np.zeros_like(out))
    return clip_to_uint8(scale * out * (255.0 / peak))


@register(
    "point.power_law",
    "Power law (gamma)",
    units=(2,),
    rationale=(
        "The standard transfer-curve correction. Gamma below one brightens the "
        "shadows of an under-exposed night frame; above one darkens a washed-out "
        "one. It is the exact inverse of the gamma the degradation study applies."
    ),
    params=(
        ParamSpec("gamma", "Gamma", "float", 1.0, 0.05, 5.0, 0.05),
        ParamSpec("scale", "Scale c", "float", 1.0, 0.01, 4.0, 0.01),
    ),
    category="point",
    reference="s = c r^gamma",
)
def power_law(image: AnyArray, gamma: float = 1.0, scale: float = 1.0) -> UInt8Array:
    """Power-law (gamma) transform on the normalised range."""
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    x = np.clip(_as_float(image) / 255.0, 0.0, 1.0)
    return clip_to_uint8(scale * np.power(x, gamma) * 255.0)


@register(
    "point.bit_plane",
    "Bit-plane slice",
    units=(1, 2),
    rationale=(
        "Extracts one bit of the binary representation of each pixel. The high "
        "planes carry the structure and the low planes are mostly noise, which "
        "is the most direct way to show what grey-level resolution actually buys."
    ),
    params=(ParamSpec("plane", "Bit plane", "int", 7, 0, 7, 1),),
    category="point",
)
def bit_plane(image: AnyArray, plane: int = 7) -> UInt8Array:
    """Extract bit ``plane`` (0 is the least significant) as a binary image."""
    if not 0 <= plane <= 7:
        raise ValueError(f"bit plane must be in 0..7 for 8-bit data, got {plane}")
    x = np.asarray(image, dtype=np.uint8)
    return (((x >> plane) & 1) * 255).astype(np.uint8)


def histogram(
    image: AnyArray, bins: int = 256, value_range: tuple[float, float] = (0, 256)
) -> AnyArray:
    """Intensity histogram as raw counts.

    Written with ``np.bincount`` for the 8-bit case because it is both faster
    and exactly integer-correct, where ``np.histogram`` on floats can put a
    value on the wrong side of a bin edge.
    """
    x = np.asarray(image)
    if bins == 256 and value_range == (0, 256) and x.dtype == np.uint8:
        return np.bincount(x.ravel(), minlength=256).astype(np.int64)
    counts, _ = np.histogram(x.ravel(), bins=bins, range=value_range)
    return counts.astype(np.int64)


def cumulative_histogram(image: AnyArray, bins: int = 256) -> AnyArray:
    """Normalised cumulative distribution of intensities, in ``[0, 1]``."""
    counts = histogram(image, bins=bins)
    total = counts.sum()
    if total == 0:
        return np.zeros(bins, dtype=np.float64)
    return np.cumsum(counts).astype(np.float64) / float(total)


@register(
    "point.histogram_equalise",
    "Histogram equalisation",
    units=(2,),
    rationale=(
        "Maps intensities through their own cumulative distribution, spreading "
        "the most-populated levels furthest apart. It fixes a low-contrast night "
        "frame without anyone choosing a level - but it also amplifies noise in "
        "flat regions, which is why the pipeline offers it rather than applying "
        "it always."
    ),
    params=(
        ParamSpec(
            "adaptive", "Adaptive (CLAHE)", "bool", False,
            description=(
                "Equalise within tiles and limit the contrast gain, so a bright "
                "headlight cannot flatten the rest of the frame."
            ),
        ),
        ParamSpec("clip_limit", "CLAHE clip limit", "float", 2.0, 0.5, 8.0, 0.1),
        ParamSpec("tile", "CLAHE tile size", "int", 8, 2, 32, 1, unit="px"),
    ),
    category="point",
    reference="s_k = (L-1) * sum_{j<=k} n_j / N",
)
def histogram_equalise(
    image: AnyArray,
    adaptive: bool = False,
    clip_limit: float = 2.0,
    tile: int = 8,
) -> UInt8Array:
    """Global histogram equalisation, or CLAHE when ``adaptive`` is set.

    The global path is implemented here from the definition - accumulate the
    histogram, scale the CDF onto the output range, and use it as a lookup
    table. The adaptive path delegates to OpenCV's CLAHE, which is a different
    algorithm rather than a faster version of this one and is labelled as such
    in the interface.
    """
    x = np.asarray(image)
    if x.ndim == 3:
        # Equalise luminance only. Equalising R, G and B independently shifts
        # the colour balance, which would corrupt the very hue and saturation
        # information S5 depends on.
        import cv2

        ycrcb = cv2.cvtColor(clip_to_uint8(x), cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = histogram_equalise(ycrcb[:, :, 0], adaptive, clip_limit, tile)
        return np.asarray(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), dtype=np.uint8)

    grey = clip_to_uint8(x)
    if adaptive:
        import cv2

        clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile), int(tile)))
        return np.asarray(clahe.apply(grey), dtype=np.uint8)

    counts = np.bincount(grey.ravel(), minlength=256)
    total = counts.sum()
    if total == 0:
        return grey
    cdf = np.cumsum(counts).astype(np.float64) / float(total)
    lut = np.rint(cdf * 255.0).astype(np.uint8)
    return np.asarray(lut[grey], dtype=np.uint8)


def match_histogram(image: AnyArray, reference: AnyArray) -> UInt8Array:
    """Map ``image`` so its histogram matches ``reference``.

    Used when comparing a degraded frame against a clean one: matching first
    removes a global brightness difference that would otherwise dominate any
    per-pixel error metric and make two visually similar frames score badly.
    """
    src = clip_to_uint8(image)
    ref = clip_to_uint8(reference)
    src_cdf = cumulative_histogram(src)
    ref_cdf = cumulative_histogram(ref)
    lut = np.searchsorted(ref_cdf, src_cdf).clip(0, 255).astype(np.uint8)
    return np.asarray(lut[src], dtype=np.uint8)
