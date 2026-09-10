"""Unit 7 - colour models and colour-based discrimination.

The headline result of this project lives here. A cast shadow darkens the road
without changing what the road is made of, so in a colour model that separates
*how bright* from *what colour*, a shadow is a large change in one channel and a
small change in the others. In RGB it is a large change in all three, and there
is no threshold on RGB that distinguishes a shadow from a dark grey car.

That is the entire argument for HSV shadow suppression, and it is why the
comparison in :func:`compare_colour_spaces` is worth running rather than
asserting: RGB, HSV, YCbCr, normalised rgb and c1c2c3 all claim some separation
of intensity from chromaticity, and they do not all deliver it equally.

Hue is an angle, and this module never forgets that. A naive ``abs(h1 - h2)``
says that hue 179 and hue 0 - two shades of the same red - are maximally
different, which fails on precisely the vehicles the pipeline needs to handle.
:func:`hue_difference` is circular, everywhere, without exception.

Conventions: images are BGR uint8, as OpenCV delivers them. Hue is in OpenCV's
0-179 range for HSV; saturation and value are normalised to 0-1 by the helpers
that return float arrays, and every such function says so in its name or its
docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, BoolArray, FloatArray, UInt8Array

__all__ = [
    "ColourSpace",
    "ColourName",
    "COLOUR_NAME_TABLE",
    "HUE_MAX",
    "bgr_to_hsv",
    "bgr_to_hsi",
    "bgr_to_ycbcr",
    "bgr_to_yuv",
    "bgr_to_normalised_rgb",
    "bgr_to_c1c2c3",
    "convert",
    "hue_difference",
    "colour_slice",
    "colour_histogram",
    "dominant_hue",
    "name_colour",
    "shadow_mask_hsv",
    "shadow_mask_in_space",
    "compare_colour_spaces",
]

ColourSpace = Literal["rgb", "hsv", "hsi", "ycbcr", "yuv", "normalised_rgb", "c1c2c3"]

HUE_MAX = 180.0
"""OpenCV packs hue into 0-179 so it fits in a byte. Every circular calculation
here uses this modulus rather than 360."""


def _as_bgr_u8(image: AnyArray) -> UInt8Array:
    x = np.asarray(image)
    if x.ndim == 2:
        x = cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected a 3-channel BGR image, got shape {x.shape}")
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return np.asarray(x, dtype=np.uint8)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def bgr_to_hsv(image: AnyArray) -> FloatArray:
    """BGR to HSV as float: hue in 0-179, saturation and value in 0-1.

    Value is ``max(R, G, B)`` and saturation is ``(max - min) / max``, so
    scaling all three channels by a constant - which is exactly what a shadow
    does to a surface - scales value and leaves hue and saturation untouched.
    That property is the reason S5 works.
    """
    hsv = cv2.cvtColor(_as_bgr_u8(image), cv2.COLOR_BGR2HSV).astype(np.float64)
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0
    return hsv


def bgr_to_hsi(image: AnyArray) -> FloatArray:
    """BGR to HSI, from the classical definition rather than a library call.

    ``I = (R + G + B) / 3``
    ``S = 1 - 3 min(R, G, B) / (R + G + B)``
    ``H = arccos( [ (R-G) + (R-B) ] / 2 sqrt( (R-G)^2 + (R-B)(G-B) ) )``,
    with ``H = 360 - H`` when ``B > G``.

    HSI differs from HSV in what it calls brightness: intensity is the *mean* of
    the channels where value is the maximum. For shadow discrimination that
    makes HSI slightly better behaved on saturated colours, since a shadow
    scales the mean as cleanly as it scales the maximum, and slightly worse on
    near-neutral surfaces where the saturation denominator gets small. The
    comparison below measures which effect wins on road scenes.

    Hue is returned on the same 0-179 scale as :func:`bgr_to_hsv` so the two are
    interchangeable in the shadow test; saturation and intensity are 0-1.
    """
    bgr = _as_bgr_u8(image).astype(np.float64) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]

    intensity = (r + g + b) / 3.0
    minimum = np.minimum(np.minimum(r, g), b)
    total = r + g + b
    with np.errstate(divide="ignore", invalid="ignore"):
        saturation = np.where(total > 1e-9, 1.0 - 3.0 * minimum / np.maximum(total, 1e-9), 0.0)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt(np.maximum((r - g) ** 2 + (r - b) * (g - b), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine = np.where(denominator > 1e-9, numerator / np.maximum(denominator, 1e-9), 0.0)
    hue = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    hue = np.where(b > g, 360.0 - hue, hue)
    # An achromatic pixel has no defined hue; reporting zero is conventional and
    # harmless because the saturation test rejects those pixels anyway.
    hue = np.where(saturation < 1e-9, 0.0, hue)

    out = np.empty(bgr.shape, dtype=np.float64)
    out[..., 0] = hue * (HUE_MAX / 360.0)
    out[..., 1] = saturation
    out[..., 2] = intensity
    return out


def bgr_to_ycbcr(image: AnyArray) -> FloatArray:
    """BGR to YCbCr, all channels normalised to 0-1.

    The broadcast colour model: one luma channel and two colour-difference
    channels. It was designed so luma could be transmitted at full resolution
    and chroma at reduced resolution, which is a statement that the two are
    largely independent - the same property S5 needs.
    """
    ycc = cv2.cvtColor(_as_bgr_u8(image), cv2.COLOR_BGR2YCrCb).astype(np.float64) / 255.0
    return ycc[..., [0, 2, 1]]  # reorder to Y, Cb, Cr


def bgr_to_yuv(image: AnyArray) -> FloatArray:
    """BGR to YUV, all channels normalised to 0-1."""
    return np.asarray(
        cv2.cvtColor(_as_bgr_u8(image), cv2.COLOR_BGR2YUV).astype(np.float64) / 255.0
    )


def bgr_to_normalised_rgb(image: AnyArray) -> FloatArray:
    """Chromaticity coordinates ``r = R/(R+G+B)`` and so on.

    Dividing out the total intensity makes the result invariant to any uniform
    scaling of the channels, so in principle a shadow changes nothing at all.
    In practice the division becomes unstable exactly where the shadow is
    darkest and the total smallest, which is the trade-off the comparison
    measures. Returned as ``(r, g, total/765)`` so the third channel still
    carries the intensity that was divided out.
    """
    bgr = _as_bgr_u8(image).astype(np.float64)
    total = bgr.sum(axis=2)
    safe = np.maximum(total, 1e-6)
    out = np.empty(bgr.shape, dtype=np.float64)
    out[..., 0] = bgr[..., 2] / safe
    out[..., 1] = bgr[..., 1] / safe
    out[..., 2] = total / 765.0
    return out


def bgr_to_c1c2c3(image: AnyArray) -> FloatArray:
    """The c1c2c3 photometric invariants of Gevers and Smeulders.

    ``c1 = arctan(R / max(G, B))`` and cyclic permutations, scaled to 0-1.

    Each channel is a ratio of two colour channels, so any illumination change
    that scales all three equally cancels exactly - which makes c1c2c3
    invariant to shadow *by construction* rather than by threshold. The catch is
    that the arctan of a ratio is extremely noisy when both channels are small,
    so its advantage on brightly lit surfaces can reverse in deep shadow. That
    is a real result and the comparison reports it either way.
    """
    bgr = _as_bgr_u8(image).astype(np.float64)
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    eps = 1e-6
    out = np.empty(bgr.shape, dtype=np.float64)
    out[..., 0] = np.arctan(r / np.maximum(np.maximum(g, b), eps))
    out[..., 1] = np.arctan(g / np.maximum(np.maximum(r, b), eps))
    out[..., 2] = np.arctan(b / np.maximum(np.maximum(r, g), eps))
    return out / (np.pi / 2.0)


_CONVERTERS = {
    "rgb": lambda img: np.asarray(_as_bgr_u8(img)[..., ::-1], dtype=np.float64) / 255.0,
    "hsv": bgr_to_hsv,
    "hsi": bgr_to_hsi,
    "ycbcr": bgr_to_ycbcr,
    "yuv": bgr_to_yuv,
    "normalised_rgb": bgr_to_normalised_rgb,
    "c1c2c3": bgr_to_c1c2c3,
}


def convert(image: AnyArray, space: ColourSpace) -> FloatArray:
    """Convert a BGR image into a named colour space."""
    try:
        return np.asarray(_CONVERTERS[space](image), dtype=np.float64)
    except KeyError:
        raise ValueError(
            f"unknown colour space {space!r}; available: {', '.join(_CONVERTERS)}"
        ) from None


# --------------------------------------------------------------------------
# Hue arithmetic
# --------------------------------------------------------------------------


def hue_difference(a: AnyArray, b: AnyArray, modulus: float = HUE_MAX) -> FloatArray:
    """Shortest angular distance between two hues, in ``[0, modulus / 2]``.

    Trap 6, in one function. Hue wraps: in OpenCV's 0-179 encoding, 179 and 0
    are adjacent, and a linear ``abs(a - b)`` calls them 179 apart - the maximum
    possible - when they are one step apart. Red is exactly where the wrap
    happens, so a naive difference fails on red vehicles specifically, which is
    the kind of bug that looks like a data problem rather than an arithmetic one.
    """
    diff = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return np.asarray(np.minimum(diff, modulus - diff), dtype=np.float64)


# --------------------------------------------------------------------------
# Shadow discrimination
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadowThresholds:
    """The four thresholds of the HSV shadow test.

    Defaults follow the classical formulation and the ranges the renderer's
    physical shadow model actually produces on road surfaces.
    """

    value_ratio_low: float = 0.40
    value_ratio_high: float = 0.85
    saturation_delta: float = 0.10
    hue_delta: float = 10.0

    def validate(self) -> None:
        if not 0.0 < self.value_ratio_low < self.value_ratio_high < 1.0:
            raise ValueError(
                f"the value-ratio band must satisfy 0 < low < high < 1, got "
                f"({self.value_ratio_low}, {self.value_ratio_high}); a band "
                f"reaching 1 would call unchanged road a shadow, and a low of 0 "
                f"would call a black car one"
            )
        if not 0.0 <= self.hue_delta <= HUE_MAX / 2:
            raise ValueError(
                f"hue tolerance must be within 0..{HUE_MAX / 2}, got {self.hue_delta}"
            )


@register(
    "colour.shadow_hsv",
    "HSV shadow suppression",
    units=(7,),
    rationale=(
        "Classifies a foreground pixel as shadow when it is darker than the "
        "background but not black, no more saturated than it, and close to it in "
        "hue. A shadow scales the surface reflectance, so value drops while hue "
        "and saturation survive; a dark vehicle changes all three. Removing "
        "shadow pixels is what splits two vehicles a shadow had bridged into one "
        "blob - which presents as a tracking failure but is a colour-processing "
        "one."
    ),
    params=(
        ParamSpec(
            "value_ratio_low", "Value ratio, low", "float", 0.40, 0.01, 0.95, 0.01,
            description="Below this the pixel is too dark to be a shadow - it is a dark object.",
        ),
        ParamSpec(
            "value_ratio_high", "Value ratio, high", "float", 0.85, 0.05, 0.999, 0.01,
            description="Above this the pixel is barely darker than the background.",
        ),
        ParamSpec(
            "saturation_delta", "Saturation tolerance", "float", 0.10, 0.0, 1.0, 0.01,
            description="How much more saturated than the background a shadow may be.",
        ),
        ParamSpec(
            "hue_delta", "Hue tolerance", "float", 10.0, 0.0, 90.0, 0.5, unit="deg/2",
            description="Circular hue tolerance on OpenCV's 0-179 scale.",
        ),
    ),
    category="colour",
)
def shadow_mask_hsv(
    frame: AnyArray,
    background: AnyArray,
    foreground: AnyArray | None = None,
    value_ratio_low: float = 0.40,
    value_ratio_high: float = 0.85,
    saturation_delta: float = 0.10,
    hue_delta: float = 10.0,
) -> BoolArray:
    """Pixels that are shadow rather than object.

    All three conditions must hold:

    * **value ratio** ``V_frame / V_background`` inside ``[low, high]``. Darker,
      but not black - a black car is not a shadow, and neither is unchanged road.
    * **saturation difference** ``S_frame - S_background <= delta``. A shadowed
      surface does not become more colourful. The test is one-sided on purpose:
      shadows commonly desaturate, so a large *negative* difference is still a
      shadow.
    * **hue difference** within ``delta``, computed circularly.

    ``foreground`` restricts the test to pixels already believed to be
    foreground, which is how S5 uses it: shadow suppression subtracts from the
    foreground mask, it does not add to it.
    """
    thresholds = ShadowThresholds(
        value_ratio_low, value_ratio_high, saturation_delta, hue_delta
    )
    thresholds.validate()

    hsv_frame = bgr_to_hsv(frame)
    hsv_background = bgr_to_hsv(background)
    if hsv_frame.shape != hsv_background.shape:
        raise ValueError(
            f"frame shape {hsv_frame.shape} does not match background shape "
            f"{hsv_background.shape}"
        )

    v_frame = hsv_frame[..., 2]
    v_background = hsv_background[..., 2]
    # Where the background is essentially black there is no ratio to speak of;
    # calling those pixels shadow would light up every dark corner of the frame.
    valid = v_background > 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, v_frame / np.maximum(v_background, 1e-6), 0.0)

    is_shadow = (
        valid
        & (ratio >= value_ratio_low)
        & (ratio <= value_ratio_high)
        & ((hsv_frame[..., 1] - hsv_background[..., 1]) <= saturation_delta)
        & (hue_difference(hsv_frame[..., 0], hsv_background[..., 0]) <= hue_delta)
    )
    if foreground is not None:
        is_shadow &= np.asarray(foreground, dtype=bool)
    return np.asarray(is_shadow, dtype=bool)


def shadow_mask_in_space(
    frame: AnyArray,
    background: AnyArray,
    space: ColourSpace,
    foreground: AnyArray | None = None,
    brightness_ratio: tuple[float, float] = (0.40, 0.85),
    chroma_tolerance: float = 0.10,
) -> BoolArray:
    """The same shadow test expressed in any of the colour spaces.

    Each space nominates one channel as brightness and the remaining two as
    chromaticity, then applies the identical logic: brightness must fall by a
    ratio inside the band, and chromaticity must be close to unchanged. Putting
    every space through the same decision is what makes the comparison fair -
    otherwise a space could look better only because it was given a better rule.

    ``rgb`` has no brightness channel to nominate, so it uses the mean of the
    three and the deviations from it, which is the most generous reading
    available and still the weakest performer.
    """
    lo, hi = brightness_ratio
    if not 0.0 < lo < hi:
        raise ValueError(f"brightness ratio band must satisfy 0 < low < high, got {(lo, hi)}")

    if space == "hsv":
        return np.asarray(
            shadow_mask_hsv(
                frame,
                background,
                foreground,
                lo,
                hi,
                chroma_tolerance,
                chroma_tolerance * HUE_MAX / 2,
            ),
            dtype=bool,
        )

    a = convert(frame, space)
    b = convert(background, space)
    if space == "hsi":
        brightness_a, brightness_b = a[..., 2], b[..., 2]
        chroma_diff = np.maximum(
            hue_difference(a[..., 0], b[..., 0]) / (HUE_MAX / 2),
            np.abs(a[..., 1] - b[..., 1]),
        )
    elif space in ("ycbcr", "yuv"):
        brightness_a, brightness_b = a[..., 0], b[..., 0]
        chroma_diff = np.maximum(
            np.abs(a[..., 1] - b[..., 1]), np.abs(a[..., 2] - b[..., 2])
        )
    elif space == "normalised_rgb":
        brightness_a, brightness_b = a[..., 2], b[..., 2]
        chroma_diff = np.maximum(
            np.abs(a[..., 0] - b[..., 0]), np.abs(a[..., 1] - b[..., 1])
        )
    elif space == "c1c2c3":
        # Every channel is already illumination invariant, so brightness has to
        # come from outside the space; the difference is measured across all
        # three invariants.
        grey_a = _as_bgr_u8(frame).astype(np.float64).mean(axis=2) / 255.0
        grey_b = _as_bgr_u8(background).astype(np.float64).mean(axis=2) / 255.0
        brightness_a, brightness_b = grey_a, grey_b
        chroma_diff = np.abs(a - b).max(axis=2)
    elif space == "rgb":
        brightness_a = a.mean(axis=2)
        brightness_b = b.mean(axis=2)
        chroma_diff = np.abs(
            (a - brightness_a[..., None]) - (b - brightness_b[..., None])
        ).max(axis=2)
    else:
        raise ValueError(f"unknown colour space {space!r}")

    valid = brightness_b > 1e-3
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(valid, brightness_a / np.maximum(brightness_b, 1e-6), 0.0)
    mask = valid & (ratio >= lo) & (ratio <= hi) & (chroma_diff <= chroma_tolerance)
    if foreground is not None:
        mask &= np.asarray(foreground, dtype=bool)
    return np.asarray(mask, dtype=bool)


def compare_colour_spaces(
    frame: AnyArray,
    background: AnyArray,
    shadow_truth: AnyArray,
    object_truth: AnyArray,
    spaces: tuple[ColourSpace, ...] = (
        "rgb", "hsv", "hsi", "ycbcr", "normalised_rgb", "c1c2c3",
    ),
    chroma_tolerance: float = 0.10,
) -> dict[str, dict[str, float]]:
    """Score shadow discrimination in each colour space against ground truth.

    Reports, per space, the fraction of true shadow pixels correctly identified
    and the fraction of true object pixels wrongly called shadow. Both matter
    and they trade against each other: a space that calls everything shadow has
    perfect recall and destroys every vehicle.

    The summary figure is the difference between the two, which is the
    separability that the decision actually depends on.
    """
    truth_shadow = np.asarray(shadow_truth, dtype=bool)
    truth_object = np.asarray(object_truth, dtype=bool)
    if truth_shadow.shape != truth_object.shape:
        raise ValueError("the shadow and object truth masks must have the same shape")
    if not truth_shadow.any() or not truth_object.any():
        raise ValueError(
            "the comparison needs both shadow and object pixels present; got "
            f"{int(truth_shadow.sum())} shadow and {int(truth_object.sum())} object"
        )

    out: dict[str, dict[str, float]] = {}
    for space in spaces:
        mask = shadow_mask_in_space(
            frame, background, space, chroma_tolerance=chroma_tolerance
        )
        recall = float((mask & truth_shadow).sum() / truth_shadow.sum())
        false_positive = float((mask & truth_object).sum() / truth_object.sum())
        out[space] = {
            "shadow_recall": recall,
            "object_false_positive": false_positive,
            "separability": recall - false_positive,
        }
    return out


# --------------------------------------------------------------------------
# Colour naming
# --------------------------------------------------------------------------


ColourName = str

COLOUR_NAME_TABLE: tuple[tuple[str, float, float], ...] = (
    ("red", 0.0, 10.0),
    ("orange", 10.0, 22.0),
    ("yellow", 22.0, 33.0),
    ("green", 33.0, 78.0),
    ("cyan", 78.0, 100.0),
    ("blue", 100.0, 130.0),
    ("purple", 130.0, 160.0),
    ("red", 160.0, 180.0),
)
"""Hue bands on OpenCV's 0-179 scale, with red appearing at both ends because
hue is circular. Written down as a table rather than buried in conditionals
because the report has to state it: a naming scheme nobody can inspect is a
naming scheme nobody can disagree with."""

ACHROMATIC_SATURATION = 0.18
"""Below this saturation a pixel has no meaningful hue and is named by value
alone. Set from the vehicle palette: silver and grey bodies sit around 0.05 and
the least saturated chromatic body colour is comfortably above 0.3."""

ACHROMATIC_BANDS: tuple[tuple[str, float], ...] = (
    ("black", 0.22),
    ("grey", 0.62),
    ("silver", 0.80),
    ("white", 1.01),
)
"""Value thresholds naming an achromatic pixel. Upper bounds, checked in order.

Silver earns its own band rather than being folded into white because it is one
of the most common body colours on the road, and because the two are far apart
in value: on the reference palette silver sits at 0.69 and white at 0.92.
Collapsing them would misname a large fraction of real traffic and would make
the class-and-colour table in the report quietly wrong."""


def colour_histogram(
    image: AnyArray, mask: AnyArray | None = None, bins: int = 36
) -> FloatArray:
    """Normalised hue histogram over the masked pixels."""
    hsv = bgr_to_hsv(image)
    hue = hsv[..., 0]
    selected = hue[np.asarray(mask, dtype=bool)] if mask is not None else hue.ravel()
    if selected.size == 0:
        return np.zeros(bins, dtype=np.float64)
    counts, _ = np.histogram(selected, bins=bins, range=(0.0, HUE_MAX))
    total = counts.sum()
    return counts.astype(np.float64) / total if total else counts.astype(np.float64)


def dominant_hue(
    image: AnyArray, mask: AnyArray, bins: int = 36
) -> tuple[float, float]:
    """Dominant hue and the fraction of pixels in its bin.

    The fraction is returned because it is the confidence: a two-tone vehicle,
    or a blob that is half road, produces a flat histogram whose peak bin means
    nothing, and the caller needs to be able to decline to name it.
    """
    hist = colour_histogram(image, mask, bins)
    if not hist.any():
        return (0.0, 0.0)
    peak = int(np.argmax(hist))
    width = HUE_MAX / bins
    return ((peak + 0.5) * width, float(hist[peak]))


@register(
    "colour.name",
    "Colour naming",
    units=(7,),
    rationale=(
        "Names a vehicle's colour from the hue histogram of its body pixels, "
        "after discarding the darkest and brightest deciles - which are windows "
        "and specular highlights, not paint. Achromatic bodies have no usable "
        "hue at all and are named by value instead, which is most of the vehicle "
        "parc and the part a hue-only scheme gets wrong."
    ),
    params=(
        ParamSpec("bins", "Hue bins", "int", 36, 6, 180, 6),
        ParamSpec(
            "trim_decile", "Trim fraction", "float", 0.10, 0.0, 0.4, 0.01,
            description="Fraction discarded from each end of the value distribution.",
        ),
        ParamSpec("achromatic_saturation", "Achromatic saturation", "float", 0.18, 0.0, 1.0, 0.01),
    ),
    category="colour",
)
def name_colour(
    image: AnyArray,
    mask: AnyArray,
    bins: int = 36,
    trim_decile: float = 0.10,
    achromatic_saturation: float = ACHROMATIC_SATURATION,
) -> tuple[ColourName, tuple[float, float, float], float]:
    """Name the dominant colour of the masked region.

    Returns ``(name, (hue, saturation, value), confidence)``.

    Windows are much darker than the paint around them and specular highlights
    much brighter, and both have hues unrelated to the body. Trimming the
    extremes of the *value* distribution removes them without needing to know
    where on the vehicle they are.
    """
    region = np.asarray(mask, dtype=bool)
    if region.shape != np.asarray(image).shape[:2]:
        raise ValueError("mask shape does not match image shape")
    if not region.any():
        return ("unknown", (0.0, 0.0, 0.0), 0.0)

    hsv = bgr_to_hsv(image)
    values = hsv[..., 2][region]
    if trim_decile > 0 and values.size >= 10:
        lo = float(np.quantile(values, trim_decile))
        hi = float(np.quantile(values, 1.0 - trim_decile))
        keep = region & (hsv[..., 2] >= lo) & (hsv[..., 2] <= hi)
        if keep.sum() >= max(8, region.sum() // 20):
            region = keep

    saturation = float(np.median(hsv[..., 1][region]))
    value = float(np.median(hsv[..., 2][region]))

    if saturation < achromatic_saturation:
        for name, upper in ACHROMATIC_BANDS:
            if value < upper:
                # Confidence for an achromatic body is how clearly it is
                # unsaturated, not how peaked its hue histogram is.
                confidence = float(
                    np.clip(1.0 - saturation / max(achromatic_saturation, 1e-6), 0.0, 1.0)
                )
                return (name, (0.0, saturation, value), confidence)

    hue, fraction = dominant_hue(image, region, bins)
    for name, low, high in COLOUR_NAME_TABLE:
        if low <= hue < high:
            return (name, (hue, saturation, value), fraction)
    return ("unknown", (hue, saturation, value), fraction)


@register(
    "colour.slice",
    "Colour slicing",
    units=(7,),
    rationale=(
        "Keeps only pixels within a tolerance of a reference colour and "
        "desaturates the rest. Used in the inspector to show exactly which "
        "pixels a colour rule selects, which is the quickest way to see why a "
        "vehicle was named as it was."
    ),
    params=(
        ParamSpec("hue", "Reference hue", "float", 0.0, 0.0, 179.0, 1.0),
        ParamSpec("hue_tolerance", "Hue tolerance", "float", 12.0, 0.5, 90.0, 0.5),
        ParamSpec("min_saturation", "Minimum saturation", "float", 0.25, 0.0, 1.0, 0.01),
        ParamSpec("keep_colour", "Keep colour outside", "bool", False),
    ),
    category="colour",
)
def colour_slice(
    image: AnyArray,
    hue: float = 0.0,
    hue_tolerance: float = 12.0,
    min_saturation: float = 0.25,
    keep_colour: bool = False,
) -> UInt8Array:
    """Highlight pixels near a reference hue, greying out the rest."""
    bgr = _as_bgr_u8(image)
    hsv = bgr_to_hsv(bgr)
    selected = (
        hue_difference(hsv[..., 0], np.full(hsv.shape[:2], float(hue))) <= hue_tolerance
    ) & (hsv[..., 1] >= min_saturation)

    if keep_colour:
        out = bgr.copy()
        out[selected] = 255
        return out
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    greyed = np.asarray(cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    greyed[selected] = bgr[selected]
    return greyed
    return np.asarray(out, dtype=np.uint8)
