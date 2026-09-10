"""Units 2 and 3 - the frequency domain.

The Fourier transform re-expresses an image as a sum of sinusoids. Two things
follow that this project needs.

**Filtering becomes multiplication.** Convolution in space is multiplication in
frequency, so a filter that would need a very large spatial kernel - suppressing
a narrow band of periodic interference, for instance - becomes a small mask
applied to the spectrum. Rain streaks and interference patterns are near
periodic and show up as localised energy away from the origin, which a notch
filter can remove without touching anything else.

**Displacement becomes phase.** Shifting an image multiplies its transform by a
linear phase ramp, and nothing else changes. Cross-correlating the normalised
transforms of two frames therefore concentrates all the energy of a pure shift
into a single impulse whose position is the shift - to sub-pixel precision, and
immune to any illumination change that scales the images. That is
:func:`phase_correlation`, and it does double duty: stabilising camera shake in
S2, and refining the block-matching displacement in S8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from kerbside.core.registry import ParamSpec, register
from kerbside.core.types import AnyArray, FloatArray
from kerbside.operators.point import clip_to_uint8

__all__ = [
    "FilterShape",
    "FilterBand",
    "PhaseCorrelationResult",
    "dft2",
    "idft2",
    "magnitude_spectrum",
    "phase_spectrum",
    "distance_grid",
    "frequency_filter",
    "notch_filter",
    "apply_filter",
    "filter_image",
    "phase_correlation",
    "detect_spectral_peaks",
    "radial_background",
    "suppress_periodic_noise",
    "hann_window",
]

FilterShape = Literal["ideal", "butterworth", "gaussian"]
FilterBand = Literal["lowpass", "highpass", "bandpass", "bandreject"]


def _shape2(array: AnyArray) -> tuple[int, int]:
    """The shape of a 2-D array as an exact pair.

    ``ndarray.shape`` is a variable-length tuple, so passing it straight to a
    helper that needs two dimensions loses the guarantee. Narrowing it here
    checks the assumption once instead of letting a 3-channel array reach a
    filter builder and fail somewhere less obvious.
    """
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {array.shape}")
    return (int(array.shape[0]), int(array.shape[1]))


def _to_grey_float(image: AnyArray) -> FloatArray:
    x = np.asarray(image, dtype=np.float64)
    if x.ndim == 3:
        import cv2

        x = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_BGR2GRAY).astype(np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {x.shape}")
    return x


def dft2(image: AnyArray, shift: bool = True) -> AnyArray:
    """Two-dimensional DFT.

    With ``shift`` set the zero-frequency term is moved to the centre of the
    array, which is how spectra are conventionally displayed and what makes a
    radially symmetric filter mask a radially symmetric array.
    """
    spectrum = np.fft.fft2(_to_grey_float(image))
    return np.fft.fftshift(spectrum) if shift else spectrum


def idft2(spectrum: AnyArray, shift: bool = True, real: bool = True) -> AnyArray:
    """Inverse of :func:`dft2`.

    ``real`` discards the imaginary residue, which for a spectrum derived from a
    real image is numerical noise at the 1e-12 level. Returning the complex
    array instead would push that decision onto every caller.
    """
    s = np.fft.ifftshift(spectrum) if shift else spectrum
    out = np.fft.ifft2(s)
    return np.real(out) if real else out


@register(
    "frequency.spectrum",
    "Magnitude spectrum",
    units=(3,),
    rationale=(
        "The log-scaled magnitude of the DFT, centred. The log is not "
        "decoration: the DC term is typically five orders of magnitude larger "
        "than the detail, so a linear display of the spectrum is a single bright "
        "dot and nothing else."
    ),
    params=(ParamSpec("log_scale", "Log scale", "bool", True),),
    category="frequency",
)
def magnitude_spectrum(image: AnyArray, log_scale: bool = True) -> AnyArray:
    """Displayable magnitude spectrum, normalised to 8 bits."""
    mag = np.abs(dft2(image))
    if log_scale:
        mag = np.log1p(mag)
    peak = float(mag.max())
    if peak < 1e-12:
        return np.zeros(mag.shape, dtype=np.uint8)
    return clip_to_uint8(mag * (255.0 / peak))


def phase_spectrum(image: AnyArray) -> FloatArray:
    """Phase angle of the DFT in radians, centred, in ``[-pi, pi]``.

    Worth looking at next to the magnitude: almost all the structure a human
    recognises in an image lives in the phase, which is the reason phase
    correlation works and amplitude correlation does not.
    """
    return np.asarray(np.angle(dft2(image)), dtype=np.float64)


def distance_grid(shape: tuple[int, int]) -> FloatArray:
    """Distance of each element from the centred zero-frequency term.

    Every radially symmetric filter below is a function of this array alone, so
    it is built once and shared.
    """
    rows, cols = shape
    v = np.arange(rows, dtype=np.float64) - rows // 2
    u = np.arange(cols, dtype=np.float64) - cols // 2
    return np.sqrt(v[:, None] ** 2 + u[None, :] ** 2)


def frequency_filter(
    shape: tuple[int, int],
    cutoff: float,
    shape_kind: FilterShape = "gaussian",
    band: FilterBand = "lowpass",
    order: int = 2,
    bandwidth: float = 10.0,
) -> FloatArray:
    """Build a radially symmetric transfer function ``H(u, v)``.

    The three profiles differ in how abruptly they cut, and that difference is
    visible in the result rather than academic:

    ``ideal``
        a hard disc. Its spatial-domain counterpart is a sinc, so a sharp edge
        filtered this way rings - concentric ripples around every transition.
    ``butterworth``
        a smooth roll-off whose sharpness is set by ``order``; order 1 barely
        rings, high orders approach the ideal filter and ring like it.
    ``gaussian``
        transforms to a Gaussian, which is non-negative everywhere, so it cannot
        ring at all. The safe default.
    """
    if cutoff <= 0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if order < 1:
        raise ValueError(f"Butterworth order must be at least 1, got {order}")
    d = distance_grid(shape)

    def low(cut: float) -> FloatArray:
        if shape_kind == "ideal":
            return (d <= cut).astype(np.float64)
        if shape_kind == "butterworth":
            return 1.0 / (1.0 + (d / cut) ** (2 * order))
        if shape_kind == "gaussian":
            return np.exp(-(d**2) / (2.0 * cut * cut))
        raise ValueError(
            f"unknown filter shape {shape_kind!r}; use ideal, butterworth or gaussian"
        )

    if band == "lowpass":
        return low(cutoff)
    if band == "highpass":
        return 1.0 - low(cutoff)
    if band in ("bandpass", "bandreject"):
        if bandwidth <= 0:
            raise ValueError(f"bandwidth must be positive, got {bandwidth}")
        inner = max(1e-6, cutoff - bandwidth / 2.0)
        outer = cutoff + bandwidth / 2.0
        ring = low(outer) - low(inner)
        ring = np.clip(ring, 0.0, 1.0)
        return ring if band == "bandpass" else 1.0 - ring
    raise ValueError(f"unknown band {band!r}")


def notch_filter(
    shape: tuple[int, int],
    centres: list[tuple[float, float]],
    radius: float = 6.0,
    shape_kind: FilterShape = "gaussian",
    order: int = 2,
) -> FloatArray:
    """Reject small neighbourhoods of the spectrum, and their conjugates.

    ``centres`` are offsets from the zero-frequency term in ``(row, col)``.
    Because the transform of a real image is conjugate symmetric, every notch is
    applied at its mirror position too - omitting the mirror leaves an imaginary
    residue in the reconstruction and a visible ripple in the result.
    """
    rows, cols = shape
    v = np.arange(rows, dtype=np.float64) - rows // 2
    u = np.arange(cols, dtype=np.float64) - cols // 2
    vv, uu = np.meshgrid(v, u, indexing="ij")

    h = np.ones(shape, dtype=np.float64)
    for dv, du in centres:
        if np.hypot(dv, du) <= radius:
            raise ValueError(
                f"notch centre ({dv:g}, {du:g}) lies within its own radius "
                f"{radius:g} of the zero-frequency term, so it would attenuate "
                f"the image mean rather than any interference"
            )
        for sv, su in ((dv, du), (-dv, -du)):
            d = np.sqrt((vv - sv) ** 2 + (uu - su) ** 2)
            if shape_kind == "ideal":
                h *= (d > radius).astype(np.float64)
            elif shape_kind == "butterworth":
                with np.errstate(divide="ignore"):
                    h *= 1.0 / (1.0 + (radius / np.maximum(d, 1e-9)) ** (2 * order))
            else:
                h *= 1.0 - np.exp(-(d**2) / (2.0 * radius * radius))

    # The zero-frequency term is the image mean. A notch centred nearby still
    # has a tail reaching the origin - a Gaussian notch of radius six placed
    # twelve pixels out leaves the DC at 0.87, and a second one takes it to
    # 0.76 - and dropping the mean by a quarter is a far larger change than the
    # interference being removed. Pin it to unity unconditionally.
    rows, cols = shape
    h[rows // 2, cols // 2] = 1.0
    return np.clip(h, 0.0, 1.0)


def apply_filter(spectrum: AnyArray, transfer: FloatArray) -> AnyArray:
    """Multiply a centred spectrum by a transfer function."""
    s = np.asarray(spectrum)
    h = np.asarray(transfer, dtype=np.float64)
    if s.shape != h.shape:
        raise ValueError(
            f"spectrum shape {s.shape} does not match transfer function shape {h.shape}"
        )
    return s * h


@register(
    "frequency.filter",
    "Frequency-domain filter",
    units=(2, 3),
    rationale=(
        "Filters by multiplying the centred spectrum with a transfer function. "
        "The ideal, Butterworth and Gaussian profiles differ in ringing, and "
        "seeing that difference on the same frame is the point of offering all "
        "three."
    ),
    params=(
        ParamSpec("cutoff", "Cutoff radius", "float", 40.0, 1.0, 500.0, 1.0, unit="cycles"),
        ParamSpec(
            "shape_kind", "Profile", "enum", "gaussian",
            choices=("ideal", "butterworth", "gaussian"),
        ),
        ParamSpec(
            "band", "Band", "enum", "lowpass",
            choices=("lowpass", "highpass", "bandpass", "bandreject"),
        ),
        ParamSpec("order", "Butterworth order", "int", 2, 1, 10, 1),
        ParamSpec("bandwidth", "Bandwidth", "float", 10.0, 1.0, 200.0, 1.0, unit="cycles"),
    ),
    category="frequency",
)
def filter_image(
    image: AnyArray,
    cutoff: float = 40.0,
    shape_kind: FilterShape = "gaussian",
    band: FilterBand = "lowpass",
    order: int = 2,
    bandwidth: float = 10.0,
) -> AnyArray:
    """Full round trip: transform, multiply, invert, rescale for display."""
    grey = _to_grey_float(image)
    spectrum = dft2(grey)
    transfer = frequency_filter(
        _shape2(grey), cutoff, shape_kind, band, order, bandwidth
    )
    out = idft2(apply_filter(spectrum, transfer))
    if band in ("highpass", "bandpass"):
        # A high-pass result is zero-mean and signed; offsetting it keeps the
        # negative lobes visible instead of clipping half the signal to black.
        return clip_to_uint8(out + 128.0)
    return clip_to_uint8(out)


def hann_window(shape: tuple[int, int]) -> FloatArray:
    """Separable raised-cosine window.

    The DFT treats an image as one period of a periodic signal, so the left edge
    is a discontinuity against the right edge. That discontinuity is a strong
    broadband feature, and in phase correlation it can produce a peak at zero
    displacement large enough to hide the real one. Tapering the borders to zero
    removes it.
    """
    rows, cols = shape
    wr = np.hanning(rows) if rows > 1 else np.ones(1)
    wc = np.hanning(cols) if cols > 1 else np.ones(1)
    return np.outer(wr, wc)


@dataclass(frozen=True, slots=True)
class PhaseCorrelationResult:
    """Estimated displacement between two images, with a confidence figure."""

    dx_px: float
    dy_px: float
    response: float
    peak_ratio: float

    @property
    def magnitude_px(self) -> float:
        return float(np.hypot(self.dx_px, self.dy_px))

    @property
    def is_confident(self) -> bool:
        """Whether the correlation peak stands out enough to be believed.

        A featureless pair of frames produces a broad, flat correlation surface
        whose maximum is essentially arbitrary. Reporting that as a displacement
        would make the stabiliser inject the very motion it exists to remove, so
        callers are given the means to reject it.
        """
        return self.peak_ratio > 1.6


def _upsampled_dft(
    data: AnyArray, region_size: int, upsample_factor: int, offsets: tuple[float, float]
) -> AnyArray:
    """Inverse DFT of ``data`` sampled on a fine grid over a small region.

    The inverse transform is a sum over frequencies, so it can be evaluated at
    *any* spacing simply by changing the sample positions in the exponential -
    there is nothing about it that requires the output grid to match the input
    grid. Writing that sum as two matrix products evaluates a ``region_size``
    square neighbourhood at ``1 / upsample_factor`` pixel spacing without ever
    building a zero-padded transform of the whole image, which for a 1/50 pixel
    grid would be fifty times the size in each direction.

    This is the matrix-multiply DFT refinement of Guizar-Sicairos, Thurman and
    Fienup, "Efficient subpixel image registration algorithms", Optics Letters
    33(2), 2008. It is ordinary Fourier analysis; nothing is learned or fitted.
    """
    out = data
    for axis_size, offset in zip(data.shape[::-1], offsets[::-1], strict=True):
        grid = (np.arange(region_size, dtype=np.float64) - offset)[:, None]
        kernel = np.exp(
            -2j * np.pi * grid * np.fft.fftfreq(axis_size, upsample_factor)[None, :]
        )
        out = np.tensordot(kernel, out, axes=(1, -1))
    return out


def phase_correlation(
    reference: AnyArray,
    current: AnyArray,
    window: bool = True,
    upsample_factor: int = 50,
    spectral_lowpass: float = 0.0,
) -> PhaseCorrelationResult:
    """Displacement of ``current`` relative to ``reference``, by phase correlation.

    Shifting an image by ``(dx, dy)`` multiplies its transform by
    ``exp(-2 pi i (u dx / M + v dy / N))`` and changes nothing else. So the
    normalised cross-power spectrum

        ``R = F1 conj(F2) / |F1 conj(F2)|``

    is exactly that phase ramp, and its inverse transform is an impulse at
    ``(dy, dx)``. Discarding the magnitude is what makes the method immune to
    illumination change: a frame that got uniformly brighter has the same phase.

    Sub-pixel refinement is not optional here. For a fractional shift the
    inverse transform of a pure phase ramp is a periodic sinc, whose side lobes
    alternate in sign - so the obvious three-point parabolic fit through the
    peak is fitting a parabola to something that is not remotely parabolic, and
    lands about a pixel out. Refining by evaluating the correlation on a fine
    grid around the peak, via :func:`_upsampled_dft`, gets it to roughly
    ``1 / upsample_factor`` of a pixel instead.

    Parameters
    ----------
    window
        Taper the borders. The DFT treats the image as one period of a periodic
        signal, so the left edge is a discontinuity against the right edge, and
        that broadband feature can produce a spurious peak at zero displacement.
        Measured on a real translated frame the window costs nothing - both
        settings land at about 0.04 px - and it only hurts on a synthetically
        *circular* shift, where there is no discontinuity to remove and the
        fixed taper cuts different content out of each frame. Real camera shake
        is not circular, so this is on by default.
    upsample_factor
        Sub-pixel resolution, as a reciprocal. 1 disables refinement.
    spectral_lowpass
        Optional Gaussian weighting of the cross-power spectrum, as a fraction
        of the Nyquist radius. Zero leaves the phase ramp pure; a small value
        suppresses the high frequencies where noise dominates the phase, at the
        cost of broadening the peak.

    Returns positive ``dx_px`` when the content of ``current`` sits to the right
    of where it sat in ``reference``.
    """
    a = _to_grey_float(reference)
    b = _to_grey_float(current)
    if a.shape != b.shape:
        raise ValueError(
            f"phase correlation needs identically shaped images, got {a.shape} and {b.shape}"
        )
    if a.size == 0:
        raise ValueError("cannot correlate an empty image")
    if upsample_factor < 1:
        raise ValueError(f"upsample factor must be at least 1, got {upsample_factor}")

    if window:
        w = hann_window(_shape2(a))
        a = a * w
        b = b * w

    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    denom = np.abs(cross)
    # Where the cross-power is zero there is no phase information; leaving those
    # bins at zero is correct, dividing by an epsilon would amplify pure noise.
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(denom > 1e-12, cross / np.maximum(denom, 1e-12), 0.0)

    if spectral_lowpass > 0:
        rows, cols = a.shape
        v = np.fft.fftfreq(rows)[:, None]
        u = np.fft.fftfreq(cols)[None, :]
        radius = np.sqrt(u**2 + v**2) / 0.5
        normalised = normalised * np.exp(-(radius**2) / (2.0 * spectral_lowpass**2))

    surface = np.real(np.fft.ifft2(normalised))
    rows, cols = surface.shape
    py, px = np.unravel_index(int(np.argmax(surface)), surface.shape)
    response = float(surface[py, px])

    # Confidence: the peak against the strongest value well away from it.
    masked = surface.copy()
    masked[
        max(0, py - 3) : min(rows, py + 4), max(0, px - 3) : min(cols, px + 4)
    ] = -np.inf
    runner_up = float(np.max(masked)) if np.isfinite(masked).any() else 0.0
    peak_ratio = float(response / runner_up) if runner_up > 1e-12 else float("inf")

    # Frequencies above the Nyquist point represent negative displacements.
    dy = float(py) - (rows if py > rows // 2 else 0)
    dx = float(px) - (cols if px > cols // 2 else 0)

    if upsample_factor > 1:
        coarse = np.array([dy, dx], dtype=np.float64)
        region = int(np.ceil(upsample_factor * 1.5))
        centre = float(region // 2)
        offsets = tuple(centre - coarse * upsample_factor)
        fine = _upsampled_dft(
            np.conj(normalised), region, upsample_factor, (offsets[0], offsets[1])
        )
        peak = np.unravel_index(int(np.argmax(np.abs(fine))), fine.shape)
        dy, dx = coarse + (np.array(peak, dtype=np.float64) - centre) / upsample_factor

    return PhaseCorrelationResult(
        dx_px=-dx, dy_px=-dy, response=response, peak_ratio=peak_ratio
    )


def radial_background(magnitude: AnyArray) -> FloatArray:
    """Median spectral magnitude at each integer radius from the origin.

    Natural image spectra fall off steeply with frequency, so a single global
    median is dominated by the vast high-frequency plateau and is minuscule
    beside anything near the centre. Scoring a candidate against it calls
    ordinary low-frequency image content a hundredfold outlier. The meaningful
    question is whether a bin stands out among *other frequencies of similar
    radius*, which is what this profile provides.
    """
    mag = np.asarray(magnitude, dtype=np.float64)
    index = np.rint(distance_grid(_shape2(mag))).astype(np.int64)
    bins = int(index.max()) + 1
    order = np.argsort(index.ravel(), kind="stable")
    sorted_index = index.ravel()[order]
    sorted_mag = mag.ravel()[order]
    starts = np.searchsorted(sorted_index, np.arange(bins), side="left")
    ends = np.searchsorted(sorted_index, np.arange(bins), side="right")

    profile = np.zeros(bins, dtype=np.float64)
    for r in range(bins):
        if ends[r] > starts[r]:
            profile[r] = float(np.median(sorted_mag[starts[r] : ends[r]]))
    return profile


def detect_spectral_peaks(
    image: AnyArray,
    exclude_radius: float = 16.0,
    count: int = 4,
    min_ratio: float = 6.0,
    min_dominance: float = 4.0,
) -> list[tuple[float, float]]:
    """Find isolated spectral peaks that indicate periodic interference.

    Periodic patterns - rain at a consistent angle and spacing, interference
    from a nearby supply, the beat between a rolling shutter and a flickering
    lamp - concentrate energy at a few frequencies away from the origin. Genuine
    image content does not: it is broadband and falls off smoothly with radius.

    Two conditions must both hold, and the second is the one that does the work.

    ``min_ratio``
        the bin must exceed this multiple of the median magnitude *at its own
        radius* (see :func:`radial_background`). A floor, cheaply computed.
    ``min_dominance``
        the bin must exceed this multiple of the 99.9th percentile of the whole
        prominence field - that is, it must stand well above the strongest
        *ordinary* content in this same image.

    The second condition exists because the absolute prominence of ordinary
    content grows with image size: measured on real road frames the strongest
    natural bin scores about 12 on a 128 pixel patch, 29 on a 256 pixel patch
    and 63 on a full 720p frame, purely because a larger annulus holds more bins
    for the maximum to be drawn from. Any fixed threshold therefore either
    misses interference in small images or shreds large ones. A genuine periodic
    component, by contrast, scores over a thousand against a runner-up of ten -
    it is not a slightly larger member of a continuum, it is a spike - so
    comparing a candidate against the same image's own ceiling is
    scale-invariant and separates the two cases by more than an order of
    magnitude.
    """
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if min_dominance < 1.0:
        raise ValueError(f"dominance must be at least 1, got {min_dominance}")

    mag = np.abs(dft2(image))
    d = distance_grid(_shape2(mag))
    profile = radial_background(mag)
    background = profile[np.clip(np.rint(d).astype(np.int64), 0, len(profile) - 1)]

    with np.errstate(divide="ignore", invalid="ignore"):
        prominence = np.where(background > 1e-12, mag / np.maximum(background, 1e-12), 0.0)
    prominence = np.where(d > exclude_radius, prominence, 0.0)
    if not np.any(prominence > 0):
        return []

    ceiling = float(np.percentile(prominence[prominence > 0], 99.9))
    threshold = max(min_ratio, min_dominance * ceiling)

    rows, cols = mag.shape
    work = prominence.copy()
    suppress = max(2, int(exclude_radius))
    peaks: list[tuple[float, float]] = []
    for _ in range(count):
        py, px = np.unravel_index(int(np.argmax(work)), work.shape)
        if work[py, px] < threshold:
            break
        peaks.append((float(py - rows // 2), float(px - cols // 2)))
        # The transform of a real image is conjugate symmetric, so a peak and
        # its mirror are the same physical frequency; suppress both or every
        # peak is reported twice and the effective count is halved.
        for cy, cx in ((py, px), (rows - py, cols - px)):
            work[
                max(0, cy - suppress) : min(rows, cy + suppress + 1),
                max(0, cx - suppress) : min(cols, cx + suppress + 1),
            ] = 0.0
    return peaks


@register(
    "frequency.suppress_periodic",
    "Periodic noise removal",
    units=(2, 3),
    rationale=(
        "Locates isolated spectral peaks and notches them out with their "
        "conjugates. Rain at a consistent angle and spacing, and electrical "
        "interference, are near periodic and localise in the spectrum; image "
        "content does not, so the notch removes them and leaves the scene."
    ),
    params=(
        ParamSpec("radius", "Notch radius", "float", 6.0, 1.0, 40.0, 0.5, unit="cycles"),
        ParamSpec("count", "Peaks to notch", "int", 4, 1, 12, 1),
        ParamSpec("exclude_radius", "Protect low frequencies", "float", 16.0, 1.0, 100.0, 1.0),
        ParamSpec("min_ratio", "Peak prominence", "float", 6.0, 1.1, 60.0, 0.1),
        ParamSpec(
            "min_dominance", "Peak dominance", "float", 4.0, 1.0, 40.0, 0.1,
            description=(
                "How far a peak must stand above the strongest ordinary "
                "content in the same image. Scale-invariant, unlike a raw "
                "prominence threshold."
            ),
        ),
    ),
    category="frequency",
)
def suppress_periodic_noise(
    image: AnyArray,
    radius: float = 6.0,
    count: int = 4,
    exclude_radius: float = 16.0,
    min_ratio: float = 6.0,
    min_dominance: float = 4.0,
) -> AnyArray:
    """Detect and notch out periodic interference.

    Returns the input unchanged when no peak is prominent enough, which is the
    honest outcome for a clean frame - inventing notches would blur it for no
    reason.

    ``exclude_radius`` is raised to at least twice ``radius`` before searching.
    A notch has a tail, and one placed closer to the origin than its own width
    attenuates the image mean far more than it attenuates the interference.
    """
    peaks = detect_spectral_peaks(
        image, max(exclude_radius, 2.0 * radius), count, min_ratio, min_dominance
    )
    grey = _to_grey_float(image)
    if not peaks:
        return clip_to_uint8(grey)
    transfer = notch_filter(_shape2(grey), peaks, radius=radius)
    return clip_to_uint8(idft2(apply_filter(dft2(grey), transfer)))
