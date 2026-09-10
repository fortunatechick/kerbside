"""Controlled degradation of rendered frames.

Real roadside footage is worse than a clean render: it is low resolution, it has
been through a lossy encoder at a bitrate someone chose to save disk, it is
blurred by the shutter, it is noisy, and at night it is under-exposed. The
resolution and compression study (interface 6.8, preset 5) answers a real
operational question - how far can CCTV bitrate drop before counting degrades -
and that question only has an answer if the degradations are applied in a
controlled, repeatable way.

Every function here is a pure frame-to-frame transform with an explicit
parameter and a seeded RNG where randomness is involved, so a degradation sweep
is reproducible.

Syllabus context: Unit 1 (spatial and grey-level resolution), Unit 2
(neighbourhood processing and point operations) and Unit 3 (DCT-based
compression), applied as the *input* to the pipeline rather than as an operator
inside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from kerbside.core.types import RESOLUTIONS_PX, UInt8Array

__all__ = [
    "DegradationConfig",
    "resample",
    "to_resolution",
    "jpeg_recompress",
    "motion_blur",
    "sensor_noise",
    "low_light",
    "quantise_grey_levels",
    "degrade",
]


@dataclass(frozen=True, slots=True)
class DegradationConfig:
    """One point in the degradation sweep.

    Applied in physical order: the optics and shutter act first, then the
    sensor, then the encoder. Applying JPEG before noise would model an
    impossible camera and would make the compression study read better than it
    should.
    """

    scale: float = 1.0
    resolution: str | None = None
    motion_blur_px: float = 0.0
    motion_blur_angle_deg: float = 90.0
    noise_sigma: float = 0.0
    gamma: float = 1.0
    exposure: float = 1.0
    grey_levels: int = 0
    jpeg_quality: int = 0
    restore_size: bool = False
    seed: int = 0

    def label(self) -> str:
        """Short label for a chart axis or a table row."""
        parts: list[str] = []
        if self.resolution:
            parts.append(self.resolution)
        elif self.scale != 1.0:
            parts.append(f"x{self.scale:g}")
        if self.jpeg_quality:
            parts.append(f"q{self.jpeg_quality}")
        if self.motion_blur_px:
            parts.append(f"blur{self.motion_blur_px:g}px")
        if self.noise_sigma:
            parts.append(f"n{self.noise_sigma:g}")
        if self.gamma != 1.0 or self.exposure != 1.0:
            parts.append(f"g{self.gamma:g}/e{self.exposure:g}")
        if self.grey_levels:
            parts.append(f"{self.grey_levels}lv")
        return " ".join(parts) if parts else "clean"


def resample(frame: UInt8Array, scale: float, restore_size: bool = False) -> UInt8Array:
    """Reduce spatial resolution by ``scale``, optionally scaling back up.

    Downsampling uses ``INTER_AREA``, which averages the source pixels covered
    by each destination pixel and is the correct anti-aliasing filter for
    minification; upsampling back uses bilinear interpolation. Restoring the
    original size is what makes a resolution sweep comparable frame-for-frame -
    the information is gone, but the array shape is unchanged, so the same
    downstream parameters apply.
    """
    if not 0 < scale <= 1.0:
        raise ValueError(f"scale must be in (0, 1], got {scale}")
    if scale == 1.0:
        return frame.copy()
    h, w = frame.shape[:2]
    small = cv2.resize(
        frame,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    if not restore_size:
        return small
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def to_resolution(
    frame: UInt8Array, resolution: str, restore_size: bool = False
) -> UInt8Array:
    """Resample to one of the named study resolutions (1080p/720p/480p/360p)."""
    if resolution not in RESOLUTIONS_PX:
        raise ValueError(
            f"unknown resolution {resolution!r}; available: "
            f"{', '.join(RESOLUTIONS_PX)}"
        )
    target_w, target_h = RESOLUTIONS_PX[resolution]
    h, w = frame.shape[:2]
    interp = cv2.INTER_AREA if target_w < w else cv2.INTER_LINEAR
    out = cv2.resize(frame, (target_w, target_h), interpolation=interp)
    if restore_size and (w, h) != (target_w, target_h):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def jpeg_recompress(frame: UInt8Array, quality: int) -> UInt8Array:
    """Round-trip through JPEG at a given quality factor.

    This is the DCT compression of Unit 3 applied as a channel impairment: the
    encoder quantises 8x8 DCT coefficients, and at low quality the blocking and
    ringing it introduces are exactly what a background-difference image
    amplifies. The study asks how far quality can fall before counting
    degrades, which is a bitrate question a CCTV operator actually faces.
    """
    if not 1 <= quality <= 100:
        raise ValueError(f"JPEG quality must be in 1..100, got {quality}")
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a JPEG buffer")
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR if frame.ndim == 3 else cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError("cv2.imdecode failed to read back the JPEG buffer")
    return decoded


def motion_blur(frame: UInt8Array, length_px: float, angle_deg: float = 90.0) -> UInt8Array:
    """Directional box blur, modelling a finite shutter at frame level.

    Note the distinction from the renderer's ``shutter_s``: that integrates each
    *vehicle* across its own motion, which is physically right and biases each
    silhouette individually. This is a whole-frame approximation used for the
    degradation sweep, where the point is a single controlled parameter rather
    than per-object fidelity. The report states which was used for which study.
    """
    if length_px <= 0:
        return frame.copy()
    n = max(3, int(round(length_px)) | 1)
    kernel = np.zeros((n, n), dtype=np.float32)
    kernel[n // 2, :] = 1.0
    m = cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, m, (n, n))
    total = float(kernel.sum())
    if total <= 0:
        return frame.copy()
    kernel /= total
    return cv2.filter2D(frame, -1, kernel, borderType=cv2.BORDER_REPLICATE)


def sensor_noise(frame: UInt8Array, sigma: float, seed: int = 0) -> UInt8Array:
    """Additive zero-mean Gaussian noise, seeded for reproducibility."""
    if sigma <= 0:
        return frame.copy()
    rng = np.random.default_rng(seed)
    noisy = frame.astype(np.float32) + rng.normal(0.0, sigma, size=frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def low_light(frame: UInt8Array, gamma: float = 1.0, exposure: float = 1.0) -> UInt8Array:
    """Exposure scaling followed by a power-law (gamma) transfer.

    The power-law transform ``s = c r^gamma`` is the Unit 2 point operation;
    here it is used to *create* an under-exposed frame, and the same operator
    with the reciprocal exponent is what S1 offers to correct one. Applying
    exposure before gamma matches the physical order - light is collected, then
    the sensor's transfer curve is applied.
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if exposure < 0:
        raise ValueError(f"exposure must be non-negative, got {exposure}")
    x = np.clip(frame.astype(np.float32) * exposure / 255.0, 0.0, 1.0)
    return np.clip(np.power(x, gamma) * 255.0, 0, 255).astype(np.uint8)


def quantise_grey_levels(frame: UInt8Array, levels: int) -> UInt8Array:
    """Reduce grey-level resolution to ``levels`` values per channel.

    The other half of Unit 1's resolution story: spatial resolution is not the
    only thing a cheap camera or an aggressive encoder throws away. Coarse
    quantisation shows up as false contouring on the road surface and as a
    difference image that jumps in steps.
    """
    if levels < 2:
        raise ValueError(f"need at least 2 grey levels, got {levels}")
    if levels >= 256:
        return frame.copy()
    step = 255.0 / (levels - 1)
    q = np.round(frame.astype(np.float32) / step) * step
    return np.clip(q, 0, 255).astype(np.uint8)


def degrade(frame: UInt8Array, config: DegradationConfig, frame_index: int = 0) -> UInt8Array:
    """Apply a full degradation configuration in physical order."""
    out = frame
    if config.motion_blur_px > 0:
        out = motion_blur(out, config.motion_blur_px, config.motion_blur_angle_deg)
    if config.resolution is not None:
        out = to_resolution(out, config.resolution, config.restore_size)
    elif config.scale != 1.0:
        out = resample(out, config.scale, config.restore_size)
    if config.exposure != 1.0 or config.gamma != 1.0:
        out = low_light(out, config.gamma, config.exposure)
    if config.noise_sigma > 0:
        out = sensor_noise(out, config.noise_sigma, seed=config.seed + frame_index)
    if config.grey_levels:
        out = quantise_grey_levels(out, config.grey_levels)
    if config.jpeg_quality:
        out = jpeg_recompress(out, config.jpeg_quality)
    return out
