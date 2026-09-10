"""Controlled degradation of frames, for the resolution and bitrate study."""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from kerbside.core.types import RESOLUTIONS_PX
from kerbside.data.degrade import (
    DegradationConfig,
    degrade,
    jpeg_recompress,
    low_light,
    motion_blur,
    quantise_grey_levels,
    resample,
    sensor_noise,
    to_resolution,
)
from kerbside.data.renderer import TrafficRenderer
from kerbside.data.scenarios import build_scene


@pytest.fixture(scope="module")
def frame() -> np.ndarray:
    scene = replace(build_scene("free_flow", seed=77, duration_s=0.2), supersample=1)
    img, _ = TrafficRenderer(scene).render_frame(2)
    return img


def test_resample_reduces_size_and_can_restore_it(frame: np.ndarray) -> None:
    small = resample(frame, 0.5)
    assert small.shape[:2] == (frame.shape[0] // 2, frame.shape[1] // 2)
    restored = resample(frame, 0.5, restore_size=True)
    assert restored.shape == frame.shape
    # The array is the same shape but the information is gone, which is the
    # whole point of the study.
    assert not np.array_equal(restored, frame)


def test_resample_rejects_an_out_of_range_scale(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="scale must be"):
        resample(frame, 1.5)
    with pytest.raises(ValueError, match="scale must be"):
        resample(frame, 0.0)


def test_named_resolutions_produce_exactly_those_sizes(frame: np.ndarray) -> None:
    for name, (w, h) in RESOLUTIONS_PX.items():
        assert to_resolution(frame, name).shape[:2] == (h, w)
    with pytest.raises(ValueError, match="unknown resolution"):
        to_resolution(frame, "240p")


def test_downsampling_destroys_high_frequency_detail_monotonically(
    frame: np.ndarray,
) -> None:
    """Lower resolution must measurably remove detail, or the resolution study
    would be comparing indistinguishable inputs.

    The sweep only runs at or below the source resolution. Asking for 1080p from
    a 720p render is an *up*-sample, which adds interpolation blur rather than
    revealing detail that was never captured - so including it would make the
    chain non-monotonic for a reason that has nothing to do with the study.
    """

    def detail(img: np.ndarray) -> float:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(grey, cv2.CV_32F).var())

    assert frame.shape[:2] == (720, 1280)
    sharpness = [detail(frame)] + [
        detail(to_resolution(frame, name, restore_size=True))
        for name in ("720p", "480p", "360p")
    ]
    assert sharpness[0] == pytest.approx(sharpness[1], rel=1e-6)
    assert all(a > b for a, b in zip(sharpness[1:-1], sharpness[2:], strict=True))


def test_jpeg_quality_trades_fidelity_monotonically(frame: np.ndarray) -> None:
    errors = []
    for quality in (95, 70, 40, 15):
        out = jpeg_recompress(frame, quality)
        assert out.shape == frame.shape
        errors.append(float(np.abs(out.astype(np.int16) - frame.astype(np.int16)).mean()))
    assert all(a < b for a, b in zip(errors[:-1], errors[1:], strict=True))


def test_jpeg_quality_is_range_checked(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="1..100"):
        jpeg_recompress(frame, 0)


def test_motion_blur_smears_along_the_requested_direction() -> None:
    """A single bright pixel blurred vertically must spread vertically and not
    horizontally - the elementary check that the kernel is oriented correctly."""
    img = np.zeros((41, 41, 3), dtype=np.uint8)
    img[20, 20] = 255
    vertical = motion_blur(img, 15, angle_deg=90.0)[:, :, 0]
    assert vertical[:, 20].sum() > 4 * vertical[20, :].sum() - vertical[20, 20] * 4
    assert int((vertical[:, 20] > 0).sum()) > int((vertical[20, :] > 0).sum())

    horizontal = motion_blur(img, 15, angle_deg=0.0)[:, :, 0]
    assert int((horizontal[20, :] > 0).sum()) > int((horizontal[:, 20] > 0).sum())


def test_motion_blur_of_zero_length_is_a_no_op(frame: np.ndarray) -> None:
    np.testing.assert_array_equal(motion_blur(frame, 0.0), frame)


def test_sensor_noise_is_seeded_and_reproducible(frame: np.ndarray) -> None:
    a = sensor_noise(frame, 5.0, seed=1)
    b = sensor_noise(frame, 5.0, seed=1)
    c = sensor_noise(frame, 5.0, seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert abs(float(a.astype(np.float32).mean() - frame.astype(np.float32).mean())) < 1.0


def test_low_light_darkens_and_gamma_is_invertible() -> None:
    """The power-law transform of Unit 2, used here to create an under-exposed
    frame; the same operator with the reciprocal exponent undoes it."""
    ramp = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
    dark = low_light(ramp, gamma=2.2)
    assert float(dark.mean()) < float(ramp.mean())
    restored = low_light(dark, gamma=1.0 / 2.2)
    # Quantisation loses the darkest codes, so compare where there is signal.
    mask = ramp > 40
    assert float(np.abs(restored[mask].astype(np.int16) - ramp[mask].astype(np.int16)).mean()) < 3.0


def test_low_light_rejects_a_non_positive_gamma(frame: np.ndarray) -> None:
    with pytest.raises(ValueError, match="gamma must be positive"):
        low_light(frame, gamma=0.0)


def test_grey_level_quantisation_reduces_the_number_of_distinct_values() -> None:
    ramp = np.tile(np.arange(256, dtype=np.uint8), (8, 1))
    for levels in (2, 4, 16, 64):
        out = quantise_grey_levels(ramp, levels)
        assert len(np.unique(out)) == levels
    with pytest.raises(ValueError, match="at least 2"):
        quantise_grey_levels(ramp, 1)


def test_full_degradation_applies_in_physical_order(frame: np.ndarray) -> None:
    config = DegradationConfig(
        resolution="360p",
        restore_size=True,
        motion_blur_px=5.0,
        noise_sigma=3.0,
        jpeg_quality=35,
        seed=9,
    )
    out = degrade(frame, config, frame_index=0)
    assert out.shape == frame.shape
    assert np.array_equal(out, degrade(frame, config, frame_index=0))
    assert not np.array_equal(out, degrade(frame, config, frame_index=1))


def test_a_clean_configuration_is_almost_a_no_op(frame: np.ndarray) -> None:
    np.testing.assert_array_equal(degrade(frame, DegradationConfig()), frame)
    assert DegradationConfig().label() == "clean"


def test_labels_describe_the_configuration() -> None:
    label = DegradationConfig(resolution="480p", jpeg_quality=30, noise_sigma=2.0).label()
    assert "480p" in label and "q30" in label and "n2" in label
