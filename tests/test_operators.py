"""Phase 2 gate: known-answer tests for the operator library.

Where an operator has a closed-form answer on a small input, that is what is
checked - a 3x3 mean on a hand-worked matrix, Otsu on a synthetic bimodal
histogram, a DFT round trip, a block match on a patch shifted by a known amount.
Where it does not, the test checks the defining property instead: that a
convolution matches its own definition, that an orthogonal transform inverts,
that a morphological opening removes exactly what is smaller than its element.

Several operators are also checked against OpenCV. That is not circular: those
are the operators written out here from their definitions, and agreeing with a
mature independent implementation is the honest basis for claiming they are
right.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from kerbside.core.registry import REGISTRY, OperatorRegistry, ParamSpec, register
from kerbside.core.types import FrameType
from kerbside.operators import (
    colour,
    edges,
    frequency,
    morphology,
    motion,
    point,
    segment,
    spatial,
    transforms,
)


@pytest.fixture(scope="module")
def road_frame() -> np.ndarray:
    """A real rendered frame: textured, with edges at several scales."""
    from dataclasses import replace

    from kerbside.data.renderer import TrafficRenderer
    from kerbside.data.scenarios import build_scene

    scene = replace(build_scene("free_flow", seed=5, duration_s=0.2), supersample=1)
    frame, _ = TrafficRenderer(scene).render_frame(0)
    return frame


@pytest.fixture(scope="module")
def road_grey(road_frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(road_frame, cv2.COLOR_BGR2GRAY)


# ==========================================================================
# Registry
# ==========================================================================


def test_every_operator_declares_a_unit_and_a_rationale() -> None:
    """The course requirement, enforced rather than trusted."""
    assert len(REGISTRY) >= 30
    for spec in REGISTRY:
        assert spec.units, f"{spec.operator_id} declares no syllabus unit"
        assert len(spec.rationale) > 40, f"{spec.operator_id} has a token rationale"
        assert spec.name and spec.category


def test_the_registry_covers_every_syllabus_unit() -> None:
    covered = set(REGISTRY.units())
    assert {2, 3, 4, 5, 6, 7, 8} <= covered, f"units missing: {{2..8}} - {covered}"


def test_parameters_are_validated_not_clamped() -> None:
    """An out-of-range parameter is a disagreement about what the system does,
    so it raises. Silently substituting a different value would produce an image
    that cannot be explained by the parameters on screen."""
    spec = REGISTRY.get("spatial.gaussian")
    assert spec.coerce_params({})["size"] == 5
    assert spec.coerce_params({"size": 7})["size"] == 7
    with pytest.raises(ValueError, match="above the maximum"):
        spec.coerce_params({"size": 999})
    with pytest.raises(ValueError, match="must be odd"):
        spec.coerce_params({"size": 4})
    with pytest.raises(ValueError, match="no parameter"):
        spec.coerce_params({"sigmaa": 1.0})


def test_enum_parameters_reject_unknown_choices() -> None:
    spec = REGISTRY.get("morphology.morph")
    with pytest.raises(ValueError, match="is not one of"):
        spec.coerce_params({"op": "dilaate"})


def test_registry_rejects_a_duplicate_id_and_an_empty_rationale() -> None:
    local = OperatorRegistry()

    @register("x.one", "One", units=(1,), rationale="a" * 50, registry=local)
    def _one() -> int:
        return 1

    with pytest.raises(ValueError, match="already registered"):
        register("x.one", "Again", units=(1,), rationale="b" * 50, registry=local)(_one)
    with pytest.raises(ValueError, match="no rationale"):
        register("x.two", "Two", units=(1,), rationale="  ", registry=local)(_one)
    with pytest.raises(ValueError, match="no syllabus unit"):
        register("x.three", "Three", units=(), rationale="c" * 50, registry=local)(_one)


def test_unknown_operator_lookup_names_the_alternatives() -> None:
    with pytest.raises(KeyError, match="unknown operator"):
        REGISTRY.get("point.nonexistent")


def test_odd_int_parameters_must_default_to_an_odd_value() -> None:
    with pytest.raises(ValueError, match="must default to an odd"):
        ParamSpec("size", "Size", "odd_int", 4)


# ==========================================================================
# Unit 2 - point processing
# ==========================================================================


def test_negative_inverts_the_scale() -> None:
    ramp = np.arange(256, dtype=np.uint8).reshape(16, 16)
    np.testing.assert_array_equal(point.negative(ramp), 255 - ramp)


def test_contrast_stretch_maps_percentiles_to_the_full_range() -> None:
    x = np.linspace(80, 160, 256).astype(np.uint8).reshape(16, 16)
    out = point.contrast_stretch(x, 0.0, 100.0)
    assert int(out.min()) == 0
    assert int(out.max()) == 255


def test_contrast_stretch_leaves_a_uniform_image_alone() -> None:
    """Nothing to stretch. Dividing by a zero range would make a flat grey frame
    turn white, which reads as a working operator rather than a degenerate one."""
    flat = np.full((8, 8), 120, np.uint8)
    np.testing.assert_array_equal(point.contrast_stretch(flat), flat)


def test_contrast_stretch_rejects_inverted_percentiles() -> None:
    with pytest.raises(ValueError, match="must be below"):
        point.contrast_stretch(np.zeros((4, 4), np.uint8), 90.0, 10.0)


def test_power_law_is_its_own_inverse_at_the_reciprocal_exponent() -> None:
    ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    restored = point.power_law(point.power_law(ramp, 2.2), 1 / 2.2)
    signal = ramp > 40  # the darkest codes are lost to quantisation
    assert float(np.abs(restored[signal].astype(int) - ramp[signal].astype(int)).mean()) < 3.0


def test_log_transform_expands_the_dark_end() -> None:
    """The property that makes a Fourier spectrum legible: small values are
    spread out and large ones compressed."""
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    out = point.log_transform(ramp).astype(float)
    low_gain = out[0, 20] - out[0, 10]
    high_gain = out[0, 250] - out[0, 240]
    assert low_gain > high_gain


def test_threshold_is_exclusive_above_the_level() -> None:
    x = np.array([[126, 127, 128, 129]], np.uint8)
    np.testing.assert_array_equal(point.threshold(x, 127), [[0, 0, 255, 255]])
    np.testing.assert_array_equal(point.threshold(x, 127, invert=True), [[255, 255, 0, 0]])


def test_bit_plane_extracts_the_right_bit() -> None:
    x = np.array([[0b10000000, 0b00000001, 0b10101010]], np.uint8)
    np.testing.assert_array_equal(point.bit_plane(x, 7), [[255, 0, 255]])
    np.testing.assert_array_equal(point.bit_plane(x, 0), [[0, 255, 0]])
    with pytest.raises(ValueError, match="0..7"):
        point.bit_plane(x, 8)


def test_histogram_equalisation_flattens_a_skewed_histogram(road_grey: np.ndarray) -> None:
    """The defining property: after equalisation the cumulative histogram should
    be much closer to a straight line than it was."""
    dark = point.power_law(road_grey, 2.5)
    before = point.cumulative_histogram(dark)
    after = point.cumulative_histogram(point.histogram_equalise(dark))
    ideal = np.linspace(0, 1, 256)
    assert float(np.abs(after - ideal).mean()) < float(np.abs(before - ideal).mean())


def test_histogram_equalisation_of_colour_preserves_hue(road_frame: np.ndarray) -> None:
    """Equalising R, G and B independently shifts the colour balance and would
    corrupt exactly the information S5 and S9 depend on, so the implementation
    equalises luminance only. This is that promise, checked."""
    out = point.histogram_equalise(road_frame)
    before = colour.bgr_to_hsv(road_frame)[..., 0]
    after = colour.bgr_to_hsv(out)[..., 0]
    assert float(np.median(colour.hue_difference(before, after))) < 3.0


def test_histogram_counts_every_pixel(road_grey: np.ndarray) -> None:
    assert int(point.histogram(road_grey).sum()) == road_grey.size


# ==========================================================================
# Unit 2 - neighbourhood processing
# ==========================================================================


def test_three_by_three_mean_on_a_hand_worked_matrix() -> None:
    """The known-answer test the brief asks for, worked by hand.

    Reflect-101 padding of [[1,2,3],[4,5,6],[7,8,9]] gives the 5x5 array

        [[5,4,5,6,5],
         [2,1,2,3,2],
         [5,4,5,6,5],
         [8,7,8,9,8],
         [5,4,5,6,5]]

    so the top-left output sees [[5,4,5],[2,1,2],[5,4,5]], which sums to 33; the
    centre sees all nine originals and averages to 5; and the bottom-right sees
    [[5,6,5],[8,9,8],[5,6,5]], which sums to 57.
    """
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    out = spatial.correlate2d(a, spatial.box_kernel(3))
    assert out[0, 0] == pytest.approx(33 / 9)
    assert out[1, 1] == pytest.approx(5.0)
    assert out[2, 2] == pytest.approx(57 / 9)


def test_convolution_and_correlation_differ_for_an_asymmetric_kernel() -> None:
    """The distinction vanishes for symmetric kernels, which is why it is so
    often skipped and occasionally the reason a gradient comes out negated."""
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
    k = np.array([[0, 0, 0], [1, 0, -1], [0, 0, 0]], dtype=float)
    np.testing.assert_allclose(
        spatial.correlate2d(a, k), -spatial.convolve2d(a, k), atol=1e-12
    )


def test_hand_written_convolution_matches_opencv(road_grey: np.ndarray) -> None:
    """The from-scratch implementation against a mature one, in float so the
    comparison is not dominated by 8-bit rounding."""
    x = road_grey.astype(np.float32)
    for kernel, reference in (
        (spatial.box_kernel(5), cv2.blur(x, (5, 5), borderType=cv2.BORDER_REFLECT_101)),
        (
            spatial.gaussian_kernel(5, 1.2),
            cv2.GaussianBlur(x, (5, 5), 1.2, borderType=cv2.BORDER_REFLECT_101),
        ),
        (
            spatial.LAPLACIAN_8,
            cv2.filter2D(x, cv2.CV_32F, spatial.LAPLACIAN_8, borderType=cv2.BORDER_REFLECT_101),
        ),
    ):
        ours = spatial.correlate2d(x, kernel)
        assert float(np.abs(ours - np.asarray(reference, np.float64)).max()) < 1e-3


def test_gaussian_and_box_kernels_are_normalised() -> None:
    for size in (3, 5, 9):
        assert spatial.box_kernel(size).sum() == pytest.approx(1.0)
        assert spatial.gaussian_kernel(size, 1.5).sum() == pytest.approx(1.0)


def test_even_kernels_are_rejected() -> None:
    with pytest.raises(ValueError, match="odd"):
        spatial.box_kernel(4)
    with pytest.raises(ValueError, match="odd"):
        spatial.median_filter(np.zeros((8, 8), np.uint8), 4)


def test_median_filter_removes_salt_and_pepper_that_a_mean_only_smears() -> None:
    """Why S4 uses a median before morphology and not a mean."""
    rng = np.random.default_rng(0)
    clean = np.full((64, 64), 120, np.uint8)
    noisy = clean.copy()
    coords = rng.integers(0, 64, size=(2, 80))
    noisy[coords[0], coords[1]] = 255

    med = spatial.median_filter(noisy, 3).astype(float)
    box = spatial.box_filter(noisy, 3).astype(float)
    assert float(np.abs(med - clean).mean()) < float(np.abs(box - clean).mean()) / 4


def test_border_mode_constant_would_invent_an_edge(road_grey: np.ndarray) -> None:
    """Why reflect is the default: a constant border creates a dark frame that a
    difference-based pipeline detects as foreground along every image edge."""
    reflected = spatial.correlate2d(road_grey, spatial.box_kernel(9), mode="reflect")
    constant = spatial.correlate2d(road_grey, spatial.box_kernel(9), mode="constant")
    # Along the top edge the constant border invents a dark frame.
    assert float(np.abs(reflected[0] - constant[0]).max()) > 20.0
    # Well inside the image, where no part of the kernel reaches a border, the
    # two must agree exactly: the border mode is a boundary decision only.
    interior = (slice(100, 200), slice(100, 200))
    assert float(np.abs(reflected[interior] - constant[interior]).max()) < 1e-9


def test_unsharp_mask_increases_local_contrast(road_grey: np.ndarray) -> None:
    blurred = spatial.gaussian_filter(road_grey, 5, 1.5)
    sharpened = spatial.unsharp_mask(blurred, 5, 1.5, amount=1.5)
    assert float(cv2.Laplacian(sharpened, cv2.CV_32F).var()) > float(
        cv2.Laplacian(blurred, cv2.CV_32F).var()
    )


def test_high_boost_requires_a_boost_of_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        spatial.high_boost(np.zeros((8, 8), np.uint8), boost=0.5)


# ==========================================================================
# Unit 3 - frequency domain
# ==========================================================================


def test_dft_round_trip_is_exact(road_grey: np.ndarray) -> None:
    """The known-answer test the brief asks for."""
    restored = frequency.idft2(frequency.dft2(road_grey))
    assert float(np.abs(restored - road_grey.astype(float)).max()) < 1e-9


def test_a_pure_sinusoid_produces_a_single_pair_of_spectral_peaks() -> None:
    """The defining property of the transform, on an input whose spectrum is
    known exactly: one frequency in, one conjugate pair out."""
    n = 128
    cycles = 8
    x = np.arange(n)
    image = 128 + 100 * np.cos(2 * np.pi * cycles * x[None, :] / n) * np.ones((n, 1))
    mag = np.abs(frequency.dft2(image))
    mag[n // 2, n // 2] = 0  # ignore the DC term
    peaks = np.argwhere(mag > 0.5 * mag.max())
    assert len(peaks) == 2
    for row, col in peaks:
        assert row == n // 2
        assert abs(abs(col - n // 2) - cycles) <= 1


def test_lowpass_and_highpass_partition_the_signal(road_grey: np.ndarray) -> None:
    low = frequency.filter_image(road_grey, cutoff=30, band="lowpass").astype(float)
    high = frequency.filter_image(road_grey, cutoff=30, band="highpass").astype(float) - 128.0
    assert float(np.abs(low + high - road_grey.astype(float)).mean()) < 3.0


def test_the_ideal_filter_rings_and_the_gaussian_does_not() -> None:
    """The reason the Gaussian is the default. A hard spectral cut transforms to
    a sinc in space, so a sharp edge acquires concentric ripples; a Gaussian
    transfer function is non-negative everywhere and cannot ring."""
    step = np.zeros((128, 128), np.uint8)
    step[:, 64:] = 255

    def ripple(name: str) -> float:
        """Oscillation amplitude in the flat region well away from the edge.

        Measured against the local trend rather than as a plain range, so the
        gentle roll-off any lowpass leaves behind is not counted as ringing.
        """
        out = frequency.filter_image(step, cutoff=12, shape_kind=name, band="lowpass")
        profile = out[64].astype(float)[:50]
        smooth = np.convolve(profile, np.ones(9) / 9, mode="same")
        return float(np.abs(profile[6:-6] - smooth[6:-6]).max())

    assert ripple("ideal") > 3.0 * max(ripple("gaussian"), 0.3)


def test_frequency_filter_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="cutoff must be positive"):
        frequency.frequency_filter((16, 16), 0.0)
    with pytest.raises(ValueError, match="unknown band"):
        frequency.frequency_filter((16, 16), 4.0, band="sideways")  # type: ignore[arg-type]


def test_phase_correlation_recovers_an_integer_shift_exactly() -> None:
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.random((128, 160)) * 255, (0, 0), 2.0)
    for dx, dy in ((3, -2), (0, 0), (7, 5)):
        shifted = _fourier_shift(base, dx, dy)
        result = frequency.phase_correlation(base, shifted, window=False)
        assert result.dx_px == pytest.approx(dx, abs=0.05)
        assert result.dy_px == pytest.approx(dy, abs=0.05)


def test_phase_correlation_recovers_sub_pixel_shifts() -> None:
    """Sub-pixel refinement is not optional: for a fractional shift the inverse
    transform of a pure phase ramp is a periodic sinc whose side lobes alternate
    in sign, so a three-point parabolic fit lands about a pixel out. The
    upsampled-DFT refinement gets it to a fraction of a pixel."""
    rng = np.random.default_rng(1)
    base = cv2.GaussianBlur(rng.random((128, 160)) * 255, (0, 0), 2.0)
    worst = 0.0
    for dx, dy in ((2.5, 1.25), (-1.75, 3.5), (0.25, -0.75), (0.5, 0.5)):
        result = frequency.phase_correlation(base, _fourier_shift(base, dx, dy), window=False)
        worst = max(worst, float(np.hypot(result.dx_px - dx, result.dy_px - dy)))
    assert worst < 0.1, f"worst sub-pixel error was {worst:.3f} px"


def test_phase_correlation_is_unmoved_by_a_brightness_change() -> None:
    """Discarding the magnitude is what buys this, and it is the property that
    makes the stabiliser survive a cloud passing over the scene."""
    rng = np.random.default_rng(2)
    base = cv2.GaussianBlur(rng.random((96, 96)) * 200 + 20, (0, 0), 2.0)
    shifted = np.clip(_fourier_shift(base, 4, -3) * 1.4 + 25.0, 0, 255)
    result = frequency.phase_correlation(base, shifted, window=False)
    assert result.dx_px == pytest.approx(4, abs=0.2)
    assert result.dy_px == pytest.approx(-3, abs=0.2)


def test_phase_correlation_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="identically shaped"):
        frequency.phase_correlation(np.zeros((8, 8)), np.zeros((8, 9)))


def test_periodic_interference_is_found_and_notched_out() -> None:
    """Rain and electrical interference localise in the spectrum; image content
    does not. The notch removes one and leaves the other."""
    rng = np.random.default_rng(4)
    scene = cv2.GaussianBlur(rng.random((128, 128)) * 120 + 60, (0, 0), 3.0)
    x = np.arange(128)
    interference = 34 * np.sin(2 * np.pi * 18 * x[None, :] / 128) * np.ones((128, 1))
    corrupted = np.clip(scene + interference, 0, 255).astype(np.uint8)

    peaks = frequency.detect_spectral_peaks(corrupted)
    assert peaks, "the interference peak was not detected"

    cleaned = frequency.suppress_periodic_noise(corrupted).astype(float)
    before = float(np.abs(corrupted.astype(float) - scene).mean())
    after = float(np.abs(cleaned - scene).mean())
    assert after < before / 2.0


def test_a_clean_frame_is_left_alone_by_the_notch_filter(road_grey: np.ndarray) -> None:
    """Inventing notches on a frame with no periodic content would blur it for
    no reason, so no prominent peak means no change."""
    patch = road_grey[300:556, 400:656]
    assert frequency.detect_spectral_peaks(patch) == []
    out = frequency.suppress_periodic_noise(patch)
    np.testing.assert_array_equal(out, patch)


def _fourier_shift(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Exact translation by a phase ramp, with no interpolation error."""
    rows, cols = image.shape
    v = np.fft.fftfreq(rows)[:, None]
    u = np.fft.fftfreq(cols)[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(image) * np.exp(-2j * np.pi * (u * dx + v * dy))))


# ==========================================================================
# Unit 3 - transforms
# ==========================================================================


def test_hadamard_matrix_is_orthonormal_with_plus_minus_one_entries() -> None:
    """The property that made it worth using: no multiplications."""
    for order in (2, 4, 8, 16):
        h = transforms.hadamard_matrix(order)
        np.testing.assert_allclose(h @ h.T, np.eye(order), atol=1e-12)
        np.testing.assert_allclose(np.abs(h) * np.sqrt(order), 1.0, atol=1e-12)


def test_sequency_ordering_sorts_rows_by_sign_changes() -> None:
    """Without it, "keep the first k coefficients" keeps an arbitrary set rather
    than the smooth ones."""
    h = transforms.hadamard_matrix(8, sequency=True)
    changes = np.count_nonzero(np.diff(np.sign(h), axis=1), axis=1)
    np.testing.assert_array_equal(changes, np.arange(8))


def test_hadamard_order_must_be_a_power_of_two() -> None:
    with pytest.raises(ValueError, match="power of two"):
        transforms.hadamard_matrix(6)


def test_all_three_transforms_round_trip(road_grey: np.ndarray) -> None:
    block = road_grey[200:208, 400:408].astype(float)
    for name in ("dct", "dft", "hadamard"):
        restored = transforms.inverse_transform2(transforms.transform2(block, name), name)
        assert float(np.abs(restored - block).max()) < 1e-9


def test_the_dct_compacts_energy_better_than_the_alternatives(road_grey: np.ndarray) -> None:
    """The reason JPEG uses it. Measured on real content rather than asserted."""
    patch = road_grey[200:456, 400:656]
    curves = {
        name: transforms.energy_compaction(patch, 8, name)
        for name in ("dct", "dft", "hadamard")
    }
    at_four = {name: float(curve[3]) for name, curve in curves.items()}
    assert at_four["dct"] > at_four["hadamard"] > at_four["dft"]


def test_compression_fidelity_rises_with_the_coefficient_budget(road_grey: np.ndarray) -> None:
    results = transforms.compression_study(road_grey[200:328, 400:528], keeps=(1, 4, 16, 32))
    psnrs = [r.psnr_db for r in results]
    assert all(a < b for a, b in zip(psnrs[:-1], psnrs[1:], strict=True))
    assert results[0].compression_ratio > results[-1].compression_ratio


def test_zonal_coding_is_no_better_than_threshold_coding(road_grey: np.ndarray) -> None:
    """Threshold coding picks the best coefficients per block with hindsight, so
    it is an upper bound; the gap is the price of not transmitting which ones
    were kept."""
    patch = road_grey[200:328, 400:528]
    threshold = transforms.compression_study(patch, keeps=(8,), zonal=False)[0]
    zonal = transforms.compression_study(patch, keeps=(8,), zonal=True)[0]
    assert threshold.psnr_db >= zonal.psnr_db


def test_psnr_is_infinite_for_identical_inputs(road_grey: np.ndarray) -> None:
    assert transforms.psnr(road_grey, road_grey) == float("inf")
    with pytest.raises(ValueError, match="shape mismatch"):
        transforms.psnr(road_grey, road_grey[:10, :10])


# ==========================================================================
# Unit 4 - morphology
# ==========================================================================


def test_erosion_and_dilation_of_a_square_are_exact() -> None:
    """A 5x5 square eroded by a 3x3 square is a 3x3 square; dilated it is 7x7."""
    m = np.zeros((11, 11), np.uint8)
    m[3:8, 3:8] = 255
    se = morphology.structuring_element(3, 3, "rect")
    assert int((morphology.erode(m, se) > 0).sum()) == 9
    assert int((morphology.dilate(m, se) > 0).sum()) == 49


def test_opening_separates_two_blobs_joined_by_a_thin_bridge() -> None:
    """Trap 2's mechanism in miniature, and the reason S4 opens before it closes."""
    m = np.zeros((21, 41), bool)
    m[6:15, 4:18] = True
    m[6:15, 24:38] = True
    m[10:11, 18:24] = True
    assert segment.label_components(m, 8).count == 1
    element = morphology.structuring_element(3, 3, "rect")
    opened = morphology.opening(m.astype(np.uint8) * 255, element)
    assert segment.label_components(opened > 0, 8).count == 2


def test_closing_fills_a_gap_narrower_than_the_element() -> None:
    m = np.zeros((21, 21), bool)
    m[5:16, 5:16] = True
    m[5:16, 10] = False  # a one-pixel slit
    element = morphology.structuring_element(3, 3, "rect")
    closed = morphology.closing(m.astype(np.uint8) * 255, element)
    assert segment.label_components(closed > 0, 4).count == 1


def test_boundary_of_a_square_is_its_perimeter() -> None:
    m = np.zeros((15, 15), bool)
    m[4:11, 4:11] = True
    assert int(morphology.boundary(m).sum()) == 24  # 7x7 minus the 5x5 interior


def test_hole_filling_recovers_an_enclosed_region() -> None:
    m = np.zeros((21, 21), bool)
    m[4:17, 4:17] = True
    m[8:13, 8:13] = False
    assert int(morphology.fill_holes(m).sum()) == 13 * 13


def test_hole_filling_does_not_fill_a_bay_open_to_the_border() -> None:
    """A notch connected to the outside is not a hole, and treating it as one
    would inflate every blob touching the frame edge."""
    m = np.zeros((21, 21), bool)
    m[4:17, 4:17] = True
    m[4:9, 8:13] = False  # open to the top of the square, not enclosed
    filled = morphology.fill_holes(m)
    assert int(filled.sum()) == int(m.sum())


def test_hit_or_miss_finds_an_isolated_pixel() -> None:
    m = np.zeros((13, 13), bool)
    m[9, 9] = True
    m[2:5, 2:5] = True
    fg = np.zeros((3, 3), np.uint8)
    fg[1, 1] = 1
    bg = np.ones((3, 3), np.uint8)
    bg[1, 1] = 0
    found = np.argwhere(morphology.hit_or_miss(m, fg, bg))
    np.testing.assert_array_equal(found, [[9, 9]])


def test_hit_or_miss_rejects_an_overlapping_element() -> None:
    with pytest.raises(ValueError, match="cannot be required to be both"):
        morphology.hit_or_miss(
            np.zeros((5, 5), bool), np.ones((3, 3), np.uint8), np.ones((3, 3), np.uint8)
        )


def test_thinning_reduces_a_bar_to_a_single_line() -> None:
    m = np.zeros((21, 41), bool)
    m[8:13, 5:36] = True
    thinned = morphology.thin(m)
    rows_used = np.nonzero(thinned.any(axis=1))[0]
    assert len(rows_used) == 1
    assert int(thinned.sum()) > 20


def test_thinning_preserves_connectivity() -> None:
    """The condition that makes Zhang-Suen topology preserving: a skeleton that
    breaks a shape into pieces has thrown away the thing it was meant to keep."""
    m = np.zeros((41, 41), bool)
    m[18:23, 5:36] = True
    m[5:36, 18:23] = True  # a cross
    assert segment.label_components(morphology.thin(m), 8).count == 1


def test_reconstruction_keeps_whole_components_containing_a_marker() -> None:
    """Strictly better than an opening for deleting objects: an opening erodes
    what it keeps, and reconstruction returns it at its original shape."""
    mask = np.zeros((21, 41), bool)
    mask[4:12, 3:15] = True
    mask[4:12, 25:37] = True
    marker = np.zeros_like(mask)
    marker[6, 6] = True
    result = morphology.reconstruct_by_dilation(marker, mask)
    assert int(result.sum()) == 8 * 12
    assert not result[4:12, 25:37].any()


def test_structuring_elements_must_be_odd() -> None:
    with pytest.raises(ValueError, match="odd"):
        morphology.structuring_element(4, 3)


def test_banded_morphology_scales_the_element_with_perspective() -> None:
    """Trap 4, checked directly: a fixed half-metre element must cover far more
    pixels near the camera than near the horizon."""
    from kerbside.data.scenarios import DEFAULT_CAMERA
    from kerbside.geometry.homography import homography_from_camera
    from kerbside.geometry.perspective import PerspectiveModel

    model = PerspectiveModel(
        homography_from_camera(DEFAULT_CAMERA), DEFAULT_CAMERA.width_px, DEFAULT_CAMERA.height_px
    )
    mask = np.zeros((DEFAULT_CAMERA.height_px, DEFAULT_CAMERA.width_px), np.uint8)
    mask[700, 600:680] = 255  # a near-field bar
    mask[300, 600:680] = 255  # a far-field bar of the same pixel length

    dilated = morphology.banded_morphology(mask, model, 0.5, op="dilate", bands=12)
    near_growth = int((dilated[690:711, :] > 0).sum())
    far_growth = int((dilated[290:311, :] > 0).sum())
    assert near_growth > 2 * far_growth


def test_banded_morphology_leaves_no_seam_at_a_band_boundary() -> None:
    """Bands overlap by the element's half-height and recombine on their
    interiors. A visible seam every fiftieth row would be read as a horizontal
    edge by S6 and would fragment any blob straddling it."""
    from kerbside.data.scenarios import DEFAULT_CAMERA
    from kerbside.geometry.homography import homography_from_camera
    from kerbside.geometry.perspective import PerspectiveModel

    model = PerspectiveModel(
        homography_from_camera(DEFAULT_CAMERA), DEFAULT_CAMERA.width_px, DEFAULT_CAMERA.height_px
    )
    mask = np.zeros((DEFAULT_CAMERA.height_px, DEFAULT_CAMERA.width_px), np.uint8)
    mask[200:700, 500:560] = 255  # one tall bar crossing every band

    closed = morphology.banded_morphology(mask, model, 0.4, op="close", bands=12)
    column = closed[210:690, 530] > 0
    assert column.all(), "a band boundary punched a hole through a continuous bar"


# ==========================================================================
# Unit 5 - edges
# ==========================================================================


def test_gradient_operators_respond_to_the_right_axis() -> None:
    vertical = np.zeros((20, 20), float)
    vertical[:, 10:] = 255
    horizontal = np.zeros((20, 20), float)
    horizontal[10:, :] = 255
    for name in ("prewitt", "sobel"):
        gv = edges.gradient(vertical, name)
        gh = edges.gradient(horizontal, name)
        assert np.abs(gv.gx).max() > 100
        assert np.abs(gv.gy).max() < 1e-9
        assert np.abs(gh.gy).max() > 100
        assert np.abs(gh.gx).max() < 1e-9


def test_roberts_measures_diagonal_differences_not_axial_ones() -> None:
    """Roberts is excluded from the axis test above because its kernels are
    diagonal by construction: its two components are the derivatives along the
    two diagonals, not along x and y. A vertical edge therefore excites both,
    equally and oppositely, and only the magnitude is axis-meaningful. That is a
    property of the operator rather than a defect, and it is part of why it
    reads differently from Prewitt and Sobel in the five-operator comparison.
    """
    vertical = np.zeros((20, 20), float)
    vertical[:, 10:] = 255
    g = edges.gradient(vertical, "roberts")
    assert np.abs(g.gx).max() > 100
    assert np.abs(g.gy).max() > 100
    assert float(np.abs(g.gx + g.gy).max()) < 1e-9
    assert np.abs(g.magnitude).max() > 100


def test_sobel_matches_opencv(road_grey: np.ndarray) -> None:
    g = edges.gradient(road_grey, "sobel")
    reference = cv2.Sobel(
        road_grey.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101
    )
    assert float(np.abs(g.gx - reference).max()) < 1e-6


def test_city_block_magnitude_overestimates_diagonals() -> None:
    """The Unit 1 distance measures, exercised: |gx| + |gy| exceeds the
    Euclidean magnitude by up to 41% on a diagonal edge and matches it exactly on
    an axis-aligned one."""
    diagonal = np.tril(np.ones((40, 40))) * 255
    g = edges.gradient(diagonal, "sobel")
    strong = g.magnitude > 0.5 * g.magnitude.max()
    ratio = float((g.magnitude_l1[strong] / g.magnitude[strong]).mean())
    assert 1.2 < ratio <= np.sqrt(2) + 1e-9

    axis = np.zeros((40, 40))
    axis[:, 20:] = 255
    ga = edges.gradient(axis, "sobel")
    strong_a = ga.magnitude > 0.5 * ga.magnitude.max()
    assert float((ga.magnitude_l1[strong_a] / ga.magnitude[strong_a]).mean()) == pytest.approx(1.0)


def test_log_kernel_sums_to_zero() -> None:
    """Otherwise a constant offset leaks through and shifts every zero crossing."""
    for size, sigma in ((9, 1.4), (15, 2.5)):
        assert abs(float(edges.laplacian_of_gaussian_kernel(size, sigma).sum())) < 1e-12


def test_log_kernel_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError, match="odd"):
        edges.laplacian_of_gaussian_kernel(8, 1.4)
    with pytest.raises(ValueError, match="sigma must be positive"):
        edges.laplacian_of_gaussian_kernel(9, 0.0)


def test_canny_produces_edges_one_pixel_wide() -> None:
    """What non-maximum suppression is for."""
    step = np.zeros((60, 60), np.uint8)
    step[:, 30:] = 255
    result = edges.canny(step, 40, 110)
    for row in range(10, 50):
        runs = np.diff(np.nonzero(np.diff(np.concatenate([[0], result[row].astype(int), [0]]))))
        assert all(r == 1 for r in runs[::2]), "an edge wider than one pixel survived"


def test_canny_agrees_with_opencv(road_grey: np.ndarray) -> None:
    """The from-scratch four stages against a mature implementation. Exact
    agreement is not expected - OpenCV uses a fixed-point Sobel internally and a
    different tie-break - but the edge maps must be substantially the same."""
    blurred = cv2.GaussianBlur(road_grey, (5, 5), 1.4)
    mine = edges.canny(road_grey, 40, 110, l2_gradient=True)
    theirs = cv2.Canny(blurred, 40, 110, L2gradient=True) > 0
    iou = float((mine & theirs).sum() / max(1, (mine | theirs).sum()))
    assert iou > 0.7, f"IoU with OpenCV was {iou:.3f}"


def test_canny_locates_a_diagonal_edge_correctly() -> None:
    """The direct check on non-maximum suppression along the diagonal sectors:
    comparing against the neighbours *across* the gradient rather than along it
    deletes diagonal edges, and does so quietly."""
    diagonal = np.zeros((60, 60), np.uint8)
    for row in range(60):
        diagonal[row, :row] = 255
    result = edges.canny(diagonal, 40, 110)
    ys, xs = np.nonzero(result)
    assert len(ys) > 40
    assert abs(float(np.mean(xs - ys))) < 2.0
    assert float(np.std(xs - ys)) < 1.5


def test_canny_rejects_inverted_thresholds() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        edges.canny(np.zeros((8, 8), np.uint8), 200, 100)


def test_hysteresis_keeps_a_weak_edge_only_when_it_touches_a_strong_one() -> None:
    """The property that keeps a fading lane marking one continuous line."""
    image = np.zeros((40, 60), np.uint8)
    image[10:14, 5:30] = 255  # strong bar
    image[10:14, 30:50] = 45  # faint continuation of the same bar
    image[30:34, 5:25] = 45  # an isolated faint bar

    # Derive the thresholds from the gradients this image actually produces, so
    # the test is about hysteresis rather than about whether two hard-coded
    # numbers happened to straddle the right values.
    grad = edges.gradient(spatial.gaussian_filter(image, 3, 0.8), "sobel").magnitude
    faint = float(grad[28:36, 5:25].max())
    strong = float(grad[8:16, 5:30].max())
    assert faint < strong, "the faint bar is not actually fainter"
    low, high = faint * 0.5, (faint + strong) / 2.0

    result = edges.canny(image, low=low, high=high, blur_size=3, blur_sigma=0.8)
    assert result[8:16, 28:48].any(), "the connected weak edge was dropped"
    assert not result[28:36, 5:25].any(), "an isolated weak edge survived"


def test_hough_finds_the_lines_of_a_synthetic_grid() -> None:
    """A known answer: two lines at known angles and offsets."""
    image = np.zeros((200, 200), bool)
    image[50, :] = True  # horizontal
    image[:, 120] = True  # vertical
    lines = edges.hough_lines(image, threshold=100, max_lines=4)
    assert len(lines) >= 2
    found = {(round(line.theta_deg), round(line.rho_px)) for line in lines}
    assert (90, 50) in found
    assert (0, 120) in found


def test_hough_survives_a_dashed_line() -> None:
    """The reason to use it for lane markings at all: each pixel votes
    independently, so a broken line still produces one peak."""
    solid = np.zeros((200, 200), bool)
    solid[:, 100] = True
    dashed = np.zeros((200, 200), bool)
    for start in range(0, 200, 20):
        dashed[start : start + 10, 100] = True

    solid_lines = edges.hough_lines(solid, threshold=60, max_lines=2)
    dashed_lines = edges.hough_lines(dashed, threshold=60, max_lines=2)
    assert solid_lines and dashed_lines
    assert dashed_lines[0].rho_px == pytest.approx(solid_lines[0].rho_px, abs=1.0)
    assert dashed_lines[0].theta_deg == pytest.approx(solid_lines[0].theta_deg, abs=1.0)


def test_hough_does_not_report_the_same_line_repeatedly() -> None:
    """A strong line lights up a neighbourhood of the accumulator, so taking the
    top n cells outright returns one line several times."""
    image = np.zeros((200, 200), bool)
    image[:, 100] = True
    lines = edges.hough_lines(image, threshold=50, max_lines=6)
    rhos = sorted(line.rho_px for line in lines)
    assert all(abs(b - a) > 5.0 for a, b in zip(rhos[:-1], rhos[1:], strict=True))


def test_hough_on_an_empty_edge_map_returns_nothing() -> None:
    assert edges.hough_lines(np.zeros((50, 50), bool)) == []


def test_edge_operator_comparison_covers_all_five(road_grey: np.ndarray) -> None:
    summary = edges.compare_edge_operators(road_grey[200:456, 400:656])
    assert set(summary) == {"roberts", "prewitt", "sobel", "laplacian", "log", "canny"}
    for name, entry in summary.items():
        assert 0.0 <= entry["edge_density"] <= 1.0, name


# ==========================================================================
# Unit 6 - segmentation
# ==========================================================================


def test_otsu_on_a_synthetic_bimodal_histogram() -> None:
    """The known-answer test the brief asks for, plus the property that makes
    the answer trustworthy."""
    rng = np.random.default_rng(0)
    values = np.clip(
        np.concatenate([rng.normal(60, 8, 5000), rng.normal(190, 8, 5000)]), 0, 255
    ).astype(np.uint8).reshape(100, 100)
    result = segment.otsu_threshold(values)
    assert 100 < result.threshold < 150
    assert result.separability > 0.95
    assert result.is_bimodal


def test_otsu_argmax_reproduces_opencv_exactly() -> None:
    """Turning the plateau handling off must give the textbook answer, so the
    difference between the two is a documented choice rather than a discrepancy."""
    rng = np.random.default_rng(0)
    values = np.clip(
        np.concatenate([rng.normal(60, 8, 5000), rng.normal(190, 8, 5000)]), 0, 255
    ).astype(np.uint8).reshape(100, 100)
    mine = segment.otsu_threshold(values, plateau_midpoint=False).threshold
    theirs, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    assert mine == pytest.approx(theirs)


def test_otsu_centres_an_empty_histogram_gap() -> None:
    """When the modes are separated by a gap, every level inside it separates
    them identically well, so the criterion is flat and a plain argmax returns
    the lowest level of the plateau - hard against the upper tail of the dark
    mode. On a difference image that admits far more noise than the middle of
    the gap would."""
    rng = np.random.default_rng(0)
    values = np.clip(
        np.concatenate([rng.normal(60, 8, 5000), rng.normal(190, 8, 5000)]), 0, 255
    ).astype(np.uint8).reshape(100, 100)
    centred = segment.otsu_threshold(values, plateau_midpoint=True)
    lowest = segment.otsu_threshold(values, plateau_midpoint=False)
    assert centred.plateau_width > 30
    assert centred.threshold > lowest.threshold + 20
    assert centred.between_class_variance == pytest.approx(lowest.between_class_variance)


def test_a_unimodal_histogram_is_not_reported_as_separable() -> None:
    """Otsu always returns a threshold. Splitting a single Gaussian at its own
    median already scores 0.64, which is why the bimodality cut sits well above
    that rather than at some small number."""
    rng = np.random.default_rng(1)
    values = np.clip(rng.normal(128, 10, 10000), 0, 255).astype(np.uint8).reshape(100, 100)
    result = segment.otsu_threshold(values)
    assert 0.55 < result.separability < 0.75
    assert not result.is_bimodal


def test_otsu_can_be_restricted_to_a_region() -> None:
    """Computing the threshold over the whole frame includes sky and verge,
    which are never foreground and drag the level towards suiting them."""
    image = np.zeros((100, 100), np.uint8)
    image[:50] = 10  # a large dark region that is not of interest
    image[50:, :50] = 90
    image[50:, 50:] = 200
    region = np.zeros((100, 100), bool)
    region[50:] = True
    assert 90 < segment.otsu_threshold(image, mask=region).threshold < 200


def test_otsu_on_an_empty_region_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty region"):
        segment.otsu_threshold(np.zeros((8, 8), np.uint8), mask=np.zeros((8, 8), bool))


def test_four_connectivity_splits_a_diagonal_chain_and_eight_does_not() -> None:
    m = np.zeros((7, 7), bool)
    for i in range(5):
        m[i + 1, i + 1] = True
    assert segment.label_components(m, 4).count == 5
    assert segment.label_components(m, 8).count == 1
    assert segment.label_components(m, "m").count == 1


def test_m_connectivity_gives_the_same_partition_as_eight_always() -> None:
    """Provable and easy to assume otherwise, which would make the connectivity
    ablation misleading. If a shared 4-neighbour of a diagonal pair is
    foreground then a 4-path already joins them; if neither is, m admits the
    diagonal directly. Either way they share a component."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        mask = rng.random((48, 48)) < rng.uniform(0.2, 0.5)
        eight = segment.label_components(mask, 8)
        m_conn = segment.label_components(mask, "m")
        assert eight.count == m_conn.count
        # Same partition, up to how the labels happen to be numbered.
        pairing = {}
        for a, b in zip(eight.labels[mask].ravel(), m_conn.labels[mask].ravel(), strict=True):
            assert pairing.setdefault(int(a), int(b)) == int(b)


def test_connectivity_disagreement_is_between_four_and_the_others() -> None:
    mask = np.zeros((9, 9), bool)
    mask[1:4, 1:4] = True
    mask[4:7, 4:7] = True
    report = segment.connectivity_disagreement(mask)
    assert report["counts"] == {"4": 2, "8": 1, "m": 1}
    assert report["disagreement_px"] > 0


def test_component_statistics_are_correct() -> None:
    mask = np.zeros((20, 30), bool)
    mask[4:10, 6:16] = True
    result = segment.label_components(mask, 8)
    assert result.count == 1
    stats = result.stats[0]
    assert (stats.x_px, stats.y_px, stats.w_px, stats.h_px) == (6, 4, 10, 6)
    assert stats.area_px == 60
    assert stats.fill_ratio == pytest.approx(1.0)
    assert stats.centroid_px == pytest.approx((10.5, 6.5))


def test_invalid_connectivity_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be 4, 8 or"):
        segment.label_components(np.zeros((4, 4), bool), 6)  # type: ignore[arg-type]


def test_region_growing_follows_a_gradient_a_global_threshold_would_cut() -> None:
    image = np.tile(np.linspace(100, 160, 60).astype(np.uint8), (40, 1))
    image[:, 30:] = 20  # a hard boundary the region must not cross
    seeds = np.zeros((40, 60), bool)
    seeds[20, 2] = True
    grown = segment.region_growing(image, seeds, tolerance=40.0)
    assert grown[:, :30].mean() > 0.9
    assert not grown[:, 32:].any()


def test_region_growing_needs_a_seed() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        segment.region_growing(np.zeros((8, 8), np.uint8), np.zeros((8, 8), bool))


def test_watershed_separates_two_touching_discs() -> None:
    """What S6 needs when a shadow bridge or a short following distance merges
    two vehicles into one component."""
    mask = np.zeros((90, 160), np.uint8)
    cv2.circle(mask, (50, 45), 26, 1, -1)
    cv2.circle(mask, (98, 45), 26, 1, -1)
    binary = mask > 0
    assert segment.label_components(binary, 8).count == 1
    split = segment.watershed_split(binary, min_distance_ratio=0.5)
    assert len(np.unique(split[split > 0])) == 2


def test_watershed_leaves_a_single_blob_intact() -> None:
    """A splitter that fragments an unmerged vehicle is worse than no splitter."""
    mask = np.zeros((80, 80), np.uint8)
    cv2.circle(mask, (40, 40), 25, 1, -1)
    split = segment.watershed_split(mask > 0, min_distance_ratio=0.5)
    assert len(np.unique(split[split > 0])) == 1


def test_split_and_merge_finds_uniform_regions() -> None:
    image = np.zeros((64, 64), np.uint8)
    image[:32] = 40
    image[32:] = 200
    labels = segment.split_and_merge(image, threshold=10.0, min_size=4)
    assert len(np.unique(labels[:32])) == 1
    assert len(np.unique(labels[32:])) == 1
    assert labels[0, 0] != labels[63, 63]


# ==========================================================================
# Unit 7 - colour
# ==========================================================================


def test_hue_difference_is_circular() -> None:
    """Trap 6. A naive absolute difference calls two shades of red maximally
    different, and red is exactly where the wrap happens."""
    a = np.array([179.0, 0.0, 90.0, 5.0])
    b = np.array([0.0, 179.0, 100.0, 175.0])
    np.testing.assert_allclose(colour.hue_difference(a, b), [1.0, 1.0, 10.0, 10.0])
    assert float(colour.hue_difference(a, b).max()) <= colour.HUE_MAX / 2


def test_a_multiplicative_darkening_preserves_hue_and_saturation() -> None:
    """The property the whole of S5 rests on, checked exactly rather than
    argued: scaling all three channels scales HSV value and leaves the other two
    channels alone."""
    rng = np.random.default_rng(0)
    bright = rng.integers(30, 240, (64, 64, 3)).astype(np.uint8)
    for scale in (0.5, 0.7, 0.85):
        dark = np.clip(bright.astype(float) * scale, 0, 255).astype(np.uint8)
        a = colour.bgr_to_hsv(bright)
        b = colour.bgr_to_hsv(dark)
        assert float(np.median(colour.hue_difference(a[..., 0], b[..., 0]))) < 0.6
        assert float(np.median(np.abs(a[..., 1] - b[..., 1]))) < 0.01
        assert float(np.median(b[..., 2] / np.maximum(a[..., 2], 1e-9))) == pytest.approx(
            scale, abs=0.01
        )


def test_hand_written_hsi_matches_hsv_hue_on_the_primaries() -> None:
    """HSI is implemented from its trigonometric definition; the primaries have
    known hues, and the two models must agree on them."""
    bgr = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0], [0, 255, 255]]], np.uint8)
    np.testing.assert_allclose(
        colour.bgr_to_hsi(bgr)[..., 0], colour.bgr_to_hsv(bgr)[..., 0], atol=0.5
    )


def test_normalised_rgb_and_c1c2c3_are_invariant_to_uniform_scaling() -> None:
    """Both claim photometric invariance; this is the claim, tested."""
    rng = np.random.default_rng(2)
    bright = rng.integers(60, 240, (32, 32, 3)).astype(np.uint8)
    dark = np.clip(bright.astype(float) * 0.55, 0, 255).astype(np.uint8)
    for space, tolerance in (("normalised_rgb", 0.02), ("c1c2c3", 0.02)):
        a = colour.convert(bright, space)  # type: ignore[arg-type]
        b = colour.convert(dark, space)  # type: ignore[arg-type]
        assert float(np.median(np.abs(a[..., :2] - b[..., :2]))) < tolerance


def test_unknown_colour_space_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown colour space"):
        colour.convert(np.zeros((4, 4, 3), np.uint8), "cielab")  # type: ignore[arg-type]


def test_shadow_mask_finds_a_darkened_region_and_spares_a_dark_object() -> None:
    """The decision S5 exists to make: a shadow scales the surface, a dark
    object replaces it."""
    background = np.full((60, 120, 3), (90, 95, 100), np.uint8)
    frame = background.copy()
    frame[:, :40] = np.clip(background[:, :40].astype(float) * 0.55, 0, 255).astype(np.uint8)
    frame[:, 80:] = (20, 21, 23)  # a near-black car, not a shadow

    mask = colour.shadow_mask_hsv(frame, background)
    assert mask[:, :40].mean() > 0.95, "the shadow was not recognised"
    assert mask[:, 80:].mean() < 0.05, "a black object was called a shadow"
    assert mask[:, 45:75].mean() < 0.05, "unchanged background was called a shadow"


def test_shadow_thresholds_are_validated() -> None:
    background = np.full((8, 8, 3), 100, np.uint8)
    with pytest.raises(ValueError, match="0 < low < high < 1"):
        colour.shadow_mask_hsv(background, background, value_ratio_low=0.9, value_ratio_high=0.4)


def test_hsv_separates_shadow_from_object_better_than_rgb() -> None:
    """The headline claim of the project, measured on a controlled case rather
    than asserted: identical logic in each space, scored against known truth."""
    rng = np.random.default_rng(5)
    background = rng.integers(70, 130, (80, 160, 3)).astype(np.uint8)
    frame = background.copy()
    shadow_truth = np.zeros((80, 160), bool)
    object_truth = np.zeros((80, 160), bool)

    shadow_truth[:, :60] = True
    frame[:, :60] = np.clip(background[:, :60].astype(float) * 0.55, 0, 255).astype(np.uint8)
    object_truth[:, 100:] = True
    frame[:, 100:] = rng.integers(40, 70, (80, 60, 3)).astype(np.uint8)

    scores = colour.compare_colour_spaces(frame, background, shadow_truth, object_truth)
    assert scores["hsv"]["separability"] > scores["rgb"]["separability"]
    assert scores["hsv"]["shadow_recall"] > 0.8


def test_colour_naming_covers_the_vehicle_palette() -> None:
    from kerbside.data.renderer import COLOUR_CATALOGUE

    for name, rgb in COLOUR_CATALOGUE:
        patch = np.full((24, 24, 3), (rgb[2], rgb[1], rgb[0]), np.uint8)
        got, _hsv, _confidence = colour.name_colour(patch, np.ones((24, 24), bool))
        if name == "brown":
            # Brown is dark orange. It gets no hue band of its own because hue
            # alone cannot express it - brown and orange differ in value, not in
            # hue - so the table names it orange. Stated here rather than
            # hidden, because the report reproduces this table verbatim.
            assert got in ("orange", "red")
        else:
            assert got == name, f"{name} was named {got}"


def test_colour_naming_ignores_windows_and_highlights() -> None:
    """Windows are much darker than the paint and highlights much brighter, and
    both have hues unrelated to the body; trimming the value distribution
    removes them without needing to know where they are."""
    body = np.full((40, 40, 3), (40, 40, 170), np.uint8)  # red in BGR
    body[5:15, 5:35] = (30, 30, 30)  # windscreen
    body[35:38, 5:35] = (250, 250, 250)  # specular highlight
    name, _hsv, confidence = colour.name_colour(body, np.ones((40, 40), bool))
    assert name == "red"
    assert confidence > 0.5


def test_colour_naming_declines_on_an_empty_mask() -> None:
    name, _hsv, confidence = colour.name_colour(
        np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), bool)
    )
    assert name == "unknown"
    assert confidence == 0.0


def test_achromatic_bodies_are_named_by_value_not_hue() -> None:
    for expected, level in (("black", 25), ("grey", 120), ("white", 235)):
        patch = np.full((20, 20, 3), level, np.uint8)
        name, _hsv, _c = colour.name_colour(patch, np.ones((20, 20), bool))
        assert name == expected


# ==========================================================================
# Unit 8 - motion
# ==========================================================================


def test_block_matching_recovers_a_synthetic_shift_exactly() -> None:
    """The known-answer test the brief asks for."""
    rng = np.random.default_rng(3)
    reference = cv2.GaussianBlur(rng.random((160, 200)) * 255, (0, 0), 1.2)
    for dx, dy in ((0, 0), (5, 3), (-7, 2), (11, -9)):
        current = cv2.warpAffine(
            reference, np.float32([[1, 0, dx], [0, 1, dy]]), (200, 160),
            borderMode=cv2.BORDER_REFLECT,
        )
        match = motion.exhaustive_search(reference, current[60:76, 80:96], 60, 80, 16)
        assert (match.dx_px, match.dy_px) == (dx, dy)


def test_the_reported_vector_is_content_displacement_not_a_codec_vector() -> None:
    """Sign conventions that disagree produce speeds of the right magnitude
    pointing the wrong way, which reads as a tracking bug. Block matching and
    phase correlation must agree, because S8 combines them."""
    rng = np.random.default_rng(9)
    reference = cv2.GaussianBlur(rng.random((128, 128)) * 255, (0, 0), 1.5)
    current = cv2.warpAffine(
        reference, np.float32([[1, 0, 6], [0, 1, -4]]), (128, 128), borderMode=cv2.BORDER_REFLECT
    )
    match = motion.exhaustive_search(reference, current[48:64, 48:64], 48, 48, 16)
    assert (match.dx_px, match.dy_px) == (6, -4)
    assert match.reference_offset_px == (-6, 4)

    correlation = frequency.phase_correlation(reference, current, window=False)
    assert np.sign(correlation.dx_px) == np.sign(match.dx_px)
    assert np.sign(correlation.dy_px) == np.sign(match.dy_px)


def test_ties_break_towards_zero_displacement() -> None:
    """Trap 10. In a uniform region many candidates share the minimum cost, and
    taking whichever the loop reached first makes the field a picture of the
    scan order."""
    flat = np.full((64, 64), 100.0)
    match = motion.exhaustive_search(flat, flat[16:32, 16:32], 16, 16, 12)
    assert (match.dx_px, match.dy_px) == (0, 0)
    for search in ("two_d_log", "three_step"):
        m = motion.search_block(flat, flat[16:32, 16:32], 16, 16, 12, search=search)  # type: ignore[arg-type]
        assert (m.dx_px, m.dy_px) == (0, 0)


def test_fast_searches_cost_at_least_twenty_times_fewer_comparisons() -> None:
    """The acceptance criterion, and the reason fast searches exist at all."""
    rng = np.random.default_rng(3)
    reference = cv2.GaussianBlur(rng.random((160, 200)) * 255, (0, 0), 1.2)
    current = cv2.warpAffine(
        reference, np.float32([[1, 0, 5], [0, 1, 3]]), (200, 160), borderMode=cv2.BORDER_REFLECT
    )
    block = current[60:76, 80:96]
    exhaustive = motion.exhaustive_search(reference, block, 60, 80, 16)
    assert exhaustive.comparisons == 33 * 33
    for search in ("two_d_log", "three_step"):
        fast = motion.search_block(reference, block, 60, 80, 16, search=search)  # type: ignore[arg-type]
        assert exhaustive.comparisons / fast.comparisons >= 20.0


def test_exhaustive_search_is_accurate_and_fast_searches_measurably_less_so(
    road_grey: np.ndarray,
) -> None:
    """The comparison the syllabus asks for, on real content with known truth.

    Exhaustive search finds the true minimum by construction. The fast searches
    assume a unimodal cost surface, and a road full of dashed markings and
    repeated texture is not unimodal, so they land in the wrong basin often
    enough to matter for a speed measurement - while still predicting the frame
    well enough for a codec, which is the distinction worth drawing.
    """
    patch = road_grey[300:620, 300:940].astype(np.float64)
    true_dx, true_dy = 6, -4
    current = cv2.warpAffine(
        patch, np.float32([[1, 0, true_dx], [0, 1, true_dy]]),
        (patch.shape[1], patch.shape[0]), borderMode=cv2.BORDER_REFLECT,
    )

    scores = {}
    for name in ("exhaustive", "two_d_log", "three_step"):
        field = motion.estimate_motion_field(current, patch, 16, 16, name, min_variance=25.0)  # type: ignore[arg-type]
        gate = field.textured
        errors = np.hypot(field.dx_px[gate] - true_dx, field.dy_px[gate] - true_dy)
        exact = (field.dx_px[gate] == true_dx) & (field.dy_px[gate] == true_dy)
        scores[name] = (float(errors.mean()), float(exact.mean()), field.comparisons)

    assert scores["exhaustive"][0] < 1.0, "exhaustive MV MAE should be sub-pixel"
    assert scores["exhaustive"][1] > 0.90, "exhaustive should be exact on most blocks"
    for name in ("two_d_log", "three_step"):
        assert scores[name][0] > scores["exhaustive"][0]
        assert scores[name][1] < scores["exhaustive"][1]
        assert scores["exhaustive"][2] / scores[name][2] >= 20.0


def test_the_median_aggregate_survives_untextured_blocks(road_grey: np.ndarray) -> None:
    """Trap 9. A plain-sided vehicle contributes arbitrary vectors; a mean is
    dragged by every one, a median ignores them while they are the minority."""
    patch = road_grey[300:500, 300:700].astype(np.float64)
    patch[:80, :200] = 128.0  # a featureless panel, like the flank of a white truck
    current = cv2.warpAffine(
        patch, np.float32([[1, 0, 5], [0, 1, -3]]), (patch.shape[1], patch.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    field = motion.estimate_motion_field(current, patch, 16, 16, "exhaustive", min_variance=25.0)
    assert field.textured_fraction < 1.0, "the featureless panel should be gated out"
    assert motion.aggregate_vector(field) == (5.0, -3.0)


def test_block_variance_gates_a_flat_block() -> None:
    assert motion.block_variance(np.full((16, 16), 100.0)) == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    assert motion.block_variance(rng.random((16, 16)) * 255) > 100.0


def test_motion_compensation_predicts_better_than_the_previous_frame(
    road_grey: np.ndarray,
) -> None:
    """The honest measure of a motion estimator: a tidy-looking field that
    predicts badly is not a good field."""
    patch = road_grey[300:500, 300:700].astype(np.float64)
    current = cv2.warpAffine(
        patch, np.float32([[1, 0, 6], [0, 1, -4]]), (patch.shape[1], patch.shape[0]),
        borderMode=cv2.BORDER_REFLECT,
    )
    field = motion.estimate_motion_field(current, patch, 16, 16, "exhaustive")
    compensated = motion.compensation_psnr(current, patch, field)
    unmoved = transforms.psnr(current, patch)
    assert compensated > unmoved + 5.0


def test_a_candidate_off_the_reference_frame_is_never_chosen() -> None:
    """Clamping the coordinates instead would invent a displacement that puts
    the block outside the image it came from."""
    rng = np.random.default_rng(6)
    reference = cv2.GaussianBlur(rng.random((64, 64)) * 255, (0, 0), 1.5)
    match = motion.exhaustive_search(reference, reference[0:16, 0:16], 0, 0, 16)
    assert match.dx_px <= 0 and match.dy_px <= 0


def test_search_and_cost_names_are_validated() -> None:
    block = np.zeros((8, 8))
    with pytest.raises(ValueError, match="unknown search"):
        motion.search_block(np.zeros((32, 32)), block, 0, 0, search="spiral")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown cost"):
        motion.block_cost(block, block, "ncc")  # type: ignore[arg-type]


def test_sad_surface_is_indexed_in_the_reference_convention() -> None:
    """The surface the Motion Lab draws is in search coordinates; a reported
    match must land at the surface minimum, and stating which convention each
    uses is what keeps that true."""
    rng = np.random.default_rng(8)
    reference = cv2.GaussianBlur(rng.random((96, 96)) * 255, (0, 0), 1.5)
    current = cv2.warpAffine(
        reference, np.float32([[1, 0, 4], [0, 1, -2]]), (96, 96), borderMode=cv2.BORDER_REFLECT
    )
    surface = motion.sad_surface(reference, current, 40, 40, 16, 12)
    match = motion.exhaustive_search(reference, current[40:56, 40:56], 40, 40, 12)
    row, col = np.unravel_index(int(np.argmin(surface)), surface.shape)
    assert (col - 12, row - 12) == match.reference_offset_px


def test_frame_classification_reports_the_evidence_with_the_verdict() -> None:
    """The threshold is a policy decision; the energy is the measurement."""
    a = np.zeros((32, 32), np.uint8)
    kind, energy = motion.classify_frame(a, None)
    assert kind is FrameType.I and energy == float("inf")

    kind, energy = motion.classify_frame(a, a)
    assert kind is FrameType.P and energy == 0.0

    b = a.copy()
    b[10:20, 10:20] = 255
    kind, energy = motion.classify_frame(b, a, threshold=2.0)
    assert kind is FrameType.I and energy > 2.0


def test_frame_classification_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        motion.classify_frame(np.zeros((8, 8)), np.zeros((8, 9)))


def test_declared_parameters_match_the_function_signatures() -> None:
    """The schema and the function must not drift apart.

    The registry's parameter list is what the inspector renders, what the API
    validates against, and what "reset to default" resets to. If it names a
    parameter the function does not take, the control silently does nothing; if
    it carries a different default, the UI opens showing a value the pipeline is
    not using. Both present as "that slider is broken" and neither raises.
    """
    import inspect

    problems: list[str] = []
    for spec in REGISTRY:
        signature = inspect.signature(spec.fn)
        for param in spec.params:
            declared = signature.parameters.get(param.name)
            if declared is None:
                problems.append(f"{spec.operator_id}: declares unknown parameter {param.name!r}")
                continue
            if declared.default is not inspect.Parameter.empty and declared.default != param.default:
                problems.append(
                    f"{spec.operator_id}.{param.name}: schema default {param.default!r} "
                    f"but signature default {declared.default!r}"
                )
    assert not problems, "\n".join(problems)
