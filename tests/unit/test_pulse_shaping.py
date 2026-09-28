"""RRC pulse-shaping (Test B): the filter that determines ISI.

These tests pin the fundamental invariants of ``root_raised_cosine``: unit
energy, symmetry, Nyquist (cascaded) zero-ISI at integer symbol periods, and
sane behaviour at the edge values of roll-off.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom._pulse import root_raised_cosine


@pytest.mark.parametrize("sps", [2, 4, 8, 16])
@pytest.mark.parametrize("roll_off", [0.0, 0.1, 0.25, 0.5, 1.0])
def test_unit_energy(sps: int, roll_off: float):
    h = root_raised_cosine(n_taps=sps * 16 + 1, samples_per_symbol=sps, roll_off=roll_off)
    np.testing.assert_allclose(float(np.sum(h**2)), 1.0, atol=1e-12)


@pytest.mark.parametrize("sps", [4, 8, 16])
@pytest.mark.parametrize("roll_off", [0.0, 0.1, 0.5, 1.0])
def test_even_symmetry(sps: int, roll_off: float):
    n_taps = sps * 32 + 1  # odd → center tap on a sample
    h = root_raised_cosine(n_taps=n_taps, samples_per_symbol=sps, roll_off=roll_off)
    np.testing.assert_allclose(h, h[::-1], atol=1e-12)


@pytest.mark.parametrize("sps", [4, 8, 16])
@pytest.mark.parametrize("roll_off", [0.1, 0.25, 0.5])
def test_cascaded_rrc_is_nyquist_zero_isi(sps: int, roll_off: float):
    """Two cascaded RRCs form a raised cosine which is zero at non-zero
    symbol-period multiples. This is the *only* reason RRC is used."""
    n_taps = sps * 64 + 1  # long enough for accurate convolution
    h = root_raised_cosine(n_taps=n_taps, samples_per_symbol=sps, roll_off=roll_off)
    rc = np.convolve(h, h)
    center = (rc.size - 1) // 2
    # Sample at integer symbol periods around the center.
    n_syms = 8
    for k in range(1, n_syms + 1):
        idx_plus = center + k * sps
        idx_minus = center - k * sps
        # Tolerance: finite-length RRC has small leakage; ~1e-3 is realistic
        # for n_taps=64·sps. Tighter as filter grows.
        assert abs(rc[idx_plus]) < 1e-3
        assert abs(rc[idx_minus]) < 1e-3


@pytest.mark.parametrize("sps", [4, 8])
def test_cascaded_rrc_peak_at_center(sps: int):
    n_taps = sps * 32 + 1
    h = root_raised_cosine(n_taps=n_taps, samples_per_symbol=sps, roll_off=0.25)
    rc = np.convolve(h, h)
    center = (rc.size - 1) // 2
    assert np.argmax(rc) == center
    # The center value should be close to 1 for a Nyquist raised cosine that
    # is itself unit-energy at the squared level.
    assert rc[center] > 0.0


def test_roll_off_zero_is_pure_sinc():
    """At β=0, RRC degenerates to a sinc — verify the canonical zero crossings
    at non-zero integer symbol multiples on the prototype before normalization."""
    sps = 8
    n_taps = sps * 32 + 1
    h = root_raised_cosine(n_taps=n_taps, samples_per_symbol=sps, roll_off=0.0)
    # The prototype t=0 value is 1 (sinc(0)); samples at t=k symbol periods
    # (k != 0) should be ~0. After unit-energy normalisation those zeros are
    # preserved (multiplicative constant).
    center = (n_taps - 1) // 2
    for k in range(1, 6):
        assert abs(h[center + k * sps]) < 1e-12
        assert abs(h[center - k * sps]) < 1e-12


def test_roll_off_one_has_no_singularity_blowup():
    """At β=1 the second singularity sits at t = ±1/4 symbol periods, which
    is within the filter span. Verify the limit value replaces the NaN cleanly."""
    sps = 8
    h = root_raised_cosine(n_taps=sps * 32 + 1, samples_per_symbol=sps, roll_off=1.0)
    assert np.all(np.isfinite(h))


def test_singularity_value_is_continuous_limit():
    """At t = 1/(4β), the RRC formula's denominator vanishes; the closed-form
    L'Hôpital limit must match the surrounding samples to within smooth-curve
    tolerances (no spike)."""
    sps = 64  # high oversampling so neighbour samples are very close to t = 1/(4β)
    beta = 0.5
    h = root_raised_cosine(n_taps=sps * 16 + 1, samples_per_symbol=sps, roll_off=beta)
    # Find the tap closest to t = 1/(4β) in symbol periods.
    n_taps = h.size
    t = (np.arange(n_taps) - (n_taps - 1) / 2.0) / sps
    idx = int(np.argmin(np.abs(t - 1.0 / (4.0 * beta))))
    # Compare to neighbours: smooth function ⇒ |h[idx] - h[idx±1]| is small.
    diff = max(abs(h[idx] - h[idx - 1]), abs(h[idx] - h[idx + 1]))
    assert diff < 5e-3, f"discontinuity at singularity sample: diff={diff}"


def test_rejects_out_of_range_roll_off():
    with pytest.raises(ValueError, match="roll_off"):
        root_raised_cosine(n_taps=33, samples_per_symbol=4, roll_off=-0.1)
    with pytest.raises(ValueError, match="roll_off"):
        root_raised_cosine(n_taps=33, samples_per_symbol=4, roll_off=1.1)


def test_rejects_invalid_sps():
    with pytest.raises(ValueError, match="samples_per_symbol"):
        root_raised_cosine(n_taps=33, samples_per_symbol=0, roll_off=0.1)


# ---------------------------------------------------------------------------
# Frequency-domain (exact) RRC and the three shaping modes
# ---------------------------------------------------------------------------

from hpcom._pulse import matched_filter, pulse_shape, raised_cosine_spectrum  # noqa: E402

MODES = [(None, "circular"), (32, "circular"), (32, "linear")]


@pytest.mark.parametrize("roll_off", [0.0, 0.05, 0.1, 0.5, 1.0])
def test_raised_cosine_satisfies_the_nyquist_folding_identity(roll_off: float):
    f = np.linspace(0.0, 1.0, 1001)
    np.testing.assert_allclose(
        raised_cosine_spectrum(f, roll_off) + raised_cosine_spectrum(f - 1.0, roll_off),
        1.0,
        atol=1e-15,
    )


@pytest.mark.parametrize("roll_off", [0.0, 0.25, 1.0])
def test_raised_cosine_support_and_peak(roll_off: float):
    f = np.linspace(-2, 2, 4001)
    rc = raised_cosine_spectrum(f, roll_off)
    assert rc.max() == 1.0
    assert np.all(rc[np.abs(f) > (1 + roll_off) / 2 + 1e-12] == 0.0)


@pytest.mark.parametrize(("span", "conv"), MODES)
@pytest.mark.parametrize("sps", [2, 3, 8])
def test_transmit_normalisation_and_receive_unit_gain(span, conv, sps):
    """Tx output power = E|a|² (for i.i.d. symbols); a single impulse comes back with gain 1."""
    rng = np.random.default_rng(0)
    a = (rng.choice([-1, 1], 8192) + 1j * rng.choice([-1, 1], 8192)) / np.sqrt(2)
    x = pulse_shape(a, sps, 0.1, span, conv)
    np.testing.assert_allclose(np.mean(np.abs(x) ** 2), 1.0, rtol=0.02)
    impulse = np.zeros(256, complex)
    impulse[128] = 1.0
    y = matched_filter(pulse_shape(impulse, sps, 0.1, span, conv), sps, 0.1, span, conv)
    assert abs(y[128] - 1.0) < 1e-12


@pytest.mark.parametrize(("span", "conv"), MODES)
def test_matched_filter_noise_gain_is_one_over_sps(span, conv):
    rng = np.random.default_rng(1)
    noise = (rng.standard_normal(1 << 20) + 1j * rng.standard_normal(1 << 20)) / np.sqrt(2)
    np.testing.assert_allclose(
        np.var(matched_filter(noise, 4, 0.1, span, conv)) * 4, 1.0, rtol=0.01
    )


def test_exact_rrc_cascade_is_isi_free_to_rounding():
    a = np.exp(2j * np.pi * np.random.default_rng(2).random(1024))
    for sps in (2, 5, 16):
        y = matched_filter(pulse_shape(a, sps, 0.2, None, "circular"), sps, 0.2, None, "circular")
        np.testing.assert_allclose(y, a, rtol=0, atol=1e-13)


def test_shaping_works_along_the_last_axis_of_any_shape():
    a = np.exp(2j * np.pi * np.random.default_rng(3).random((2, 3, 64)))
    x = pulse_shape(a, 4, 0.1, None, "circular")
    assert x.shape == (2, 3, 256)
    np.testing.assert_allclose(x[1, 2], pulse_shape(a[1, 2], 4, 0.1, None, "circular"))


def test_linear_mode_requires_a_finite_span():
    with pytest.raises(ValueError, match="finite"):
        pulse_shape(np.ones(8), 4, 0.1, None, "linear")
    with pytest.raises(ValueError, match="finite"):
        matched_filter(np.ones(32), 4, 0.1, None, "linear")


def test_fir_longer_than_the_signal_is_rejected():
    with pytest.raises(ValueError, match="needs 257 taps"):
        pulse_shape(np.ones(8), 8, 0.1, 32, "circular")


def test_matched_filter_needs_whole_symbols():
    with pytest.raises(ValueError, match="multiple of samples_per_symbol"):
        matched_filter(np.ones(33), 4, 0.1, None, "circular")
