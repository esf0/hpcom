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
