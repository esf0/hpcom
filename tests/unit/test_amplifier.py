"""EDFA gain and ASE noise (convention: Essiambre et al., JLT 2010; see EDFA docstring).

Expected noise levels are computed here from the formulas themselves —
``S_ASE = n_sp hν (G−1)`` per polarisation with ``n_sp = (GF−1)/(2(G−1))`` —
not from the class's own helper methods.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.amplifier import EDFA

H = 6.62607015e-34
C = 299_792_458.0


def _expected_psd_per_pol(gain_db: float, nf_db: float, wavelength_m: float = 1550e-9) -> float:
    g, f = 10 ** (gain_db / 10), 10 ** (nf_db / 10)
    n_sp = (g * f - 1) / (2 * (g - 1))
    return n_sp * H * (C / wavelength_m) * (g - 1)


@pytest.mark.parametrize(("gain_db", "nf_db"), [(20.0, 5.0), (16.0, 4.5), (10.0, 0.0), (30.0, 3.0)])
def test_psd_per_polarization_matches_literature_formula(gain_db, nf_db):
    edfa = EDFA(gain_db=gain_db, noise_figure_db=nf_db)
    np.testing.assert_allclose(
        edfa.ase_psd_per_polarization_w_per_hz, _expected_psd_per_pol(gain_db, nf_db), rtol=1e-14
    )


def test_high_gain_psd_is_half_of_nf_h_nu_g():
    """For G ≫ 1: S_ASE → F·hν·G/2 per polarisation (both pols together ≈ F·hν·G)."""
    edfa = EDFA(gain_db=40.0, noise_figure_db=5.0)
    approx = 10**0.5 * H * (C / 1550e-9) * 1e4 / 2
    np.testing.assert_allclose(edfa.ase_psd_per_polarization_w_per_hz, approx, rtol=1e-4)


def test_quantum_limited_amplifier_has_unit_inversion_factor():
    """F = 2 − 1/G (≈ 3 dB) ⇔ n_sp = 1, the ideal fully inverted amplifier."""
    g = 1000.0
    edfa = EDFA(gain_db=30.0, noise_figure_db=10 * np.log10(2 - 1 / g))
    np.testing.assert_allclose(edfa.spontaneous_emission_factor, 1.0, rtol=1e-12)


def test_new_convention_is_half_the_pre_v1_level_at_high_gain():
    """Pre-v1 hpcom used hν(G−1)F per polarisation (see EDFA docstring note)."""
    edfa = EDFA(gain_db=30.0, noise_figure_db=5.0)
    old = H * (C / 1550e-9) * (1e3 - 1) * 10**0.5
    np.testing.assert_allclose(edfa.ase_psd_per_polarization_w_per_hz / old, 0.5, rtol=1e-3)


@pytest.mark.parametrize("shape", [(200_000,), (2, 100_000), (4, 2, 25_000)])
def test_noise_on_zero_input_has_the_predicted_variance_in_every_mode(shape):
    edfa = EDFA(gain_db=20.0, noise_figure_db=5.0)
    dt_s = 1e-11
    out = edfa.amplify(np.zeros(shape, dtype=complex), dt_s, rng=np.random.default_rng(0))
    expected = _expected_psd_per_pol(20.0, 5.0) / dt_s
    assert out.shape == shape
    per_mode_var = np.var(out.reshape(-1, shape[-1]), axis=-1)
    np.testing.assert_allclose(per_mode_var, expected, rtol=0.02)
    np.testing.assert_allclose(np.var(out.real), expected / 2, rtol=0.02)
    np.testing.assert_allclose(np.var(out.imag), expected / 2, rtol=0.02)


def test_polarisation_noise_is_independent():
    edfa = EDFA(gain_db=20.0, noise_figure_db=5.0)
    out = edfa.amplify(np.zeros((2, 200_000), dtype=complex), 1e-11, rng=np.random.default_rng(1))
    corr = np.abs(np.mean(out[0] * np.conj(out[1]))) / np.sqrt(
        np.mean(np.abs(out[0]) ** 2) * np.mean(np.abs(out[1]) ** 2)
    )
    assert corr < 1e-2  # ≈ 1/√n


def test_gain_is_exact_on_the_deterministic_part():
    """Same RNG seed with and without a signal: the difference is exactly √G·signal."""
    edfa = EDFA(gain_db=16.0, noise_figure_db=5.0)
    signal = np.random.default_rng(2).standard_normal(4096) + 0j
    with_signal = edfa.amplify(signal, 1e-11, rng=np.random.default_rng(9))
    noise_only = edfa.amplify(np.zeros_like(signal), 1e-11, rng=np.random.default_rng(9))
    np.testing.assert_allclose(with_signal - noise_only, signal * 10 ** (16 / 20), rtol=1e-13)


def test_output_power_is_gain_times_input_plus_ase():
    edfa = EDFA(gain_db=16.0, noise_figure_db=5.0)
    rng = np.random.default_rng(1)
    field = (rng.standard_normal(200_000) + 1j * rng.standard_normal(200_000)) * 1e-4
    out = edfa.amplify(field, 1e-11, rng=np.random.default_rng(2))
    expected = 10**1.6 * np.mean(np.abs(field) ** 2) + _expected_psd_per_pol(16.0, 5.0) / 1e-11
    np.testing.assert_allclose(np.mean(np.abs(out) ** 2), expected, rtol=1e-2)


def test_unit_gain_is_noiseless_identity_and_draws_no_random_numbers():
    edfa = EDFA(gain_db=0.0, noise_figure_db=6.0)
    field = np.array([1 + 1j, 2 - 0.5j, -0.3 + 0.7j])
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    np.testing.assert_array_equal(edfa.amplify(field, 1e-11, rng=rng), field)
    assert rng.bit_generator.state == state
    assert edfa.ase_psd_per_polarization_w_per_hz == 0.0
    assert np.isnan(edfa.spontaneous_emission_factor)


def test_zero_noise_figure_still_adds_ase():
    """NF = 0 dB (F = 1) is below the quantum limit but still gives n_sp = 1/2 > 0."""
    edfa = EDFA(gain_db=20.0, noise_figure_db=0.0)
    assert edfa.spontaneous_emission_factor == pytest.approx(0.5)
    assert edfa.ase_psd_per_polarization_w_per_hz > 0


def test_seeded_rng_makes_noise_deterministic():
    edfa = EDFA(gain_db=15.0)
    field = np.zeros(1024, dtype=complex)
    a = edfa.amplify(field, 1e-11, rng=np.random.default_rng(42))
    b = edfa.amplify(field, 1e-11, rng=np.random.default_rng(42))
    np.testing.assert_array_equal(a, b)


def test_apply_inverse_is_noiseless_division_by_sqrt_gain():
    edfa = EDFA(gain_db=20.0)
    field = np.array([1.0 + 2j, -3.0])
    np.testing.assert_allclose(edfa.apply_inverse(field, 1e-11), field / 10.0, rtol=1e-15)


def test_apply_is_amplify():
    edfa = EDFA(gain_db=10.0)
    field = np.ones(8, dtype=complex)
    np.testing.assert_array_equal(
        edfa.apply(field, 1e-11, rng=np.random.default_rng(3)),
        edfa.amplify(field, 1e-11, rng=np.random.default_rng(3)),
    )


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"gain_db": -1.0}, "gain_db"),
        ({"gain_db": 10.0, "noise_figure_db": -1.0}, "noise_figure_db"),
        ({"gain_db": 10.0, "wavelength_m": 0.0}, "wavelength_m"),
        ({"gain_db": 10.0, "gain": 3.0}, "gain"),  # unknown keyword → error, not ignored
    ],
)
def test_invalid_parameters_are_rejected(kwargs, field):
    with pytest.raises(ValueError, match=field):
        EDFA(**kwargs)
