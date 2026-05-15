"""Tests for :class:`hpcom.amplifier.EDFA`."""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.amplifier import EDFA


def test_zero_noise_figure_zero_input_gives_only_noise_at_predicted_variance():
    edfa = EDFA(gain_db=20.0, noise_figure_db=5.0, wavelength_m=1550e-9)
    n_samples = 200_000
    field = np.zeros(n_samples, dtype=np.complex128)
    dt_s = 1e-11
    rng = np.random.default_rng(0)
    out = edfa.amplify(field, dt_s, rng=rng)

    measured_var = float(np.var(out.real) + np.var(out.imag))
    expected_var = edfa.ase_noise_density_w_per_hz() * (1.0 / dt_s)
    np.testing.assert_allclose(measured_var, expected_var, rtol=1e-2)


def test_gain_restores_attenuated_signal_power_modulo_ase():
    """Gain matching span loss should restore signal power; ASE adds on top.

    Verifies p_after ≈ p_before + ASE_variance, where ASE is computed from the
    closed-form amplifier formula. Signal level chosen well above ASE so the
    test is dominated by the deterministic gain.
    """
    edfa = EDFA(gain_db=16.0, noise_figure_db=5.0)
    n = 16384
    rng = np.random.default_rng(1)
    field = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)
    attenuated = field * 10 ** (-16.0 / 20.0)
    out = edfa.amplify(attenuated, dt_s=1e-11, rng=np.random.default_rng(2))

    p_before = float(np.mean(np.abs(field) ** 2))
    p_after = float(np.mean(np.abs(out) ** 2))
    expected_ase_power = edfa.ase_noise_density_w_per_hz() * (1.0 / 1e-11)
    np.testing.assert_allclose(p_after, p_before + expected_ase_power, rtol=2e-2)


def test_zero_gain_is_identity_on_field():
    edfa = EDFA(gain_db=0.0, noise_figure_db=0.0)
    field = np.array([1 + 1j, 2 - 0.5j, -0.3 + 0.7j], dtype=np.complex128)
    out = edfa.amplify(field, dt_s=1e-11)
    np.testing.assert_array_equal(out, field)


def test_invalid_params_rejected():
    with pytest.raises(ValueError, match="gain_db"):
        EDFA(gain_db=-1.0)
    with pytest.raises(ValueError, match="noise_figure_db"):
        EDFA(gain_db=10.0, noise_figure_db=-1.0)
    with pytest.raises(ValueError, match="wavelength_m"):
        EDFA(gain_db=10.0, wavelength_m=0.0)


def test_dual_pol_input_gets_independent_per_pol_ase():
    """For a 2-D field, each polarization must independently get N₀·f_s ASE."""
    edfa = EDFA(gain_db=20.0, noise_figure_db=5.0)
    n = 200_000
    dt_s = 1e-11
    field = np.zeros((2, n), dtype=np.complex128)
    out = edfa.amplify(field, dt_s, rng=np.random.default_rng(0))

    expected_var = edfa.ase_noise_density_w_per_hz() * (1.0 / dt_s)
    var_x = float(np.var(out[0].real) + np.var(out[0].imag))
    var_y = float(np.var(out[1].real) + np.var(out[1].imag))
    np.testing.assert_allclose(var_x, expected_var, rtol=1e-2)
    np.testing.assert_allclose(var_y, expected_var, rtol=1e-2)
    # Pols are independent: cross-correlation should be near zero.
    cross = float(np.mean(out[0] * np.conj(out[1])).real)
    assert abs(cross) < expected_var * 1e-1


def test_invalid_field_shape_rejected():
    edfa = EDFA(gain_db=10.0)
    with pytest.raises(ValueError, match="2-D field with leading axis 2"):
        edfa.amplify(np.zeros((3, 100), dtype=np.complex128), dt_s=1e-11)
    with pytest.raises(ValueError, match="1-D or 2-D"):
        edfa.amplify(np.zeros((2, 2, 100), dtype=np.complex128), dt_s=1e-11)


def test_seeded_rng_makes_noise_deterministic():
    edfa = EDFA(gain_db=15.0, noise_figure_db=5.0)
    field = np.zeros(1024, dtype=np.complex128)
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    out_a = edfa.amplify(field, dt_s=1e-11, rng=rng_a)
    out_b = edfa.amplify(field, dt_s=1e-11, rng=rng_b)
    np.testing.assert_array_equal(out_a, out_b)
