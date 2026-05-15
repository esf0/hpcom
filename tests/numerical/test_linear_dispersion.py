"""Closed-form linear dispersion correctness for :class:`LinearNumpyPropagator`.

A Gaussian pulse propagated through pure dispersion has an analytical solution:

  A(t, 0) = exp(-t² / (2 T₀²))
  A(t, z) = T₀ / sqrt(T₀² - i β₂ z) * exp(-t² / (2 (T₀² - i β₂ z)))

Reference: Agrawal, *Nonlinear Fiber Optics* (5th ed.), Eq. 3.2.7.

Tolerance: matches to within 1e-10 on a well-resolved grid (γ = 0, α = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hpcom.backends import get_propagator


@dataclass(frozen=True)
class _LossLessFiber:
    length_km: float = 50.0
    alpha_db_per_km: float = 0.0
    beta2_s2_per_m: float = -2.16e-26  # SSMF at 1550 nm
    gamma_per_w_per_km: float = 0.0
    n_steps: int = 1
    polarization: str = "single"


def test_gaussian_pulse_matches_closed_form_dispersion():
    fiber = _LossLessFiber()
    z_m = fiber.length_km * 1e3

    # Time grid: 4096 samples over a window much wider than the pulse.
    n = 4096
    dt_s = 1e-12  # 1 ps
    t = (np.arange(n) - n // 2) * dt_s

    # Input Gaussian, T0 = 50 ps (well-resolved on this grid).
    t0 = 50e-12
    field_in = np.exp(-(t**2) / (2 * t0**2)).astype(np.complex128)

    # Closed-form propagated field.
    denom = t0**2 - 1j * fiber.beta2_s2_per_m * z_m
    expected = (t0 / np.sqrt(denom)) * np.exp(-(t**2) / (2 * denom))

    # Numerical propagation. ifftshift so the pulse is at index 0 in FFT order
    # (matches NumPy's FFT convention used inside the propagator).
    propagator = get_propagator("numpy-linear")
    field_in_shifted = np.fft.ifftshift(field_in)
    expected_shifted = np.fft.ifftshift(expected)
    out_shifted = propagator.propagate(field_in_shifted, fiber, dt_s=dt_s)

    np.testing.assert_allclose(out_shifted, expected_shifted, atol=1e-10, rtol=0)


def test_round_trip_recovers_input():
    """Forward then backward propagation should recover the input identically (linear, lossless)."""
    fiber = _LossLessFiber(length_km=100.0)
    n = 2048
    dt_s = 2e-12
    t = (np.arange(n) - n // 2) * dt_s
    field_in = np.exp(-(t**2) / (2 * (30e-12) ** 2)).astype(np.complex128)
    field_in = np.fft.ifftshift(field_in)

    propagator = get_propagator("numpy-linear")
    forward = propagator.propagate(field_in, fiber, dt_s=dt_s, direction="forward")
    recovered = propagator.propagate(forward, fiber, dt_s=dt_s, direction="backward")

    np.testing.assert_allclose(recovered, field_in, atol=1e-12, rtol=0)


def test_loss_only_attenuates_amplitude():
    """With β₂ = 0, propagation should only scale the field by exp(-α L / 2)."""

    @dataclass(frozen=True)
    class LossyDispersionFreeFiber:
        length_km: float = 80.0
        alpha_db_per_km: float = 0.2
        beta2_s2_per_m: float = 0.0
        gamma_per_w_per_km: float = 0.0
        n_steps: int = 1
        polarization: str = "single"

    fiber = LossyDispersionFreeFiber()
    field_in = np.ones(64, dtype=np.complex128)
    out = get_propagator("numpy-linear").propagate(field_in, fiber, dt_s=1e-11)

    alpha_np_per_m = fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3
    expected_amp = np.exp(-alpha_np_per_m * fiber.length_km * 1e3 / 2.0)
    np.testing.assert_allclose(out, field_in * expected_amp, atol=1e-12, rtol=0)
