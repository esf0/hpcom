"""Fundamental soliton shape preservation under nonlinear propagation.

The fundamental soliton of the NLSE (Agrawal §5.2) is

    A(t, 0) = A₀ · sech(t / T₀)

with the soliton amplitude

    A₀ = sqrt(|β₂| / (γ · T₀²))

Over any propagation distance ``z`` the *intensity* profile is invariant
(only an overall phase ``exp(i z / 2 L_D)`` accumulates). We propagate one
soliton period over a lossless fiber and check the post-propagation
intensity profile against the input within a tight tolerance.

Tolerance: a fine SSFM grid (``n_steps = 400``) gets within ~1e-3 in
relative shape error; we assert 5e-3 as a robust gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from hpcom.backends import get_propagator


@dataclass(frozen=True)
class _LosslessFiber:
    length_km: float
    alpha_db_per_km: float = 0.0
    beta2_s2_per_m: float = -2.16e-26  # SSMF @ 1550 nm, anomalous (soliton-supporting)
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 400
    polarization: str = "single"


def test_fundamental_soliton_preserves_intensity_profile():
    # Soliton width and amplitude.
    t0_s = 25e-12
    beta2 = -2.16e-26  # s²/m
    gamma_per_w_per_km = 1.27
    gamma_per_w_per_m = gamma_per_w_per_km * 1e-3

    a0 = float(np.sqrt(abs(beta2) / (gamma_per_w_per_m * t0_s**2)))

    # Time grid wide enough to contain the sech tails to floating-point zero.
    n = 4096
    dt_s = 0.5e-12  # 0.5 ps → 8·T₀ window centered
    t = (np.arange(n) - n // 2) * dt_s
    field = (a0 / np.cosh(t / t0_s)).astype(np.complex128)
    # Re-centre to FFT order so the propagator's frequency grid is correct.
    field = np.fft.ifftshift(field)

    # One soliton period z₀ = (π / 2) · L_D, with L_D = T₀² / |β₂|.
    l_d_m = t0_s**2 / abs(beta2)
    z_period_m = (np.pi / 2.0) * l_d_m
    fiber = _LosslessFiber(length_km=z_period_m * 1e-3)

    out = get_propagator("tensorflow").propagate(field, fiber, dt_s=dt_s)

    # Compare normalized intensity profiles (drop the global phase).
    p_in = np.abs(field) ** 2
    p_out = np.abs(out) ** 2
    # Both are concentrated; normalize peak to compare shape.
    p_in /= p_in.max()
    p_out /= p_out.max()

    rel_shape_err = float(np.sqrt(np.mean((p_out - p_in) ** 2)))
    assert rel_shape_err < 5e-3, f"soliton shape drift {rel_shape_err:.2e} exceeds tolerance"
