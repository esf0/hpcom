"""Energy conservation under lossless / noiseless propagation.

The NLSE conserves the L² norm (total optical energy) when α = 0 and no ASE
is added. This is a global, model-agnostic invariant — any backend that
violates it has a bug regardless of which equation it claims to solve.

Tolerance: 1e-6 relative drift over 1000 km of fibre. Tighter than this is
sensitive to SSFM step size; looser would let real bugs through.
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
    beta2_s2_per_m: float = -2.16e-26
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 200
    polarization: str = "single"


def _energy(field: np.ndarray, dt_s: float) -> float:
    return float(np.sum(np.abs(field) ** 2) * dt_s)


def test_energy_conserved_over_long_distance():
    fiber = _LosslessFiber(length_km=1000.0)
    n = 2048
    dt_s = 1e-12
    rng = np.random.default_rng(0)
    # Modest power so nonlinearity is non-trivial but doesn't blow up SSFM.
    field = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 2e-2).astype(np.complex128)

    e_in = _energy(field, dt_s)
    out = get_propagator("tensorflow").propagate(field, fiber, dt_s=dt_s)
    e_out = _energy(out, dt_s)

    np.testing.assert_allclose(e_out, e_in, rtol=1e-6)


def test_energy_decays_with_loss_per_beer_lambert():
    """With α > 0 and γ = 0, energy decays by exp(-α · L) (intensity)."""

    @dataclass(frozen=True)
    class LossyLinearFiber:
        length_km: float = 500.0
        alpha_db_per_km: float = 0.2
        beta2_s2_per_m: float = -2.16e-26
        gamma_per_w_per_km: float = 0.0
        n_steps: int = 1
        polarization: str = "single"

    fiber = LossyLinearFiber()
    n = 1024
    dt_s = 1e-12
    rng = np.random.default_rng(1)
    field = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3).astype(np.complex128)

    e_in = _energy(field, dt_s)
    out = get_propagator("numpy-linear").propagate(field, fiber, dt_s=dt_s)
    e_out = _energy(out, dt_s)

    alpha_np_per_m = fiber.alpha_db_per_km * np.log(10.0) / 10.0 / 1e3
    expected_factor = np.exp(-alpha_np_per_m * fiber.length_km * 1e3)
    np.testing.assert_allclose(e_out / e_in, expected_factor, rtol=1e-10)
