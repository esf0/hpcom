"""Forward → backward propagation (digital back-propagation) on the SSFM backend.

Reference: a perfect, noise-free DBP must invert forward propagation
exactly (modulo SSFM step error). This is the strongest single test that
the TF backend's nonlinear step is the algebraic inverse of itself under
the sign flip in ``direction="backward"``.

Tolerance: 1e-4 RMS over a single noiseless 80 km span at modest power
(soliton-like regime where the nonlinear phase shift per step is ~1e-2).
SSFM step error scales like O(dz²) — bumping ``n_steps`` would tighten
this further.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from hpcom.backends import get_propagator


@dataclass(frozen=True)
class _NoiselessFiber:
    length_km: float = 80.0
    alpha_db_per_km: float = 0.0
    beta2_s2_per_m: float = -2.16e-26
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 200
    polarization: str = "single"


def test_forward_then_backward_recovers_input_with_nonlinearity():
    fiber = _NoiselessFiber()
    n = 1024
    dt_s = 1e-12
    rng = np.random.default_rng(0)
    field = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 5e-2).astype(np.complex128)

    prop = get_propagator("tensorflow")
    forward = prop.propagate(field, fiber, dt_s=dt_s, direction="forward")
    recovered = prop.propagate(forward, fiber, dt_s=dt_s, direction="backward")

    rms_err = float(np.sqrt(np.mean(np.abs(recovered - field) ** 2)))
    rms_signal = float(np.sqrt(np.mean(np.abs(field) ** 2)))
    relative = rms_err / rms_signal
    assert relative < 1e-4, f"DBP residual {relative:.2e} exceeds 1e-4"
