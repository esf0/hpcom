"""Cross-backend equivalence: NumPy linear vs. TensorFlow with γ = 0.

Pins the convention seam between the two propagator implementations. Catches
any future drift in unit conversions (β₂ s²/m → s²/km, α dB/km → Np/km, length
scaling) — without this, a subtle bug in either backend's adapter would only
show up much later in physics-level tests.

For the linear-only case (γ = 0) the two backends solve the *same* equation
and must agree to many digits; the only differences are arithmetic precision
and SSFM step discretisation in the TF kernel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from hpcom.backends import get_propagator


@dataclass(frozen=True)
class _Fiber:
    length_km: float = 50.0
    alpha_db_per_km: float = 0.2
    beta2_s2_per_m: float = -2.16e-26
    gamma_per_w_per_km: float = 0.0
    n_steps: int = 200  # fine SSFM grid → close to closed-form linear answer
    polarization: str = "single"


@pytest.mark.parametrize("alpha_db_per_km", [0.0, 0.2])
def test_numpy_and_tf_match_in_linear_regime(alpha_db_per_km: float):
    fiber = _Fiber(alpha_db_per_km=alpha_db_per_km)

    rng = np.random.default_rng(0)
    n = 1024
    field = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3).astype(np.complex128)
    dt_s = 1e-11

    out_np = get_propagator("numpy-linear").propagate(field, fiber, dt_s=dt_s)
    out_tf = get_propagator("tensorflow").propagate(field, fiber, dt_s=dt_s)

    # SSFM with a fine grid converges to the closed-form linear operator;
    # 1e-8 leaves headroom for arithmetic precision but is tight enough to
    # catch any unit-conversion drift.
    np.testing.assert_allclose(out_tf, out_np, atol=1e-8, rtol=1e-6)


def test_backward_direction_consistent_across_backends():
    fiber = _Fiber(alpha_db_per_km=0.0)  # lossless so backward is well-defined
    n = 512
    rng = np.random.default_rng(1)
    field = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * 1e-3).astype(np.complex128)
    dt_s = 2e-11

    np_back = get_propagator("numpy-linear").propagate(field, fiber, dt_s=dt_s, direction="backward")
    tf_back = get_propagator("tensorflow").propagate(field, fiber, dt_s=dt_s, direction="backward")
    np.testing.assert_allclose(tf_back, np_back, atol=1e-8, rtol=1e-6)
