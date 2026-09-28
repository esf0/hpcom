"""Numerical parity between :class:`TensorFlowPropagator` and the original ssfm-gpu.

The TF backend in v1 wraps a vendored copy of the ``ssfm-gpu`` package
(see ``src/hpcom/backends/tensorflow/_vendor/``). This test pins the wrapper
to byte-for-byte numerical equivalence with the upstream reference output.

The golden ``.npz`` was generated once by running the original
``ssfm_gpu.propagation.tf_fiber_propogate`` (and ``tf_manakov_fiber_propogate``)
on a fixed seeded input. If the vendor is intentionally updated, regenerate
the golden via ``scripts/regen_tf_parity.py`` (TODO) and bump the file
suffix (``_v0.npz`` → ``_v1.npz``).

Tolerance: 1e-12 (per spec — the wrapper is a thin adapter, not a
reimplementation; differences should be exactly zero modulo TF nondeterminism
on GPU, which we don't exercise here).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("tensorflow")

from hpcom.backends import get_propagator

GOLDEN = Path(__file__).parent / "_data" / "tf_parity_v0.npz"


@dataclass(frozen=True)
class _Fiber:
    length_km: float
    alpha_db_per_km: float
    beta2_s2_per_m: float
    gamma_per_w_per_km: float
    n_steps: int
    polarization: str = "single"


def _load_golden() -> dict[str, np.ndarray]:
    if not GOLDEN.exists():
        pytest.fail(
            f"golden parity file missing: {GOLDEN}. Regenerate via the script in "
            "tests/numerical/_data/ before running CI."
        )
    with np.load(GOLDEN) as data:
        return {k: data[k] for k in data.files}


def _fiber_from_golden(g: dict[str, np.ndarray], *, polarization: str = "single") -> _Fiber:
    return _Fiber(
        length_km=float(g["length_km"]),
        alpha_db_per_km=float(g["alpha_db_per_km"]),
        # vendor uses s²/km; FiberLike stores s²/m. Convert back here.
        beta2_s2_per_m=float(g["beta2_s2_per_km"]) * 1e-3,
        gamma_per_w_per_km=float(g["gamma"]),
        n_steps=int(g["n_steps"]),
        polarization=polarization,
    )


def test_scalar_propagation_matches_original_ssfm_gpu():
    g = _load_golden()
    fiber = _fiber_from_golden(g)
    out = get_propagator("tensorflow").propagate(
        g["field_scalar_in"].astype(np.complex128),
        fiber,
        dt_s=float(g["dt_s"]),
    )
    np.testing.assert_allclose(out, g["out_scalar"], atol=1e-12, rtol=0)


def test_manakov_propagation_matches_original_ssfm_gpu():
    g = _load_golden()
    fiber = _fiber_from_golden(g, polarization="dual")
    out = get_propagator("tensorflow").propagate(
        g["field_manakov_in"].astype(np.complex128),
        fiber,
        dt_s=float(g["dt_s"]),
    )
    np.testing.assert_allclose(out, g["out_manakov"], atol=1e-12, rtol=0)


def test_backward_undoes_forward_within_lossless_dispersion_only():
    """Round-trip on a lossless, γ=0 fiber must recover the input nearly exactly."""
    fiber = _Fiber(
        length_km=50.0,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=-2.16e-26,
        gamma_per_w_per_km=0.0,
        n_steps=50,
    )
    rng = np.random.default_rng(7)
    field = ((rng.standard_normal(512) + 1j * rng.standard_normal(512)) * 1e-3).astype(
        np.complex128
    )

    prop = get_propagator("tensorflow")
    forward = prop.propagate(field, fiber, dt_s=2e-11, direction="forward")
    recovered = prop.propagate(forward, fiber, dt_s=2e-11, direction="backward")
    np.testing.assert_allclose(recovered, field, atol=1e-10, rtol=0)
