"""Polarisation-mode dispersion on every backend (coarse-step Manakov-PMD model).

* the span's DGD is Maxwellian with mean D_PMD·√L over realisations;
* a pulse launched in a principal state is delayed by ±τ/2 without distortion;
* without DGD, PMD is the frequency-independent rotation Q_N of the
  Manakov solution (the Manakov Kerr term is rotation invariant);
* split-step schemes stay exactly invertible and keep their orders;
* every backend agrees with the NumPy reference.

Fibers are duck-typed (:class:`StubFiber`), so these pin the backend contract.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest
from scipy import stats

from hpcom.backends import get_propagator
from hpcom.backends.base import apply_jones, pmd_realization
from tests.backends.test_propagator_conformance import StubFiber, StubScheme, _random_field

PMD = {"polarization": "dual", "pmd_ps_per_sqrt_km": 1.0, "pmd_sections": 100}
LINEAR = {"alpha_db_per_km": 0.0, "beta2_s2_per_m": 0.0, "gamma_per_w_per_km": 0.0}


def _mean_dgd(fiber: StubFiber) -> float:
    return fiber.pmd_ps_per_sqrt_km * 1e-12 * math.sqrt(fiber.length_km)


def _jones(
    backend: str, fiber: StubFiber, n: int = 8, dt: float = 0.5e-9
) -> tuple[np.ndarray, np.ndarray]:
    """T(ω) on the grid from the impulse responses of X and Y: ``(n, 2, 2)``, and ω."""
    cols = []
    for pol in range(2):
        x = np.zeros((2, n), complex)
        x[pol, 0] = 1.0
        cols.append(np.fft.fft(get_propagator(backend).propagate(x, fiber, dt), axis=-1))
    return np.moveaxis(np.stack(cols, axis=1), -1, 0), 2 * np.pi * np.fft.fftfreq(n, dt)


def _dgd_and_psp(t: np.ndarray, w: np.ndarray) -> tuple[float, np.ndarray]:
    """DGD τ and the input principal state of the slow axis, at ω = 0 (central difference)."""
    dt_dw = (t[1] - t[-1]) / (w[1] - w[-1])
    m = 1j * t[0].conj().T @ dt_dw  # input-side group-delay operator (Hermitian)
    values, vectors = np.linalg.eigh((m + m.conj().T) / 2)
    return float(values[1] - values[0]), vectors[:, 1]


def test_dgd_is_maxwellian_with_mean_d_pmd_sqrt_length():
    fiber = StubFiber(length_km=100.0, n_steps=1, **LINEAR, **PMD)
    taus = np.array(
        [_dgd_and_psp(*_jones("numpy-linear", replace(fiber, pmd_seed=s)))[0] for s in range(1500)]
    )
    mean = _mean_dgd(fiber)
    assert taus.mean() == pytest.approx(mean, rel=0.03)  # standard error ≈ 1.1 %
    assert np.mean(taus**2) == pytest.approx(3 * math.pi / 8 * mean**2, rel=0.06)
    scale = mean * math.sqrt(math.pi) / (2 * math.sqrt(2))  # Maxwell: mean = 2a√(2/π)
    assert stats.kstest(taus, stats.maxwell(scale=scale).cdf).pvalue > 0.01


def test_every_backend_has_the_same_jones_matrix(any_backend):
    fiber = StubFiber(length_km=80.0, n_steps=7, **LINEAR, **PMD, pmd_seed=5)
    t, _ = _jones(any_backend, fiber)
    ref, _ = _jones("numpy-linear", fiber)
    np.testing.assert_allclose(t, ref, atol=1e-13)
    np.testing.assert_allclose(t[0] @ t[0].conj().T, np.eye(2), atol=1e-13)  # lossless → unitary


@pytest.mark.parametrize("seed", [1, 2])
def test_principal_states_are_delayed_by_plus_and_minus_half_the_dgd(nonlinear_backend, seed):
    """First-order PMD: a pulse much longer than τ, launched in a PSP, only moves by ±τ/2."""
    fiber = StubFiber(length_km=400.0, n_steps=5, **LINEAR, **PMD, pmd_seed=seed)
    tau, slow = _dgd_and_psp(*_jones("numpy-linear", fiber))
    fast = np.array([-np.conj(slow[1]), np.conj(slow[0])])
    n, dt = 4096, 2e-12
    t = (np.arange(n) - n // 2) * dt
    pulse = np.fft.ifftshift(np.exp(-(t**2) / (2 * (60e-12) ** 2)))

    def centroid(state: np.ndarray) -> float:
        out = get_propagator(nonlinear_backend).propagate(state[:, None] * pulse, fiber, dt)
        p = np.fft.fftshift(np.sum(np.abs(out) ** 2, axis=0))
        return float(np.sum(t * p) / np.sum(p))

    assert tau > 5e-12  # a clearly measurable DGD
    assert centroid(slow) == pytest.approx(tau / 2, rel=0.01)
    assert centroid(fast) == pytest.approx(-tau / 2, rel=0.01)


def test_without_dgd_pmd_rotates_the_manakov_solution(nonlinear_backend):
    """Q_N commutes with dispersion, loss and the Manakov Kerr term."""
    base = StubFiber(length_km=80.0, alpha_db_per_km=0.2, n_steps=20, polarization="dual")
    tiny = replace(base, pmd_ps_per_sqrt_km=1e-9, pmd_seed=4)
    x = _random_field((2, 1024), 0.1)
    prop = get_propagator(nonlinear_backend)
    rotated = apply_jones(prop.propagate(x, base, 1e-11), pmd_realization(tiny).output_rotation)
    out = prop.propagate(x, tiny, 1e-11)
    np.testing.assert_allclose(out, rotated, atol=1e-10 * np.max(np.abs(out)))


@pytest.mark.parametrize("method", ["ssfm", "ssfm4-yoshida", "ssfm4-suzuki"])
def test_split_step_with_pmd_is_exactly_invertible(nonlinear_backend, method):
    fiber = StubFiber(
        length_km=80.0,
        alpha_db_per_km=0.2,
        beta3_s3_per_m=1.2e-40,
        n_steps=12,
        step_distribution="logarithmic",
        scheme=StubScheme(method=method),
        **PMD,
        pmd_seed=9,
    )
    x = _random_field((2, 2048), 0.1)
    prop = get_propagator(nonlinear_backend)
    back = prop.propagate(prop.propagate(x, fiber, 1e-11), fiber, 1e-11, direction="backward")
    np.testing.assert_allclose(back, x, atol=1e-11 * np.max(np.abs(x)))


@pytest.mark.parametrize(
    ("method", "steps", "order"),
    [
        ("ssfm", (100, 200, 400), 2),
        ("ssfm4-suzuki", (100, 200, 400), 4),
        ("rk4ip", (100, 200, 400), 4),
    ],
)
def test_schemes_keep_their_order_with_pmd(nonlinear_backend, method, steps, order):
    fiber = StubFiber(
        length_km=80.0,
        alpha_db_per_km=0.2,
        polarization="dual",
        pmd_ps_per_sqrt_km=0.5,
        pmd_sections=50,
        pmd_seed=3,
    )
    x = _random_field((2, 1024), 0.05)
    prop = get_propagator(nonlinear_backend)
    ref = prop.propagate(
        x, replace(fiber, n_steps=3000, scheme=StubScheme(method="ssfm4-suzuki")), 1e-11
    )
    errors = [
        np.max(
            np.abs(
                prop.propagate(
                    x, replace(fiber, n_steps=k, scheme=StubScheme(method=method)), 1e-11
                )
                - ref
            )
        )
        for k in steps
    ]
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine == pytest.approx(2**order, rel=0.2)


@pytest.mark.parametrize("method", ["ssfm", "rk4ip"])
def test_pmd_agrees_with_numpy_reference(nonlinear_backend, method):
    fiber = StubFiber(
        length_km=80.0,
        alpha_db_per_km=0.2,
        n_steps=30,
        scheme=StubScheme(method=method, nonlinear_position=0.5),
        **PMD,
        pmd_seed=11,
    )
    x = _random_field((3, 2, 2048), 0.08)
    ref = get_propagator("numpy").propagate(x, fiber, 1e-11)
    out = get_propagator(nonlinear_backend).propagate(x, fiber, 1e-11)
    np.testing.assert_allclose(out, ref, atol=1e-12 * np.max(np.abs(ref)))
