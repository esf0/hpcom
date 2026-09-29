"""Multicore fibers on every backend: exact solutions, invariants, cross-backend agreement.

Coupled NLSE per core with linear coupling ``+iΣκ_mn A_n`` (see
``hpcom.multicore``). Checks:

* uncoupled cores are independent single-core fibers (exact);
* linear coupled-mode theory: two-core coupler with and without phase
  mismatch, 7-core hexagonal centre launch (exact to rounding);
* Jensen's nonlinear coherent coupler, P₁ = P/2·[1 + cn(2κz | (P/P_c)²)],
  P_c = 4κ/γ (second-order convergence of the split step);
* the symmetric two-core soliton supermode (exact phase κ + γP₀/2);
* skew delays each core by δβ₁,m·L exactly;
* energy conservation, exact inverse, and agreement with NumPy for every
  scheme, with skew and mismatch.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import ellipj

from hpcom.backends import get_propagator
from hpcom.multicore import hexagonal_coupling, linear_array_coupling, ring_coupling


@dataclass(frozen=True)
class StubScheme:
    method: str = "ssfm"
    nonlinear_position: float = 0.5
    kerr_length: str = "step"
    kerr_filter_bandwidth_hz: float | None = None
    kerr_filter_shape: str = "gaussian"
    local_error: float | None = None


@dataclass(frozen=True)
class StubMulticoreFiber:
    """Duck-typed multicore FiberLike (independent of :class:`hpcom.multicore.MulticoreFiber`)."""

    coupling_per_m: np.ndarray = field(default_factory=lambda: linear_array_coupling(2, 1e-3))
    core_beta0_offsets_per_m: np.ndarray | None = None
    core_beta1_offsets_s_per_m: np.ndarray | None = None
    length_km: float = 1.0
    alpha_db_per_km: float = 0.0
    beta1_s_per_m: float = 0.0
    beta2_s2_per_m: float = 0.0
    beta3_s3_per_m: float = 0.0
    gamma_per_w_per_km: float = 0.0
    n_steps: int = 5
    step_distribution: str = "uniform"
    step_log_factor: float = 1.0
    scheme: StubScheme = StubScheme()
    polarization: str = "single"

    @property
    def n_cores(self) -> int:
        return int(self.coupling_per_m.shape[0])


@dataclass(frozen=True)
class StubFiber:
    """The single-core fiber with the same parameters."""

    length_km: float
    alpha_db_per_km: float
    beta1_s_per_m: float
    beta2_s2_per_m: float
    beta3_s3_per_m: float
    gamma_per_w_per_km: float
    n_steps: int
    step_distribution: str = "uniform"
    step_log_factor: float = 1.0
    scheme: StubScheme = StubScheme()
    polarization: str = "single"
    n_cores: int = 1


def _single(f: StubMulticoreFiber, beta1_extra: float = 0.0) -> StubFiber:
    return StubFiber(
        length_km=f.length_km,
        alpha_db_per_km=f.alpha_db_per_km,
        beta1_s_per_m=f.beta1_s_per_m + beta1_extra,
        beta2_s2_per_m=f.beta2_s2_per_m,
        beta3_s3_per_m=f.beta3_s3_per_m,
        gamma_per_w_per_km=f.gamma_per_w_per_km,
        n_steps=f.n_steps,
        scheme=f.scheme,
        polarization=f.polarization,
    )


def _field(shape: tuple[int, ...], power_w: float, seed: int = 0, band: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[..., np.abs(np.fft.fftfreq(shape[-1])) > band] = 0
    x = np.fft.ifft(spectrum, axis=-1)
    return x * np.sqrt(power_w / np.mean(np.abs(x) ** 2))


DT = 1e-12
KAPPA = 1e-3  # 1/m
GAMMA = 1.27e-3  # 1/(W·m)


def _propagate(backend: str, fiber, field, direction="forward"):
    return get_propagator(backend).propagate(field, fiber, dt_s=DT, direction=direction)


# ---------------------------------------------------------------------------
# Linear coupled-mode theory (exact)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_uncoupled_cores_are_independent_fibers(nonlinear_backend, polarization):
    f = StubMulticoreFiber(
        coupling_per_m=np.zeros((3, 3)),
        length_km=20.0,
        alpha_db_per_km=0.2,
        beta2_s2_per_m=-2.16e-26,
        beta3_s3_per_m=1.2e-40,
        gamma_per_w_per_km=1.27,
        n_steps=30,
        polarization=polarization,
    )
    shape = (3, 1024) if polarization == "single" else (3, 2, 1024)
    x = _field(shape, 5e-3)
    out = _propagate(nonlinear_backend, f, x)
    for m in range(3):
        np.testing.assert_allclose(
            out[m], _propagate(nonlinear_backend, _single(f), x[m]), rtol=0, atol=1e-13
        )


@pytest.mark.parametrize("z_km", [0.3, 1.0, 2.5])
def test_two_core_coupler_matches_coupled_mode_theory(any_backend, z_km):
    """P₁ = cos²κz; with mismatch ±δ: P₂ = κ²/(κ²+δ²)·sin²(√(κ²+δ²) z)."""
    cw = np.zeros((2, 64), complex)
    cw[0] = 1.0
    z = z_km * 1e3
    f = StubMulticoreFiber(length_km=z_km)
    out = _propagate(any_backend, f, cw)
    np.testing.assert_allclose(np.abs(out[0]) ** 2, np.cos(KAPPA * z) ** 2, atol=1e-13)
    np.testing.assert_allclose(np.abs(out[1]) ** 2, np.sin(KAPPA * z) ** 2, atol=1e-13)
    d = 0.5e-3
    mismatched = StubMulticoreFiber(length_km=z_km, core_beta0_offsets_per_m=np.array([d, -d]))
    out = _propagate(any_backend, mismatched, cw)
    q = np.hypot(KAPPA, d)
    np.testing.assert_allclose(
        np.abs(out[1]) ** 2, (KAPPA / q) ** 2 * np.sin(q * z) ** 2, atol=1e-13
    )


def test_hexagonal_centre_launch_matches_the_reduced_system(any_backend):
    """Centre a, ring cores b (all equal by symmetry): [a, b]′ = iκ[[0, 6], [1, 2]]·[a, b]."""
    z = 1300.0
    f = StubMulticoreFiber(coupling_per_m=hexagonal_coupling(KAPPA), length_km=z / 1e3)
    cw = np.zeros((7, 16), complex)
    cw[0] = 1.0
    out = _propagate(any_backend, f, cw)
    a, b = expm(1j * KAPPA * np.array([[0, 6], [1, 2]]) * z) @ [1.0, 0.0]
    np.testing.assert_allclose(out[0], a, atol=1e-13)
    for m in range(1, 7):
        np.testing.assert_allclose(out[m], b, atol=1e-13)


def test_skew_delays_each_core_exactly(any_backend):
    f = StubMulticoreFiber(
        coupling_per_m=np.zeros((3, 3)),
        length_km=10.0,
        core_beta1_offsets_s_per_m=np.array([0.0, 7, -4]) * DT / 10e3,
    )
    x = _field((3, 256), 1.0)
    out = _propagate(any_backend, f, x)
    for m, shift in enumerate((0, 7, -4)):
        np.testing.assert_allclose(out[m], np.roll(x[m], shift), atol=1e-13)


# ---------------------------------------------------------------------------
# Nonlinear exact solutions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("power_fraction", [0.5, 0.9])
def test_nonlinear_coupler_matches_jensen(nonlinear_backend, power_fraction):
    """CW, lossless, no dispersion: P₁(z) = P/2·[1 + cn(2κz | m)], m = (P/P_c)², P_c = 4κ/γ."""
    z = 2000.0
    p = power_fraction * 4 * KAPPA / GAMMA
    exact = p / 2 * (1 + ellipj(2 * KAPPA * z, power_fraction**2)[1])
    cw = np.zeros((2, 16), complex)
    cw[0] = np.sqrt(p)
    errors = []
    for n_steps in (200, 400, 800):
        f = StubMulticoreFiber(length_km=z / 1e3, gamma_per_w_per_km=GAMMA * 1e3, n_steps=n_steps)
        errors.append(
            abs(np.mean(np.abs(_propagate(nonlinear_backend, f, cw)[0]) ** 2) - exact) / p
        )
    assert errors[-1] < 1e-5
    assert all(3.7 < a / b < 4.3 for a, b in itertools.pairwise(errors)), errors


def test_two_core_soliton_supermode(nonlinear_backend):
    """A₁ = A₂ = √P₀ sech(t/T₀), γP₀T₀² = |β₂|: stationary, phase (κ + γP₀/2)z."""
    b2, t0 = -2.16e-26, 5e-12
    p0 = abs(b2) / (GAMMA * t0**2)
    length = 3 * t0**2 / abs(b2)
    n, dt = 2048, t0 / 40
    tau = (np.arange(n) - n // 2) * dt / t0
    s = np.sqrt(p0) / np.cosh(tau)
    f = StubMulticoreFiber(
        length_km=length / 1e3,
        beta2_s2_per_m=b2,
        gamma_per_w_per_km=GAMMA * 1e3,
        n_steps=800,
    )
    out = get_propagator(nonlinear_backend).propagate(np.stack([s, s]), f, dt_s=dt)
    expected = s * np.exp(1j * (KAPPA + GAMMA * p0 / 2) * length)
    assert np.max(np.abs(out - expected)) / np.sqrt(p0) < 1e-5


# ---------------------------------------------------------------------------
# Invariants and cross-backend agreement, with mismatch and skew
# ---------------------------------------------------------------------------


_STRANG = StubScheme()


def _general(scheme: StubScheme = _STRANG, alpha: float = 0.0, **kw) -> StubMulticoreFiber:
    kw.setdefault("n_steps", 13)
    return StubMulticoreFiber(
        coupling_per_m=ring_coupling(4, 2e-4),
        core_beta0_offsets_per_m=np.array([0.0, 1e-4, -2e-4, 3e-5]),
        core_beta1_offsets_s_per_m=np.array([0.0, 1e-15, -2e-15, 5e-16]),
        length_km=50.0,
        alpha_db_per_km=alpha,
        beta2_s2_per_m=-2.16e-26,
        beta3_s3_per_m=1.2e-40,
        gamma_per_w_per_km=1.27,
        scheme=scheme,
        polarization="dual",
        **kw,
    )


SPLIT_SCHEMES = {
    "strang": ({}, StubScheme()),
    "log-cle-suzuki": (
        {"step_distribution": "logarithmic", "step_log_factor": 1 / 3},
        StubScheme(method="ssfm4-suzuki"),
    ),
    "effective-filtered": (
        {},
        StubScheme(kerr_length="effective", kerr_filter_bandwidth_hz=40e9),
    ),
}


@pytest.mark.parametrize(("placement", "scheme"), SPLIT_SCHEMES.values(), ids=SPLIT_SCHEMES)
def test_energy_is_conserved_and_backward_is_exact(nonlinear_backend, placement, scheme):
    x = _field((4, 2, 1024), 1e-2)
    lossless = _general(scheme, **placement)
    out = _propagate(nonlinear_backend, lossless, x)
    np.testing.assert_allclose(np.sum(np.abs(out) ** 2), np.sum(np.abs(x) ** 2), rtol=1e-12)
    lossy = _general(scheme, alpha=0.2, **placement)
    back = _propagate(
        nonlinear_backend, lossy, _propagate(nonlinear_backend, lossy, x), direction="backward"
    )
    np.testing.assert_allclose(back, x, rtol=0, atol=1e-12 * np.max(np.abs(x)))


@pytest.mark.parametrize(
    "scheme",
    [
        StubScheme(),
        StubScheme(method="ssfm4-yoshida"),
        StubScheme(method="rk4ip"),
        StubScheme(method="ssfm-local-error", local_error=1e-5),
        StubScheme(nonlinear_position=0.2, kerr_length="effective"),
    ],
    ids=["strang", "yoshida", "rk4ip", "local-error", "r0.2-effective"],
)
def test_multicore_agrees_with_numpy_reference(nonlinear_backend, scheme):
    fiber = _general(scheme, alpha=0.2)
    x = _field((4, 2, 1024), 1e-2)
    ref = get_propagator("numpy").propagate(x, fiber, dt_s=DT)
    out = get_propagator(nonlinear_backend).propagate(x, fiber, dt_s=DT)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.max(np.abs(ref)))


@pytest.mark.parametrize(
    ("method", "steps", "low", "high"),
    [("ssfm", (64, 128, 256), 3.6, 4.6), ("ssfm4-suzuki", (128, 256, 512), 12.0, 24.0)],
)
def test_multicore_convergence_order(nonlinear_backend, method, steps, low, high):
    """Coupling, mismatch and skew live in the exact linear step: the order is the scheme's.

    Asymptotic regime: a 30 GHz field (at 200 GHz both single- and multicore
    runs are still pre-asymptotic at these step counts).
    """
    x = _field((4, 2, 1024), 1e-2, band=0.03)
    ref = _propagate(
        nonlinear_backend, _general(StubScheme(method="ssfm4-suzuki"), n_steps=4096), x
    )
    errors = [
        np.linalg.norm(
            _propagate(nonlinear_backend, _general(StubScheme(method=method), n_steps=k), x) - ref
        )
        / np.linalg.norm(ref)
        for k in steps
    ]
    ratios = [a / b for a, b in itertools.pairwise(errors)]
    assert all(low < q < high for q in ratios), ratios
