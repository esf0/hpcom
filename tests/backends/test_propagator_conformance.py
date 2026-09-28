"""Conformance suite for fiber-propagation backends.

Every available registered backend runs every test here (fixtures
``any_backend`` / ``nonlinear_backend`` in ``conftest.py``), so a new backend
(JAX, Rust, …) is accepted only if it reproduces the same physics. Physics
expectations are analytic (closed forms, invariants, known solutions), not
another implementation — except the explicit cross-implementation checks at
the end.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from hpcom.backends import get_propagator

GOLDEN = Path(__file__).parents[1] / "numerical" / "_data" / "tf_parity_v0.npz"


@dataclass(frozen=True)
class StubFiber:
    """Duck-typed :class:`~hpcom._typing.FiberLike` (independent of :class:`hpcom.Fiber`)."""

    length_km: float = 80.0
    alpha_db_per_km: float = 0.0
    beta1_s_per_m: float = 0.0
    beta2_s2_per_m: float = -2.16e-26
    beta3_s3_per_m: float = 0.0
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 100
    step_distribution: str = "uniform"
    step_log_factor: float = 1.0
    polarization: str = "single"


#: Step placements every nonlinear backend must support (docs/numerics.rst).
PLACEMENTS = {
    "uniform": {},
    "log-k1": {"step_distribution": "logarithmic", "step_log_factor": 1.0},
    "log-k1/3": {"step_distribution": "logarithmic", "step_log_factor": 1 / 3},
}


def _placement_params():
    return [pytest.param(kw, id=name) for name, kw in PLACEMENTS.items()]


def _random_field(shape: tuple[int, ...], scale: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * scale


def _gaussian(n: int, dt_s: float, t0_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Unchirped Gaussian exp(−t²/2T₀²) centred at t = 0, in FFT order."""
    t = (np.arange(n) - n // 2) * dt_s
    return np.fft.ifftshift(t), np.fft.ifftshift(np.exp(-(t**2) / (2 * t0_s**2))).astype(complex)


# ---------------------------------------------------------------------------
# Interface contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("polarization", "shape"),
    [("single", (256,)), ("single", (3, 256)), ("dual", (2, 256)), ("dual", (4, 2, 256))],
)
def test_output_shape_dtype_and_input_untouched(any_backend, polarization, shape):
    fiber = StubFiber(length_km=10.0, n_steps=5, polarization=polarization)
    field = _random_field(shape, 1e-2)
    before = field.copy()
    out = get_propagator(any_backend).propagate(field, fiber, dt_s=1e-11)
    assert out.shape == shape
    assert out.dtype == np.complex128
    np.testing.assert_array_equal(field, before)


@pytest.mark.parametrize("shape", [(256,), (3, 256), (2, 3, 256)])
def test_dual_polarisation_requires_axis_of_two(any_backend, shape):
    with pytest.raises(ValueError, match="dual-polarisation"):
        get_propagator(any_backend).propagate(
            _random_field(shape, 1e-2), StubFiber(polarization="dual"), dt_s=1e-11
        )


def test_rejects_unknown_direction(any_backend):
    with pytest.raises(ValueError, match="direction"):
        get_propagator(any_backend).propagate(
            _random_field((64,), 1e-2), StubFiber(), dt_s=1e-11, direction="sideways"
        )


def test_batch_axes_are_independent(nonlinear_backend):
    """Propagating a batch equals propagating each member alone (nonlinear, Manakov)."""
    fiber = StubFiber(length_km=40.0, n_steps=20, polarization="dual", alpha_db_per_km=0.2)
    batch = _random_field((3, 2, 512), 3e-2)
    prop = get_propagator(nonlinear_backend)
    together = prop.propagate(batch, fiber, dt_s=1e-12)
    for i in range(3):
        np.testing.assert_allclose(
            together[i], prop.propagate(batch[i], fiber, dt_s=1e-12), rtol=0, atol=1e-14
        )


# ---------------------------------------------------------------------------
# Linear propagation
# ---------------------------------------------------------------------------


def test_gaussian_dispersion_matches_closed_form(any_backend):
    """A(t,z) = T₀/√(T₀² − iβ₂z) · exp(−t²/2(T₀² − iβ₂z)) (Agrawal, NFO, eq. 3.2.7)."""
    fiber = StubFiber(length_km=50.0, gamma_per_w_per_km=0.0, n_steps=3)
    t, a0 = _gaussian(4096, 1e-12, 50e-12)
    z = fiber.length_km * 1e3
    denom = (50e-12) ** 2 - 1j * fiber.beta2_s2_per_m * z
    expected = 50e-12 / np.sqrt(denom) * np.exp(-(t**2) / (2 * denom))
    out = get_propagator(any_backend).propagate(a0, fiber, dt_s=1e-12)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-11)


@pytest.mark.parametrize("beta3_sign", [+1.0, -1.0])
def test_third_order_dispersion_delays_pulse_centroid(any_backend, beta3_sign):
    """With β₂ = 0 the intensity centroid moves by β₃L/(4T₀²) (trailing edge for β₃ > 0).

    The group delay of the TOD phase is τ(ω) = β₃ω²L/2 ≥ 0 for β₃ > 0; the
    centroid shift is its average over |Ã(ω)|² ∝ exp(−ω²T₀²), i.e.
    β₃L⟨ω²⟩/2 = β₃L/(4T₀²). The sign pins hpcom's convention to Agrawal's
    (oscillatory tail on the trailing edge for β₃ > 0).
    """
    t0, dt = 2e-12, 0.05e-12
    beta3 = beta3_sign * 1.2e-40
    fiber = StubFiber(
        length_km=500.0, beta2_s2_per_m=0.0, beta3_s3_per_m=beta3, gamma_per_w_per_km=0.0, n_steps=2
    )
    t, a0 = _gaussian(1 << 14, dt, t0)
    out = get_propagator(any_backend).propagate(a0, fiber, dt_s=dt)
    intensity = np.abs(out) ** 2
    centroid = float(np.sum(t * intensity) / np.sum(intensity))
    expected = beta3 * fiber.length_km * 1e3 / (4.0 * t0**2)
    assert np.sign(centroid) == beta3_sign
    np.testing.assert_allclose(centroid, expected, rtol=1e-6)


@pytest.mark.parametrize("delay_samples", [5, -12])
def test_group_delay_shifts_the_field_exactly(any_backend, delay_samples):
    """β₁ delays the field by β₁L: an exact circular shift for a whole number of samples."""
    dt, length_km = 1e-12, 40.0
    fiber = StubFiber(
        length_km=length_km,
        beta1_s_per_m=delay_samples * dt / (length_km * 1e3),
        beta2_s2_per_m=0.0,
        gamma_per_w_per_km=0.0,
        n_steps=3,
    )
    a0 = _random_field((1024,), 1.0)
    out = get_propagator(any_backend).propagate(a0, fiber, dt_s=dt)
    np.testing.assert_allclose(out, np.roll(a0, delay_samples), atol=1e-12)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_group_delay_commutes_with_dispersion_and_nonlinearity(nonlinear_backend, polarization):
    """A whole-sample β₁h per step shifts the β₁ = 0 output (the Kerr step is pointwise).

    Fractional per-step delays commute only up to the grid's aliasing of the
    nonlinear products — like any frame change of a sampled NLSE.
    """
    dt, n, n_steps, per_step = 1e-12, 1024, 25, 2  # whole samples per full step
    shape = (2, n) if polarization == "dual" else (n,)
    base = StubFiber(
        length_km=50.0, beta3_s3_per_m=1.2e-40, n_steps=n_steps, polarization=polarization
    )
    delay = n_steps * per_step  # the two half steps add up to one more full step
    delayed = replace(base, beta1_s_per_m=delay * dt / (base.length_km * 1e3))
    a0 = _random_field(shape, 0.05)
    prop = get_propagator(nonlinear_backend)
    np.testing.assert_allclose(
        prop.propagate(a0, delayed, dt_s=dt),
        np.roll(prop.propagate(a0, base, dt_s=dt), delay, axis=-1),
        atol=1e-11,
    )


def test_loss_follows_beer_lambert(any_backend):
    fiber = StubFiber(length_km=120.0, alpha_db_per_km=0.2, gamma_per_w_per_km=0.0, n_steps=7)
    field = _random_field((2048,), 1e-3)
    out = get_propagator(any_backend).propagate(field, fiber, dt_s=1e-11)
    ratio = np.sum(np.abs(out) ** 2) / np.sum(np.abs(field) ** 2)
    np.testing.assert_allclose(ratio, 10 ** (-0.2 * 120.0 / 10.0), rtol=1e-12)


def test_loss_without_dispersion_only_scales_amplitude(any_backend):
    """β₂ = β₃ = γ = 0: A(L) = A(0)·exp(−αL/2) sample by sample."""
    fiber = StubFiber(
        length_km=80.0, alpha_db_per_km=0.2, beta2_s2_per_m=0.0, gamma_per_w_per_km=0.0
    )
    field = _random_field((512,), 1e-3)
    out = get_propagator(any_backend).propagate(field, fiber, dt_s=1e-11)
    np.testing.assert_allclose(
        out, field * 10 ** (-0.2 * 80.0 / 20.0), rtol=0, atol=1e-13 * np.max(np.abs(field))
    )


def test_linear_regime_matches_exact_linear_solution(nonlinear_backend):
    """γ = 0: the SSFM reduces to the exact linear operator whatever n_steps is."""
    fiber = StubFiber(
        length_km=77.0,
        alpha_db_per_km=0.21,
        beta3_s3_per_m=1.2e-40,
        gamma_per_w_per_km=0.0,
        n_steps=13,
    )
    field = _random_field((2, 1024), 1e-3)
    exact = get_propagator("numpy-linear").propagate(field, fiber, dt_s=1e-11)
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=1e-11)
    np.testing.assert_allclose(out, exact, rtol=0, atol=1e-13 * np.max(np.abs(exact)))


# ---------------------------------------------------------------------------
# Nonlinear propagation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_energy_conserved_without_loss(nonlinear_backend, polarization):
    """α = 0: every SSFM step is unitary, so energy is conserved to rounding error."""
    fiber = StubFiber(length_km=1000.0, n_steps=200, polarization=polarization)
    shape = (2048,) if polarization == "single" else (2, 2048)
    field = _random_field(shape, 2e-2)
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=1e-12)
    np.testing.assert_allclose(np.sum(np.abs(out) ** 2), np.sum(np.abs(field) ** 2), rtol=1e-12)


def test_self_phase_modulation_without_dispersion(nonlinear_backend):
    """β₂ = β₃ = α = 0: A(L) = A(0)·exp(iγ|A(0)|²L) exactly (a single Kerr step)."""
    fiber = StubFiber(length_km=50.0, beta2_s2_per_m=0.0, n_steps=10)
    field = _random_field((1024,), 3e-2)
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=1e-12)
    expected = field * np.exp(1j * 1.27e-3 * np.abs(field) ** 2 * 50e3)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-14)


def test_manakov_phase_without_dispersion(nonlinear_backend):
    """Manakov, β₂ = 0: both pols rotate by (8/9)γ(|A_x|²+|A_y|²)L."""
    fiber = StubFiber(length_km=50.0, beta2_s2_per_m=0.0, n_steps=10, polarization="dual")
    field = _random_field((2, 1024), 3e-2)
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=1e-12)
    phase = (8.0 / 9.0) * 1.27e-3 * np.sum(np.abs(field) ** 2, axis=0) * 50e3
    np.testing.assert_allclose(out, field * np.exp(1j * phase), rtol=0, atol=1e-14)


def test_manakov_with_empty_y_equals_scalar_with_eight_ninths_gamma(nonlinear_backend):
    fiber = StubFiber(length_km=80.0, alpha_db_per_km=0.2, n_steps=40)
    x = _random_field((1024,), 5e-2)
    dual = get_propagator(nonlinear_backend).propagate(
        np.stack([x, np.zeros_like(x)]), replace(fiber, polarization="dual"), dt_s=1e-12
    )
    scalar = get_propagator(nonlinear_backend).propagate(
        x, replace(fiber, gamma_per_w_per_km=1.27 * 8.0 / 9.0), dt_s=1e-12
    )
    np.testing.assert_allclose(dual[0], scalar, rtol=0, atol=1e-15)
    np.testing.assert_array_equal(dual[1], 0.0)


def test_fundamental_soliton_is_stationary_up_to_its_phase(nonlinear_backend):
    """A(t,z) = A₀ sech(t/T₀) e^{iz/2L_D}, A₀² = |β₂|/(γT₀²) (Agrawal, NFO, §5.2)."""
    t0, beta2, gamma = 10e-12, -2.16e-26, 1.27e-3
    a0 = np.sqrt(abs(beta2) / (gamma * t0**2))
    l_d = t0**2 / abs(beta2)
    z = np.pi / 2 * l_d  # one soliton period
    fiber = StubFiber(length_km=z / 1e3, beta2_s2_per_m=beta2, n_steps=2000)
    t = np.fft.ifftshift((np.arange(4096) - 2048) * 0.25e-12)
    field = a0 / np.cosh(t / t0) + 0j
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=0.25e-12)
    expected = field * np.exp(1j * z / (2 * l_d))
    err = np.max(np.abs(out - expected)) / a0
    assert err < 2e-5, f"soliton deviates by {err:.2e} of its amplitude"


@pytest.mark.parametrize("placement", _placement_params())
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_backward_propagation_is_exact_inverse(any_backend, polarization, placement):
    """The steps in reverse order with negated lengths undo forward propagation to rounding."""
    fiber = StubFiber(
        length_km=80.0,
        alpha_db_per_km=0.2,
        beta3_s3_per_m=1.2e-40,
        n_steps=50,
        polarization=polarization,
        **placement,
    )
    shape = (1024,) if polarization == "single" else (2, 1024)
    field = _random_field(shape, 5e-2)
    prop = get_propagator(any_backend)
    back = prop.propagate(
        prop.propagate(field, fiber, dt_s=1e-12), fiber, dt_s=1e-12, direction="backward"
    )
    np.testing.assert_allclose(back, field, rtol=0, atol=1e-12 * np.max(np.abs(field)))


def _order_test_field() -> tuple[np.ndarray, float]:
    """30 GHz band-limited random signal at 10 mW (asymptotic regime over 80 km)."""
    n, dt = 4096, 2e-12
    spectrum = np.fft.fft(_random_field((n,), 1.0))
    spectrum[np.abs(np.fft.fftfreq(n, dt)) > 30e9] = 0.0
    field = np.fft.ifft(spectrum)
    return field * np.sqrt(10e-3 / np.mean(np.abs(field) ** 2)), dt


_ORDER_BASE = StubFiber(length_km=80.0, alpha_db_per_km=0.2)


@functools.cache
def _order_reference(backend: str) -> np.ndarray:
    field, dt = _order_test_field()
    return get_propagator(backend).propagate(field, replace(_ORDER_BASE, n_steps=16384), dt_s=dt)


def _order_errors(backend: str, placement: dict, steps: tuple[int, ...]) -> list[float]:
    field, dt = _order_test_field()
    ref = _order_reference(backend)
    prop = get_propagator(backend)
    return [
        float(
            np.linalg.norm(
                prop.propagate(field, replace(_ORDER_BASE, n_steps=k, **placement), dt_s=dt) - ref
            )
            / np.linalg.norm(ref)
        )
        for k in steps
    ]


@pytest.mark.parametrize(
    ("placement", "steps"),
    [
        pytest.param(PLACEMENTS["uniform"], (80, 160, 320), id="uniform"),
        pytest.param(PLACEMENTS["log-k1/3"], (80, 160, 320), id="log-k1/3"),
        # k = 1 stretches the tail steps ≈ e^{αL} = 40×; it reaches the asymptote later.
        pytest.param(PLACEMENTS["log-k1"], (640, 1280, 2560), id="log-k1"),
    ],
)
def test_split_step_error_is_second_order(nonlinear_backend, placement, steps):
    """Halving every step divides the global error by ≈ 4, for any smooth placement.

    Symmetric (Strang) splitting has local error O(h³); a placement that
    scales as L/K keeps Σ h³ = O(K⁻²). Needs the asymptotic regime: a 30 GHz
    signal at 10 mW over 80 km with 0.2 dB/km.
    """
    errors = _order_errors(nonlinear_backend, placement, steps)
    ratios = [a / b for a, b in itertools.pairwise(errors)]
    assert all(3.8 < r < 4.6 for r in ratios), f"convergence ratios {ratios} (expected ≈ 4)"


def test_placement_changes_the_error_constant_not_the_order(nonlinear_backend):
    """Same regime at 320 steps: constant-local-error placement (k = 1/3) beats uniform,
    equal-nonlinear-phase placement (k = 1) is worse — its long tail steps dominate
    (cf. Musetti, Serena & Bononi 2018; docs/numerics.rst)."""
    uniform, cle, nlp = (
        _order_errors(nonlinear_backend, PLACEMENTS[name], (320,))[0]
        for name in ("uniform", "log-k1/3", "log-k1")
    )
    assert cle < 0.6 * uniform
    assert nlp > 2 * uniform


def test_logarithmic_steps_without_loss_are_uniform(nonlinear_backend):
    field = _random_field((2, 1024), 5e-2)
    lossless = StubFiber(length_km=80.0, beta3_s3_per_m=1.2e-40, n_steps=40, polarization="dual")
    prop = get_propagator(nonlinear_backend)
    np.testing.assert_allclose(
        prop.propagate(field, replace(lossless, **PLACEMENTS["log-k1"]), dt_s=1e-12),
        prop.propagate(field, lossless, dt_s=1e-12),
        rtol=0,
        atol=1e-13 * np.max(np.abs(field)),
    )


# ---------------------------------------------------------------------------
# Exact solutions of the NLSE / Manakov equation
# ---------------------------------------------------------------------------

_B2, _GAMMA = -2.16e-26, 1.27e-3  # s²/m, 1/(W·m)
_T0 = 5e-12
_LD = _T0**2 / abs(_B2)  # dispersion length ≈ 1.16 km


def _soliton_grid(n: int = 2048, dt_s: float = _T0 / 40) -> tuple[np.ndarray, float]:
    return (np.arange(n) - n // 2) * dt_s / _T0, dt_s


def _satsuma_yajima(tau: np.ndarray, xi: float, p0: float) -> np.ndarray:
    """Second-order soliton of iu_ξ + ½u_ττ + |u|²u = 0 with u(0, τ) = 2 sech τ (period ξ = π/2)."""
    num = 4 * (np.cosh(3 * tau) + 3 * np.exp(4j * xi) * np.cosh(tau)) * np.exp(1j * xi / 2)
    den = np.cosh(4 * tau) + 4 * np.cosh(2 * tau) + 3 * np.cos(4 * xi)
    return np.sqrt(p0) * num / den


@pytest.mark.parametrize("xi", [np.pi / 4, np.pi / 2])
def test_second_order_soliton_matches_the_satsuma_yajima_solution(nonlinear_backend, xi):
    """N = 2 soliton: the full complex field against the closed form, converging as h²."""
    tau, dt = _soliton_grid()
    p0 = abs(_B2) / (_GAMMA * _T0**2)
    a0 = _satsuma_yajima(tau, 0.0, p0)
    np.testing.assert_allclose(a0, 2 * np.sqrt(p0) / np.cosh(tau), rtol=0, atol=1e-12)
    expected = _satsuma_yajima(tau, xi, p0)
    errors = []
    for n_steps in (200, 400, 800):
        fiber = StubFiber(
            length_km=xi * _LD / 1e3,
            beta2_s2_per_m=_B2,
            gamma_per_w_per_km=_GAMMA * 1e3,
            n_steps=n_steps,
        )
        out = get_propagator(nonlinear_backend).propagate(a0, fiber, dt_s=dt)
        errors.append(np.max(np.abs(out - expected)) / np.max(np.abs(a0)))
    assert errors[-1] < 5e-4
    for coarse, fine in itertools.pairwise(errors):
        assert 3.7 < coarse / fine < 4.3  # second order


@pytest.mark.parametrize(("theta", "phi"), [(0.0, 0.0), (np.pi / 4, 0.0), (0.6, 1.1)])
def test_manakov_vector_soliton_keeps_its_shape_polarisation_and_exact_phase(
    nonlinear_backend, theta, phi
):
    """A = √P₀ sech(t/T₀)·(cos θ, sin θ e^{iφ})·e^{iγ_eff P₀ z/2} with (8/9)γP₀T₀² = |β₂|."""
    tau, dt = _soliton_grid()
    p0 = abs(_B2) / (8 / 9 * _GAMMA * _T0**2)
    pol = np.array([np.cos(theta), np.sin(theta) * np.exp(1j * phi)])[:, None]
    a0 = np.sqrt(p0) / np.cosh(tau) * pol
    length_m = 5 * _LD
    fiber = StubFiber(
        length_km=length_m / 1e3,
        beta2_s2_per_m=_B2,
        gamma_per_w_per_km=_GAMMA * 1e3,
        n_steps=800,
        polarization="dual",
    )
    out = get_propagator(nonlinear_backend).propagate(a0, fiber, dt_s=dt)
    expected = a0 * np.exp(1j * (8 / 9) * _GAMMA * p0 * length_m / 2)
    assert np.max(np.abs(out - expected)) / np.sqrt(p0) < 5e-5


def _fwm_idler_ratio(
    backend: str, spacing_bins: int, n_steps: int, placement: dict | None = None
) -> float:
    """Simulated / closed-form idler power of degenerate FWM (pumps at 0 and +Δf, idler at −Δf)."""
    n, fs, p = 512, 1.28e12, 1e-5  # 2.5 GHz bins; 10 µW pumps keep the Kerr phase mismatch ≈ 2e-4
    alpha_db, length_m = 0.2, 50e3
    alpha = alpha_db * np.log(10) / 10 / 1e3
    t = np.arange(n) / fs
    df = spacing_bins * fs / n
    a0 = np.sqrt(p) * (1 + np.exp(2j * np.pi * df * t))
    fiber = StubFiber(
        length_km=length_m / 1e3,
        alpha_db_per_km=alpha_db,
        beta2_s2_per_m=_B2,
        gamma_per_w_per_km=_GAMMA * 1e3,
        n_steps=n_steps,
        **(placement or {}),
    )
    out = get_propagator(backend).propagate(a0, fiber, dt_s=1 / fs)
    idler = np.abs(np.fft.fft(out)[-spacing_bins] / n) ** 2
    # dA₃/dz = −α/2·A₃ + iγA₁²A₂*·e^{iΔβz}, pumps undepleted, Δβ = β₂Δω²:
    # P₃(L) = γ²P₁²P₂·e^{−αL}·|(1 − e^{(−α+iΔβ)L})/(α − iΔβ)|².
    dbeta = abs(_B2) * (2 * np.pi * df) ** 2
    integral = (1 - np.exp((-alpha + 1j * dbeta) * length_m)) / (alpha - 1j * dbeta)
    theory = _GAMMA**2 * p**3 * np.exp(-alpha * length_m) * np.abs(integral) ** 2
    return float(idler / theory)


@pytest.mark.parametrize("spacing_bins", [2, 8, 20])  # 5, 20, 50 GHz: ΔβL ≈ 1, 17, 107
def test_degenerate_fwm_matches_the_closed_form_efficiency(nonlinear_backend, spacing_bins):
    assert abs(_fwm_idler_ratio(nonlinear_backend, spacing_bins, 1600) - 1) < 1e-3


@pytest.mark.parametrize("placement", _placement_params())
def test_spurious_fwm_of_coarse_steps_vanishes_as_h_squared(nonlinear_backend, placement):
    """At 100 GHz (ΔβL ≈ 430) coarse steps misestimate FWM; the error falls ≈ 16× per 4× steps."""
    errors = [
        abs(_fwm_idler_ratio(nonlinear_backend, 40, s, placement) - 1) for s in (400, 1600, 6400)
    ]
    assert errors[0] > 1e-2 and errors[-1] < 1e-3
    for coarse, fine in itertools.pairwise(errors):
        assert coarse / fine > 10


@pytest.mark.parametrize("omega_over_omega_c", [0.3, 0.5, 0.7, 0.96, 1.12, 1.44])
def test_modulation_instability_matches_linear_stability_theory(
    nonlinear_backend, omega_over_omega_c
):
    """CW pump P₀ with weak sidebands at ±Ω (anomalous dispersion, no loss).

    Linearising around the pump, the sideband amplitude u (= v by symmetry)
    obeys x' = (|β₂|Ω²/2)·y, y' = c·x with u = x + iy, c = 2γP₀ − |β₂|Ω²/2:
    x = s·cosh κz, y = c·s·sinh(κz)/κ, κ = g/2 = |β₂Ω|√(Ω_c² − Ω²)/2,
    Ω_c² = 4γP₀/|β₂| — exponential growth inside the band, oscillation outside.
    """
    p0, n, fs, length_m = 0.1, 256, 1e12, 30e3
    omega_c = np.sqrt(4 * _GAMMA * p0 / abs(_B2))
    k = round(omega_over_omega_c * omega_c / (2 * np.pi) / (fs / n))
    omega = 2 * np.pi * k * fs / n
    t = np.arange(n) / fs
    s = 1e-5 * np.sqrt(p0)
    a0 = np.sqrt(p0) + 2 * s * np.cos(omega * t)
    fiber = StubFiber(
        length_km=length_m / 1e3,
        beta2_s2_per_m=_B2,
        gamma_per_w_per_km=_GAMMA * 1e3,
        n_steps=600,
    )
    out = get_propagator(nonlinear_backend).propagate(a0, fiber, dt_s=1 / fs)
    sideband = np.abs(np.fft.fft(out)[k] / n) ** 2
    kappa = np.sqrt(complex(_B2**2 * omega**2 * (omega_c**2 - omega**2))) / 2
    c = 2 * _GAMMA * p0 - abs(_B2) * omega**2 / 2
    x = (s * np.cosh(kappa * length_m)).real
    y = (c * s * np.sinh(kappa * length_m) / kappa).real
    np.testing.assert_allclose(sideband, x**2 + y**2, rtol=1e-3)


# ---------------------------------------------------------------------------
# Cross-implementation checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("placement", _placement_params())
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_agrees_with_numpy_reference(nonlinear_backend, polarization, placement):
    fiber = StubFiber(
        length_km=80.0,
        alpha_db_per_km=0.2,
        beta3_s3_per_m=1.2e-40,
        n_steps=60,
        polarization=polarization,
        **placement,
    )
    shape = (4096,) if polarization == "single" else (2, 4096)
    field = _random_field(shape, 0.1)
    ref = get_propagator("numpy").propagate(field, fiber, dt_s=1e-12)
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=1e-12)
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.max(np.abs(ref)))


def _golden() -> dict[str, np.ndarray]:
    with np.load(GOLDEN) as data:
        return {k: data[k] for k in data.files}


@pytest.mark.parametrize(("polarization", "key"), [("single", "scalar"), ("dual", "manakov")])
def test_matches_ssfm_gpu_golden_output(nonlinear_backend, polarization, key):
    """Golden outputs of the original ``ssfm-gpu`` kernels (the pre-v1 propagator)."""
    g = _golden()
    fiber = StubFiber(
        length_km=float(g["length_km"]),
        alpha_db_per_km=float(g["alpha_db_per_km"]),
        beta2_s2_per_m=float(g["beta2_s2_per_km"]) * 1e-3,
        gamma_per_w_per_km=float(g["gamma"]),
        n_steps=int(g["n_steps"]),
        polarization=polarization,
    )
    out = get_propagator(nonlinear_backend).propagate(
        g[f"field_{key}_in"], fiber, dt_s=float(g["dt_s"])
    )
    expected = g[f"out_{key}"]
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_matches_installed_ssfm_gpu(nonlinear_backend, polarization):
    """Live comparison with the ``ssfm_gpu`` package (skipped if not installed)."""
    pytest.importorskip("ssfm_gpu")
    tf = pytest.importorskip("tensorflow")
    from ssfm_gpu.propagation import tf_fiber_propogate, tf_manakov_fiber_propogate

    fiber = StubFiber(length_km=80.0, alpha_db_per_km=0.2, n_steps=32, polarization=polarization)
    n, dt = 2048, 1.0 / 80e9
    field = _random_field((2, n), 0.03)
    alpha_per_km = 0.2 * np.log(10.0) / 10.0
    args = (n * dt, 80.0, 32, 1.27, fiber.beta2_s2_per_m * 1e3)
    if polarization == "single":
        expected = np.asarray(tf_fiber_propogate(tf.constant(field[0]), *args, alpha=alpha_per_km))
        field = field[0]
    else:
        x, y = tf_manakov_fiber_propogate(
            tf.constant(field[0]), tf.constant(field[1]), *args, alpha=alpha_per_km
        )
        expected = np.stack([np.asarray(x), np.asarray(y)])
    out = get_propagator(nonlinear_backend).propagate(field, fiber, dt_s=dt)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))
