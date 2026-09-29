"""Differentiable propagation (``hpcom.autodiff``): values, gradients, memory, applications.

* forward values equal the NumPy reference for every scheme, term and layout;
* gradients equal Richardson-extrapolated finite differences for every
  parameter class, and closed-form derivatives where they exist;
* the three gradient modes agree, and the reversible one needs no memory
  per step;
* gradient-based fits recover fiber parameters and the optimal DBP scale.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from scipy.optimize import minimize  # noqa: E402

from hpcom import (  # noqa: E402
    EDFA,
    DigitalBackPropagation,
    Fiber,
    Link,
    LossSpectrum,
    MulticoreFiber,
    RamanResponse,
    Scheme,
)
from hpcom import autodiff as ad  # noqa: E402
from hpcom.multicore import hexagonal_coupling, ring_coupling  # noqa: E402


@pytest.fixture(autouse=True)
def _x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _field(shape: tuple[int, ...], scale: float, seed: int = 0, band: float = 0.25) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    spectrum = np.fft.fft(x, axis=-1)
    spectrum[..., np.abs(np.fft.fftfreq(shape[-1])) > band] = 0
    return np.fft.ifft(spectrum, axis=-1) * scale


_SSMF = Fiber.ssmf(30.0, n_steps=12).replace(beta3_s3_per_m=1.2e-40)
_SPECTRUM = LossSpectrum(wavelengths_m=[1.50e-6, 1.60e-6], alpha_db_per_km=[0.26, 0.17])

#: name → (fiber, field shape, dt): every scheme, term and layout autodiff supports.
CASES: dict[str, tuple[Fiber, tuple[int, ...], float]] = {
    "ssfm": (_SSMF, (512,), 2e-12),
    "manakov": (_SSMF.replace(polarization="dual"), (2, 512), 2e-12),
    "kerr-position-effective-log": (
        _SSMF.replace(
            scheme=Scheme(nonlinear_position=0.2, kerr_length="effective"),
            step_distribution="logarithmic",
            step_log_factor=1 / 3,
        ),
        (512,),
        2e-12,
    ),
    "filtered": (_SSMF.replace(scheme=Scheme(kerr_filter_bandwidth_hz=60e9)), (512,), 2e-12),
    "yoshida": (_SSMF.replace(scheme="ssfm4-yoshida", n_steps=4), (512,), 2e-12),
    "suzuki-gnlse": (
        _SSMF.replace(
            scheme="ssfm4-suzuki",
            n_steps=4,
            higher_order_betas=(-3e-55, 2e-69),
            raman=RamanResponse(model="lin-agrawal"),
        ).with_loss_spectrum(_SPECTRUM),
        (512,),
        2e-13,
    ),
    "rk4ip-raman-shock": (
        _SSMF.replace(
            scheme="rk4ip", raman=RamanResponse(), self_steepening=True, shock_time_s=5e-14
        ),
        (512,),
        2e-13,
    ),
    "manakov-pmd": (
        _SSMF.replace(polarization="dual").with_pmd(0.5, sections=40, seed=7),
        (2, 512),
        2e-12,
    ),
    "rk4ip-pmd": (
        _SSMF.replace(polarization="dual", scheme="rk4ip").with_pmd(0.5, sections=40, seed=8),
        (2, 512),
        2e-12,
    ),
    "multicore-ring": (
        MulticoreFiber.from_fiber(
            _SSMF, ring_coupling(3, 2e-3), core_beta0_offsets_per_m=[0, 1e-3, -2e-3]
        ),
        (3, 512),
        2e-12,
    ),
    "multicore-skew-dual": (
        MulticoreFiber.from_fiber(
            _SSMF.replace(polarization="dual"),
            hexagonal_coupling(1e-3),
            core_beta1_offsets_s_per_m=np.linspace(-1e-13, 1e-13, 7),
        ),
        (7, 2, 256),
        2e-12,
    ),
}


def _case(name: str) -> tuple[Fiber, np.ndarray, float]:
    fiber, shape, dt = CASES[name]
    return fiber, _field(shape, 0.12 if len(shape) == 1 else 0.06), dt


def _modes(fiber: Fiber) -> tuple[str, ...]:
    return ("store", "checkpoint") + (() if fiber.scheme.method == "rk4ip" else ("reversible",))


# ---------------------------------------------------------------------------
# Forward values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("direction", ["forward", "backward"])
@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_values_equal_the_numpy_reference(name, direction):
    """To 1e-10: the band-edge dispersion phase reaches ~1e5 rad, and computing it in ps/km
    units instead of s/m changes its last bits."""
    fiber, x, dt = _case(name)
    ref = fiber.propagate(x, dt, direction=direction)
    for mode in _modes(fiber):
        out = np.asarray(
            ad.propagate(jnp.asarray(x), fiber, dt, direction=direction, gradient=mode)
        )
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10 * np.max(np.abs(ref)))


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def _loss_fn(fiber: Fiber, x: np.ndarray, dt: float, mode: str) -> Callable:
    """A smooth real loss that weighs every sample differently."""
    weight = jnp.asarray(_field(x.shape, 1.0, seed=7))

    def loss(params: ad.FiberParameters) -> jax.Array:
        y = ad.propagate(jnp.asarray(x), fiber, dt, params=params, gradient=mode)
        return jnp.sum(jnp.abs(y) ** 4 * jnp.abs(weight)) + jnp.sum(jnp.real(jnp.conj(weight) * y))

    return loss


def _richardson(f: Callable[[float], float], h: float) -> float:
    """Central difference with Richardson extrapolation (error O(h⁴))."""
    d1 = (f(h) - f(-h)) / (2 * h)
    d2 = (f(h / 2) - f(-h / 2)) / h
    return (4 * d2 - d1) / 3


def _finite_difference(f: Callable[[float], float], scale: float) -> tuple[float, float]:
    """Richardson differences at steps scale·10⁻³…10⁻⁸; the most stable adjacent pair.

    Returns the estimate and its spread (truncation dominates at large
    steps, rounding at small ones; the plateau between is the derivative).
    """
    estimates = [_richardson(f, scale * 10.0**-k) for k in range(3, 9)]
    spreads = [abs(a - b) for a, b in itertools.pairwise(estimates)]
    best = int(np.argmin(spreads))
    return estimates[best + 1], spreads[best]


@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_gradients_equal_finite_differences_for_every_parameter(name):
    fiber, x, dt = _case(name)
    params = ad.FiberParameters.from_fiber(fiber)
    loss = jax.jit(_loss_fn(fiber, x, dt, "store"))
    grads = jax.grad(loss)(params)
    leaves, tree = jax.tree_util.tree_flatten(params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(grad_leaves) == len(leaves)
    for i, leaf in enumerate(leaves):
        flat = np.asarray(leaf, dtype=np.float64).ravel()
        for j in range(min(flat.size, 4)):

            def at(
                dx: float, i: int = i, j: int = j, flat: np.ndarray = flat, leaf: object = leaf
            ) -> float:
                moved = flat.copy()
                moved[j] += dx
                new = list(leaves)
                new[i] = jnp.asarray(moved.reshape(np.shape(leaf)))
                return float(loss(jax.tree_util.tree_unflatten(tree, new)))

            numeric, spread = _finite_difference(at, max(abs(flat[j]), 1e-2))
            exact = float(np.asarray(grad_leaves[i]).ravel()[j])
            assert exact == pytest.approx(numeric, rel=1e-6, abs=3 * spread), (i, j)


@pytest.mark.parametrize("name", list(CASES), ids=list(CASES))
def test_gradient_modes_agree(name):
    """Stored, checkpointed and reversible (inverse-step) adjoints: the same gradient."""
    fiber, x, dt = _case(name)
    params = ad.FiberParameters.from_fiber(fiber)
    ref = None
    for mode in _modes(fiber):
        g = jax.grad(_loss_fn(fiber, x, dt, mode))(params)
        leaves = [np.asarray(v) for v in jax.tree_util.tree_leaves(g)]
        if ref is None:
            ref = leaves
            continue
        for a, b in zip(leaves, ref, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12 * np.max(np.abs(b)))


def test_gradient_with_respect_to_the_input_field():
    """∇_x of Re⟨c, U x⟩ for the (linear, unitary) lossless dispersive fiber is U†c.

    JAX's gradient of a real function of a complex array is ∂/∂Re − i∂/∂Im,
    here conj(U†c).
    """
    fiber = Fiber(length_km=50.0, alpha_db_per_km=0.0, gamma_per_w_per_km=0.0, n_steps=3)
    n, dt = 256, 5e-12
    c = jnp.asarray(_field((n,), 1.0, seed=3))
    g = jax.grad(lambda x: jnp.real(jnp.vdot(c, ad.propagate(x, fiber, dt))))(
        jnp.asarray(_field((n,), 0.1))
    )
    expected = fiber.propagate(np.asarray(c), dt, direction="backward")  # U† = U⁻¹
    np.testing.assert_allclose(np.asarray(g), np.conj(expected), atol=1e-12)


def test_gamma_gradient_of_self_phase_modulation_is_exact():
    """Without dispersion y = x·exp(iγL_eff|x|²) — exactly, with the effective Kerr length."""
    fiber = Fiber(
        length_km=40.0, beta2_s2_per_m=0.0, n_steps=5, scheme=Scheme(kerr_length="effective")
    )
    x = jnp.asarray(_field((128,), 0.3))
    c = jnp.asarray(_field((128,), 1.0, seed=5))
    leff = fiber.effective_length_km
    loss = 10 ** (-fiber.loss_db / 20)
    y = loss * x * jnp.exp(1j * fiber.gamma_per_w_per_km * leff * jnp.abs(x) ** 2)
    exact = float(jnp.real(jnp.vdot(c, 1j * leff * jnp.abs(x) ** 2 * y)))
    params = ad.FiberParameters.from_fiber(fiber)
    grad = jax.grad(lambda p: jnp.real(jnp.vdot(c, ad.propagate(x, fiber, 1e-11, params=p))))(
        params
    )
    assert float(grad.gamma_per_w_per_km) == pytest.approx(exact, rel=1e-12)


def test_beta2_gradient_matches_the_gaussian_closed_form():
    """d/dβ₂ of |A(t,z)|² at t = 0 for A = T₀/√(T₀² − iβ₂z)·e^{−t²/2(T₀² − iβ₂z)}: −... in closed form."""
    t0, length_km, n, dt = 20e-12, 40.0, 4096, 0.5e-12
    fiber = Fiber(length_km=length_km, alpha_db_per_km=0.0, gamma_per_w_per_km=0.0, n_steps=2)
    t = (np.arange(n) - n // 2) * dt
    x = jnp.asarray(np.fft.ifftshift(np.exp(-(t**2) / (2 * t0**2))) + 0j)
    params = ad.FiberParameters.from_fiber(fiber)
    grad = jax.grad(lambda p: jnp.abs(ad.propagate(x, fiber, dt, params=p)[0]) ** 2)(params)
    # |A(0,z)|² = T₀²/|T₀² − iβ₂z| = T₀²/√(T₀⁴ + β₂²z²); in ps²/km units for β₂:
    b2 = float(params.betas_psn_per_km[1])
    t0_ps, z = t0 / 1e-12, length_km
    exact = -(t0_ps**2) * b2 * z**2 / (t0_ps**4 + b2**2 * z**2) ** 1.5
    assert float(grad.betas_psn_per_km[1]) == pytest.approx(exact, rel=1e-9)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_reversible_gradient_memory_does_not_grow_with_the_step_count():
    n, dt = 2**13, 1e-12
    x = jnp.asarray(_field((n,), 0.1))
    temp = {}
    for steps in (40, 320):
        fiber = Fiber.ssmf(80.0, n_steps=steps)
        params = ad.FiberParameters.from_fiber(fiber)
        for mode in ("store", "checkpoint", "reversible"):
            f = jax.jit(
                jax.grad(
                    lambda p, fiber=fiber, mode=mode: jnp.sum(
                        jnp.abs(ad.propagate(x, fiber, dt, params=p, gradient=mode)) ** 4
                    )
                )
            )
            analysis = f.lower(params).compile().memory_analysis()
            if analysis is None:
                pytest.skip("this JAX backend reports no memory analysis")
            temp[mode, steps] = analysis.temp_size_in_bytes
    field_bytes = n * 16
    # (only the per-step scalars — Kerr phases, lengths — grow with the step count)
    assert temp["reversible", 320] < temp["reversible", 40] + 64 * 320 < 20 * field_bytes
    assert temp["store", 320] > 6 * temp["store", 40] > 40 * field_bytes
    assert temp["checkpoint", 320] < temp["store", 320] / 10


# ---------------------------------------------------------------------------
# Links, amplifiers, DBP
# ---------------------------------------------------------------------------


def _link(polarization: str = "single", noise_figure_db: float | None = None) -> Link:
    fiber = Fiber.ssmf(60.0, n_steps=20, polarization=polarization).replace(beta3_s3_per_m=1.2e-40)
    return Link.from_spans(3, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=noise_figure_db))


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_noiseless_link_equals_link_propagate(polarization):
    link = _link(polarization)
    shape = (1024,) if polarization == "single" else (2, 1024)
    x = _field(shape, 0.05)
    ref = link.propagate(x, 1e-11)
    out = np.asarray(ad.propagate_link(jnp.asarray(x), link, 1e-11))
    np.testing.assert_allclose(out, ref, atol=1e-12 * np.max(np.abs(ref)))


def test_amplifier_noise_has_the_edfa_variance_and_is_reparameterised():
    amp = EDFA(gain_db=20.0, noise_figure_db=5.0)
    dt = 1e-11
    zeros = jnp.zeros((2, 200_000), jnp.complex128)
    noise = np.asarray(ad.amplify(zeros, amp, dt, key=jax.random.key(0)))
    assert np.var(noise) == pytest.approx(amp.ase_variance_per_sample(dt), rel=0.01)
    assert abs(np.mean(noise.real * noise.imag)) < 0.01 * np.var(noise)
    with pytest.raises(ValueError, match="PRNG key"):
        ad.amplify(zeros, amp, dt)
    # gradient of the mean noise power with respect to the noise figure (reparameterised)
    params = ad.AmplifierParameters.from_amplifier(amp)
    g = jax.grad(
        lambda p: jnp.mean(
            jnp.abs(ad.amplify(zeros, amp, dt, key=jax.random.key(1), params=p)) ** 2
        )
    )(params)
    power = float(jnp.mean(jnp.abs(ad.amplify(zeros, amp, dt, key=jax.random.key(1))) ** 2))
    gain, nf = amp.gain_linear, amp.noise_figure_linear
    dpower_dnf_db = power * gain * nf * math.log(10) / 10 / (gain * nf - 1)
    assert float(g.noise_figure_db) == pytest.approx(dpower_dnf_db, rel=1e-10)


@pytest.mark.parametrize("offset_bins", [0, 37])
@pytest.mark.parametrize("scale", [1.0, 0.6])
def test_backpropagate_equals_digital_backpropagation(offset_bins, scale):
    link = _link("dual")
    n, dt = 1024, 1e-11
    x = _field((2, n), 0.05, band=0.05)
    f_k = offset_bins / (n * dt)
    dbp = DigitalBackPropagation(
        link=link,
        steps_per_span=3,
        samples_per_symbol=None,
        nonlinearity_scale=scale,
        scheme=Scheme(kerr_length="effective"),
        step_distribution="logarithmic",
        step_log_factor=1 / 3,
    )
    ref = dbp.compensate(x, dt, frequency_offset_hz=f_k)
    out = np.asarray(ad.compensate(jnp.asarray(x), dbp, dt, frequency_offset_hz=f_k))
    np.testing.assert_allclose(out, ref, atol=1e-12 * np.max(np.abs(ref)))


# ---------------------------------------------------------------------------
# Applications
# ---------------------------------------------------------------------------


def test_fiber_parameters_are_recovered_from_input_and_output():
    """System identification: fit (α, β₂, β₃, γ) to a measured output by L-BFGS on exact gradients."""
    truth = Fiber.ssmf(20.0, n_steps=40).replace(beta3_s3_per_m=1.2e-40)
    n, dt = 512, 5e-12  # 200 GHz: band-edge dispersion phase ≈ 40 rad, one basin
    x = jnp.asarray(_field((n,), 0.3))
    measured = ad.propagate(x, truth, dt)
    start = ad.FiberParameters.from_fiber(truth)
    true_vec = np.array(
        [
            float(start.alpha_db_per_km),
            *np.asarray(start.betas_psn_per_km)[1:],
            float(start.gamma_per_w_per_km),
        ]
    )
    scale = true_vec  # signed: v = 1 is the truth

    def unpack(v: jax.Array) -> ad.FiberParameters:
        u = v * scale
        return start._replace(
            alpha_db_per_km=u[0],
            betas_psn_per_km=jnp.concatenate([start.betas_psn_per_km[:1], u[1:3]]),
            gamma_per_w_per_km=u[3],
        )

    @jax.jit
    def objective(v: jax.Array) -> jax.Array:
        y = ad.propagate(x, truth, dt, params=unpack(v))
        return jnp.sum(jnp.abs(y - measured) ** 2) / jnp.sum(jnp.abs(measured) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(objective))
    guess = np.array([1.1, 0.97, 1.3, 0.9])  # 3–30 % off

    def fun(v: np.ndarray) -> tuple[float, np.ndarray]:
        val, grad = value_and_grad(jnp.asarray(v))
        return float(val), np.asarray(grad, dtype=np.float64)

    result = minimize(
        fun, guess, jac=True, method="L-BFGS-B", options={"gtol": 1e-14, "ftol": 1e-30}
    )
    np.testing.assert_allclose(result.x, np.ones(4), rtol=1e-6)


def test_learned_nonlinearity_scale_matches_the_grid_search_optimum():
    """One-step-per-span DBP: gradient descent on ξ lands on the minimum of the MSE curve."""
    link = _link("single")
    n, dt = 2048, 1 / (32e9 * 4)
    x = _field((n,), math.sqrt(4e-3), band=0.15)
    y = link.propagate(x, dt)
    dbp = DigitalBackPropagation(link=link, steps_per_span=1, samples_per_symbol=None)

    def mse(xi: jax.Array) -> jax.Array:
        z = ad.compensate(jnp.asarray(y), dbp, dt, nonlinearity_scale=xi)
        return jnp.mean(jnp.abs(z - x) ** 2)

    grid = np.linspace(0.5, 2.0, 151)
    curve = [float(mse(jnp.asarray(xi))) for xi in grid]
    best = grid[int(np.argmin(curve))]
    value_and_grad = jax.jit(jax.value_and_grad(mse))
    result = minimize(
        lambda v: tuple(np.asarray(o, np.float64) for o in value_and_grad(jnp.asarray(v[0]))),
        np.array([1.0]),
        jac=True,
        method="L-BFGS-B",
    )
    assert result.x[0] == pytest.approx(best, abs=0.011)
    assert result.fun <= min(curve) + 1e-15
