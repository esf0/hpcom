"""hpcom.autodiff API: parameter pytrees, validation, jit/vmap, float32."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpcom import (  # noqa: E402
    EDFA,
    AdditiveNoise,
    Fiber,
    GainFiber,
    Link,
    LossSpectrum,
    MulticoreFiber,
    RamanResponse,
    Scheme,
)
from hpcom import autodiff as ad  # noqa: E402
from hpcom.multicore import linear_array_coupling  # noqa: E402


@pytest.fixture(autouse=True)
def _x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _x(shape: tuple[int, ...] = (256,), scale: float = 0.1) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * scale


GNLSE = Fiber(
    length_km=0.5,
    higher_order_betas=(2e-55, -1e-69),
    raman=RamanResponse(model="lin-agrawal"),
    self_steepening=True,
    shock_time_s=0.7e-15,
    scheme="rk4ip",
    n_steps=20,
).with_loss_spectrum(LossSpectrum(wavelengths_m=[1.5e-6, 1.6e-6], alpha_db_per_km=[0.3, 0.2]))
LASER = GainFiber(
    length_km=4e-3,
    beta2_s2_per_m=-2e-26,
    gain_db_per_m=1.5,
    gain_bandwidth_hz=3e12,
    gain_shape="parabolic",
    saturation_energy_j=5e-13,
    nonlinear_gain_per_w_per_km=-50.0,
    n_steps=12,
)
MULTICORE = MulticoreFiber.from_fiber(
    Fiber.ssmf(5.0, n_steps=10),
    linear_array_coupling(3, 1e-3),
    core_beta0_offsets_per_m=[0.0, 1e-3, -1e-3],
    core_beta1_offsets_s_per_m=[0.0, 2e-14, -1e-14],
)


@pytest.mark.parametrize(
    "fiber", [Fiber.ssmf(80.0), GNLSE, MULTICORE, LASER], ids=["ssmf", "gnlse", "mcf", "laser"]
)
def test_parameters_round_trip_through_the_fiber(fiber):
    params = ad.FiberParameters.from_fiber(fiber)
    back = params.to_fiber(fiber)
    assert type(back) is type(fiber)
    np.testing.assert_allclose(back.dispersion_betas, fiber.dispersion_betas, rtol=1e-14)
    assert back.gamma_per_w_per_km == pytest.approx(fiber.gamma_per_w_per_km, rel=1e-15)
    assert back.alpha_db_per_km == pytest.approx(fiber.alpha_db_per_km, rel=1e-15)
    assert back.effective_shock_time_s == pytest.approx(fiber.effective_shock_time_s, rel=1e-14)
    if fiber.raman is not None:
        assert back.raman.tau_b_s == pytest.approx(fiber.raman.tau_b_s, rel=1e-14)
    if isinstance(fiber, MulticoreFiber):
        np.testing.assert_allclose(
            back.core_beta1_offsets_s_per_m, fiber.core_beta1_offsets_s_per_m, rtol=1e-14
        )
    if isinstance(fiber, GainFiber):
        for name in (
            "gain_db_per_m",
            "gain_bandwidth_hz",
            "saturation_energy_j",
            "nonlinear_gain_per_w_per_km",
        ):
            assert getattr(back, name) == pytest.approx(getattr(fiber, name), rel=1e-14)


def test_units_keep_every_parameter_o1():
    """ps/km units: SSMF β₂ = −21.7 ps²/km, β₃ = 0.13 ps³/km — representable in float32."""
    params = ad.FiberParameters.from_fiber(Fiber.ssmf(1.0).replace(beta3_s3_per_m=1.3e-40))
    np.testing.assert_allclose(params.betas_psn_per_km, [0.0, -21.68, 0.13], atol=0.01)
    assert params.raman_fraction is None and params.coupling_per_km is None


def test_params_must_match_the_fibers_terms():
    params = ad.FiberParameters.from_fiber(Fiber.ssmf(1.0))
    with pytest.raises(ValueError, match="structure"):
        ad.propagate(jnp.asarray(_x()), GNLSE, 1e-13, params=params)


@pytest.mark.parametrize(
    ("fiber", "kwargs", "error", "match"),
    [
        (
            Fiber.ssmf(1.0).replace(scheme=Scheme(method="ssfm-local-error", local_error=1e-5)),
            {},
            NotImplementedError,
            "not differentiable",
        ),
        (GNLSE, {"gradient": "reversible"}, ValueError, "invertible"),
        (LASER, {"gradient": "reversible"}, ValueError, "saturated-gain"),
        (Fiber.ssmf(1.0), {"gradient": "adjoint"}, ValueError, "gradient must be"),
        (Fiber.ssmf(1.0), {"direction": "sideways"}, ValueError, "direction"),
    ],
)
def test_unsupported_requests_are_rejected(fiber, kwargs, error, match):
    with pytest.raises(error, match=match):
        ad.propagate(jnp.asarray(_x()), fiber, 1e-12, **kwargs)


def test_field_shape_is_checked():
    with pytest.raises(ValueError, match="dual-polarisation"):
        ad.propagate(jnp.asarray(_x()), Fiber.ssmf(1.0, polarization="dual"), 1e-12)


def test_jit_with_a_static_fiber_and_vmap_over_parameters():
    fiber = Fiber.ssmf(20.0, n_steps=10)
    x = jnp.asarray(_x(scale=0.3))
    params = ad.FiberParameters.from_fiber(fiber)
    run = jax.jit(ad.propagate, static_argnames=("fiber", "dt_s"))
    np.testing.assert_allclose(run(x, fiber=fiber, dt_s=1e-11), ad.propagate(x, fiber, 1e-11))
    gammas = jnp.array([0.5, 1.27, 2.0])
    batched = jax.vmap(
        lambda g: ad.propagate(x, fiber, 1e-11, params=params._replace(gamma_per_w_per_km=g))
    )(gammas)
    for g, out in zip(gammas, batched, strict=True):
        ref = fiber.replace(gamma_per_w_per_km=float(g)).propagate(np.asarray(x), 1e-11)
        np.testing.assert_allclose(np.asarray(out), ref, atol=1e-12)


@pytest.mark.parametrize(
    ("fiber", "dt", "scale", "tol"),
    [
        (Fiber.ssmf(80.0, n_steps=40), 1e-11, 0.3, 2e-4),
        (GNLSE, 2e-14, 1.0, 1e-2),
        (LASER, 2e-14, 1.0, 1e-4),
    ],
    ids=["ssmf-100ghz", "gnlse-25thz", "laser-25thz"],
)
def test_complex64_runs_in_single_precision(fiber, dt, scale, tol):
    """float32 resolves a phase φ to ~1e-7·φ: the band-edge dispersion phase per step
    (≈ 1 rad at 100 GHz over 2 km; ≈ 6e3 rad at 25 THz over 25 m) sets the accuracy."""
    x64 = _x((1024,), scale)
    ref = fiber.propagate(x64, dt)
    out = ad.propagate(jnp.asarray(x64, jnp.complex64), fiber, dt)
    assert out.dtype == jnp.complex64
    assert np.max(np.abs(np.asarray(out) - ref)) < tol * np.max(np.abs(ref))
    grad = jax.grad(
        lambda p: jnp.sum(
            jnp.abs(ad.propagate(jnp.asarray(x64, jnp.complex64), fiber, dt, params=p)) ** 4
        )
    )(
        jax.tree_util.tree_map(
            lambda v: jnp.asarray(v, jnp.float32), ad.FiberParameters.from_fiber(fiber)
        )
    )
    assert all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves(grad))


def test_link_parameters_cover_fibers_and_amplifiers_only():
    fiber = Fiber.ssmf(50.0)
    link = Link.from_spans(2, fiber, EDFA(gain_db=10.0))
    params = ad.link_parameters(link)
    assert isinstance(params[0], ad.FiberParameters) and isinstance(
        params[1], ad.AmplifierParameters
    )
    assert float(params[1].noise_figure_db) == 5.0
    assert (
        ad.AmplifierParameters.from_amplifier(
            EDFA(gain_db=10.0, noise_figure_db=None)
        ).noise_figure_db
        is None
    )
    with pytest.raises(TypeError, match="no differentiable model"):
        ad.link_parameters(
            Link(elements=(fiber, AdditiveNoise(psd_per_polarization_w_per_hz=1e-20)))
        )


def test_noisy_link_needs_a_key_and_is_reproducible():
    link = Link.from_spans(2, Fiber.ssmf(50.0, n_steps=10), EDFA(gain_db=10.0))
    x = jnp.asarray(_x())
    with pytest.raises(ValueError, match="PRNG key"):
        ad.propagate_link(x, link, 1e-11)
    a = ad.propagate_link(x, link, 1e-11, key=jax.random.key(3))
    b = ad.propagate_link(x, link, 1e-11, key=jax.random.key(3))
    c = ad.propagate_link(x, link, 1e-11, key=jax.random.key(4))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    assert np.max(np.abs(np.asarray(a) - np.asarray(c))) > 0
