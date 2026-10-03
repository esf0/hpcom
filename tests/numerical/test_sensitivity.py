"""Jacobian singular values of links (``hpcom.autodiff.jacobian_singular_values``).

* the randomized subspace iteration equals the full SVD of the Jacobian;
* a linear link that restores the launch power is unitary: every σ = 1;
* a nonlinear transparent link is symplectic: σᵢ·σ₂ₙ₊₁₋ᵢ = 1 (scalar and
  Manakov), so gradient growth in one direction is shrinkage in another;
* the largest σ grows with launch power.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpcom import EDFA, Fiber, Link  # noqa: E402
from hpcom import autodiff as ad  # noqa: E402

DT = 1 / 64e9


@pytest.fixture(autouse=True)
def _x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


def _link(polarization: str = "single", gamma: float = 1.27, spans: int = 2) -> Link:
    fiber = Fiber.ssmf(80.0, n_steps=20, polarization=polarization).replace(
        gamma_per_w_per_km=gamma
    )
    return Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))


def _field(shape: tuple[int, ...], power_w: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    return x * np.sqrt(power_w / np.mean(np.abs(x) ** 2) / (shape[0] if len(shape) > 1 else 1))


def _full_singular_values(link: Link, x: np.ndarray) -> np.ndarray:
    n = x.size

    def f_real(v: jax.Array) -> jax.Array:
        y = ad.propagate_link((v[:n] + 1j * v[n:]).reshape(x.shape), link, DT).reshape(-1)
        return jnp.concatenate([jnp.real(y), jnp.imag(y)])

    jac = jax.jacfwd(f_real)(jnp.concatenate([x.real.ravel(), x.imag.ravel()]))
    return np.linalg.svd(np.asarray(jac), compute_uv=False)


def test_randomized_values_and_vectors_equal_the_full_svd():
    link, x = _link(), _field((64,), 10e-3)
    full = _full_singular_values(link, x)
    spectrum = ad.jacobian_singular_values(
        lambda a: ad.propagate_link(a, link, DT), jnp.asarray(x), k=8, iterations=60
    )
    np.testing.assert_allclose(np.asarray(spectrum.singular_values)[:4], full[:4], rtol=1e-8)
    # the top vector is amplified by σ₁
    v = spectrum.vectors[0]
    _, jvp = jax.jvp(lambda a: ad.propagate_link(a, link, DT), (jnp.asarray(x),), (v,))
    assert float(jnp.linalg.norm(jvp)) == pytest.approx(full[0], rel=1e-8)
    assert float(jnp.linalg.norm(v)) == pytest.approx(1.0, rel=1e-12)


def test_linear_transparent_link_is_unitary():
    link = _link(gamma=0.0)
    spectrum = ad.jacobian_singular_values(
        lambda a: ad.propagate_link(a, link, DT), jnp.asarray(_field((256,), 1e-3)), k=4
    )
    np.testing.assert_allclose(np.asarray(spectrum.singular_values), 1.0, atol=1e-10)


@pytest.mark.parametrize(
    ("polarization", "shape"), [("single", (64,)), ("dual", (2, 32))], ids=["scalar", "manakov"]
)
def test_nonlinear_transparent_link_is_symplectic(polarization, shape):
    """σᵢ·σ₂ₙ₊₁₋ᵢ = 1 to rounding, while the spectrum itself spreads well away from 1."""
    s = _full_singular_values(_link(polarization), _field(shape, 10e-3))
    assert s[0] > 1.5
    np.testing.assert_allclose(s * s[::-1], 1.0, atol=1e-10)


def test_largest_singular_value_grows_with_power():
    link = _link(spans=3)
    sigma = [
        float(
            ad.jacobian_singular_values(
                lambda a: ad.propagate_link(a, link, DT), jnp.asarray(_field((256,), p)), k=4
            ).singular_values[0]
        )
        for p in (1e-4, 1e-3, 1e-2)
    ]
    assert sigma[0] < sigma[1] < sigma[2]
    assert sigma[0] == pytest.approx(1.0, abs=0.05)
