"""Learned DBP (``hpcom.autodiff.learned_dbp``): exact start, exact gradients, it learns.

* with its initial parameters it is standard DBP (``backpropagate``) for
  scalar and Manakov fields, uniform and logarithmic steps, a channel at an
  offset, with and without power filters;
* gradients equal finite differences for every kind of parameter;
* trained by gradient descent at one step per span it removes much more of
  the nonlinearity than standard DBP with the same steps.
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


def _link(polarization: str = "single", spans: int = 3) -> Link:
    fiber = Fiber.ssmf(80.0, n_steps=10, polarization=polarization).replace(beta3_s3_per_m=1e-40)
    return Link.from_spans(spans, fiber, EDFA(gain_db=fiber.loss_db + 0.3, noise_figure_db=None))


def _field(shape: tuple[int, ...], scale: float = 0.05, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * scale


@pytest.mark.parametrize("taps", [0, 4])
@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps_per_span": 4},
        {"steps_per_span": 2, "step_distribution": "logarithmic", "step_log_factor": 1 / 3},
        {"steps_per_span": 3, "frequency_offset_hz": 3e9},
    ],
    ids=["uniform", "logarithmic", "offset"],
)
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_initial_parameters_are_standard_dbp(polarization, kwargs, taps):
    link = _link(polarization)
    shape = (2, 1024) if polarization == "dual" else (1024,)
    x = jnp.asarray(_field(shape))
    ref = ad.backpropagate(x, link, DT, **kwargs)
    structure, params = ad.learned_dbp_init(link, shape[-1], DT, intensity_taps=taps, **kwargs)
    out = ad.learned_dbp(x, structure, params)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(ref), atol=1e-12 * float(jnp.max(jnp.abs(ref)))
    )
    assert params.phase.shape == (structure.base.shape[0], 33)
    assert params.intensity.shape == (structure.base.shape[0] - 1, 2 * taps + 1)


def test_gradients_equal_finite_differences():
    link = _link(spans=2)
    x = jnp.asarray(_field((512,), 0.1))
    structure, params = ad.learned_dbp_init(
        link, 512, DT, steps_per_span=2, intensity_taps=2, knots=9
    )
    rng = np.random.default_rng(1)
    params = jax.tree_util.tree_map(
        lambda v: v + 0.01 * rng.standard_normal(np.shape(v)) * (np.abs(v).max() + 1.0), params
    )
    weight = jnp.asarray(_field((512,), 1.0, seed=2))

    def loss(p: ad.LearnedDBPParameters) -> jax.Array:
        y = ad.learned_dbp(x, structure, p)
        return jnp.sum(jnp.abs(y) ** 4 * jnp.abs(weight)) + jnp.sum(jnp.real(jnp.conj(weight) * y))

    grads = jax.grad(loss)(params)
    for name, index in (
        ("phase", (1, 3)),
        ("log_gain", (2, 5)),
        ("nonlinear", (1,)),
        ("intensity", (2, 1)),
    ):
        leaf = np.asarray(getattr(params, name))
        h = 1e-6 * max(abs(leaf[index]), 1.0)

        def at(
            dx: float, name: str = name, index: tuple[int, ...] = index, leaf: np.ndarray = leaf
        ) -> float:
            moved = leaf.copy()
            moved[index] += dx
            return float(loss(params._replace(**{name: jnp.asarray(moved)})))

        numeric = (at(h) - at(-h)) / (2 * h)
        assert float(np.asarray(getattr(grads, name))[index]) == pytest.approx(
            numeric, rel=1e-5, abs=1e-9
        )


def test_training_at_one_step_per_span_beats_standard_dbp():
    """Noiseless 5 × 80 km, one DBP step per span: Adam on the mean-square error to the sent field."""
    fiber = Fiber.ssmf(80.0, n_steps=40)
    link = Link.from_spans(5, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=None))
    n = 1024
    spectrum = np.fft.fft(_field((n,), 1.0, seed=3))
    spectrum[np.abs(np.fft.fftfreq(n)) > 0.25] = 0  # 32 GHz of the 64 GHz grid
    sent = np.fft.ifft(spectrum)
    sent *= np.sqrt(4e-3 / np.mean(np.abs(sent) ** 2))  # 6 dBm: strongly nonlinear
    received = jnp.asarray(link.propagate(sent, DT))
    structure, params = ad.learned_dbp_init(link, n, DT, steps_per_span=1, intensity_taps=4)

    def error(p: ad.LearnedDBPParameters) -> jax.Array:
        y = ad.learned_dbp(received, structure, p)
        return jnp.mean(jnp.abs(y - sent) ** 2) / jnp.mean(jnp.abs(sent) ** 2)

    value_and_grad = jax.jit(jax.value_and_grad(error))
    start = float(error(params))
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    lr = {"phase": 1e-2, "log_gain": 1e-3, "nonlinear": 2e-2, "intensity": 1e-2}
    for t in range(1, 301):
        _, g = value_and_grad(params)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b**2, v, g)
        params = params._replace(
            **{
                k: getattr(params, k)
                - lr[k]
                * (np.abs(getattr(params, k)).max() if k == "nonlinear" else 1.0)
                * (getattr(m, k) / (1 - 0.9**t))
                / (jnp.sqrt(getattr(v, k) / (1 - 0.999**t)) + 1e-12)
                for k in lr
            }
        )
    end = float(error(params))
    assert 10 * np.log10(start / end) > 6.0  # the error falls by more than 6 dB
