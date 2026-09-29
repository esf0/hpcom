"""Differentiable transceivers (``hpcom.autodiff``): equivalence, gradients, end-to-end learning.

* :func:`modulate` / :func:`demodulate` equal the NumPy transmitter and
  receiver (pulse modes, pilots, WDM, dual polarisation, DBP stage);
* :func:`information_rates` equals :func:`hpcom.metrics.information_rates`;
* gradients through the whole chain equal finite differences;
* learning a constellation (geometric) or a PMF (probabilistic, Gumbel)
  beats square QAM when scored by the NumPy pipeline on fresh data.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from hpcom import (  # noqa: E402
    EDFA,
    AdditiveNoise,
    ConstellationConfig,
    DigitalBackPropagation,
    Fiber,
    Link,
    Simulation,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)
from hpcom import autodiff as ad  # noqa: E402
from hpcom.metrics import information_rates  # noqa: E402
from hpcom.pilots import PilotConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _x64() -> Iterator[None]:
    with jax.enable_x64(True):
        yield


_QAM64 = ConstellationConfig.from_square_qam(64)
_MB = np.exp(-1.5 * np.abs(_QAM64.points) ** 2)

TRANSMITTERS = {
    "sc": SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=32e9, n_symbols=512, launch_power_per_channel_dbm=3.0
    ),
    "dual-pilots-fir": SingleCarrierTransmitter(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=512,
        polarization="dual",
        pilots=PilotConfig(period=8),
        pulse_span_symbols=32,
    ),
    "linear-fir": SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=512,
        pulse_span_symbols=16,
        convolution="linear",
    ),
    "wdm-dual": WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=512,
        polarization="dual",
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
    ),
    "probabilistic": SingleCarrierTransmitter(
        constellation=_QAM64.with_probabilities(_MB / _MB.sum()),
        symbol_rate_hz=32e9,
        n_symbols=512,
    ),
}


def _link(tx) -> Link:
    fiber = Fiber.ssmf(80.0, n_steps=20, polarization=tx.polarization)
    return Link.from_spans(2, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))


@pytest.mark.parametrize("name", list(TRANSMITTERS), ids=list(TRANSMITTERS))
def test_modulate_equals_the_transmitter(name):
    tx = TRANSMITTERS[name]
    sig = tx.generate(rng=np.random.default_rng(1))
    field = ad.modulate(jnp.asarray(sig.data_symbols()), tx)
    np.testing.assert_allclose(np.asarray(field), sig.field, atol=1e-14 * np.max(np.abs(sig.field)))


def _receivers(tx, link: Link) -> dict[str, object]:
    out = {
        "cdc": tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2),
        "dbp-2sps": tx.matched_receiver(
            compensation=DigitalBackPropagation(link=link, steps_per_span=3, samples_per_symbol=2)
        ),
        "dbp-full-field": tx.matched_receiver(
            compensation=DigitalBackPropagation(
                link=link, steps_per_span=3, samples_per_symbol=None
            )
        ),
        "unit-power": tx.matched_receiver(
            cd_compensation_s2=link.total_dispersion_s2, equalize="unit_power"
        ),
    }
    if isinstance(tx, WDMTransmitter):
        out["subset-band-limited"] = tx.matched_receiver(
            channels=(0, 2),
            compensation=DigitalBackPropagation(
                link=link, steps_per_span=3, samples_per_symbol=2, bandwidth_hz=40e9
            ),
        )
    return out


@pytest.mark.parametrize("name", list(TRANSMITTERS), ids=list(TRANSMITTERS))
def test_demodulate_equals_the_receiver(name):
    tx = TRANSMITTERS[name]
    link = _link(tx)
    sig = tx.generate(rng=np.random.default_rng(2))
    y = link.propagate(sig.field, tx.dt_s, rng=np.random.default_rng(3))
    for label, rx in _receivers(tx, link).items():
        ref = rx.reference(sig)[1]
        expected = rx.receive(y, tx.dt_s, reference=sig).symbols
        out = ad.demodulate(jnp.asarray(y), rx, tx.dt_s, reference=jnp.asarray(ref))
        np.testing.assert_allclose(
            np.asarray(out), expected, atol=1e-12 * np.max(np.abs(expected)), err_msg=label
        )


@pytest.mark.parametrize(
    ("config", "snr"),
    [
        (ConstellationConfig.from_square_qam(16), 9.0),
        (_QAM64.with_probabilities(_MB / _MB.sum()), 14.0),
        (ConstellationConfig.from_square_qam(256), 20.0),
    ],
    ids=["16qam", "ps-64qam", "256qam"],
)
def test_information_rates_equal_the_numpy_metrics(config, snr):
    rng = np.random.default_rng(4)
    idx = rng.choice(config.modulation_order, size=20_000, p=config.pmf)
    noise = (rng.standard_normal(idx.size) + 1j * rng.standard_normal(idx.size)) / math.sqrt(2)
    y = config.points[idx] + noise * 10 ** (-snr / 20)
    mi, gmi = information_rates(y, idx, config.points, config.bit_labels, config.pmf)
    mi2, gmi2 = ad.information_rates(
        jnp.asarray(y),
        jnp.asarray(idx),
        jnp.asarray(config.points),
        config.bit_labels,
        jnp.asarray(config.pmf),
    )
    assert 1.0 < gmi < mi < config.entropy_bits  # a non-trivial regime
    assert float(mi2) == pytest.approx(mi, abs=1e-12)
    assert float(gmi2) == pytest.approx(gmi, abs=1e-12)
    # GMI from the bit LLRs: H − Σᵢ E[log₂(1 + e^{−(1−2bᵢ)Lᵢ})] (uniform labels only)
    if config.is_uniform:
        sigma2 = np.mean(np.abs(y - config.points[idx]) ** 2)
        llr = np.asarray(
            ad.bit_llrs(jnp.asarray(y), jnp.asarray(config.points), config.bit_labels, sigma2)
        )
        bits = config.bit_labels[idx]
        from_llr = config.bits_per_symbol - np.mean(
            np.sum(np.logaddexp(0.0, -(1 - 2.0 * bits) * llr), axis=1)
        ) / math.log(2)
        assert from_llr == pytest.approx(gmi, abs=1e-10)


def test_gradients_through_the_whole_chain_equal_finite_differences():
    """GMI after modulate → link (fibers, EDFAs with fixed noise) → DBP receiver, in every input."""
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=32e9, n_symbols=256, samples_per_symbol=4
    )
    link = _link(tx)
    rx = tx.matched_receiver(
        compensation=DigitalBackPropagation(link=link, steps_per_span=2, samples_per_symbol=2)
    )
    idx = jnp.asarray(np.random.default_rng(5).integers(0, 16, 256))
    labels = tx.resolved_constellation.bit_labels
    key = jax.random.key(0)

    def gmi(tp, link_params, xi):
        points, pmf = ad.constellation(tx, tp)
        x = ad.modulate(points[idx], tx, launch_power_dbm=tp.launch_power_dbm)
        y = ad.propagate_link(x, link, tx.dt_s, key=key, params=link_params)
        z = ad.demodulate(y, rx, tx.dt_s, reference=points[idx], nonlinearity_scale=xi)
        return ad.information_rates(z, idx, points, labels, pmf)[1]

    tp = ad.TransmitterParameters.from_transmitter(tx)._replace(launch_power_dbm=jnp.asarray(6.0))
    lp = ad.link_parameters(link)
    xi = jnp.asarray(0.8)
    f = jax.jit(gmi)
    g_tp, g_lp, g_xi = jax.grad(f, argnums=(0, 1, 2))(tp, lp, xi)

    def check(exact: float, shift) -> None:
        h = 1e-5
        numeric = (float(shift(h)) - float(shift(-h))) / (2 * h)
        assert exact == pytest.approx(numeric, rel=2e-5, abs=1e-9)

    check(float(g_xi), lambda h: f(tp, lp, xi + h))
    check(
        float(g_tp.launch_power_dbm),
        lambda h: f(tp._replace(launch_power_dbm=tp.launch_power_dbm + h), lp, xi),
    )
    for k in (0, 5):  # a point's real and imaginary parts (JAX: ∂/∂Re − i∂/∂Im)
        e = jnp.zeros(16, complex).at[k].set(1.0)
        check(
            float(jnp.real(g_tp.points[k])),
            lambda h, e=e: f(tp._replace(points=tp.points + h * e), lp, xi),
        )
        check(
            -float(jnp.imag(g_tp.points[k])),
            lambda h, e=e: f(tp._replace(points=tp.points + 1j * h * e), lp, xi),
        )
    gamma = float(g_lp[0].gamma_per_w_per_km)
    check(
        gamma,
        lambda h: f(
            tp, (lp[0]._replace(gamma_per_w_per_km=lp[0].gamma_per_w_per_km + h), *lp[1:]), xi
        ),
    )


def test_gumbel_samples_send_exact_points_with_the_softmax_frequencies():
    points = jnp.asarray(ConstellationConfig.from_square_qam(16).points)
    logits = jnp.asarray(np.linspace(-1.0, 1.0, 16))
    symbols, idx = ad.sample_symbols(jax.random.key(1), points, logits, (200_000,))
    np.testing.assert_allclose(np.asarray(symbols), np.asarray(points[idx]), atol=1e-15)
    freq = np.bincount(np.asarray(idx), minlength=16) / idx.size
    np.testing.assert_allclose(freq, np.asarray(jax.nn.softmax(logits)), atol=4e-3)
    grad = jax.grad(
        lambda lg: jnp.sum(
            jnp.abs(ad.sample_symbols(jax.random.key(1), points, lg, (1000,))[0]) ** 2
        )
    )(logits)
    assert np.all(np.isfinite(np.asarray(grad))) and np.any(np.asarray(grad) != 0)


# ---------------------------------------------------------------------------
# End-to-end learning, scored by the NumPy pipeline on fresh data
# ---------------------------------------------------------------------------


def _adam(grad_fn, params, steps: int, lr: float):
    """Plain Adam on a pytree (maximises: grad_fn returns the ascent direction)."""
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(lambda p: jnp.zeros_like(jnp.real(p)), params)
    for t in range(1, steps + 1):
        g = jax.tree_util.tree_map(jnp.conj, grad_fn(params))  # JAX gives the conjugate
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * jnp.abs(b) ** 2, v, g)
        params = jax.tree_util.tree_map(
            lambda p, a, b, t=t: (
                p + lr * (a / (1 - 0.9**t)) / (jnp.sqrt(b / (1 - 0.999**t)) + 1e-9)
            ),
            params,
            m,
            v,
        )
    return params


def _awgn_simulation(config: ConstellationConfig, snr_db: float, seed: int) -> float:
    tx = SingleCarrierTransmitter(
        constellation=config, symbol_rate_hz=32e9, n_symbols=1 << 14, seed=seed
    )
    noise = AdditiveNoise.for_snr(
        snr_db, power_per_polarization_w=tx.launch_power_per_polarization_w, symbol_rate_hz=32e9
    )
    sim = Simulation(
        transmitter=tx, link=Link(elements=(noise,)), receiver=tx.matched_receiver(), seed=seed + 1
    )
    return sim.run().metrics.mi_bits


def test_learned_geometric_shaping_beats_16qam_on_awgn():
    """Maximise the MI of 16 free points at Es/N0 = 12 dB; score with Simulation on new data."""
    snr = 12.0
    qam = ConstellationConfig.from_square_qam(16)
    sigma = 10 ** (-snr / 20) / math.sqrt(2)

    def mi(points, key):
        points = points / jnp.sqrt(jnp.mean(jnp.abs(points) ** 2))
        k1, k2, k3 = jax.random.split(key, 3)
        idx = jax.random.randint(k1, (8192,), 0, 16)
        n = jax.random.normal(k2, (8192,)) + 1j * jax.random.normal(k3, (8192,))
        return ad.information_rates(points[idx] + sigma * n, idx, points, qam.bit_labels)[0]

    grad = jax.jit(jax.grad(mi))
    keys = iter(jax.random.split(jax.random.key(0), 400))
    learned = _adam(lambda p: grad(p, next(keys)), jnp.asarray(qam.points), 400, 0.01)
    config = ConstellationConfig.from_points(np.asarray(learned), bit_labels=qam.bit_labels)
    gain = _awgn_simulation(config, snr, 11) - _awgn_simulation(qam, snr, 11)
    assert gain > 0.02  # bits/symbol (≈ 0.05 expected: geometric shaping gain at 12 dB)


def test_learned_probabilistic_shaping_beats_uniform_64qam_on_awgn():
    """Gumbel-softmax training of the 64-QAM PMF at Es/N0 = 12 dB; the MB-like PMF wins."""
    snr = 12.0
    qam = ConstellationConfig.from_square_qam(64)
    points_u = jnp.asarray(qam.points)
    sigma = 10 ** (-snr / 20) / math.sqrt(2)

    def mi(logits, key):
        pmf = jax.nn.softmax(logits)
        points = points_u / jnp.sqrt(jnp.sum(pmf * jnp.abs(points_u) ** 2))
        k1, k2, k3 = jax.random.split(key, 3)
        symbols, idx = ad.sample_symbols(k1, points, logits, (8192,))
        n = jax.random.normal(k2, (8192,)) + 1j * jax.random.normal(k3, (8192,))
        return ad.information_rates(symbols + sigma * n, idx, points, qam.bit_labels, pmf)[0]

    grad = jax.jit(jax.grad(mi))
    keys = iter(jax.random.split(jax.random.key(2), 300))
    logits = _adam(lambda lg: grad(lg, next(keys)), jnp.zeros(64), 300, 0.05)
    pmf = np.asarray(jax.nn.softmax(logits), np.float64)
    config = qam.with_probabilities(pmf / pmf.sum())
    gain = _awgn_simulation(config, snr, 21) - _awgn_simulation(qam, snr, 21)
    assert gain > 0.05
    inner = np.abs(qam.points) < np.median(np.abs(qam.points))
    assert pmf[inner].mean() > 1.5 * pmf[~inner].mean()  # low-energy points favoured
