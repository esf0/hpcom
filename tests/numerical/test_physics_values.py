"""Set-X-measure-X physics invariants (launch power, loss, gain, carriers, CD)."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import b2b_link, matched_receiver_kwargs, wdm_pair
from hpcom import (
    EDFA,
    Fiber,
    Link,
    Simulation,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMTransmitter,
)
from hpcom.units import dbm_to_w

# ---------------------------------------------------------------------------
# Launch power
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["single", "dual"])
@pytest.mark.parametrize("dbm", [-10.0, -3.0, 0.0, 3.0])
def test_received_power_is_the_launch_power_per_channel_summed_over_polarisations(
    dbm, polarization
):
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=16384,
        samples_per_symbol=4,
        seed=0,
        polarization=polarization,
        launch_power_per_channel_dbm=dbm,
    )
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx))
    result = Simulation(transmitter=tx, link=b2b_link(), receiver=rx).run()
    total = np.sum(
        np.mean(np.abs(result.received_field.reshape(tx.n_polarizations, -1)) ** 2, axis=-1)
    )
    np.testing.assert_allclose(total, dbm_to_w(dbm), rtol=0.02)


@pytest.mark.parametrize("n_channels", [3, 5])
def test_total_launch_power_of_a_wdm_comb(n_channels):
    tx, _ = wdm_pair(
        n_channels=n_channels,
        modulation_order=16,
        n_symbols=8192,
        samples_per_symbol=32,
        spacing_hz=50e9,
        symbol_rate_hz=10e9,
        seed=0,
        launch_power_total_dbm=7.0,
    )
    assert tx.launch_power_per_channel_w == pytest.approx(dbm_to_w(7.0) / n_channels)
    np.testing.assert_allclose(tx.generate().mean_power_w, dbm_to_w(7.0), rtol=0.02)


# ---------------------------------------------------------------------------
# Fiber loss and EDFA gain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["numpy-linear", "numpy"])
@pytest.mark.parametrize(("alpha", "length"), [(0.2, 80.0), (0.3, 50.0), (1.0, 10.0)])
def test_fiber_loss_matches_beer_lambert(alpha, length, backend):
    fiber = Fiber(
        length_km=length,
        alpha_db_per_km=alpha,
        beta2_s2_per_m=0.0,
        gamma_per_w_per_km=0.0,
        n_steps=10,
        backend=backend,
    )
    field = np.random.default_rng(0).standard_normal(2048) + 1j
    out = fiber.propagate(field, dt_s=1e-11)
    np.testing.assert_allclose(
        np.mean(np.abs(out) ** 2),
        np.mean(np.abs(field) ** 2) * 10 ** (-alpha * length / 10),
        rtol=1e-12,
    )


@pytest.mark.parametrize("gain_db", [3.0, 10.0, 20.0])
def test_edfa_gain_is_exact_on_the_signal(gain_db):
    """Twin RNG: amplify(signal) − amplify(0) with the same noise draws = √G·signal exactly."""
    edfa = EDFA(gain_db=gain_db, noise_figure_db=5.0)
    sig = np.random.default_rng(0).standard_normal(2048) + 1j * np.random.default_rng(
        1
    ).standard_normal(2048)
    out = edfa.amplify(sig, 1e-11, rng=np.random.default_rng(42)) - edfa.amplify(
        np.zeros_like(sig), 1e-11, rng=np.random.default_rng(42)
    )
    np.testing.assert_allclose(out, np.sqrt(10 ** (gain_db / 10)) * sig, rtol=1e-12)


def test_gain_compensated_span_restores_the_signal_power():
    fiber = Fiber.ssmf(80.0, backend="numpy-linear")
    link = Link(elements=(fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0)))
    x = (np.random.default_rng(3).standard_normal(1 << 16) + 1j) * 1e-2
    out = link.propagate(x, 1e-11, rng=np.random.default_rng(4))
    ase = EDFA(gain_db=16.0, noise_figure_db=5.0).ase_variance_per_sample(1e-11)
    np.testing.assert_allclose(np.mean(np.abs(out) ** 2), np.mean(np.abs(x) ** 2) + ase, rtol=5e-3)


# ---------------------------------------------------------------------------
# Carriers and polarisations
# ---------------------------------------------------------------------------


def test_wdm_carriers_sit_exactly_on_their_frequencies():
    """Constant symbols → each channel is a pure tone, exactly on one FFT bin at f_i."""
    cfg = WDMConfig(n_channels=5, spacing_hz=37.5e9)
    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=4096, samples_per_symbol=32, wdm=cfg
    )
    sig = tx.generate(symbols=np.ones((5, 4096)))
    psd = np.abs(np.fft.fft(sig.field)) ** 2
    f = np.fft.fftfreq(sig.field.size, sig.dt_s)
    peaks = np.sort(f[np.argsort(psd)[-5:]])
    np.testing.assert_allclose(peaks, cfg.channel_frequencies_hz(), rtol=0, atol=1e-3)
    assert np.sort(psd)[-6] < 1e-20 * psd.max()


def test_dual_polarisation_power_is_split_equally():
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=16384,
        seed=0,
        polarization="dual",
        launch_power_per_channel_dbm=0.0,
    )
    sig = tx.generate()
    p = np.mean(np.abs(sig.field) ** 2, axis=-1)
    np.testing.assert_allclose(p, 0.5e-3, rtol=0.02)
    np.testing.assert_allclose(sig.mean_power_w, p.sum(), rtol=1e-12)


# ---------------------------------------------------------------------------
# Chromatic-dispersion compensation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_receiver_cd_compensation_undoes_a_linear_link_exactly(polarization):
    fiber = Fiber(
        length_km=800.0,
        alpha_db_per_km=0.0,
        gamma_per_w_per_km=0.0,
        backend="numpy-linear",
        polarization=polarization,
    )
    link = Link(elements=(fiber,))
    tx = SingleCarrierTransmitter(
        modulation_order=64,
        symbol_rate_hz=32e9,
        n_symbols=4096,
        samples_per_symbol=2,
        seed=1,
        polarization=polarization,
    )
    rx = SingleCarrierReceiver(
        **matched_receiver_kwargs(tx, equalize="none", cd_compensation_s2=link.total_dispersion_s2)
    )
    sent = tx.generate()
    out = rx.receive(link.propagate(sent.field, sent.dt_s), sent.dt_s)
    np.testing.assert_allclose(
        out.symbols / np.sqrt(tx.launch_power_per_polarization_w), sent.symbols, rtol=0, atol=1e-10
    )
