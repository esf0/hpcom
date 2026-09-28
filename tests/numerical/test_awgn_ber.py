"""End-to-end AWGN performance vs. exact theory.

Every case runs the public pipeline — Transmitter → AdditiveNoise → Receiver
→ Simulation metrics — at a configured per-symbol SNR ``Es/N0`` and checks:

* the measured (data-aided) SNR equals the configured SNR (±0.05 dB, i.e.
  the matched-filter noise bandwidth and launch-power scaling are right);
* the BER equals the exact BER of Gray-coded square QAM (Cho & Yoon, IEEE
  Trans. Commun. 50(7), 2002) within a 5σ Monte-Carlo bound. σ is taken as
  √(k·p/n) — bits of one symbol are correlated, so the plain binomial
  √(p/n) would be too tight.

Receivers use ``equalize="data_aided"`` (an unbiased least-squares gain):
the blind ``"unit_power"`` estimator normalises signal *plus* noise power and
is biased at low SNR.
"""

from __future__ import annotations

import numpy as np
import pytest

from helpers import PULSE_MODES, ber_square_qam_awgn, matched_receiver_kwargs
from hpcom import (
    AdditiveNoise,
    Link,
    OFDMConfig,
    OFDMReceiver,
    OFDMTransmitter,
    PilotConfig,
    Simulation,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMReceiver,
    WDMTransmitter,
)
from hpcom.metrics import Metrics

pytestmark = pytest.mark.slow


def _noise_for(tx, snr_db: float, symbol_rate_hz: float) -> Link:
    return Link(
        elements=(
            AdditiveNoise.for_snr(
                snr_db,
                power_per_polarization_w=tx.launch_power_per_polarization_w,
                symbol_rate_hz=symbol_rate_hz,
            ),
        )
    )


def _check(m: Metrics, snr_db: float, order: int, snr_offset_db: float = 0.0) -> None:
    """``snr_offset_db``: known difference between the configured and per-symbol SNR (OFDM)."""
    k = int(np.log2(order))
    snr = snr_db + snr_offset_db
    p = ber_square_qam_awgn(10 ** (snr / 10), order)
    sigma = np.sqrt(k * p / m.n_bits)
    assert abs(m.snr_db - snr) < 0.05, f"measured SNR {m.snr_db:.3f} dB vs {snr:.3f} dB"
    assert abs(m.ber - p) < 5 * sigma, f"BER {m.ber:.4e} vs theory {p:.4e} (5σ = {5 * sigma:.1e})"


@pytest.mark.parametrize("polarization", ["single", "dual"])
@pytest.mark.parametrize(
    ("order", "snr_db"), [(4, 6.0), (4, 9.0), (16, 12.0), (16, 15.0), (64, 18.0), (64, 21.0)]
)
def test_single_carrier(order, snr_db, polarization):
    tx = SingleCarrierTransmitter(
        modulation_order=order,
        symbol_rate_hz=32e9,
        n_symbols=1 << 16,
        samples_per_symbol=2,
        polarization=polarization,
        launch_power_per_channel_dbm=1.0,
    )
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx, equalize="data_aided"))
    result = Simulation(
        transmitter=tx, link=_noise_for(tx, snr_db, 32e9), receiver=rx, seed=11
    ).run()
    _check(result.metrics, snr_db, order)


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_single_carrier_every_pulse_mode(pulse):
    """The truncated FIR's ISI (≈ −51 dB, span 32) is negligible at 14 dB SNR."""
    tx = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=32e9, n_symbols=1 << 16, samples_per_symbol=4, **pulse
    )
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx, equalize="data_aided"))
    result = Simulation(transmitter=tx, link=_noise_for(tx, 14.0, 32e9), receiver=rx, seed=3).run()
    _check(result.metrics, 14.0, 16)


def test_pilot_aided_equalisation_reaches_theory():
    pilots = PilotConfig(period=16, symbol=np.exp(1j * np.pi / 4))
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 16,
        samples_per_symbol=2,
        pilots=pilots,
    )
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx, equalize="pilot"))
    result = Simulation(transmitter=tx, link=_noise_for(tx, 13.0, 32e9), receiver=rx, seed=5).run()
    _check(result.metrics, 13.0, 16)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_wdm_every_channel(polarization):
    wdm = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=1 << 15,
        samples_per_symbol=8,
        wdm=wdm,
        polarization=polarization,
    )
    rx = WDMReceiver(wdm=wdm, **matched_receiver_kwargs(tx, equalize="data_aided"))
    result = Simulation(transmitter=tx, link=_noise_for(tx, 14.0, 32e9), receiver=rx, seed=7).run()
    assert len(result.stream_metrics) == 3 * tx.n_polarizations
    _check(result.metrics, 14.0, 16)
    for stream in result.stream_metrics:  # every (polarisation, channel) on its own
        _check(stream, 14.0, 16)


@pytest.mark.parametrize("oversampling", [1, 2])
@pytest.mark.parametrize(("order", "snr_db"), [(4, 8.0), (16, 14.0), (64, 20.0)])
def test_ofdm(order, snr_db, oversampling):
    """Per-subcarrier SNR = SNR_B · N / (n_data + n_pilot·|p|²/E_s), SNR_B = P/(N₀·N·Δf).

    Only the occupied subcarriers carry power, so the SNR per data subcarrier
    exceeds the SNR over the full subcarrier grid by N/(n_data + …).
    """
    ofdm = OFDMConfig(n_subcarriers=256, cp_len=16, n_guard=8, oversampling=oversampling)
    tx = OFDMTransmitter(
        modulation_order=order, subcarrier_spacing_hz=125e6, n_frames=256, ofdm=ofdm
    )
    rx = OFDMReceiver(modulation_order=order, ofdm=ofdm, equalize="data_aided")
    result = Simulation(
        transmitter=tx, link=_noise_for(tx, snr_db, tx.bandwidth_hz), receiver=rx, seed=9
    ).run()
    offset = 10 * np.log10(ofdm.n_subcarriers / ofdm.frame_energy(1.0))
    _check(result.metrics, snr_db, order, snr_offset_db=offset)


def test_qpsk_generalised_mutual_information_matches_theory():
    """Gray QPSK on AWGN: GMI = MI = 2·C_BPSK(SNR/2 per dimension)."""
    tx = SingleCarrierTransmitter(
        modulation_order=4, symbol_rate_hz=32e9, n_symbols=1 << 16, samples_per_symbol=2
    )
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx, equalize="data_aided"))
    result = Simulation(transmitter=tx, link=_noise_for(tx, 3.0, 32e9), receiver=rx, seed=1).run()
    snr = 10**0.3
    x, w = np.polynomial.hermite.hermgauss(120)
    a, s = 1 / np.sqrt(2), np.sqrt(1 / (2 * snr))
    c_bpsk = 1 - np.sum(
        w * np.log2(1 + np.exp(-2 * a * (a + np.sqrt(2) * s * x) / s**2))
    ) / np.sqrt(np.pi)
    assert result.metrics.gmi_bits == pytest.approx(2 * c_bpsk, abs=0.01)
    assert result.metrics.mi_bits == pytest.approx(2 * c_bpsk, abs=0.01)
