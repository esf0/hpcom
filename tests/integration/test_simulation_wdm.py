"""End-to-end Simulation with WDM Tx/Rx (Phase 2 / commit F).

Replaces the prior ``NotImplementedError`` guard with a real per-channel
scoring loop. Covers the four Tx/Rx × polarisation combinations:

- single-pol single-carrier (regression — existing behaviour preserved)
- single-pol WDM
- dual-pol  single-carrier
- dual-pol  WDM
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.amplifier import EDFA
from hpcom.fiber import Fiber
from hpcom.link import Link
from hpcom.receiver import Receiver, WDMReceiver
from hpcom.simulation import Simulation
from hpcom.transmitter import Transmitter, WDMTransmitter
from hpcom.wdm import WDMConfig


def _passthrough_link() -> Link:
    """Lossless 1-span fiber + 0 dB amplifier — exactly preserves power."""
    fiber = Fiber(
        length_km=1.0, alpha_db_per_km=0.0,
        beta2_s2_per_m=0.0, gamma_per_w_per_km=0.0,
        n_steps=1, backend="numpy-linear",
    )
    amp = EDFA(gain_db=0.0, noise_figure_db=0.0)
    return Link.from_spans(n_spans=1, fiber=fiber, amplifier=amp, strict=True)


def _b2b_dual_pol_passthrough_link() -> Link:
    """Same as above but with a dual-pol-capable fiber."""
    fiber = Fiber(
        length_km=1.0, alpha_db_per_km=0.0,
        beta2_s2_per_m=0.0, gamma_per_w_per_km=0.0,
        n_steps=1, backend="numpy-linear", polarization="dual",
    )
    amp = EDFA(gain_db=0.0, noise_figure_db=0.0)
    return Link.from_spans(n_spans=1, fiber=fiber, amplifier=amp, strict=True)


def test_single_pol_single_carrier_simulation_ber_zero():
    """Regression: the pre-refactor pipeline still produces BER=0 in B2B."""
    tx = Transmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=512,
        samples_per_symbol=8, pulse_span_symbols=32, seed=0,
    )
    rx = Receiver(
        modulation_order=4, samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off, pulse_span_symbols=tx.pulse_span_symbols,
    )
    sim = Simulation(transmitter=tx, link=_passthrough_link(), receiver=rx)
    result = sim.run()
    assert result.metrics.ber == 0.0
    assert result.metrics.n_bits > 0


@pytest.mark.parametrize("ch", [0, 1, 2])
def test_single_pol_wdm_simulation_per_channel_ber_zero(ch: int):
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=1024,
        samples_per_symbol=16, pulse_span_symbols=32, seed=0, wdm=cfg,
    )
    rx = WDMReceiver(
        modulation_order=4, samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off, pulse_span_symbols=tx.pulse_span_symbols,
        wdm=cfg, wdm_channel=ch,
    )
    sim = Simulation(transmitter=tx, link=_passthrough_link(), receiver=rx)
    result = sim.run()
    assert result.metrics.ber == 0.0, (
        f"channel {ch} BER must be 0 in noiseless passthrough"
    )
    # Metadata records which channel was scored.
    assert result.metadata.config["scored_wdm_channel"] == ch
    assert result.metadata.config["n_channels"] == 3


def test_dual_pol_single_carrier_simulation_aggregates_per_pol_ber_zero():
    """Dual-pol B2B: BER aggregated over both polarisations equals 0."""
    tx = Transmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=512,
        samples_per_symbol=8, pulse_span_symbols=32, seed=0,
        polarization="dual",
    )
    rx = Receiver(
        modulation_order=4, samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off, pulse_span_symbols=tx.pulse_span_symbols,
        polarization="dual",
    )
    sim = Simulation(
        transmitter=tx, link=_b2b_dual_pol_passthrough_link(), receiver=rx,
    )
    result = sim.run()
    assert result.metrics.ber == 0.0
    # n_bits should reflect both polarisations.
    bps = 2
    interior = tx.n_symbols - 2 * tx.pulse_span_symbols
    assert result.metrics.n_bits == 2 * interior * bps


@pytest.mark.parametrize("ch", [0, 1, 2])
def test_dual_pol_wdm_simulation_per_channel_ber_zero(ch: int):
    """The full Tx/Rx matrix's hardest combo: dual-pol × 3-channel WDM."""
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=1024,
        samples_per_symbol=16, pulse_span_symbols=32, seed=0,
        polarization="dual", wdm=cfg,
    )
    rx = WDMReceiver(
        modulation_order=4, samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off, pulse_span_symbols=tx.pulse_span_symbols,
        polarization="dual", wdm=cfg, wdm_channel=ch,
    )
    sim = Simulation(
        transmitter=tx, link=_b2b_dual_pol_passthrough_link(), receiver=rx,
    )
    result = sim.run()
    assert result.metrics.ber == 0.0


def test_simulation_rejects_mismatched_polarization():
    tx = Transmitter(
        modulation_order=4, symbol_rate_hz=1e9, n_symbols=64,
        samples_per_symbol=4, polarization="dual",
    )
    rx = Receiver(modulation_order=4, samples_per_symbol=4, polarization="single")
    with pytest.raises(ValueError, match="polarization must match"):
        Simulation(transmitter=tx, link=_passthrough_link(), receiver=rx)


def test_simulation_rejects_mismatched_wdm_configs():
    cfg_a = WDMConfig(n_channels=3, spacing_hz=50e9)
    cfg_b = WDMConfig(n_channels=3, spacing_hz=37.5e9)
    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=64,
        samples_per_symbol=32, wdm=cfg_a,
    )
    rx = WDMReceiver(
        modulation_order=4, samples_per_symbol=32, wdm=cfg_b,
    )
    with pytest.raises(ValueError, match="same WDMConfig"):
        Simulation(transmitter=tx, link=_passthrough_link(), receiver=rx)


def test_simulation_rejects_wdm_tx_with_single_carrier_rx():
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    tx = WDMTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=64,
        samples_per_symbol=32, wdm=cfg,
    )
    rx = Receiver(modulation_order=4, samples_per_symbol=32)  # not WDMReceiver
    with pytest.raises(ValueError, match="WDM Tx requires WDM Rx"):
        Simulation(transmitter=tx, link=_passthrough_link(), receiver=rx)


def test_simulation_seed_reproducibility_across_runs():
    """Same seed → identical Result.metrics and Result.received_field bytes."""
    cfg = WDMConfig(n_channels=3, spacing_hz=50e9)
    kw_tx = dict(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=256,
        samples_per_symbol=16, pulse_span_symbols=32, wdm=cfg,
    )
    rx = WDMReceiver(
        modulation_order=4, samples_per_symbol=16, pulse_span_symbols=32,
        wdm=cfg, wdm_channel=1,
    )
    edfa = EDFA(gain_db=10.0, noise_figure_db=5.0)
    fiber = Fiber(
        length_km=80.0, alpha_db_per_km=0.2, beta2_s2_per_m=-2.16e-26,
        gamma_per_w_per_km=0.0, n_steps=20, backend="numpy-linear",
    )
    link = Link.from_spans(n_spans=1, fiber=fiber, amplifier=edfa, strict=True)
    sim1 = Simulation(
        transmitter=WDMTransmitter(**kw_tx), link=link, receiver=rx, seed=99,
    )
    sim2 = Simulation(
        transmitter=WDMTransmitter(**kw_tx), link=link, receiver=rx, seed=99,
    )
    r1 = sim1.run()
    r2 = sim2.run()
    np.testing.assert_array_equal(r1.received_field, r2.received_field)
    assert r1.metrics.ber == r2.metrics.ber
    assert r1.metrics.n_bit_errors == r2.metrics.n_bit_errors
