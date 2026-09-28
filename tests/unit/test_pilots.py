"""Time-domain pilots: configuration, insertion, extraction, pilot-aided equalisation.

Convention (since the v1 hardening): pilots carry no information bits —
``TransmittedSignal.bits`` / ``ReceivedSymbols.bits`` / ``.symbols`` cover the
data slots only; pilot samples are reported separately.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from helpers import PULSE_MODES, sc_pair, trim, wdm_pair
from hpcom import PilotConfig, SingleCarrierReceiver, SingleCarrierTransmitter


def test_period_below_two_is_rejected():
    with pytest.raises(ValidationError, match="period"):
        PilotConfig(period=1)


def test_indices_and_counts():
    cfg = PilotConfig(period=4)
    np.testing.assert_array_equal(cfg.pilot_indices(16), [0, 4, 8, 12])
    np.testing.assert_array_equal(cfg.data_indices(16), [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15])
    assert (cfg.n_pilots(16), cfg.n_data(16)) == (4, 12)
    np.testing.assert_array_equal(cfg.pilot_indices(14), [0, 4, 8, 12])


@pytest.mark.parametrize("period", [2, 3, 7, 16])
@pytest.mark.parametrize("n", [1, 2, 15, 16, 17, 100])
def test_pilot_and_data_slots_partition_the_block(period, n):
    cfg = PilotConfig(period=period)
    pilots, data = cfg.pilot_indices(n), cfg.data_indices(n)
    assert (pilots.size, data.size) == (cfg.n_pilots(n), cfg.n_data(n))
    np.testing.assert_array_equal(np.sort(np.concatenate([pilots, data])), np.arange(n))


def test_transmitter_places_pilots_and_maps_bits_to_data_slots():
    cfg = PilotConfig(period=8, symbol=complex(0.7, -0.4))
    tx = SingleCarrierTransmitter(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=64, seed=0, pilots=cfg
    )
    assert tx.n_data_symbols_per_stream == 56
    assert tx.n_bits_per_stream == 112
    bits = np.random.default_rng(0).integers(0, 2, 112, dtype=np.uint8)
    sent = tx.generate(bits=bits)
    np.testing.assert_array_equal(sent.symbols[cfg.pilot_indices(64)], cfg.symbol)
    np.testing.assert_array_equal(sent.data_symbols(), tx.resolved_constellation.map(bits))
    np.testing.assert_array_equal(sent.bits, bits)


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_pilot_samples_and_data_are_recovered_b2b(pulse):
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx, rx = sc_pair(
        modulation_order=16,
        n_symbols=512,
        seed=1,
        pilots=cfg,
        rx_kwargs={"equalize": "pilot"},
        **pulse,
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    np.testing.assert_array_equal(received.bits.size, sent.bits.size)
    k, skip = tx.bits_per_symbol, tx.transient_symbols
    np.testing.assert_array_equal(trim(received.bits, k, skip), trim(sent.bits, k, skip))
    atol = 1e-12 if not pulse else 2e-2
    pskip = -(-skip // cfg.period)
    np.testing.assert_allclose(trim(received.pilot_symbols, 1, pskip), 1.0, rtol=0, atol=atol)


@pytest.mark.parametrize("gain", [1e-3, 0.02 * np.exp(1j * 2.1), 5.0 * np.exp(-1j * 0.7)])
def test_pilot_equaliser_removes_an_arbitrary_complex_gain(gain):
    cfg = PilotConfig(period=16, symbol=np.exp(1j * 0.3))
    tx, rx = sc_pair(
        modulation_order=64, n_symbols=1024, seed=2, pilots=cfg, rx_kwargs={"equalize": "pilot"}
    )
    sent = tx.generate()
    received = rx.receive(sent.field * gain, sent.dt_s)
    np.testing.assert_allclose(received.symbols, sent.data_symbols(), rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        received.gain, gain * np.sqrt(tx.launch_power_per_polarization_w), rtol=1e-12
    )
    np.testing.assert_array_equal(received.bits, sent.bits)


def test_unit_power_equaliser_cannot_remove_a_phase_rotation():
    """Contrast with the pilot equaliser: blind amplitude normalisation leaves the phase."""
    cfg = PilotConfig(period=16)
    tx, rx = sc_pair(modulation_order=16, n_symbols=1024, seed=2, pilots=cfg)
    sent = tx.generate()
    received = rx.receive(sent.field * np.exp(1j * 0.5), sent.dt_s)
    assert np.mean(received.bits != sent.bits) > 0.1


def test_pilot_equaliser_requires_pilots():
    with pytest.raises(ValidationError, match="requires `pilots`"):
        SingleCarrierReceiver(modulation_order=4, equalize="pilot")


def test_pilot_energy_is_part_of_the_launch_power():
    """A 6 dB pilot (|p|² = 4) every 4th slot still gives the configured average power."""
    cfg = PilotConfig(period=4, symbol=2 + 0j)
    tx = SingleCarrierTransmitter(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=16384,
        seed=0,
        pilots=cfg,
        launch_power_per_channel_dbm=0.0,
    )
    np.testing.assert_allclose(tx.generate().mean_power_w, 1e-3, rtol=0.02)


def test_wdm_pilots_are_recovered_per_channel():
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx, rx = wdm_pair(
        n_channels=3,
        modulation_order=4,
        n_symbols=512,
        seed=0,
        pilots=cfg,
        rx_kwargs={"equalize": "pilot"},
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.pilot_symbols.shape == (3, cfg.n_pilots(512))
    np.testing.assert_allclose(received.pilot_symbols, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(received.bits, sent.bits)


def test_without_pilots_every_slot_is_data():
    tx, rx = sc_pair(modulation_order=4, n_symbols=64, seed=0)
    sent = tx.generate()
    np.testing.assert_array_equal(sent.data_symbols(), sent.symbols)
    assert tx.pilot_indices.size == 0
    assert rx.receive(sent.field, sent.dt_s).pilot_symbols is None
