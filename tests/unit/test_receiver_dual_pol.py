"""Dual-polarisation single-carrier receiver."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from helpers import PULSE_MODES, sc_pair, trim
from hpcom import PilotConfig, SingleCarrierReceiver, WDMConfig, WDMReceiver


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_output_shapes(pulse):
    tx, rx = sc_pair(modulation_order=16, n_symbols=512, polarization="dual", seed=0, **pulse)
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.symbols.shape == (2, 512)
    assert received.decisions.shape == (2, 512)
    assert received.bits.shape == (2, 512 * 4)
    assert received.gain.shape == (2,)
    assert received.polarization == "dual"
    assert received.pilot_symbols is None


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("order", [4, 16, 64])
def test_b2b_each_polarisation_error_free(order, pulse):
    tx, rx = sc_pair(modulation_order=order, n_symbols=1024, polarization="dual", seed=2, **pulse)
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    k, skip = tx.bits_per_symbol, tx.transient_symbols
    for pol in (0, 1):
        np.testing.assert_array_equal(
            trim(received.bits[pol], k, skip), trim(sent.bits[pol], k, skip)
        )


def test_polarisations_are_not_coupled():
    tx, rx = sc_pair(modulation_order=16, n_symbols=512, polarization="dual", seed=0)
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert not np.array_equal(received.bits[0], received.bits[1])
    # Swapping X and Y at the input swaps the outputs.
    swapped = rx.receive(sent.field[::-1], sent.dt_s)
    np.testing.assert_array_equal(swapped.bits, received.bits[::-1])


def test_dual_receiver_rejects_single_polarisation_field():
    rx = SingleCarrierReceiver(modulation_order=4, polarization="dual")
    with pytest.raises(ValueError, match=r"expects shape \(2, n\)"):
        rx.receive(np.zeros(64, dtype=complex), 1e-11)


def test_single_receiver_rejects_dual_polarisation_field():
    rx = SingleCarrierReceiver(modulation_order=4)
    with pytest.raises(ValueError, match="1-D field"):
        rx.receive(np.zeros((2, 64), dtype=complex), 1e-11)


@pytest.mark.parametrize("pulse", PULSE_MODES)
def test_pilots_are_recovered_in_each_polarisation(pulse):
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx, rx = sc_pair(
        modulation_order=4,
        n_symbols=512,
        polarization="dual",
        seed=2,
        pilots=cfg,
        rx_kwargs={"equalize": "pilot"},
        **pulse,
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.pilot_symbols.shape == (2, cfg.n_pilots(512))
    skip = -(-tx.transient_symbols // cfg.period)
    atol = 1e-12 if not pulse else 2e-2
    np.testing.assert_allclose(trim(received.pilot_symbols, 1, skip), cfg.symbol, rtol=0, atol=atol)


def test_bits_cover_data_slots_only():
    cfg = PilotConfig(period=4)
    tx, rx = sc_pair(modulation_order=4, n_symbols=64, polarization="dual", seed=0, pilots=cfg)
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    n_data = cfg.n_data(64)
    assert sent.bits.shape == received.bits.shape == (2, n_data * 2)
    assert received.symbols.shape == (2, n_data)
    np.testing.assert_array_equal(received.bits, sent.bits)


def test_dual_polarisation_wdm_receiver_configuration():
    rx = WDMReceiver(
        modulation_order=4,
        polarization="dual",
        wdm=WDMConfig(n_channels=3, spacing_hz=50e9),
        channels=(1,),
    )
    assert rx.polarization == "dual"
    assert rx.selected_channels == (1,)


def test_invalid_polarization_is_rejected():
    with pytest.raises(ValidationError, match="polarization"):
        SingleCarrierReceiver(modulation_order=4, polarization="triple")


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_received_symbols_report_polarisation(polarization):
    tx, rx = sc_pair(modulation_order=4, n_symbols=16, polarization=polarization, seed=0)
    sent = tx.generate()
    assert rx.receive(sent.field, sent.dt_s).polarization == polarization
