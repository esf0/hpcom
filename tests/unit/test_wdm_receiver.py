"""WDM receiver: demultiplexing, channel selection, CD compensation of every channel."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import PULSE_MODES, trim, wdm_pair
from hpcom import Fiber, Link, PilotConfig


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_b2b_every_channel_every_polarisation_error_free(polarization, pulse):
    tx, rx = wdm_pair(
        modulation_order=16, n_symbols=1024, polarization=polarization, seed=1, **pulse
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.bits.shape == sent.bits.shape
    assert received.channels == (0, 1, 2)
    k, skip = tx.bits_per_symbol, tx.transient_symbols
    np.testing.assert_array_equal(trim(received.bits, k, skip), trim(sent.bits, k, skip))


def test_exact_rrc_b2b_recovers_symbols_to_rounding():
    tx, rx = wdm_pair(
        modulation_order=64,
        n_symbols=1024,
        polarization="dual",
        seed=2,
        rx_kwargs={"equalize": "none"},
    )
    sent = tx.generate()
    out = rx.receive(sent.field, sent.dt_s).symbols / np.sqrt(tx.launch_power_per_polarization_w)
    # 1e-10: the carrier phase 2πf·t reaches ~1e4 rad, costing a few digits in exp().
    np.testing.assert_allclose(out, sent.symbols, rtol=0, atol=1e-10)


@pytest.mark.parametrize("channels", [(1,), (2, 0), (0, 1, 2, 3, 4)])
def test_selected_channels_come_back_in_the_requested_order(channels):
    tx, rx = wdm_pair(
        n_channels=5,
        modulation_order=4,
        n_symbols=512,
        samples_per_symbol=16,
        seed=3,
        rx_kwargs={"channels": channels},
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.channels == channels
    assert received.bits.shape == (len(channels), sent.bits.shape[-1])
    np.testing.assert_array_equal(received.bits, sent.bits[list(channels)])
    ref_bits, ref_syms = rx.reference(sent)
    np.testing.assert_array_equal(ref_bits, sent.bits[list(channels)])
    np.testing.assert_array_equal(ref_syms, sent.symbols[list(channels)])


def test_dual_polarisation_single_channel_keeps_the_channel_axis():
    tx, rx = wdm_pair(
        modulation_order=4, n_symbols=256, polarization="dual", rx_kwargs={"channels": (1,)}
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.symbols.shape == (2, 1, 256)
    np.testing.assert_array_equal(received.bits, sent.bits[:, [1]])


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_cd_compensation_restores_every_channel_after_a_dispersive_link(polarization, pulse):
    """Regression: CD must be undone on the composite field, before the channel
    downshift — otherwise off-centre channels keep a walk-off of β₂·ω_ch·L
    (≈17 symbols here) and fail completely."""
    span = Fiber.ssmf(80.0, backend="numpy-linear", polarization=polarization).replace(
        alpha_db_per_km=0.0
    )
    link = Link(elements=(span, span))  # 160 km, dispersion only, noiseless
    tx, rx = wdm_pair(
        modulation_order=16,
        n_symbols=1024,
        polarization=polarization,
        seed=5,
        rx_kwargs={"cd_compensation_s2": link.total_dispersion_s2, "equalize": "data_aided"},
        **pulse,
    )
    sent = tx.generate()
    received = rx.receive(link.propagate(sent.field, sent.dt_s), sent.dt_s, reference=sent)
    k, skip = tx.bits_per_symbol, tx.transient_symbols
    np.testing.assert_array_equal(trim(received.bits, k, skip), trim(sent.bits, k, skip))
    if not pulse:
        np.testing.assert_allclose(received.symbols, sent.symbols, rtol=0, atol=1e-9)


def test_pilots_are_recovered_per_channel_and_polarisation():
    cfg = PilotConfig(period=8, symbol=1 + 0j)
    tx, rx = wdm_pair(
        modulation_order=4,
        n_symbols=512,
        polarization="dual",
        seed=0,
        pilots=cfg,
        rx_kwargs={"equalize": "pilot"},
    )
    sent = tx.generate()
    received = rx.receive(sent.field, sent.dt_s)
    assert received.pilot_symbols.shape == (2, 3, cfg.n_pilots(512))
    np.testing.assert_allclose(received.pilot_symbols, 1.0, rtol=0, atol=1e-12)


def test_dt_s_is_required():
    tx, rx = wdm_pair(modulation_order=4, n_symbols=64)
    with pytest.raises(TypeError):
        rx.receive(tx.generate().field)  # type: ignore[call-arg]
