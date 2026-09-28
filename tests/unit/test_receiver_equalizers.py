"""Receiver equaliser modes, chromatic-dispersion compensation, compatibility checks."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from helpers import sc_pair
from hpcom import (
    Fiber,
    OFDMConfig,
    OFDMTransmitter,
    PilotConfig,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
)
from hpcom.modulation import apsk

GAIN = 0.37 * np.exp(1j * 1.1)  # an arbitrary link gain and phase rotation


@pytest.fixture
def scenario():
    tx, _ = sc_pair(
        modulation_order=16,
        n_symbols=2048,
        seed=0,
        pilots=PilotConfig(period=16),
        launch_power_per_channel_dbm=-3.0,
    )
    sent = tx.generate()
    return tx, sent, sent.field * GAIN


def _rx(tx, **kw):
    return SingleCarrierReceiver(
        modulation_order=16, samples_per_symbol=tx.samples_per_symbol, pilots=tx.pilots, **kw
    )


def test_none_returns_raw_samples(scenario):
    tx, sent, field = scenario
    out = _rx(tx, equalize="none").receive(field, sent.dt_s)
    expected = sent.data_symbols() * GAIN * np.sqrt(tx.launch_power_per_polarization_w)
    np.testing.assert_allclose(out.symbols, expected, rtol=0, atol=1e-14)
    assert out.gain == 1.0


def test_unit_power_fixes_amplitude_but_not_phase(scenario):
    tx, sent, field = scenario
    out = _rx(tx, equalize="unit_power").receive(field, sent.dt_s)
    np.testing.assert_allclose(np.mean(np.abs(out.symbols) ** 2), 1.0, rtol=1e-12)
    assert np.angle(out.gain) == 0.0
    np.testing.assert_allclose(
        np.angle(np.vdot(sent.data_symbols(), out.symbols)), np.angle(GAIN), atol=1e-2
    )


@pytest.mark.parametrize("mode", ["pilot", "data_aided"])
def test_pilot_and_data_aided_remove_gain_and_phase_exactly(scenario, mode):
    tx, sent, field = scenario
    out = _rx(tx, equalize=mode).receive(field, sent.dt_s, reference=sent)
    np.testing.assert_allclose(out.symbols, sent.data_symbols(), rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        out.gain, GAIN * np.sqrt(tx.launch_power_per_polarization_w), rtol=1e-12
    )
    np.testing.assert_array_equal(out.bits, sent.bits)


def test_data_aided_gain_is_the_least_squares_estimate_under_noise(scenario):
    tx, sent, field = scenario
    rng = np.random.default_rng(1)
    noisy = field + 0.02 * np.abs(field).max() * (
        rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
    )
    rx = _rx(tx, equalize="data_aided")
    raw = _rx(tx, equalize="none").receive(noisy, sent.dt_s).symbols
    a = sent.data_symbols()
    ls = np.vdot(a, raw) / np.vdot(a, a)
    np.testing.assert_allclose(rx.receive(noisy, sent.dt_s, reference=sent).gain, ls, rtol=1e-12)


def test_data_aided_needs_the_reference(scenario):
    tx, sent, field = scenario
    with pytest.raises(ValueError, match="reference=sent"):
        _rx(tx, equalize="data_aided").receive(field, sent.dt_s)


def test_data_aided_reference_must_match(scenario):
    tx, sent, field = scenario
    other = tx.replace(n_symbols=1024).generate()
    with pytest.raises(ValueError, match="reference data symbols have shape"):
        _rx(tx, equalize="data_aided").receive(field, sent.dt_s, reference=other)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_cd_compensation_undoes_beta2_and_beta3_exactly(polarization):
    fiber = Fiber(
        length_km=500.0,
        alpha_db_per_km=0.0,
        beta2_s2_per_m=-2.17e-26,
        beta3_s3_per_m=1.28e-40,
        backend="numpy-linear",
        polarization=polarization,
    )
    tx, rx0 = sc_pair(
        modulation_order=64, n_symbols=4096, polarization=polarization, seed=2, samples_per_symbol=2
    )
    rx = rx0.replace(
        cd_compensation_s2=fiber.accumulated_dispersion_s2,
        cd_compensation_s3=fiber.accumulated_beta3_s3,
        equalize="none",
    )
    sent = tx.generate()
    out = rx.receive(fiber.propagate(sent.field, sent.dt_s), sent.dt_s)
    np.testing.assert_allclose(
        out.symbols / np.sqrt(tx.launch_power_per_polarization_w), sent.symbols, rtol=0, atol=1e-10
    )
    # Compensating β₂ only leaves the β₃ distortion.
    partial = rx.replace(cd_compensation_s3=0.0).receive(
        fiber.propagate(sent.field, sent.dt_s), sent.dt_s
    )
    assert (
        np.max(np.abs(partial.symbols / np.sqrt(tx.launch_power_per_polarization_w) - sent.symbols))
        > 1e-3
    )


# ---- compatibility checks -----------------------------------------------------


def test_matched_pair_is_compatible():
    tx, rx = sc_pair(modulation_order=16, n_symbols=64)
    assert rx.compatibility_issues(tx) == []


@pytest.mark.parametrize(
    ("rx_change", "word"),
    [
        ({"modulation_order": 64}, "constellations differ"),
        ({"samples_per_symbol": 8}, "samples_per_symbol"),
        ({"roll_off": 0.2}, "roll_off"),
        ({"pulse_span_symbols": 16}, "pulse_span_symbols"),
        ({"pilots": PilotConfig(period=4)}, "pilots"),
        ({"polarization": "dual"}, "polarization"),
        ({"constellation": apsk((4, 12), (1.0, 2.8))}, "constellations differ"),
    ],
)
def test_mismatches_are_reported(rx_change, word):
    tx, rx = sc_pair(modulation_order=16, n_symbols=64, samples_per_symbol=4)
    assert any(word in issue for issue in rx.replace(**rx_change).compatibility_issues(tx))


def test_waveform_mismatch_is_reported():
    rx = SingleCarrierReceiver(modulation_order=4)
    ofdm_tx = OFDMTransmitter(
        modulation_order=4, subcarrier_spacing_hz=1e9, n_frames=2, ofdm=OFDMConfig(n_subcarriers=16)
    )
    assert rx.compatibility_issues(ofdm_tx) == [
        "SingleCarrierReceiver cannot receive a OFDMTransmitter"
    ]


def test_receiver_modulation_validation():
    with pytest.raises(ValidationError, match="exactly one"):
        SingleCarrierReceiver()
    with pytest.raises(ValidationError, match="square_qam supports"):
        SingleCarrierReceiver(modulation_order=12)
    with pytest.raises(ValidationError, match="square_qam supports"):
        SingleCarrierTransmitter(modulation_order=8, symbol_rate_hz=1e9, n_symbols=8)


# ---- "auto" resolution ---------------------------------------------------------


@pytest.mark.parametrize(
    ("pilots", "with_reference", "expected"),
    [
        (False, False, "unit_power"),
        (False, True, "data_aided"),
        (True, False, "pilot"),
        (True, True, "pilot"),  # a real receiver's own pilots beat the genie
    ],
)
def test_auto_equaliser_resolution(pilots, with_reference, expected):
    tx, rx = sc_pair(n_symbols=512, seed=0, pilots=PilotConfig(period=8) if pilots else None)
    assert rx.equalize == "auto"
    sent = tx.generate()
    out = rx.receive(sent.field * GAIN, sent.dt_s, reference=sent if with_reference else None)
    assert out.equalizer == expected
    if expected != "unit_power":
        np.testing.assert_array_equal(out.bits, sent.bits)


@pytest.mark.parametrize(("pilots", "expected"), [((4, 20, 44, 60), "pilot"), ((), "data_aided")])
def test_auto_equaliser_resolution_for_ofdm(pilots, expected):
    from hpcom import OFDMReceiver

    cfg = OFDMConfig(n_subcarriers=64, cp_len=8, n_guard=2, pilot_subcarrier_indices=pilots)
    tx = OFDMTransmitter(
        modulation_order=16, subcarrier_spacing_hz=1e9, n_frames=8, ofdm=cfg, seed=0
    )
    sent = tx.generate()
    out = OFDMReceiver(modulation_order=16, ofdm=cfg).receive(
        sent.field * GAIN, sent.dt_s, reference=sent
    )
    assert out.equalizer == expected
    np.testing.assert_array_equal(out.bits, sent.bits)


def test_explicit_modes_are_reported_unchanged(scenario):
    tx, sent, field = scenario
    for mode in ("none", "unit_power", "pilot", "data_aided"):
        assert _rx(tx, equalize=mode).receive(field, sent.dt_s, reference=sent).equalizer == mode
