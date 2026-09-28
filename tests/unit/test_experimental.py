"""The custom-waveform template: it must satisfy the full Transmitter/Receiver contract."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import b2b_link
from hpcom import EDFA, Fiber, Link, Simulation
from hpcom.experimental import ExperimentalReceiver, ExperimentalTransmission


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_b2b_round_trip(polarization):
    tx = ExperimentalTransmission(
        modulation_order=16, symbol_rate_hz=10e9, n_symbols=512, seed=0, polarization=polarization
    )
    rx = ExperimentalReceiver(modulation_order=16, polarization=polarization)
    sent = tx.generate()
    assert sent.field.shape[-1] == 512 * 8
    received = rx.receive(sent.field, sent.dt_s)
    np.testing.assert_array_equal(received.bits, sent.bits)


def test_field_power_is_the_launch_power():
    tx = ExperimentalTransmission(
        modulation_order=16,
        symbol_rate_hz=10e9,
        n_symbols=8192,
        seed=0,
        launch_power_per_channel_dbm=3.0,
    )
    np.testing.assert_allclose(tx.generate().mean_power_w, 10**0.3 * 1e-3, rtol=0.03)


def test_runs_through_simulation_with_cd_compensation():
    fiber = Fiber.ssmf(80.0, n_steps=20)
    link = Link.from_spans(2, fiber, EDFA(gain_db=16.0, noise_figure_db=4.0))
    tx = ExperimentalTransmission(
        modulation_order=4, symbol_rate_hz=10e9, n_symbols=1024, launch_power_per_channel_dbm=-3.0
    )
    rx = ExperimentalReceiver(
        modulation_order=4, equalize="data_aided", cd_compensation_s2=link.total_dispersion_s2
    )
    result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
    assert result.metrics.ber < 1e-3
    b2b = Simulation(
        transmitter=tx, link=b2b_link(), receiver=rx.replace(cd_compensation_s2=0.0), seed=0
    ).run()
    assert b2b.metrics.ber == 0.0 and b2b.metrics.evm_rms < 1e-12


def test_compatibility_checks():
    tx = ExperimentalTransmission(modulation_order=4, symbol_rate_hz=10e9, n_symbols=16)
    assert ExperimentalReceiver(modulation_order=4).compatibility_issues(tx) == []
    assert "samples_per_symbol differs" in ExperimentalReceiver(
        modulation_order=4, samples_per_symbol=4
    ).compatibility_issues(tx)
