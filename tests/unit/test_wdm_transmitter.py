"""WDM transmitter: shapes, stream independence, deterministic input, pilots, power."""

from __future__ import annotations

import numpy as np
import pytest

from helpers import PULSE_MODES, wdm_pair
from hpcom import PilotConfig, SingleCarrierTransmitter
from hpcom.units import dbm_to_w


@pytest.mark.parametrize(
    ("polarization", "field_shape", "bits_shape"),
    [("single", (8192,), (3, 2048)), ("dual", (2, 8192), (2, 3, 2048))],
)
def test_output_shapes(polarization, field_shape, bits_shape):
    tx, _ = wdm_pair(modulation_order=4, n_symbols=1024, polarization=polarization, seed=0)
    sent = tx.generate()
    assert sent.field.shape == field_shape
    assert sent.bits.shape == bits_shape
    assert sent.symbols.shape == (*bits_shape[:-1], 1024)
    assert tx.stream_shape == bits_shape[:-1]


def test_every_polarisation_and_channel_carries_independent_data():
    tx, _ = wdm_pair(modulation_order=16, n_symbols=512, polarization="dual", seed=0)
    streams = tx.generate().bits.reshape(6, -1)
    for a in range(6):
        for b in range(a + 1, 6):
            assert not np.array_equal(streams[a], streams[b])


def test_first_stream_equals_single_carrier_stream_with_same_seed():
    tx, _ = wdm_pair(modulation_order=16, n_symbols=512, seed=4)
    sc = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=512, seed=4)
    np.testing.assert_array_equal(tx.generate().bits[0], sc.generate().bits)


def test_seed_reproducible():
    tx, _ = wdm_pair(modulation_order=4, n_symbols=256, polarization="dual", seed=7)
    a, b = tx.generate(), tx.generate()
    np.testing.assert_array_equal(a.field, b.field)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_deterministic_bits_and_symbols(polarization):
    tx, _ = wdm_pair(modulation_order=4, n_symbols=128, polarization=polarization)
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, (*tx.stream_shape, tx.n_bits_per_stream), dtype=np.uint8)
    np.testing.assert_array_equal(tx.generate(bits=bits).bits, bits)
    symbols = tx.resolved_constellation.points[rng.integers(0, 4, (*tx.stream_shape, 128))]
    from_symbols = tx.generate(symbols=symbols)
    np.testing.assert_array_equal(from_symbols.symbols, symbols)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_wrongly_shaped_input_is_rejected(polarization):
    tx, _ = wdm_pair(modulation_order=4, n_symbols=128, polarization=polarization)
    with pytest.raises(ValueError, match="`bits` must have shape"):
        tx.generate(bits=np.zeros(256, dtype=np.uint8))
    with pytest.raises(ValueError, match="`symbols` must have shape"):
        tx.generate(symbols=np.zeros((5, 128), dtype=complex))


def test_pilots_on_every_channel_and_polarisation():
    cfg = PilotConfig(period=8, symbol=0.5 + 0.5j)
    tx, _ = wdm_pair(modulation_order=4, n_symbols=256, polarization="dual", seed=0, pilots=cfg)
    sent = tx.generate()
    np.testing.assert_array_equal(sent.symbols[..., cfg.pilot_indices(256)], cfg.symbol)
    assert sent.bits.shape == (2, 3, cfg.n_data(256) * 2)


@pytest.mark.parametrize("pulse", PULSE_MODES)
@pytest.mark.parametrize("n_channels", [2, 3, 5])
def test_composite_power_is_n_channels_times_the_per_channel_launch_power(n_channels, pulse):
    tx, _ = wdm_pair(
        n_channels=n_channels,
        modulation_order=16,
        n_symbols=4096,
        samples_per_symbol=16,
        seed=0,
        launch_power_per_channel_dbm=-2.0,
        **pulse,
    )
    np.testing.assert_allclose(tx.generate().mean_power_w, n_channels * dbm_to_w(-2.0), rtol=0.03)
    assert tx.launch_power_total_w == pytest.approx(n_channels * dbm_to_w(-2.0))


def test_total_launch_power_is_shared_equally():
    tx, _ = wdm_pair(
        n_channels=5,
        modulation_order=4,
        n_symbols=512,
        samples_per_symbol=16,
        launch_power_total_dbm=10.0,
    )
    assert tx.launch_power_per_channel_w == pytest.approx(dbm_to_w(10.0) / 5)
