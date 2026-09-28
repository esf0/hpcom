"""T11.2: Transmitter + Receiver accept a ConstellationConfig.

Verifies the new `constellation=` keyword on both Tx and Rx:

* mutual exclusion against `modulation_order`,
* `from_square_qam` produces bit-identical output to the modulation_order path,
* an arbitrary (still-square-QAM) config round-trips at BER == 0 noiselessly,
* and an exotic 8-PSK constellation round-trips B2B.
"""

from __future__ import annotations

import numpy as np
import pytest

from hpcom.modulation import ConstellationConfig
from hpcom.receiver import SingleCarrierReceiver
from hpcom.transmitter import SingleCarrierTransmitter

# ---------------------------------------------------------------------------
# Mutual exclusion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (SingleCarrierTransmitter, {"symbol_rate_hz": 32e9, "n_symbols": 256}),
        (SingleCarrierReceiver, {}),
    ],
)
def test_both_modulation_order_and_constellation_raises(cls, kwargs):
    cfg = ConstellationConfig.from_square_qam(16)
    with pytest.raises(ValueError, match="exactly one"):
        cls(modulation_order=16, constellation=cfg, **kwargs)


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (SingleCarrierTransmitter, {"symbol_rate_hz": 32e9, "n_symbols": 256}),
        (SingleCarrierReceiver, {}),
    ],
)
def test_neither_modulation_order_nor_constellation_raises(cls, kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        cls(**kwargs)


# ---------------------------------------------------------------------------
# from_square_qam path matches the modulation_order path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [4, 16, 64])
def test_constellation_config_b2b_matches_modulation_order(m):
    """Same seed, same constellation factory → byte-identical field."""
    cfg = ConstellationConfig.from_square_qam(m)
    common = dict(
        symbol_rate_hz=32e9, n_symbols=512, samples_per_symbol=8, pulse_span_symbols=16, seed=42
    )

    tx_int = SingleCarrierTransmitter(modulation_order=m, **common)
    tx_cfg = SingleCarrierTransmitter(constellation=cfg, **common)

    sig_int = tx_int.generate()
    sig_cfg = tx_cfg.generate()

    np.testing.assert_array_equal(sig_int.bits, sig_cfg.bits)
    np.testing.assert_array_equal(sig_int.symbols, sig_cfg.symbols)
    np.testing.assert_allclose(sig_int.field, sig_cfg.field, atol=0.0)
    assert sig_cfg.modulation_order == m


# ---------------------------------------------------------------------------
# Noiseless B2B round-trip
# ---------------------------------------------------------------------------


def _trim_rrc_edges(bits: np.ndarray, k: int, span: int) -> np.ndarray:
    """Drop the first/last ``span`` symbols' worth of bits — RRC truncation
    corrupts the matched-filter output there. Matches the convention used by
    the existing dual-pol B2B tests."""
    return bits.reshape(-1, k)[span:-span].ravel()


@pytest.mark.parametrize("m", [4, 16, 64])
def test_b2b_noiseless_recovers_bits_via_constellation_config(m):
    cfg = ConstellationConfig.from_square_qam(m)
    span = 16
    tx = SingleCarrierTransmitter(
        constellation=cfg,
        symbol_rate_hz=32e9,
        n_symbols=1024,
        samples_per_symbol=8,
        pulse_span_symbols=span,
        seed=7,
    )
    rx = SingleCarrierReceiver(constellation=cfg, samples_per_symbol=8, pulse_span_symbols=span)

    sent = tx.generate()
    received = rx.receive(sent.field)

    k = int(np.log2(m))
    np.testing.assert_array_equal(
        _trim_rrc_edges(received.bits, k, span),
        _trim_rrc_edges(sent.bits, k, span),
    )


def test_b2b_with_8psk_constellation():
    """Arbitrary 8-PSK geometric constellation round-trips noiselessly."""
    raw_points = np.exp(2j * np.pi * np.arange(8) / 8)
    cfg = ConstellationConfig.from_points(raw_points)
    span = 16
    tx = SingleCarrierTransmitter(
        constellation=cfg,
        symbol_rate_hz=32e9,
        n_symbols=512,
        samples_per_symbol=8,
        pulse_span_symbols=span,
        seed=11,
    )
    rx = SingleCarrierReceiver(constellation=cfg, samples_per_symbol=8, pulse_span_symbols=span)

    sent = tx.generate()
    received = rx.receive(sent.field)

    np.testing.assert_array_equal(
        _trim_rrc_edges(received.bits, 3, span),
        _trim_rrc_edges(sent.bits, 3, span),
    )
    assert sent.modulation_order == 8


def test_b2b_dual_pol_with_constellation_config():
    cfg = ConstellationConfig.from_square_qam(16)
    span = 16
    tx = SingleCarrierTransmitter(
        constellation=cfg,
        symbol_rate_hz=32e9,
        n_symbols=512,
        samples_per_symbol=8,
        pulse_span_symbols=span,
        polarization="dual",
        seed=3,
    )
    rx = SingleCarrierReceiver(
        constellation=cfg, samples_per_symbol=8, pulse_span_symbols=span, polarization="dual"
    )

    sent = tx.generate()
    received = rx.receive(sent.field)

    for pol in (0, 1):
        np.testing.assert_array_equal(
            _trim_rrc_edges(received.bits[pol], 4, span),
            _trim_rrc_edges(sent.bits[pol], 4, span),
        )


# ---------------------------------------------------------------------------
# effective_modulation_order is always present
# ---------------------------------------------------------------------------


def test_effective_modulation_order_via_int():
    tx = SingleCarrierTransmitter(modulation_order=64, symbol_rate_hz=32e9, n_symbols=64)
    assert tx.effective_modulation_order == 64


def test_effective_modulation_order_via_constellation():
    cfg = ConstellationConfig.from_points(np.exp(2j * np.pi * np.arange(8) / 8))
    tx = SingleCarrierTransmitter(constellation=cfg, symbol_rate_hz=32e9, n_symbols=64)
    assert tx.effective_modulation_order == 8
