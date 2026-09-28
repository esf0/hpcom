"""Shared builders for the test suite (matched Tx/Rx pairs, trivial links, theory)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy.special import erfc

from hpcom import (
    EDFA,
    Link,
    SingleCarrierReceiver,
    SingleCarrierTransmitter,
    WDMConfig,
    WDMReceiver,
    WDMTransmitter,
)

#: The three pulse-shaping implementations; every SC/WDM behaviour is tested in each.
PULSE_MODES = [
    pytest.param({}, id="exact-circular"),
    pytest.param({"pulse_span_symbols": 32}, id="fir-circular"),
    pytest.param({"pulse_span_symbols": 32, "convolution": "linear"}, id="fir-linear"),
]

_PULSE_KEYS = ("samples_per_symbol", "roll_off", "pulse_span_symbols", "convolution", "pilots")
_MOD_KEYS = ("modulation_order", "constellation", "dm", "polarization")


def matched_receiver_kwargs(tx: Any, **overrides: Any) -> dict[str, Any]:
    """Receiver kwargs that mirror ``tx``'s modulation, polarisation and pulse."""
    kw = {k: getattr(tx, k) for k in _MOD_KEYS + _PULSE_KEYS if hasattr(tx, k)}
    kw = {k: v for k, v in kw.items() if v is not None or k in ("pulse_span_symbols",)}
    kw.update(overrides)
    return kw


def sc_pair(
    *,
    modulation_order: int | None = 16,
    n_symbols: int = 1024,
    rx_kwargs: dict[str, Any] | None = None,
    **tx_kwargs: Any,
) -> tuple[SingleCarrierTransmitter, SingleCarrierReceiver]:
    """Matched single-carrier transmitter/receiver."""
    tx_kwargs.setdefault("symbol_rate_hz", 32e9)
    tx_kwargs.setdefault("samples_per_symbol", 4)
    if modulation_order is not None and "constellation" not in tx_kwargs:
        tx_kwargs["modulation_order"] = modulation_order
    tx = SingleCarrierTransmitter(n_symbols=n_symbols, **tx_kwargs)
    rx = SingleCarrierReceiver(**matched_receiver_kwargs(tx, **(rx_kwargs or {})))
    return tx, rx


def wdm_pair(
    *,
    n_channels: int = 3,
    spacing_hz: float = 50e9,
    modulation_order: int = 16,
    n_symbols: int = 1024,
    rx_kwargs: dict[str, Any] | None = None,
    **tx_kwargs: Any,
) -> tuple[WDMTransmitter, WDMReceiver]:
    """Matched WDM transmitter/receiver (all channels received)."""
    tx_kwargs.setdefault("symbol_rate_hz", 32e9)
    tx_kwargs.setdefault("samples_per_symbol", 8)
    wdm = WDMConfig(n_channels=n_channels, spacing_hz=spacing_hz)
    tx = WDMTransmitter(
        modulation_order=modulation_order, n_symbols=n_symbols, wdm=wdm, **tx_kwargs
    )
    rx = WDMReceiver(wdm=wdm, **matched_receiver_kwargs(tx, **(rx_kwargs or {})))
    return tx, rx


def b2b_link() -> Link:
    """Noiseless unit-gain 'link' (back-to-back)."""
    return Link(elements=(EDFA(gain_db=0.0),))


def trim(arr: np.ndarray, per_symbol: int, skip: int) -> np.ndarray:
    """Drop ``skip`` symbols (``per_symbol`` entries each) at both ends of the last axis."""
    if not skip:
        return arr
    return arr[..., skip * per_symbol : arr.shape[-1] - skip * per_symbol]


def q_function(x: np.ndarray | float) -> np.ndarray | float:
    return 0.5 * erfc(np.asarray(x) / np.sqrt(2.0))


def ber_square_qam_awgn(snr_linear: float, m: int) -> float:
    """Exact BER of Gray-coded square M-QAM on AWGN at per-symbol SNR ``Es/N0``.

    Cho & Yoon, "On the general BER expression of one- and two-dimensional
    amplitude modulations", IEEE Trans. Commun. 50(7), 2002, eq. (16). Each
    quadrature is a √M-PAM with ``Es/2`` energy.
    """
    side = math.isqrt(m)
    k_axis = int(np.log2(side))
    total = 0.0
    for kk in range(1, k_axis + 1):
        s = 0.0
        for i in range(int((1 - 2.0 ** (-kk)) * side)):
            w = (-1) ** int(i * 2 ** (kk - 1) / side) * (
                2 ** (kk - 1) - int(i * 2 ** (kk - 1) / side + 0.5)
            )
            s += w * erfc((2 * i + 1) * np.sqrt(3.0 * snr_linear / (2.0 * (m - 1))))
        total += s / side
    return float(total / k_axis)
