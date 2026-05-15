"""Optical receiver: matched filter + downsample + symbol demap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import nearest_symbol_indices, square_qam, symbols_to_bits

#: How (or whether) to rescale received-symbol amplitudes before hard decision.
EqualizerMode = Literal["none", "unit_power"]


@dataclass(frozen=True)
class ReceivedSymbols:
    """Output of :meth:`Receiver.receive`."""

    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    decisions: npt.NDArray[np.intp]  # constellation index per received symbol
    modulation_order: int


@dataclass(frozen=True)
class Receiver:
    """Single-polarization receiver matched to a :class:`hpcom.transmitter.Transmitter`.

    Applies the matched RRC filter, downsamples at the symbol rate, optionally
    equalizes amplitude, then hard-decodes to nearest constellation point.

    The ``equalize`` option is **explicit** by design. ``"unit_power"`` rescales
    the post-downsample symbols so their measured average power equals 1
    (matching the unit-power QAM grid); this is convenient for noiseless
    benchmarks but it silently absorbs gain/loss errors along the link, which
    can mask bugs in physics tests. ``"none"`` (the default) lets gain/loss
    surface as BER — required for any meaningful AWGN BER curve.
    """

    modulation_order: int
    samples_per_symbol: int = 16
    roll_off: float = 0.1
    pulse_span_symbols: int = 64
    equalize: EqualizerMode = "none"

    def __post_init__(self) -> None:
        if self.equalize not in ("none", "unit_power"):
            raise ValueError(
                f"equalize must be 'none' or 'unit_power', got {self.equalize!r}"
            )

    def receive(self, field: ComplexArray) -> ReceivedSymbols:
        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        n_samples = field.shape[-1]
        # Same delay-compensated full convolution as the Tx, so symbol m
        # remains at index m * sps after the matched filter.
        full = np.convolve(field, rrc, mode="full")
        delay = (n_taps - 1) // 2
        filtered = full[delay : delay + n_samples]

        symbols_rx = filtered[:: self.samples_per_symbol]

        if self.equalize == "unit_power":
            avg_power = float(np.mean(np.abs(symbols_rx) ** 2))
            if avg_power > 0:
                symbols_rx = symbols_rx / np.sqrt(avg_power)

        constellation, _ = square_qam(self.modulation_order)
        decisions = nearest_symbol_indices(symbols_rx, constellation)
        bits = symbols_to_bits(symbols_rx, self.modulation_order)

        return ReceivedSymbols(
            symbols=symbols_rx,
            bits=bits,
            decisions=decisions,
            modulation_order=self.modulation_order,
        )
