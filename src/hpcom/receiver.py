"""Optical receiver: matched filter + downsample + symbol demap.

Mirrors the Transmitter's three composable feature axes: polarisation (single
or dual), WDM demux (when paired with a WDMConfig), and pilot extraction.
For dual-pol input the receiver demodulates each polarisation independently
and returns 2-D ``(2, …)`` bits/symbols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from hpcom._pulse import root_raised_cosine
from hpcom._typing import ComplexArray
from hpcom.modulation import bits_per_symbol, nearest_symbol_indices, square_qam, symbols_to_bits
from hpcom.pilots import PilotConfig
from hpcom.transmitter import Polarization
from hpcom.wdm import WDMConfig

#: How (or whether) to rescale received-symbol amplitudes before hard decision.
EqualizerMode = Literal["none", "unit_power"]


@dataclass(frozen=True)
class ReceivedSymbols:
    """Output of :meth:`Receiver.receive`.

    When ``pilots`` is set, ``pilot_symbols`` is the post-matched-filter
    complex amplitudes at the pilot positions (no estimation applied) and the
    bits at pilot positions in ``bits`` are kept (zero-equivalent under
    hard-decision demap; use :meth:`data_bits` for the pilot-stripped view).
    """

    symbols: ComplexArray
    bits: npt.NDArray[np.uint8]
    decisions: npt.NDArray[np.intp]  # constellation index per received symbol
    modulation_order: int
    polarization: Polarization = "single"
    pilots: PilotConfig | None = None
    pilot_symbols: ComplexArray | None = None

    def data_bits(self) -> npt.NDArray[np.uint8]:
        """Bits at data positions only — Option α convention.

        Matches the shape conventions of :class:`TransmittedSignal.data_bits`:
        1-D for single-pol single-channel, 2-D for dual-pol or WDM.
        """
        if self.pilots is None:
            return self.bits
        k = bits_per_symbol(self.modulation_order)
        n_symbols = self.symbols.shape[-1]
        data_idx = self.pilots.data_indices(n_symbols)
        if self.bits.ndim == 1:
            return self.bits.reshape(-1, k)[data_idx].ravel()
        per_stream = self.bits.reshape(self.bits.shape[0], -1, k)
        return per_stream[:, data_idx, :].reshape(self.bits.shape[0], -1)

    def data_symbols(self) -> ComplexArray:
        if self.pilots is None:
            return self.symbols
        n_symbols = self.symbols.shape[-1]
        return self.symbols[..., self.pilots.data_indices(n_symbols)]


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
    cd_compensation_s2: float = 0.0
    """Accumulated dispersion ``β₂ · L_total`` (in s²) to undo before the
    matched filter. Set to ``fiber.beta2_s2_per_m * fiber.length_km * 1e3 * n_spans``
    to remove the chromatic dispersion of an EDFA-compensated link. Leave at
    ``0.0`` for back-to-back / dispersion-free scenarios."""
    polarization: Polarization = "single"
    """Match the Transmitter's polarisation: ``"dual"`` expects a ``(2, N)``
    field input and returns 2-D bits/symbols."""
    wdm: WDMConfig | None = None
    """If set, the receiver demuxes a single channel from a WDM-multiplexed
    field. Pair with :attr:`wdm_channel` to pick which channel."""
    wdm_channel: int = 0
    """Index of the WDM channel to demux. Ignored when :attr:`wdm` is ``None``."""
    pilots: PilotConfig | None = None
    """If set, post-matched-filter samples at pilot positions are returned as
    :attr:`ReceivedSymbols.pilot_symbols` and stripped from :meth:`data_bits`."""

    def __post_init__(self) -> None:
        if self.equalize not in ("none", "unit_power"):
            raise ValueError(
                f"equalize must be 'none' or 'unit_power', got {self.equalize!r}"
            )
        if self.polarization not in ("single", "dual"):
            raise ValueError(
                f"polarization must be 'single' or 'dual', got {self.polarization!r}"
            )
        if self.polarization == "dual" and self.wdm is not None:
            raise NotImplementedError(
                "dual-pol × WDM demux is handled by WDMReceiver (planned subclass); "
                "for now use polarization='single' when wdm is set."
            )
        if self.wdm is not None and not 0 <= self.wdm_channel < self.wdm.n_channels:
            raise ValueError(
                f"wdm_channel must be in [0, {self.wdm.n_channels}), "
                f"got {self.wdm_channel}"
            )

    def receive(self, field: ComplexArray, dt_s: float | None = None) -> ReceivedSymbols:
        """Demodulate a received field.

        For ``polarization="single"`` the input is 1-D ``(n_samples,)``; for
        ``"dual"`` it is ``(2, n_samples)`` and each polarisation is demodulated
        independently.

        ``dt_s`` (sample period in seconds) is required when WDM demux is
        enabled or when :attr:`cd_compensation_s2` is non-zero.
        """
        if self.polarization == "single":
            if field.ndim != 1:
                raise ValueError(
                    f"single-pol receiver expects 1-D field, got shape {field.shape}"
                )
            return self._receive_one_stream(field, dt_s, demux_wdm=True)

        # Dual-pol: input must be (2, N).
        if field.ndim != 2 or field.shape[0] != 2:
            raise ValueError(
                f"dual-pol receiver expects 2-D field with leading axis 2, "
                f"got shape {field.shape}"
            )
        per_pol = [
            self._receive_one_stream(field[i], dt_s, demux_wdm=False)
            for i in (0, 1)
        ]
        # Stack the 1-D outputs into 2-D arrays.
        pilot_syms: ComplexArray | None
        if per_pol[0].pilot_symbols is not None:
            pilot_syms = np.stack(
                [p.pilot_symbols for p in per_pol], axis=0  # type: ignore[misc]
            )
        else:
            pilot_syms = None
        return ReceivedSymbols(
            symbols=np.stack([p.symbols for p in per_pol], axis=0),
            bits=np.stack([p.bits for p in per_pol], axis=0),
            decisions=np.stack([p.decisions for p in per_pol], axis=0),
            modulation_order=self.modulation_order,
            polarization="dual",
            pilots=self.pilots,
            pilot_symbols=pilot_syms,
        )

    def _receive_one_stream(
        self,
        field: ComplexArray,
        dt_s: float | None,
        *,
        demux_wdm: bool,
    ) -> ReceivedSymbols:
        """Single 1-D demodulation pipeline; used directly for single-pol and
        once per polarisation for dual-pol. ``demux_wdm=False`` skips the WDM
        downshift step (dual-pol × WDM is rejected earlier)."""
        if demux_wdm and self.wdm is not None:
            if dt_s is None:
                raise ValueError(
                    "dt_s is required when wdm is set (needed to downshift "
                    "the selected channel to baseband)"
                )
            f_ch = float(self.wdm.channel_frequencies_hz()[self.wdm_channel])
            n = field.shape[-1]
            t = np.arange(n) * dt_s
            field = field * np.exp(-2j * np.pi * f_ch * t)

        if self.cd_compensation_s2 != 0.0:
            if dt_s is None:
                raise ValueError(
                    "dt_s is required when cd_compensation_s2 != 0 "
                    "(needed for the frequency grid)"
                )
            field = self._undo_dispersion(field, dt_s)

        n_taps = self.pulse_span_symbols * self.samples_per_symbol + 1
        rrc = root_raised_cosine(n_taps, self.samples_per_symbol, self.roll_off)
        n_samples = field.shape[-1]
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

        pilot_symbols: ComplexArray | None = None
        if self.pilots is not None:
            n_sym = symbols_rx.shape[-1]
            pilot_symbols = symbols_rx[self.pilots.pilot_indices(n_sym)]

        return ReceivedSymbols(
            symbols=symbols_rx,
            bits=bits,
            decisions=decisions,
            modulation_order=self.modulation_order,
            polarization="single",
            pilots=self.pilots,
            pilot_symbols=pilot_symbols,
        )

    def _undo_dispersion(self, field: ComplexArray, dt_s: float) -> ComplexArray:
        """Undo accumulated linear chromatic dispersion (β₂ · L) on the field.

        Forward dispersion in :class:`hpcom.backends.numpy.linear.LinearNumpyPropagator`
        applies ``exp(+i β₂/2 ω² L)`` in the frequency domain (NumPy FFT
        convention); the inverse is ``exp(-i β₂/2 ω² L)``.
        """
        n = field.shape[-1]
        omega = 2.0 * np.pi * np.fft.fftfreq(n, d=dt_s)
        spectrum = np.fft.fft(field, axis=-1)
        spectrum = spectrum * np.exp(-1j * self.cd_compensation_s2 / 2.0 * omega**2)
        out: ComplexArray = np.fft.ifft(spectrum, axis=-1)
        return out
