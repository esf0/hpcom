"""WDM (wavelength-division multiplexing) configuration, transmitter, signal.

A :class:`WDMConfig` describes a uniform grid of ``n_channels`` channels with
constant ``spacing_hz`` between adjacent channel centers, centered on the
optical carrier (baseband 0 Hz). :class:`WDMTransmitter` superimposes
per-channel RRC-shaped baseband fields onto this grid; the receiver
(:class:`hpcom.receiver.WDMReceiver`) downshifts a single selected channel
back to baseband before its matched filter.

For ``n_channels = N`` and spacing :math:`\\Delta f`, the channel center
frequencies (Hz, baseband) are

.. math:: f_i = (i - (N-1)/2) \\cdot \\Delta f, \\quad i = 0, ..., N-1

so the grid straddles 0 Hz symmetrically (with a center channel exactly at 0
for odd ``N``).

:class:`WDMTransmitter` inherits from :class:`SingleCarrierTransmitter`
because every per-channel stream is itself a single-carrier RRC-shaped
signal; WDM only adds the frequency-shift + sum composition on top.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hpcom._typing import ComplexArray
from hpcom.receiver import (
    ReceivedSymbols,
    SingleCarrierReceivedSymbols,
    SingleCarrierReceiver,
)
from hpcom.transmitter import (
    Polarization,
    SingleCarrierSignal,
    SingleCarrierTransmitter,
    _seed_seq_from,
)


class WDMConfig(BaseModel):
    """Uniform WDM grid: ``n_channels`` channels at constant ``spacing_hz``."""

    model_config = ConfigDict(frozen=True)

    n_channels: int = Field(
        ge=2,
        description="Number of WDM channels (>= 2; use SingleCarrierTransmitter "
                    "for one channel).",
    )
    spacing_hz: float = Field(gt=0, description="Inter-channel spacing in Hz.")

    def channel_frequencies_hz(self) -> npt.NDArray[np.float64]:
        """Per-channel center frequencies (Hz, baseband), centered on 0 Hz."""
        idx = np.arange(self.n_channels, dtype=np.float64)
        return (idx - (self.n_channels - 1) / 2.0) * self.spacing_hz

    @property
    def total_bandwidth_hz(self) -> float:
        """Outer-to-outer channel-center span, in Hz. (Channel bandwidth is
        added separately by the caller based on symbol rate and roll-off.)"""
        return float((self.n_channels - 1) * self.spacing_hz)


# ---------------------------------------------------------------------------
# WDMSignal
# ---------------------------------------------------------------------------


class WDMSignal(SingleCarrierSignal):
    """Wideband WDM signal: composite field carrying multiple QAM channels.

    Shape conventions:

    * Single-pol: ``field=(n_samples,)`` composite,
      ``bits=(n_channels, n_symbols*k)``,
      ``symbols=(n_channels, n_symbols)``.
    * Dual-pol:   ``field=(2, n_samples)`` composite,
      ``bits=(2, n_channels, n_symbols*k)``,
      ``symbols=(2, n_channels, n_symbols)``.
    """

    wdm: WDMConfig


# ---------------------------------------------------------------------------
# WDMTransmitter
# ---------------------------------------------------------------------------


class WDMTransmitter(SingleCarrierTransmitter):
    """Wavelength-division-multiplexed transmitter.

    Generates ``wdm.n_channels`` independent QAM streams (per polarisation),
    frequency-shifts each onto the configured grid, and sums them into a
    composite wideband field. With ``polarization="dual"``, each polarisation
    independently carries the full N-channel composite; X and Y streams are
    independent (PDM-WDM).
    """

    wdm: WDMConfig

    @model_validator(mode="after")
    def _validate_wdm_bandwidth(self) -> WDMTransmitter:
        # Wideband Nyquist: sample rate must cover the outer channel edges.
        # Per-channel one-sided bandwidth is Rs · (1 + roll_off) / 2.
        sample_rate_hz = self.symbol_rate_hz * self.samples_per_symbol
        outer_center = abs(self.wdm.channel_frequencies_hz()[-1])
        needed = 2.0 * (
            outer_center + self.symbol_rate_hz * (1.0 + self.roll_off) / 2.0
        )
        if sample_rate_hz < needed:
            raise ValueError(
                f"WDM wideband sample rate ({sample_rate_hz:.3e} Hz) is below "
                f"the Nyquist requirement ({needed:.3e} Hz) for "
                f"{self.wdm.n_channels} channels at {self.wdm.spacing_hz:.3e} Hz "
                f"spacing with roll-off {self.roll_off}. Increase "
                f"samples_per_symbol (or reduce spacing/channel count)."
            )
        return self

    def generate(
        self,
        *,
        rng: np.random.Generator | None = None,
        bits: npt.NDArray[np.uint8] | None = None,
        symbols: ComplexArray | None = None,
    ) -> WDMSignal:
        """Generate the WDM composite field.

        Deterministic-input shapes (single carrier per polarisation per channel):

        - single-pol: ``bits=(n_channels, n_symbols*k)``,
          ``symbols=(n_channels, n_symbols)``.
        - dual-pol:   ``bits=(2, n_channels, n_symbols*k)``,
          ``symbols=(2, n_channels, n_symbols)``.
        """
        if bits is not None and symbols is not None:
            raise ValueError("pass at most one of `bits` or `symbols`, not both")

        n_ch = self.wdm.n_channels
        bps = self._bits_per_stream()
        n_pol = 2 if self.polarization == "dual" else 1

        # Validate deterministic-input shapes.
        if bits is not None:
            b = np.asarray(bits, dtype=np.uint8)
            expected = (
                (2, n_ch, bps) if n_pol == 2 else (n_ch, bps)
            )
            if b.shape != expected:
                raise ValueError(
                    f"`bits` must have shape {expected}, got {b.shape}"
                )
        if symbols is not None:
            s = np.asarray(symbols, dtype=np.complex128)
            expected_s = (
                (2, n_ch, self.n_symbols) if n_pol == 2 else (n_ch, self.n_symbols)
            )
            if s.shape != expected_s:
                raise ValueError(
                    f"`symbols` must have shape {expected_s}, got {s.shape}"
                )

        # Seed tree: root → per-pol → per-channel.
        seed_seq = _seed_seq_from(rng, self.seed)
        pol_seqs = seed_seq.spawn(n_pol) if n_pol == 2 else [seed_seq]

        n_samples = self.n_symbols * self.samples_per_symbol
        t = np.arange(n_samples) * self.dt_s
        freqs = self.wdm.channel_frequencies_hz()

        per_pol_field: list[ComplexArray] = []
        per_pol_bits: list[npt.NDArray[np.uint8]] = []
        per_pol_syms: list[ComplexArray] = []

        for pol_idx, pol_seq in enumerate(pol_seqs):
            ch_sub_seqs = pol_seq.spawn(n_ch)
            composite = np.zeros(n_samples, dtype=np.complex128)
            this_pol_bits: list[npt.NDArray[np.uint8]] = []
            this_pol_syms: list[ComplexArray] = []
            for ch_idx, (f_ch, sub_seq) in enumerate(
                zip(freqs, ch_sub_seqs, strict=True)
            ):
                ch_rng = np.random.default_rng(sub_seq)
                if bits is not None:
                    b_in = bits[pol_idx, ch_idx] if n_pol == 2 else bits[ch_idx]
                else:
                    b_in = None
                if symbols is not None:
                    s_in = symbols[pol_idx, ch_idx] if n_pol == 2 else symbols[ch_idx]
                else:
                    s_in = None
                ch_bits, ch_syms, field_ch = self._generate_one_channel(
                    ch_rng, bits=b_in, symbols=s_in
                )
                this_pol_bits.append(ch_bits)
                this_pol_syms.append(ch_syms)
                composite += field_ch * np.exp(2j * np.pi * float(f_ch) * t)
            per_pol_field.append(composite)
            per_pol_bits.append(np.stack(this_pol_bits, axis=0))
            per_pol_syms.append(np.stack(this_pol_syms, axis=0))

        if n_pol == 1:
            return self._make_wdm_signal(
                field=per_pol_field[0],
                symbols=per_pol_syms[0],
                bits=per_pol_bits[0],
                polarization="single",
            )

        return self._make_wdm_signal(
            field=np.stack(per_pol_field, axis=0),
            symbols=np.stack(per_pol_syms, axis=0),
            bits=np.stack(per_pol_bits, axis=0),
            polarization="dual",
        )

    def _make_wdm_signal(
        self,
        *,
        field: ComplexArray,
        symbols: ComplexArray,
        bits: npt.NDArray[np.uint8],
        polarization: Polarization,
    ) -> WDMSignal:
        return WDMSignal(
            field=field,
            dt_s=self.dt_s,
            symbols=symbols,
            bits=bits,
            samples_per_symbol=self.samples_per_symbol,
            modulation_order=self.effective_modulation_order,
            roll_off=self.roll_off,
            pulse_span_symbols=self.pulse_span_symbols,
            n_symbols=self.n_symbols,
            polarization=polarization,
            pilots=self.pilots,
            wdm=self.wdm,
        )


# ---------------------------------------------------------------------------
# WDMReceivedSymbols
# ---------------------------------------------------------------------------


class WDMReceivedSymbols(SingleCarrierReceivedSymbols):
    """Output of :class:`WDMReceiver` — single demuxed channel's bits / symbols,
    annotated with which channel was scored."""

    wdm_channel: int


# ---------------------------------------------------------------------------
# WDMReceiver
# ---------------------------------------------------------------------------


class WDMReceiver(SingleCarrierReceiver):
    """WDM-aware receiver: downshifts the selected channel to baseband, then
    runs the single-carrier matched-filter pipeline.

    Pair with :class:`WDMTransmitter`. Works for both ``polarization="single"``
    (1-D input) and ``"dual"`` (``(2, N)`` input). The output carries the
    bits / symbols / decisions of the *single* demuxed channel — to score all
    channels, instantiate one WDMReceiver per channel.
    """

    wdm: WDMConfig
    wdm_channel: int = 0

    @model_validator(mode="after")
    def _validate_wdm_channel(self) -> WDMReceiver:
        if not 0 <= self.wdm_channel < self.wdm.n_channels:
            raise ValueError(
                f"wdm_channel must be in [0, {self.wdm.n_channels}), "
                f"got {self.wdm_channel}"
            )
        return self

    def receive(
        self, field: ComplexArray, dt_s: float | None = None
    ) -> WDMReceivedSymbols:
        if dt_s is None:
            raise ValueError(
                "dt_s is required for WDMReceiver (needed to downshift the "
                "selected channel to baseband)"
            )
        f_ch = float(self.wdm.channel_frequencies_hz()[self.wdm_channel])
        t = np.arange(field.shape[-1]) * dt_s
        downshifted = field * np.exp(-2j * np.pi * f_ch * t)
        base: ReceivedSymbols = super().receive(downshifted, dt_s)
        return WDMReceivedSymbols(
            symbols=base.symbols,
            bits=base.bits,
            decisions=base.decisions,
            modulation_order=base.modulation_order,
            polarization=base.polarization,
            pilots=base.pilots,
            pilot_symbols=base.pilot_symbols,
            wdm_channel=self.wdm_channel,
        )
