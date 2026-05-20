"""End-to-end simulation runner: Transmitter → Link → Receiver → Result."""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, model_validator

from hpcom._typing import ComplexArray
from hpcom.link import Link
from hpcom.metrics import Metrics, bit_error_rate, evm_rms
from hpcom.modulation import bits_per_symbol, bits_to_symbols
from hpcom.receiver import Receiver
from hpcom.result import Result, SimulationMetadata
from hpcom.transmitter import SingleCarrierTransmitter, TransmittedSignal, Transmitter
from hpcom.wdm import WDMReceiver, WDMTransmitter


class Simulation(BaseModel):
    """One Tx → Link → Rx pipeline, runnable to a :class:`hpcom.result.Result`.

    Supports the full Tx/Rx matrix today:

    * single-pol single-carrier
    * single-pol WDM (Tx = :class:`WDMTransmitter`, Rx = :class:`WDMReceiver`
      with a specific ``wdm_channel`` — Result reports BER for that channel)
    * dual-pol single-carrier (per-pol metrics aggregated into a single BER)
    * dual-pol × WDM (same: pick one channel via WDMReceiver, aggregate
      across polarisations)

    Reproducibility: a single ``seed`` (set on the ``Simulation``) is the
    canonical source of randomness, split via :class:`numpy.random.SeedSequence`
    so the transmitter and each amplifier draw from non-overlapping RNGs. An
    explicit ``Transmitter.seed`` takes precedence for the transmitter only.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    transmitter: Transmitter
    link: Link
    receiver: Receiver
    seed: int | None = None
    launch_power_dbm: float | None = None
    """Average launch power per polarization, in dBm. ``None`` (default) leaves
    the transmitter's native power untouched."""

    @model_validator(mode="after")
    def _validate(self) -> Simulation:
        if self.transmitter.modulation_order != self.receiver.modulation_order:
            raise ValueError(
                "transmitter and receiver modulation_order must match: "
                f"{self.transmitter.modulation_order} vs {self.receiver.modulation_order}"
            )
        tx = self._sc_transmitter()
        if tx.samples_per_symbol != self.receiver.samples_per_symbol:
            raise ValueError(
                "transmitter and receiver samples_per_symbol must match"
            )
        if self.transmitter.polarization != self.receiver.polarization:
            raise ValueError(
                "transmitter and receiver polarization must match: "
                f"{self.transmitter.polarization!r} vs {self.receiver.polarization!r}"
            )

        tx_is_wdm = isinstance(self.transmitter, WDMTransmitter)
        rx_is_wdm = isinstance(self.receiver, WDMReceiver)
        if tx_is_wdm != rx_is_wdm:
            raise ValueError(
                "WDM Tx requires WDM Rx and vice versa; got "
                f"transmitter={type(self.transmitter).__name__}, "
                f"receiver={type(self.receiver).__name__}"
            )
        if tx_is_wdm:
            assert isinstance(self.transmitter, WDMTransmitter)
            assert isinstance(self.receiver, WDMReceiver)
            if self.transmitter.wdm != self.receiver.wdm:
                raise ValueError(
                    "WDM Tx and Rx must share the same WDMConfig; got "
                    f"{self.transmitter.wdm} vs {self.receiver.wdm}"
                )

        # Need at least one symbol in the steady-state region after edge skip.
        skip = tx.pulse_span_symbols
        if self.transmitter.n_symbols <= 2 * skip:
            raise ValueError(
                f"n_symbols ({self.transmitter.n_symbols}) must exceed "
                f"2 · pulse_span_symbols ({2 * skip}) so metrics have a "
                "transient-free region to score."
            )
        return self

    def run(self) -> Result:
        t0 = time.perf_counter()

        # Seed split: Tx and link draw from independent sub-streams.
        seed_seq = np.random.SeedSequence(self.seed) if self.seed is not None else None
        tx_seed_seq, link_seed_seq = (
            seed_seq.spawn(2) if seed_seq is not None else (None, None)
        )
        tx_rng = (
            None
            if self.transmitter.seed is not None
            else np.random.default_rng(tx_seed_seq) if tx_seed_seq is not None else None
        )
        link_rng = np.random.default_rng(link_seed_seq) if link_seed_seq is not None else None

        sent = self.transmitter.generate(rng=tx_rng)
        field = sent.field
        if self.launch_power_dbm is not None:
            target_power_w = 10.0 ** (self.launch_power_dbm / 10.0) * 1e-3
            current_power_w = float(np.mean(np.abs(field) ** 2))
            if current_power_w > 0:
                field = field * float(np.sqrt(target_power_w / current_power_w))

        propagated = self.link.propagate(field, sent.dt_s, rng=link_rng)
        received = self.receiver.receive(propagated, dt_s=sent.dt_s)

        # Pick the reference bits / symbols that should match what the Rx
        # produced. For WDM, slice the demuxed channel out of the Tx's
        # per-channel arrays. For dual-pol, the leading axis is the
        # polarisation and is preserved through scoring.
        ref_bits, _ref_symbols = self._reference_streams(sent)

        tx = self._sc_transmitter()
        skip = tx.pulse_span_symbols
        end = -skip if skip else None
        bps = bits_per_symbol(self.transmitter.effective_modulation_order)

        # Trim pulse-edge transients along the symbol-time axis (the last axis
        # of bits, viewed as (..., n_symbols, bps)). Works for 1-D and 2-D.
        sent_bits = _trim_symbol_axis(ref_bits, bps, skip, end)
        rx_bits = _trim_symbol_axis(received.bits, bps, skip, end)
        ber, n_errors, n_bits = bit_error_rate(sent_bits.ravel(), rx_bits.ravel())

        rx_syms = received.symbols[..., skip:end]
        # EVM measured against each received symbol's hard decision.
        decision_bits = (received.bits.reshape(*received.bits.shape[:-1], -1, bps)
                         [..., skip:end, :]
                         .reshape(*received.bits.shape[:-1], -1))
        ref_for_evm = bits_to_symbols(decision_bits.ravel(), self.receiver.effective_modulation_order)
        evm = evm_rms(rx_syms.ravel(), ref_for_evm)

        metrics = Metrics(ber=ber, evm_rms=evm, n_bits=n_bits, n_bit_errors=n_errors)
        runtime = time.perf_counter() - t0
        metadata = SimulationMetadata(
            seed=self.seed,
            runtime_s=runtime,
            config=self._metadata_config(),
        )

        return Result(
            received_field=propagated,
            decoded_symbols=rx_syms,
            decoded_bits=rx_bits.reshape(*received.bits.shape[:-1], -1),
            metrics=metrics,
            metadata=metadata,
        )

    def _reference_streams(
        self, sent: TransmittedSignal
    ) -> tuple[npt.NDArray[np.uint8], ComplexArray]:
        """Select the bits / symbols on the Tx side that correspond to what
        the Rx returns. For WDMReceiver this picks the demuxed channel out of
        the per-channel arrays; otherwise it's the full Tx output."""
        bits = sent.bits
        symbols = sent.symbols
        if isinstance(self.receiver, WDMReceiver):
            ch = self.receiver.wdm_channel
            # sent.bits shape:
            #   single-pol WDM: (n_channels, n_bits) → pick channel axis 0
            #   dual-pol  WDM: (2, n_channels, n_bits) → pick channel axis 1
            if sent.polarization == "dual":
                bits = bits[:, ch]
                symbols = symbols[:, ch]
            else:
                bits = bits[ch]
                symbols = symbols[ch]
        return bits, symbols

    def _sc_transmitter(self) -> SingleCarrierTransmitter:
        """Return the transmitter narrowed to SingleCarrierTransmitter.

        Simulation currently requires SC-specific attributes
        (``samples_per_symbol``, ``pulse_span_symbols``).  WDMTransmitter
        also satisfies this because it inherits from SingleCarrierTransmitter.
        """
        tx = self.transmitter
        if not isinstance(tx, SingleCarrierTransmitter):
            raise TypeError(
                f"Simulation currently requires a SingleCarrierTransmitter "
                f"(or WDMTransmitter), got {type(tx).__name__}"
            )
        return tx

    def _metadata_config(self) -> dict[str, object]:
        tx = self._sc_transmitter()
        cfg: dict[str, object] = {
            "n_symbols": self.transmitter.n_symbols,
            "modulation_order": self.transmitter.modulation_order,
            "symbol_rate_hz": self.transmitter.symbol_rate_hz,
            "samples_per_symbol": tx.samples_per_symbol,
            "n_spans": self.link.n_spans,
            "polarization": self.transmitter.polarization,
        }
        if isinstance(self.transmitter, WDMTransmitter):
            cfg["n_channels"] = self.transmitter.wdm.n_channels
            cfg["channel_spacing_hz"] = self.transmitter.wdm.spacing_hz
        if isinstance(self.receiver, WDMReceiver):
            cfg["scored_wdm_channel"] = self.receiver.wdm_channel
        return cfg


def _trim_symbol_axis(
    bits: npt.NDArray[np.uint8], bps: int, skip: int, end: int | None
) -> npt.NDArray[np.uint8]:
    """View ``bits`` (any leading dims) as ``(..., n_symbols, bps)``, trim the
    symbol-time axis to ``[skip:end]``, and reshape back to ``(..., n_trimmed*bps)``."""
    reshaped = bits.reshape(*bits.shape[:-1], -1, bps)
    trimmed = reshaped[..., skip:end, :]
    return trimmed.reshape(*bits.shape[:-1], -1)
