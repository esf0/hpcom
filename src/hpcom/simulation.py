"""End-to-end simulation runner: Transmitter → Link → Receiver → Result."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from hpcom.link import Link
from hpcom.metrics import Metrics, bit_error_rate, evm_rms
from hpcom.modulation import bits_per_symbol, bits_to_symbols
from hpcom.receiver import Receiver
from hpcom.result import Result, SimulationMetadata
from hpcom.transmitter import Transmitter


@dataclass(frozen=True)
class Simulation:
    """One Tx → Link → Rx pipeline, runnable to a :class:`hpcom.result.Result`.

    Reproducibility: a single ``seed`` (set on the ``Simulation``) is the
    canonical source of randomness. Internally it is split into independent
    sub-streams via :class:`numpy.random.SeedSequence` so the transmitter and
    each amplifier draw from non-overlapping RNGs. An explicit
    ``Transmitter.seed`` takes precedence for the transmitter only — useful
    when scanning link parameters with a fixed bit pattern.
    """

    transmitter: Transmitter
    link: Link
    receiver: Receiver
    seed: int | None = None
    launch_power_dbm: float | None = None
    """Average launch power per polarization, in dBm. ``None`` (default) leaves
    the transmitter's native power untouched. Set to a realistic value (e.g.
    ``-3.0`` for SSMF + EDFA) to scale the field to that average power before
    it enters the link — without this, defaults often land in the highly
    nonlinear regime where the signal is unrecoverable."""

    def __post_init__(self) -> None:
        if self.transmitter.modulation_order != self.receiver.modulation_order:
            raise ValueError(
                "transmitter and receiver modulation_order must match: "
                f"{self.transmitter.modulation_order} vs {self.receiver.modulation_order}"
            )
        from hpcom.transmitter import WDMTransmitter

        if isinstance(self.transmitter, WDMTransmitter):
            raise NotImplementedError(
                "Simulation does not yet integrate WDMTransmitter. Construct "
                "the Tx → Link → Rx pipeline manually and demux per channel via "
                "Receiver(wdm=..., wdm_channel=i)."
            )
        if self.transmitter.samples_per_symbol != self.receiver.samples_per_symbol:
            raise ValueError(
                "transmitter and receiver samples_per_symbol must match"
            )
        # Need at least one symbol in the steady-state region after edge skip.
        skip = self.transmitter.pulse_span_symbols
        if self.transmitter.n_symbols <= 2 * skip:
            raise ValueError(
                f"n_symbols ({self.transmitter.n_symbols}) must exceed "
                f"2 · pulse_span_symbols ({2 * skip}) so metrics have a "
                "transient-free region to score."
            )

    def run(self) -> Result:
        t0 = time.perf_counter()

        # Split seed into one stream for the Tx (if it doesn't have its own
        # seed) and one for the link's amplifiers.
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

        # Drop pulse-shape edge transients before scoring metrics. When
        # pulse_span_symbols=0, Python's [0:-0] would silently return empty —
        # use None as the upper bound instead so the full range is kept.
        skip = self.transmitter.pulse_span_symbols
        end = -skip if skip else None
        bps = bits_per_symbol(self.transmitter.modulation_order)
        sent_bits = sent.bits.reshape(-1, bps)[skip:end].ravel()
        rx_bits = received.bits.reshape(-1, bps)[skip:end].ravel()
        ber, n_errors, n_bits = bit_error_rate(sent_bits, rx_bits)

        rx_syms = received.symbols[skip:end]
        # EVM measured against each received symbol's hard decision.
        ref = bits_to_symbols(rx_bits, self.receiver.modulation_order)
        evm = evm_rms(rx_syms, ref)

        metrics = Metrics(ber=ber, evm_rms=evm, n_bits=n_bits, n_bit_errors=n_errors)
        runtime = time.perf_counter() - t0
        metadata = SimulationMetadata(
            seed=self.seed,
            runtime_s=runtime,
            config={
                "n_symbols": self.transmitter.n_symbols,
                "modulation_order": self.transmitter.modulation_order,
                "symbol_rate_hz": self.transmitter.symbol_rate_hz,
                "samples_per_symbol": self.transmitter.samples_per_symbol,
                "n_spans": self.link.n_spans,
            },
        )

        return Result(
            received_field=propagated,
            decoded_symbols=rx_syms,
            decoded_bits=rx_bits,
            metrics=metrics,
            metadata=metadata,
        )
