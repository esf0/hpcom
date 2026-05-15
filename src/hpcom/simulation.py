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

    def __post_init__(self) -> None:
        if self.transmitter.modulation_order != self.receiver.modulation_order:
            raise ValueError(
                "transmitter and receiver modulation_order must match: "
                f"{self.transmitter.modulation_order} vs {self.receiver.modulation_order}"
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
        propagated = self.link.propagate(sent.field, sent.dt_s, rng=link_rng)
        received = self.receiver.receive(propagated)

        # Drop pulse-shape edge transients before scoring metrics.
        skip = self.transmitter.pulse_span_symbols
        bps = bits_per_symbol(self.transmitter.modulation_order)
        sent_bits = sent.bits.reshape(-1, bps)[skip:-skip].ravel()
        rx_bits = received.bits.reshape(-1, bps)[skip:-skip].ravel()
        ber, n_errors, n_bits = bit_error_rate(sent_bits, rx_bits)

        rx_syms = received.symbols[skip:-skip]
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
