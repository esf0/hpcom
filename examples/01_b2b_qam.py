"""01 — Back-to-back 16-QAM, no fiber.

The smallest possible hpcom scenario: a transmitter wired straight into a
receiver, with no link in between. Useful as a sanity check that your
modulation, pulse shaping, and matched filter all line up.

Run::

    python examples/01_b2b_qam.py
"""

from __future__ import annotations

import numpy as np

from hpcom.modulation import bits_per_symbol
from hpcom.receiver import Receiver
from hpcom.transmitter import Transmitter


def main() -> None:
    tx = Transmitter(
        modulation_order=16,
        symbol_rate_hz=32e9,
        n_symbols=8192,
        samples_per_symbol=8,
        roll_off=0.1,
        pulse_span_symbols=32,
        seed=0,
    )
    rx = Receiver(
        modulation_order=16,
        samples_per_symbol=tx.samples_per_symbol,
        roll_off=tx.roll_off,
        pulse_span_symbols=tx.pulse_span_symbols,
    )

    sent = tx.generate()
    received = rx.receive(sent.field)

    skip = tx.pulse_span_symbols
    bps = bits_per_symbol(tx.modulation_order)
    sent_bits = sent.bits.reshape(-1, bps)[skip:-skip].ravel()
    rx_bits = received.bits.reshape(-1, bps)[skip:-skip].ravel()
    n_errors = int(np.sum(sent_bits != rx_bits))
    ber = n_errors / sent_bits.size

    print(f"16-QAM back-to-back, {tx.n_symbols} symbols, {sent.field.size} samples")
    print(f"  bits scored:    {sent_bits.size}")
    print(f"  bit errors:     {n_errors}")
    print(f"  BER:            {ber:.2e}")


if __name__ == "__main__":
    main()
