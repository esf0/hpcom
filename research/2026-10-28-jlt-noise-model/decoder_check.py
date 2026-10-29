"""Sanity check of the triplet forward–backward of ``analyse.py``.

Synthetic channels whose memory is confined to the triplet (two-sided linear
ISI; a cubic term x_{t-1} x_{t+1} x_t*), 16-QAM at SNR 15 dB: the decoder must
bring the BER back to that of the ISI-free AWGN channel, by marginals (forward–backward) and by
the most probable sequence (Viterbi).

Run::

    python decoder_check.py
"""

from __future__ import annotations

import numpy as np
from analyse import awgn, ber, fit_gmm, smd, triplet_decode, triplet_ids

from hpcom import SingleCarrierTransmitter


def main() -> None:
    points = SingleCarrierTransmitter(
        modulation_order=16, symbol_rate_hz=1e9, n_symbols=16, samples_per_symbol=2
    ).resolved_constellation.points
    rng = np.random.default_rng(0)
    idx = rng.integers(0, 16, (24, 16384))
    x = points[idx]
    left, right = np.roll(x, 1, axis=-1), np.roll(x, -1, axis=-1)
    var = 10 ** (-15.0 / 10)
    tr, te = slice(0, 20), slice(20, None)
    inner = (slice(None), slice(1, -1))
    for a, b, c in ((0.0, 0.0, 0.0), (0.25, 0.2, 0.0), (0.0, 0.0, 0.08)):
        clean = x + a * left + b * right + c * left * right * np.conj(x)
        z = clean + awgn(rng, x.shape, var)
        gmm = fit_gmm(
            clean[tr].ravel() + awgn(rng, (clean[tr].size,), 1e-6), triplet_ids(idx[tr]).ravel(), 1
        )
        dec = triplet_decode(z[te], gmm, var)
        seq = triplet_decode(z[te], gmm, var, sequence=True)
        ref = ber(smd(x[te] + awgn(rng, x[te].shape, var), points)[inner], idx[te][inner])
        print(
            f"z = x + {a}·x(t-1) + {b}·x(t+1) + {c}·x(t-1)x(t+1)x*(t), SNR 15 dB: "
            f"SMD {ber(smd(z[te], points)[inner], idx[te][inner]):.3e}  triplet FB {ber(dec, idx[te][inner]):.3e}  "
            f"Viterbi {ber(seq, idx[te][inner]):.3e}  ISI-free AWGN {ref:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
