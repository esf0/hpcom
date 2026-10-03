# Häger & Pfister 2021: DBP and learned DBP at one step per span

C. Häger, H. D. Pfister, "Physics-based deep learning for fiber-optic
communication systems," *IEEE J. Sel. Areas Commun.* 39(1), 280–294, 2021
(arXiv:2010.14258) — Fig. 5.

**Claim.** In a 10.7 GBd single-channel link over 25 × 80 km, DBP with one
step per span (frequency-domain filters) gains about 8 dB of effective SNR
over linear equalisation. Learned DBP (LDBP) with short time-domain filters,
77 taps in total, reaches 23.5 dB at a small fraction of the complexity.

**System.** One polarisation, 10.7 GBd, Gaussian symbols, RRC roll-off 0.1;
25 × 80 km SSMF (β₂ = −21.683 ps²/km, γ = 1.3 /W/km, 0.2 dB/km), EDFA
NF 5 dB; receiver at 2 samples/symbol. The quantity is the effective SNR.

**hpcom.**
- Linear equalisation: `learn.CDC()`.
- DBP: `learn.DBP(steps_per_span=1)`. Each Kerr step uses the span's
  effective length (the default of `learn.DBP`).
- Learned DBP: `learn.LearnedDBP(steps_per_span=1)`. Every step's frequency
  response, Kerr coefficient and power filter train for 2000 steps on the
  effective SNR. Its filters are frequency-domain, with no tap limit. It is
  therefore *not* the paper's 77-tap LDBP.
- Gaussian symbols: 1024 points drawn from a circular Gaussian, sent with
  equal probability. Training simulation: 100 steps per span. Score: 16
  blocks of 1024 symbols on 400 steps per span.

## Results

Effective SNR in dB. The paper's values are read off its Fig. 5 (about
±0.2 dB).

| power | paper, linear | hpcom, linear | paper, DBP | hpcom, DBP | paper, LDBP 77 taps | hpcom, learned DBP |
|---|---|---|---|---|---|---|
| −10 dBm | 16.4 | 16.36 | — | 16.71 | — | — |
| −8 dBm | 17.3 | 17.41 | — | 18.63 | — | — |
| −6 dBm | 16.9 | 17.00 | 20.6 | 20.59 | — | — |
| −4 dBm | — | 14.70 | 22.6 | 22.50 | 22.1 | 22.51 |
| −2 dBm | — | 10.92 | 24.2 | 24.33 | 23.3 | 24.43 |
| 0 dBm | — | 6.69 | 25.3 | 25.45 | 22.8 | 25.85 |
| +2 dBm | — | 1.56 | 23.6 | 23.65 | 17.6 | 25.60 |

The standard errors of the hpcom values are 0.04–0.35 dB (largest at
+2 dBm).

**Agreement.**
- Linear equalisation and DBP with one step per span agree with the paper
  within 0.15 dB at every power. DBP peaks at 25.45 dB at 0 dBm (paper:
  25.3 dB at 0 dBm).
- hpcom's learned DBP is not the paper's LDBP. Without a tap limit it
  reaches 25.85 dB, 0.4 dB above DBP at 0 dBm and 2.0 dB above it at
  +2 dBm. The paper's LDBP spends at most 77 taps and peaks at 23.5 dB.
  hpcom has no tap-constrained, pruned time-domain LDBP, so the paper's
  green curve is not reproduced.

**Note.** With a Kerr phase over the whole step length at the mid-step
power (the default split-step scheme of a `Fiber`), DBP at one step per
span reaches only 19.8 dB at −4 dBm and 13.9 dB at 0 dBm. The effective
length is necessary for this comparison.

Full table: `results/results.json`, `results/summary.md`.

Run (GPU): `python run.py --out results` (about 13 minutes on an RTX 3070:
about 130 s of training per power).
