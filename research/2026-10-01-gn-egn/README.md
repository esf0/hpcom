# GN, EGN and first-order NLI models against the split-step simulation

2026-10-01 · hpcom `dev` · `hpcom.nli`

**Question.** How accurate are the NLI models of `hpcom.nli` — the GN
reference formula, the closed-form EGN correction and the first-order
perturbation of the actual signal — against the noiseless split-step
simulation, for QAM and Gaussian symbols, 1 to 30 spans?

**Answer.** The **first-order perturbation equals the simulation within
0.05 dB** at every one of the 28 points. The **GN model overestimates the
NLI of QAM**: by 6.9 / 3.6 / 3.1 dB at one span (QPSK / 16-QAM / 64-QAM),
still 1.8 / 1.2 / 0.9 dB at 30 spans. The **closed-form EGN correction**
fixes most of that from five spans on (within ±0.23 dB; QPSK +0.50 dB at
three spans), but is 1.1–3.0 dB high at one span, as its authors state.
For Gaussian symbols GN agrees with the simulation: within ±0.10 dB up to
10 spans, and within the statistical error at 20 and 30 spans.

**Correction (2026-10-04).** The first run sampled at 5 samples/symbol.
Kerr products of the 135 GHz band reach ±203 GHz and aliased into the
centre channel: η was 0.1–0.6 dB too high in both the simulation and the
first-order model (they share the grid, so they still agreed). This page
now shows the run at 10 samples/symbol (8 gives the same η). The aliased
run is in `results-5sps-aliased/`.

## Setup

| Item | Value |
|---|---|
| Channels | 3 × 32 GBd, dual polarisation, RRC β = 0.1, 50 GHz grid; the centre channel is scored |
| Link | 1–30 spans of 80 km SSMF (0.2 dB/km, D = 17 ps/(nm·km), γ = 1.27 /W/km), ideal noiseless EDFAs |
| Simulation | split-step, 100 steps/span (logarithmic), 10 samples/symbol, at the power where GN predicts SNR_NLI = 30 dB and 3 dB above; η = 1/(SNR·P²) |
| Models | `nli_coefficient` (GN coherent / incoherent, `model="egn"`), `first_order_nli` (96 nodes per span) |
| Statistics | 4 blocks of 4096 symbols; the first-order estimate and the simulation use the *same* blocks (η itself varies by up to 0.3 dB from block to block) |

The simulated η at P and P + 3 dB differ by 0.01–0.05 dB: the simulation is
in the first-order regime, and far above its own numerical noise.

## Results

![η vs spans](results/eta_vs_spans.png)

η of the centre channel in dB(1/W²) (full table: `results/summary.md`):

| format | spans | GN | GN incoh. | EGN closed form | first order | SSFM |
|---|---|---|---|---|---|---|
| PM-QPSK | 1 | 25.49 | 25.49 | 21.58 | 18.60 | 18.61 |
| PM-QPSK | 5 | 33.60 | 32.48 | 30.95 | 30.72 | 30.73 |
| PM-QPSK | 30 | 42.45 | 40.26 | 40.53 | 40.61 | 40.62 |
| PM-16QAM | 1 | 25.49 | 25.49 | 23.24 | 21.88 | 21.90 |
| PM-16QAM | 5 | 33.60 | 32.48 | 31.99 | 32.07 | 32.10 |
| PM-16QAM | 30 | 42.45 | 40.26 | 41.24 | 41.27 | 41.28 |
| PM-64QAM | 1 | 25.49 | 25.49 | 23.50 | 22.39 | 22.41 |
| PM-64QAM | 5 | 33.60 | 32.48 | 32.16 | 32.21 | 32.23 |
| PM-64QAM | 30 | 42.45 | 40.26 | 41.36 | 41.58 | 41.59 |
| PM-Gaussian | 1 | 25.49 | 25.49 | — | 25.55 | 25.59 |
| PM-Gaussian | 5 | 33.60 | 32.48 | — | 33.53 | 33.55 |
| PM-Gaussian | 30 | 42.45 | 40.26 | — | 42.06 | 42.07 |

## Findings

1. **First order is the whole story here.** At the powers that matter
   (SNR_NLI ≈ 30 dB, near the optimum) the first-order field reproduces the
   simulated NLI to 0.05 dB, for every format and length.
2. **GN is pessimistic for QAM**, most at few spans and for QPSK — the
   modulation's fourth moment (Φ = 1, 17/25, 13/21) sets the size. Only for
   Gaussian symbols does GN describe the signal.
3. **Coherence matters.** The incoherent GN model under-counts long links
   (−1.8 dB vs the simulation at 30 spans, Gaussian symbols). For QAM it
   looks close around 5–10 spans only because that under-count cancels GN's
   overestimate.
4. **The closed-form EGN** is a good engineering formula from five spans on
   (|error| ≤ 0.23 dB), not at one or two spans (1.1–3.0 dB and
   0.3–0.9 dB high).
5. **GN for Gaussian symbols** agrees with the simulation within ±0.10 dB
   up to 10 spans. (The first run's "open residual" of −0.08 … −0.26 dB was
   the aliasing.) At 20 and 30 spans GN reads 0.22 and 0.38 dB above the
   mean of the 4 blocks, but for Gaussian symbols η scatters by about
   0.4 dB from block to block there. With 8 blocks the first-order η at
   30 spans is 42.43 ± 0.15 dB, GN 42.45 dB (`block_length.py`). The GN
   integral has converged (4× the channel nodes: +0.004 dB; 16× the
   points: 0.000 dB). Longer blocks (16384 symbols, spectral lines 2 MHz
   apart instead of 7.8 MHz) change nothing beyond the error
   (42.21 ± 0.26 dB).

Cross-check with the literature (`tests/numerical/test_nli.py`): the GN
integral reproduces the published GN curve of Poggiolini et al. 2015
(arXiv:1503.04132, Fig. 1a) at 1 and 50 spans, and the closed-form EGN their
50-span value.

## Reproduce

```bash
python run.py --out results     # job.sh on bugatti: CPU, 4 cores, 21 min
python plot.py results
```
