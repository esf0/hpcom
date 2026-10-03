# Poggiolini et al. 2015: GN and EGN models of nonlinear interference

P. Poggiolini, Y. Jiang, A. Carena, F. Forghieri, "A simple and accurate
closed-form EGN model formula," arXiv:1503.04132, 2015 — Figs. 1(a) and 2(a).

**Claim.** The GN model overestimates the nonlinear interference (NLI) of
PM-QPSK. A closed-form correction (the "approximate EGN") follows the exact
EGN model and the simulation once the link has a few spans.

**System.** PM-QPSK, 32 GBd, RRC roll-off 0.05, 33.6 GHz grid, 3 or 15
channels; SMF spans of 100 km (D = 16.7 ps/(nm·km), γ = 1.3 /W/km,
0.22 dB/km); ideal noiseless amplification. The quantity is the NLI
coefficient η of the centre channel, in dB(1/W²).

**hpcom.**
- `hpcom.nli.nli_coefficient`: the GN model (`model="gn"`) and the
  closed-form EGN correction (`model="egn"`).
- `hpcom.nli.first_order_nli`: the first-order perturbation of the actual
  signal. It agrees with the split-step simulation within 0.05 dB
  (`research/2026-10-01-gn-egn`), so it stands in for the paper's
  simulation curve (3 channels; 4 blocks of 1000 symbols per point).

## Results

The paper's values are read off its figures (about ±0.2 dB).

| system | quantity | spans: 1 | 2 | 5 | 10 | 20 | 50 |
|---|---|---|---|---|---|---|---|
| 3 channels | GN, paper | 26.7 | 30.0 | 34.3 | 37.6 | 40.9 | 45.3 |
| | GN, hpcom | 26.67 | 29.97 | 34.34 | 37.63 | 40.91 | 45.23 |
| | closed-form EGN, paper | 24.5 | 27.9 | 32.5 | 36.0 | 39.4 | 43.9 |
| | closed-form EGN, hpcom | 24.50 | 27.97 | 32.55 | 35.98 | 39.38 | 43.84 |
| | simulation, paper | 21.2 | 26.8 | 32.2 | 35.9 | 39.4 | 44.2 |
| | first order, hpcom | 20.90 | 26.55 | 32.12 | 35.85 | 39.39 | 43.81 |
| 15 channels | GN, paper | 29.5 | 32.7 | 36.9 | 40.1 | 43.3 | 47.5 |
| | GN, hpcom | 29.50 | 32.67 | 36.86 | 40.03 | 43.20 | 47.38 |
| | closed-form EGN, paper | 27.5 | 30.8 | 35.1 | 38.4 | 41.6 | 45.9 |
| | closed-form EGN, hpcom | 27.53 | 30.78 | 35.08 | 38.34 | 41.58 | 45.85 |

**Agreement.**
- The GN model and the closed-form EGN agree with the paper within
  0.12 dB at every point. This is within the reading error of the figures.
- The first-order perturbation agrees with the paper's simulation within
  0.01–0.39 dB. The largest differences are at 1 span and at 50 spans. At
  1 span, η varies by up to 0.3 dB from one block of symbols to the next
  (`research/2026-10-01-gn-egn`). At 50 spans, higher-order nonlinearity
  at the paper's launch power (−3 dBm) can add to the simulated NLI. The
  first-order model does not include it.

Full table: `results/summary.md`.

Run: `python run.py --out results` (about 1 minute on 4 CPU cores).
