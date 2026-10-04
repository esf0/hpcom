# Jovanovic et al. 2023: a constellation learned through the split-step simulation

O. Jovanovic, F. Da Ros, D. Zibar, M. P. Yankov, "Geometric constellation
shaping for fiber-optic channels via end-to-end learning," *J. Lightw.
Technol.* 41(12), 2023 — Fig. 5.

**Claim.** A 64-point constellation, trained by back-propagation through
the split-step simulation of a dual-polarisation WDM link, gains about
0.21 bit/4D-symbol over 64-QAM at the optimum launch power.

**System.** 5 × 32 GBd dual-polarisation channels on a 50 GHz grid, RRC
roll-off 0.01, 64 points; 10 × 100 km SSMF (D = 16.464 ps/(nm·km),
γ = 1.3 /W/km, 0.2 dB/km), EDFA NF 5 dB. The quantity is the
mismatched-Gaussian-receiver MI of the centre channel after CD
compensation, in bit/4D-symbol (both polarisations).

**hpcom.** The `jovanovic2023` benchmark scenario (`hpcom.bench`), methods
`qam+cdc` (64-QAM) and `gs+cdc` (`learn.Constellation`), at 16 samples per
symbol as in the paper. The constellation trains for 2000 steps on the MI,
with the gradient through 50 split steps per span (reversible adjoint). The
score is the mean of 16 fixed blocks of 2000 symbols on a simulation with
200 steps per span: a constellation cannot profit from the training
simulation's step error. One constellation per power.

## Results

The paper's values are read off its Fig. 5 (about ±0.01 bit/4D).

| power | paper QAM | hpcom QAM | paper learned | hpcom learned | paper gain | hpcom gain |
|---|---|---|---|---|---|---|
| −1.5 dBm | 10.37 | 10.35 | 10.59 | 10.60 | +0.22 | +0.24 |
| −0.5 dBm | 10.63 | 10.62 | 10.83 | 10.86 | +0.20 | +0.24 |
| +0.5 dBm | 10.71 | 10.71 | 10.92 | 10.94 | +0.21 | +0.23 |
| +1.5 dBm | 10.57 | 10.62 | 10.78 | 10.85 | +0.21 | +0.23 |

The standard errors of the hpcom values are 0.010–0.017 bit/4D.

**Agreement.**
- At and below the optimum (+0.5 dBm, as in the paper), 64-QAM agrees
  within 0.02 bit/4D and the learned constellation within 0.03 bit/4D.
- Above the optimum (+1.5 dBm) hpcom is 0.05 (QAM) and 0.07 (learned)
  bit/4D higher than the paper. The cause is not determined. The paper
  steps 100 m (1000 uniform steps per span); hpcom scores on 200
  logarithmic steps per span.
- The gain of the learned constellation is +0.23 to +0.24 bit/4D, the
  paper's +0.20 to +0.22.

**Sampling.** At 8 samples/symbol the simulated band (256 GHz) is narrower
than twice the signal band (2 × 232 GHz). Kerr products then alias into the
signal band and 64-QAM loses 0.07–0.10 bit/4D at and above the optimum.
`hpcom.bench` now rejects such scenarios.

Full table: `results/summary.md`.

Run (GPU): `python run.py --out results` (about 10 minutes on an RTX 3070:
about 125 s of training per power).
