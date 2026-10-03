# End-to-end learning through a dual-polarisation 5-channel WDM link

2026-10-03 · hpcom `dev` · `hpcom.autodiff` (reversible adjoint)

**Question.** In a realistic dual-polarisation WDM system with lumped EDFA
spans, what does learning by gradient descent *through the split-step
simulation* buy: a learned constellation (the published state of the art,
Jovanovic et al., JLT 2023: +0.21 bit/4D for DP 5 × 32 GBd, 64 points,
10 × 100 km), a transmitter *with memory*, a learned wideband DBP, or all
of them trained jointly?

## Setup

| Item | Value |
|---|---|
| Transmitter | 5 × 32 GBd on a 50 GHz grid, dual polarisation, RRC β = 0.1, 8 samples/symbol of the composite field, 2048 symbols/block, 64 points (one constellation for all channels and polarisations) |
| Link | 10 × 80 km SSMF (Manakov), EDFA NF 5 dB after every span; split-step 50 steps/span (logarithmic, k = 1/3) for training, 200 for evaluation |
| Receiver | centre channel: CD compensation or DBP, matched filter, data-aided complex gain per polarisation, Gaussian demapper |
| Rate | Gaussian-demapper MI per polarisation, averaged over the two (bit/symbol/pol; ×2 for bit/4D), on 16 paired blocks |
| Training | Adam, 2000 steps, fresh data and noise per step, cosine decay, Polyak average (0.99); gradients through the whole field's split-step simulation by the **reversible adjoint** (memory independent of the 500 steps) |

## What is learned where

| Method | Transmitter | Receiver front end |
|---|---|---|
| `qam+cdc` | 64-QAM | CD compensation |
| `gs+cdc` | **learned constellation** | CD compensation |
| `gs+txcnn+cdc` | **learned constellation + residual dilated-CNN pre-distorter** on each channel's symbols (both polarisations, receptive field 169 symbols; starts at identity) | CD compensation |
| `gs+wdbp2` | learned constellation | wideband DBP of all 5 channels, 2 steps/span (fixed) |
| `gs+wldbp2` | learned constellation | **learned** wideband DBP, 2 steps/span (`ad.learned_dbp`) |
| `gs+txcnn+wldbp2` | learned constellation + pre-distorter | learned wideband DBP — all trained jointly |
| `ffdbp` | 64-QAM | full-field DBP with the forward steps (ideal reference) |

The pre-distorter's output is renormalised to unit mean power per channel;
the receiver decodes the constellation points.

## Results

Pending (job `dp-wdm-e2e` on bugatti).

## Reproduce

```bash
# GPU environment: UV_PROJECT_ENVIRONMENT=.venv-jax-gpu uv sync --extra dev --extra jax-cuda
python run.py --cosine --ema 0.99 --out results   # job.sh
python plot.py results
```
