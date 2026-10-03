# Gallery: published results, reproduced with hpcom

Each entry reproduces a figure of a published paper. It shows the paper's
numbers, read off its figures, beside the numbers hpcom computes for the
same system. An entry is a script (`run.py`) and a page (`README.md`).

| Entry | Paper | What is reproduced | Status |
|---|---|---|---|
| [poggiolini2015](poggiolini2015/) | Poggiolini et al., arXiv:1503.04132 (2015) | NLI coefficient vs spans: GN model, closed-form EGN, simulation | done: GN and EGN within 0.12 dB, simulation within 0.4 dB |
| [jovanovic2023](jovanovic2023/) | Jovanovic et al., JLT 41(12) (2023) | constellation learned through the SSFM, dual-pol 5-channel WDM: +0.21 bit/4D | results pending |
| [hager2021](hager2021/) | Häger & Pfister, IEEE JSAC 39(1) (2021) | linear equalisation, DBP and learned DBP at one step per span | done: linear and DBP within 0.15 dB; the 77-tap LDBP is not reproduced (hpcom's learned DBP has no tap limit) |
| [kelly1992](kelly1992/) | Kelly, Electron. Lett. 28(8) (1992) | Kelly sidebands of a soliton fiber laser | done |
| uhlemann2020 | Uhlemann et al., ITG Photonic Networks (2020) | autoencoder through the SSFM with ideal distributed Raman amplification | planned: needs distributed amplification |

Run one entry: `python gallery/<entry>/run.py --out gallery/<entry>/results`
(`--quick` for a check in seconds). The GPU entries run on a node with
`gallery/job.sh`. The values read off the figures carry about ±0.2 dB
(±0.01 bit/symbol) of reading error.
