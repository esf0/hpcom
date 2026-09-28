# Session resume — hpcom v1

**Last updated: 2026-09-28.** Branch `dev`. 1281 tests pass (≈90 s); ruff,
ruff-format, mypy (strict) and the Sphinx build (`-W`) are clean. A core-only
install (numpy, scipy, pydantic) passes too, with TF/legacy tests skipped.

The files `hpcom-v1-plan.md`, `hpcom-v1-tasks.md` and `test-coverage-plan.md`
describe the original development plan and are kept for history; the code,
`docs/physics.rst` and the tests are the source of truth.

## State after the v1 hardening pass (commit `5b6b68b` and follow-ups)

A full audit replaced the earlier dev work where it was wrong. Decisions taken
with the maintainer:

| Topic | Decision |
|---|---|
| Launch power | per WDM channel, summed over polarisations, on the transmitter (`launch_power_per_channel_dbm` / `launch_power_total_dbm`) |
| ASE | per-pol PSD `n_sp·hν·(G−1)`, `n_sp = (G·F−1)/(2(G−1))` (Essiambre 2010) |
| Pulse shaping | exact RRC (default) or truncated FIR; circular (default) or linear |
| QAM bit labelling | default `q_descending` = pre-v1 mapping (same bits → same field); `q_ascending` (former dev mapping, complex conjugate) as an option; proven equivalent (`test_labeling.py`) |
| Noiseless amplifier | `EDFA(noise_figure_db=None)` (replaces pre-v1 `noise_figure_db=-200`) |
| DBP | core `hpcom.dsp.DigitalBackPropagation` (a `Compensator`), called by pulse-shaped receivers on realistic front-end samples (baseband channel, band-limited, resampled to 1/2/4 sps); `Simulation.evaluate` for paired comparisons |
| Git | commit on `dev` |

Architecture: `HpcomModel` base (strict, immutable, JSON round-trip);
`Transmitter`/`Receiver` ABCs with `PulseShaped*` (SC, WDM siblings) and OFDM
subclasses; propagator registry (numpy default, tensorflow, numpy-linear) and
DSP-kernel registry (python) with conformance suites in `tests/backends/`.

## Open items / candidates

- DBP follow-ups: enhanced/filtered SSFM variants and non-uniform step
  placement (as further `Compensator`s); DBP for OFDM receivers; batching the
  per-channel stage across channels.
- Cross-checks still to add: OptiCommPy (SSFM/Manakov, EDFA), N = 2 soliton
  recurrence, Manakov vector soliton, FWM efficiency vs closed form,
  modulation-instability gain.
- Laser phase noise, PMD, frequency offset — not modelled (would need
  carrier-phase recovery beyond the per-stream complex gain).
- Rust kernel set (`hpcom.backends.rust`) — seam and conformance tests exist.
- JAX propagator — seam and conformance tests exist.
- OptiCommPy gap survey (old Phase B) — not started.
- `hpcom._legacy` removal in v2.

## Ideas (not planned)

- **OFDM over WDM** — several OFDM bands, each on its own WDM carrier (the
  "multi-band OFDM superchannel" of the 2010s literature). hpcom could build
  it from `OFDMTransmitter` streams placed on a `WDMConfig` grid. Kept as an
  idea: no current study needs it (maintainer, 2026-09-28).
