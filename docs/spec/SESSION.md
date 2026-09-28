# Session resume — hpcom v1

**Last updated: 2026-09-28.** Branch `dev`. 1059 tests pass (≈45 s); ruff,
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
| Git | commit on `dev` |

Architecture: `HpcomModel` base (strict, immutable, JSON round-trip);
`Transmitter`/`Receiver` ABCs with `PulseShaped*` (SC, WDM siblings) and OFDM
subclasses; propagator registry (numpy default, tensorflow, numpy-linear) and
DSP-kernel registry (python) with conformance suites in `tests/backends/`.

## Open items / candidates

- OFDM × WDM (multi-carrier OFDM) — not implemented.
- Laser phase noise, PMD, frequency offset — not modelled (would need
  carrier-phase recovery beyond the per-stream complex gain).
- Rust kernel set (`hpcom.backends.rust`) — seam and conformance tests exist.
- JAX propagator — seam and conformance tests exist.
- OptiCommPy gap survey (old Phase B) — not started.
- `hpcom._legacy` removal in v2.
