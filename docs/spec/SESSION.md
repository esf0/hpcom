# Session resume — hpcom v1

**Last updated: 2026-09-28.** Branch `dev`. 1731 tests pass (≈4 min); ruff,
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

- Non-uniform steps done (`Fiber.step_distribution`/`step_log_factor`, DBP and
  `Link.backpropagate`; docs/numerics.rst with derivations and references).
  Schemes done (`hpcom.numerics.Scheme`): Kerr position, effective Kerr
  length, filtered Kerr step, Yoshida/Suzuki 4th-order compositions, RK4IP,
  FWM-aware step count, adaptive local-error method (Sinkin 2003);
  NumPy + TF (GPU); CPU-fallback warning mechanism.
  DBP for OFDM receivers done. JAX backend (`backend="jax"`, every scheme)
  added; the full suite passes with JAX on the local RTX 3070 (separate env,
  `benchmarks/gpu_validation.sh`), 4–41× faster than NumPy (docs/backends.rst). TensorFlow backend:
  kept and supported alongside JAX for systems where it suits better
  (maintainer, 2026-09-28); new schemes go into both. NFT: not in hpcom for now — the
  maintainer's separate NFT project will be integrated later. Rust: the maintainer will write it
  themselves later.
- Multicore fibers done (`hpcom.MulticoreFiber`, weakly coupled, exact linear
  step, all schemes/backends, validated vs coupler/Jensen/supermode/hexagonal
  exact solutions). Next for SDM: single-core → multicore transceivers
  (per-core Tx/Rx, Simulation scoring per core); random coupling;
  strongly coupled (generalised Manakov). SDM transceivers done (2026-09-30):
  `n_cores` on transmitters/receivers (streams (core, pol, ch)), Simulation
  scores per core (`StreamMetrics.core`), `equalize="mimo"` with
  `mimo_across_cores=True` (joint 2M×2M), autodiff chain too.
- Cross-checks done (physics.rst "Validation"): OptiCommPy 0.10 golden data
  (regenerate with `tests/numerical/_data/make_opticommpy_golden.py` in a
  separate env — OptiCommPy needs numba), N = 2 soliton, Manakov vector
  soliton, degenerate FWM, modulation instability. Candidate: GN-model NLI
  power vs simulation.
- Generalised NLSE done (2026-09-29; physics.rst "Generalised NLSE",
  numerics.rst "Generalised NLSE terms"): `Fiber.higher_order_betas`,
  `wavelength_m`, `loss_spectrum` (`LossSpectrum`), `raman`
  (`RamanResponse`, Blow–Wood / Lin–Agrawal, analytic H_R), `self_steepening`
  (+`shock_time_s`, RK4IP only), `Fiber.recentred` (DBP per channel uses
  it). Raman keeps every split-step scheme exact/invertible; single
  polarisation only (vector Raman planned with PMD). Validated on every
  backend vs exact solutions (moment theorem, tone losses, linearised GNLSE
  2×2, shock characteristics, photon number, Gordon SSFS) and vs gnlse-python
  on Dudley's supercontinuum benchmark (golden data built on a node with
  `tests/numerical/_data/make_gnlse_golden.sh`). Next (maintainer's order):
  differentiable JAX mode → PMD → SDM transceivers → lasers (CGLE, saturable
  gain) → GN/EGN. Planned numerics: adaptive RK4IP.
- Differentiable JAX mode done (`hpcom.autodiff`, docs/autodiff.rst):
  `propagate` / `propagate_link` / `amplify` / `backpropagate` /
  `compensate` as pure JAX functions; `FiberParameters` (ps/km units,
  float32-safe), `AmplifierParameters`; gradient modes store / checkpoint
  (√J) / reversible (exact inverse-step adjoint, O(1) memory); multicore via
  eigen-decomposition + Daleckii–Krein custom JVP. Validated vs NumPy
  (values), finite differences and closed forms (gradients); example 10
  (learned per-span DBP, +4.3 dB at 1 step/span). Differentiable
  transceivers done: constellation (GS points, PS logits + Gumbel
  straight-through), modulate, demodulate (CDC, WDM selection, front end,
  DBP or any JAX compensator, matched filter, equalisers), MI/GMI/LLRs;
  equal to the NumPy chain to 1e-12; example 11 (end-to-end GS + launch
  power through 20×80 km, +0.05 b/sym). Open: JaxPropagator could delegate
  to the autodiff core; OFDM not in autodiff yet.
- PMD done (2026-09-29): `Fiber.pmd_ps_per_sqrt_km/pmd_sections/pmd_seed`,
  coarse-step Manakov-PMD (Haar rotations, exact pieces per section,
  output rotation Q_N), all fixed-step schemes, numpy/numpy-linear/jax/tf;
  `Link.from_spans` seeds spans independently; receiver `equalize="mimo"`
  (least-squares butterfly FIR at 2 sps, trained on the sent data — the
  constant pilots cannot train it); DBP drops PMD unless `include_pmd`;
  autodiff: PMD coefficient differentiable, MIMO by normal equations.
  Validated: Maxwellian DGD (KS), PSP delays ±τ/2, rotation invariance,
  orders, inverse. Open: vector Raman in Manakov (needs a full CNLSE
  reference to validate the averaging), per-pol pilot sequences for
  pilot-trained MIMO, CMA/LMS adaptive equalisers.
- Fiber lasers done (2026-10-01): `hpcom.laser` — `GainFiber` (small-signal
  gain, Lorentzian/parabolic bandwidth, gain saturating on the window
  energy by a mid-step predictor), `Fiber.nonlinear_gain_per_w_per_km`
  (cubic gain / two-photon absorption, exact Kerr sub-step
  P → P/(1 − 2εcP)), `SaturableAbsorber`, `SpectralFilter`,
  `OutputCoupler`, `RingCavity` (converges on the output power profile).
  numpy/jax/tf kernels agree to 1e-14; autodiff has every gain parameter
  (reversible mode refused with saturation). Validated: gain spectrum
  exact, saturated CW amplifier closed form (order 2 in every split-step
  scheme), TPA exact and inverted, Pereira–Stenflo CGLE soliton (Strang 2,
  RK4IP 4), ring-cavity CW steady state; example 12 self-starts a soliton
  laser with Kelly sidebands at the predicted offsets. Parabolic gain
  rejects the fourth-order compositions (backward heat equation on the
  negative sub-steps); Lorentzian keeps them at order 4. Also fixed: the
  TensorFlow local-error kernel failed on a field of another rank after a
  relaxed retrace (`tf.norm` lost its scalar shape). Open: fast gain
  dynamics, gain-fiber ASE, rate equations, quintic CGLE.
- Laser phase noise, frequency offset — not modelled (would need
  carrier-phase recovery beyond the per-stream complex gain).
- Rust kernel set (`hpcom.backends.rust`) — seam and conformance tests exist.
- JAX propagator — seam and conformance tests exist.
- OptiCommPy feature-gap survey (old Phase B) — not started.
- `hpcom._legacy` removal in v2.

## Ideas (not planned)

- **OFDM over WDM** — several OFDM bands, each on its own WDM carrier (the
  "multi-band OFDM superchannel" of the 2010s literature). hpcom could build
  it from `OFDMTransmitter` streams placed on a `WDMConfig` grid. Kept as an
  idea: no current study needs it (maintainer, 2026-09-28).
