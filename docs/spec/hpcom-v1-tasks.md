# Tasks: hpcom v1

Source plan: [`hpcom-v1-plan.md`](./hpcom-v1-plan.md). Tasks are ordered by dependency. Each is sized for a single focused session and touches ≤ ~5 files.

## M0 — Scaffold

- [ ] **T0.1 — Switch to `src/` layout**
  - Acceptance: `src/hpcom/__init__.py` exists; old top-level `hpcom/` dir removed; package importable after `pip install -e .`.
  - Verify: `python -c "import hpcom; print(hpcom.__file__)"` resolves under `src/`.
  - Files: `pyproject.toml`, move `hpcom/` → `src/hpcom/`.

- [ ] **T0.2 — Configure `pyproject.toml` extras and tooling**
  - Acceptance: `[project.optional-dependencies]` defines `tf`, `tf-gpu`, `dev`. `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]` present. Python ≥ 3.10.
  - Verify: `pip install -e ".[dev]"` succeeds in clean venv.
  - Files: `pyproject.toml`, delete `requirements.txt`.

- [ ] **T0.3 — Create empty test + example dirs and minimal CI placeholder**
  - Acceptance: `tests/{unit,numerical,integration,fixtures}/__init__.py`, `examples/`, `notebooks/` all exist with `.gitkeep` or stub.
  - Verify: `pytest` runs cleanly with zero collected tests; `ruff check .` passes.
  - Files: directory adds + `.gitignore` update for `_build/`.

## M1 — Legacy preservation

- [ ] **T1.1 — Move existing modules to `_legacy/`**
  - Acceptance: all current `.py` files under `src/hpcom/_legacy/`; original imports inside them updated to relative.
  - Verify: `python -c "from hpcom._legacy import channel"` works.
  - Files: move 8 files; `src/hpcom/_legacy/__init__.py` (with `DeprecationWarning`).

- [ ] **T1.2 — Top-level shim re-exports legacy API**
  - Acceptance: `import hpcom; hpcom.channel.get_default_channel_parameters()` still works (via re-export).
  - Verify: ad-hoc smoke test from a clean shell session.
  - Files: `src/hpcom/__init__.py`.

- [ ] **T1.3 — Triage existing tests**
  - Acceptance: real test files (test_*.py) moved to `tests/legacy/` and pass; `.ipynb` and `data_collection.py` moved to `notebooks/`.
  - Verify: `pytest tests/legacy` green (or all-skip with documented reason).
  - Files: `tests/` reorganization.

## M2 — Propagator interface

- [ ] **T2.1 — `Propagator` ABC**
  - Acceptance: `src/hpcom/backends/base.py` defines `Propagator` with abstract `propagate(field, fiber, dt_s, direction)`.
  - Verify: subclass without implementing raises `TypeError`.
  - Files: `src/hpcom/backends/base.py`, `src/hpcom/_typing.py` (`ComplexArray`, `Direction`).

- [ ] **T2.2 — Backend registry**
  - Acceptance: `register_propagator(name, cls)` and `get_propagator(name)` in `backends/__init__.py`; unknown name raises with helpful message listing registered backends.
  - Verify: unit test registers a fake backend and resolves it.
  - Files: `src/hpcom/backends/__init__.py`, `tests/unit/test_backend_registry.py`.

- [ ] **T2.3 — Stub `NumpyPropagator` (linear-only)**
  - Acceptance: `backends/numpy/linear.py` implements dispersion-only propagation in pure NumPy.
  - Verify: `tests/numerical/test_linear_dispersion.py` — propagating a Gaussian pulse matches closed-form chirp within 1e-10.
  - Files: `src/hpcom/backends/numpy/__init__.py`, `linear.py`, test.
  - **🛑 USER CHECKPOINT after T2.3:** confirm `Propagator` interface signature.

## M3 — Absorb ssfm-gpu

- [ ] **T3.1 — Vendor ssfm-gpu source into the tree**
  - Acceptance: ssfm-gpu code under `src/hpcom/backends/tensorflow/_vendor/` with original copyright/attribution preserved in a `THIRD_PARTY_NOTICES.md`.
  - Verify: `from hpcom.backends.tensorflow._vendor import propagation` works.
  - Files: copy ssfm-gpu source; add NOTICE file.

- [ ] **T3.2 — `TensorFlowPropagator` wrapper**
  - Acceptance: `backends/tensorflow/propagator.py` implements `Propagator`, dispatches scalar vs. Manakov by fiber config.
  - Verify: unit test runs `propagate()` end-to-end on small input without error.
  - Files: `backends/tensorflow/__init__.py`, `propagator.py`.

- [ ] **T3.3 — Numerical parity test (golden values)**
  - Acceptance: `tests/numerical/test_tf_parity.py` propagates a fixed seeded signal and asserts output matches a checked-in golden array within 1e-12.
  - Verify: test green; golden generated once via the original ssfm-gpu pip package on a separate branch and committed.
  - Files: test + golden `.npz` fixture.

- [ ] **T3.4 — Drop external `ssfm-gpu` dependency**
  - Acceptance: `pyproject.toml` no longer lists `ssfm-gpu`; project installs without it.
  - Verify: `pip install -e ".[tf]"` in clean venv succeeds.
  - Files: `pyproject.toml`.
  - **🛑 USER CHECKPOINT after T3.3:** confirm numerical parity is acceptable.

## M4 — Fiber

- [ ] **T4.1 — `Fiber` dataclass**
  - Acceptance: per spec code-style example. `propagate()` resolves backend and delegates.
  - Verify: `tests/unit/test_fiber.py` — construct, propagate noise input, assert shape and dtype preserved.
  - Files: `src/hpcom/fiber.py`, test.

- [ ] **T4.2 — SSMF default factory**
  - Acceptance: `Fiber.ssmf(length_km)` classmethod returning standard SSMF parameters.
  - Verify: unit test on default values.
  - Files: `src/hpcom/fiber.py` (extend), test.

## M5 — Transmitter & Receiver

- [ ] **T5.1 — `modulation/` package extracted from legacy**
  - Acceptance: pure-NumPy QAM/PSK constellation gen, mapping, demapping under `src/hpcom/modulation/`. No legacy imports.
  - Verify: `tests/unit/test_modulation.py` — round-trip mapping for QAM-4/16/64.
  - Files: `src/hpcom/modulation/{__init__,constellations,mapping}.py`, test.

- [ ] **T5.2 — `Transmitter` class**
  - Acceptance: `Transmitter(modulation_order, symbol_rate_hz, n_symbols, pulse="rrc", roll_off=0.1).generate() -> (field, dt_s, symbols)`.
  - Verify: unit test on shape, power, and that returned `symbols` round-trip via mapping.
  - Files: `src/hpcom/transmitter.py`, test.

- [ ] **T5.3 — `Receiver` class**
  - Acceptance: matched filter + dispersion comp + demap. `receive(field, dt_s, link_meta) -> (symbols, soft_metrics)`.
  - Verify: unit test back-to-back with `Transmitter` (no fiber): noiseless BER == 0.
  - Files: `src/hpcom/receiver.py`, test.

- [x] **T5.4 — WDM mode on `Transmitter`/`Receiver`** *(Simulation integration deferred — current guard raises NotImplementedError)*
  - Acceptance: `Transmitter(..., wdm=WDMConfig(n_channels=5, spacing_hz=50e9))`. Receiver demuxes channel `i`.
  - Verify: 3-channel back-to-back; center channel BER == 0.
  - Files: `src/hpcom/transmitter.py`, `receiver.py`, `wdm.py`, test.

## M6 — Amplifier

- [ ] **T6.1 — `Amplifier` ABC + `EDFA`**
  - Acceptance: `EDFA(gain_db, noise_figure_db).amplify(field, dt_s) -> field`. ASE noise model documented.
  - Verify: unit test — gain restores power within 0.1 dB; ASE variance matches `h*f*(G-1)*NF*B` formula within 1%.
  - Files: `src/hpcom/amplifier.py`, test.

## M7 — Link, Simulation, Result

- [ ] **T7.1 — `Link` composition**
  - Acceptance: `Link([Fiber, Amplifier, Fiber, Amplifier, ...]).propagate(field, dt_s) -> field`. Validates ordering (fiber/amp alternation if `strict=True`).
  - Verify: unit test composes 3-span link, asserts output power consistent with link budget.
  - Files: `src/hpcom/link.py`, test.

- [ ] **T7.2 — `Result` dataclass**
  - Acceptance: fields = `received_field`, `decoded_symbols`, `metrics: Metrics`, `metadata: SimulationMetadata` (seed, timestamp, runtime_s, config snapshot as dict).
  - Verify: unit test on construction + serialization to dict.
  - Files: `src/hpcom/result.py`, test.

- [ ] **T7.3 — `Simulation` runner**
  - Acceptance: `Simulation(transmitter, link, receiver, seed=None).run() -> Result`. Seed stamps RNG and is recorded in `Result.metadata`.
  - Verify: integration test — runs the SSMF+EDFA scenario end-to-end; `Result.metrics.ber < 0.1` for low-noise config.
  - Files: `src/hpcom/simulation.py`, `tests/integration/test_ssmf_edfa.py`.

- [ ] **T7.4 — Promote new API to top-level, remove legacy re-exports**
  - Acceptance: `from hpcom import Transmitter, Fiber, Amplifier, EDFA, Receiver, Link, Simulation, Result` works. Legacy reachable only via `hpcom._legacy.*` with `DeprecationWarning`.
  - Verify: smoke test imports + `pytest -W error::DeprecationWarning` on new tests is clean.
  - Files: `src/hpcom/__init__.py`.
  - **🛑 USER CHECKPOINT after T7.4:** end-to-end UX review of the new top-level API.

## M8 — Numerical correctness suite

- [ ] **T8.1 — Soliton fundamental shape preservation**
  - Acceptance: launch fundamental soliton, propagate N=10 spans of dispersion-managed link, peak power and FWHM within documented tolerance.
  - Verify: test green; tolerance + reference cited in docstring.
  - Files: `tests/numerical/test_soliton.py`.

- [ ] **T8.2 — Energy conservation (lossless / noiseless)**
  - Acceptance: with α=0, NF=0, total energy preserved within 1e-6 over 1000 km.
  - Verify: test green.
  - Files: `tests/numerical/test_energy_conservation.py`.

- [ ] **T8.3 — Linear-only matches closed-form dispersion**
  - Acceptance: with γ=0, propagated Gaussian matches analytical chirped-Gaussian within 1e-8.
  - Verify: test green.
  - Files: `tests/numerical/test_linear_dispersion.py` (extend M2 stub).

- [ ] **T8.4 — AWGN BER curve match**
  - Acceptance: back-to-back QAM with AWGN noise; measured BER vs SNR curve within 0.5 dB of theoretical at BER=1e-3.
  - Verify: test green at three SNR points.
  - Files: `tests/numerical/test_awgn_ber.py`.

- [ ] **T8.5 — Forward → backward round-trip**
  - Acceptance: propagate forward then backward (DBP) recovers input within 1e-4 RMS error on a noiseless single-span.
  - Verify: test green.
  - Files: `tests/numerical/test_round_trip.py`.
  - **🛑 USER CHECKPOINT after T8.5:** review tolerances; tighten or loosen with physics judgment.

## M9 — Docs + examples

- [ ] **T9.1 — Sphinx skeleton**
  - Acceptance: `docs/source/{conf.py, index.rst}`, furo theme, autodoc + napoleon enabled. `sphinx-build` clean.
  - Verify: local HTML build succeeds; readthedocs config updated.
  - Files: `docs/source/*`, `.readthedocs.yml`.

- [ ] **T9.2 — Concepts + quickstart pages**
  - Acceptance: pages for install, quickstart (10-line example), each domain class, "how backends work", "how to add a backend".
  - Verify: every page renders; no broken cross-refs.
  - Files: `docs/source/*.rst`.

- [ ] **T9.3 — Four example scripts**
  - Acceptance: `examples/{01_b2b_qam,02_ssmf_edfa_chain,03_wdm_manakov,04_dataset_generation}.py` each runnable standalone, prints `Result` summary.
  - Verify: each runs in <60s on CPU (or skipped with marker if GPU-only); integration test runs example 02.
  - Files: 4 example files + 1 integration test.

- [ ] **T9.4 — README rewrite**
  - Acceptance: badges → 1-paragraph pitch → install → 10-line minimal example → links to examples and docs.
  - Verify: rendered preview reads well; PyPI long-description renders clean.
  - Files: `README.md`.

## M10 — CI + packaging

- [x] **T10.1 — GitHub Actions CI**
  - Acceptance: `.github/workflows/ci.yml` runs ruff + mypy + `pytest tests/unit tests/numerical` on Python 3.10/3.11/3.12 × {ubuntu-latest, macos-latest}.
  - Verify: green on a draft PR.
  - Files: `.github/workflows/ci.yml`.

- [x] **T10.2 — GPU-marked tests opt-in**
  - Acceptance: `@pytest.mark.gpu` skipped by default in CI; `pytest -m gpu` runs them locally / on self-hosted.
  - Verify: standard CI run skips them; manual run executes them.
  - Files: `pyproject.toml` (pytest config), test markers.

- [x] **T10.3 — Wheel build + clean-venv install verification**
  - Acceptance: `python -m build` produces sdist + wheel; clean venv `pip install dist/*.whl` runs example 02.
  - Verify: workflow job that builds wheel and runs the example.
  - Files: `.github/workflows/ci.yml` (extend).

- [x] **T10.4 — Release workflow (tag → PyPI)** *(workflow written; live tag awaiting user checkpoint)*
  - Acceptance: `.github/workflows/release.yml` publishes on `v*` tag via PyPI Trusted Publishing.
  - Verify: dry-run against TestPyPI succeeds.
  - Files: `.github/workflows/release.yml`.
  - **🛑 USER CHECKPOINT before first real `vX.Y.Z` tag:** full pre-release walkthrough.

## M11 — Phase 4: Constellation shaping

Geometric and probabilistic shaping. Per user directive 2026-05-17, this is
the next active milestone after M10. Tasks are pure-NumPy, framework-free,
and keep the `Transmitter`/`Receiver` API stable by routing through a new
`ConstellationConfig` object that can substitute for `modulation_order`.

- [ ] **T11.1 — `ConstellationConfig` dataclass**
  - Acceptance: `hpcom.modulation.ConstellationConfig(points, bit_labels, probabilities=None)` validates power-of-2 size, `bit_labels` shape `(M, log₂ M)` with one-to-one bit→symbol coverage, unit-mean-power normalisation, and optional probability vector summing to 1. Factories `from_square_qam(M)` and `from_points(points, bit_labels=None)` (auto-Gray for arbitrary points via nearest-neighbour labelling).
  - Verify: round-trip `bits_to_symbols / symbols_to_bits` matches the existing `square_qam` path bit-for-bit when constructed via `from_square_qam`.
  - Files: `src/hpcom/modulation/{__init__.py, shaping.py}`, `tests/unit/test_constellation_config.py`.

- [ ] **T11.2 — Wire `ConstellationConfig` into `Transmitter`/`Receiver`**
  - Acceptance: `Transmitter(..., constellation=ConstellationConfig(...))` accepted as alternative to `modulation_order`; if both given, raise. Receiver uses the same config for demap (`symbols_to_bits` reads bit_labels from config when present).
  - Verify: noiseless B2B with a custom (but still square-QAM-equivalent) `ConstellationConfig` recovers bits → BER == 0.
  - Files: `src/hpcom/transmitter.py`, `src/hpcom/receiver.py`, `tests/unit/test_transmitter_constellation_config.py`.

- [ ] **T11.3 — Geometric shaping: non-square constellations**
  - Acceptance: factories for APSK (concentric rings) and one 16-point geometrically-shaped constellation. Each returns a valid `ConstellationConfig`.
  - Verify: round-trip B2B BER == 0 noiseless; constellation mean power == 1.0 within 1e-12; minimum-distance ≥ documented threshold.
  - Files: `src/hpcom/modulation/shaping.py` (extend), `tests/unit/test_geometric_shaping.py`.

- [ ] **T11.4 — Probabilistic shaping: Maxwell–Boltzmann + simple DM**
  - Acceptance: `maxwell_boltzmann(constellation, nu)` returns a probability vector minimising `E[|s|²]` at fixed entropy. A minimal distribution matcher — shell-mapping lookup for v1, *not* CCDM — maps uniform bits → symbol indices following the target distribution. Decoder inverts the lookup.
  - Verify: empirical symbol distribution from the DM matches target within KL-divergence < 1e-3 over ≥ 2¹⁶ symbols; round-trip noiseless recovers input bits.
  - Files: `src/hpcom/modulation/shaping.py` (extend), `src/hpcom/modulation/dm.py`, `tests/unit/test_probabilistic_shaping.py`.

- [ ] **T11.5 — Numerical shaping-gain test (AWGN)**
  - Acceptance: 64-QAM shaped vs unshaped over AWGN at matched entropy; measured SNR-gap-to-Shannon for shaped is smaller by ≥ 0.5 dB where the gain is expected (~3 bit/s/Hz region).
  - Verify: `tests/numerical/test_shaping_gain.py` green within documented tolerance.
  - Files: `tests/numerical/test_shaping_gain.py`.
  - **🛑 USER CHECKPOINT after T11.5:** review numbers + decide whether to extend to CCDM or stay with shell mapping for v1.

## M12 — Phase 5: OFDM Tx/Rx

Minimal OFDM transmitter/receiver, reusing the `Transmitter`/`Receiver`
base via subclassing (same pattern as `WDMTransmitter`). Single-pol first;
dual-pol if straightforward after the single-pol API is fixed.

- [ ] **T12.1 — `OFDMConfig` dataclass**
  - Acceptance: `OFDMConfig(n_subcarriers, cp_len, n_guard, pilot_subcarrier_indices=())` with validation (power-of-2 `n_subcarriers`, `0 ≤ n_guard < n_subcarriers/2`, disjoint pilot/data indices).
  - Verify: malformed configs raise; valid configs round-trip via dataclass replace.
  - Files: `src/hpcom/ofdm.py`, `tests/unit/test_ofdm_config.py`.

- [ ] **T12.2 — `OFDMTransmitter(Transmitter)`**
  - Acceptance: required kw-only `ofdm: OFDMConfig`. Generates IFFT-based OFDM symbols with cyclic prefix, guard bands, optional pilot subcarriers (taken from `pilots: PilotConfig`). Returns `(field, dt_s, symbols)`. Single-pol first.
  - Verify: output length = `n_symbols * (n_subcarriers + cp_len)`; power matches `p_ave` within 1e-10; constellation on data subcarriers is the configured QAM (or `ConstellationConfig`).
  - Files: `src/hpcom/ofdm.py` (extend), `tests/unit/test_ofdm_transmitter.py`.

- [ ] **T12.3 — `OFDMReceiver(Receiver)`**
  - Acceptance: removes CP, FFT, extracts data subcarriers, demaps with the configured constellation. Pilot-based one-tap channel equalisation when pilots are present.
  - Verify: noiseless B2B `OFDMTransmitter → OFDMReceiver` recovers bits at BER == 0 for QPSK/16-QAM/64-QAM.
  - Files: `src/hpcom/ofdm.py` (extend), `tests/unit/test_ofdm_receiver.py`.

- [ ] **T12.4 — OFDM AWGN BER curve**
  - Acceptance: measured AWGN BER over a swept SNR matches the per-subcarrier theoretical QAM-AWGN curve within 0.5 dB at BER=1e-3.
  - Verify: `tests/numerical/test_ofdm_awgn.py` green at three SNR points.
  - Files: `tests/numerical/test_ofdm_awgn.py`.
  - **🛑 USER CHECKPOINT after T12.4:** decide on dual-pol OFDM scope and whether to add OFDM × WDM composition.

## M13 — Phase 6: OptiCommPy gap analysis

Survey-driven. Implementation tasks added after the survey.

- [ ] **T13.1 — OptiCommPy API survey**
  - Acceptance: `docs/spec/opticomm_gap.md` lists every public function/class in OptiCommPy, classifies each as {already in hpcom, planned, intentionally omitted, candidate for adoption}, with a one-line rationale.
  - Verify: list is exhaustive (cross-check against `dir(opticommpy)` from a clean install); candidates ranked by user value.
  - Files: `docs/spec/opticomm_gap.md`.
  - **🛑 USER CHECKPOINT after T13.1:** pick which candidates make it into v1.

- [ ] **T13.2 — Implement adopted candidates** *(spawned per T13.1 outcome)*

## Summary

41 → 53 tasks across 13 milestones. 7 user checkpoints (after T2.3, T3.3, T7.4, T8.5, before T10.4, after T11.5, T12.4, T13.1). Critical path through M0–M10 unchanged; M11–M13 chain sequentially after M10. T11.1–11.2 must complete before T11.3–11.5; M12 reuses M11 for shaped-OFDM but otherwise depends only on M5/M7.
