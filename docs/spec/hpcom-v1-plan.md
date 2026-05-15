# Plan: hpcom v1 Implementation

Source spec: [`hpcom-v1.md`](./hpcom-v1.md).

## Strategy

Build outside-in: **scaffold first, then the seam (Propagator), then absorb ssfm-gpu behind it, then the domain API on top, then tests + docs + CI.** This order keeps the codebase in a runnable state at every checkpoint and means the riskiest piece (the propagator interface) is validated before any user-facing code is written against it.

The **legacy code keeps working throughout** — it moves to `_legacy/` in step 1 and is not touched again until v2.

## Components & Dependencies

```
                    ┌──────────────────────┐
                    │ M0: Repo scaffold    │
                    │ src/ layout, tooling │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ M1: Move old → _legacy │
                    │ Existing tests still pass │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ M2: Propagator ABC   │
                    │ Backend registry      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │ M3: Absorb ssfm-gpu  │
                    │ TF backend impl      │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────▼─────┐   ┌──────▼─────┐   ┌─────▼──────┐
       │ M4: Fiber  │   │ M5: Tx/Rx  │   │ M6: Amp    │
       │ (uses prop)│   │ (modulation)│  │ (EDFA)     │
       └──────┬─────┘   └──────┬─────┘   └─────┬──────┘
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼───────────┐
                    │ M7: Link + Simulation │
                    │ + Result              │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────▼─────┐   ┌──────▼─────┐   ┌─────▼──────┐
       │ M8: Tests  │   │ M9: Docs   │   │ M10: CI +  │
       │ (numerical)│   │ + examples │   │ packaging  │
       └────────────┘   └────────────┘   └────────────┘
```

## Milestones

### M0 — Scaffold (1 day)
- Adopt `src/hpcom/` layout; update `pyproject.toml` with extras (`[tf]`, `[tf-gpu]`, `[dev]`).
- Add `ruff`, `mypy`, `pytest` config to `pyproject.toml`.
- Create empty `tests/{unit,numerical,integration,fixtures}/` and `examples/` dirs.
- **Verify:** `pip install -e ".[dev]"` works; `pytest` runs (zero tests is fine); `ruff check .` passes.

### M1 — Legacy preservation (1 day)
- Move existing `hpcom/{channel,signal,modulation,metrics,plot,decorators,_sklearn_functions,hpcom_old}.py` → `src/hpcom/_legacy/`.
- Add `_legacy/__init__.py` that emits a `DeprecationWarning` on import.
- Triage `tests/`: keep real tests (move to `tests/legacy/`), move `.ipynb` and `data_collection.py` to `notebooks/`.
- Top-level `hpcom/__init__.py` re-exports from `_legacy` for now (so existing code keeps importing).
- **Verify:** existing legacy tests still pass; `import hpcom` still works.

### M2 — Propagator interface (2 days)
- Define `src/hpcom/backends/base.py`: `Propagator` ABC with `propagate(field, fiber, dt_s, direction) -> field`.
- Define `src/hpcom/backends/__init__.py`: registry + `get_propagator(name)` factory.
- Write a stub `NumpyPropagator` (linear-only, no nonlinearity) in `backends/numpy/` purely to validate the interface and enable tests without TF.
- **Verify:** unit test that registers a fake backend, calls `get_propagator("fake")`, runs `.propagate()` on a sine wave through a zero-loss fiber and gets the input back.
- **Checkpoint with user:** confirm interface signature before building TF backend on top of it.

### M3 — Absorb ssfm-gpu as TensorFlow backend (3 days)
- Vendor `ssfm-gpu` source into `src/hpcom/backends/tensorflow/` (preserve original copyright/attribution).
- Wrap as `TensorFlowPropagator(Propagator)` implementing the M2 interface.
- Drop `ssfm-gpu` from `requirements.txt` and `pyproject.toml`.
- Register backend; opportunistic cleanup *only* on the wrapper layer, not the absorbed numerics.
- **Verify:** numerical test — propagate a known signal through `TensorFlowPropagator`, get the same numerical output as the original `ssfm-gpu` package (golden-value test).
- **Risk:** TF GPU env in CI. Mitigation: tag GPU tests `@pytest.mark.gpu`, skip in standard CI, run nightly on a self-hosted runner or locally.

### M4 — Fiber class (1 day)
- `src/hpcom/fiber.py`: dataclass per spec example. `propagate()` delegates to backend.
- Wire dispersion/loss/gamma defaults to SSMF.
- **Verify:** unit test constructs a `Fiber`, calls `propagate()` on a synthetic field, gets a field of the same shape back.

### M5 — Transmitter & Receiver (3 days)
- `src/hpcom/transmitter.py`: `Transmitter` dataclass (modulation order, symbol rate, pulse shape, optional WDM mode). Returns time-domain field.
- `src/hpcom/receiver.py`: `Receiver` (matched filter + dispersion comp + demod). Returns symbols + soft metrics.
- Build by extracting/cleaning the relevant pieces from `_legacy/signal.py` and `_legacy/modulation.py` — copy into new modules, refactor for clarity, do not import from `_legacy`.
- **Verify:** back-to-back test (Transmitter → Receiver, no fiber) recovers symbols with BER = 0 in noiseless case.

### M6 — Amplifier (1 day)
- `src/hpcom/amplifier.py`: `Amplifier` ABC + `EDFA` concrete class (gain, NF, ASE noise model).
- **Verify:** unit test — gain restores power within tolerance; ASE noise statistics match expected variance.

### M7 — Link, Simulation, Result (2 days)
- `src/hpcom/link.py`: `Link` composes ordered list of `[Fiber, Amplifier, ...]` spans.
- `src/hpcom/simulation.py`: `Simulation(transmitter, link, receiver, seed=None).run() -> Result`.
- `src/hpcom/result.py`: `Result` dataclass (received signal, decoded symbols, BER, EVM, MI, metadata: seed, config snapshot, runtime).
- Public re-exports in top-level `__init__.py`. Remove the legacy re-exports from M1 (legacy now only reachable via `hpcom._legacy.*` with deprecation warning).
- **Verify:** end-to-end test — build the example scenario in code, `.run()` it, assert `Result.ber < threshold`.

### M8 — Numerical correctness suite (3 days)
- Implement the five reference cases from the spec:
  1. Soliton shape preservation
  2. Energy conservation (lossless/noiseless)
  3. Linear-only matches closed-form dispersion
  4. AWGN BER curve match
  5. Forward → backward round-trip
- Each test cites its reference and documents tolerance.
- **Verify:** all five pass on `TensorFlowPropagator`.
- **Checkpoint with user:** review tolerances; this is one of the spec's open questions.

### M9 — Docs + examples (3 days)
- Sphinx skeleton: install, quickstart, concepts (Transmitter/Fiber/Amplifier/Receiver/Link), API reference (autodoc), "how to add a backend" guide, examples gallery.
- Write the four example scripts from the spec; each runnable standalone.
- Rewrite README: badges → install → 10-line minimal example → links.
- **Verify:** `sphinx-build` clean; readthedocs build green; each example runs end-to-end.

### M10 — CI + packaging (2 days)
- `.github/workflows/ci.yml`: lint (ruff), type-check (mypy), unit + numerical tests on Python 3.10/3.11/3.12 × {Linux, macOS}, wheel build.
- `.github/workflows/release.yml`: tag-triggered PyPI publish via trusted publishing.
- Verify clean-venv install of the built wheel runs the minimal example.
- **Verify:** CI green on `main`; test-PyPI publish dry-run succeeds.

## Parallelization

After M3, three streams can proceed in parallel if there's bandwidth: M4+M6 (small), M5 (largest), M9 docs skeleton (can start once M2 lands).

If solo: serial M0→M10 is fine; the plan is sized in days, not weeks.

## Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Propagator interface needs to change after M3 (ripple cost) | Medium | M2 checkpoint with user before building on it; design with future JAX/Rust backends in mind |
| Absorbed ssfm-gpu numerics differ from original after vendoring | Low–Medium | Golden-value test in M3 against the original package output |
| TF GPU not available in CI | High | Mark GPU tests, run nightly on self-hosted/local; non-GPU CI uses `NumpyPropagator` for interface tests |
| Numerical tolerances in M8 too tight (false failures) or too loose (hides bugs) | Medium | M8 user checkpoint; document derivation of each tolerance |
| Scope creep: "while we're rewriting Receiver, let's also add new equalizers" | High | Boundaries in spec — new physical models are out of scope; reject in PR review |
| Legacy re-exports in M1 hide breaking changes from users until M7 removes them | Low | Documented in CHANGELOG; release notes call out the boundary |

## Verification Checkpoints (user review gates)

1. **After M2:** Propagator interface signature — once we build on it, changing it is expensive.
2. **After M3:** Numerical parity with original ssfm-gpu — confirms the absorption preserved physics.
3. **After M7:** End-to-end example runs — first moment a user could realistically try the new API.
4. **After M8:** Numerical tolerances — physics-judgment call, needs your eyes.
5. **Before M10 release:** Full README + docs + example walkthrough — last chance to catch UX issues.

## Estimated Effort

~22 working days end-to-end if serial, less with parallelization. Real calendar time depends on your part-time availability.
