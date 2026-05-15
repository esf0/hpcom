# hpcom v1 — Confirmed Intent

Captured via `/agent-skills:interview-me` on 2026-05-15.

## Outcome
Transform hpcom from a research codebase into an adoptable open-source library that optical-comm scientists will pick up, cite, and recommend — with data-mining/GPU workflows as the differentiating layer on top.

## User
- **Primary:** optical-comm scientists and PhD students who know the physics and want a clean, trustworthy simulation library.
- **Strategic:** ML / data-science people generating optical-channel datasets, served by workflows built on the same API.

## Why now
OptiCommPy launched after hpcom and is winning mindshare despite hpcom having stronger GPU/data-mining capabilities. The gap is presentation and API, not capability.

## Success (v1)
A new user can `pip install hpcom`, open an example like *"16-QAM over 1000 km SSMF + EDFA chain"*, run it, get a structured result object back, and trust the numbers — without reading the source.

## Constraint
Backend-agnostic API is the spine, with an explicit **propagator interface layer** in v1 (e.g., a `Propagator` / `FiberBackend` abstraction). ssfm-gpu (TensorFlow) plugs in as the first implementation; a JAX implementation can drop in later via the same interface without breaking user code.

## v1 scope
- Domain-shaped composable API: `Transmitter`, `Fiber`, `Amplifier`, `Receiver`, `Link` (names TBD in spec).
- Structured result classes (no raw dicts as user-facing return values).
- Propagator interface defined; ssfm-gpu absorbed into the package behind it as the sole v1 backend.
- Numerical-correctness test suite (analytical references, conservation laws, comparison vs. published results).
- Docs site + runnable example scenarios + pip-installable packaging.
- Opportunistic cleanup as modules move behind the new API — refactor messy areas, fix latent bugs, improve readability.

## Out of scope for v1
- TensorFlow → JAX migration of SSFM.
- Rust rewrites of signal generation or supplementary functions.
- High-order SSFM schemes.
- Full OptiCommPy capability parity.
- New physical-effect models beyond what hpcom already has.
- Standalone refactor sweeps decoupled from the API migration.

## Discipline notes
- Cleanup happens *as code moves behind the new interfaces*, never as a separate project.
- Rust experiments and JAX migration are deliberately deferred — the API seam where they will land is built in v1, the implementations are not.
- "I want Rust experience" is a personal motivation, not a user-facing requirement; it does not gate v1.
