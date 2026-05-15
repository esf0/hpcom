# Spec: hpcom v1

Source intent: [`docs/intent/hpcom-v1.md`](../intent/hpcom-v1.md). This spec translates that intent into concrete, testable requirements.

## Objective

Ship hpcom v1 — a Python library for GPU-accelerated optical-comm simulation that an optical-comm scientist or PhD student can `pip install`, run an example, and trust without reading the source. Data-mining/dataset-generation workflows ride on the same API as a strategic second audience.

**User stories:**
- As an optical-comm scientist, I can build a `16-QAM over 1000 km SSMF + EDFA chain` simulation in <30 lines and inspect a structured result object — without learning hpcom internals.
- As an ML researcher, I can take any scenario built with hpcom and bulk-generate a labeled dataset with one extra call.
- As a future contributor, I can read the codebase and add a new amplifier or fiber model without rewriting the core.

**Non-goals (v1):** JAX backend, Rust extensions, high-order SSFM schemes, full OptiCommPy capability parity, new physical-effect models.

## Tech Stack

- **Language:** Python ≥ 3.10
- **Numerics:** NumPy, SciPy
- **GPU backend (v1):** TensorFlow ≥ 2.15 (absorbed `ssfm-gpu` code, no external dep)
- **Plotting:** Matplotlib
- **Tests:** pytest, pytest-cov, hypothesis (for property tests where useful)
- **Lint/format:** ruff (lint + format)
- **Type checking:** mypy (strict on new code, lenient on `_legacy`)
- **Docs:** Sphinx + sphinx-autodoc + furo theme, hosted on readthedocs
- **CI:** GitHub Actions (lint, type-check, test on Linux/macOS, build wheel)
- **Build:** `setuptools` via `pyproject.toml`, `src/` layout
- **License:** GPLv3 (unchanged)

Install extras:
- `pip install hpcom` — CPU-only core (NumPy/SciPy)
- `pip install hpcom[tf]` — adds TensorFlow CPU
- `pip install hpcom[tf-gpu]` — adds TensorFlow with GPU support
- `pip install hpcom[dev]` — pytest, ruff, mypy, sphinx, etc.

## Commands

```
Install (dev):     pip install -e ".[dev,tf]"
Test:              pytest
Test w/ coverage:  pytest --cov=hpcom --cov-report=term-missing
Lint:              ruff check .
Format:            ruff format .
Type check:        mypy src/hpcom
Build docs:        sphinx-build -b html docs/source docs/_build/html
Build wheel:       python -m build
Run examples:      python examples/<scenario>.py
```

## Project Structure

```
src/hpcom/
    __init__.py            # public API re-exports (Transmitter, Fiber, Amplifier, Receiver, Link, Simulation, Result)
    transmitter.py         # Transmitter class (modulation + pulse shaping + WDM)
    fiber.py               # Fiber class (parameters + propagate() that delegates to backend)
    amplifier.py           # Amplifier classes (EDFA first; ABC for future types)
    receiver.py            # Receiver class (matched filter, dispersion comp, demod)
    link.py                # Link = composable chain of components
    simulation.py          # Simulation runner; turns a Link into a Result
    result.py              # Result dataclass(es): signal, metrics, metadata
    metrics/               # BER, EVM, MI, energy — pure functions, NumPy in/out
    modulation/            # constellations, mapping/demapping
    backends/
        __init__.py        # Propagator registry + factory
        base.py            # Propagator ABC: propagate(field, fiber, direction) -> field
        tensorflow/        # absorbed ssfm-gpu code, behind Propagator interface
            __init__.py
            ssfm.py
            manakov.py
    _legacy/               # old top-level modules preserved verbatim for one release
        channel.py
        signal.py
        ...
        README.md          # "Deprecated; will be removed in v2. New code: see hpcom.*"
tests/
    unit/                  # fast, no GPU, pure-Python correctness
    integration/           # end-to-end scenarios; may be slow; opt-in via marker
    numerical/             # analytical reference cases (soliton, energy conservation, BER vs theory)
    fixtures/              # shared sample signals/configs
examples/
    01_b2b_qam.py
    02_ssmf_edfa_chain.py
    03_wdm_manakov.py
    04_dataset_generation.py
notebooks/                 # exploratory; not run in CI
docs/
    intent/                # confirmed-intent docs (this folder is input, not output)
    spec/                  # this spec and future specs
    source/                # Sphinx sources
    _build/                # generated; gitignored
.github/workflows/
    ci.yml                 # lint, type, test, build
    docs.yml               # build docs on tag
pyproject.toml
README.md
LICENSE
```

## Code Style

```python
# src/hpcom/fiber.py
from dataclasses import dataclass
from typing import Literal

import numpy as np

from hpcom.backends import get_propagator
from hpcom._typing import ComplexArray, Direction


@dataclass(frozen=True)
class Fiber:
    """Single span of optical fiber.

    Parameters use SI units except where noted (km for length, dB/km for loss).
    Frozen so a Fiber instance can be shared across simulations safely.
    """

    length_km: float
    alpha_db_per_km: float = 0.2
    beta2_s2_per_m: float = -2.16e-26
    gamma_per_w_per_km: float = 1.27
    n_steps: int = 100
    backend: Literal["tensorflow"] = "tensorflow"

    def propagate(self, field: ComplexArray, dt_s: float, direction: Direction = "forward") -> ComplexArray:
        propagator = get_propagator(self.backend)
        return propagator.propagate(field=field, fiber=self, dt_s=dt_s, direction=direction)
```

**Conventions:**
- `snake_case` functions/variables, `PascalCase` classes, `_leading_underscore` for internals.
- Units in variable names when ambiguous: `length_km`, `power_dbm`, `dt_s`.
- Public functions and classes have type hints and docstrings (NumPy style).
- Domain classes are dataclasses where state is config; mutable state lives on `Simulation`/`Result`.
- One concept per module; modules > 600 lines are a code smell.
- No raw dicts as user-facing return values — use a dataclass or `Result`.
- No `from x import *` outside of `__init__.py` re-exports.
- Side-effecting code (file I/O, plotting) is opt-in, never inside numerical functions.

## Testing Strategy

**Levels:**
- `tests/unit/` — pure-function correctness. Fast (<1s each). No GPU. Run on every commit.
- `tests/numerical/` — analytical references that prove the physics:
  - Soliton fundamental: shape preserved over N spans within tolerance
  - Energy conservation in lossless/noiseless propagation
  - Linear-only propagation matches closed-form dispersion
  - BER vs. theoretical AWGN curve for back-to-back QAM
  - Round-trip: propagate forward then backward → recover input within tolerance
- `tests/integration/` — full scenarios end-to-end. Marked `@pytest.mark.slow`, run in nightly CI and pre-release.
- `tests/fixtures/` — shared synthetic signals and golden config dicts.

**Coverage targets:**
- New code in `src/hpcom/` (excluding `_legacy/`): **≥ 85%** line coverage.
- `_legacy/` is excluded from coverage and lint enforcement.

**Conventions:**
- Test file mirrors source: `src/hpcom/fiber.py` ↔ `tests/unit/test_fiber.py`.
- One behavior per test; test names describe the behavior, not the function.
- Numerical tests state tolerance and cite the reference (paper, textbook, or analytical derivation).

## Boundaries

**Always do:**
- Write a numerical-correctness test for any new physical model (fiber, amplifier, propagator).
- Run `ruff check`, `mypy`, and `pytest` before committing.
- Update the spec when scope or interfaces change.
- Use SI units in public APIs; if a non-SI unit is exposed (km, dBm), put it in the variable name.

**Ask first:**
- Adding a new top-level public class or module.
- Changing the `Propagator` interface signature (any backend implementor depends on it).
- Adding a new runtime dependency.
- Touching `_legacy/` for anything other than deletion.
- Changing GitHub Actions config or release process.

**Never do:**
- Commit secrets, datasets >1 MB, or generated artifacts.
- Return raw dicts from public APIs.
- Add silent fallbacks that hide numerical errors.
- Skip a failing test by marking it `xfail` without an issue link explaining why.
- Re-export anything from `_legacy/` in the new public API.

## Success Criteria

v1 is shippable when **all** of these are true:

1. `pip install hpcom[tf]` from PyPI in a clean venv works on Linux + macOS, Python 3.10/3.11/3.12.
2. `examples/02_ssmf_edfa_chain.py` runs end-to-end, prints a `Result` summary, and produces sane numbers — verified by a test in `tests/integration/`.
3. All five `tests/numerical/` reference cases listed above pass with documented tolerances.
4. `pytest tests/unit tests/numerical` runs in <60s on a developer laptop with no GPU.
5. CI green on `main`: lint, type-check, unit + numerical tests, wheel build, docs build.
6. `Propagator` interface defined and the absorbed ssfm-gpu code lives behind it. A documented "how to add a new backend" page exists in the docs.
7. README rewritten: install → 10-line minimal example → link to full examples and docs.
8. Sphinx docs published on readthedocs covering every public class with at least one runnable example.
9. Coverage ≥ 85% on `src/hpcom/` excluding `_legacy/`.
10. Old top-level functions still importable from `hpcom._legacy` with a `DeprecationWarning` on import.

## Open Questions

1. **Naming of the chain class:** `Link` vs. `System` vs. `Chain` vs. `Line`. I'm using `Link` as a placeholder; defer until first prototype. Link is fine. One link is combined with spans, right? spans can be different types of fiber.
2. **WDM as a `Transmitter` mode or a separate `WDMTransmitter` class?** Lean toward a mode/parameter on `Transmitter` to keep the surface area small, but this depends on how different the WDM signal-gen code path actually is. Transmitter and it takes details about transmission. wdm is mode, also can have some geometric canstellation shaping, or probablility shaping or even some arbitrary symbol sequence, e.g. we can add pilot symbols.
3. **Result persistence:** does v1 need built-in save/load (HDF5? Parquet? pickle?), or is `Result` just an in-memory dataclass that users serialize themselves? I lean toward in-memory only for v1. What is the best standard now? parquet? 
4. **Reproducibility:** should `Simulation` take an explicit `seed` and stamp it on the `Result`? I'd say yes — cheap, high-value for a scientific library. yes, reproducibility is extremely needed.
5. **Concrete numerical tolerances** for the five reference cases — needs a pass with the actual physics before locking numbers in. tolerance is good question, should we have some flexability depending on the simulation, e.g. for telecom we can simulate every 1km step but for some we will need 1mm step.
6. **Manakov vs. scalar SSFM as separate fibers or a single `Fiber(polarization="dual")` mode?** Affects how the API feels for dual-pol users. yes, models should be defined by fiber, also there we will add higer order schemes etc
