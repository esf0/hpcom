# hpcom — High-Performance Optical Communication Library

[![PyPI](https://img.shields.io/pypi/v/hpcom.svg)](https://pypi.python.org/pypi/hpcom)
[![Python](https://img.shields.io/pypi/pyversions/hpcom.svg)](https://pypi.python.org/pypi/hpcom)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hpcom.readthedocs.io)
[![DOI](https://zenodo.org/badge/564802755.svg)](https://zenodo.org/badge/latestdoi/564802755)
[![Licence: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

GPU-accelerated, backend-agnostic simulation of coherent optical communication
links — designed for optical-comm scientists who want a clean, trustworthy
library and for ML researchers who need labelled optical-channel datasets.

The split-step Fourier method (SSFM) and Manakov solvers run on TensorFlow
today, behind a swappable backend interface; JAX and Rust backends are on the
roadmap without breaking user code.

## Install

```bash
pip install "hpcom[tf]"        # CPU TensorFlow, ready to run on a laptop
pip install "hpcom[tf-gpu]"    # GPU TensorFlow (Linux + CUDA)
pip install "hpcom[dev]"       # add tests, docs, lint, type checking
```

Python ≥ 3.10. The base install (`pip install hpcom`) gives you the public
domain API and the NumPy linear backend; TensorFlow is optional, pulled in by
the `[tf]` extra.

## Hello, fibre

```python
from hpcom import EDFA, Fiber, Link, Receiver, Simulation, Transmitter

fiber = Fiber.ssmf(length_km=80.0, n_steps=20, backend="tensorflow")
edfa  = EDFA(gain_db=fiber.alpha_db_per_km * fiber.length_km, noise_figure_db=5.0)
link  = Link.from_spans(n_spans=3, fiber=fiber, amplifier=edfa)

tx = Transmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=4096,
                 samples_per_symbol=4, seed=0)
rx = Receiver(modulation_order=16, samples_per_symbol=4,
              equalize="unit_power",
              cd_compensation_s2=link.total_dispersion_s2)

result = Simulation(tx, link, rx, seed=0, launch_power_dbm=-3.0).run()
print(result)
# Result(ber=0.000e+00, evm_rms=9.7e-02, n_bits=16256, runtime=0.27s)
```

More worked examples in [`examples/`](examples/):

- `01_b2b_qam.py` — back-to-back QAM, no fibre.
- `02_ssmf_edfa_chain.py` — flagship 3-span SSMF + EDFA scenario.
- `03_dual_pol_manakov.py` — dual-polarization Manakov propagation.
- `04_dataset_generation.py` — bulk dataset generation for ML workflows.

## What's in the box

- **Composable domain API.** `Transmitter`, `Fiber`, `Amplifier` (`EDFA`
  built-in), `Receiver`, `Link`, `Simulation` — all frozen dataclasses with
  validated config. Add your own `LinkComponent` (filter, DCM, coupler) and
  drop it into a `Link` with no plumbing changes.
- **Swappable propagator backends.** A `Propagator` ABC sits between every
  `Fiber.propagate()` call and the actual numerics. NumPy linear and
  TensorFlow (SSFM + Manakov) ship today; JAX is the next backend.
- **Numerical correctness, pinned.** Soliton shape preservation, energy
  conservation, AWGN BER vs. theory, DBP round-trip — all in
  `tests/numerical/`, all gated by tolerances cited from the literature.
- **Reproducibility built in.** A single `seed` on `Simulation` is split into
  independent sub-streams (transmitter, link amplifiers) via
  `numpy.random.SeedSequence`. Two runs with the same seed produce identical
  results, including ASE noise samples.
- **Pre-v1 modules preserved.** Anything that lived under `hpcom.channel`,
  `hpcom.signal`, etc. is reachable for one release through `hpcom._legacy`
  (with a `DeprecationWarning`). New code should not touch them.

## Development

```bash
git clone https://github.com/esf0/hpcom.git
cd hpcom
pip install -e ".[dev,tf]"

pytest                         # unit + numerical tests
pytest -m slow                 # add the integration scenario
ruff check . && mypy src/hpcom
sphinx-build -b html docs docs/_build/html
```

Design docs are committed under [`docs/spec/`](docs/spec/): start with
`hpcom-v1.md` (the spec), then `hpcom-v1-plan.md` and `hpcom-v1-tasks.md`.

## Cite

If you use hpcom in research, please cite the Zenodo DOI in the badge above.

## Licence

GNU GPL v3.0 — see [`LICENSE`](LICENSE). The vendored `ssfm-gpu` numerics are
MIT-licensed; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
