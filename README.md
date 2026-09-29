# hpcom — High-Performance Optical Communication

[![PyPI](https://img.shields.io/pypi/v/hpcom.svg)](https://pypi.python.org/pypi/hpcom)
[![Python](https://img.shields.io/pypi/pyversions/hpcom.svg)](https://pypi.python.org/pypi/hpcom)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hpcom.readthedocs.io)
[![DOI](https://zenodo.org/badge/564802755.svg)](https://zenodo.org/badge/latestdoi/564802755)
[![Licence: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

Simulation of coherent optical communication links for researchers who need
numbers they can trust and ML practitioners who need large, reproducible
datasets.

- **Transceivers**: RRC-shaped QAM (single carrier and WDM), CP-OFDM, pilots,
  geometric (APSK, arbitrary points) and probabilistic shaping (Maxwell–Boltzmann,
  constant-composition distribution matching), single or dual polarisation.
- **Channel**: scalar NLSE and Manakov solvers with loss, β₂, β₃ and Kerr
  nonlinearity — symmetric split-step (uniform or logarithmic steps, Kerr
  position, effective Kerr length), fourth-order Yoshida/Suzuki compositions,
  RK4IP and adaptive local-error control; the generalised NLSE (dispersion of
  any order, loss spectrum, Raman response, self-steepening); polarisation-mode
  dispersion (coarse-step Manakov-PMD) with a 2×2 MIMO receiver equaliser;
  weakly coupled multicore fibers; EDFAs with ASE; noise loading by SNR/OSNR;
  digital back-propagation at realistic sampling rates.
- **Backends**: NumPy (reference, always available), JAX (recommended for
  GPU) and TensorFlow (GPU, kept for systems where it suits better); a
  registry and conformance test suites for adding more.
- **Differentiable mode** (`hpcom.autodiff`, JAX): gradients of fibers,
  links and DBP in every physical parameter and the input field —
  parameter fitting, learned DBP, end-to-end training — with checkpointed
  or constant-memory (reversible) adjoints.
- **Metrics**: data-aided BER, SER, EVM, SNR, Q-factor, MI and GMI, per
  polarisation and WDM channel.
- **Trustworthy**: every convention is written down as formulas in the
  [physics page](docs/physics.rst) and pinned by tests against theory — BER
  within 5σ of the exact QAM formula, ASE-limited SNR within 0.1 dB of the
  closed form, split-step solver second-order convergent and exactly
  invertible, fields identical to the pre-v1 transmitter.
- **Reproducible**: one seed controls data and noise; every result carries
  its full configuration as JSON and can be replayed bit for bit.

## Install

```bash
pip install hpcom              # NumPy backend
pip install "hpcom[jax-cuda]"  # + JAX backend on NVIDIA GPUs (CPU: "hpcom[jax]")
pip install "hpcom[tf-gpu]"    # + TensorFlow backend with CUDA (CPU: "hpcom[tf]")
```

Python ≥ 3.10.

## Example

Dual-polarisation 16-QAM at 32 GBd over 10 × 80 km of standard fibre:

```python
from hpcom import EDFA, Fiber, Link, Simulation, SingleCarrierTransmitter

fiber = Fiber.ssmf(length_km=80.0, polarization="dual")        # backend="jax" for GPU
link = Link.from_spans(10, fiber, EDFA(gain_db=fiber.loss_db, noise_figure_db=5.0))

tx = SingleCarrierTransmitter(modulation_order=16, symbol_rate_hz=32e9, n_symbols=4096,
                              polarization="dual", launch_power_per_channel_dbm=0.0)
rx = tx.matched_receiver(cd_compensation_s2=link.total_dispersion_s2)

result = Simulation(transmitter=tx, link=link, receiver=rx, seed=0).run()
print(result)
# Result(ber=0.000e+00, snr=21.19 dB, evm=8.722e-02, gmi=4.000 b/sym, n_bits=32768, runtime=...)
```

`tx.matched_receiver()` builds the receiver with the transmitter's modulation,
pulse and polarisation; its default equaliser removes the mean nonlinear phase
rotation. `result.save("run.npz")` / `Result.load(...)` store a run with its
full configuration.

More in [`examples/`](examples/): BER vs SNR against theory, a launch-power
sweep, Manakov propagation, dataset generation, custom waveforms, WDM, OFDM, learned DBP,
probabilistic shaping and digital back-propagation.

## Conventions at a glance

| Quantity | Convention |
|---|---|
| Launch power | per WDM channel, summed over polarisations (`launch_power_per_channel_dbm`, or `launch_power_total_dbm`) |
| EDFA ASE | per-polarisation PSD n_sp·hν·(G−1), n_sp = (G·F−1)/(2(G−1)) |
| Equation | Agrawal's NLSE / Manakov (8/9 γ), optionally generalised (β₄…, α(λ), Raman, shock); NumPy FFT convention, +f = longer wavelength |
| Pulse | exact RRC by default; truncated FIR (circular or linear) on request |
| SNR | Es/N0 = P_pol/(N0·R_s) at the matched-filter output |

Details and derivations: [docs/physics.rst](docs/physics.rst). Moving from
pre-v1 hpcom: [docs/migration.rst](docs/migration.rst).

## Development

```bash
git clone https://github.com/esf0/hpcom.git && cd hpcom
pip install -e ".[dev,tf,legacy]"

pytest                                   # ~40 s, all suites
pytest -m "not slow"                     # fast subset
ruff check . && ruff format --check . && mypy src
sphinx-build -W -b html docs docs/_build/html
```

Tests are organised as `tests/unit` (behaviour), `tests/numerical` (physics
against theory), `tests/backends` (conformance suites every propagator and
kernel set must pass), `tests/integration` (end-to-end, examples) and
`tests/legacy` (pre-v1 modules).

## Cite

If you use hpcom in research, please cite the Zenodo DOI in the badge above.

## Licence

GNU GPL v3.0 — see [LICENSE](LICENSE).
