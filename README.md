# hpcom — High-Performance Optical Communication

[![PyPI](https://img.shields.io/pypi/v/hpcom.svg)](https://pypi.python.org/pypi/hpcom)
[![Python](https://img.shields.io/pypi/pyversions/hpcom.svg)](https://pypi.python.org/pypi/hpcom)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://hpcom.readthedocs.io)
[![DOI](https://zenodo.org/badge/564802755.svg)](https://zenodo.org/badge/latestdoi/564802755)
[![Licence: GPLv3](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

hpcom simulates coherent optical fiber links, from the bits at the
transmitter to the information rate at the receiver.

## Goals

1. **Correct physics.** Each convention is a formula in
   [docs/physics.rst](docs/physics.rst). A test checks each formula against
   theory or an independent code.
2. **Reproducible results.** One seed sets the data and the noise. Each
   result holds its full configuration as JSON and replays bit for bit.
3. **A differentiable channel.** With JAX, gradients go through the full
   chain: transmitter, split-step fiber, amplifiers, DBP, receiver and
   information rate.
4. **Fair comparisons.** Fixed benchmark scenarios score each method on
   the same data and noise (paired), on a finer simulation than the
   training one.

## What hpcom does

| Area | Content |
|---|---|
| Transmitters | QAM on one carrier or WDM (exact or truncated RRC), CP-OFDM, pilots. Geometric shaping (APSK, free points), probabilistic shaping (Maxwell–Boltzmann, CCDM). 1 or 2 polarisations, multicore (SDM) |
| Fiber | NLSE and Manakov equation: loss, β₂, β₃, Kerr. Generalised NLSE: dispersion of any order, loss spectrum, Raman, self-steepening. PMD. Weakly coupled multicore fiber |
| Solvers | Symmetric split step (uniform or logarithmic steps, Kerr position, effective Kerr length, filtered Kerr step). 4th-order Yoshida and Suzuki compositions. RK4IP. Adaptive local-error step |
| Amplifiers, noise | EDFA with ASE. Noise loading by SNR or OSNR |
| Transceiver impairments | Laser and LO phase noise, LO frequency offset, IQ imbalance, DAC/ADC quantisation, Mach–Zehnder modulator |
| Receiver DSP | CD compensation. DBP for one channel, a wide band or the full field, at 1, 2 or 4 samples/symbol. 2×2 MIMO (least squares), CMA, LMS. Blind phase search |
| NLI models | GN model for any link and channel plan. Closed-form EGN. First-order perturbation of the actual signal |
| Learning | `hpcom.autodiff`: gradients in each fiber, amplifier and DBP parameter and in the field. `hpcom.learn`: end-to-end systems from encoder, pre-distorter, front end and decoder blocks. `hpcom.bench`: scenarios and a leaderboard |
| Fiber lasers | Gain fiber with saturable gain of finite bandwidth (Ginzburg–Landau), saturable absorber, filters, couplers, ring cavity |
| Metrics | BER, SER, EVM, SNR, Q factor, MI, GMI. Per polarisation and per WDM channel |
| Backends | NumPy (reference), JAX (GPU, recommended), TensorFlow (GPU). One conformance test suite for all |

## Checks against theory and other codes

| Check | Result |
|---|---|
| QAM BER over AWGN | within 5σ of the exact formula |
| ASE-limited SNR | within 0.1 dB of the closed form |
| Split-step schemes | converge at their stated order. Without saturable gain, each scheme inverts exactly |
| OptiCommPy 0.10 `ssfm` | equal to 1e-14 |
| gnlse-python 2.0, supercontinuum of Dudley et al. (2006) | 7e-5 to 5e-4 relative L2 error (the floor of the reference) |
| Pre-v1 hpcom, full WDM chain | 6e-6 RMS |
| GN model, Poggiolini et al. (2015) Fig. 1a | 26.67 dB at 1 span, 45.25 dB at 50 spans (paper: ≈26.8, ≈45.3) |
| NLI models against the split step, 1–30 spans | first order within 0.05 dB. Closed-form EGN within 0.23 dB from 5 spans |
| Blind phase search | within 0.5 dB of the SNR without phase noise |
| JAX on GPU against NumPy | equal to 3e-13. 27–35 times faster at 2¹⁸ samples (RTX 3070 Laptop) |

The test suite has more than 2300 tests. Each check above names its test
in [docs/physics.rst](docs/physics.rst) or [docs/numerics.rst](docs/numerics.rst).

## Results

**Published results, reproduced** ([gallery/](gallery/)):

| Paper | Result |
|---|---|
| Poggiolini et al., arXiv:1503.04132 (2015): NLI coefficient against spans | GN and EGN within 0.12 dB, simulation within 0.4 dB |
| Jovanovic et al., JLT 41(12) (2023): constellation learned through the SSFM, 5 × 32 GBd | within 0.03 bit/4D at and below the optimum, gain +0.23 bit/4D (paper: +0.21) |
| Häger & Pfister, IEEE JSAC 39(1) (2021): linear equalisation, DBP at 1 step/span | within 0.15 dB. The 77-tap learned DBP is not reproduced |
| Kelly, Electron. Lett. 28(8) (1992): Kelly sidebands of a soliton fiber laser | reproduced |

**Benchmark** ([leaderboard](benchmarks/leaderboard/results/leaderboard.md)),
best rate over the power sweep in bit/symbol/polarisation, 10 × 80 km SSMF,
2000 training steps per power:

| Scenario | QAM + CDC | learned points + CDC | best learned | best fixed DBP |
|---|---|---|---|---|
| `sc-dp-10x80`, 1 × 32 GBd, 1024 points | 6.566 | 6.872 | 8.344 (points + learned DBP, 1 step/span) | 9.070 (4 steps/span) |
| `wdm5-dp-10x80`, 5 × 32 GBd, 256 points | 6.369 | 6.585 | 6.971 (points + wideband learned DBP, 2 steps/span) | 7.355 (full-field, ideal) |
| `jovanovic2023`, 5 × 32 GBd, 64 points | 5.357 | 5.472 | — | — |

## Install

Version 0.2 (this branch) is not on PyPI yet. `pip install hpcom` gives
the pre-v1 version 0.1.13. Install from GitHub:

```bash
pip install "hpcom @ git+https://github.com/esf0/hpcom"              # NumPy backend
pip install "hpcom[jax-cuda] @ git+https://github.com/esf0/hpcom"    # + JAX on NVIDIA GPUs (CPU: [jax])
pip install "hpcom[tf-gpu] @ git+https://github.com/esf0/hpcom"      # + TensorFlow with CUDA (CPU: [tf])
```

Python 3.10 or later.

## Quick start

Dual-polarisation 16-QAM at 32 GBd over 10 × 80 km of standard fiber:

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

`tx.matched_receiver()` makes a receiver with the modulation, pulse and
polarisation of the transmitter. `result.save("run.npz")` stores the run
with its full configuration. `Result.load(...)` reads it.

## Cookbook and examples

[cookbook/](cookbook/) has one recipe for each task:

1. [Find the optimum launch power: EGN model and split step](cookbook/01_optimum_launch_power.md)
2. [Compare CDC and DBP receivers on one received field](cookbook/02_compare_receivers.md)
3. [Add laser phase noise, recover the phase with blind phase search](cookbook/03_phase_noise.md)
4. [Make a dataset of sent and received symbols, replay a run](cookbook/04_ml_dataset.md)
5. [Fit γ and β₂ to a measured field with gradients](cookbook/05_fit_fiber_parameters.md)
6. [Train a constellation end to end through the fiber](cookbook/06_learn_constellation.md)
7. [Score a new method on a benchmark scenario](cookbook/07_benchmark_a_method.md)

[examples/](examples/) has 12 full scripts: BER against theory, launch-power
sweep, Manakov propagation, dataset files, custom waveform, WDM, OFDM,
probabilistic shaping, DBP, learned DBP, end-to-end learning and a
mode-locked fiber laser.

## Documentation

| Page | Content |
|---|---|
| [concepts](docs/concepts.rst) | transmitters, link, receivers, simulation and result |
| [physics](docs/physics.rst) | each equation and convention, with its checks |
| [numerics](docs/numerics.rst) | solvers, step placement, accuracy |
| [impairments](docs/impairments.rst) | transceiver impairments and receiver DSP |
| [autodiff](docs/autodiff.rst) | what is differentiable, gradient modes |
| [learn](docs/learn.rst) | end-to-end learning, benchmark, leaderboard |
| [backends](docs/backends.rst) | NumPy, JAX, TensorFlow, how to add a backend |
| [migration](docs/migration.rst) | from pre-v1 hpcom |

## Conventions

| Quantity | Convention |
|---|---|
| Launch power | per WDM channel, sum of the two polarisations (`launch_power_per_channel_dbm` or `launch_power_total_dbm`) |
| EDFA ASE | PSD per polarisation n_sp·hν·(G−1), n_sp = (G·F−1)/(2(G−1)) |
| Equation | NLSE and Manakov (8/9 γ) of Agrawal, optionally generalised (β₄…, α(λ), Raman, shock). NumPy FFT convention, +f = longer wavelength |
| Pulse | exact RRC by default. Truncated FIR (circular or linear) on request |
| SNR | Es/N0 = P_pol/(N0·R_s) at the matched-filter output |

## Not done yet

- Release of version 0.2 on PyPI.
- Gallery entry Uhlemann et al. (2020). It needs distributed Raman amplification.
- Raman scattering with two polarisations. Raman is for one polarisation only now.
- GN model with inter-channel stimulated Raman scattering (ISRS).
- Learned DBP with a limited number of filter taps.
- Rust kernels. The backend interfaces for them are ready.
- JAX 0.12: one gradient test fails
  (`test_gradients_equal_finite_differences_for_every_parameter[saturated-gain-dual-log]`).
  The gradient of the fiber loss differs from the finite difference by
  8e-6 relative. With JAX 0.11.2 all tests pass.

## Development

```bash
git clone https://github.com/esf0/hpcom.git && cd hpcom
pip install -e ".[dev,jax,tf,legacy]"

pytest                                   # all suites
pytest -m "not slow"                     # fast subset
ruff check . && ruff format --check . && mypy src
sphinx-build -W -b html docs docs/_build/html
```

Test folders: `tests/unit` (behaviour), `tests/numerical` (physics against
theory), `tests/backends` (the conformance suites for each propagator and
kernel set), `tests/integration` (end to end, examples, README and
cookbook code) and `tests/legacy` (pre-v1 modules).

## Cite

If you use hpcom in research, cite the Zenodo DOI in the badge above.

## Licence

GNU GPL v3.0. See [LICENSE](LICENSE).
