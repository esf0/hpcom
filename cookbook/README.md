# Cookbook

Short recipes for one task each. Each recipe gives the code, the output it
printed and notes on what to change. The test suite runs the code of each
recipe (`tests/integration/test_readme.py`). The times are for a laptop
CPU without a GPU.

| Recipe | Task | Needs |
|---|---|---|
| [01](01_optimum_launch_power.md) | Find the optimum launch power with the EGN model, check it with the split step | NumPy |
| [02](02_compare_receivers.md) | Compare CDC and DBP receivers on one received field (paired) | NumPy |
| [03](03_phase_noise.md) | Add laser phase noise, recover the phase with blind phase search | NumPy |
| [04](04_ml_dataset.md) | Make a dataset of sent and received symbols, replay a stored run | NumPy |
| [05](05_fit_fiber_parameters.md) | Fit γ and β₂ to a measured field with gradients through the solver | JAX |
| [06](06_learn_constellation.md) | Train a constellation end to end through 10 × 80 km | JAX |
| [07](07_benchmark_a_method.md) | Score a new method on a benchmark scenario | JAX |

Full scripts with more cases: [`examples/`](../examples/). Reproduced
papers: [`gallery/`](../gallery/).
