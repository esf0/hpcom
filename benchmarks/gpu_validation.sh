#!/usr/bin/env bash
# GPU validation of the JAX backend: conformance suite + timings.
# Queue on a GPU node:  nodes run <node> hpcom-gpu-check -p hpcom -- bash benchmarks/gpu_validation.sh
set -euo pipefail
uv sync --extra dev --extra jax-cuda
uv run python -c "import jax; print('JAX', jax.__version__, 'devices:', jax.devices())"
uv run pytest -q -p no:cacheprovider --rungpu tests/backends -k "jax"
uv run python benchmarks/backends.py
