#!/usr/bin/env bash
# GPU validation of the JAX backend and hpcom.autodiff: conformance suite, autodiff tests, timings.
# Run on any GPU machine:      bash benchmarks/gpu_validation.sh
# or on a compute node:        nodes run <node> hpcom-gpu-check -p hpcom -- bash benchmarks/gpu_validation.sh
# It builds its own environment (.venv-jax-gpu) and never touches .venv, whose
# TensorFlow CUDA libraries could clash with JAX's.
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # do not reserve 75 % of the GPU up front
uv sync --extra dev --extra jax-cuda
uv run python -c "import jax; print('JAX', jax.__version__, 'devices:', jax.devices())"
uv run pytest -q -p no:cacheprovider --rungpu tests/backends -k "jax"
uv run pytest -q -p no:cacheprovider tests/numerical/test_autodiff.py tests/unit/test_autodiff_api.py
uv run python benchmarks/backends.py
