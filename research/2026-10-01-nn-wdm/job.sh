#!/usr/bin/env bash
# Node job: GPU environment as benchmarks/gpu_validation.sh, then the sweep.
#   nodes run <node> nn-wdm -p hpcom --cpus 4 --ram 8 --vram 6 -- bash research/2026-10-01-nn-wdm/job.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
uv run python -c "import jax; print('JAX', jax.__version__, jax.devices())"
uv run python research/2026-10-01-nn-wdm/run.py --order 256 --out research/2026-10-01-nn-wdm/results-256
