#!/usr/bin/env bash
# Node job:  nodes run <node> dp-wdm-e2e -p hpcom --cpus 4 --ram 12 --vram 6 -- bash research/2026-10-03-dp-wdm-e2e/job.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
uv run python -c "import jax; print('JAX', jax.__version__, jax.devices())"
uv run python research/2026-10-03-dp-wdm-e2e/run.py --cosine --ema 0.99 --out research/2026-10-03-dp-wdm-e2e/results
