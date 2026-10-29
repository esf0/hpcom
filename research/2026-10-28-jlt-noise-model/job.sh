#!/usr/bin/env bash
# Node job (GPU):  nodes run bugatti jlt-noise -p hpcom --cpus 6 --ram 16 --vram 5 -- bash research/2026-10-28-jlt-noise-model/job.sh
# The paired symbol streams (~1.2 GB) stay on the node in ~/data/jlt-noise-model.
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=6
uv sync -q --extra dev --extra jax-cuda
uv run python -c "import jax; print('JAX', jax.__version__, jax.devices())"
here=research/2026-10-28-jlt-noise-model
data="$HOME/data/jlt-noise-model"
uv run python "$here/run.py" --out "$data" --backend jax
uv run python "$here/analyse.py" "$data" --out "$here/results"
