#!/usr/bin/env bash
# Fig. 3 top-left of the paper: 10 × 100 km, −1 … 3 dBm, otherwise as job.sh.
# Node job (GPU):  nodes run bugatti jlt-noise-10x100 -p hpcom --cpus 6 --ram 16 --vram 5 -- bash research/2026-10-28-jlt-noise-model/job_10x100.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=6
uv sync -q --extra dev --extra jax-cuda
here=research/2026-10-28-jlt-noise-model
data="$HOME/data/jlt-noise-model-10x100"
uv run python "$here/run.py" --out "$data" --backend jax --spans 10 --span-km 100 --steps-per-span 100 \
    --powers -1 0 1 2 3
uv run python "$here/analyse.py" "$data" --out "$here/results-10x100"
