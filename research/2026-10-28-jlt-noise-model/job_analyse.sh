#!/usr/bin/env bash
# Analysis only, on paired runs already on the node:
#   nodes run bugatti jlt-analyse -p hpcom --cpus 4 --ram 10 --vram 2 -- \
#       bash research/2026-10-28-jlt-noise-model/job_analyse.sh jlt-noise-model-10x100 results-10x100
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=4
here=research/2026-10-28-jlt-noise-model
uv run python "$here/analyse.py" "$HOME/data/$1" --out "$here/$2"
