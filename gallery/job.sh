#!/usr/bin/env bash
# Gallery entries on a node (default: all; or pass entry names):
#   GPU:  nodes run <node> gallery -p hpcom --cpus 4 --ram 12 --vram 2 -- bash gallery/job.sh hager2021 jovanovic2023
#   CPU:  nodes run <node> gallery-cpu -p hpcom --cpus 4 --ram 8 --gpu none -- bash gallery/job.sh poggiolini2015
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
entries=("$@")
[ ${#entries[@]} -eq 0 ] && entries=(poggiolini2015 hager2021 jovanovic2023)
for entry in "${entries[@]}"; do
    uv run python "gallery/$entry/run.py" --out "gallery/$entry/results"
done
