#!/usr/bin/env bash
# The gallery entries, on a GPU node:
#   nodes run <node> gallery -p hpcom --cpus 4 --ram 12 --vram 6 -- bash gallery/job.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
for entry in poggiolini2015 hager2021 jovanovic2023; do
    uv run python "gallery/$entry/run.py" --out "gallery/$entry/results"
done
