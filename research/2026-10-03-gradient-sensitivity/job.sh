#!/usr/bin/env bash
# Node job:  nodes run <node> grad-sensitivity -p hpcom --cpus 2 --ram 6 --vram 1 -- bash research/2026-10-03-gradient-sensitivity/job.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
uv run python research/2026-10-03-gradient-sensitivity/run.py --out research/2026-10-03-gradient-sensitivity/results
