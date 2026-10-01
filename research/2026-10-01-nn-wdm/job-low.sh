#!/usr/bin/env bash
# Node job: the two lowest powers, appended to results-256 (the CDC optimum lies below -4 dBm).
#   nodes run <node> nn-wdm-low -p hpcom --cpus 4 --ram 8 --vram 6 -- bash research/2026-10-01-nn-wdm/job-low.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
uv run python research/2026-10-01-nn-wdm/run.py --order 256 --powers -8 -6 --append \
    --out research/2026-10-01-nn-wdm/results-256
