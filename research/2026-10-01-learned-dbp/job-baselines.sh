#!/usr/bin/env bash
# Node job, after job.sh: the fixed baselines at the learned methods' step counts, appended.
#   nodes run <node> learned-dbp-base -p hpcom --cpus 4 --ram 8 --vram 6 -- bash research/2026-10-01-learned-dbp/job-baselines.sh dbp1 wdbp2
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
uv run python research/2026-10-01-learned-dbp/run.py --cosine --ema 0.99 --methods "$@" \
    --append --out research/2026-10-01-learned-dbp/results
