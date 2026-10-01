#!/usr/bin/env bash
# Node job (CPU):  nodes run <node> gn-egn -p hpcom --cpus 4 --ram 8 --gpu none -- bash research/2026-10-01-gn-egn/job.sh
set -euo pipefail
export OMP_NUM_THREADS=4
uv sync -q --extra dev
uv run python research/2026-10-01-gn-egn/run.py --out research/2026-10-01-gn-egn/results
