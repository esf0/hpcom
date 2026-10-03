#!/usr/bin/env bash
# The benchmark leaderboard (hpcom.bench) for the standard methods, on a GPU node:
#   nodes run <node> leaderboard -p hpcom --cpus 4 --ram 12 --vram 6 -- bash benchmarks/leaderboard/job.sh
set -euo pipefail
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-jax-gpu}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
uv sync -q --extra dev --extra jax-cuda
out=benchmarks/leaderboard/results
mkdir -p "$out"
run() { uv run python -m hpcom.bench --scenario "$1" --out "$out/$1.json" --method "${@:2}"; }
run sc-dp-10x80 qam+cdc gs+cdc qam+dbp1 qam+dbp4 gs+ldbp1
run jovanovic2023 qam+cdc gs+cdc
run wdm5-dp-10x80 qam+cdc gs+cdc qam+dbp4 gs+ldbp1 qam+wdbp2 gs+wldbp2 qam+ffdbp
uv run python -c "
import json, pathlib
from hpcom import bench
results = [bench.MethodResult.model_validate(r) for f in sorted(pathlib.Path('$out').glob('*.json')) for r in json.loads(f.read_text())]
pathlib.Path('$out/leaderboard.md').write_text(bench.leaderboard(results))
print(bench.leaderboard(results))
"
