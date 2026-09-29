#!/usr/bin/env bash
# Run make_gnlse_golden.py in a throw-away environment (e.g. as a node job):
#   nodes run <node> gnlse-golden -p hpcom --cpus 1 --ram 2 --gpu none -- bash tests/numerical/_data/make_gnlse_golden.sh
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ENV_DIR="${TMPDIR:-/tmp}/hpcom-gnlse-env"
uv venv -q --python 3.12 "$ENV_DIR"
VIRTUAL_ENV="$ENV_DIR" uv pip install -q "setuptools<70" wheel numpy scipy
VIRTUAL_ENV="$ENV_DIR" uv pip install -q --no-build-isolation gnlse==2.0.0
"$ENV_DIR/bin/python" tests/numerical/_data/make_gnlse_golden.py
