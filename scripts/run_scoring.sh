#!/usr/bin/env bash
# Score a run and build the result tables / report. Usage: scripts/run_scoring.sh full_v1 [n_boot]
set -e
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3} -B"
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
RUN=${1:-full_v1}; NB=${2:-1000}
$PY scripts/score.py --run $RUN --manifest manifests/episode_manifest_v1.parquet --eligible-only --n-boot 200 2>&1 | grep -v Warning | tail -40
$PY scripts/make_tables.py --run $RUN --manifest manifests/episode_manifest_v1.parquet --n-boot $NB 2>&1 | grep -v Warning | tail -2
