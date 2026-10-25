#!/bin/bash
# After the warm-start chain and the seed 1-2 learned map replays finish: score the new parts, rebuild every table,
# refresh the paper tables / figures data, re-export the release tree and re-stage the gated benchmark folder.
# Commit + push and the Hub upload stay explicit (see progress.md).
set -u
cd "$(dirname "$0")/.."
PY="nice -n 10 ionice -c2 -n7 ${PYTHON:-python3} -B"
LOG=${CANTRAJ_LOGS:-logs}/can2gps_finalize_round3.log
M=manifests/episode_manifest_v1.parquet
echo "=== finalize start $(date)" | tee -a $LOG
$PY scripts/score.py --run full_v1 --manifest $M --eligible-only --workers ${WORKERS:-4} --resume 2>&1 | grep --line-buffered -vE "Warning|cKDTree" | tee -a $LOG
$PY scripts/make_tables.py --run full_v1 --n-boot 1000 2>&1 | grep --line-buffered -vE "Warning|cKDTree" | tee -a $LOG
$PY scripts/make_paper_tables.py 2>&1 | tee -a $LOG
$PY scripts/export_release.py --out ${RELEASE_DIR:-../GPS--CAN} --clean 2>&1 | tail -n 3 | tee -a $LOG
$PY scripts/stage_benchmark_release.py 2>&1 | tail -n 2 | tee -a $LOG
echo "=== finalize done $(date)" | tee -a $LOG
