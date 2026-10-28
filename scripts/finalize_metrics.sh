#!/bin/bash
# Rescore every prediction group with the recovery metrics, complete the seed-0 learned map replays on the episodes that
# had no map at the time (per-episode tile planning came later), rescore those groups, rebuild the tables.
set -u
cd "$(dirname "$0")/.."
PY="nice -n 10 ionice -c2 -n7 ${PYTHON:-python3} -B"
LOG=${CANTRAJ_LOGS:-logs}/can2gps_finalize_metrics.log
M=manifests/episode_manifest_v1.parquet
S=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/scores/full_v1
echo "=== rescoring start $(date)" | tee -a $LOG
$PY scripts/score.py --run full_v1 --manifest $M --eligible-only --workers ${WORKERS:-5} 2>&1 | grep --line-buffered -vE "Warning|cKDTree" | tee -a $LOG
echo "=== seed-0 map fill start $(date)" | tee -a $LOG
MODE=eval-val-map KINDS="resid_nogru resid gru" SEEDS="0" EVAL_WORKERS=3 bash scripts/run_learned.sh 2>&1 | grep --line-buffered -vE "Warning|cKDTree|exp\(" | tee -a $LOG
MODE=eval-test-map KINDS="resid_nogru resid gru" SEEDS="0" EVAL_WORKERS=3 bash scripts/run_learned.sh 2>&1 | grep --line-buffered -vE "Warning|cKDTree|exp\(" | tee -a $LOG
echo "=== seed-0 rescoring start $(date)" | tee -a $LOG
rm -f $S/_parts/O-M-Hp__map_pf_learned_*_s0__G3__shard*.parquet
$PY scripts/score.py --run full_v1 --manifest $M --eligible-only --workers ${WORKERS:-5} --resume --methods map_pf_learned_gru_s0__G3 map_pf_learned_resid_s0__G3 map_pf_learned_resid_nogru_s0__G3 2>&1 | grep --line-buffered -vE "Warning|cKDTree" | tee -a $LOG
echo "=== tables start $(date)" | tee -a $LOG
$PY scripts/make_tables.py --run full_v1 --n-boot 1000 2>&1 | grep --line-buffered -vE "Warning|cKDTree" | tail -n 2 | tee -a $LOG
$PY scripts/make_paper_tables.py 2>&1 | tee -a $LOG
$PY scripts/failure_analysis.py 2>&1 | grep --line-buffered -vE "Warning|cKDTree" | tail -n 2 | tee -a $LOG
echo "=== finalize_metrics done $(date)" | tee -a $LOG
