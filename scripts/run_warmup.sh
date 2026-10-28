#!/bin/bash
# Warm-start protocol variant (research rows, not the frozen main board): the baseline may calibrate on the drive's own
# CAN / IMU history before t0 (never GPS). W = 300 s (stem suffix _wu5) and all past (_wuall; the learned provider caps
# its recurrent warm-up at the last 3,600 s, dead reckoning consumes the whole history in one vectorised pass).
set -u
cd "$(dirname "$0")/.."
PY="nice -n 10 ionice -c2 -n7 ${PYTHON:-python3} -B"
M=manifests/episode_manifest_v1.parquet
WORKERS=${WORKERS:-3}
FOLDS=${FOLDS:-"val test few_shot_heldout"}
WINS=${WINS:-"w05"}
LOG=${CANTRAJ_LOGS:-logs}/can2gps_warmup.log
CK=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/models/learn/gru/seed0
COMMON="--run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins $WINS --workers $WORKERS --resume --version v1"
for W in 300 inf; do
  if [ "$W" = 300 ]; then sfx=_wu5; else sfx=_wuall; fi
  echo "=== warmup W=$W start $(date)" | tee -a $LOG
  $PY scripts/run_baseline.py --baseline dr_steer_offonline --group G3 --protocols O-N-Hp O-N-Hm $COMMON --warmup-s $W --stem-suffix $sfx 2>&1 | grep --line-buffered -v Warning | tee -a $LOG
  $PY scripts/run_baseline.py --baseline dr_imu_yaw_rls --group G8 --drives-file manifests/drives_G8.csv --protocols O-N-Hp O-N-Hm $COMMON --warmup-s $W --stem-suffix $sfx 2>&1 | grep --line-buffered -v Warning | tee -a $LOG
  $PY scripts/run_baseline.py --baseline learned --tag learned_gru_s0 --ckpt $CK --group G3 --protocols O-N-Hp O-N-Hm $COMMON --warmup-s $W --stem-suffix $sfx 2>&1 | grep --line-buffered -v Warning | tee -a $LOG
  echo "=== warmup W=$W done $(date)" | tee -a $LOG
done
