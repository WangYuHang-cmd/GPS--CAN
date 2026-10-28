#!/usr/bin/env bash
# Learned motion baselines: train (one process at a time, GPU) and evaluate through the online replay.
#   MODE=train     KINDS="resid_nogru resid gru" SEEDS="0 1 2"  scripts/run_learned.sh
#   MODE=eval-val  KINDS=... SEEDS=...  scripts/run_learned.sh      # val only (model selection fold), no map + map PF
#   MODE=eval-test KINDS=... SEEDS=...  scripts/run_learned.sh      # run ONCE for the frozen configuration
set -e
cd "$(dirname "$0")/.."
PY="nice -n 10 ionice -c2 -n7 ${PYTHON:-python3} -B"
export PYTHONPATH=$OPENPILOT_DIR
M=manifests/episode_manifest_v1.parquet
LOG=${CANTRAJ_LOGS:-logs}/can2gps_learned.log
MODELS=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/models/learn
MODE=${MODE:-train}; KINDS=${KINDS:-"resid_nogru resid gru"}; SEEDS=${SEEDS:-"0 1 2"}
WORKERS=${EVAL_WORKERS:-4}   # replay workers hold ~2 GB each (native tables); keep 4 while map runs are active
case $MODE in
  train) FOLDS="";;
  eval-val) FOLDS="val"; DO_NOMAP=1; DO_MAP=0;;            # no-map replay (O-N-Hp, O-N-Hm), all seeds
  eval-val-map) FOLDS="val"; DO_NOMAP=0; DO_MAP=1;;        # map PF with the learned omega (O-M-Hp); run for seed 0 only to save graph memory/time
  eval-test) FOLDS="test few_shot_heldout"; DO_NOMAP=1; DO_MAP=0;;
  eval-test-map) FOLDS="test few_shot_heldout"; DO_NOMAP=0; DO_MAP=1;;
  *) echo "unknown MODE $MODE"; exit 1;;
esac
{
  echo "=== $MODE start $(date) kinds: $KINDS seeds: $SEEDS"
  for kind in $KINDS; do for seed in $SEEDS; do
    CK=$MODELS/$kind/seed$seed
    if [ "$MODE" = train ]; then
      $PY scripts/train_motion.py --model $kind --seed $seed 2>&1 | grep --line-buffered -v Warning | grep -E "^ep|done|Traceback|Error"
    else
      [ "$DO_NOMAP" = 1 ] && $PY scripts/run_baseline.py --baseline learned --tag learned_${kind}_s${seed} --ckpt $CK --group G3 --protocols O-N-Hp O-N-Hm --run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins w05 w15 full --workers $WORKERS --resume --version v1 2>&1 | grep --line-buffered -v Warning
      [ "$DO_MAP" = 1 ] && $PY scripts/run_baseline.py --baseline map_pf_learned --tag map_pf_learned_${kind}_s${seed} --ckpt $CK --group G3 --protocols O-M-Hp --run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins w05 w15 --workers $WORKERS --resume --version v1 2>&1 | grep --line-buffered -v Warning
    fi
  done; done
  echo "=== $MODE done $(date)"
} >> $LOG 2>&1
