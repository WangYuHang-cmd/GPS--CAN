#!/usr/bin/env bash
# Map protocols over all eligible v1 episodes with map tiles (resumable; episodes without tiles are recorded as map_missing).
# Usage: scripts/run_full_maps.sh [folds...]   e.g. scripts/run_full_maps.sh val test few_shot_heldout
set -e
cd "$(dirname "$0")/.."
PY="nice -n 10 ionice -c2 -n7 ${PYTHON:-python3} -B"
WORKERS=${WORKERS:-6}
export PYTHONPATH=$OPENPILOT_DIR
M=manifests/episode_manifest_v1.parquet
FOLDS=${*:-val test few_shot_heldout train}
LOG=${CANTRAJ_LOGS:-logs}/can2gps_full_v1_maps.log
{
  echo "=== map passes start $(date) folds: $FOLDS"
  # online PF, G3 (nominal and calibrated steering) and G1 (speed only); w05 and w15 first, then full, then w01
  WINSETS=${WINSETS:-"w05 w15|full|w01"}       # window sets to run, separated by |
  IFS='|' read -ra WSETS <<< "$WINSETS"
  for W in "${WSETS[@]}"; do
    $PY scripts/run_baseline.py --baselines map_pf_steer map_pf_steer_calib --group G3 --protocols O-M-Hp O-M-Hm --run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins $W --workers $WORKERS --resume --version v1 2>&1 | grep --line-buffered -v Warning
    $PY scripts/run_baseline.py --baselines map_pf_none --group G1 --protocols O-M-Hp O-M-Hm --run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins $W --workers $WORKERS --resume --version v1 2>&1 | grep --line-buffered -v Warning
    # offline beam (F-M), whole-path Top-1 and candidates
    $PY scripts/run_baseline.py --baselines map_beam_steer map_beam_steer_calib --group G3 --protocols F-M-Hp F-M-Hm --run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins $W --workers $WORKERS --resume --version v1 2>&1 | grep --line-buffered -v Warning
    $PY scripts/run_baseline.py --baselines map_beam_none --group G1 --protocols F-M-Hp F-M-Hm --run full_v1 --manifest $M --eligible-only --folds $FOLDS --wins $W --workers $WORKERS --resume --version v1 2>&1 | grep --line-buffered -v Warning
    echo "--- windows $W done $(date)"
  done
  echo "=== map passes done $(date)"
} >> $LOG 2>&1
