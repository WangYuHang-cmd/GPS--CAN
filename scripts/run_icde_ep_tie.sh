#!/usr/bin/env bash
# Elastic Pathing tie-order sensitivity (validation only, run icde_dev): the paper does not say how paths with equal
# error are ordered; ours takes the furthest path first, then successors in edge-id order. Three shuffled successor
# orders (tie_seed 1, 2, 3) on the 60 selection drives (w05, heading-known variant, the selected f = 0.15,
# v_zero = 1.0), compared with the selection run elastic_pathing_hp_f015v10. Then scored on val. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
COMMON="--group G1 --run icde_dev --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds val --wins w05 --drives-file manifests/icde_val_select_drives.csv --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --workers 2 --flush-min 0 --resume"
for s in 1 2 3; do
  echo "$(date '+%F %T') START tie_seed=$s"
  nice -n 10 ionice -c3 "$PY" scripts/run_baseline.py --baselines elastic_pathing_hp --protocols F-M-Hp $COMMON --set "tie_seed=$s" --stem-suffix "_tie$s"
  rc=$?
  echo "$(date '+%F %T') END rc=$rc tie_seed=$s"
done
nice -n 10 ionice -c3 "$PY" scripts/score.py --run icde_dev --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds val --workers 2 --resume --methods elastic_pathing_hp_tie1__G1 elastic_pathing_hp_tie2__G1 elastic_pathing_hp_tie3__G1 elastic_pathing_hp_f015v10__G1
echo "$(date '+%F %T') EP TIE DONE"
