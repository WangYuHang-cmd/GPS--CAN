#!/usr/bin/env bash
# Elastic Pathing: choose the two values the paper leaves open (side friction f, "speed ~ 0" threshold v_zero) on
# validation drives only (manifests/icde_val_select_drives.csv: 60 val drives of 36 drivers, w05, heading-known
# variant), run icde_dev; then score and pick by median ADE. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
COMMON="--group G1 --run icde_dev --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds val --wins w05 --drives-file manifests/icde_val_select_drives.csv --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --workers 3 --flush-min 0 --resume"
for f in 0.15 0.3 0.5 0.7; do
  for vz in 0.3 0.5 1.0; do
    tag="_f${f/./}v${vz/./}"
    echo "$(date '+%F %T') START f=$f v_zero=$vz"
    nice -n 10 ionice -c3 "$PY" scripts/run_baseline.py --baselines elastic_pathing_hp --protocols F-M-Hp $COMMON --set "f=$f,v_zero=$vz" --stem-suffix "$tag"
  done
done
echo "$(date '+%F %T') EP SELECT DONE"
