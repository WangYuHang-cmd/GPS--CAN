#!/usr/bin/env bash
# ICDE chain 3 (run icde_v11, test + few-shot): signal-combination study on the accelerator-pedal subset (w05, heading
# known; no map and the fixed turn-budget map decoder), then the three-seed learned map replays with the corrected
# motion-source timing on all main-fold w05 episodes. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-${PYTHON:-python3}3}
W=${WORKERS:-3}
M=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/models/learn
COMMON="--run icde_v11 --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15"
run() {
  echo "$(date '+%F %T') START $*"
  nice -n 10 ionice -c3 "$PY" scripts/run_baseline.py "$@"
  echo "$(date '+%F %T') END rc=$? $*"
}
declare -A GRP=( [S1]=G1 [S2c]=G2 [S2b]=G2 [S3]=G3 [S4c]=G7 [S4b]=G7 )
for g in S1 S2c S3 S4c S2b S4b; do
  for s in 0 1 2; do
    d="$M/sig/$g/seed$s"
    [ -f "$d/meta.json" ] || continue
    run --baseline sig_dr --ckpt "$d" --tag "sig_dr_${g}_s$s" --group "${GRP[$g]}" --protocols O-N-Hp --wins w05 --workers "$W" --drives-file manifests/icde_pedal_main_drives.csv $COMMON
    run --baseline map_pf_sig --ckpt "$d" --tag "map_pf_sig_${g}_s$s" --group "${GRP[$g]}" --protocols O-M-Hp --wins w05 --workers "$W" --drives-file manifests/icde_pedal_main_drives.csv $COMMON
  done
done
for kind in gru resid; do
  for s in 0 1 2; do
    run --baseline map_pf_learned --ckpt "$M/$kind/seed$s" --tag "map_pf_learned_${kind}_s$s" --group G3 --protocols O-M-Hp --wins w05 --workers "$W" $COMMON
  done
done
echo "$(date '+%F %T') CHAIN DONE"
