#!/usr/bin/env bash
# ICDE chain 2 (run icde_v11, test + few-shot): published methods and controls. Elastic Pathing (val-selected values),
# CAN-Trace on the accelerator-pedal subset, the PF lineage read-out, OPD-II (needs maps/valhalla_v2), the A3
# visibility check, then whole drives for the published methods. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
W=${WORKERS:-3}
BASE="--version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15"
COMMON="--run icde_v11 --folds test few_shot_heldout $BASE"
run() {
  echo "$(date '+%F %T') START $*"
  nice -n 10 ionice -c3 "$PY" scripts/run_baseline.py "$@"
  echo "$(date '+%F %T') END rc=$? $*"
}
run --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp --wins w05 w15 --workers "$W" $COMMON
run --baselines elastic_pathing --group G1 --protocols F-M-Hm --wins w05 w15 --workers "$W" $COMMON
run --baselines can_trace can_trace_area --group G2 --protocols F-M-Hm --wins w05 w15 --workers "$W" --drives-file manifests/icde_pedal_main_drives.csv $COMMON
run --baselines map_pf_steer_lineage --group G3 --protocols F-M-Hp --wins w05 w15 --workers "$W" $COMMON
run --baselines opd2 opd2_wrapfix opd2_calib --group G3 --protocols F-M-Hp --wins w05 w15 --workers "$W" $COMMON
run --baselines opd2_online --group G3 --protocols O-M-Hp --wins w05 w15 --workers "$W" $COMMON
# A3: the same 60 test drives with the whole per-drive union visible (v1 behaviour) and a second PF seed
run --baselines map_beam_steer --group G3 --protocols F-M-Hp --wins w05 w15 --workers "$W" --drives-file manifests/icde_a3_drives.csv --no-visibility --run icde_a3_union --folds test $BASE
run --baselines map_pf_steer --group G3 --protocols O-M-Hp --wins w05 w15 --workers "$W" --drives-file manifests/icde_a3_drives.csv --no-visibility --run icde_a3_union --folds test $BASE
run --baselines map_pf_steer_s1 --group G3 --protocols O-M-Hp --wins w05 w15 --workers "$W" --drives-file manifests/icde_a3_drives.csv --run icde_a3_union --folds test $BASE
run --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp --wins full --workers "$W" $COMMON
run --baselines opd2 --group G3 --protocols F-M-Hp --wins full --workers "$W" $COMMON
echo "$(date '+%F %T') CHAIN DONE"
