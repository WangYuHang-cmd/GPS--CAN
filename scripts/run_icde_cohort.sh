#!/usr/bin/env bash
# ICDE same-cohort horizon study (manifests/cohort_v11_episodes.csv: one 15-minute start per drive, 426 drives).
# Offline methods are rerun on the 60 s and 300 s prefixes with only the prefix's events and the prefix's visible map
# (runs icde_cohort_p060 / icde_cohort_p300); their 900 s results are the regular w15 runs. Online methods are not
# rerun: their w15 predictions are causal, so the prefix is scored from the same files. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
WAIT_PID=${1:-0}                       # at most two run chains at a time: wait for this chain to exit first
while [ "$WAIT_PID" -gt 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
PY=${PYTHON:-python3}
W=${WORKERS:-3}
BASE="--version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout --wins w15 --episodes-file manifests/cohort_v11_episodes.csv --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15"
run() {
  echo "$(date '+%F %T') START $*"
  nice -n 10 ionice -c3 "$PY" scripts/run_baseline.py "$@"
  rc=$?
  echo "$(date '+%F %T') END rc=$rc $*"
}
for P in 60 300; do
  R=$(printf "icde_cohort_p%03d" "$P")
  run --baselines map_beam_steer --group G3 --protocols F-M-Hp F-M-Hm --workers "$W" --run "$R" --prefix-s "$P" $BASE
  run --baselines map_beam_none --group G1 --protocols F-M-Hp F-M-Hm --workers "$W" --run "$R" --prefix-s "$P" $BASE
  run --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp --workers "$W" --run "$R" --prefix-s "$P" $BASE
  run --baselines elastic_pathing --group G1 --protocols F-M-Hm --workers "$W" --run "$R" --prefix-s "$P" $BASE
  run --baselines opd2 --group G3 --protocols F-M-Hp --workers "$W" --run "$R" --prefix-s "$P" $BASE
  run --baselines map_pf_steer_lineage --group G3 --protocols F-M-Hp --workers "$W" --run "$R" --prefix-s "$P" $BASE
done
echo "$(date '+%F %T') CHAIN DONE"
