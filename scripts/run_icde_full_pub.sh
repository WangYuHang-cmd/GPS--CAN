#!/usr/bin/env bash
# ICDE whole drives for the published methods (run icde_v11, test + few-shot): OPD-II (3 s delayed and online read-out)
# and Elastic Pathing (heading known, adapted; heading unknown, native) with their per-episode search budget (0.2 s per
# second of drive, 30 to 300 s; 70 branch calls per second, 20k to 200k): a search that ends without a complete path is
# a failure and counted. Waits for >= GATE_MB available before each step; scores at the end. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-${PYTHON:-python3}3}
GATE_MB=${GATE_MB:-12000}
COMMON="--run icde_v11 --folds test few_shot_heldout --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15 --wins full"
gate() {
  while [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt "$GATE_MB" ]; do sleep 60; done
}
run() {
  gate
  echo "$(date '+%F %T') START $*"
  nice -n 10 ionice -c3 "$PY" scripts/run_baseline.py "$@"
  rc=$?
  echo "$(date '+%F %T') END rc=$rc $*"
}
run --baselines opd2 --group G3 --protocols F-M-Hp --workers 3 $COMMON
run --baselines opd2_online --group G3 --protocols O-M-Hp --workers 3 $COMMON
run --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp --workers 3 $COMMON
run --baselines elastic_pathing --group G1 --protocols F-M-Hm --workers 3 $COMMON
gate
echo "$(date '+%F %T') START score"
nice -n 10 ionice -c3 "$PY" scripts/score.py --run icde_v11 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout --workers 2 --resume
rc=$?
echo "$(date '+%F %T') END rc=$rc score"
echo "$(date '+%F %T') CHAIN DONE"
