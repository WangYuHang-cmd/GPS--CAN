#!/usr/bin/env bash
# ICDE plan v0.9 item D: external check on the Controlled supplement (admission == "supplement"; never part of the
# CAN-only board or of any selection, see scripts/icde_supplement.py). Frozen settings, no parameter changed: the nine
# methods of the main five-minute comparison run on the supplement's 46 map-eligible five-minute windows
# (manifests/icde_supplement_episodes.csv, map eligibility manifests/episode_map_v11_supp.parquet) and are scored with
# the same scorer. Run id icde_supp (icde_v11 untouched). Only summary numbers go to the paper (no map, place or
# platform names). Usage: run_icde_supp.sh. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
GATE_MB=${GATE_MB:-12000}
WORKERS=${WORKERS:-3}
BASE="--run icde_supp --version v1.1 --manifest manifests/episode_manifest_v1.parquet --folds supplement --episodes-file manifests/icde_supplement_episodes.csv --wins w05 --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11_supp.parquet --resume --flush-min 15 --workers $WORKERS"
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
run --baselines dr_steer --group G3 --protocols O-N-Hp $BASE
run --baselines map_pf_steer macrotracking --group G3 --protocols O-M-Hp $BASE
run --baselines map_beam_steer --group G3 --protocols F-M-Hp $BASE
run --baselines opd2 --group G3 --protocols F-M-Hp $BASE
run --baselines map_pf_none --group G1 --protocols O-M-Hp $BASE
run --baselines map_beam_none elastic_pathing_hp dewri_hp --group G1 --protocols F-M-Hp $BASE
gate
echo "$(date '+%F %T') START score"
nice -n 10 ionice -c3 "$PY" scripts/score.py --run icde_supp --manifest manifests/episode_manifest_v1.parquet --folds supplement --episodes-file manifests/icde_supplement_episodes.csv --workers 2 --resume
rc=$?
echo "$(date '+%F %T') END rc=$rc score"
echo "$(date '+%F %T') CHAIN DONE"
