#!/usr/bin/env bash
# ICDE cost study (RQ5): every method on the same fixed validation subset (manifests/icde_cost_episodes_valhalla.csv:
# the episodes of manifests/icde_cost_episodes.csv whose disc lies inside the OPD-II Valhalla clusters, w05 + w15),
# ONE worker, peak memory from /usr/bin/time -v, per-episode inference seconds from the runner's timing files
# (run icde_cost). Budget points for the two search methods: PF with 500 particles, beam width 50.
# Run when the machine is otherwise quiet; do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-${PYTHON:-python3}3}
LOG=${CANTRAJ_LOGS:-${CANTRAJ_LOGS:-logs}}/icde_cost
mkdir -p "$LOG"
M=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/models/learn
BASE="--run icde_cost --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds val --wins w05 w15 --episodes-file manifests/icde_cost_episodes_valhalla.csv --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --workers 1 --flush-min 0"
ONLY=${ONLY:-}                     # optional space-separated tags to run (default: every method)
run() {
  tag=$1; shift
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $tag "* ]]; then return; fi
  echo "$(date '+%F %T') START $tag"
  /usr/bin/time -v "$PY" scripts/run_baseline.py "$@" $BASE > "$LOG/$tag.out" 2> "$LOG/$tag.time"
  rc=$?
  echo "$(date '+%F %T') END $tag rc=$rc"
}
run dr_steer --baselines dr_steer --group G3 --protocols O-N-Hp
run dr_none --baselines dr_none --group G1 --protocols O-N-Hp
run learned_gru --baseline learned --ckpt "$M/gru/seed0" --tag learned_gru_s0 --group G3 --protocols O-N-Hp
run map_pf_steer --baselines map_pf_steer --group G3 --protocols O-M-Hp
run map_pf_none --baselines map_pf_none --group G1 --protocols O-M-Hp
run map_pf_steer_n500 --baselines map_pf_steer --group G3 --protocols O-M-Hp --set N=500 --stem-suffix _n500
run map_beam_steer --baselines map_beam_steer --group G3 --protocols F-M-Hp
run map_beam_steer_b50 --baselines map_beam_steer --group G3 --protocols F-M-Hp --set B=50 --stem-suffix _b50
run map_beam_none --baselines map_beam_none --group G1 --protocols F-M-Hp
run macrotracking --baselines macrotracking --group G3 --protocols O-M-Hp
run opd2 --baselines opd2 --group G3 --protocols F-M-Hp
run elastic_pathing_hp --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp
run elastic_pathing --baselines elastic_pathing --group G1 --protocols F-M-Hm
run map_pf_learned --baseline map_pf_learned --ckpt "$M/gru/seed0" --tag map_pf_learned_gru_s0 --group G3 --protocols O-M-Hp
run dewri_hp --baselines dewri_hp --group G1 --protocols F-M-Hp
run dewri --baselines dewri --group G1 --protocols F-M-Hm
run spy_hp --baselines spy_hp --group G3 --protocols F-M-Hp
run spy --baselines spy --group G3 --protocols F-M-Hm
run spy_area --baselines spy_area --group G3 --protocols F-M-Hm
echo "$(date '+%F %T') CHAIN DONE"
