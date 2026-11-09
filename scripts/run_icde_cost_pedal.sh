#!/usr/bin/env bash
# ICDE cost study, pedal subset (RQ5). CAN-Trace needs a continuous accelerator pedal, which no drive of the 92-episode
# cost subset has, so it is timed on the fixed pedal-valid validation subset manifests/icde_cost_episodes_pedal.csv
# (scripts/icde_cost_pedal_subset.py: a rule, not a selection on results), together with the methods it is compared
# with on the same windows. Same protocol as run_icde_cost.sh: ONE worker, peak memory from /usr/bin/time -v,
# per-episode inference seconds from the runner's timing files (a failed search counts with the time it ran), run
# icde_cost_pedal; the subset is scored at the end for the accuracy-on-the-same-windows table.
# Run when the machine is otherwise quiet; do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
LOG=${CANTRAJ_LOGS:-${CANTRAJ_LOGS:-logs}}/icde_cost_pedal
mkdir -p "$LOG"
EPIS=manifests/icde_cost_episodes_pedal.csv
BASE="--run icde_cost_pedal --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds val --wins w05 w15 --episodes-file $EPIS --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --workers 1 --flush-min 0"
ONLY=${ONLY:-}                     # optional space-separated tags to run (default: every method below)
run() {
  tag=$1; shift
  if [ -n "$ONLY" ] && [[ " $ONLY " != *" $tag "* ]]; then return; fi
  echo "$(date '+%F %T') START $tag"
  /usr/bin/time -v "$PY" scripts/run_baseline.py "$@" $BASE > "$LOG/$tag.out" 2> "$LOG/$tag.time"
  rc=$?
  echo "$(date '+%F %T') END $tag rc=$rc"
}
run can_trace --baselines can_trace --group G2 --protocols F-M-Hm
run can_trace_area --baselines can_trace_area --group G2 --protocols F-M-Hm
run dr_steer --baselines dr_steer --group G3 --protocols O-N-Hp
run map_pf_steer --baselines map_pf_steer --group G3 --protocols O-M-Hp
run map_beam_steer --baselines map_beam_steer --group G3 --protocols F-M-Hp
run map_beam_none --baselines map_beam_none --group G1 --protocols F-M-Hm
run elastic_pathing_hp --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp
run elastic_pathing --baselines elastic_pathing --group G1 --protocols F-M-Hm
run dewri --baselines dewri --group G1 --protocols F-M-Hm
echo "$(date '+%F %T') START score"
"$PY" scripts/score.py --run icde_cost_pedal --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds val --episodes-file $EPIS --workers 1 > "$LOG/score.out" 2>&1
echo "$(date '+%F %T') END score rc=$?"
echo "$(date '+%F %T') CHAIN DONE"
