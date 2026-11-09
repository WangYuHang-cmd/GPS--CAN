#!/usr/bin/env bash
# ICDE remaining runs as ONE sequential chain. Replaces chains 1 to 4 after the 17:08 memory squeeze of 2026-10-28
# (map beam workers reach 2.5 to 3.5 GB each; two chains plus the scorer left under 4 GB for the user). Before every
# step it waits until GATE_MB of memory is available; workers per step are sized by the method's memory use (beam 2,
# filters / searches 3, light trackers 4). Every step resumes from its flushed parts, so a stopped step is rerun as is.
# Order: published methods -> signal study -> speed-only map baselines -> controls (lineage, A3, curvature) -> learned
# map replays -> same-cohort prefixes -> whole drives, with scoring of icde_v11 after each block.
# Usage: run_icde_seq.sh <pid-to-wait-for>. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
WAIT_PID=${1:-0}
while [ "$WAIT_PID" -gt 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
PY=${PYTHON:-python3}
GATE_MB=${GATE_MB:-12000}
M=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/models/learn
BASE="--version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15"
COMMON="--run icde_v11 --folds test few_shot_heldout $BASE"
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
score() {
  gate
  echo "$(date '+%F %T') START score $*"
  nice -n 10 ionice -c3 "$PY" scripts/score.py --manifest manifests/episode_manifest_v1.parquet --eligible-only --workers 2 --resume "$@"
  rc=$?
  echo "$(date '+%F %T') END rc=$rc score $*"
}
# A1: published methods, five- and fifteen-minute windows
run --baselines macrotracking macrotracking_calib macrotracking_wrapfix --group G3 --protocols O-M-Hp --wins w05 w15 --workers 4 $COMMON
run --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp --wins w05 w15 --workers 3 $COMMON
run --baselines can_trace can_trace_area --group G2 --protocols F-M-Hm --wins w05 w15 --workers 3 --drives-file manifests/icde_pedal_main_drives.csv $COMMON
run --baselines opd2 opd2_wrapfix opd2_calib --group G3 --protocols F-M-Hp --wins w05 w15 --workers 3 $COMMON
run --baselines opd2_online --group G3 --protocols O-M-Hp --wins w05 w15 --workers 3 $COMMON
score --run icde_v11 --folds test few_shot_heldout
# B1: signal study on the accelerator-pedal subset (fixed learner, fixed decoder, 3 seeds)
declare -A GRP=( [S1]=G1 [S2c]=G2 [S2b]=G2 [S3]=G3 [S4c]=G7 [S4b]=G7 )
for g in S1 S2c S3 S4c S2b S4b; do
  for s in 0 1 2; do
    d="$M/sig/$g/seed$s"
    [ -f "$d/meta.json" ] || continue
    run --baseline sig_dr --ckpt "$d" --tag "sig_dr_${g}_s$s" --group "${GRP[$g]}" --protocols O-N-Hp --wins w05 --workers 3 --drives-file manifests/icde_pedal_main_drives.csv $COMMON
    run --baseline map_pf_sig --ckpt "$d" --tag "map_pf_sig_${g}_s$s" --group "${GRP[$g]}" --protocols O-M-Hp --wins w05 --workers 3 --drives-file manifests/icde_pedal_main_drives.csv $COMMON
  done
done
# A2: speed-only map baselines
run --baselines map_pf_none --group G1 --protocols O-M-Hp O-M-Hm --wins w05 w15 --workers 3 $COMMON
run --baselines map_beam_none --group G1 --protocols F-M-Hp F-M-Hm --wins w05 w15 --workers 2 $COMMON
score --run icde_v11 --folds test few_shot_heldout
# C: decode-output control, A3 visibility outcome check (60 test drives), class-F reference
run --baselines map_pf_steer_lineage --group G3 --protocols F-M-Hp --wins w05 w15 --workers 2 $COMMON
run --baselines map_beam_steer --group G3 --protocols F-M-Hp --wins w05 w15 --workers 2 --drives-file manifests/icde_a3_drives.csv --no-visibility --run icde_a3_union --folds test $BASE
run --baselines map_pf_steer --group G3 --protocols O-M-Hp --wins w05 w15 --workers 3 --drives-file manifests/icde_a3_drives.csv --no-visibility --run icde_a3_union --folds test $BASE
run --baselines map_pf_steer_s1 --group G3 --protocols O-M-Hp --wins w05 w15 --workers 3 --drives-file manifests/icde_a3_drives.csv --run icde_a3_union --folds test $BASE
run --baselines map_pf_curvature --group G5 --protocols O-M-Hp O-M-Hm --wins w05 w15 --workers 3 $COMMON
# B2: learned motion sources inside the map filter after the timing fix (3 seeds each)
for kind in gru resid; do
  for s in 0 1 2; do
    run --baseline map_pf_learned --ckpt "$M/$kind/seed$s" --tag "map_pf_learned_${kind}_s$s" --group G3 --protocols O-M-Hp --wins w05 --workers 3 $COMMON
  done
done
score --run icde_v11 --folds test few_shot_heldout
score --run icde_a3_union --folds test
# D: same-cohort horizon (offline methods rerun on the 60 s and 300 s prefixes; online ones are scored from w15)
CB="--version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout --wins w15 --episodes-file manifests/cohort_v11_episodes.csv --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15"
for P in 60 300; do
  R=$(printf "icde_cohort_p%03d" "$P")
  run --baselines map_beam_steer --group G3 --protocols F-M-Hp F-M-Hm --workers 2 --run "$R" --prefix-s "$P" $CB
  run --baselines map_beam_none --group G1 --protocols F-M-Hp F-M-Hm --workers 2 --run "$R" --prefix-s "$P" $CB
  run --baselines elastic_pathing_hp --group G1 --protocols F-M-Hp --workers 3 --run "$R" --prefix-s "$P" $CB
  run --baselines elastic_pathing --group G1 --protocols F-M-Hm --workers 3 --run "$R" --prefix-s "$P" $CB
  run --baselines opd2 --group G3 --protocols F-M-Hp --workers 3 --run "$R" --prefix-s "$P" $CB
  run --baselines map_pf_steer_lineage --group G3 --protocols F-M-Hp --workers 2 --run "$R" --prefix-s "$P" $CB
done
# E: whole drives (long; published searches last)
run --baselines map_pf_steer --group G3 --protocols O-M-Hp O-M-Hm --wins full --workers 2 $COMMON
run --baselines map_beam_steer --group G3 --protocols F-M-Hp F-M-Hm --wins full --workers 2 $COMMON
run --baselines macrotracking --group G3 --protocols O-M-Hp --wins full --workers 3 $COMMON
run --baselines map_pf_none --group G1 --protocols O-M-Hp O-M-Hm --wins full --workers 2 $COMMON
run --baselines map_beam_none --group G1 --protocols F-M-Hp F-M-Hm --wins full --workers 2 $COMMON
score --run icde_v11 --folds test few_shot_heldout
echo "$(date '+%F %T') CHAIN DONE"
