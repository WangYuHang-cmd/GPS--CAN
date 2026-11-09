#!/usr/bin/env bash
# ICDE sensitivity sweep (plan v0.9, item C): how the FIXED map attacks degrade when the CAN inputs they see are
# coarsened. Research question: how many routes do these attacks still recover at each degradation level? Not a
# protection threshold: an adapted attacker (re-tuned noise, re-calibration, re-training) is not measured here.
# Grid fixed before the first run and reported in full (no level is added or dropped on results):
#   steering-wheel angle quantised to 2 / 5 / 10 / 20 deg; steering held at 1 / 0.2 Hz; steering offset noise 2 / 5 deg
#   (one release seed; constant within each drive second, shared by overlapping windows); steering delayed 0.5 s (a
#   time-misalignment probe, reported apart); speed quantised to 1 / 2 / 5 m/s; speed held at 1 Hz.
# Methods: map PF with steering (O-M-Hp) and map beam with steering (F-M-Hp) on every level; the speed-only beam
# (F-M-Hp) on the speed levels (steering levels leave speed-only methods unchanged: their clean rows are the reference).
# test + few-shot, 5-minute windows, run icde_probe (icde_v11 untouched), 3 workers, memory gate, resume, scoring
# after the core levels, after the speed-only beam and at the end. Core levels first, the rest after.
# Usage: run_icde_probe.sh <pid-to-wait-for>. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
WAIT_PID=${1:-0}
while [ "$WAIT_PID" -gt 0 ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
PY=${PYTHON:-python3}
GATE_MB=${GATE_MB:-12000}
WORKERS=${WORKERS:-3}
BASE="--run icde_probe --version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout --wins w05 --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume --flush-min 15 --workers $WORKERS"
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
CORE_STEER="sq2:steer_quant=2 sq5:steer_quant=5 sq10:steer_quant=10 sq20:steer_quant=20 s1hz:steer_hz=1 s02hz:steer_hz=0.2"
CORE_SPEED="pq1:speed_quant=1 pq5:speed_quant=5 v1hz:speed_hz=1"
REST_STEER="sn5:steer_noise=5 sn2:steer_noise=2 sd05:steer_delay=0.5"
REST_SPEED="pq2:speed_quant=2"
steer_rows() {           # PF and beam with steering on one level
  tag=${1%%:*}; spec=${1#*:}
  run --baselines map_pf_steer --group G3 --protocols O-M-Hp --perturb "$spec" --stem-suffix "_$tag" $BASE
  run --baselines map_beam_steer --group G3 --protocols F-M-Hp --perturb "$spec" --stem-suffix "_$tag" $BASE
}
speed_only_row() {       # speed-only beam on one speed level
  tag=${1%%:*}; spec=${1#*:}
  run --baselines map_beam_none --group G1 --protocols F-M-Hp --perturb "$spec" --stem-suffix "_$tag" $BASE
}
for lv in $CORE_STEER $CORE_SPEED; do steer_rows "$lv"; done
score --run icde_probe --folds test few_shot_heldout
for lv in $CORE_SPEED; do speed_only_row "$lv"; done
score --run icde_probe --folds test few_shot_heldout
for lv in $REST_STEER $REST_SPEED; do steer_rows "$lv"; done
for lv in $REST_SPEED; do speed_only_row "$lv"; done
score --run icde_probe --folds test few_shot_heldout
echo "$(date '+%F %T') CHAIN DONE"
