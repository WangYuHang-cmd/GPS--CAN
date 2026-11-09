#!/usr/bin/env bash
# ICDE: the two published methods added on 10/30, in one sequential chain (memory gate before every step):
# (1) SPy open values (moving-average window, turn-ramp level) chosen on the 60 val selection drives (run icde_dev,
#     known-start variant, w05) by a pre-set rule: highest route_ok, then path recall at 50 m, then lowest median ADE
#     (scripts/icde_val_select.py -> reports/icde/spy_val_selection.csv); the test folds are not read for this choice;
# (2) Dewri et al. (no value tuned) and SPy with the selected values on test + few-shot (run icde_v11): w05, w15, whole
#     drives (SPy's area variant: w05, w15); (3) scoring. Do not edit while it runs.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
GATE_MB=${GATE_MB:-12000}
BASE="--version v1.1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --tiles-dir ${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/maps/tiles_v2 --map-episodes manifests/episode_map_v11.parquet --resume"
COMMON="--run icde_v11 --folds test few_shot_heldout $BASE --flush-min 15 --workers 3"
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
  echo "$(date '+%F %T') END rc=$rc score"
}

# (1) SPy selection on val
SEL="--group G3 --run icde_dev --folds val --wins w05 --drives-file manifests/icde_val_select_drives.csv $BASE --flush-min 0 --workers 3"
STEMS=""
for ma in 5 15; do
  for rd in 20 45; do
    STEMS="$STEMS spy_ma${ma}r${rd}__G3"
    run --baselines spy --protocols F-M-Hm $SEL --set "ma=$ma,ramp_deg=$rd" --stem-suffix "_ma${ma}r${rd}"
  done
done
score --run icde_dev --folds val --methods $STEMS
"$PY" scripts/icde_val_select.py --stems $STEMS --protocol F-M-Hm --req G3 --out reports/icde/spy_val_selection.csv
TOP=$("$PY" -c "import pandas as pd; print(pd.read_csv('reports/icde/spy_val_selection.csv').stem.iloc[0])" 2>/dev/null)
MA=$(echo "$TOP" | sed -n 's/^spy_ma\([0-9]*\)r[0-9]*__G3$/\1/p')
RD=$(echo "$TOP" | sed -n 's/^spy_ma[0-9]*r\([0-9]*\)__G3$/\1/p')
echo "$(date '+%F %T') SPY SELECTED $TOP -> ma=$MA ramp_deg=$RD"

# (2) Dewri et al.: no value tuned
run --baselines dewri --group G1 --protocols F-M-Hm --wins w05 w15 $COMMON
run --baselines dewri_hp --group G1 --protocols F-M-Hp --wins w05 w15 $COMMON
# SPy with the selected values
if [ -n "$MA" ] && [ -n "$RD" ]; then
  run --baselines spy --group G3 --protocols F-M-Hm --wins w05 w15 $COMMON --set "ma=$MA,ramp_deg=$RD"
  run --baselines spy_hp --group G3 --protocols F-M-Hp --wins w05 w15 $COMMON --set "ma=$MA,ramp_deg=$RD"
  run --baselines spy_area --group G3 --protocols F-M-Hm --wins w05 w15 $COMMON --set "ma=$MA,ramp_deg=$RD"
else
  echo "$(date '+%F %T') SPY SELECTION MISSING: SPy formal runs skipped"
fi
score --run icde_v11 --folds test few_shot_heldout
# whole drives
run --baselines dewri --group G1 --protocols F-M-Hm --wins full $COMMON
run --baselines dewri_hp --group G1 --protocols F-M-Hp --wins full $COMMON
if [ -n "$MA" ] && [ -n "$RD" ]; then
  run --baselines spy --group G3 --protocols F-M-Hm --wins full $COMMON --set "ma=$MA,ramp_deg=$RD"
  run --baselines spy_hp --group G3 --protocols F-M-Hp --wins full $COMMON --set "ma=$MA,ramp_deg=$RD"
fi
score --run icde_v11 --folds test few_shot_heldout
echo "$(date '+%F %T') CHAIN DONE"
