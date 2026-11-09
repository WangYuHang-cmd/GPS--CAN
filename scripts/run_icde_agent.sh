#!/usr/bin/env bash
# Agent study chain (supplementary cohort). Run AFTER configs/agent/protocol_v1.md is frozen. Do not edit while it runs.
#   bash scripts/run_icde_agent.sh cache      # tool cache for the formal cohort (3 workers) + identity stems
#   bash scripts/run_icde_agent.sh score      # score every stem of the run (test + few-shot, cohort episodes only)
#   bash scripts/run_icde_agent.sh selector   # B on the cohort
#   bash scripts/run_icde_agent.sh llm 0      # C, one nominal seed (needs the vLLM tunnel on 127.0.0.1:8000)
#   bash scripts/run_icde_agent.sh tables     # tables, LaTeX table, macros
# Logs: $CANTRAJ_LOGS/icde_agent_<step>.log. One heavy local job at a time; the LLM runs on the remote H100.
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
RUN=${RUN:-icde_agent}
COHORT=${COHORT:-manifests/icde_agent_cohort.csv}
WORKERS=${WORKERS:-3}
CONC=${CONC:-12}
GATE_MB=${GATE_MB:-12000}
LOGS=${CANTRAJ_LOGS:-${CANTRAJ_LOGS:-logs}}
step=${1:-}; seed=${2:-0}

gate() {
  while [ "$(awk '/MemAvailable/ {print int($2/1024)}' /proc/meminfo)" -lt "$GATE_MB" ]; do sleep 60; done
}
run() {
  gate
  echo "$(date '+%F %T') START $*"
  nice -n 10 ionice -c3 "$PY" "$@"
  echo "$(date '+%F %T') END rc=$?"
}

case "$step" in
  cache)
    run scripts/run_agent_tools.py --run "$RUN" --manifest "$COHORT" --workers "$WORKERS" --resume 2>&1 | grep -v -E "UserWarning|from scipy" | tee -a "$LOGS/icde_agent_cache.log" ;;
  score)
    run scripts/score.py --run "$RUN" --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout \
        --episodes-file "$COHORT" --workers 2 2>&1 | grep -v -E "UserWarning|from scipy" | tee -a "$LOGS/icde_agent_score.log" ;;
  selector)
    run scripts/run_agent.py --run "$RUN" --mode selector --manifest "$COHORT" 2>&1 | grep -v -E "UserWarning|from scipy" | tee -a "$LOGS/icde_agent_selector.log" ;;
  llm)
    run scripts/run_agent.py --run "$RUN" --mode llm --seed "$seed" --manifest "$COHORT" --concurrency "$CONC" --resume 2>&1 | grep -v -E "UserWarning|from scipy" | tee -a "$LOGS/icde_agent_llm_s$seed.log" ;;
  tables)
    run scripts/icde_agent_tables.py --run "$RUN" --cohort "$COHORT" --n-boot 1000 2>&1 | grep -v -E "UserWarning|from scipy" | tee -a "$LOGS/icde_agent_tables.log"
    run scripts/icde_macros.py 2>&1 | grep -v -E "UserWarning|from scipy" | tail -1 ;;
  *) echo "usage: $0 {cache|score|selector|llm <seed>|tables}"; exit 2 ;;
esac
echo "$(date '+%F %T') CHAIN STEP DONE $step"
