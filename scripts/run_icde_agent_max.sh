#!/usr/bin/env bash
# C-max ceiling exploration (exploratory; the frozen study icde_agent is untouched). Needs the vLLM server at 32k context.
#   bash scripts/run_icde_agent_max.sh pilot          # 40 validation windows, seed 0, concurrency 12 (C-max config, selector tool, critic)
#   bash scripts/run_icde_agent_max.sh selector       # B-max on the formal cohort
#   bash scripts/run_icde_agent_max.sh llm 0          # C-max, one nominal seed (no critic)
#   bash scripts/run_icde_agent_max.sh critic 0       # C-max + critic, one seed (separate tag)
#   bash scripts/run_icde_agent_max.sh vote           # three-seed vote stems for both tags
#   bash scripts/run_icde_agent_max.sh score          # score every stem of icde_agent_max
#   bash scripts/run_icde_agent_max.sh tables         # agent_max_*.csv, agent_max_results.md
set -u
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
RUN=${RUN:-icde_agent_max}
COHORT=${COHORT:-manifests/icde_agent_cohort.csv}
CFG=${CFG:-configs/agent_max}
CONC=${CONC:-24}
LOGS=${CANTRAJ_LOGS:-${CANTRAJ_LOGS:-logs}}
step=${1:-}; seed=${2:-0}
run() { echo "$(date '+%F %T') START $*"; nice -n 10 "$PY" "$@" 2>&1 | grep -v -E "UserWarning|from scipy"; echo "$(date '+%F %T') END rc=$?"; }
case "$step" in
  pilot)
    run scripts/run_agent.py --run icde_agent_max_val --mode llm --seed 0 --manifest manifests/icde_agent_pilot.csv --concurrency "$CONC" \
        --config-dir "$CFG" --selector "$CFG/selector_max.json" --selector-tool --critic --tag agent_qwen3max_pilot | tee -a "$LOGS/icde_agent_max_pilot.log" ;;
  selector)
    run scripts/run_agent.py --run "$RUN" --mode selector --manifest "$COHORT" --config-dir "$CFG" --selector "$CFG/selector_max.json" --tag selector_max | tee -a "$LOGS/icde_agent_max_selector.log" ;;
  llm)
    run scripts/run_agent.py --run "$RUN" --mode llm --seed "$seed" --manifest "$COHORT" --concurrency "$CONC" --config-dir "$CFG" \
        --selector "$CFG/selector_max.json" --selector-tool --tag "agent_qwen3max_s$seed" --resume | tee -a "$LOGS/icde_agent_max_llm_s$seed.log" ;;
  critic)
    run scripts/run_agent.py --run "$RUN" --mode llm --seed "$seed" --manifest "$COHORT" --concurrency "$CONC" --config-dir "$CFG" \
        --selector "$CFG/selector_max.json" --selector-tool --critic --tag "agent_qwen3maxcritic_s$seed" --resume | tee -a "$LOGS/icde_agent_max_critic_s$seed.log" ;;
  vote)
    for tag in agent_qwen3max agent_qwen3maxcritic; do
      run scripts/agent_vote.py --run "$RUN" --tag "$tag" --seeds 0 1 2 --manifest "$COHORT" --selector "$CFG/selector_max.json" | tee -a "$LOGS/icde_agent_max_vote.log"
    done ;;
  score)
    run scripts/score.py --run "$RUN" --manifest manifests/episode_manifest_v1.parquet --eligible-only --folds test few_shot_heldout \
        --episodes-file "$COHORT" --workers 2 --resume | grep -E "groups|written|agent_|selector" | tee -a "$LOGS/icde_agent_max_score.log" ;;
  tables)
    run scripts/icde_agent_tables.py --run "$RUN" --cohort "$COHORT" --n-boot 1000 --agent-tag agent_qwen3max --selector-stem selector_max__G3 \
        --selector-results selector_max.csv --extra-stems agent_qwen3max_vote__G3 agent_qwen3maxcritic_s0__G3 agent_qwen3maxcritic_s1__G3 agent_qwen3maxcritic_s2__G3 agent_qwen3maxcritic_vote__G3 \
        --suffix _max | tee -a "$LOGS/icde_agent_max_tables.log" ;;
  *) echo "usage: $0 {pilot|selector|llm <seed>|critic <seed>|vote|score|tables}"; exit 2 ;;
esac
