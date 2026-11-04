#!/usr/bin/env bash
# H100 side of the agent study (run ON the GPU host, over ssh): environment, pinned weights, vLLM server. Idempotent.
#   bash h100_setup.sh check      # read-only facts (GPU, disk, python, network); prints JSON lines
#   nohup bash h100_setup.sh setup > ~/h100_setup.log 2>&1 &   # env + weights in the background (10 to 25 min); then: tail ~/h100_setup.log
#   bash h100_setup.sh env        # python venv + vllm + huggingface_hub (records pip freeze); safe to rerun
#   bash h100_setup.sh weights    # download Qwen/Qwen3-14B at the pinned revision (sha from the HF API, kept in ~/qwen3_14b.revision)
#   bash h100_setup.sh serve      # start vLLM in tmux session "vllm" (or nohup), OpenAI-compatible on 127.0.0.1:8000
#   bash h100_setup.sh status     # health of the server and the last log lines
# Nothing here touches other users' processes.
set -u
MODEL=${MODEL:-Qwen/Qwen3-14B}
VENV=${VENV:-$HOME/venvs/vllm}
PORT=${PORT:-8000}
LOG=${LOG:-$HOME/vllm_qwen3_14b.log}
REV_FILE=${REV_FILE:-$HOME/qwen3_14b.revision}
cmd=${1:-check}

case "$cmd" in
  check)
    echo "{\"host\": \"$(hostname)\", \"os\": \"$(. /etc/os-release; echo $PRETTY_NAME)\", \"nproc\": $(nproc), \"mem_total_gb\": $(free -g | awk '/Mem/{print $2}'), \"mem_avail_gb\": $(free -g | awk '/Mem/{print $7}')}"
    nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv,noheader | sed 's/^/{"gpu": "/; s/$/"}/'
    echo "{\"gpu_procs\": \"$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')\"}"
    echo "{\"disk_home_avail\": \"$(df -h $HOME | awk 'NR==2{print $4}')\", \"disk_root_avail\": \"$(df -h / | awk 'NR==2{print $4}')\"}"
    echo "{\"python3\": \"$(python3 --version 2>&1)\", \"has_conda\": \"$(command -v conda || true)\", \"has_uv\": \"$(command -v uv || true)\", \"has_docker\": \"$(command -v docker || true)\", \"has_tmux\": \"$(command -v tmux || true)\"}"
    echo "{\"hf_cache\": \"$(du -sh $HOME/.cache/huggingface 2>/dev/null | cut -f1)\", \"hf_reachable\": \"$(curl -sI --max-time 10 https://huggingface.co | head -1 | tr -d '\r')\", \"pypi_reachable\": \"$(curl -sI --max-time 10 https://pypi.org | head -1 | tr -d '\r')\"}"
    echo "{\"venv_vllm\": \"$([ -x $VENV/bin/python ] && $VENV/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo none)\", \"weights\": \"$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/* 2>/dev/null | head -1)\", \"revision_file\": \"$(cat $REV_FILE 2>/dev/null)\"}"
    ;;
  setup)
    date; bash "$0" env && bash "$0" weights && echo "SETUP DONE $(date)" || echo "SETUP FAILED $(date)"
    ;;
  env)
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || { echo "venv creation failed (python3-venv missing?)"; exit 1; }
    "$VENV/bin/python" -m pip install -q -U pip || exit 1
    "$VENV/bin/python" -c "import vllm" 2>/dev/null || "$VENV/bin/python" -m pip install -q vllm || { echo "vllm install failed"; exit 1; }
    "$VENV/bin/python" -c "import huggingface_hub, requests" 2>/dev/null || "$VENV/bin/python" -m pip install -q huggingface_hub requests || exit 1
    "$VENV/bin/python" -m pip freeze > "$HOME/vllm_pip_freeze.txt"
    "$VENV/bin/python" -c "import vllm, torch, transformers; print({'vllm': vllm.__version__, 'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda': torch.version.cuda})"
    ;;
  weights)
    if [ ! -s "$REV_FILE" ]; then
      SHA=$(curl -s "https://huggingface.co/api/models/${MODEL}" | python3 -c "import sys, json; print(json.load(sys.stdin)['sha'])") || exit 1
      echo "$SHA" > "$REV_FILE"
    fi
    SHA=$(cat "$REV_FILE"); echo "revision $SHA"
    "$VENV/bin/python" - "$MODEL" "$SHA" <<'PY'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2])
print("downloaded", p)
PY
    du -sh "$HOME/.cache/huggingface/hub" | cut -f1
    ;;
  serve)
    "$VENV/bin/python" -c "import vllm" 2>/dev/null || { echo "vllm not installed in $VENV: run setup first"; exit 1; }
    [ -s "$REV_FILE" ] || { echo "no revision file: run weights first"; exit 1; }
    SHA=$(cat "$REV_FILE")
    ls -d "$HOME/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/$SHA" >/dev/null 2>&1 || { echo "weights for $SHA not downloaded: run weights first"; exit 1; }
    if curl -s --max-time 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then echo "a server already answers on port $PORT"; exit 0; fi
    CMD="VLLM_USE_FLASHINFER_SAMPLER=0 $VENV/bin/vllm serve $MODEL --revision $SHA --dtype bfloat16 --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 --host 127.0.0.1 --port $PORT --seed 0"
    echo "$CMD" > "$HOME/vllm_serve_cmd.txt"
    if command -v tmux >/dev/null; then tmux kill-session -t vllm 2>/dev/null; tmux new-session -d -s vllm "$CMD 2>&1 | tee $LOG"; else nohup bash -c "$CMD" > "$LOG" 2>&1 & fi
    echo "started; log $LOG (model load takes a few minutes; check with: bash $0 status)"
    ;;
  status)
    curl -s --max-time 10 "http://127.0.0.1:$PORT/v1/models" || echo "server not ready"; echo; tail -3 "$LOG" 2>/dev/null; tail -2 "$HOME/h100_setup.log" 2>/dev/null
    ;;
  *) echo "unknown command $cmd"; exit 2;;
esac
