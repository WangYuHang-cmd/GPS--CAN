#!/usr/bin/env python3
"""Agent study, configurations B and C on a cohort whose tool cache exists (scripts/run_agent_tools.py).

    --mode selector : B, the supervised selector (every reconstructor called; selector_v1.json frozen on validation)
    --mode llm      : C, the Qwen tool-calling controller through an OpenAI-compatible server (one seed per run)

This process never builds a graph, never loads a drive and never reads the reference: it only reads the cache.
Outputs: results/<mode>[_s<seed>].csv (one row per window, failures explicit), per-window JSONL logs (llm), and the
prediction stems selector__G3, agent_qwen3_s<k>__G3 and agent_qwen3_fb_s<k>__G3 (fallback to the selector when the
agent produced no valid output; its extra cost = the tools the agent had not run).

    python scripts/run_agent.py --run icde_agent_val --mode selector --manifest manifests/icde_agent_val.csv
    python scripts/run_agent.py --run icde_agent_val --mode llm --seed 0 --manifest manifests/icde_agent_pilot.csv --concurrency 12
    python scripts/run_agent.py ... --live --concurrency 1      # uncached cost verification: each reconstructor call really runs
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths, predfile  # noqa: E402
from cantraj.agent.llm import QwenClient  # noqa: E402
from cantraj.agent.loop import Budget, run_agent_episode, run_critic  # noqa: E402
from cantraj.agent.selector import Selector  # noqa: E402
from cantraj.agent.toolbox import TOOLS, EpisodeToolBox  # noqa: E402

PROTO = "F-M-Hp"
CFG = os.path.join(paths.GPS_DIR, "configs", "agent")
PY = sys.executable


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def cache_root(run):
    return os.path.join(paths.BENCH, "agent_toolcache", run)


def logs_root(run):
    return os.path.join(paths.BENCH, "agent_logs", run)


def cache_dir(root, r):
    return os.path.join(root, r.driver, r.drive, f"k{int(r.k):03d}")


def cost_of(tb, extra_tools=()):
    return float(sum(tb.logical_time(t) for t in set(tb.called_tools) | set(extra_tools)))


def write_stem(run, stem, rows, meta):
    """rows: list of (episode_id, t_abs, xy)."""
    if not rows:
        print(f"{stem}: no output rows", flush=True)
        return
    df = pd.concat([predfile.new_frame(eid, t, xy, None, candidate_id=0, prob=1.0) for eid, t, xy in rows], ignore_index=True)
    path = os.path.join(paths.PREDS, run, PROTO, f"{stem}.part0.parquet")
    predfile.write(df, path, dict(meta, baseline=stem.split("__")[0], group=stem.split("__")[1], protocol=PROTO, run=run, n_episodes=int(df.episode_id.nunique())))
    print(f"{stem}: {df.episode_id.nunique()} windows -> {path}", flush=True)


def write_skipped(run, stem, rows):
    path = os.path.join(paths.PREDS, run, PROTO, f"{stem}.part0.skipped.csv")
    if rows:
        pd.DataFrame(rows, columns=["episode_id", "reason"]).to_csv(path, index=False)
    elif os.path.exists(path):
        os.remove(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True, choices=["selector", "llm"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--config-dir", default=CFG, help="directory with budget.json, sampling.json, system_prompt.txt, tools_schema.json, selector json")
    ap.add_argument("--budget", default=None)
    ap.add_argument("--sampling", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--selector", default=None)
    ap.add_argument("--selector-tool", action="store_true", help="give the agent the score_candidates tool backed by the selector (C-max)")
    ap.add_argument("--critic", action="store_true", help="C-max: a second critic conversation reviews the controller's choice")
    ap.add_argument("--tag", default=None, help="result / stem tag override (default selector | agent_qwen3_s<seed>)")
    ap.add_argument("--live", action="store_true", help="uncached verification: every reconstructor call really runs in a subprocess")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfgd = args.config_dir
    args.budget = args.budget or os.path.join(cfgd, "budget.json")
    args.sampling = args.sampling or os.path.join(cfgd, "sampling.json")
    args.prompt = args.prompt or os.path.join(cfgd, "system_prompt.txt")
    args.schema = args.schema or os.path.join(cfgd, "tools_schema.json")
    if args.selector is None:
        cands = sorted(glob.glob(os.path.join(cfgd, "selector*.json")))
        args.selector = cands[0] if cands else os.path.join(cfgd, "selector_v1.json")
    cohort = pd.read_csv(args.manifest)
    if args.limit:
        cohort = cohort.head(args.limit)
    root = cache_root(args.run)
    rm_cache = json.load(open(os.path.join(root, "run_manifest.json")))
    tag = args.tag or ("selector" if args.mode == "selector" else f"agent_qwen3_s{args.seed}")
    res_dir = os.path.join(logs_root(args.run), "results")
    os.makedirs(res_dir, exist_ok=True)
    res_path = os.path.join(res_dir, f"{tag}{'_live' if args.live else ''}.csv")
    budget = Budget.from_json(args.budget)
    sampling = json.load(open(args.sampling))
    manifest = dict(run=args.run, mode=args.mode, tag=tag, seed=args.seed, cohort=os.path.basename(args.manifest), cohort_sha=sha256_file(args.manifest),
                    cache_manifest=rm_cache, budget_sha=sha256_file(args.budget), sampling_sha=sha256_file(args.sampling), prompt_sha=sha256_file(args.prompt),
                    schema_sha=sha256_file(args.schema), selector_sha=sha256_file(args.selector) if os.path.exists(args.selector) else None,
                    model=sampling.get("model"), revision=sampling.get("revision"), mode_thinking=sampling.get("mode"), live=args.live, concurrency=args.concurrency,
                    config_dir=cfgd, selector_tool=args.selector_tool, critic=args.critic)
    man_path = os.path.join(res_dir, f"{tag}{'_live' if args.live else ''}.run_manifest.json")
    if os.path.exists(man_path) and args.resume:
        old = json.load(open(man_path))
        diff = [k for k in manifest if k not in ("concurrency",) and old.get(k) != manifest[k]]
        if diff:
            raise SystemExit(f"--resume refused: run manifest differs in {diff}")
    json.dump(manifest, open(man_path, "w"), indent=1)
    done = set()
    rows = []
    if args.resume and os.path.exists(res_path):
        prev = pd.read_csv(res_path)
        rows = prev.to_dict("records")
        done = set(prev.episode_id)
    todo = [r for r in cohort.itertuples() if r.episode_id not in done and os.path.exists(os.path.join(cache_dir(root, r), "diagnostics.json"))]
    missing = [r.episode_id for r in cohort.itertuples() if not os.path.exists(os.path.join(cache_dir(root, r), "diagnostics.json"))]
    print(f"{len(cohort)} windows, {len(todo)} to run, {len(done)} done, {len(missing)} without cache", flush=True)
    sel = Selector.load(args.selector) if os.path.exists(args.selector) else None
    if args.mode == "selector" and sel is None:
        raise SystemExit(f"selector file missing: {args.selector}")
    meta = dict(layer="events", time_comp="train", manifest=args.manifest, benchmark_version="v1.1", shard="0/1", study="agent",
                code_sha=rm_cache.get("code_sha"), map_version=rm_cache.get("tiles_dir"), tiles_manifest_sha=rm_cache.get("tiles_manifest_sha"),
                manifest_sha=manifest["cohort_sha"], episode_visibility=True, agent_manifest=man_path)
    lock = threading.Lock()
    t_run = time.time()

    def flush():
        pd.DataFrame(rows).to_csv(res_path, index=False)

    if args.mode == "selector":
        for r in todo:
            tb = EpisodeToolBox(cache_dir(root, r)).call_all()
            t0 = time.perf_counter()
            feats = {c: tb.features(c) for c in tb.all_candidates()}
            choice = sel.select(feats) if feats else None
            rows.append(dict(episode_id=r.episode_id, driver=r.driver, drive=r.drive, fold=r.fold, seed=-1, status="submitted" if choice else "no_candidates",
                             choice=choice, choice_tool=choice.rsplit("-", 1)[0] if choice else None, n_candidates=len(feats),
                             tool_logical_s=cost_of(tb), select_s=time.perf_counter() - t0, cost_reconstructed_s=cost_of(tb) + (time.perf_counter() - t0),
                             tools_run="|".join(TOOLS)))
        flush()
    else:
        client = QwenClient(args.base_url, sampling["model"], sampling)
        system_prompt = open(args.prompt).read()
        schema = json.load(open(args.schema))
        thinking = sampling.get("mode", "thinking") == "thinking"
        critic_prompt = open(os.path.join(cfgd, "critic_prompt.txt")).read() if args.critic else None
        critic_budget = Budget.from_json(os.path.join(cfgd, "critic_budget.json")) if args.critic else None
        sub = f"s{args.seed}" if args.tag is None else tag          # custom tags (pilot variants) keep their own log directory
        log_dir = os.path.join(logs_root(args.run), f"{sub}{'_live' if args.live else ''}")
        os.makedirs(log_dir, exist_ok=True)

        def live_runner_for(r):
            def run(tool):
                cmd = [PY, os.path.join(HERE, "run_agent_tools.py"), "--run", args.run, "--manifest", args.manifest, "--only", tool, "--episode", r.episode_id]
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                line = [l for l in out.stdout.splitlines() if l.startswith("{")]
                info = json.loads(line[-1]) if line else {"error": out.stderr[-300:]}
                live_log.setdefault(r.episode_id, []).append(info)
                return float(info.get("tool_s", 0.0) or 0.0) + float(info.get("graph_s", 0.0) or 0.0) + float(info.get("load_s", 0.0) or 0.0)
            return run

        live_log = {}

        def one(r):
            tb = EpisodeToolBox(cache_dir(root, r), selector=sel if args.selector_tool else None)
            res = run_agent_episode(tb, client, budget, system_prompt, schema, seed=args.seed, enable_thinking=thinking, episode_id=r.episode_id,
                                    live_runner=live_runner_for(r) if args.live else None)
            critic_rec = None
            if args.critic and res.choice is not None:
                vis = [c for c in tb.visible if c != res.choice]        # alternatives: best others by selector probability, else other Top-1s
                if sel is not None and vis:
                    alts = sorted(vis, key=lambda c: -sel.proba(tb.features(c, called_only=True)))[:2]
                else:
                    alts = [c for c in vis if c.endswith("-1")][:2]
                if alts:
                    cres = run_critic(tb, client, critic_budget, critic_prompt, schema, seed=args.seed, enable_thinking=thinking, chosen=res.choice,
                                      alternatives=alts, episode_id=r.episode_id)
                    critic_rec = dict(status=cres.status, choice=cres.choice, overturned=bool(cres.choice and cres.choice != res.choice),
                                      n_decisions=cres.n_decisions, n_tool_calls=cres.n_tool_calls, completion_tokens=cres.completion_tokens,
                                      llm_wall_s=cres.llm_wall_s, messages=cres.messages)
                    if cres.status == "submitted" and cres.choice:
                        res.choice = cres.choice
                    res.llm_wall_s += cres.llm_wall_s; res.virtual_clock_s += cres.virtual_clock_s; res.completion_tokens += cres.completion_tokens
                    res.prompt_tokens += cres.prompt_tokens; res.n_tool_calls += cres.n_tool_calls; res.n_decisions += cres.n_decisions
                    if cres.reasoning_tokens is not None and res.reasoning_tokens is not None:
                        res.reasoning_tokens += cres.reasoning_tokens
            fb_choice, fb_extra = None, 0.0
            if res.choice is None and sel is not None:          # pre-fixed fallback: the selector over every tool, reusing what the agent ran
                tb2 = EpisodeToolBox(cache_dir(root, r)).call_all()
                feats = {c: tb2.features(c) for c in tb2.all_candidates()}
                fb_choice = sel.select(feats) if feats else None
                fb_extra = float(sum(tb.logical_time(t) for t in TOOLS if t not in tb.called_tools))
            rec = res.to_dict()
            msgs = rec.pop("messages")
            with open(os.path.join(log_dir, r.episode_id.replace("/", "__") + ".jsonl"), "w") as f:
                f.write(json.dumps(dict(rec, messages=msgs, live=live_log.get(r.episode_id), critic=critic_rec), ensure_ascii=False, default=str) + "\n")
            row = dict(episode_id=r.episode_id, driver=r.driver, drive=r.drive, fold=r.fold, seed=args.seed, status=res.status, choice=res.choice,
                       choice_tool=res.choice.rsplit("-", 1)[0] if res.choice else None, n_decisions=res.n_decisions, n_tool_calls=res.n_tool_calls,
                       tools_run="|".join(res.tools_run), prompt_tokens=res.prompt_tokens, completion_tokens=res.completion_tokens,
                       reasoning_tokens=res.reasoning_tokens, llm_wall_s=res.llm_wall_s, tool_logical_s=res.tool_logical_s, tool_exec_s=res.tool_exec_s,
                       cost_reconstructed_s=res.virtual_clock_s, elapsed_s=res.elapsed_s, truncations=res.truncations, retries=res.retries,
                       forced_submit=res.forced_submit, error=(res.error or "")[:200], fallback_choice=fb_choice, fallback_extra_tool_s=fb_extra,
                       n_candidates_visible=len(tb.visible), critic_overturned=(critic_rec or {}).get("overturned"), critic_status=(critic_rec or {}).get("status"))
            with lock:
                rows.append(row)
                if len(rows) % 20 == 0:
                    flush()
                    n_ok = sum(1 for x in rows if x.get("status") == "submitted")
                    print(f"  {len(rows)}/{len(cohort)} windows, submitted {n_ok}, {time.time() - t_run:.0f}s", flush=True)

        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as ex:
            list(ex.map(one, todo))
        flush()
    # prediction stems from the result rows
    df = pd.DataFrame(rows)
    by_id = {r.episode_id: r for r in cohort.itertuples()}
    out_rows, fb_rows, skipped, fb_skipped = [], [], [], []
    for rec in df.to_dict("records"):
        r = by_id.get(rec["episode_id"])
        if r is None:
            continue
        cdir = cache_dir(root, r)
        t_abs = np.load(os.path.join(cdir, "t_abs.npz"))["t_abs"]
        arr = np.load(os.path.join(cdir, "candidates.npz"))
        ch = rec.get("choice")
        if isinstance(ch, str) and ch:
            out_rows.append((rec["episode_id"], t_abs, arr[f"xy__{ch}"]))
            fb_rows.append((rec["episode_id"], t_abs, arr[f"xy__{ch}"]))
        else:
            skipped.append((rec["episode_id"], f"{tag}: {rec.get('status')}: {str(rec.get('error', ''))[:60]}"))
            fb = rec.get("fallback_choice")
            if isinstance(fb, str) and fb:
                fb_rows.append((rec["episode_id"], t_abs, arr[f"xy__{fb}"]))
            else:
                fb_skipped.append((rec["episode_id"], f"{tag}_fb: no candidate"))
    stem = f"{tag}__G3"
    write_stem(args.run, stem, out_rows, dict(meta, n_skipped=len(skipped), skipped=skipped[:50]))
    write_skipped(args.run, stem, skipped)
    if args.mode == "llm":
        fstem = f"agent_qwen3_fb_s{args.seed}__G3" if args.tag is None else f"{tag}_fb__G3"
        write_stem(args.run, fstem, fb_rows, dict(meta, n_skipped=len(fb_skipped), skipped=fb_skipped[:50], fallback="selector"))
        write_skipped(args.run, fstem, fb_skipped)
    st = df.status.value_counts().to_dict() if len(df) else {}
    print(f"status: {st}; results -> {res_path}; {(time.time() - t_run) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
