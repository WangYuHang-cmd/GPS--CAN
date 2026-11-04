#!/usr/bin/env python3
"""Pilot report for the agent study: interface health (parse / invalid / budget / truncation rates), decision and tool-call
distributions, tokens and timing per window, throughput for the formal-run estimate, and the live (uncached) cost check.

    python scripts/agent_pilot_report.py --run icde_agent_val --tags agent_qwen3_s0 agent_qwen3_s1 agent_qwen3_s2 [--live-tag agent_qwen3_s0_live]
Writes reports/icde/tables/agent_pilot_summary.csv and (with --live-tag) reports/icde/tables/agent_live_check.csv.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths  # noqa: E402

OUT = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")


def summarise(df, tag):
    n = len(df)
    q = lambda c, p: float(np.nanpercentile(df[c].astype(float), p)) if c in df and df[c].notna().any() else np.nan  # noqa: E731
    row = dict(tag=tag, n=n, submitted_pct=100 * (df.status == "submitted").mean())
    for st in ("invalid_choice", "no_submit", "total_token_budget", "timeout", "parse_error", "llm_error", "no_candidates"):
        row[f"{st}_pct"] = 100 * (df.status == st).mean()
    row.update(truncation_pct=100 * (df.truncations > 0).mean(), forced_submit_pct=100 * df.forced_submit.astype(bool).mean(),
               decisions_mean=df.n_decisions.mean(), tool_calls_mean=df.n_tool_calls.mean(), tool_calls_max=int(df.n_tool_calls.max()),
               completion_tokens_mean=df.completion_tokens.mean(), completion_tokens_p95=q("completion_tokens", 95), completion_tokens_max=int(df.completion_tokens.max()),
               reasoning_tokens_mean=float(df.reasoning_tokens.mean()) if df.reasoning_tokens.notna().any() else np.nan,
               prompt_tokens_mean=df.prompt_tokens.mean(), llm_wall_med=q("llm_wall_s", 50), llm_wall_p95=q("llm_wall_s", 95),
               cost_med=q("cost_reconstructed_s", 50), cost_p95=q("cost_reconstructed_s", 95), elapsed_med=q("elapsed_s", 50),
               retries_total=int(df.retries.sum()))
    for t in ("beam", "pf", "ep"):
        row[f"use_{t}_pct"] = 100 * df.tools_run.fillna("").str.contains(t).mean()
    row["choice_tool"] = df.choice_tool.value_counts().to_dict()
    row["tokens_per_s_aggregate"] = float(df.completion_tokens.sum() / max(df.elapsed_s.max(), 1e-9)) if "elapsed_s" in df else np.nan
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="icde_agent_val")
    ap.add_argument("--tags", nargs="+", default=["agent_qwen3_s0", "agent_qwen3_s1", "agent_qwen3_s2"])
    ap.add_argument("--live-tag", default=None)
    ap.add_argument("--formal-n", type=int, default=659)
    args = ap.parse_args()
    res_dir = os.path.join(paths.BENCH, "agent_logs", args.run, "results")
    rows, frames = [], []
    for tag in args.tags:
        p = os.path.join(res_dir, f"{tag}.csv")
        if not os.path.exists(p):
            print(f"missing {p}")
            continue
        df = pd.read_csv(p)
        frames.append(df.assign(tag=tag))
        rows.append(summarise(df, tag))
    if frames:
        allf = pd.concat(frames, ignore_index=True)
        rows.append(summarise(allf, "all"))
        tab = pd.DataFrame(rows)
        os.makedirs(OUT, exist_ok=True)
        tab.to_csv(os.path.join(OUT, "agent_pilot_summary.csv"), index=False)
        pd.set_option("display.width", 250)
        print(tab.drop(columns=["choice_tool"]).round(1).T.to_string())
        print("choice tools:", {r["tag"]: r["choice_tool"] for r in rows})
        # wall-clock based formal estimate: per-window elapsed at the pilot's concurrency, scaled to the formal cohort and 3 seeds
        wall = allf.groupby("tag").elapsed_s.sum().mean()      # one seed's serial sum
        print(f"formal estimate: {args.formal_n} windows x 3 seeds at the pilot's per-window elapsed mean {allf.elapsed_s.mean():.1f} s "
              f"-> serial {args.formal_n * 3 * allf.elapsed_s.mean() / 3600:.1f} h; with the run's concurrency divide accordingly (sum per seed in pilot {wall:.0f} s)")
        # status reasons
        err = allf[allf.status != "submitted"][["tag", "episode_id", "status", "error"]]
        if len(err):
            print(err.head(20).to_string(index=False))
    if args.live_tag:
        live_p = os.path.join(res_dir, f"{args.live_tag}.csv")
        base = args.live_tag.replace("_live", "")
        base_p = os.path.join(res_dir, f"{base}.csv")
        if os.path.exists(live_p) and os.path.exists(base_p):
            lv = pd.read_csv(live_p).set_index("episode_id")
            bs = pd.read_csv(base_p).set_index("episode_id")
            common = lv.index.intersection(bs.index)
            # live end-to-end = LLM wall + real tool execution (load, graph, search per call; the subprocess time is in tool_logical_s)
            live_e2e = lv.loc[common, "llm_wall_s"] + lv.loc[common, "tool_logical_s"]
            ratio = (live_e2e / bs.loc[common, "cost_reconstructed_s"]).replace([np.inf], np.nan)
            chk = pd.DataFrame([dict(n=len(common), live_med_s=float(live_e2e.median()),
                                     reconstructed_med_s=float(bs.loc[common, "cost_reconstructed_s"].median()), ratio_median=float(ratio.median()),
                                     ratio_p90=float(ratio.quantile(0.9)), live_tool_med_s=float(lv.loc[common, "tool_logical_s"].median()),
                                     cached_tool_med_s=float(bs.loc[common, "tool_logical_s"].median()), same_choice_pct=100 * float((lv.loc[common, "choice"] == bs.loc[common, "choice"]).mean()))])
            chk.to_csv(os.path.join(OUT, "agent_live_check.csv"), index=False)
            print("live check:", chk.round(2).to_dict("records")[0])
        else:
            print("live check: result files missing", live_p, base_p)
    # per-window log sanity: every log has a final status and no forbidden key
    logs = glob.glob(os.path.join(paths.BENCH, "agent_logs", args.run, "s*", "*.jsonl"))
    bad = 0
    for f in logs[:200]:
        txt = open(f).read()
        if any(k in txt for k in ('"p0_lat"', '"latitude"', '"route_ok"', '"ade"')):
            bad += 1
    print(f"{len(logs)} window logs checked (first 200 for forbidden keys): {bad} with forbidden keys")


if __name__ == "__main__":
    main()
