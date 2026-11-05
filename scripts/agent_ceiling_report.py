#!/usr/bin/env python3
"""Ceiling exploration of the agent (C-max): reports/icde/agent_ceiling.md from the C-max tables, result CSVs, conversation logs
and candidate-pool scores. Exploratory and reported in full: every variant, failure shares, costs, tool use, critic overturns,
vote agreement, and the pre-specified decision rule for the paper.

    python scripts/agent_ceiling_report.py --run icde_agent_max --prev-run icde_agent --served-context 16384

Reads (all optional, sections are skipped when a file is absent):
  reports/icde/tables/agent_max_w05.csv, agent_max_pairs_w05.csv, agent_max_status_w05.csv  (icde_agent_tables.py --suffix _max)
  reports/icde/tables/agent_w05.csv (frozen study), agent_selector_max_val.csv, agent_max_val_hints.csv
  $CANTRAJ_BENCH/agent_logs/<run>/results/*.csv and <tag>/*.jsonl, $CANTRAJ_BENCH/scores/<run>/episodes_all.parquet (poolcand_* stems)
  configs/agent_max/{budget,critic_budget,sampling,selector_max}.json, the seed-0 run manifest
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cantraj import paths  # noqa: E402

PROTO = "F-M-Hp"
TABLES = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
FOLD_LABEL = {"test": "test", "few_shot_heldout": "few-shot", "all": "both folds pooled"}
ROW_LABEL = {"beam": "A: offline beam (published setting, Top-1)", "selector": "B-max: supervised selector over every preset (validation-fitted)",
             "agent_prev": "C (frozen first configuration, three seeds pooled)", "agent": "C-max: single controller, three seeds pooled",
             "agent_s0": "C-max seed 0", "agent_s1": "C-max seed 1", "agent_s2": "C-max seed 2", "agent_fb": "C-max with selector fallback",
             "agent_qwen3max_vote": "C-max + three-seed vote", "agent_qwen3maxcritic_s0": "C-max + critic, seed 0",
             "agent_qwen3maxcritic_s1": "C-max + critic, seed 1", "agent_qwen3maxcritic_s2": "C-max + critic, seed 2",
             "agent_qwen3maxcritic_vote": "C-max + critic, seed vote", "oracle": "pool oracle over every preset (diagnostic, uses the reference)"}
PAIR_LABEL = {"agent_vs_selector": "C-max minus B-max", "agent_vs_beam": "C-max minus A", "agent_vs_prev": "C-max minus C (frozen)",
              "agent_qwen3max_vote_vs_selector": "vote minus B-max", "agent_qwen3max_vote_vs_beam": "vote minus A",
              "agent_qwen3maxcritic_s0_vs_selector": "critic seed 0 minus B-max", "agent_qwen3maxcritic_s0_vs_beam": "critic seed 0 minus A",
              "agent_qwen3maxcritic_vote_vs_selector": "critic vote minus B-max", "agent_qwen3maxcritic_vote_vs_beam": "critic vote minus A"}


def _f(x, nd=1):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "fail" if xf == np.inf else "n/a"
    return f"{xf:,.{nd}f}"


def _load_json(p):
    with open(p) as f:
        return json.load(f)


def pool_scores(run):
    """(episode_id, candidate id) -> (route_ok, ade) from the poolcand_* stems of `run`."""
    p = os.path.join(paths.SCORES, run, "episodes_all.parquet")
    if not os.path.exists(p):
        return {}
    ep = pd.read_parquet(p, columns=["episode_id", "method", "protocol", "ade", "route_ok"])
    ep = ep[(ep.protocol == PROTO) & ep.method.str.startswith("poolcand_")]
    cid = ep.method.str.split("__").str[0].str.replace("poolcand_", "", regex=False)
    return {(e, c): (float(r), float(a)) for e, c, r, a in zip(ep.episode_id, cid, ep.route_ok, ep.ade)}


def submit_before_critic(rec):
    """Candidate id of the controller's own submit_choice call (before the critic), from the conversation log."""
    last = None
    for msg in rec.get("messages") or []:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            if fn.get("name") == "submit_choice":
                try:
                    args = fn.get("arguments")
                    args = json.loads(args) if isinstance(args, str) else (args or {})
                    last = args.get("candidate_id", last)
                except (TypeError, ValueError):
                    pass
    return last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="icde_agent_max")
    ap.add_argument("--prev-run", default="icde_agent")
    ap.add_argument("--cohort", default=os.path.join(paths.MANIFESTS, "icde_agent_cohort.csv"))
    ap.add_argument("--config-dir", default=os.path.join(paths.GPS_DIR, "configs", "agent_max"))
    ap.add_argument("--served-context", type=int, default=16384, help="max-model-len of the served model during the runs")
    ap.add_argument("--tags", nargs="+", default=["agent_qwen3max", "agent_qwen3maxcritic"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default=os.path.join(paths.GPS_DIR, "reports", "icde", "agent_ceiling.md"))
    ap.add_argument("--decision", default="", help="free text: what was decided for the paper after the rule's verdict (recorded under section 8)")
    args = ap.parse_args()
    cohort = pd.read_csv(args.cohort)
    fold_of = cohort.set_index("episode_id").fold
    res_dir = os.path.join(paths.BENCH, "agent_logs", args.run, "results")
    L = ["# Agent ceiling exploration (C-max): what a less constrained single-controller agent reaches", "",
         "Generated by `scripts/agent_ceiling_report.py` (do not edit). **Exploratory, not part of the frozen protocol**: the configuration was",
         "specified after the frozen study (`reports/icde/agent_protocol_v1.md`, results in `agent_results.md`) from validation data, then run once on the",
         "same 659-window cohort. Every variant is reported; failures stay in the denominator (ADE +inf, route 0); rates pool the nominal seeds and are",
         "bootstrapped by driver (1,000 resamples, 95 % percentile intervals); the frozen study's files were not touched.", ""]
    # 1 configuration
    L += ["## 1. Configuration", ""]
    man_p = os.path.join(res_dir, f"{args.tags[0]}_s{args.seeds[0]}.run_manifest.json")
    if os.path.exists(man_p):
        man = _load_json(man_p)
        cm = man.get("cache_manifest", {})
        L += [f"- Model `{man.get('model')}` revision `{man.get('revision')}`, bf16, vLLM, thinking mode with the model card's sampling values; served context "
              f"{args.served_context:,} tokens (the run manifests record the config hashes: budget `{man.get('budget_sha', '')[:12]}`, sampling `{man.get('sampling_sha', '')[:12]}`, "
              f"prompt `{man.get('prompt_sha', '')[:12]}`, schema `{man.get('schema_sha', '')[:12]}`, selector `{man.get('selector_sha', '')[:12]}`; tool-cache code sha `{str(cm.get('code_sha', ''))[:12]}`, "
              f"cache version {cm.get('cache_version')}).",
              f"- Reconstructor presets ({len(cm.get('tools', {}))}): " + ", ".join(sorted(cm.get("tools", {}))) + "; each may be run once per window; candidates come from the cache that "
              "replays `run_baseline.py` exactly (identity check in `agent_max_results.md`).",
              f"- Concurrency {man.get('concurrency')}; selector tool {'on' if man.get('selector_tool') else 'off'}; cohort sha `{man.get('cohort_sha', '')[:12]}`."]
    for name, lab in (("budget.json", "Controller budget"), ("critic_budget.json", "Critic budget")):
        p = os.path.join(args.config_dir, name)
        if os.path.exists(p):
            b = _load_json(p)
            L.append(f"- {lab}: {b.get('max_decisions')} turns, {b.get('max_tool_calls')} tool calls plus {b.get('reserved_submit_calls', 1)} reserved submit, "
                     f"{b.get('total_completion_tokens'):,} generated tokens (reserve {b.get('reserve_tokens')}, per turn {b.get('per_turn_max_tokens'):,}), virtual clock {b.get('timeout_virtual_s'):.0f} s; "
                     f"forced final submit with thinking {'off' if b.get('forced_submit_disable_thinking') else 'on'}.")
    p = os.path.join(args.config_dir, "selector_max.json")
    if os.path.exists(p):
        s = _load_json(p)
        feats = s.get("features") or []
        L.append(f"- B-max selector: logistic model over {len(feats)} observable features (`configs/agent_max/selector_max.json`; fitted on the cached validation windows, "
                 f"validation cross-fit in `reports/icde/tables/agent_selector_max_val.csv`).")
    p = os.path.join(args.config_dir, "tools_schema.json")
    if os.path.exists(p):
        names = [t["function"]["name"] for t in _load_json(p)]
        L.append(f"- Tools ({len(names)}): " + ", ".join(f"`{x}`" for x in names) + "; the controller sees only candidates of reconstructors it has run and may submit only those.")
    L.append("- Prompt: `configs/agent_max/system_prompt.txt` with calibration hints computed on validation candidates only (`reports/icde/tables/agent_max_val_hints.csv`); "
             "critic prompt `configs/agent_max/critic_prompt.txt`.")
    L.append("")
    # 2 results
    tab_p = os.path.join(TABLES, "agent_max_w05.csv")
    tab = pd.read_csv(tab_p) if os.path.exists(tab_p) else None
    pairs_p = os.path.join(TABLES, "agent_max_pairs_w05.csv")
    pairs = pd.read_csv(pairs_p) if os.path.exists(pairs_p) else None
    if tab is not None:
        L += ["## 2. Results on the 659-window cohort (one five-minute window per held-out drive)", ""]
        order = ["beam", "selector", "agent_prev", "agent", "agent_s0", "agent_s1", "agent_s2", "agent_fb", "agent_qwen3max_vote", "agent_qwen3maxcritic_s0",
                 "agent_qwen3maxcritic_s1", "agent_qwen3maxcritic_s2", "agent_qwen3maxcritic_vote", "oracle"]
        for fold in ["test", "few_shot_heldout", "all"]:
            t = tab[tab.fold == fold]
            if t.empty:
                continue
            L += [f"### {FOLD_LABEL.get(fold, fold)} ({int(t.iloc[0].n)} windows, {int(t.iloc[0].n_drivers)} drivers)", "",
                  "| system | route % [CI] | median ADE m [CI] | no output % | cost s (median, reconstructed serial) | calls | decisions | gen. tokens | same as B-max % |",
                  "|---|---|---|---|---|---|---|---|---|"]
            for key in order + [r for r in t.row if r not in order]:
                r = t[t.row == key]
                if r.empty:
                    continue
                r = r.iloc[0]
                g = lambda k, nd=1: _f(r[k], nd) if k in r and pd.notna(r[k]) else ""  # noqa: E731
                L.append(f"| {ROW_LABEL.get(key, key)} | {r.route_pct:.1f} [{r.route_lo:.1f}, {r.route_hi:.1f}] | {_f(r.ade_median, 0)} [{_f(r.ade_lo, 0)}, {_f(r.ade_hi, 0)}] | "
                         f"{r.noout_pct:.1f} | {g('cost_reconstructed_med')} | {g('tool_calls_mean')} | {g('decisions_mean')} | {g('gen_tokens_mean', 0)} | {g('same_sel_pct', 0)} |")
            L.append("")
    if pairs is not None and len(pairs):
        L += ["## 3. Paired route-recovery differences (percentage points, same windows, driver-clustered CIs; seeds aggregated inside every resample)", "",
              "| pair | fold | route diff [CI] | per seed | median ADE diff on common outputs [CI] |", "|---|---|---|---|---|"]
        for r in pairs.itertuples():
            seeds_r = ", ".join(f"{getattr(r, f'route_diff_s{s}_pp'):+.1f}" for s in (0, 1, 2) if hasattr(r, f"route_diff_s{s}_pp") and pd.notna(getattr(r, f"route_diff_s{s}_pp")))
            L.append(f"| {PAIR_LABEL.get(r.pair, r.pair)} | {FOLD_LABEL.get(r.fold, r.fold)} | {r.route_diff_pp:+.1f} [{r.route_lo_pp:+.1f}, {r.route_hi_pp:+.1f}] | {seeds_r} | "
                     f"{_f(r.ade_med_diff)} [{_f(r.ade_lo)}, {_f(r.ade_hi)}] |")
        L.append("")
    # 4 behaviour from the result CSVs
    scores = pool_scores(args.run)
    sel_p = os.path.join(res_dir, "selector_max.csv")
    sel = pd.read_csv(sel_p).set_index("episode_id").choice if os.path.exists(sel_p) else None
    L += ["## 4. Controller behaviour (result CSVs under `agent_logs/" + args.run + "/results/`)", ""]
    beh = []
    for tag in args.tags:
        for s in args.seeds:
            p = os.path.join(res_dir, f"{tag}_s{s}.csv")
            if not os.path.exists(p):
                continue
            t = pd.read_csv(p)
            st = t.status.value_counts()
            tools = t.choice_tool.fillna("none").value_counts(normalize=True) * 100
            top_tools = ", ".join(f"{k} {v:.0f}" for k, v in tools.head(5).items())
            run_cov = t.tools_run.fillna("").str.split("|").explode()
            run_cov = run_cov[run_cov != ""].value_counts()
            cov = ", ".join(f"{k} {100 * v / len(t):.0f}" for k, v in run_cov.head(10).items())
            same_sel = 100 * float((t.set_index("episode_id").choice == sel.reindex(t.episode_id).to_numpy()).mean()) if sel is not None else np.nan
            same_b1 = 100 * float((t.choice == "beam-1").mean())
            hit = np.nan
            if scores:
                h = [scores.get((e, c), (0.0, np.inf))[0] if isinstance(c, str) else 0.0 for e, c in zip(t.episode_id, t.choice)]
                hit = 100 * float(np.mean(h))
            beh.append(dict(tag=tag, seed=s, n=len(t), submitted=int(st.get("submitted", 0)), other=", ".join(f"{k} {v}" for k, v in st.items() if k != "submitted") or "none",
                            dec_med=t.n_decisions.median(), dec_mean=t.n_decisions.mean(), calls_med=t.n_tool_calls.median(), calls_mean=t.n_tool_calls.mean(),
                            tok_med=t.completion_tokens.median(), reason_share=100 * t.reasoning_tokens.sum() / max(t.completion_tokens.sum(), 1),
                            prompt_med=t.prompt_tokens.median(), wall_med=t.llm_wall_s.median(), cost_med=t.cost_reconstructed_s.median(), vis_med=t.n_candidates_visible.median(),
                            trunc=100 * float((t.truncations > 0).mean()), forced=100 * float(t.forced_submit.astype(bool).mean()) if "forced_submit" in t else np.nan,
                            top_tools=top_tools, cov=cov, same_sel=same_sel, same_b1=same_b1, hit=hit,
                            overturn=100 * float(t.critic_overturned.fillna(False).astype(bool).mean()) if t.critic_overturned.notna().any() else np.nan))
    if beh:
        L += ["| tag | seed | windows | submitted | other statuses | decisions (median / mean) | tool calls (median / mean) | generated tokens (median) | thinking share % | "
              "prompt tokens per window (median, cumulative over turns) | LLM wall s (median) | reconstructed cost s (median) | visible candidates (median) | truncated % | forced submit % | "
              "same as B-max % | same as beam Top-1 % | route rate of the choice % | critic overturned % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for b in beh:
            L.append(f"| {b['tag']} | {b['seed']} | {b['n']} | {b['submitted']} | {b['other']} | {b['dec_med']:.0f} / {b['dec_mean']:.1f} | {b['calls_med']:.0f} / {b['calls_mean']:.1f} | "
                     f"{b['tok_med']:,.0f} | {b['reason_share']:.0f} | {b['prompt_med']:,.0f} | {b['wall_med']:.0f} | {b['cost_med']:.0f} | {b['vis_med']:.0f} | {b['trunc']:.1f} | "
                     f"{_f(b['forced'])} | {_f(b['same_sel'], 0)} | {b['same_b1']:.0f} | {_f(b['hit'])} | {_f(b['overturn'])} |")
        L += ["", "Chosen reconstructor (share of windows, %) and reconstructors run (share of windows in which the tool was called, %):", ""]
        for b in beh:
            L.append(f"- {b['tag']} seed {b['seed']}: chosen {b['top_tools']}; run {b['cov']}.")
        L.append("")
    # 5 critic: did the overturns help?
    for tag in [t for t in args.tags if "critic" in t]:
        for s in args.seeds:
            d = os.path.join(paths.BENCH, "agent_logs", args.run, f"{tag}_s{s}")
            files = sorted(glob.glob(os.path.join(d, "*.jsonl")))
            if not files or not scores:
                continue
            helped = hurt = neutral = unknown = total = 0
            for fpath in files:
                with open(fpath) as f:
                    rec = json.loads(f.readline())
                cr = rec.get("critic") or {}
                if not cr.get("overturned"):
                    continue
                total += 1
                before, after = submit_before_critic(rec), rec.get("choice")
                e = rec["episode_id"]
                if before is None or (e, before) not in scores or (e, after) not in scores:
                    unknown += 1
                    continue
                rb, ra = scores[(e, before)][0], scores[(e, after)][0]
                helped += int(ra > rb); hurt += int(ra < rb); neutral += int(ra == rb)
            L += [f"## 5. Critic overturns ({tag}, seed {s})", "",
                  f"{total} of {len(files)} windows were overturned by the critic: the new choice recovered the route where the controller's did not in {helped}, "
                  f"lost it in {hurt}, same route outcome in {neutral}" + (f", undetermined {unknown}" if unknown else "") + ".", ""]
    # 6 vote agreement
    for tag in args.tags:
        p = os.path.join(res_dir, f"{tag}_vote.csv")
        if os.path.exists(p):
            v = pd.read_csv(p)
            sub = v[v.status == "submitted"]
            L += [f"## 6. Seed vote ({tag})", "",
                  f"{len(sub)} windows with a vote over {int(sub.n_seeds.max()) if len(sub) else 0} seeds: unanimous in {100 * float((sub.votes == sub.n_seeds).mean()):.0f} %, "
                  f"a 2-of-3 majority in {100 * float((sub.votes == 2).mean()):.0f} %, ties broken by the selector's probability in {100 * float((sub.votes == 1).mean()):.0f} %.", ""]
    # 7 selector-max validation and hints
    p = os.path.join(TABLES, "agent_selector_max_val.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        L += ["## 7. B-max on the cached validation windows (cross-fitted by driver)", "", "| row | route % [CI] | windows |", "|---|---|---|"]
        L += [f"| {r.row} | {r.route_pct:.1f} [{r.route_lo:.1f}, {r.route_hi:.1f}] | {int(r.n_windows)} |" for r in t.itertuples()]
        L += ["", "Calibration hints shown to the controller: `reports/icde/tables/agent_max_val_hints.csv` (validation candidates only).", ""]
    # 8 decision rule
    L += ["## 8. Pre-specified rule for the paper", "",
          "Stated before the three seeds, the critic and the vote had finished: the candidate for the main text is the single controller with three seeds",
          "(row `agent`); critic and vote stay in the supplement as variants. It replaces the frozen configuration in Section VI-F only if its seed-pooled",
          "route recovery is at least the frozen configuration's on both folds; if it is higher on one fold and lower on the other, the main text keeps the",
          "frozen study and gains one sentence pointing to this exploration in the supplement. Interpretation follows the stats wording rules (a CI that",
          "includes 0 is 'no clear evidence').", ""]
    if tab is not None and "agent_prev" in set(tab.row):
        out = []
        for fold in ["test", "few_shot_heldout", "all"]:
            a = tab[(tab.row == "agent") & (tab.fold == fold)]
            b = tab[(tab.row == "agent_prev") & (tab.fold == fold)]
            if len(a) and len(b):
                out.append((fold, float(a.iloc[0].route_pct), float(b.iloc[0].route_pct)))
        if out:
            L += ["| fold | C-max route % | frozen C route % | difference pp |", "|---|---|---|---|"]
            L += [f"| {FOLD_LABEL.get(f, f)} | {x:.1f} | {y:.1f} | {x - y:+.1f} |" for f, x, y in out]
            per_fold = [(x, y) for f, x, y in out if f != "all"]
            verdict = "C-max is at least the frozen configuration on both folds: the rule says update the main text." if per_fold and all(x >= y for x, y in per_fold) else \
                      "C-max is not at least the frozen configuration on both folds: the rule says the main text keeps the frozen study."
            if pairs is not None and len(pairs):
                pv = pairs[(pairs.pair == "agent_vs_prev")]
                ci = "; ".join(f"{FOLD_LABEL.get(r.fold, r.fold)} {r.route_diff_pp:+.1f} [{r.route_lo_pp:+.1f}, {r.route_hi_pp:+.1f}]" for r in pv.itertuples())
                if ci:
                    verdict += f" Paired C-max minus frozen C (pp): {ci}."
            L += ["", verdict, ""]
    if args.decision:
        L += [f"**Decision taken:** {args.decision}", ""]
    L += ["## 9. Files", "",
          "- Tables: `reports/icde/tables/agent_max_w05.csv`, `agent_max_pairs_w05.csv`, `agent_max_status_w05.csv`; report `reports/icde/agent_max_results.md`; LaTeX `paper_icde/tables/agent_max.tex`.",
          f"- Results and conversation logs: `$CANTRAJ_BENCH/agent_logs/{args.run}/`; predictions `$CANTRAJ_BENCH/preds/{args.run}/F-M-Hp/`; scores `$CANTRAJ_BENCH/scores/{args.run}/`.",
          "- Chain: `scripts/run_icde_agent_max.sh {pilot|selector|llm <seed>|critic <seed>|vote|score|tables}`; configuration `configs/agent_max/`.", ""]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(L))
    print(f"report -> {args.out} ({len(L)} lines)")


if __name__ == "__main__":
    main()
