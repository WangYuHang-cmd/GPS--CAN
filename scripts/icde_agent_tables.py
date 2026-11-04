#!/usr/bin/env python3
"""Agent study tables (supplementary cohort; one five-minute window per held-out drive), failure-aware.

Rows per fold: A = map beam (icde_v11 output re-aggregated on the cohort), B = supervised selector, C = Qwen agent
(three nominal seeds; rates pooled over seeds and bootstrapped by driver, failure-aware ADE medians averaged over seeds
inside every resample), per-seed agent rows, C with the pre-fixed selector fallback, and the evaluator-side pool
oracle. Costs: reconstructed serial cost (LLM wall + logical tool time + tool execution), cached-mode elapsed, LLM
wall; tool calls, generated and reasoning tokens, truncations, status shares, tool-use shares, fallback share.
Pairs: C - B (primary) and C - A route-recovery differences, unconditional, driver-clustered; ADE paired medians per
seed and aggregated inside the resample (windows with output on both sides).

    python scripts/icde_agent_tables.py --run icde_agent --cohort manifests/icde_agent_cohort.csv --n-boot 1000
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths, stats  # noqa: E402

OUT = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
TEX = os.path.join(paths.GPS_DIR, "paper_icde", "tables")
PROTO = "F-M-Hp"
FOLDS = [("test", "test"), ("few_shot_heldout", "few-shot")]
FOLD_LABEL = {"test": "test", "few_shot_heldout": "few-shot", "val": "validation"}
A_STEM, A_RUN = "map_beam_steer__G3", "icde_v11"
B_STEM = "selector__G3"


def values(ep, stem, ids, col):
    """Per-episode values for `ids` (failure-aware: ADE +inf, rates 0 when the stem has no output)."""
    g = ep[(ep.method == stem) & (ep.protocol == PROTO)].drop_duplicates("episode_id").set_index("episode_id")
    v = g[col].reindex(ids).to_numpy(dtype=float)
    return np.where(np.isnan(v), np.inf if col == "ade" else 0.0, v)


def rate_row(route_mat, ade_mat, noout_mat, fde_mat, drivers, n_boot):
    """route_mat etc: (S, N) matrices; rates pooled over seeds (driver resampling), ADE through seed_median."""
    S_, N = route_mat.shape
    tiled = np.tile(drivers, S_)
    r, (rlo, rhi) = stats.mean_ci(route_mat.ravel(), tiled, n_boot=n_boot)
    med, (mlo, mhi) = stats.seed_median(ade_mat, drivers, n_boot=n_boot)
    return dict(n=N, n_drivers=int(len(np.unique(drivers))), route_pct=100 * r, route_lo=100 * rlo, route_hi=100 * rhi, ade_median=med, ade_lo=mlo, ade_hi=mhi,
                noout_pct=100 * float(noout_mat.mean()), fde500_pct=100 * float(fde_mat.mean()), n_seeds=S_)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="icde_agent")
    ap.add_argument("--cohort", default=os.path.join(paths.MANIFESTS, "icde_agent_cohort.csv"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--folds", nargs="+", default=["test", "few_shot_heldout"])
    args = ap.parse_args()
    global FOLDS
    FOLDS = [(f, FOLD_LABEL.get(f, f)) for f in args.folds]
    cohort = pd.read_csv(args.cohort)
    cols = ["episode_id", "method", "protocol", "ade", "route_ok", "fde_gt_500"]
    ep = pd.read_parquet(os.path.join(paths.SCORES, args.run, "episodes_all.parquet"), columns=cols)
    epA = pd.read_parquet(os.path.join(paths.SCORES, A_RUN, "episodes_all.parquet"), columns=cols)
    epA = epA[(epA.method == A_STEM) & (epA.protocol == PROTO) & epA.episode_id.isin(set(cohort.episode_id))]
    res_dir = os.path.join(paths.BENCH, "agent_logs", args.run, "results")
    res = {s: pd.read_csv(os.path.join(res_dir, f"agent_qwen3_s{s}.csv")) for s in args.seeds if os.path.exists(os.path.join(res_dir, f"agent_qwen3_s{s}.csv"))}
    sel_res = pd.read_csv(os.path.join(res_dir, "selector.csv")) if os.path.exists(os.path.join(res_dir, "selector.csv")) else None
    tm_beam = os.path.join(paths.PREDS, args.run, PROTO, "tool_beam__G3.part0.timing.csv")
    tm_beam = pd.read_csv(tm_beam).set_index("key").secs if os.path.exists(tm_beam) else None
    pool = ep[ep.method.str.startswith("poolcand_") & (ep.protocol == PROTO)]
    pool_route = pool.groupby("episode_id").route_ok.max()
    pool_ade = pool.groupby("episode_id").ade.min()
    rows, pairs, status_rows = [], [], []
    # identity check A (icde_v11) vs the cached tool beam
    chk = pd.DataFrame()
    if "tool_beam__G3" in set(ep.method):
        a = epA.set_index("episode_id").ade
        b = ep[(ep.method == "tool_beam__G3")].set_index("episode_id").ade
        common = a.index.intersection(b.index)
        chk = pd.DataFrame(dict(n_common=[len(common)], max_abs_ade_diff=[float(np.max(np.abs(a.loc[common] - b.loc[common]))) if len(common) else np.nan],
                                n_only_v11=[len(a.index.difference(b.index))], n_only_tool=[len(b.index.difference(a.index))]))
        chk.to_csv(os.path.join(OUT, "agent_identity_check.csv"), index=False)
    for fold, flab in FOLDS:
        c = cohort[cohort.fold == fold]
        ids, drivers = list(c.episode_id), c.driver.to_numpy()
        if not ids:
            continue

        def mats(ep_, stem):
            return (values(ep_, stem, ids, "route_ok")[None], values(ep_, stem, ids, "ade")[None],
                    (~np.isfinite(values(ep_, stem, ids, "ade")))[None], values(ep_, stem, ids, "fde_gt_500")[None])

        # A
        rA, aA, nA, fA = mats(epA, A_STEM)
        row = dict(row="beam", fold=fold, label="map beam (A)", **rate_row(rA, aA, nA, fA, drivers, args.n_boot))
        if tm_beam is not None:
            row["cost_reconstructed_med"] = float(np.median(tm_beam.reindex(ids).dropna())) if tm_beam.reindex(ids).notna().any() else np.nan
        rows.append(row)
        # B
        rB, aB, nB, fB = mats(ep, B_STEM)
        row = dict(row="selector", fold=fold, label="supervised selector (B)", **rate_row(rB, aB, nB, fB, drivers, args.n_boot))
        if sel_res is not None:
            s = sel_res.set_index("episode_id").reindex(ids)
            row["cost_reconstructed_med"] = float(s.cost_reconstructed_s.median())
            row["tool_calls_mean"] = 3.0
        rows.append(row)
        # C per seed and pooled
        R, A_, N_, F_, Rfb, Afb, Nfb, Ffb = [], [], [], [], [], [], [], []
        for s in args.seeds:
            stem = f"agent_qwen3_s{s}__G3"
            r_, a_, n_, f_ = mats(ep, stem)
            R.append(r_[0]); A_.append(a_[0]); N_.append(n_[0]); F_.append(f_[0])
            rf, af, nf, ff = mats(ep, f"agent_qwen3_fb_s{s}__G3")
            Rfb.append(rf[0]); Afb.append(af[0]); Nfb.append(nf[0]); Ffb.append(ff[0])
            row = dict(row=f"agent_s{s}", fold=fold, label=f"Qwen agent, seed {s}", **rate_row(r_, a_, n_, f_, drivers, args.n_boot))
            if s in res:
                rs = res[s].set_index("episode_id").reindex(ids)
                if sel_res is not None:
                    sc = sel_res.set_index("episode_id").reindex(ids).choice
                    row["same_sel_pct"] = 100 * float((rs.choice == sc).mean())
                row["same_beam1_pct"] = 100 * float((rs.choice == "beam-1").mean())
                row.update(cost_reconstructed_med=float(rs.cost_reconstructed_s.median()), elapsed_cached_med=float(rs.elapsed_s.median()),
                           llm_wall_med=float(rs.llm_wall_s.median()), tool_calls_mean=float(rs.n_tool_calls.mean()), decisions_mean=float(rs.n_decisions.mean()),
                           gen_tokens_mean=float(rs.completion_tokens.mean()), reasoning_tokens_mean=float(rs.reasoning_tokens.mean()) if rs.reasoning_tokens.notna().any() else np.nan,
                           truncation_pct=100 * float((rs.truncations > 0).mean()), fallback_pct=100 * float((rs.status != "submitted").mean()),
                           **{f"tool_use_{t}_pct": 100 * float(rs.tools_run.fillna("").str.contains(t).mean()) for t in ("beam", "pf", "ep")},
                           **{f"status_{k}_pct": 100 * float((rs.status == k).mean()) for k in ("submitted", "invalid_choice", "total_token_budget", "no_submit", "timeout", "parse_error", "llm_error", "no_candidates")})
                for k, v in rs.status.value_counts().items():
                    status_rows.append(dict(fold=fold, seed=s, status=k, n=int(v), pct=100 * v / len(rs)))
            rows.append(row)
        R, A_, N_, F_ = (np.stack(x) for x in (R, A_, N_, F_))
        row = dict(row="agent", fold=fold, label="Qwen agent (C), seeds pooled", **rate_row(R, A_, N_, F_, drivers, args.n_boot))
        seed_rows = [r for r in rows if r["fold"] == fold and r["row"].startswith("agent_s")]
        for k in ("cost_reconstructed_med", "elapsed_cached_med", "llm_wall_med", "tool_calls_mean", "decisions_mean", "gen_tokens_mean", "reasoning_tokens_mean",
                  "truncation_pct", "fallback_pct", "tool_use_beam_pct", "tool_use_pf_pct", "tool_use_ep_pct", "same_sel_pct", "same_beam1_pct"):
            vals = [r[k] for r in seed_rows if k in r and np.isfinite(r[k])]
            if vals:
                row[k] = float(np.mean(vals))
        row["route_seed_min"], row["route_seed_max"] = 100 * float(R.mean(axis=1).min()), 100 * float(R.mean(axis=1).max())
        rows.append(row)
        Rfb, Afb, Nfb, Ffb = (np.stack(x) for x in (Rfb, Afb, Nfb, Ffb))
        rows.append(dict(row="agent_fb", fold=fold, label="Qwen agent with selector fallback", **rate_row(Rfb, Afb, Nfb, Ffb, drivers, args.n_boot)))
        # oracle
        ro = pool_route.reindex(ids).fillna(0).to_numpy()[None]
        ao = pool_ade.reindex(ids).to_numpy(); ao = np.where(np.isnan(ao), np.inf, ao)[None]
        rows.append(dict(row="oracle", fold=fold, label="pool oracle (diagnostic)", **rate_row(ro, ao, ~np.isfinite(ao), np.zeros_like(ro), drivers, args.n_boot)))
        # pairs: C - B and C - A route (unconditional), ADE paired on common outputs
        for other, (rO, aO) in (("selector", (rB, aB)), ("beam", (rA, aA))):
            d = R - rO                                                   # (S, N)
            m, (lo, hi) = stats.mean_ci(d.ravel(), np.tile(drivers, len(args.seeds)), n_boot=args.n_boot)
            pr = dict(pair=f"agent_vs_{other}", fold=fold, n=len(ids), n_drivers=int(len(np.unique(drivers))), route_diff_pp=100 * m, route_lo_pp=100 * lo, route_hi_pp=100 * hi)
            for i, s in enumerate(args.seeds):
                ms, (ls, hs) = stats.mean_ci(d[i], drivers, n_boot=args.n_boot)
                pr[f"route_diff_s{s}_pp"] = 100 * ms
                pd_ = stats.paired(A_[i], aO[0], drivers, n_boot=args.n_boot)
                pr[f"ade_med_diff_s{s}"], pr[f"ade_lo_s{s}"], pr[f"ade_hi_s{s}"], pr[f"n_out_s{s}"] = pd_.get("median"), pd_.get("median_lo"), pd_.get("median_hi"), pd_.get("n", 0)
            diff = np.where(np.isfinite(A_) & np.isfinite(aO), A_ - aO, np.nan)
            sp = stats.seed_paired(diff, drivers, n_boot=args.n_boot)
            pr.update(ade_med_diff=sp["median"], ade_lo=sp["lo"], ade_hi=sp["hi"])
            pairs.append(pr)
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit("no rows: check --folds against the cohort manifest")
    os.makedirs(OUT, exist_ok=True)
    tab.to_csv(os.path.join(OUT, "agent_w05.csv"), index=False)
    pd.DataFrame(pairs).to_csv(os.path.join(OUT, "agent_pairs_w05.csv"), index=False)
    if status_rows:
        pd.DataFrame(status_rows).to_csv(os.path.join(OUT, "agent_status_w05.csv"), index=False)
    write_tex(tab)
    write_report(tab, pd.DataFrame(pairs), pd.DataFrame(status_rows) if status_rows else None, chk)
    show = ["row", "fold", "n", "route_pct", "route_lo", "route_hi", "ade_median", "noout_pct", "cost_reconstructed_med", "tool_calls_mean", "gen_tokens_mean"]
    print(tab[[c for c in show if c in tab]].round(1).to_string(index=False))
    print(pd.DataFrame(pairs)[["pair", "fold", "route_diff_pp", "route_lo_pp", "route_hi_pp", "ade_med_diff", "ade_lo", "ade_hi"]].round(1).to_string(index=False))
    if len(chk):
        print("identity check (icde_v11 beam vs cached tool beam):", chk.to_dict("records")[0])


def _isfin(x):
    try:
        return x is not None and np.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _f(x, nd=1):
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(xf):
        return "fail" if xf == np.inf else "n/a"
    return f"{xf:,.{nd}f}"


def write_report(tab, pairs, status, chk):
    """Reproduction report (everything that does not fit the paper): per-seed rows, fallback, oracle, status shares, tool use,
    cost quantities, identity and live checks, selector validation."""
    rep = os.path.join(paths.GPS_DIR, "reports", "icde", "agent_results.md")
    L = ["# Agent study results (supplementary cohort; generated by scripts/icde_agent_tables.py, do not edit)", ""]
    L += ["Failure-aware: windows without a valid output count with ADE +inf and route 0. Rates pool the three nominal seeds and are",
          "bootstrapped by driver (1,000 resamples, 95 % percentile intervals); ADE medians average the per-seed medians inside every resample.", ""]
    for fold, flab in FOLDS:
        t = tab[tab.fold == fold]
        if t.empty:
            continue
        L += [f"## {flab} fold ({int(t.iloc[0].n)} windows, {int(t.iloc[0].n_drivers)} drivers)", "",
              "| row | route % [CI] | median ADE m [CI] | no output % | end > 500 m % | cost s (reconstructed) | elapsed s (cached) | LLM s | calls | gen. tokens | reasoning tokens | truncated % | fallback % | use beam/pf/ep % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in t.itertuples():
            g = lambda k, nd=1: _f(getattr(r, k, np.nan), nd) if hasattr(r, k) else "" # noqa: E731
            use = f"{g('tool_use_beam_pct', 0)}/{g('tool_use_pf_pct', 0)}/{g('tool_use_ep_pct', 0)}" if hasattr(r, "tool_use_beam_pct") else ""
            L.append(f"| {r.label} | {r.route_pct:.1f} [{r.route_lo:.1f}, {r.route_hi:.1f}] | {_f(r.ade_median, 0)} [{_f(r.ade_lo, 0)}, {_f(r.ade_hi, 0)}] | {r.noout_pct:.1f} | {r.fde500_pct:.1f} | "
                     f"{g('cost_reconstructed_med')} | {g('elapsed_cached_med')} | {g('llm_wall_med')} | {g('tool_calls_mean')} | {g('gen_tokens_mean', 0)} | {g('reasoning_tokens_mean', 0)} | {g('truncation_pct')} | {g('fallback_pct')} | {use} |")
        L.append("")
        pr = pairs[pairs.fold == fold] if len(pairs) else pairs
        if len(pr):
            L += ["Paired differences (route: unconditional, percentage points; ADE: median difference on windows with output on both sides, m):", "",
                  "| pair | route diff [CI] | per seed | ADE diff [CI] (seeds aggregated) | per seed ADE diff [CI], n |", "|---|---|---|---|---|"]
            for r in pr.itertuples():
                seeds_r = ", ".join(f"{getattr(r, f'route_diff_s{s}_pp'):+.1f}" for s in (0, 1, 2) if _isfin(getattr(r, f"route_diff_s{s}_pp", None)))
                seeds_a = "; ".join(f"{_f(getattr(r, f'ade_med_diff_s{s}'))} [{_f(getattr(r, f'ade_lo_s{s}'))}, {_f(getattr(r, f'ade_hi_s{s}'))}], n {int(getattr(r, f'n_out_s{s}', 0) or 0)}"
                                    for s in (0, 1, 2) if _isfin(getattr(r, f"ade_med_diff_s{s}", None)))
                L.append(f"| {r.pair} | {r.route_diff_pp:+.1f} [{r.route_lo_pp:+.1f}, {r.route_hi_pp:+.1f}] | {seeds_r} | {_f(r.ade_med_diff)} [{_f(r.ade_lo)}, {_f(r.ade_hi)}] | {seeds_a} |")
            L.append("")
        if status is not None and len(status):
            st = status[status.fold == fold]
            if len(st):
                L += ["Agent status shares (per seed):", ""]
                piv = st.pivot_table(index="status", columns="seed", values="pct", aggfunc="first").fillna(0.0)
                L += ["| status | " + " | ".join(f"seed {c}" for c in piv.columns) + " |", "|---|" + "---|" * len(piv.columns)]
                for idx, row in piv.iterrows():
                    L.append(f"| {idx} | " + " | ".join(f"{v:.1f}" for v in row.values) + " |")
                L.append("")
    sel = os.path.join(OUT, "agent_selector_val.csv")
    if os.path.exists(sel):
        t = pd.read_csv(sel)
        L += ["## Selector on the validation cohort (cross-fitted by driver)", "", "| row | route % [CI] | n |", "|---|---|---|"]
        L += [f"| {r.row} | {r.route_pct:.1f} [{r.route_lo:.1f}, {r.route_hi:.1f}] | {int(r.n_windows)} |" for r in t.itertuples()]
        L.append("")
    if chk is not None and len(chk):
        r = chk.iloc[0]
        L += ["## Identity check", "", f"icde_v11 map_beam_steer__G3 vs the cached tool beam: {int(r.n_common)} common windows, max |ADE difference| {r.max_abs_ade_diff:.2e}; "
              f"only in icde_v11 {int(r.n_only_v11)}, only in the cache {int(r.n_only_tool)}.", ""]
    lv = os.path.join(OUT, "agent_live_check.csv")
    if os.path.exists(lv):
        r = pd.read_csv(lv).iloc[0]
        L += ["## Live (uncached) cost check", "", f"{int(r.n)} pilot windows, seed 0, concurrency 1, every reconstructor call executed for real (load, graph, search): "
              f"median end-to-end {r.live_med_s:.1f} s against a reconstructed serial cost of {r.reconstructed_med_s:.1f} s on the same windows "
              f"(median ratio {r.ratio_median:.2f}, 90th percentile {r.ratio_p90:.2f}); same final choice in {r.same_choice_pct:.0f} % of the windows.", ""]
    pil = os.path.join(OUT, "agent_pilot_summary.csv")
    if os.path.exists(pil):
        L += ["## Pilot summary", "", "See `reports/icde/tables/agent_pilot_summary.csv` (validation pilot, 40 windows x 3 seeds).", ""]
    with open(rep, "w") as f:
        f.write("\n".join(L))
    print(f"report -> {rep}")


def fmt_m(x):
    if not np.isfinite(x):
        return "fail"
    return f"{x:,.0f}".replace(",", "{,}")


def write_tex(tab):
    os.makedirs(TEX, exist_ok=True)
    if tab.empty:
        print("no rows: no LaTeX table written")
        return
    lines = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}",
             r"\caption{Agent study, supplementary cohort (one five-minute window per held-out drive): beam (A), supervised selector (B), "
             r"Qwen3-14B agent (C, three seeds pooled); route recovery, failure-aware median ADE, median reconstructed serial cost; "
             r"95\,\% driver-clustered CIs; every window produced an output.}\label{tab:agent}",
             r"\begin{tabular}{llrrrr}\toprule",
             r"fold & method & $n$ & route \% [CI] & ADE m [CI] & cost s \\\midrule"]
    labels = {"beam": "beam (A)", "selector": "selector (B)", "agent": "agent (C)"}
    for fold, flab in FOLDS:
        for key in ("beam", "selector", "agent"):
            r = tab[(tab.row == key) & (tab.fold == fold)]
            if r.empty:
                continue
            r = r.iloc[0]
            cost = r.get("cost_reconstructed_med", np.nan)
            lines.append(f"{flab} & {labels[key]} & {int(r.n)} & {r.route_pct:.0f} [{r.route_lo:.0f}, {r.route_hi:.0f}] & "
                         f"{fmt_m(r.ade_median)} [{fmt_m(r.ade_lo)}, {fmt_m(r.ade_hi)}] & {cost:.1f} \\\\")
        if fold == FOLDS[0][0]:
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    with open(os.path.join(TEX, "agent.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
