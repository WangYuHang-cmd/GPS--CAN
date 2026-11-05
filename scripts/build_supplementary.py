#!/usr/bin/env python3
"""Supplementary material (EAB): every table is generated from reports/icde/tables/*.csv (no hand-typed number) into
paper_icde/supplementary/tables/*.tex; the document paper_icde/supplementary/supp.tex inputs them and the paper's macros.

    python scripts/build_supplementary.py            # tables
    cd paper_icde/supplementary && pdflatex supp.tex && pdflatex supp.tex
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths  # noqa: E402

T = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
OUT = os.path.join(paths.GPS_DIR, "paper_icde", "supplementary", "tables")
FOLD = {"test": "test", "few_shot_heldout": "few-shot", "val": "validation", "all": "both folds"}
WIN = {"w05": "5 min", "w15": "15 min", "full": "whole drive", "w01": "1 min"}


def esc(s):
    return str(s).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def m(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    if np.isnan(x):
        return "--"
    if np.isinf(x):
        return "fail"
    return f"{x:,.{nd}f}".replace(",", "{,}")


def pct(x, nd=0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if np.isnan(x) else f"{100 * x:.{nd}f}"


def ci(a, lo, hi, nd=0):
    return f"{m(a, nd)} [{m(lo, nd)}, {m(hi, nd)}]"


def write(name, lines):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("  ", name)


def table(name, caption, label, header, rows, colspec, star=False, size="\\footnotesize", notes=None):
    env = "table*" if star else "table"
    L = [f"\\begin{{{env}}}[t]\\centering{size}\\setlength{{\\tabcolsep}}{{3pt}}", f"\\caption{{{caption}}}\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}\\toprule", " & ".join(header) + " \\\\\\midrule"]
    L += [" & ".join(r) + " \\\\" for r in rows]
    L += ["\\bottomrule\\end{tabular}"]
    if notes:
        L.append(f"\\par\\smallskip\\raggedright {notes}")
    L.append(f"\\end{{{env}}}")
    write(name, L)


def main_board():
    t = pd.read_csv(os.path.join(T, "main_board.csv"))
    for win in ("w05", "w15", "full"):
        rows = []
        sub = t[t.win == win]
        for (key, lab, proto), g in sub.groupby(["key", "label", "protocol"], sort=False):
            r = [esc(lab), proto]
            for fold in ("test", "few_shot_heldout"):
                x = g[g.fold == fold]
                if x.empty:
                    r += ["--"] * 4
                    continue
                x = x.iloc[0]
                r += [m(x.n_expected), pct(x.fail_rate, 1), ci(x.ade_median, x.ade_lo, x.ade_hi), pct(x.route_ok)]
            rows.append(r)
        table(f"main_{win}.tex", f"Complete board, {WIN[win]} windows, benchmark v1.1: expected windows $n$, failure share (no output), "
              f"failure-aware median ADE in metres with the 95\\,\\% driver-clustered CI, and the route recovery rate (path recall at 50\\,m $\\ge 0.8$), "
              f"test and few-shot drivers.", f"tab:supp_main_{win}",
              ["method", "protocol", "$n$", "fail \\%", "median ADE m [CI]", "route \\%", "$n$", "fail \\%", "median ADE m [CI]", "route \\%"], rows,
              "llrrrrrrrr", star=True, size="\\scriptsize",
              notes="Left block: test drivers; right block: few-shot drivers. Protocols O/F (online / offline), N/M (no map / map), Hp/Hm (start heading given / withheld).")


def cohort():
    t = pd.read_csv(os.path.join(T, "cohort_horizon.csv"))
    rows = []
    for (meth, proto), g in t.groupby(["method", "protocol"], sort=False):
        r = [esc(meth.split("__")[0]), proto]
        for fold in ("test", "few_shot_heldout"):
            for h in (60, 300, 900):
                x = g[(g.fold == fold) & (g.horizon_s == h)]
                r.append("--" if x.empty else f"{m(x.iloc[0].ade_median)} / {pct(x.iloc[0].route_ok)}")
        rows.append(r)
    table("cohort_horizon.tex", "Same-cohort horizon study: one start per drive, scored 1, 5 and 15 minutes after the start (median ADE in metres / route recovery \\%). "
          "Online methods on prefixes of their 15-minute runs; offline methods rerun on each prefix.", "tab:supp_cohort",
          ["method", "protocol", "1 min", "5 min", "15 min", "1 min", "5 min", "15 min"], rows, "llrrrrrr", star=True, size="\\scriptsize",
          notes="Left: test drives; right: few-shot drives.")


def signal():
    s = pd.read_csv(os.path.join(T, "signal_sets_w05.csv"))
    rows = [[esc(r.setting), esc(r.group), str(int(r.seed)), m(r.n), m(r.fail), ci(r.ade_median, r.lo, r.hi), pct(r.route_ok), pct(r.c100)] for r in s.itertuples()]
    table("signal_sets.tex", "Signal study (pedal-valid five-minute windows of the held-out drives): one learner, one decoder, per signal set and seed; "
          "failure-aware median ADE with 95\\,\\% driver-clustered CI, route recovery and 100\\,m coverage.", "tab:supp_signal_sets",
          ["setting", "set", "seed", "$n$", "fail", "median ADE m [CI]", "route \\%", "c100 \\%"], rows, "llrrrrrr", star=True, size="\\scriptsize")
    g = pd.read_csv(os.path.join(T, "signal_gains_w05.csv"))
    rows = [[esc(r.setting), esc(r.added), esc(r.a), esc(r.b), str(int(r.seed)), m(r.n), ci(r.median, r.median_lo, r.median_hi, 1), ci(r.mean, r.mean_lo, r.mean_hi), pct(r.win_a)] for r in g.itertuples()]
    table("signal_gains.tex", "Signal study, paired gains of adding a signal (a $-$ b, metres; negative = better): median and mean paired differences with 95\\,\\% CIs and the share of windows that improve.",
          "tab:supp_signal_gains", ["setting", "added", "a", "b", "seed", "$n$", "median diff [CI]", "mean diff [CI]", "improved \\%"], rows, "lllllrrrr", star=True, size="\\scriptsize")


def topk():
    t = pd.read_csv(os.path.join(T, "topk_w05.csv"))
    rows = [[esc(r.stem.split("__")[0]), r.protocol, FOLD.get(r.fold, r.fold), m(r.n_expected), m(r.n_output), m(r.n_single_path), m(r.top1_ade_median), m(r.top3_ade_median), m(r.top5_ade_median), m(r.top10_ade_median), pct(r.best_rank_is_1), m(r.n_candidates_median)] for r in t.itertuples()]
    table("topk.tex", "Candidate sets on five-minute windows: whole-path oracle median ADE among the first $K$ candidates, share of windows whose best candidate is ranked first, and the median number of candidates.",
          "tab:supp_topk", ["method", "protocol", "fold", "$n$", "out", "single", "$K$=1", "3", "5", "10", "best 1st \\%", "cands"], rows, "lllrrrrrrrrr", star=True, size="\\scriptsize")


def robustness():
    t = pd.read_csv(os.path.join(T, "robustness_map_w05.csv"))
    rows = []
    for r in t.itertuples():
        rows.append([esc(r.method), r.protocol, esc(r.perturbation), FOLD.get(r.fold, r.fold), m(r.n_expected), m(r.median_clean), m(r.median_pert), pct(r.route_clean), pct(r.route_pert),
                     f"{100 * r.route_diff:+.1f} [{100 * r.route_diff_lo:+.1f}, {100 * r.route_diff_hi:+.1f}]"])
    table("robustness_map.tex", "Fixed map attacks on degraded inputs (five-minute windows, grid fixed before the first run): median ADE and route recovery on the clean and the degraded signal, and the paired change of the route recovery rate in percentage points with 95\\,\\% driver-clustered CI.",
          "tab:supp_robust_map", ["method", "protocol", "degradation", "fold", "$n$", "med. clean", "med. degraded", "route clean \\%", "route degraded \\%", "$\\Delta$ route pp [CI]"], rows, "llllrrrrrr", star=True, size="\\scriptsize")
    t2 = pd.read_csv(os.path.join(T, "robustness_w05.csv"))
    rows = [[esc(r.method), esc(r.perturbation), FOLD.get(r.fold, r.fold), m(r.median_clean), m(r.median_perturbed), m(r.n), ci(r.median, r.median_lo, r.median_hi, 1), ci(r.mean, r.mean_lo, r.mean_hi)] for r in t2.itertuples()]
    table("robustness_dr.tex", "Degraded inputs for dead reckoning (no map): clean and degraded medians and the paired median / mean differences (metres).",
          "tab:supp_robust_dr", ["method", "degradation", "fold", "med. clean", "med. degraded", "$n$", "paired median [CI]", "paired mean [CI]"], rows, "lllrrrrr", star=True, size="\\scriptsize")


def cost():
    t = pd.read_csv(os.path.join(T, "cost.csv"))
    rows = [[esc(r.tag), m(r.n_w05), m(r.n_fail_w05), m(r.infer_median_s_w05, 2), m(r.infer_p90_s_w05, 2), m(r.n_w15), m(r.n_fail_w15), m(r.infer_median_s_w15, 2), m(r.infer_p90_s_w15, 2), m(r.peak_rss_gb, 1), m(r.graph_build_median_s, 1)] for r in t.itertuples()]
    table("cost.tex", "Cost study on one fixed validation subset, one worker per method: inference seconds per window (median and 90th percentile; failed searches count with the time they ran), peak memory and the per-drive graph build.",
          "tab:supp_cost", ["method", "$n_5$", "fail", "med s", "p90 s", "$n_{15}$", "fail", "med s", "p90 s", "RSS GB", "graph s"], rows, "lrrrrrrrrrr", star=True, size="\\scriptsize")
    t2 = pd.read_csv(os.path.join(T, "cost_accuracy.csv"))
    rows = [[esc(r.method), r.protocol, WIN.get(r.win, r.win), m(r.n), m(r.fail), m(r.ade_median), pct(r.route)] for r in t2.itertuples()]
    table("cost_accuracy.tex", "Accuracy on the cost subset at the published and at smaller search budgets (validation windows).", "tab:supp_cost_acc",
          ["method / budget", "protocol", "window", "$n$", "fail", "median ADE m", "route \\%"], rows, "lllrrrr", star=True, size="\\scriptsize")


def confidence():
    t = pd.read_csv(os.path.join(T, "confidence_w05_w15.csv"))
    s = t[t.kind == "summary"]
    rows = [[esc(r.stem.split("__")[0]), r.protocol, WIN.get(r.win, r.win), FOLD.get(r.fold, r.fold), m(r.n_expected), m(r.n_fail), ci(r.auroc, r.auroc_lo, r.auroc_hi, 2), m(r.spearman_ade, 2), pct(r.share_prob_eq_1)] for r in s.itertuples()]
    table("confidence.tex", "Candidate confidence score of the offline beam: AUROC of the Top-1 score for route recovery (driver-clustered CI), Spearman correlation with the ADE, and the share of windows whose score is 1.",
          "tab:supp_conf", ["method", "protocol", "window", "fold", "$n$", "fail", "AUROC [CI]", "Spearman", "score = 1 \\%"], rows, "llllrrrrr", star=True, size="\\scriptsize")
    th = t[(t.kind == "threshold") & (t.tau.isin([0.5, 0.7, 0.9]))]
    rows = [[esc(r.stem.split("__")[0]), r.protocol, WIN.get(r.win, r.win), FOLD.get(r.fold, r.fold), m(r.tau, 1), pct(r.coverage), ci(r.route_rate_accepted * 100 if pd.notna(r.route_rate_accepted) else np.nan, r.route_rate_lo * 100 if pd.notna(r.route_rate_lo) else np.nan, r.route_rate_hi * 100 if pd.notna(r.route_rate_hi) else np.nan), pct(r.error_share_accepted)] for r in th.itertuples()]
    table("confidence_thresholds.tex", "Selective prediction by the beam's confidence score: windows accepted at a threshold (coverage, all expected windows in the denominator), route recovery among the accepted and the share of accepted routes that are wrong.",
          "tab:supp_conf_thr", ["method", "protocol", "window", "fold", "$\\tau$", "coverage \\%", "route among accepted \\% [CI]", "wrong \\%"], rows, "llllrrrr", star=True, size="\\scriptsize")
    tx = pd.read_csv(os.path.join(T, "window_taxonomy_w05.csv"))
    rows = [[esc(r.stem.split("__")[0]), r.protocol, FOLD.get(r.fold, r.fold), m(r.threshold_m), m(r.n), pct(r.no_output), pct(r.no_candidate), pct(r.ranking_miss), pct(r.top1_ok)] for r in tx.itertuples()]
    table("taxonomy.tex", "Window taxonomy (five minutes, whole-path criterion at the given threshold): no output, no candidate within the threshold, a lower-ranked candidate within it while the Top-1 is not (ranking failure), Top-1 within it.",
          "tab:supp_tax", ["method", "protocol", "fold", "thr. m", "$n$", "no output \\%", "no cand. \\%", "ranking \\%", "Top-1 ok \\%"], rows, "lllrrrrrr", star=True, size="\\scriptsize")


def complementarity():
    t = pd.read_csv(os.path.join(T, "complementarity_w05.csv"))
    rows = [[esc(r.set), FOLD.get(r.fold, r.fold), m(r.n), ci(r.oracle_ade_median, r.oracle_ade_lo, r.oracle_ade_hi), ci(100 * r.oracle_route, 100 * r.oracle_route_lo, 100 * r.oracle_route_hi), ci(100 * r.all_fail_share, 100 * r.all_fail_lo, 100 * r.all_fail_hi, 1), esc(r.reference), m(r.ref_ade_median), pct(r.ref_route) if "ref_route" in t.columns else "--"] for r in t.itertuples()]
    table("complementarity.tex", "Complementarity of the map methods (five-minute windows): per-window oracles over a method set (lowest ADE; any route recovered), the share of windows every member fails, and the fixed reference method.",
          "tab:supp_comp", ["set", "fold", "$n$", "oracle median m [CI]", "oracle route \\% [CI]", "all fail \\% [CI]", "reference", "ref. median", "ref. route \\%"], rows, "llrrrrlrr", star=True, size="\\scriptsize")
    s = pd.read_csv(os.path.join(T, "common_failure_strata_w05.csv"))
    rows = [[esc(r.stratum), esc(r.level), m(r.n), m(r.n_drivers), ci(100 * r.all_fail_share, 100 * r.all_fail_lo, 100 * r.all_fail_hi, 1), pct(r.oracle_route), pct(r.ref_route)] for r in s.itertuples()]
    table("common_failure.tex", "Common failure of the steering and speed map methods by reference turn count and road density (folds pooled).", "tab:supp_cfail",
          ["stratum", "level", "$n$", "drivers", "all fail \\% [CI]", "oracle route \\%", "beam route \\%"], rows, "llrrrrr", star=True, size="\\scriptsize")
    d = pd.read_csv(os.path.join(T, "driver_route_w05.csv"))
    rows = [[esc(r.method), FOLD.get(r.fold, r.fold), m(r.n_windows), m(r.n_drivers), m(r.n_drivers_ge5), pct(r.pooled_route), pct(r.q10), pct(r.q25), pct(r.q50), pct(r.q75), pct(r.q90), pct(r.share_drivers_below_half)] for r in d.itertuples()]
    table("driver_route.tex", "Per-driver route recovery share (five-minute windows; drivers with at least five windows): pooled rate, quantiles over drivers and the share of drivers below 50\\,\\%.",
          "tab:supp_driver", ["method", "fold", "windows", "drivers", "$\\ge 5$", "pooled \\%", "q10", "q25", "q50", "q75", "q90", "below 50 \\%"], rows, "llrrrrrrrrrr", star=True, size="\\scriptsize")


def strata():
    t = pd.read_csv(os.path.join(T, "T7b_density_strata_v11.csv"))
    rows = [[esc(r.method), r.protocol, WIN.get(r.win, r.win), esc(r.level), m(r.n_episodes), m(r.n_drivers), ci(r.ade_median, r.ade_median_lo, r.ade_median_hi), pct(r.c100), pct(r.lost_road)] for r in t.itertuples()]
    table("density.tex", "Road-density strata around the start (tertiles; folds pooled): median ADE with CI, 100\\,m coverage and tail-lost share.", "tab:supp_density",
          ["method", "protocol", "window", "tertile", "$n$", "drivers", "median ADE m [CI]", "c100 \\%", "tail lost \\%"], rows, "lllrrrrrr", star=True, size="\\scriptsize")
    s = pd.read_csv(os.path.join(T, "supplement_summary.csv"))
    rows = [[esc(r.method), r.protocol, m(r.n), m(r.n_drives), m(r.n_drivers), m(r.fail), ci(r.median, r.median_lo, r.median_hi), m(r.p90), ci(100 * r.route, 100 * r.route_lo, 100 * r.route_hi)] for r in s.itertuples()]
    table("controlled.tex", "External check on the controlled supplement (one vehicle, one metropolitan area, other drivers; five-minute windows), methods unchanged after freezing.", "tab:supp_controlled",
          ["method", "protocol", "$n$", "drives", "drivers", "fail", "median ADE m [CI]", "p90 m", "route \\% [CI]"], rows, "llrrrrrrr", star=True, size="\\scriptsize")
    a = pd.read_csv(os.path.join(T, "area_coverage.csv"))
    rows = [[esc(r.check), esc(r.method), WIN.get(r.win, r.win), FOLD.get(r.fold, r.fold), m(r.n), m(r.n_out), pct(r.share, 1), m(r.max_m_median)] for r in a.itertuples()]
    table("area.tex", "Map-extent checks: share of windows whose reference route leaves the 5\\,km area adaptation square or the visible disc.", "tab:supp_area",
          ["check", "method", "window", "fold", "$n$", "out", "share \\%", "median max m"], rows, "llllrrrr", star=True, size="\\scriptsize")


def agent():
    if not os.path.exists(os.path.join(T, "agent_w05.csv")):
        return
    t = pd.read_csv(os.path.join(T, "agent_w05.csv"))
    rows = []
    for r in t.itertuples():
        rows.append([esc(r.label), FOLD.get(r.fold, r.fold), m(r.n), ci(r.route_pct, r.route_lo, r.route_hi), ci(r.ade_median, r.ade_lo, r.ade_hi), m(r.noout_pct, 1),
                     m(getattr(r, "cost_reconstructed_med", np.nan), 1), m(getattr(r, "tool_calls_mean", np.nan), 1), m(getattr(r, "gen_tokens_mean", np.nan))])
    table("agent_rows.tex", "Agent study (supplementary cohort, one five-minute window per held-out drive): every configuration and seed, with the selector fallback row and the pool oracle.",
          "tab:supp_agent", ["row", "fold", "$n$", "route \\% [CI]", "median ADE m [CI]", "no out. \\%", "cost s", "calls", "tokens"], rows, "p{0.26\\textwidth}lrrrrrrr", star=True, size="\\scriptsize")
    p = pd.read_csv(os.path.join(T, "agent_pairs_w05.csv"))
    rows = [[esc(r.pair), FOLD.get(r.fold, r.fold), m(r.n), f"{r.route_diff_pp:+.1f} [{r.route_lo_pp:+.1f}, {r.route_hi_pp:+.1f}]",
             ", ".join(f"{getattr(r, f'route_diff_s{s}_pp'):+.1f}" for s in (0, 1, 2) if hasattr(r, f"route_diff_s{s}_pp") and pd.notna(getattr(r, f"route_diff_s{s}_pp"))),
             ci(r.ade_med_diff, r.ade_lo, r.ade_hi, 1) if pd.notna(r.ade_med_diff) else "--"] for r in p.itertuples()]
    table("agent_pairs.tex", "Agent study, paired comparisons on the same windows: route recovery difference in percentage points (seeds pooled inside the driver resamples; per-seed values) and the median ADE difference on windows with output on both sides.",
          "tab:supp_agent_pairs", ["pair", "fold", "$n$", "$\\Delta$ route pp [CI]", "per seed", "$\\Delta$ median ADE m [CI]"], rows, "llrrrr", star=True, size="\\scriptsize")
    for name, title, lab in (("agent_selector_val.csv", "Supervised selector on the validation cohort (cross-fitted by driver).", "tab:supp_agent_sel"),
                             ("agent_pilot_summary.csv", "Validation pilot of the agent (40 windows, three seeds).", "tab:supp_agent_pilot")):
        pth = os.path.join(T, name)
        if not os.path.exists(pth):
            continue
        d = pd.read_csv(pth)
        if name.startswith("agent_selector"):
            rows = [[esc(r.row), m(r.n_windows), m(r.n_drivers), ci(r.route_pct, r.route_lo, r.route_hi)] for r in d.itertuples()]
            table("agent_selector_val.tex", title, lab, ["row", "windows", "drivers", "route \\% [CI]"], rows, "lrrr", size="\\scriptsize")
        else:
            cols = ["tag", "n", "submitted_pct", "truncation_pct", "forced_submit_pct", "decisions_mean", "tool_calls_mean", "completion_tokens_mean", "reasoning_tokens_mean", "llm_wall_med"]
            rows = [[esc(r.tag), m(r.n), m(r.submitted_pct, 1), m(r.truncation_pct, 1), m(r.forced_submit_pct, 1), m(r.decisions_mean, 1), m(r.tool_calls_mean, 1), m(r.completion_tokens_mean), m(r.reasoning_tokens_mean), m(r.llm_wall_med, 1)] for r in d.itertuples()]
            table("agent_pilot.tex", title, lab, ["tag", "$n$", "valid \\%", "trunc. \\%", "forced \\%", "turns", "calls", "tokens", "thinking", "LLM s"], rows, "lrrrrrrrrr", star=True, size="\\scriptsize")
    lv = os.path.join(T, "agent_live_check.csv")
    if os.path.exists(lv):
        r = pd.read_csv(lv).iloc[0]
        write("agent_live.tex", [f"On {int(r.n)} validation windows every reconstructor call was executed uncached and single-stream: median end-to-end time {r.live_med_s:.1f}\\,s against a reconstructed serial cost of {r.reconstructed_med_s:.1f}\\,s on the same windows (ratio {r.ratio_median:.2f}, 90th percentile {r.ratio_p90:.2f}).", ""])
    # the C-max exploration, if present
    mx = os.path.join(T, "agent_max_w05.csv")
    if os.path.exists(mx):
        t = pd.read_csv(mx)
        rows = [[esc(r.label), FOLD.get(r.fold, r.fold), m(r.n), ci(r.route_pct, r.route_lo, r.route_hi), ci(r.ade_median, r.ade_lo, r.ade_hi), m(r.noout_pct, 1), m(getattr(r, "cost_reconstructed_med", np.nan), 1), m(getattr(r, "tool_calls_mean", np.nan), 1), m(getattr(r, "gen_tokens_mean", np.nan))] for r in t.itertuples()]
        table("agent_max_rows.tex", "Ceiling exploration of the agent (C-max: ten reconstructor presets, comparison and scoring tools, relaxed budgets, optional critic and seed vote); exploratory, not part of the frozen protocol.",
              "tab:supp_agent_max", ["row", "fold", "$n$", "route \\% [CI]", "median ADE m [CI]", "no out. \\%", "cost s", "calls", "tokens"], rows, "llrrrrrrr", star=True, size="\\scriptsize")
        p = pd.read_csv(os.path.join(T, "agent_max_pairs_w05.csv"))
        rows = [[esc(r.pair), FOLD.get(r.fold, r.fold), m(r.n), f"{r.route_diff_pp:+.1f} [{r.route_lo_pp:+.1f}, {r.route_hi_pp:+.1f}]"] for r in p.itertuples()]
        table("agent_max_pairs.tex", "Ceiling exploration, paired route-recovery differences (percentage points, driver-clustered CIs).", "tab:supp_agent_max_pairs",
              ["pair", "fold", "$n$", "$\\Delta$ route pp [CI]"], rows, "llrr", size="\\scriptsize")
        sv = os.path.join(T, "agent_selector_max_val.csv")
        if os.path.exists(sv):
            d = pd.read_csv(sv)
            rows = [[esc(r.row), m(r.n_windows), m(r.n_drivers), ci(r.route_pct, r.route_lo, r.route_hi)] for r in d.itertuples()]
            table("agent_selector_max_val.tex", "Ceiling exploration: the selector over the ten presets and the extended diagnostics on the cached validation windows (cross-fitted by driver).",
                  "tab:supp_agent_sel_max", ["row", "windows", "drivers", "route \\% [CI]"], rows, "lrrr", size="\\scriptsize")
        hv = os.path.join(T, "agent_max_val_hints.csv")
        if os.path.exists(hv):
            d = pd.read_csv(hv)
            rows = [[esc(r.kind), esc(r.key), m(r.n), m(r.route_pct, 0)] for r in d.itertuples()]
            table("agent_max_hints.tex", "Calibration hints given to the C-max controller (validation candidates: share that recovered the route, by tool Top-1 and by diagnostic bin).",
                  "tab:supp_agent_hints", ["diagnostic", "bin", "candidates", "route \\%"], rows, "llrr", size="\\scriptsize")


def prompt_listing():
    cfg = os.path.join(paths.GPS_DIR, "configs", "agent")
    txt = open(os.path.join(cfg, "system_prompt.txt")).read()
    schema = json.load(open(os.path.join(cfg, "tools_schema.json")))
    budget = json.load(open(os.path.join(cfg, "budget.json")))
    lines = ["\\begin{scriptsize}\\begin{verbatim}"] + _wrap(txt, 58) + ["\\end{verbatim}\\end{scriptsize}"]
    write("agent_prompt.tex", lines)
    sl = ["\\begin{scriptsize}\\begin{verbatim}"]
    for t in schema:
        f = t["function"]
        props = f["parameters"].get("properties", {})
        sl.append(f"{f['name']}({', '.join(props)})")
        sl += _wrap("  " + f["description"], 58)
    sl += _wrap(f"budget: {budget['max_decisions']} decision turns, {budget['max_tool_calls']} tool calls + 1 submit, {budget['total_completion_tokens']} completion tokens "
                f"(per turn {budget['per_turn_max_tokens']}, reserve {budget['reserve_tokens']}), virtual clock {budget['timeout_virtual_s']:.0f} s", 58) + ["\\end{verbatim}\\end{scriptsize}"]
    write("agent_schema.tex", sl)


def _wrap(text, width):
    out = []
    for para in text.split("\n"):
        words, cur = para.split(" "), ""
        for w in words:
            if len(cur) + len(w) + 1 > width:
                out.append(cur)
                cur = w
            else:
                cur = (cur + " " + w).strip()
        out.append(cur)
    return out


if __name__ == "__main__":
    print("supplementary tables:")
    for fn in (main_board, cohort, signal, topk, robustness, cost, confidence, complementarity, strata, agent, prompt_listing):
        fn()
