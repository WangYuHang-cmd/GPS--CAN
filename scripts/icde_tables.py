#!/usr/bin/env python3
"""ICDE tables from per-episode results (benchmark v1.1, run icde_v11), failure-aware.

For every row (method stem x protocol) and fold the EXPECTED episodes are fixed before looking at outputs: eligible
main-board episodes of the window, map_ok for map protocols, drives where the method's signals are valid, heading-known
protocols only for methods that need a start heading. Expected episodes without a scored output are failures (reason
from the runner's skipped files): ADE = +inf in medians, 0 / False in rates. Medians carry driver-clustered bootstrap
CIs; the driver-equal median summarises each driver first. Every number in the paper comes from these files.

    python scripts/icde_tables.py [--run icde_v11] [--n-boot 1000]
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths, stats  # noqa: E402

OUT = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
TEX = os.path.join(paths.GPS_DIR, "paper_icde", "tables")
FOLDS = {"test": "test", "few_shot_heldout": "few-shot"}
RATE_COLS = ["c100", "s50_90", "path_recall50", "route_ok", "end_within_100", "end_within_500", "tail_lost100", "terminal_lost100"]
FAIL_RATE = {"c100": 0.0, "s50_90": 0.0, "path_recall50": 0.0, "route_ok": 0.0, "end_within_100": 0.0, "end_within_500": 0.0,
             "tail_lost100": 1.0, "terminal_lost100": 1.0}

# (key, stem, protocol, label, family, requirement, heading_only)
MAIN = [
    ("hold", "p0_hold__G1", "O-N-Hp", "hold start", "no map, online", "G1", False),
    ("straight", "dr_none__G1", "O-N-Hp", "straight line (speed)", "no map, online", "G1", False),
    ("prior", "prior__G1", "O-N-Hp", "start prior (speed, train stats)", "no map, online", "G1", False),
    ("retrieval", "retrieval__G1", "O-N-Hp", "route retrieval (speed, train routes)", "no map, online", "G1", False),
    ("dr", "dr_steer__G3", "O-N-Hp", "steering DR, nominal", "no map, online", "G3", False),
    ("dr_off", "dr_steer_offonline__G3", "O-N-Hp", "steering DR, online offset", "no map, online", "G3", False),
    ("dr_cal", "dr_steer_calib__G3", "O-N-Hp", "steering DR, train calibrated", "no map, online", "G3", False),
    ("gru", ["learned_gru_s0__G3", "learned_gru_s1__G3", "learned_gru_s2__G3"], "O-N-Hp", "learned GRU (3 seeds)", "no map, online", "G3", False),
    ("resid", ["learned_resid_s0__G3", "learned_resid_s1__G3", "learned_resid_s2__G3"], "O-N-Hp", "learned physics residual (3 seeds)", "no map, online", "G3", False),
    ("pf_none", "map_pf_none__G1", "O-M-Hp", "map PF, speed only", "map, online", "G1", False),
    ("pf", "map_pf_steer__G3", "O-M-Hp", "map PF, steering", "map, online", "G3", False),
    ("pf_gru", ["map_pf_learned_gru_s0__G3", "map_pf_learned_gru_s1__G3", "map_pf_learned_gru_s2__G3"], "O-M-Hp", "map PF, learned GRU motion (3 seeds)", "map, online", "G3", False),
    ("pf_resid", ["map_pf_learned_resid_s0__G3", "map_pf_learned_resid_s1__G3", "map_pf_learned_resid_s2__G3"], "O-M-Hp", "map PF, learned residual motion (3 seeds)", "map, online", "G3", False),
    ("macro", "macrotracking__G3", "O-M-Hp", "MacroTracking", "map, online", "G3", True),
    ("macro_cal", "macrotracking_calib__G3", "O-M-Hp", "MacroTracking, train calibrated", "map, online", "G3", True),
    ("opd2_on", "opd2_online__G3", "O-M-Hp", "OPD-II, online read-out", "map, online", "G3", True),
    ("beam_none", "map_beam_none__G1", "F-M-Hp", "map beam, speed only", "map, offline", "G1", False),
    ("beam", "map_beam_steer__G3", "F-M-Hp", "map beam, steering", "map, offline", "G3", False),
    ("lineage", "map_pf_steer_lineage__G3", "F-M-Hp", "map PF, steering, best-lineage read-out", "map, offline", "G3", False),
    ("ep_hp", "elastic_pathing_hp__G1", "F-M-Hp", "Elastic Pathing (start heading used)", "map, offline", "G1", True),
    ("opd2", "opd2__G3", "F-M-Hp", "OPD-II (3 s delayed)", "map, offline", "G3", True),
    ("pf_hm", "map_pf_steer__G3", "O-M-Hm", "map PF, steering", "map, heading unknown", "G3", False),
    ("pf_none_hm", "map_pf_none__G1", "O-M-Hm", "map PF, speed only", "map, heading unknown", "G1", False),
    ("beam_hm", "map_beam_steer__G3", "F-M-Hm", "map beam, steering", "map, heading unknown", "G3", False),
    ("beam_none_hm", "map_beam_none__G1", "F-M-Hm", "map beam, speed only", "map, heading unknown", "G1", False),
    ("ep", "elastic_pathing__G1", "F-M-Hm", "Elastic Pathing", "map, heading unknown", "G1", False),
    ("ct", "can_trace__G2", "F-M-Hm", "CAN-Trace (pedal subset), start known", "map, heading unknown", "pedal", False),
    ("ct_area", "can_trace_area__G2", "F-M-Hm", "CAN-Trace (pedal subset), 5 km area", "map, heading unknown", "pedal", False),
    ("curv", "dr_curvature__G5", "O-N-Hp", "curvature DR (class F, not CAN-only)", "reference", "G5", False),
    ("pf_curv", "map_pf_curvature__G5", "O-M-Hp", "map PF, curvature (class F)", "reference", "G5", False),
]


def load(run):
    ep = pd.read_parquet(os.path.join(paths.SCORES, run, "episodes_all.parquet"))
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "driver", "drive", "fold", "win", "eligible_main"])
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"), columns=["episode_id", "map_ok"])
    aw = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"))
    man = man[man.eligible_main & man.fold.isin(list(FOLDS))].merge(me, on="episode_id", how="left").merge(aw, on=["driver", "drive"], how="left")
    man["map_ok"] = man.map_ok.fillna(False).astype(bool)
    man["pedal_ok"] = man.gas == "valid"
    skipped = {}
    for f in glob.glob(os.path.join(paths.PREDS, run, "*", "*.skipped.csv")):
        proto = os.path.basename(os.path.dirname(f))
        stem = os.path.basename(f).split(".part")[0]
        s = pd.read_csv(f)
        skipped.setdefault((proto, stem), []).append(s)
    skipped = {k: pd.concat(v).drop_duplicates("episode_id", keep="last").set_index("episode_id").reason for k, v in skipped.items()}
    return ep, man, skipped


def expected(man, win, fold, proto, req, heading_only):
    m = man[(man.win == win) & (man.fold == fold)]
    if proto.split("-")[1] == "M":
        m = m[m.map_ok]
    if req == "pedal":
        m = m[m.pedal_ok & m.G3.fillna(False).astype(bool)]
    elif req:
        m = m[m[req].fillna(False).astype(bool)]
    if heading_only and proto.endswith("Hm"):
        m = m.iloc[0:0]
    return m


def summarise(ep, man, skipped, stem, proto, win, fold, req, heading_only, n_boot):
    exp = expected(man, win, fold, proto, req, heading_only)
    if not len(exp):
        return None
    got = ep[(ep.method == stem) & (ep.protocol == proto)].set_index("episode_id")
    got = got[~got.index.duplicated()]
    d = exp[["episode_id", "driver"]].set_index("episode_id")
    ok = d.index.isin(got.index)
    ade = np.full(len(d), np.inf)
    ade[ok] = got.loc[d.index[ok], "ade"].to_numpy(dtype=float)
    ade = np.where(np.isnan(ade), np.inf, ade)
    drv = d.driver.to_numpy()
    med, (lo, hi) = stats.median_inf(ade, drv, n_boot)
    deq, (dlo, dhi), n_drv = stats.driver_equal(ade, drv, n_boot=n_boot)
    row = dict(stem=stem, protocol=proto, win=win, fold=fold, n_expected=len(d), n_output=int(ok.sum()), n_fail=int((~ok).sum()),
               fail_rate=float((~ok).mean()), n_drivers=int(n_drv), ade_median=med, ade_lo=lo, ade_hi=hi,
               ade_driver_median=deq, ade_driver_lo=dlo, ade_driver_hi=dhi,
               ade_mean_outputs=float(np.mean(ade[np.isfinite(ade)])) if np.isfinite(ade).any() else np.nan)
    for c in RATE_COLS:
        v = np.full(len(d), FAIL_RATE[c])
        if c in got.columns:
            vv = got.loc[d.index[ok], c].astype(float).to_numpy()
            v[ok] = np.where(np.isnan(vv), FAIL_RATE[c], vv)
        row[c] = float(v.mean())
    sk = skipped.get((proto, stem))
    if sk is not None and (~ok).any():
        reasons = sk.reindex(d.index[~ok]).dropna().str.slice(0, 50).value_counts()
        row["fail_reasons"] = "; ".join(f"{k} x{v}" for k, v in reasons.head(3).items())
    return row


def fmt_m(x):
    if not np.isfinite(x):
        return "fail" if x == np.inf else "n/a"
    return f"{x:,.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="icde_v11")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--wins", nargs="+", default=["w05"])
    ap.add_argument("--extras", nargs="*", default=[], help="also build: a3 topk signal pairs")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TEX, exist_ok=True)
    ep, man, skipped = load(args.run)
    if "a3" in args.extras:
        print(a3_compare())
    if "topk" in args.extras:
        print(topk_table(ep, man).round(2).to_string())
    if "signal" in args.extras:
        t, gt = signal_tables(ep, man)
        print(t.round(1).to_string()); print(gt.round(1).to_string())
    if "pairs" in args.extras:
        print(pair_table(ep, man).round(3).to_string())
    if "cohort" in args.extras:
        print(cohort_table().round(1).to_string())
    if "cost" in args.extras:
        print(cost_table().round(3).to_string())
    if "robust" in args.extras:
        print(robust_table(ep, man).round(2).to_string())
    rows = []
    for win in args.wins:
        for key, stem, proto, label, fam, req, hp in MAIN:
            for fold in FOLDS:
                stems = stem if isinstance(stem, list) else [stem]
                per = [summarise(ep, man, skipped, s, proto, win, fold, req, hp, args.n_boot) for s in stems]
                per = [p for p in per if p is not None]
                if not per or all(p["n_output"] == 0 for p in per):
                    continue
                if len(per) == 1:
                    r = per[0]
                else:                                     # seeds: mean of the per-seed statistics, SD across seeds
                    r = {k: (np.mean([p[k] for p in per]) if isinstance(per[0][k], (int, float, np.floating)) else per[0][k]) for k in per[0]}
                    r["ade_median_seed_sd"] = float(np.std([p["ade_median"] for p in per], ddof=1))
                    r["n_seeds"] = len(per)
                r.update(key=key, label=label, family=fam)
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "main_board.csv"), index=False)
    write_main(t, args.run)


def ci_str(r):
    s = f"{fmt_m(r['ade_median'])} [{fmt_m(r['ade_lo'])}, {fmt_m(r['ade_hi'])}]"
    sd = r.get("ade_median_seed_sd", np.nan)
    if isinstance(sd, float) and np.isfinite(sd):
        s += f" (seed SD {sd:.0f})"
    return s


WIN_NAME = {"w01": "1-minute", "w05": "5-minute", "w15": "15-minute", "full": "whole-drive"}


def write_main(t, run):
    """Markdown, one section per window (both folds side by side), and the paper's LaTeX table (w05, test | few-shot)."""
    md = []
    for win in dict.fromkeys(t.win):
        lines = main_rows(t[t.win == win])[0]
        md.append(f"## {WIN_NAME.get(win, win)} windows\n\n" + "\n".join(lines) + "\n")
        print("\n".join(lines))
    with open(os.path.join(OUT, "main_board.md"), "w") as f:
        f.write(f"# Main board (run {run}, benchmark v1.1; failures count as +inf / 0)\n\n" + "\n".join(md))
    if (t.win == "w05").any():
        open(os.path.join(TEX, "main_board.tex"), "w").write("\n".join(main_rows(t[t.win == "w05"])[1]) + "\n")


def main_rows(t):
    """Markdown lines and LaTeX lines for the rows of one window."""
    lines = ["| family | method | test n (fail) | test median ADE [CI] | test drv-eq | test c100 | test route | few-shot n (fail) | few-shot median ADE [CI] | few-shot drv-eq | few-shot c100 | few-shot route |",
             "|" + "---|" * 12]
    tex = [r"\begin{table*}[t]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}", r"\caption{Five-minute windows, test and few-shot drivers (benchmark v1.1). Median ADE in metres with "
           r"95\% CI from a driver-resampling bootstrap; drv-eq: median of per-driver medians; c100: share of seconds within 100\,m; "
           r"route: path recall at 50\,m $\ge 0.8$; fail: expected episodes without output (counted as $+\infty$ / 0). "
           r"Learned rows: mean of three seeds.}\label{tab:main}", r"\begin{tabular}{llrrrrrrrr}\toprule",
           r" & & \multicolumn{4}{c}{test} & \multicolumn{4}{c}{few-shot} \\\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
           r"setting & method & ADE [CI] & drv-eq & c100 & route & ADE [CI] & drv-eq & c100 & route \\\midrule"]
    last_fam = None
    for key in dict.fromkeys(t.key):
        a = t[(t.key == key) & (t.fold == "test")]
        b = t[(t.key == key) & (t.fold == "few_shot_heldout")]
        if a.empty and b.empty:
            continue
        ref = (a if len(a) else b).iloc[0]
        cells, texcells = [], []
        for d in (a, b):
            if d.empty:
                cells += ["", "", "", "", ""]
                texcells += ["--", "--", "--", "--"]
                continue
            r = d.iloc[0].to_dict()
            cells += [f"{int(r['n_expected'])} ({int(r['n_fail'])})", ci_str(r), fmt_m(r["ade_driver_median"]), f"{r['c100']:.2f}", f"{r['route_ok']:.2f}"]
            texcells += [f"{fmt_m(r['ade_median'])} [{fmt_m(r['ade_lo'])}, {fmt_m(r['ade_hi'])}]" + (r"$^\dagger$" if r["n_fail"] else ""),
                         fmt_m(r["ade_driver_median"]), f"{r['c100']:.2f}", f"{r['route_ok']:.2f}"]
        lines.append(f"| {ref['family']} | {ref['label']} | " + " | ".join(cells) + " |")
        fam = ref["family"] if ref["family"] != last_fam else ""
        if fam and last_fam is not None:
            tex.append(r"\midrule")
        last_fam = ref["family"]
        tex.append(f"{fam} & {ref['label']} & " + " & ".join(texcells) + r" \\")
    tex += [r"\bottomrule\end{tabular}", r"\\[2pt]\footnotesize $^\dagger$ some expected episodes without output (counted as failures).", r"\end{table*}"]
    return lines, tex


# ---------------------------------------------------------------- extra tables (called from extras())
def a3_compare(n_boot=500):
    """A3: union visibility (v1 behaviour) vs the episode's own tiles on the same 60 test drives; PF seed noise as scale."""
    p_u = os.path.join(paths.SCORES, "icde_a3_union", "episodes_all.parquet")
    p_m = os.path.join(paths.SCORES, "icde_v11", "episodes_all.parquet")
    if not (os.path.exists(p_u) and os.path.exists(p_m)):
        return None
    u, m = pd.read_parquet(p_u), pd.read_parquet(p_m)
    rows = []
    for stem, proto, other_run, other_stem, what in (("map_beam_steer__G3", "F-M-Hp", None, None, "union vs own tiles"),
                                                      ("map_pf_steer__G3", "O-M-Hp", None, None, "union vs own tiles"),
                                                      ("map_pf_steer__G3", "O-M-Hp", "u", "map_pf_steer_s1__G3", "seed 1 vs seed 0 (own tiles)")):
        a = u[(u.method == (other_stem or stem)) & (u.protocol == proto)].set_index("episode_id")
        b = m[(m.method == stem) & (m.protocol == proto)].set_index("episode_id")
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        for win in ("w05", "w15"):
            c = [e for e in common if a.loc[e, "win"] == win] if "win" in a.columns else list(common)
            if not c:
                continue
            d = a.loc[c, "ade"].to_numpy(dtype=float) - b.loc[c, "ade"].to_numpy(dtype=float)
            r = stats.paired(a.loc[c, "ade"].to_numpy(dtype=float), b.loc[c, "ade"].to_numpy(dtype=float), b.loc[c, "driver"].to_numpy(), n_boot)
            rows.append(dict(method=stem, protocol=proto, comparison=what, win=win, n=len(c), share_abs_diff_lt_1m=float(np.mean(np.abs(d) < 1.0)),
                             median_diff=r.get("median"), median_lo=r.get("median_lo"), median_hi=r.get("median_hi"),
                             mean_diff=r.get("mean"), mean_lo=r.get("mean_lo"), mean_hi=r.get("mean_hi")))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "a3_visibility.csv"), index=False)
    return t


def topk_table(ep, man, n_boot=300):
    """RQ4: offline candidates. Top-1, admissible whole-path oracle among the first K, per-second union coverage."""
    rows = []
    for stem, proto, req, hp in (("map_beam_steer__G3", "F-M-Hp", "G3", False), ("map_beam_steer__G3", "F-M-Hm", "G3", False),
                                 ("map_beam_none__G1", "F-M-Hp", "G1", False), ("map_beam_none__G1", "F-M-Hm", "G1", False),
                                 ("elastic_pathing_hp__G1", "F-M-Hp", "G1", True), ("elastic_pathing__G1", "F-M-Hm", "G1", False),
                                 ("can_trace__G2", "F-M-Hm", "pedal", False), ("can_trace_area__G2", "F-M-Hm", "pedal", False)):
        for fold in FOLDS:
            exp = expected(man, "w05", fold, proto, req, hp)
            got = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id")
            if not len(exp) or not len(got):
                continue
            g = got.reindex(exp.episode_id)
            drv = exp.driver.to_numpy()
            out = g.ade.notna().to_numpy()
            # an output without a candidate set (a single path) is its own top-K: K = 1, rank 1
            single = out & (g.topk_K.isna().to_numpy() if "topk_K" in g.columns else np.ones(len(g), dtype=bool))
            row = dict(stem=stem, protocol=proto, fold=fold, n_expected=len(exp), n_output=int(out.sum()), n_single_path=int(single.sum()))
            for K in (1, 3, 5, 10):
                col = f"top{K}_ade"
                v = g[col].to_numpy(dtype=float) if col in g.columns else np.full(len(g), np.nan)
                v = np.where(single, g.ade.to_numpy(dtype=float), v)
                row[f"top{K}_ade_median"], _ = stats.median_inf(np.where(np.isnan(v), np.inf, v), drv, n_boot)
                cv = g[f"top{K}_cov100"].to_numpy(dtype=float) if f"top{K}_cov100" in g.columns else np.full(len(g), np.nan)
                cv = np.where(single, g.c100.to_numpy(dtype=float), cv)
                row[f"top{K}_cov100"] = float(np.nan_to_num(cv, nan=0.0).mean())
            br = g.best_rank.to_numpy(dtype=float) if "best_rank" in g.columns else np.full(len(g), np.nan)
            br = np.where(single, 1.0, br)
            row["best_rank_is_1"] = float(np.mean(br[out] == 1)) if out.any() else np.nan
            kk = g.topk_K.to_numpy(dtype=float) if "topk_K" in g.columns else np.full(len(g), np.nan)
            kk = np.where(single, 1.0, kk)
            row["n_candidates_median"] = float(np.nanmedian(kk[out])) if out.any() else np.nan
            rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "topk_w05.csv"), index=False)
    return t


# (label, a = (stem or seed list, protocol, requirement, heading_only), b = same); a - b on the intersection of both
# expected sets; failures stay in as +inf (medians) and count as losses in the win rate
PAIRS = [
    ("map gain, steering, heading known", ("map_pf_steer__G3", "O-M-Hp", "G3", False), ("dr_steer__G3", "O-N-Hp", "G3", False)),
    ("map gain, steering, heading unknown", ("map_pf_steer__G3", "O-M-Hm", "G3", False), ("dr_steer__G3", "O-N-Hm", "G3", False)),
    ("start heading, map PF steering", ("map_pf_steer__G3", "O-M-Hp", "G3", False), ("map_pf_steer__G3", "O-M-Hm", "G3", False)),
    ("offline beam vs online PF, steering", ("map_beam_steer__G3", "F-M-Hp", "G3", False), ("map_pf_steer__G3", "O-M-Hp", "G3", False)),
    ("decode output: PF lineage vs PF weighted mean", ("map_pf_steer_lineage__G3", "F-M-Hp", "G3", False), ("map_pf_steer__G3", "O-M-Hp", "G3", False)),
    ("decode output: beam vs PF lineage", ("map_beam_steer__G3", "F-M-Hp", "G3", False), ("map_pf_steer_lineage__G3", "F-M-Hp", "G3", False)),
    ("start heading, map beam steering", ("map_beam_steer__G3", "F-M-Hp", "G3", False), ("map_beam_steer__G3", "F-M-Hm", "G3", False)),
    ("steering value, map PF, heading known", ("map_pf_steer__G3", "O-M-Hp", "G3", False), ("map_pf_none__G1", "O-M-Hp", "G1", False)),
    ("steering value, map PF, heading unknown", ("map_pf_steer__G3", "O-M-Hm", "G3", False), ("map_pf_none__G1", "O-M-Hm", "G1", False)),
    ("steering value, map beam, heading known", ("map_beam_steer__G3", "F-M-Hp", "G3", False), ("map_beam_none__G1", "F-M-Hp", "G1", False)),
    ("steering value, map beam, heading unknown", ("map_beam_steer__G3", "F-M-Hm", "G3", False), ("map_beam_none__G1", "F-M-Hm", "G1", False)),
    ("Elastic Pathing vs map beam, speed only, heading unknown", ("elastic_pathing__G1", "F-M-Hm", "G1", False), ("map_beam_none__G1", "F-M-Hm", "G1", False)),
    ("Elastic Pathing, start heading used vs not", ("elastic_pathing_hp__G1", "F-M-Hp", "G1", True), ("elastic_pathing__G1", "F-M-Hm", "G1", False)),
    ("MacroTracking vs map PF, steering, online", ("macrotracking__G3", "O-M-Hp", "G3", True), ("map_pf_steer__G3", "O-M-Hp", "G3", False)),
    ("OPD-II (3 s delayed) vs map PF, steering", ("opd2__G3", "F-M-Hp", "G3", True), ("map_pf_steer__G3", "O-M-Hp", "G3", False)),
    ("OPD-II vs MacroTracking", ("opd2__G3", "F-M-Hp", "G3", True), ("macrotracking__G3", "O-M-Hp", "G3", True)),
    ("OPD-II calibrated vs as released", ("opd2_calib__G3", "F-M-Hp", "G3", True), ("opd2__G3", "F-M-Hp", "G3", True)),
    ("MacroTracking calibrated vs as released", ("macrotracking_calib__G3", "O-M-Hp", "G3", True), ("macrotracking__G3", "O-M-Hp", "G3", True)),
    ("learned GRU vs nominal DR, no map", (["learned_gru_s0__G3", "learned_gru_s1__G3", "learned_gru_s2__G3"], "O-N-Hp", "G3", False),
     ("dr_steer__G3", "O-N-Hp", "G3", False)),
    ("learned residual vs nominal DR, no map", (["learned_resid_s0__G3", "learned_resid_s1__G3", "learned_resid_s2__G3"], "O-N-Hp", "G3", False),
     ("dr_steer__G3", "O-N-Hp", "G3", False)),
    ("learned GRU motion in map PF vs nominal steering", (["map_pf_learned_gru_s0__G3", "map_pf_learned_gru_s1__G3", "map_pf_learned_gru_s2__G3"], "O-M-Hp", "G3", False),
     ("map_pf_steer__G3", "O-M-Hp", "G3", False)),
    ("learned residual motion in map PF vs nominal steering", (["map_pf_learned_resid_s0__G3", "map_pf_learned_resid_s1__G3", "map_pf_learned_resid_s2__G3"], "O-M-Hp", "G3", False),
     ("map_pf_steer__G3", "O-M-Hp", "G3", False)),
    ("CAN yaw rate vs steering, no map", ("dr_can_yaw__G4", "O-N-Hp", "G4", False), ("dr_steer__G3", "O-N-Hp", "G3", False)),
    ("device gyro vs steering, no map", ("dr_imu_yaw_rls__G8", "O-N-Hp", "G8", False), ("dr_steer__G3", "O-N-Hp", "G3", False)),
    ("Elastic Pathing (speed) vs hold start", ("elastic_pathing__G1", "F-M-Hm", "G1", False), ("p0_hold__G1", "O-N-Hp", "G1", False)),
    ("Elastic Pathing (speed) vs map PF steering, heading unknown", ("elastic_pathing__G1", "F-M-Hm", "G1", False),
     ("map_pf_steer__G3", "O-M-Hm", "G3", False)),
    ("Elastic Pathing (speed) vs map beam steering, heading unknown", ("elastic_pathing__G1", "F-M-Hm", "G1", False),
     ("map_beam_steer__G3", "F-M-Hm", "G3", False)),
]


def pair_table(ep, man, wins=("w05", "w15"), n_boot=1000):
    """Paired differences a - b per fold and window on the episodes both rows expect (fixed before outputs)."""
    def ade(stem, proto, ids):
        got = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id").ade
        v = got.reindex(ids).to_numpy(dtype=float)
        return np.where(np.isnan(v), np.inf, v)

    rows = []
    for label, (sa, pa, ra, ha), (sb, pb, rb, hb) in PAIRS:
        for win in wins:
            for fold in FOLDS:
                ea, eb = expected(man, win, fold, pa, ra, ha), expected(man, win, fold, pb, rb, hb)
                d = ea[ea.episode_id.isin(set(eb.episode_id))]
                if not len(d):
                    continue
                ids, drv = d.episode_id.to_numpy(), d.driver.to_numpy()
                b = ade(sb, pb, ids)
                if np.isinf(b).all():
                    continue
                per = []
                for s in (sa if isinstance(sa, list) else [sa]):
                    a = ade(s, pa, ids)
                    if np.isinf(a).all():
                        continue
                    r = stats.paired(a, b, drv, n_boot)
                    both_fail = np.isinf(a) & np.isinf(b)
                    win_a = float(np.mean((a < b)[~both_fail])) if (~both_fail).any() else np.nan
                    per.append(dict(r, median_a=float(np.median(a)), median_b=float(np.median(b)), fail_a=int(np.isinf(a).sum()), fail_b=int(np.isinf(b).sum()), win_a_with_fail=win_a))
                if not per:
                    continue
                row = {k: float(np.mean([p[k] for p in per])) if isinstance(per[0].get(k), (int, float, np.floating)) else per[0].get(k) for k in per[0]}
                if len(per) > 1:
                    row["median_diff_seed_sd"] = float(np.std([p["median"] for p in per], ddof=1))
                rows.append(dict(comparison=label, a=sa if isinstance(sa, str) else f"{len(per)} seeds", b=sb, win=win, fold=fold, n_common=len(d), **row))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "pairs.csv"), index=False)
    return t


COHORT_SOURCES = {60: ["icde_cohort_online_p060", "icde_cohort_p060"], 300: ["icde_cohort_online_p300", "icde_cohort_p300"],
                  900: ["icde_cohort_online_p900", "icde_v11"]}


def cohort_table(n_boot=500):
    """RQ3: the same cohort (one 15-minute start per drive, manifests/cohort_v11.csv) scored at 60 / 300 / 900 s. Online
    methods come from their w15 files scored with --prefix-s; offline ones from their prefix reruns (and w15 at 900 s).
    Every cohort episode a method should answer counts; missing outputs are failures (+inf)."""
    coh = pd.read_csv(os.path.join(paths.MANIFESTS, "cohort_v11.csv"))
    coh = coh[coh.cohort]
    rows = []
    for P, runs in COHORT_SOURCES.items():
        for run in runs:
            f = os.path.join(paths.SCORES, run, "episodes_all.parquet")
            if not os.path.exists(f):
                continue
            e = pd.read_parquet(f, columns=["episode_id", "method", "protocol", "ade", "route_ok"])
            e = e[e.episode_id.isin(set(coh.episode_id))]
            for (meth, proto), g in e.groupby(["method", "protocol"]):
                g = g.drop_duplicates("episode_id").set_index("episode_id")
                for fold in FOLDS:
                    c = coh[coh.fold == fold]
                    a = g.ade.reindex(c.episode_id).to_numpy(dtype=float)
                    a = np.where(np.isnan(a), np.inf, a)
                    if np.isinf(a).all():
                        continue
                    med, (lo, hi) = stats.median_inf(a, c.driver.to_numpy(), n_boot)
                    deq, _, nd = stats.driver_equal(a, c.driver.to_numpy(), n_boot=n_boot)
                    rows.append(dict(method=meth, protocol=proto, fold=fold, horizon_s=P, source=run, n=len(a), n_output=int(np.isfinite(a).sum()),
                                     n_drivers=nd, ade_median=med, ade_lo=lo, ade_hi=hi, ade_driver_median=deq,
                                     route_ok=float(g.route_ok.reindex(c.episode_id).fillna(0).astype(float).mean())))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "cohort_horizon.csv"), index=False)
    # paired differences on the cohort episodes at every horizon (offline runs: prefix reruns; 900 s: the w15 runs)
    prs = []
    runs = {60: ["icde_cohort_p060", "icde_cohort_online_p060"], 300: ["icde_cohort_p300", "icde_cohort_online_p300"], 900: ["icde_v11"]}
    for P, rr in runs.items():
        e = pd.concat([pd.read_parquet(os.path.join(paths.SCORES, r, "episodes_all.parquet"), columns=["episode_id", "method", "protocol", "ade"])
                       for r in rr if os.path.exists(os.path.join(paths.SCORES, r, "episodes_all.parquet"))])
        for name, (sa, pa), (sb, pb) in COHORT_PAIRS:
            for fold in FOLDS:
                c = coh[coh.fold == fold]

                def ade(stem, proto):
                    g = e[(e.method == stem) & (e.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id").ade
                    v = g.reindex(c.episode_id).to_numpy(dtype=float)
                    return np.where(np.isnan(v), np.inf, v)
                a, b = ade(sa, pa), ade(sb, pb)
                if np.isinf(a).all() or np.isinf(b).all():
                    continue
                r = stats.paired(a, b, c.driver.to_numpy(), n_boot)
                prs.append(dict(comparison=name, horizon_s=P, fold=fold, **r))
    pd.DataFrame(prs).to_csv(os.path.join(OUT, "cohort_pairs.csv"), index=False)
    return t


ROBUST = [("dr_steer__G3", "steering DR"), ("dr_none__G1", "straight line")]
PERTURB = [("pq1", "speed quantised to 1 m/s"), ("pn05", "speed noise 0.5 m/s"), ("s1hz", "steering at 1 Hz")]


def robust_table(ep, man, n_boot=1000):
    """RQ5: degraded inputs (v1 probes, no map, unchanged in v1.1) against the clean signal, paired, w05 O-N-Hp."""
    rows = []
    for stem, label in ROBUST:
        base = stem.split("__")[0]
        for tag, what in PERTURB:
            pert = f"{base}_{tag}__G3"
            if not ((ep.method == pert) & (ep.protocol == "O-N-Hp")).any():
                continue
            for fold in FOLDS:
                exp = expected(man, "w05", fold, "O-N-Hp", "G3", False)
                ids, drv = exp.episode_id.to_numpy(), exp.driver.to_numpy()

                def ade(st):
                    g = ep[(ep.method == st) & (ep.protocol == "O-N-Hp")].drop_duplicates("episode_id").set_index("episode_id").ade
                    v = g.reindex(ids).to_numpy(dtype=float)
                    return np.where(np.isnan(v), np.inf, v)
                a, b = ade(pert), ade(stem)
                r = stats.paired(a, b, drv, n_boot)
                rows.append(dict(method=label, perturbation=what, tag=tag, fold=fold, median_clean=float(np.median(b)),
                                 median_perturbed=float(np.median(a)), **r))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "robustness_w05.csv"), index=False)
    return t


COST_LOG = paths.LOGS


def cost_table():
    """RQ5: run icde_cost (scripts/run_icde_cost.sh: one worker, the same 92 validation episodes for every method).
    Per-episode inference seconds from the runner's timing files (graph build and drive loading listed separately),
    peak resident memory and wall time of the whole run from /usr/bin/time -v."""
    rows = []
    for tf in sorted(glob.glob(os.path.join(COST_LOG, "icde_cost", "*.time"))):
        tag = os.path.basename(tf)[:-5]
        txt = open(tf).read()
        rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        wall = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): ([\d:.]+)", txt)
        secs = np.nan
        if wall:
            parts = [float(x) for x in wall.group(1).split(":")]
            secs = sum(v * 60 ** k for k, v in enumerate(reversed(parts)))
        timing = glob.glob(os.path.join(paths.PREDS, "icde_cost", "*", "*.timing.csv"))
        out = open(os.path.join(COST_LOG, "icde_cost", tag + ".out")).read() if os.path.exists(os.path.join(COST_LOG, "icde_cost", tag + ".out")) else ""
        stem = re.search(r"-> \S+/([^/]+)\.part0\.parquet", out)
        t = pd.DataFrame()
        if stem:
            f = [x for x in timing if os.path.basename(x).startswith(stem.group(1) + ".part")]
            if f:
                t = pd.read_csv(f[0])
        row = dict(tag=tag, peak_rss_gb=int(rss.group(1)) / 1048576 if rss else np.nan, wall_s=secs)
        if len(t):
            e = t[~t.protocol.str.startswith("_")].copy()
            e["win"] = e.key.str.split("/").str[2]
            for win, g in e.groupby("win"):
                row[f"n_{win}"] = len(g)
                row[f"infer_median_s_{win}"] = float(g.secs.median())
                row[f"infer_p90_s_{win}"] = float(g.secs.quantile(0.9))
            gr = t[t.protocol == "_graph"].secs
            row["graph_build_median_s"] = float(gr.median()) if len(gr) else np.nan
            row["drive_load_median_s"] = float(t[t.protocol == "_load"].secs.median())
        rows.append(row)
    c = pd.DataFrame(rows)
    c.to_csv(os.path.join(OUT, "cost.csv"), index=False)
    return c


COHORT_PAIRS = [("beam vs PF lineage", ("map_beam_steer__G3", "F-M-Hp"), ("map_pf_steer_lineage__G3", "F-M-Hp")),
                ("beam vs PF", ("map_beam_steer__G3", "F-M-Hp"), ("map_pf_steer__G3", "O-M-Hp")),
                ("PF lineage vs PF", ("map_pf_steer_lineage__G3", "F-M-Hp"), ("map_pf_steer__G3", "O-M-Hp"))]


SIG_GROUPS = ["S1", "S2c", "S3", "S4c", "S2b", "S4b"]
SIG_PAIRS = [("S2c", "S1", "pedals | speed"), ("S3", "S1", "steering | speed"), ("S4c", "S3", "pedals | speed+steering"),
             ("S4c", "S2c", "steering | speed+pedals"), ("S2b", "S1", "pressed flags | speed"), ("S4b", "S3", "pressed flags | speed+steering")]
SIG_RUNNER_GROUP = {"S1": "G1", "S2c": "G2", "S2b": "G2", "S3": "G3", "S4c": "G7", "S4b": "G7"}


def signal_tables(ep, man, n_boot=500):
    """RQ2: signal-set rows (seed means) and paired gains on the common pedal-valid episodes, no map and map."""
    rows, gains = [], []
    for kind, proto, prefix in (("no map", "O-N-Hp", "sig_dr"), ("map", "O-M-Hp", "map_pf_sig")):
        exp = pd.concat([expected(man, "w05", f, proto, "pedal", False) for f in FOLDS])
        drv = exp.set_index("episode_id").driver
        per = {}
        for g in SIG_GROUPS:
            for s in (0, 1, 2):
                stem = f"{prefix}_{g}_s{s}__{SIG_RUNNER_GROUP[g]}"
                got = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id")
                if not len(got):
                    continue
                a = got.ade.reindex(exp.episode_id).to_numpy(dtype=float)
                per[(g, s)] = np.where(np.isnan(a), np.inf, a)
                med, (lo, hi) = stats.median_inf(per[(g, s)], drv.to_numpy(), n_boot)
                rows.append(dict(setting=kind, group=g, seed=s, n=len(a), fail=int(np.isinf(per[(g, s)]).sum()), ade_median=med, lo=lo, hi=hi,
                                 route_ok=float(got.route_ok.reindex(exp.episode_id).fillna(0).astype(float).mean()),
                                 c100=float(got.c100.reindex(exp.episode_id).fillna(0).mean())))
        for a_g, b_g, label in SIG_PAIRS:
            for s in (0, 1, 2):
                if (a_g, s) in per and (b_g, s) in per:
                    r = stats.paired(per[(a_g, s)], per[(b_g, s)], drv.to_numpy(), n_boot)
                    gains.append(dict(setting=kind, added=label, a=a_g, b=b_g, seed=s, **r))
    t, gt = pd.DataFrame(rows), pd.DataFrame(gains)
    t.to_csv(os.path.join(OUT, "signal_sets_w05.csv"), index=False)
    gt.to_csv(os.path.join(OUT, "signal_gains_w05.csv"), index=False)
    # the benchmark's map methods on the same pedal-valid windows (map protocol expected set), for scale
    exp = pd.concat([expected(man, "w05", f, "O-M-Hp", "pedal", False) for f in FOLDS])
    drv = exp.driver.to_numpy()
    ref = []
    for key, stem, proto in SIG_REFERENCE:
        got = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id")
        if not len(got):
            continue
        a = got.ade.reindex(exp.episode_id).to_numpy(dtype=float)
        a = np.where(np.isnan(a), np.inf, a)
        med, (lo, hi) = stats.median_inf(a, drv, n_boot)
        ref.append(dict(key=key, stem=stem, protocol=proto, n=len(a), fail=int(np.isinf(a).sum()), ade_median=med, lo=lo, hi=hi))
    pd.DataFrame(ref).to_csv(os.path.join(OUT, "signal_reference_w05.csv"), index=False)
    return t, gt


SIG_REFERENCE = [("pf", "map_pf_steer__G3", "O-M-Hp"), ("beam", "map_beam_steer__G3", "F-M-Hp"), ("opd2", "opd2__G3", "F-M-Hp"),
                 ("macro", "macrotracking__G3", "O-M-Hp"), ("pf_none", "map_pf_none__G1", "O-M-Hp"), ("beam_none", "map_beam_none__G1", "F-M-Hp"),
                 ("ep", "elastic_pathing__G1", "F-M-Hm"), ("ep_hp", "elastic_pathing_hp__G1", "F-M-Hp"),
                 ("ct", "can_trace__G2", "F-M-Hm"), ("ct_area", "can_trace_area__G2", "F-M-Hm")]


if __name__ == "__main__":
    main()
