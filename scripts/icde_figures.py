#!/usr/bin/env python3
"""ICDE figures from per-episode results (run icde_v11 and the cohort runs), failure-aware (expected episodes without
output count as +inf / not recovered). Palette = the validated v1 categorical palette; published methods share the hue
of their closest baseline family and are told apart by dashed lines, markers and direct labels.
Outputs paper_icde/figures/*.pdf|png and their data CSVs in reports/icde/figures_data/."""
import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths, stats  # noqa: E402
import icde_tables as T  # noqa: E402

FIG = os.path.join(paths.GPS_DIR, "paper_icde", "figures")
DATA = os.path.join(paths.GPS_DIR, "reports", "icde", "figures_data")
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FORBIDDEN = re.compile(r"dongle|route_id|\b[0-9a-f]{16}\b|lat[_ ]?0|lon[_ ]?0|the Controlled-supplement city|/data/", re.I)
# entity -> (label, colour, linestyle, marker)
STYLE = {
    ("dr_none__G1", "O-N-Hp"): ("straight line", "#e34948", "-", None),
    ("dr_steer__G3", "O-N-Hp"): ("steering DR", "#2a78d6", "-", None),
    ("map_pf_none__G1", "O-M-Hp"): ("map PF, speed", "#6250d6", "-", None),
    ("map_pf_steer__G3", "O-M-Hp"): ("map PF, steering", "#eb6834", "-", None),
    ("macrotracking__G3", "O-M-Hp"): ("MacroTracking", "#2a78d6", "--", "o"),
    ("opd2_online__G3", "O-M-Hp"): ("OPD-II online", "#eb6834", "--", "s"),
    ("map_beam_none__G1", "F-M-Hp"): ("map beam, speed", "#6250d6", ":", None),
    ("map_beam_steer__G3", "F-M-Hp"): ("map beam, steering", "#1baf7a", "-", None),
    ("map_pf_steer_lineage__G3", "F-M-Hp"): ("PF lineage (offline)", "#eb6834", ":", None),
    ("opd2__G3", "F-M-Hp"): ("OPD-II (3 s delay)", "#eb6834", "--", "s"),
    ("elastic_pathing_hp__G1", "F-M-Hp"): ("Elastic Pathing", "#6250d6", "--", "^"),
    ("can_trace__G2", "F-M-Hm"): ("CAN-Trace", "#e87ba4", "--", "D"),
    ("map_pf_steer__G3", "O-M-Hm"): ("map PF, steering (H-)", "#eb6834", "-.", None),
    ("map_beam_steer__G3", "F-M-Hm"): ("map beam, steering (H-)", "#1baf7a", "-.", None),
    ("elastic_pathing__G1", "F-M-Hm"): ("Elastic Pathing (H-)", "#6250d6", "-.", "^"),
}


def rc():
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                         "grid.linewidth": 0.6, "legend.frameon": False, "lines.linewidth": 1.6, "figure.dpi": 150})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text)]
    bad = [t for t in texts if FORBIDDEN.search(t)]
    assert not bad, f"identifier gate: {bad}"
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def ade_vector(ep, man, stem, proto, win, req, hp):
    """Failure-aware ADE (+inf) of every expected test / few-shot episode, with driver ids."""
    rows = []
    for fold in T.FOLDS:
        exp = T.expected(man, win, fold, proto, req, hp)
        got = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id").ade
        a = got.reindex(exp.episode_id).to_numpy(dtype=float)
        rows.append(pd.DataFrame({"episode_id": exp.episode_id.to_numpy(), "driver": exp.driver.to_numpy(), "fold": fold,
                                  "ade": np.where(np.isnan(a), np.inf, a)}))
    return pd.concat(rows, ignore_index=True)


def fig_cdf(ep, man):
    rc()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    groups = [("heading known", [k for k in STYLE if k[1].endswith("Hp")]), ("heading unknown", [k for k in STYLE if k[1].endswith("Hm")])]
    req_of = {(m[1], m[2]): (m[5], m[6]) for m in T.MAIN if isinstance(m[1], str)}
    req_of[("can_trace__G2", "F-M-Hm")] = ("pedal", False)
    req_of[("map_pf_steer_lineage__G3", "F-M-Hp")] = ("G3", False)
    rows = []
    for ax, (title, keys) in zip(axes, groups):
        for key in keys:
            if key not in req_of:
                continue
            req, hp = req_of[key]
            v = ade_vector(ep, man, key[0], key[1], "w05", req, hp)
            if not np.isfinite(v.ade).any():
                continue
            x = np.sort(v.ade.to_numpy())
            y = np.arange(1, len(x) + 1) / len(x)
            lab, col, ls, mk = STYLE[key]
            fin = np.isfinite(x)
            ax.plot(x[fin], y[fin], color=col, ls=ls, label=lab)
            rows.append(dict(stem=key[0], protocol=key[1], n=len(x), fail=int((~fin).sum()), median=float(np.median(x))))
        ax.set_xscale("log")
        ax.set_xlim(5, 2e4)
        ax.set_xlabel("5-minute ADE (m)")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("share of episodes")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.32), fontsize=7)
    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig_cdf.csv"), index=False)
    save(fig, "fig_cdf_w05")


COHORT_RUNS = {60: "icde_cohort_p060", 300: "icde_cohort_p300", 900: "icde_v11"}
ONLINE_KEYS = [("dr_none__G1", "O-N-Hp"), ("dr_steer__G3", "O-N-Hp"), ("map_pf_none__G1", "O-M-Hp"), ("map_pf_steer__G3", "O-M-Hp"),
               ("macrotracking__G3", "O-M-Hp"), ("opd2_online__G3", "O-M-Hp")]
OFFLINE_KEYS = [("map_beam_none__G1", "F-M-Hp"), ("map_beam_steer__G3", "F-M-Hp"), ("map_pf_steer_lineage__G3", "F-M-Hp"),
                ("opd2__G3", "F-M-Hp"), ("elastic_pathing_hp__G1", "F-M-Hp")]


def cohort_scores():
    """Per (horizon, run) episode scores of the cohort; online methods are scored from the w15 files at every prefix
    (files icde_cohort_online_p060 / p300 written by `score.py --prefix-s`), offline ones from their prefix reruns."""
    coh = pd.read_csv(os.path.join(paths.MANIFESTS, "cohort_v11.csv"))
    coh = coh[coh.cohort]
    out = []
    for P, run in COHORT_RUNS.items():
        for r in (run, f"icde_cohort_online_p{P:03d}"):
            f = os.path.join(paths.SCORES, r, "episodes_all.parquet")
            if os.path.exists(f):
                e = pd.read_parquet(f, columns=["episode_id", "method", "protocol", "ade", "route_ok", "c100"])
                out.append(e[e.episode_id.isin(set(coh.episode_id))].assign(horizon=P, source=r))
    return coh, (pd.concat(out, ignore_index=True) if out else pd.DataFrame())


def fig_horizon():
    rc()
    coh, sc = cohort_scores()
    if sc.empty:
        print("no cohort scores yet")
        return
    drv = coh.set_index("episode_id").driver
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, keys, title in ((axes[0], ONLINE_KEYS, "online"), (axes[1], OFFLINE_KEYS, "offline")):
        for key in keys:
            xs, ys, lo, hi = [], [], [], []
            for P in COHORT_RUNS:
                s = sc[(sc.method == key[0]) & (sc.protocol == key[1]) & (sc.horizon == P)].drop_duplicates("episode_id").set_index("episode_id")
                if not len(s):
                    continue
                a = s.ade.reindex(coh.episode_id).to_numpy(dtype=float)
                a = np.where(np.isnan(a), np.inf, a)
                med, (l, h) = stats.median_inf(a, drv.reindex(coh.episode_id).to_numpy(), 300)
                xs.append(P / 60.0); ys.append(med); lo.append(l); hi.append(h)
                rows.append(dict(stem=key[0], protocol=key[1], horizon_s=P, n=len(a), fail=int(np.isinf(a).sum()), median=med, lo=l, hi=h,
                                 route_ok=float(s.route_ok.reindex(coh.episode_id).fillna(0).astype(float).mean())))
            if xs:
                lab, col, ls, mk = STYLE[key]
                ax.plot(xs, ys, color=col, ls=ls, marker=mk or "o", ms=4, label=lab)
                ax.fill_between(xs, lo, hi, color=col, alpha=0.08, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 5, 15])
        ax.set_xticklabels(["1", "5", "15"])
        ax.set_xlabel("minutes after the start (same drives)")
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6.5, loc="upper left")
    axes[0].set_ylabel("median ADE (m)")
    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig_horizon.csv"), index=False)
    save(fig, "fig_horizon_cohort")


def fig_topk():
    rc()
    f = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "topk_w05.csv")
    if not os.path.exists(f):
        print("no topk table yet")
        return
    t = pd.read_csv(f)
    t = t.groupby(["stem", "protocol"]).mean(numeric_only=True).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4))
    Ks = [1, 3, 5, 10]
    for r in t.itertuples():
        key = (r.stem, r.protocol)
        if key not in STYLE:
            continue
        lab, col, ls, mk = STYLE[key]
        axes[0].plot(Ks, [getattr(r, f"top{K}_ade_median") for K in Ks], color=col, ls=ls, marker=mk or "o", ms=4, label=lab)
        axes[1].plot(Ks, [getattr(r, f"top{K}_cov100") for K in Ks], color=col, ls=ls, marker=mk or "o", ms=4, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("median ADE of the best of K paths (m)")
    axes[1].set_ylabel("share of seconds within 100 m\nof any of the K paths")
    for ax in axes:
        ax.set_xticks(Ks)
        ax.set_xlabel("K (candidates in rank order)")
    axes[1].legend(fontsize=6.5, loc="lower right")
    save(fig, "fig_topk_w05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="icde_v11")
    ap.add_argument("--figs", nargs="+", default=["cdf"])
    args = ap.parse_args()
    if "cdf" in args.figs:
        ep, man, _ = T.load(args.run)
        fig_cdf(ep, man)
    if "horizon" in args.figs:
        fig_horizon()
    if "topk" in args.figs:
        fig_topk()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
