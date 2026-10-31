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
# Encoding (validated with the dataviz validator, --pairs all, light): hue = signal set, the first three slots of the
# reference palette (blue = speed only, orange = speed + steering, aqua = steering only: SPy); line style = decoder
# (solid: online map filter, dashed: offline beam, dotted: no map, dash-dot + marker: a published method). At most five
# lines per panel, each direct-labelled (aqua is below 3:1 contrast on white: the direct label carries it); a shared
# legend below. Same entity, same style in every panel.
SPEED, STEER, STEER_ONLY = "#2a78d6", "#eb6834", "#1baf7a"
STYLE = {
    ("dr_none__G1", "O-N-Hp"): ("straight line, no map", SPEED, ":", None),
    ("dr_steer__G3", "O-N-Hp"): ("steering DR, no map", STEER, ":", None),
    ("map_pf_none__G1", "O-M-Hp"): ("map PF, speed", SPEED, "-", None),
    ("map_pf_steer__G3", "O-M-Hp"): ("map PF, steering", STEER, "-", None),
    ("map_beam_none__G1", "F-M-Hp"): ("map beam, speed", SPEED, "--", None),
    ("map_beam_steer__G3", "F-M-Hp"): ("map beam, steering", STEER, "--", None),
    ("macrotracking__G3", "O-M-Hp"): ("MacroTracking", STEER, "-.", "o"),
    ("opd2__G3", "F-M-Hp"): ("OPD-II", STEER, "-.", "s"),
    ("opd2_online__G3", "O-M-Hp"): ("OPD-II, online read-out", STEER, "-.", "s"),
    ("map_pf_steer_lineage__G3", "F-M-Hp"): ("map PF lineage, steering", STEER, (0, (5, 1, 1, 1)), None),
    ("elastic_pathing_hp__G1", "F-M-Hp"): ("Elastic Pathing", SPEED, "-.", "^"),
    ("map_pf_steer__G3", "O-M-Hm"): ("map PF, steering", STEER, "-", None),
    ("map_beam_steer__G3", "F-M-Hm"): ("map beam, steering", STEER, "--", None),
    ("map_pf_none__G1", "O-M-Hm"): ("map PF, speed", SPEED, "-", None),
    ("map_beam_none__G1", "F-M-Hm"): ("map beam, speed", SPEED, "--", None),
    ("elastic_pathing__G1", "F-M-Hm"): ("Elastic Pathing", SPEED, "-.", "^"),
    ("dewri_hp__G1", "F-M-Hp"): ("Dewri et al.", SPEED, "-.", "v"),
    ("dewri__G1", "F-M-Hm"): ("Dewri et al.", SPEED, "-.", "v"),
    ("spy_hp__G3", "F-M-Hp"): ("SPy", STEER_ONLY, "-.", "D"),
    ("spy__G3", "F-M-Hm"): ("SPy", STEER_ONLY, "-.", "D"),
}
CDF_PANELS = [
    ("speed vs steering, same decoders", [("dr_steer__G3", "O-N-Hp"), ("map_pf_none__G1", "O-M-Hp"), ("map_pf_steer__G3", "O-M-Hp"),
                                          ("map_beam_none__G1", "F-M-Hp"), ("map_beam_steer__G3", "F-M-Hp")]),
    ("published methods", [("map_pf_steer__G3", "O-M-Hp"), ("macrotracking__G3", "O-M-Hp"), ("opd2__G3", "F-M-Hp"),
                           ("elastic_pathing_hp__G1", "F-M-Hp"), ("spy_hp__G3", "F-M-Hp")]),
    ("start heading unknown", [("map_pf_steer__G3", "O-M-Hm"), ("map_beam_steer__G3", "F-M-Hm"), ("map_beam_none__G1", "F-M-Hm"),
                               ("elastic_pathing__G1", "F-M-Hm"), ("dewri__G1", "F-M-Hm")]),
]
LABEL_Y = [0.18, 0.36, 0.54, 0.72, 0.9]


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


def ade_vector(ep, man, stem, proto, win, req, hp, folds=tuple(T.FOLDS)):
    """Failure-aware ADE (+inf) of every expected episode of the folds, with driver ids."""
    rows = []
    for fold in folds:
        exp = T.expected(man, win, fold, proto, req, hp)
        got = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id").ade
        a = got.reindex(exp.episode_id).to_numpy(dtype=float)
        rows.append(pd.DataFrame({"episode_id": exp.episode_id.to_numpy(), "driver": exp.driver.to_numpy(), "fold": fold,
                                  "ade": np.where(np.isnan(a), np.inf, a)}))
    return pd.concat(rows, ignore_index=True)


def style_args(key):
    lab, col, ls, mk = STYLE[key]
    kw = dict(color=col, ls=ls, label=lab)
    if mk:
        kw.update(marker=mk, ms=3.5, markevery=0.12, mfc="white", mew=1.0)
    return kw


def direct_label(ax, x, y, text):
    """Ink-coloured label at the curve point, nudged right (labels never wear the series colour)."""
    ax.annotate(text, (x, y), xytext=(4, -1), textcoords="offset points", fontsize=6, color=INK2, va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))


def fig_cdf(ep, man):
    """Five-minute ADE CDFs on test drivers: failures stay in the denominator, so a curve tops out at 1 - failure rate."""
    rc()
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), sharey=True)
    req_of = {(m[1], m[2]): (m[5], m[6]) for m in T.MAIN if isinstance(m[1], str)}
    rows, handles = [], {}
    for ax, (title, keys) in zip(axes, CDF_PANELS):
        for i, key in enumerate(keys):
            req, hp = req_of[key]
            v = ade_vector(ep, man, key[0], key[1], "w05", req, hp, folds=("test",))
            x = np.sort(v.ade.to_numpy())
            if not np.isfinite(x).any():
                continue
            y = np.arange(1, len(x) + 1) / len(x)
            fin = np.isfinite(x)
            (line,) = ax.plot(x[fin], y[fin], **style_args(key))
            handles.setdefault(STYLE[key][0], line)
            yl = LABEL_Y[i]
            if fin.mean() >= yl:
                direct_label(ax, float(np.interp(yl, y[fin], x[fin])), yl, STYLE[key][0])
            rows.append(dict(panel=title, stem=key[0], protocol=key[1], n=len(x), fail=int((~fin).sum()), median=float(np.median(x))))
        ax.set_xscale("log")
        ax.set_xlim(5, 3e4)
        ax.set_ylim(0, 1)
        ax.set_xlabel("5-minute ADE (m), test drivers")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("share of episodes")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=5 if len(handles) > 8 else 4, bbox_to_anchor=(0.5, -0.2), fontsize=6.5)
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


HORIZON_PANELS = [
    ("online, start heading known", [("map_pf_steer__G3", "O-M-Hp"), ("map_pf_none__G1", "O-M-Hp"), ("macrotracking__G3", "O-M-Hp"),
                                     ("opd2_online__G3", "O-M-Hp"), ("dr_steer__G3", "O-N-Hp")]),
    ("offline, start heading known", [("map_beam_steer__G3", "F-M-Hp"), ("map_beam_none__G1", "F-M-Hp"), ("map_pf_steer_lineage__G3", "F-M-Hp"),
                                      ("elastic_pathing_hp__G1", "F-M-Hp")]),
    ("start heading unknown", [("map_pf_steer__G3", "O-M-Hm"), ("map_beam_steer__G3", "F-M-Hm"), ("map_beam_none__G1", "F-M-Hm"),
                               ("elastic_pathing__G1", "F-M-Hm")]),
]


SHORT = {"map PF, steering": "PF, steering", "map PF, speed": "PF, speed", "steering DR, no map": "DR, no map",
         "OPD-II, online read-out": "OPD-II", "map beam, steering": "beam, steering", "map beam, speed": "beam, speed",
         "map PF lineage, steering": "PF lineage"}


def label_ends(ax, items, min_dec=0.11):
    """End-of-line labels (ink colour) at the right of the last point, spread vertically (log axis) so none overlap."""
    items = sorted(items, key=lambda z: z[1])
    ys = [np.log10(y) for _, y, _ in items]
    for k in range(1, len(ys)):
        ys[k] = max(ys[k], ys[k - 1] + min_dec)
    for (x, _, text), yl in zip(items, ys):
        ax.annotate(text, (x, 10 ** yl), xytext=(5, 0), textcoords="offset points", fontsize=6, color=INK2, va="center")


def fig_horizon():
    """Same cohort (one start per test drive) at 1, 5 and 15 minutes, from tables/cohort_horizon.csv (icde_tables.py
    --extras cohort): online methods scored on prefixes of their 15-minute runs, offline ones rerun on each prefix."""
    rc()
    f = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "cohort_horizon.csv")
    if not os.path.exists(f):
        print("no cohort table yet")
        return
    t = pd.read_csv(f)
    t = t[(t.fold == "test") & ~((t.horizon_s == 900) & (t.source == "icde_v11") & t.protocol.str.startswith("O-"))]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.45), sharey=True)
    rows, handles = [], {}
    for ax, (title, keys) in zip(axes, HORIZON_PANELS):
        ends = []
        for key in keys:
            d = t[(t.method == key[0]) & (t.protocol == key[1])].sort_values("horizon_s")
            if len(d) < 2:
                continue
            xs = d.horizon_s.to_numpy() / 60.0
            kw = style_args(key)
            kw.update(marker=kw.get("marker", "o"), ms=3.5, markevery=1, mfc="white", mew=1.0)
            (line,) = ax.plot(xs, d.ade_median, **kw)
            ax.fill_between(xs, d.ade_lo, d.ade_hi, color=kw["color"], alpha=0.07, lw=0)
            handles.setdefault(STYLE[key][0], line)
            ends.append((xs[-1], float(d.ade_median.iloc[-1]), SHORT.get(STYLE[key][0], STYLE[key][0])))
            rows.extend(d.assign(panel=title).to_dict("records"))
        label_ends(ax, ends)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([1, 5, 15])
        ax.set_xticklabels(["1", "5", "15"])
        ax.set_xlim(0.8, 110)
        ax.set_xlabel("minutes after the start")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("median ADE (m), same test drives")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.24), fontsize=6.5)
    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig_horizon.csv"), index=False)
    save(fig, "fig_horizon_cohort")


def fig_topk():
    """Best of the first K candidate paths (median ADE, test drivers): offline searches with and without steering."""
    rc()
    f = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "topk_w05.csv")
    if not os.path.exists(f):
        print("no topk table yet")
        return
    t = pd.read_csv(f)
    t = t[t.fold == "test"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.3), sharey=True)
    Ks = [1, 3, 5, 10]
    panels = [("start heading known", [("map_beam_none__G1", "F-M-Hp"), ("map_beam_steer__G3", "F-M-Hp"), ("elastic_pathing_hp__G1", "F-M-Hp"),
                                       ("dewri_hp__G1", "F-M-Hp")]),
              ("start heading unknown", [("map_beam_none__G1", "F-M-Hm"), ("map_beam_steer__G3", "F-M-Hm"), ("elastic_pathing__G1", "F-M-Hm"),
                                         ("dewri__G1", "F-M-Hm")])]
    rows, handles = [], {}
    for ax, (title, keys) in zip(axes, panels):
        for key in keys:
            r = t[(t.stem == key[0]) & (t.protocol == key[1])]
            if r.empty:
                continue
            r = r.iloc[0]
            yv = [r[f"top{K}_ade_median"] for K in Ks]
            kw = style_args(key)
            kw.update(marker=kw.get("marker", "o"), ms=4, markevery=1, mfc="white", mew=1.0)
            (line,) = ax.plot(Ks, yv, **kw)
            handles.setdefault(STYLE[key][0], line)
            direct_label(ax, Ks[-1], yv[-1], STYLE[key][0])
            rows.append(dict(panel=title, stem=key[0], protocol=key[1], **{f"top{K}": y for K, y in zip(Ks, yv)}, best_first=r["best_rank_is_1"]))
        ax.set_yscale("log")
        ax.set_xticks(Ks)
        ax.set_xlim(0.5, 17)
        ax.set_xlabel("K (candidate paths, in rank order)")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("median ADE of the best\nof the first K paths (m)")
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), fontsize=6.5)
    os.makedirs(DATA, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig_topk.csv"), index=False)
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
