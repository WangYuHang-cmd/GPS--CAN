"""Paper figures for the CAN-to-GPS benchmark v1 (reports/paper/figures/*.pdf|png).

Data: reports/paper/figures/data/*.parquet from scripts/paper_curves.py (per-point curves, coverage, lost-road timing,
anonymised example trajectories) and the per-episode score table (ADE CDF, strata). Figures show episode-frame metres
only: no coordinates, place names, device or route identifiers, and no Controlled-supplement drives.

  python scripts/make_paper_figures.py [--data DIR] [--out DIR] [--folds test few_shot_heldout]
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cantraj import paths  # noqa: E402

GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(GPS_DIR, "reports", "paper", "figures", "data")
OUT = os.path.join(GPS_DIR, "reports", "paper", "figures")

# colour follows the method in every figure (validated categorical palette, light surface); class F dashed, hold dotted
STYLE = {
    "p0_hold__G1": ("hold start", "#9a9890", ":"),
    "dr_none__G1": ("speed only (straight)", "#e34948", "-"),
    "dr_steer__G3": ("steering DR", "#2a78d6", "-"),
    "dr_steer_calib__G3": ("calibrated DR", "#e87ba4", "-"),
    "learned_gru_s0__G3": ("learned GRU", "#eda100", "-"),
    "dr_curvature__G5": ("curvature DR (class F)", "#008300", "--"),
    "map_pf_steer__G3": ("map PF, steering", "#eb6834", "-"),
    "map_pf_none__G1": ("map PF, speed only", "#6250d6", "-"),
    "map_beam_steer__G3": ("map beam, steering (offline)", "#1baf7a", "-"),
}
NOMAP = ["p0_hold__G1", "dr_none__G1", "dr_steer__G3", "dr_steer_calib__G3", "learned_gru_s0__G3", "dr_curvature__G5"]
MAP = ["dr_steer__G3", "map_pf_steer__G3", "map_pf_none__G1", "map_beam_steer__G3"]
PROTO = {m: "O-N-Hp" for m in NOMAP} | {"map_pf_steer__G3": "O-M-Hp", "map_pf_none__G1": "O-M-Hp", "map_beam_steer__G3": "F-M-Hp"}
HM = {"dr_steer__G3": "O-N-Hm", "map_pf_steer__G3": "O-M-Hm", "map_beam_steer__G3": "F-M-Hm"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FORBIDDEN = re.compile(r"dongle|route_id|\b[0-9a-f]{16}\b|lat[_ ]?0|lon[_ ]?0|the Controlled-supplement city|/data/", re.I)
TEXT = []


def rc():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def t(s):
    TEXT.append(s)
    return s


def save(fig, name, out):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def end_labels(ax, series, x_right, pad=1.06):
    """Direct labels at the right edge, pushed apart in log space so they do not collide."""
    ys = np.log10(np.array([max(y, 1e-3) for _, y, _ in series]))
    order = np.argsort(ys)
    lo, hi = ax.get_ylim()
    step = 0.055 * (np.log10(hi) - np.log10(lo))
    for a, b in zip(order[:-1], order[1:]):
        if ys[b] - ys[a] < step:
            ys[b] = ys[a] + step
    for i, (label, _, color) in enumerate(series):
        ax.text(x_right * pad, 10 ** ys[i], t(label), color=color, va="center", ha="left", fontsize=6.5, clip_on=False)


SHORT = {"p0_hold__G1": "hold", "dr_none__G1": "straight", "dr_steer__G3": "steer DR", "dr_steer_calib__G3": "calib DR",
         "learned_gru_s0__G3": "GRU", "dr_curvature__G5": "curv (F)", "map_pf_steer__G3": "PF steer", "map_pf_none__G1": "PF speed",
         "map_beam_steer__G3": "beam"}


def fig_error_vs_time(curves, out):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), sharey="row")
    for row, (win, xmax, unit, ymax, what) in enumerate([("w15", 15.0, 60.0, 3e4, "15-minute episodes"), ("full", 60.0, 60.0, 2e5, "whole-drive episodes")]):
        n_eps = int(curves[(curves.win == win) & (curves.method == "dr_steer__G3") & (curves.protocol == "O-N-Hp")].n.max() or 0)
        for col, (methods, title) in enumerate([(NOMAP, "CAN only, no map"), (MAP, "with road map")]):
            ax = axes[row, col]
            labels = []
            for m in methods:
                c = curves[(curves.method == m) & (curves.protocol == PROTO[m]) & (curves.win == win)].sort_values("t_bin_s")
                c = c[(c.t_bin_s <= xmax * unit) & (c.n >= 30)]
                if c.empty:
                    continue
                label, color, ls = STYLE[m]
                x = (c.t_bin_s + (0.5 if win != "full" else 15.0)) / unit
                ax.plot(x, c.median_m, color=color, ls=ls, label=t(label))
                if m in ("dr_steer__G3", "map_pf_steer__G3"):
                    ax.fill_between(x, c.q25_m, c.q75_m, color=color, alpha=0.12, lw=0)
                labels.append((SHORT[m], float(c.median_m.iloc[-1]), color))
            ax.set_yscale("log")
            ax.set_xlim(0, xmax)
            ax.set_ylim(1, ymax)
            ax.set_xlabel(t("elapsed time (min)"))
            if col == 0:
                ax.set_ylabel(t(f"median error (m)\n{what} (n = {n_eps:,})"))
            if row == 0:
                ax.set_title(t(title), loc="left")
            end_labels(ax, labels, xmax, pad=1.02)
            for r in (50, 100):
                ax.axhline(r, color=INK2, lw=0.5, ls=(0, (1, 3)))
    handles, labels = [], []
    for ax in axes.ravel():
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(t("start and heading known, main folds; bands: interquartile range of steering DR and map PF"), x=0.01, ha="left", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.42, hspace=0.35, top=0.9, bottom=0.17)
    save(fig, "fig1_error_vs_time", out)


def fig_heading_unknown(curves, out):
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    labels = []
    for m, proto in HM.items():
        label, color, ls = STYLE[m]
        for p, alpha, lw, suffix in ((proto, 1.0, 1.4, ", heading unknown"), (PROTO[m], 0.45, 0.9, ", heading known")):
            c = curves[(curves.method == m) & (curves.protocol == p) & (curves.win == "w15")].sort_values("t_bin_s")
            c = c[c.n >= 30]
            if c.empty:
                continue
            ax.plot((c.t_bin_s + 0.5) / 60.0, c.median_m, color=color, alpha=alpha, lw=lw, ls="-" if alpha == 1.0 else (0, (2, 2)), label=t(label + suffix))
            if alpha == 1.0:
                labels.append((label, float(c.median_m.iloc[-1]), color))
    ax.set_yscale("log"); ax.set_xlim(0, 15); ax.set_ylim(1, 3e4)
    ax.set_xlabel(t("elapsed time (min)")); ax.set_ylabel(t("median position error (m)"))
    ax.set_title(t("start known; heading unknown (solid) vs known (dashed); 15-minute episodes"), loc="left", fontsize=7.2)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, fontsize=6)
    end_labels(ax, labels, 15.0, pad=1.02)
    save(fig, "fig1b_heading_unknown", out)


def load_episodes(folds):
    cols = ["episode_id", "protocol", "method", "win", "eligible_main", "ade", "c50", "c100", "lost_road", "driver", "parent"]
    ep = pd.read_parquet(os.path.join(paths.SCORES, "full_v1", "episodes_all.parquet"), columns=cols)
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "fold", "platform"])
    ep = ep.merge(man, on="episode_id")
    train_plat = set(man.platform[man.fold == "train"].dropna())
    ep["platform_seen"] = ep.platform.isin(train_plat)
    return ep[ep.eligible_main & ep.fold.isin(folds)]


def fig_ade_cdf(ep, out):
    e = ep[ep.win == "w05"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, methods, title in zip(axes, (NOMAP, MAP), ("CAN only, no map", "with road map")):
        for m in methods:
            v = e[(e.method == m) & (e.protocol == PROTO[m])].ade.dropna().to_numpy()
            if len(v) == 0:
                continue
            v = np.sort(np.maximum(v, 1.0))
            label, color, ls = STYLE[m]
            ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=color, ls=ls, label=t(label))
            n_ep = len(v)
        ax.set_xscale("log"); ax.set_xlim(1, 2e4); ax.set_ylim(0, 1)
        ax.set_xlabel(t("episode ADE (m)"))
        for r in (50, 100):
            ax.axvline(r, color=INK2, lw=0.5, ls=(0, (1, 3)))
        ax.set_title(t(title), loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6)
    axes[0].set_ylabel(t("fraction of episodes"))
    fig.suptitle(t(f"5-minute episodes, start and heading known, main folds (n = {n_ep:,} episodes per method)"), x=0.01, ha="left", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.08, top=0.82)
    save(fig, "fig2_ade_cdf", out)


def fig_coverage(cov, out):
    c5 = cov[cov.win == "w05"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, methods, title in zip(axes, (NOMAP, MAP), ("CAN only, no map", "with road map")):
        for m in methods:
            c = c5[(c5.method == m) & (c5.protocol == PROTO[m])].sort_values("r_m")
            if c.empty:
                continue
            label, color, ls = STYLE[m]
            ax.plot(c.r_m, c.c_r, color=color, ls=ls, label=t(label))
        ax.set_xscale("log"); ax.set_xlim(1, 1e4); ax.set_ylim(0, 1)
        ax.set_xlabel(t("radius r (m)"))
        for r in (50, 100):
            ax.axvline(r, color=INK2, lw=0.5, ls=(0, (1, 3)))
        ax.set_title(t(title), loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=6)
    axes[0].set_ylabel(t("coverage c(r): fraction of seconds within r"))
    fig.suptitle(t("5-minute episodes, start and heading known, main folds"), x=0.01, ha="left", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.08, top=0.82)
    save(fig, "fig3_coverage", out)


def fig_example(data, out):
    files = [(w, os.path.join(data, f"example_trajectory_{w}.parquet")) for w in ("w05", "w15")]
    files = [(w, f) for w, f in files if os.path.exists(f)]
    if not files:
        return
    fig, axes = plt.subplots(1, len(files), figsize=(3.4 * len(files), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (w, f) in zip(axes, files):
        df = pd.read_parquet(f)
        ax.plot(df.ref_x_m / 1e3, df.ref_y_m / 1e3, color=INK, lw=1.6, label=t("GPS reference"))
        for m in ("dr_steer__G3", "map_pf_steer__G3", "map_beam_steer__G3"):
            key = f"{PROTO[m]}__{m}"
            if f"{key}__x_m" in df:
                label, color, ls = STYLE[m]
                ax.plot(df[f"{key}__x_m"] / 1e3, df[f"{key}__y_m"] / 1e3, color=color, lw=1.1, ls=ls, label=t(label))
        ax.plot([0], [0], marker="o", color=INK, ms=5, mfc="white", mew=1.2, ls="none", label=t("start (given)"))
        ax.set_aspect("equal")
        ax.set_xlabel(t("east of start (km)")); ax.set_ylabel(t("north of start (km)"))
        ax.set_title(t(f"{'5' if w == 'w05' else '15'}-minute episode"), loc="left")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=6.5, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(t("validation episodes in the episode frame (median map-PF case of their region; start and heading known)"), x=0.01, ha="left", fontsize=7.5, color=INK2)
    fig.subplots_adjust(wspace=0.3, top=0.88, bottom=0.2)
    save(fig, "fig4_example_trajectory", out)


def fig_strata(ep, out):
    e = ep[ep.win == "w05"]
    methods = ["dr_steer__G3", "map_pf_steer__G3", "map_beam_steer__G3"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    # (a) platform seen in training vs unseen: per-episode ADE
    ax = axes[0]
    pos, ticks, lab = 0, [], []
    for m in methods:
        label, color, _ = STYLE[m]
        for seen, name in ((True, "seen"), (False, "unseen")):
            v = e[(e.method == m) & (e.protocol == PROTO[m]) & (e.platform_seen == seen)].ade.dropna().to_numpy()
            if len(v) < 30:
                pos += 1
                continue
            bp = ax.boxplot([np.maximum(v, 1.0)], positions=[pos], widths=0.7, showfliers=False, patch_artist=True, whis=(10, 90))
            bp["boxes"][0].set(facecolor=color, alpha=0.45 if seen else 0.9, edgecolor=color, lw=0.8, hatch=None if seen else "///")
            for part in ("whiskers", "caps", "medians"):
                for art in bp[part]:
                    art.set(color=color if part != "medians" else INK, lw=0.9)
            ticks.append(pos); lab.append(t(f"{name}\n{len(v):,}"))
            pos += 1.15
        pos += 0.9
    ax.set_yscale("log"); ax.set_ylim(1, 3e4)
    ax.set_xticks(ticks); ax.set_xticklabels(lab, fontsize=6)
    ax.set_ylabel(t("episode ADE (m)"))
    ax.set_title(t("platform seen vs unseen (hatched) in training"), loc="left", fontsize=7.5)
    handles = [plt.Line2D([], [], color=STYLE[m][1], lw=3, label=t(STYLE[m][0])) for m in methods]
    ax.legend(handles=handles, loc="upper right", fontsize=6)
    # (b) per-driver median ADE
    ax = axes[1]
    pos, ticks, lab = 0, [], []
    for m in methods:
        label, color, _ = STYLE[m]
        d = e[(e.method == m) & (e.protocol == PROTO[m])].groupby("driver").ade.median().dropna()
        if len(d) < 5:
            continue
        bp = ax.boxplot([np.maximum(d.to_numpy(), 1.0)], positions=[pos], widths=0.7, showfliers=True, patch_artist=True, whis=(10, 90),
                        flierprops=dict(marker="o", ms=2.5, mfc=color, mec="none", alpha=0.7))
        bp["boxes"][0].set(facecolor=color, alpha=0.6, edgecolor=color, lw=0.8)
        for part in ("whiskers", "caps", "medians"):
            for art in bp[part]:
                art.set(color=color if part != "medians" else INK, lw=0.9)
        ticks.append(pos); lab.append(t(label.replace(", ", "\n").replace(" (offline)", "")))
        pos += 1
    ax.set_yscale("log"); ax.set_ylim(1, 3e4)
    ax.set_xticks(ticks); ax.set_xticklabels(lab, fontsize=6)
    ax.set_title(t(f"per-driver median ({len(d)} drivers)"), loc="left", fontsize=7.5)
    fig.suptitle(t("5-minute episodes, start and heading known, main folds"), x=0.01, ha="left", fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.25, top=0.8)
    save(fig, "fig5_strata", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--folds", nargs="+", default=["test", "few_shot_heldout"])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rc()
    curves = pd.read_parquet(os.path.join(args.data, "curve_time.parquet"))
    cov = pd.read_parquet(os.path.join(args.data, "coverage_curve.parquet"))
    fig_error_vs_time(curves, args.out)
    fig_heading_unknown(curves, args.out)
    fig_coverage(cov, args.out)
    fig_example(args.data, args.out)
    ep = load_episodes(args.folds)
    fig_ade_cdf(ep, args.out)
    fig_strata(ep, args.out)
    bad = [s for s in TEXT if FORBIDDEN.search(str(s))]
    assert not bad, f"forbidden text in figures: {bad}"
    print(f"identifier gate: 0 hits in {len(TEXT)} text elements")


if __name__ == "__main__":
    main()
