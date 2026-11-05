#!/usr/bin/env python3
"""Table IV of the paper (paper_icde/tables/agent.tex) with both agent configurations: the frozen study (reports/icde/tables/agent_w05.csv,
rows beam / selector / agent) and the extended configuration (reports/icde/tables/agent_max_w05.csv, row agent -> "agent, extended").
Numbers are read from the two CSVs written by icde_agent_tables.py; nothing is recomputed, so the frozen rows stay bit-identical.

    python scripts/icde_agent_table_tex.py                      # writes paper_icde/tables/agent.tex
    python scripts/icde_agent_table_tex.py --no-extended        # the frozen three-row version
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cantraj import paths  # noqa: E402

TABLES = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
TEX = os.path.join(paths.GPS_DIR, "paper_icde", "tables")
FOLDS = [("test", "test"), ("few_shot_heldout", "few-shot")]


def fmt_m(x):
    if not np.isfinite(x):
        return "fail"
    return f"{x:,.0f}".replace(",", "{,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", default=os.path.join(TABLES, "agent_w05.csv"))
    ap.add_argument("--extended", default=os.path.join(TABLES, "agent_max_w05.csv"))
    ap.add_argument("--no-extended", action="store_true")
    ap.add_argument("--out", default=os.path.join(TEX, "agent.tex"))
    args = ap.parse_args()
    fr = pd.read_csv(args.frozen)
    ex = None if args.no_extended or not os.path.exists(args.extended) else pd.read_csv(args.extended)
    cap = r"Agent study (one five-minute window per held-out drive): beam (A), supervised selector (B), Qwen3-14B agent (C, three seeds pooled)"
    if ex is not None:
        cap += r"; B$^{+}$ and C$^{+}$: the extended configuration (ten reconstructor presets, comparison and scoring tools, twelve turns)"
    cap += r"; route recovery, failure-aware median ADE, median reconstructed serial cost; 95\,\% driver-clustered CIs; every window produced an output."
    lines = [r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{3.5pt}", r"\caption{" + cap + r"}\label{tab:agent}",
             r"\begin{tabular}{llrrrr}\toprule", r"fold & method & $n$ & route \% [CI] & ADE m [CI] & cost s \\\midrule"]

    def line(flab, lab, r):
        cost = r.get("cost_reconstructed_med", np.nan)
        return (f"{flab} & {lab} & {int(r.n)} & {r.route_pct:.0f} [{r.route_lo:.0f}, {r.route_hi:.0f}] & "
                f"{fmt_m(r.ade_median)} [{fmt_m(r.ade_lo)}, {fmt_m(r.ade_hi)}] & {cost:.1f} \\\\")

    for i, (fold, flab) in enumerate(FOLDS):
        for key, lab in (("beam", "beam (A)"), ("selector", "selector (B)"), ("agent", "agent (C)")):
            r = fr[(fr.row == key) & (fr.fold == fold)]
            if len(r):
                lines.append(line(flab, lab, r.iloc[0]))
        if ex is not None:
            for key, lab in (("selector", r"selector, every preset (B$^{+}$)"), ("agent", r"agent, extended (C$^{+}$)")):
                r = ex[(ex.row == key) & (ex.fold == fold)]
                if len(r):
                    lines.append(line(flab, lab, r.iloc[0]))
        if i == 0:
            lines.append(r"\midrule")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines) - 6} rows -> {args.out}")


if __name__ == "__main__":
    main()
