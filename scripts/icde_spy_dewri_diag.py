#!/usr/bin/env python3
"""Diagnostics for the two speed-only / steering-only path searches added last (run icde_v11, test + few-shot).

(a) Dewri et al. vs Elastic Pathing: both search road paths from the known start with 1 Hz speed only; share of the
    windows where both have an output and their ADEs agree within 0.5 m (the same path and placement in practice).
(b) Route recovery by the number of turns of the reference path, fifteen-minute windows (a turn = a change of the
    reference heading by more than 45 deg within 10 s, from scripts/paper_curves.py failure_events.parquet; analysis
    only, never an input). Failure-aware as the paper tables: an expected window without output counts as not recovered.

    python scripts/icde_spy_dewri_diag.py [--run icde_v11]   ->  reports/icde/tables/spy_dewri_diag.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icde_tables as T  # noqa: E402
from cantraj import paths  # noqa: E402

SAME = [("dewri__G1", "elastic_pathing__G1", "F-M-Hm", "G1", False), ("dewri_hp__G1", "elastic_pathing_hp__G1", "F-M-Hp", "G1", True)]
TURN_ROWS = [("spy_hp__G3", "F-M-Hp", "G3", True), ("spy__G3", "F-M-Hm", "G3", False), ("map_beam_steer__G3", "F-M-Hp", "G3", False),
             ("dewri_hp__G1", "F-M-Hp", "G1", True), ("elastic_pathing_hp__G1", "F-M-Hp", "G1", True)]
BINS, LABELS = [-1, 1, 3, 6, 10 ** 6], ["0-1", "2-3", "4-6", "7+"]
EVENTS = os.path.join(paths.GPS_DIR, "reports", "icde", "figures_data", "curves_v11", "failure_events.parquet")


def outputs(ep, stem, proto, win):
    return ep[(ep.method == stem) & (ep.protocol == proto) & (ep.win == win)].drop_duplicates("episode_id").set_index("episode_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="icde_v11")
    args = ap.parse_args()
    ep, man, _ = T.load(args.run)
    rows = []
    for win in ("w05", "w15"):
        for fold in T.FOLDS:
            for a, b, proto, req, hp in SAME:
                exp = T.expected(man, win, fold, proto, req, hp).episode_id
                j = outputs(ep, a, proto, win)[["ade"]].join(outputs(ep, b, proto, win)[["ade"]], lsuffix="_a", rsuffix="_b", how="inner")
                j = j[j.index.isin(exp)]
                same = (j.ade_a - j.ade_b).abs() < 0.5
                rows.append(dict(kind="same_output", stem=a, other=b, protocol=proto, win=win, fold=fold, bin="", n=len(j),
                                 value=float(same.mean())))
    turns = pd.read_parquet(EVENTS).drop_duplicates("episode_id").set_index("episode_id").n_turns
    for stem, proto, req, hp in TURN_ROWS:
        exp = pd.concat([T.expected(man, "w15", fold, proto, req, hp) for fold in T.FOLDS])
        exp = exp[exp.episode_id.isin(turns.index)]
        got = outputs(ep, stem, proto, "w15").route_ok.astype(float)
        ok = got.reindex(exp.episode_id).fillna(0.0).to_numpy()
        b = pd.cut(turns.reindex(exp.episode_id).to_numpy(), BINS, labels=LABELS)
        for lab in LABELS:
            m = np.asarray(b == lab)
            rows.append(dict(kind="route_by_turns", stem=stem, other="", protocol=proto, win="w15", fold="test+few-shot", bin=lab,
                             n=int(m.sum()), value=float(ok[m].mean()) if m.any() else np.nan))
        # where the path goes wrong on the simplest windows (at most one turn): median path recall / precision at 50 m
        # over the windows with an output (recall low with precision high = right road, too short a stretch of it)
        low = exp.episode_id[np.asarray(b == LABELS[0])]
        g = outputs(ep, stem, proto, "w15").reindex(low).dropna(subset=["ade"])
        for c in ("path_recall50", "path_prec50"):
            rows.append(dict(kind=f"lowturn_{c}", stem=stem, other="", protocol=proto, win="w15", fold="test+few-shot", bin=LABELS[0],
                             n=len(g), value=float(g[c].median())))
    t = pd.DataFrame(rows)
    out = os.path.join(T.OUT, "spy_dewri_diag.csv")
    t.round(4).to_csv(out, index=False)
    print(t.round(3).to_string(index=False))
    print("written", out)


if __name__ == "__main__":
    main()
