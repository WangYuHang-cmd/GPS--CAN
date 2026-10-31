#!/usr/bin/env python3
"""Validation-only selection table for values a published method leaves open (run icde_dev, val fold).

Same failure-aware statistics as the paper tables (scripts/icde_tables.py: an expected episode without output is a
failure: ADE = inf inside the median, rates = 0), restricted to the val fold and to the given stems. The test folds are
never read. Output: one CSV row per stem, sorted by the selection key (route_ok, then path recall at 50 m, then median
ADE; ties keep the listed order).

    python scripts/icde_val_select.py --stems spy_ma5r20__G3 spy_ma15r20__G3 --protocol F-M-Hm --req G3 \
        --drives-file manifests/icde_val_select_drives.csv --out reports/icde/spy_val_selection.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import icde_tables as T  # noqa: E402
from cantraj import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="icde_dev")
    ap.add_argument("--stems", nargs="+", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--win", default="w05")
    ap.add_argument("--req", default="G1", help="availability column (G1, G3, ...) or 'pedal'")
    ap.add_argument("--heading-only", action="store_true")
    ap.add_argument("--drives-file", default=os.path.join("manifests", "icde_val_select_drives.csv"),
                    help="CSV with driver, drive: the selection drives the runs used (default: the 60 val drives)")
    ap.add_argument("--key", nargs="+", default=["route_ok", "path_recall50", "ade_median"],
                    help="sort keys; rates descending, ADE ascending")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    T.FOLDS = {"val": "val"}                     # load() keeps only these folds of the manifest
    ep, man, skipped = T.load(args.run)
    drives = pd.read_csv(args.drives_file if os.path.isabs(args.drives_file) else os.path.join(paths.GPS_DIR, args.drives_file))
    man = man.merge(drives[["driver", "drive"]].drop_duplicates(), on=["driver", "drive"])
    rows = []
    for stem in args.stems:
        r = T.summarise(ep, man, skipped, stem, args.protocol, args.win, "val", args.req, args.heading_only, args.n_boot)
        if r is None:
            sys.exit(f"no expected val episodes for {stem}")
        rows.append(r)
    t = pd.DataFrame(rows)
    t["order"] = range(len(t))
    asc = [k.startswith("ade") for k in args.key]
    t = t.sort_values(args.key + ["order"], ascending=asc + [True]).drop(columns="order")
    cols = ["stem", "n_expected", "n_fail", "n_drivers", "ade_median", "ade_lo", "ade_hi", "c100", "route_ok", "path_recall50", "fail_reasons"]
    t = t[[c for c in cols if c in t.columns]]
    out = args.out if os.path.isabs(args.out) else os.path.join(paths.GPS_DIR, args.out)
    t.round(3).to_csv(out, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
