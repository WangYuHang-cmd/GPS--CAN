#!/usr/bin/env python3
"""Fixed cohort for the tool-using LLM agent study (supplementary; benchmark v1.1).

Rule (deterministic, frozen before any agent run): for every drive of the requested folds, the FIRST eligible
five-minute window (smallest k) that is also map_ok (start on the v2 map, every tile present); one window per drive.
No error, success or output of any method is consulted. The manifest written here carries no GPS-derived column.

    python scripts/build_agent_cohort.py                       # test + few_shot_heldout -> manifests/icde_agent_cohort.csv
    python scripts/build_agent_cohort.py --folds val --out manifests/icde_agent_val.csv --pilot 40
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402

TABLES = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def build(folds, win="w05"):
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"),
                          columns=["episode_id", "driver", "drive", "win", "k", "t0_s", "fold", "eligible_main"])
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"), columns=["episode_id", "map_ok"])
    m = man[(man.win == win) & man.eligible_main & man.fold.isin(folds)].merge(me, on="episode_id", how="left")
    m["map_ok"] = m.map_ok.fillna(False).astype(bool)
    m = m[m.map_ok].sort_values(["driver", "drive", "k"], kind="stable")
    first = m.groupby(["driver", "drive"], sort=True).head(1)
    return first[["episode_id", "driver", "drive", "fold", "k", "t0_s"]].reset_index(drop=True)


def counts(c):
    rows = []
    for fold, g in c.groupby("fold", sort=True):
        rows.append(dict(fold=fold, n_windows=len(g), n_drives=g[["driver", "drive"]].drop_duplicates().shape[0], n_drivers=g.driver.nunique()))
    rows.append(dict(fold="all", n_windows=len(c), n_drives=c[["driver", "drive"]].drop_duplicates().shape[0], n_drivers=c.driver.nunique()))
    return pd.DataFrame(rows)


def pilot(c, n, per_driver=4):
    """Deterministic pilot subset: drives in (driver, drive) order, at most `per_driver` per driver, until n windows."""
    out, used = [], {}
    for _, r in c.sort_values(["driver", "drive"], kind="stable").iterrows():
        if used.get(r.driver, 0) >= per_driver:
            continue
        used[r.driver] = used.get(r.driver, 0) + 1
        out.append(r)
        if len(out) >= n:
            break
    return pd.DataFrame(out).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", nargs="+", default=["test", "few_shot_heldout"])
    ap.add_argument("--out", default=os.path.join(paths.MANIFESTS, "icde_agent_cohort.csv"))
    ap.add_argument("--counts-out", default=None, help="default: reports/icde/tables/agent_cohort_counts[_<tag>].csv")
    ap.add_argument("--pilot", type=int, default=0, help="also write a deterministic pilot subset of N windows (<= 4 per driver)")
    ap.add_argument("--pilot-out", default=os.path.join(paths.MANIFESTS, "icde_agent_pilot.csv"))
    args = ap.parse_args()
    c = build(args.folds)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    c.to_csv(args.out, index=False)
    tag = "" if args.folds == ["test", "few_shot_heldout"] else "_" + "_".join(args.folds)
    cnt_path = args.counts_out or os.path.join(TABLES, f"agent_cohort_counts{tag}.csv")
    os.makedirs(os.path.dirname(cnt_path), exist_ok=True)
    cnt = counts(c)
    cnt.to_csv(cnt_path, index=False)
    print(cnt.to_string(index=False))
    print(f"{args.out}: {len(c)} windows, sha256 {sha256(args.out)}")
    if args.pilot:
        p = pilot(c, args.pilot)
        p.to_csv(args.pilot_out, index=False)
        print(f"{args.pilot_out}: {len(p)} windows, {p.driver.nunique()} drivers, sha256 {sha256(args.pilot_out)}")


if __name__ == "__main__":
    main()
