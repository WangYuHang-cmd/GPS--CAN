#!/usr/bin/env python3
"""build_learn_cache.py: per-drive caches for the learned motion models (train and val folds only).

    python scripts/build_learn_cache.py --workers 6 [--folds train val] [--limit N]
Writes /data/backup/DriveDNA-GPS_bench/learn/v1/grid10/<driver>__<drive>.npz and index.parquet.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths  # noqa: E402
from cantraj.learn.data import LEARN_DIR, drive_cache  # noqa: E402
from cantraj.splits import drive_folds  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _job(args):
    driver, drive, fold, static = args
    try:
        return drive_cache(driver, drive, fold, static)
    except Exception as e:  # noqa: BLE001
        print("skip", driver, drive, type(e).__name__, str(e)[:80], flush=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["train", "val"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    import importlib.util
    spec = importlib.util.spec_from_file_location("rb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_baseline.py"))
    rb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rb)
    get_static = rb.static_lookup()
    man = io.load_manifest()
    folds = drive_folds()
    av = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet")).set_index(["driver", "drive"])
    ok = man[(man.status == "ok") & (man.set == "main")]
    jobs = []
    for d, r in zip(ok.driver, ok.drive):
        fold = folds.get((d, r))
        if fold not in args.folds:
            continue
        if (d, r) in av.index and not bool(av.loc[(d, r), "G3"]):
            continue
        st, plat = get_static(d, r)
        if st.wheelbase_m is None or st.steer_ratio is None:
            continue
        s = dict(platform=plat, brand=st.brand, wheelbase_m=st.wheelbase_m, steer_ratio=st.steer_ratio, angle_offset_train_deg=st.angle_offset_train_deg, steer_ratio_eff_train=st.steer_ratio_eff_train)
        jobs.append((d, r, fold, s))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} drives to cache", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, out in enumerate(ex.map(_job, jobs, chunksize=4)):
            if out:
                rows.append(out)
            if i % 200 == 0:
                print(i, flush=True)
    idx = pd.DataFrame(rows)
    os.makedirs(LEARN_DIR, exist_ok=True)
    idx.to_parquet(os.path.join(LEARN_DIR, "index.parquet"), index=False)
    print(idx.groupby("fold").agg(drives=("drive", "size"), hires=("hires", "sum"), ticks=("n_ticks", "sum"), anchors=("n_anchor", "sum")).to_string())
    print("written", os.path.join(LEARN_DIR, "index.parquet"))


if __name__ == "__main__":
    main()
