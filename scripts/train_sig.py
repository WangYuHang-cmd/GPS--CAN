#!/usr/bin/env python3
"""Signal-combination study (ICDE RQ2): train the same small GRU (learn/sig.py) on S1 / S2c / S2b / S3 / S4c / S4b with one
budget on one set of drives: train / val drives of the main learning cache whose accelerator pedal AND steering are
valid (availability manifest), so every set sees the same data. Models: $CANTRAJ_BENCH/models/learn/sig/<set>/seed<k>.

    python scripts/train_sig.py cache [--workers 3]          # pedal channels on the cache grid (train / val only)
    python scripts/train_sig.py train --groups S1 S2c S3 S4c --seeds 0 1 2 [--epochs 20 --steps 150]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402
from cantraj.learn.data import LEARN_DIR  # noqa: E402

OUT = os.path.join(paths.BENCH, "models", "learn", "sig")


def study_index():
    ix = pd.read_parquet(os.path.join(LEARN_DIR, "index.parquet"))
    aw = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"))[["driver", "drive", "gas", "brake", "steer_angle"]]
    ix = ix.merge(aw, on=["driver", "drive"], how="left")
    ix["gas_valid"] = ix.gas == "valid"
    ix["brake_valid"] = ix.brake == "valid"
    return ix[ix.gas_valid & (ix.steer_angle == "valid")].reset_index(drop=True)


def _cache(row):
    from cantraj.learn.sig_train import pedal_cache
    try:
        pedal_cache(row[0], row[1], row[2])
        return None
    except Exception as e:  # noqa: BLE001
        return f"{row[0]}/{row[1]}: {type(e).__name__}: {str(e)[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cache", "train"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--groups", nargs="+", default=["S1", "S2c", "S3", "S4c"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    ix = study_index()
    print(f"study drives: {len(ix)} ({ix.fold.value_counts().to_dict()}), brands {ix.brand.value_counts().head(8).to_dict()}", flush=True)
    if args.cmd == "cache":
        t0 = time.time()
        with Pool(args.workers) as pool:
            errs = [e for e in pool.imap_unordered(_cache, list(zip(ix.driver, ix.drive, ix.fold)), chunksize=4) if e]
        print(f"pedal cache done in {time.time() - t0:.0f}s, {len(errs)} errors {errs[:3]}", flush=True)
        return
    from cantraj.learn.sig_train import brand_scales, train_sig
    cfg = yaml.safe_load(open(os.path.join(paths.GPS_DIR, "configs", "baselines", "learn.yaml")))
    scales = brand_scales(ix)
    print("brand pedal scales (train):", {k: [round(x, 3) if x is not None else None for x in v] for k, v in scales.items()}, flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump({k: list(v) for k, v in scales.items()}, open(os.path.join(OUT, "brand_scales.json"), "w"), indent=1)
    for g in args.groups:
        for s in args.seeds:
            out = os.path.join(OUT, g, f"seed{s}")
            if os.path.exists(os.path.join(out, "meta.json")):
                print(f"{g} seed {s}: exists", flush=True)
                continue
            t0 = time.time()
            meta = train_sig(g, s, cfg, ix, scales, out, args.epochs, args.steps, log=lambda m: print(m, flush=True))
            print(f"{g} seed {s}: best val {meta['best_val_loss']:.3f} in {time.time() - t0:.0f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
