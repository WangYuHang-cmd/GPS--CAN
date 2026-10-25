#!/usr/bin/env python3
"""train_motion.py: train one learned motion model.

    python scripts/train_motion.py --model gru --seed 0 [--epochs 40] [--quick]
Models: gru (causal increment GRU), resid (physics residual GRU), resid_nogru (calibration-only control b').
Outputs /data/backup/DriveDNA-GPS_bench/models/learn/<model>/seed<k>/{model.pt, weights.npz, meta.json}.
"""
import argparse
import os
import sys

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402
from cantraj.learn.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gru", "resid", "resid_nogru"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config", default=os.path.join(paths.GPS_DIR, "configs", "baselines", "learn.yaml"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--augment-speed-noise", type=float, default=0.0, help="training augmentation: Gaussian speed noise std (m/s) on the speed-derived features")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    models_dir = cfg["paths"]["models"] if os.path.isabs(cfg["paths"]["models"]) else os.path.join(paths.BENCH, cfg["paths"]["models"])
    out = args.out or os.path.join(models_dir, args.model, f"seed{args.seed}")
    meta = train(args.model, args.seed, cfg, out, epochs=args.epochs, quick=args.quick, steps_per_epoch=args.steps_per_epoch, augment_speed_noise=args.augment_speed_noise)
    print("done", out, "best val loss", meta["best_val_loss"])


if __name__ == "__main__":
    main()
