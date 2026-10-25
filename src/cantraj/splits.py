"""Split assignment. split_main = DriveDNA driver-disjoint folds (train / val / test / few_shot_heldout); the other
splits (seen_route, cross_route_within_driver, cross_metro) are produced by scripts/m0_gps_quality.py."""
from __future__ import annotations

import pandas as pd

from . import io


def drive_folds():
    idx = io.load_drive_index()
    f = idx[["driver", "drive", "fold"]].copy() if "fold" in idx.columns else idx[["driver", "drive"]].assign(fold="")
    f["fold"] = f["fold"].fillna("").replace("", "unassigned")
    return f.set_index(["driver", "drive"])["fold"].to_dict()


def split_main(driver, drive, folds=None):
    folds = folds if folds is not None else drive_folds()
    return folds.get((driver, drive), "unassigned")
