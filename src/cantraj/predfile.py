"""Prediction file schema, writer and validator.

Columns: episode_id (str), t (float, reference time on the drive axis), pred_x_m, pred_y_m (float, episode ENU),
candidate_id (int16, 0 = Top-1), prob (float32, sums to 1 over candidates of an episode), optional pred_theta_rad.
Whole-path candidates: the same candidate ids at every t of an episode.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED = ["episode_id", "t", "pred_x_m", "pred_y_m", "candidate_id", "prob"]


def new_frame(episode_id, t, xy, theta=None, candidate_id=0, prob=1.0):
    xy = np.asarray(xy, dtype=float)
    df = pd.DataFrame({"episode_id": episode_id, "t": np.asarray(t, dtype=float), "pred_x_m": xy[:, 0],
                       "pred_y_m": xy[:, 1], "candidate_id": np.int16(candidate_id), "prob": np.float32(prob)})
    if theta is not None:
        df["pred_theta_rad"] = np.asarray(theta, dtype=float)
    return df


def write(df: pd.DataFrame, path: str, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = df.sort_values(["episode_id", "candidate_id", "t"], kind="stable").reset_index(drop=True)
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)
    with open(os.path.splitext(path)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1, default=str)
    return path


def read(path: str):
    return pq.read_table(path).to_pandas()


def validate(df: pd.DataFrame, expected: pd.DataFrame, allow_missing=False, atol=1e-6):
    """expected: DataFrame[episode_id, t] of the reference times that must be predicted (Top-1).
    Returns a report dict; raises ValueError on schema problems or (unless allow_missing) missing points."""
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"prediction file lacks columns {missing_cols}")
    if not np.isfinite(df[["pred_x_m", "pred_y_m"]].to_numpy(dtype=float)).all():
        raise ValueError("non-finite predictions")
    top1 = df[df["candidate_id"] == 0]
    key_pred = set(zip(top1["episode_id"], np.round(top1["t"] / atol).astype(np.int64)))
    key_exp = set(zip(expected["episode_id"], np.round(expected["t"] / atol).astype(np.int64)))
    missing = key_exp - key_pred
    extra = key_pred - key_exp
    report = {"n_expected": len(key_exp), "n_pred_top1": len(key_pred), "n_missing": len(missing), "n_extra": len(extra),
              "episodes_incomplete": sorted({e for e, _ in missing})}
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} expected reference points have no Top-1 prediction (first: {sorted(missing)[:3]})")
    # whole-path candidate consistency
    for eid, g in df.groupby("episode_id"):
        cands = g.groupby("candidate_id")["t"].apply(lambda s: tuple(np.round(s.values / atol).astype(np.int64)))
        if len(set(cands.values)) > 1:
            raise ValueError(f"episode {eid}: candidates do not share the same time set (whole-path rule)")
        probs = g.groupby("candidate_id")["prob"].first()
        if abs(float(probs.sum()) - 1.0) > 1e-3:
            raise ValueError(f"episode {eid}: candidate probabilities sum to {probs.sum():.4f}")
    return report
