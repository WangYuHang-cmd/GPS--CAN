"""Loaders. `load_gps` is the ONLY reader of reference positions; baselines never import it (run_baseline.py replaces it
with a function that raises, and test_whitelist checks that)."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import paths

GPS_COLS = ["time_s", "source", "latitude", "longitude", "speed_mps", "bearing_deg", "bearing_acc_deg", "h_acc_m", "fix_ok"]


def load_gps(driver, drive, delta_s=0.0):
    """Valid GPS fixes of a drive on the released time axis. `delta_s`: training-fold time alignment compensation
    for this drive's receiver stratum (manifests/time_alignment.json); time_s = time_s_raw - delta_s, i.e. a fix
    stamped at t describes the vehicle state at CAN time t - delta_s. 0.0 = no compensation."""
    df = pd.read_csv(paths.gps_path(driver, drive), usecols=GPS_COLS)
    df = df[df["fix_ok"] == 1].copy()
    df = df.rename(columns={"time_s": "time_s_raw", "latitude": "lat", "longitude": "lon"})
    df["time_s"] = df["time_s_raw"].to_numpy(dtype=float) - float(delta_s or 0.0)
    return df.sort_values("time_s", kind="stable").reset_index(drop=True)


def time_compensation(source, gps_rate_hz, table=None):
    """Delta for a drive from manifests/time_alignment.json (stratum = source + _hires/_lowrate); 0 if unknown."""
    import json
    if table is None:
        p = os.path.join(paths.MANIFESTS, "time_alignment.json")
        if not os.path.exists(p):
            return 0.0
        table = json.load(open(p))["delta_s"]
    key = f"{source}_{'hires' if (gps_rate_hz or 0) >= 8 else 'lowrate'}"
    return float(table.get(key, table.get(f"{str(source).split(';')[-1]}_{'hires' if (gps_rate_hz or 0) >= 8 else 'lowrate'}", 0.0)))


def load_grid(driver, drive, columns=None):
    return pq.read_table(paths.grid_path(driver, drive), columns=columns).to_pandas()


def load_native(driver, drive, tables=("carstate",)):
    d = paths.native_dir(driver, drive)
    out = {}
    for t in tables:
        p = os.path.join(d, f"{t}.parquet")
        if os.path.exists(p):
            if t == "imu":        # only the calibrated gyro and the accelerometer rows (a 16 h drive has 30 M IMU rows otherwise)
                out[t] = pq.read_table(p, columns=["t_s", "kind", "which", "v0", "v1", "v2"], filters=[("kind", "in", ["gyro", "accel", "gyroUncalibrated"])]).to_pandas()
            else:
                out[t] = pq.read_table(p).to_pandas()
    return out


def load_manifest():
    return pd.read_csv(paths.GPS_MANIFEST)


def load_drive_index():
    idx = pd.read_parquet(paths.DRIVES_INDEX)
    if os.path.exists(paths.DRIVES_INDEX_EXTRA):
        try:
            ex = pd.read_parquet(paths.DRIVES_INDEX_EXTRA)
            ex = ex[[c for c in ex.columns if c in idx.columns]]
            idx = pd.concat([idx, ex], ignore_index=True)
        except Exception:
            pass
    return idx
