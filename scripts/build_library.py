#!/usr/bin/env python3
"""build_library.py: training-fold reference library for the retrieval and prior baselines.

Writes /data/backup/DriveDNA-GPS_bench/library/train_tracks.parquet: one row per 1 Hz reference point of every
TRAIN-fold GPS-ok drive (driver, drive, i, time_s, lat, lon, s_m = cumulative CAN distance, v_can) plus a 250 m cell key,
and prior_rho.json: median net-displacement / path-length ratio as a function of CAN distance (the start-prior baseline).
Only training drives enter the library; baselines load the file, never the GPS CSVs. Pilot runs on train drives use
leave-one-driver-out (handled in the baseline).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths  # noqa: E402
from cantraj.episodes import view_1hz  # noqa: E402
from cantraj.splits import drive_folds  # noqa: E402

LIB = f"{paths.BENCH}/library"
CELL_DEG = 0.00225


def main():
    man = io.load_manifest()
    folds = drive_folds()
    ok = man[(man.status == "ok") & (man.set == "main")]
    ok = ok[[folds.get((a, b)) == "train" for a, b in zip(ok.driver, ok.drive)]]
    rows, rho_rows = [], []
    for i, r in enumerate(ok.itertuples(index=False)):
        delta = io.time_compensation(r.gps_sources, r.gps_rate_hz)
        ref = view_1hz(io.load_gps(r.driver, r.drive, delta))
        if len(ref) < 30:
            continue
        g = pq.read_table(paths.grid_path(r.driver, r.drive), columns=["time_s", "vEgo"]).to_pandas()
        gt, gv = g.time_s.values, g.vEgo.values
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (gv[1:] + gv[:-1]) * np.diff(gt))])
        s = np.interp(ref.time_s.values, gt, cum)
        s -= s[0]
        lat, lon = ref.lat.values, ref.lon.values
        df = pd.DataFrame(dict(driver=r.driver, drive=r.drive, i=np.arange(len(ref)), time_s=ref.time_s.values, lat=lat, lon=lon,
                               s_m=s, v_can=np.interp(ref.time_s.values, gt, gv),
                               cx=np.round(lon * np.cos(np.radians(lat)) / CELL_DEG).astype(np.int64), cy=np.round(lat / CELL_DEG).astype(np.int64)))
        rows.append(df)
        # prior: net displacement from the first point as a function of CAN distance
        x = (lon - lon[0]) * 111320 * np.cos(np.radians(lat[0]))
        y = (lat - lat[0]) * 111320
        rho_rows.append(pd.DataFrame(dict(s_m=s, net=np.hypot(x, y))))
        if i % 300 == 0:
            print(i, flush=True)
    lib = pd.concat(rows, ignore_index=True)
    os.makedirs(LIB, exist_ok=True)
    lib.to_parquet(os.path.join(LIB, "train_tracks.parquet"), index=False)
    rho = pd.concat(rho_rows, ignore_index=True)
    bins = np.array([0, 250, 500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 1e9])
    rho["bin"] = np.digitize(rho.s_m, bins) - 1
    tab = rho[rho.s_m > 0].groupby("bin").apply(lambda d: float(np.median(d.net / d.s_m))).to_dict()
    out = dict(bins_m=bins[:-1].tolist(), rho=[tab.get(b, np.nan) for b in range(len(bins) - 1)], n_drives=int(lib.drive.nunique()))
    json.dump(out, open(os.path.join(LIB, "prior_rho.json"), "w"), indent=1)
    print(f"library: {len(lib)} points from {lib.drive.nunique()} train drives, {lib.driver.nunique()} drivers; rho:", [round(v, 2) for v in out["rho"]])


if __name__ == "__main__":
    main()
