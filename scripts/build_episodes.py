#!/usr/bin/env python3
"""build_episodes.py: select drives, build episodes per window class, write the public episode manifest.

    python scripts/build_episodes.py --pilot                # 20-drive pilot selection (fixed seed, rules below)
    python scripts/build_episodes.py --only driver_154/drive_008 --wins w05 full
    python scripts/build_episodes.py --all                  # every GPS-ok drive (M4)

Pilot rule: GPS-ok main-set drives with a native (rlog) re-decode available, align_method k_min*, duration 15 to 60 min,
not in the the Controlled-supplement city window, train fold; 10 u-blox >= 8 Hz + 10 Qualcomm 1 Hz, at least one Ford and one Hyundai, at
least three drives > 30 min. Output: manifests/episode_manifest.parquet (+ .csv) and manifests/pilot_drives.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
from cantraj.splits import drive_folds  # noqa: E402

TAMPA = dict(lat=(27.5, 28.6), lon=(-83.0, -82.0))


def native_available(driver, drive):
    return os.path.exists(os.path.join(paths.native_dir(driver, drive), "carstate.parquet"))


def pilot_selection(man, idx, folds, n_each=10, seed=20260722, require_native=True):
    m = man[(man.status == "ok") & (man.set == "main")].merge(idx[["driver", "drive", "model_canon"]], on=["driver", "drive"], how="left")
    m["fold"] = [folds.get((a, b), "") for a, b in zip(m.driver, m.drive)]
    m = m[(m.fold == "train") & m.align_method.str.startswith("k_min") & (m.align_cov >= 0.999) & m.duration_s.between(900, 3600)]
    m = m[~(m.lat_med.between(*TAMPA["lat"]) & m.lon_med.between(*TAMPA["lon"]))]
    if require_native:
        m = m[[native_available(a, b) for a, b in zip(m.driver, m.drive)]]
    rng = np.random.default_rng(seed)
    ub = m[(m.gps_sources == "ublox") & (m.gps_rate_hz >= 8)]
    qc = m[(m.gps_sources == "qcomdiag")]
    picks = []
    for pool, n in ((ub, n_each), (qc, n_each)):
        pool = pool.copy()
        pool["brand"] = pool.model_canon.fillna("").str.split("_").str[0]
        chosen = []
        for brand in ("FORD", "HYUNDAI"):                      # guarantee the CAN yawRate brands
            cand = pool[(pool.brand == brand) & ~pool.index.isin(chosen)]
            if len(cand):
                chosen.append(cand.sample(1, random_state=int(rng.integers(1e9))).index[0])
        long = pool[(pool.duration_s > 1800) & ~pool.index.isin(chosen)]
        for i in long.sample(min(2, len(long)), random_state=int(rng.integers(1e9))).index:
            chosen.append(i)
        rest = pool[~pool.index.isin(chosen)]
        # spread over drivers: one drive per driver first
        rest = rest.sample(frac=1.0, random_state=int(rng.integers(1e9))).drop_duplicates("driver")
        for i in rest.index:
            if len(chosen) >= n:
                break
            chosen.append(i)
        picks.append(pool.loc[chosen[:n]])
    sel = pd.concat(picks)
    return sel[["driver", "drive", "model_canon", "fold", "gps_sources", "gps_rate_hz", "duration_s", "op_version", "n_seg_rlog", "n_seg_qlog"]]


def _drive_rows(job):
    driver, drive, delta, layer, wins, dmeta, full_load = job
    rows = []
    try:
        dd = DriveData(driver, drive, delta_s=delta, input_layer=layer, light=not full_load)
    except Exception as e:  # noqa: BLE001
        print("skip", driver, drive, type(e).__name__, str(e)[:80], flush=True)
        return rows
    dmeta = dict(dmeta, log_native=bool(dd.has_native))
    for win in wins:
        for ep in dd.episodes(win, dmeta):
            a = ep.init_full
            rows.append(dict(episode_id=ep.episode_id, driver=driver, drive=drive, win=win, k=ep.k, t0_s=ep.t0, t_end_s=ep.t_end,
                             dur_s=ep.t_end - ep.t0, theta0_rad=a["theta0"], anchor_kind=a["anchor_kind"],
                             anchor_speed_mps=a["anchor_speed_mps"], anchor_bearing_acc_deg=a["anchor_bearing_acc_deg"],
                             anchor_h_acc_m=a["anchor_h_acc_m"], **ep.meta))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--pilot-no-native-filter", action="store_true", help="select pilot drives before their native tables exist")
    ap.add_argument("--reselect", action="store_true", help="ignore the frozen manifests/pilot_drives.csv and select again")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fold", action="append", default=[], help="restrict --all to these folds (train/val/test/few_shot_heldout/supplement)")
    ap.add_argument("--version", default="", help="benchmark version tag written into the manifest (e.g. v1)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("--wins", nargs="+", default=["w01", "w05", "w15", "full"])
    ap.add_argument("--layer", default="events", choices=["events", "grid_compat"])
    ap.add_argument("--time-comp", default="train", choices=["train", "none"], help="GPS time alignment compensation (training-fold table) or none")
    ap.add_argument("--out", default=os.path.join(paths.MANIFESTS, "episode_manifest.parquet"))
    args = ap.parse_args()
    man = io.load_manifest()
    idx = io.load_drive_index()
    folds = drive_folds()
    if args.pilot:
        frozen = os.path.join(paths.MANIFESTS, "pilot_drives.csv")
        if os.path.exists(frozen) and not args.reselect:
            sel = pd.read_csv(frozen)
            print(f"using the frozen pilot set {frozen}")
        else:
            sel = pilot_selection(man, idx, folds, require_native=not args.pilot_no_native_filter)
            sel.to_csv(frozen, index=False)
        print(f"pilot drives: {len(sel)}  drivers {sel.driver.nunique()}  ublox>=8Hz {int((sel.gps_rate_hz >= 8).sum())}  >30 min {int((sel.duration_s > 1800).sum())}")
        print(sel.to_string(index=False))
        drives = list(zip(sel.driver, sel.drive))
    elif args.only:
        drives = [tuple(o.split("/")) for o in args.only]
    else:
        ok = man[man.status == "ok"]
        if args.fold:
            ok = ok[[folds.get((a, b), "supplement" if s_ == "controlled" else "unassigned") in args.fold for a, b, s_ in zip(ok.driver, ok.drive, ok.set)]]
        drives = list(zip(ok.driver, ok.drive))
    rows = []
    platform = pd.read_csv(os.path.join(paths.MANIFESTS, "drive_platform.csv")).set_index(["driver", "drive"]) if os.path.exists(os.path.join(paths.MANIFESTS, "drive_platform.csv")) else None
    mi = idx.set_index(["driver", "drive"])
    mm = man.set_index(["driver", "drive"])
    jobs = []
    for driver, drive in drives:
        r = mm.loc[(driver, drive)]
        delta = io.time_compensation(r["gps_sources"], r["gps_rate_hz"]) if args.time_comp == "train" else 0.0
        plat = platform.loc[(driver, drive), "platform"] if platform is not None and (driver, drive) in platform.index else None
        dmeta = dict(model_canon=mi.loc[(driver, drive), "model_canon"] if (driver, drive) in mi.index else None,
                     fold=folds.get((driver, drive), "supplement" if r["set"] == "controlled" else "unassigned"), platform=plat,
                     gps_rate_hz=float(r["gps_rate_hz"]), gps_source=str(r["gps_sources"]), set=str(r["set"]))
        jobs.append((driver, drive, delta, args.layer, args.wins, dmeta, args.pilot or bool(args.only)))
    if args.workers > 1 and len(jobs) > 8:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for out in ex.map(_drive_rows, jobs, chunksize=4):
                rows.extend(out)
    else:
        for j in jobs:
            rows.extend(_drive_rows(j))
    df = pd.DataFrame(rows)
    if len(df):
        # main-board admission (benchmark v1): mock / unconfirmed platforms and the Controlled supplement stay out of the CAN-only main board
        unconfirmed = df["platform"].isna() | df["platform"].astype(str).str.lower().eq("mock")
        df["admission"] = np.where(~df["eligible_main"], "ineligible_window", np.where(unconfirmed, "platform_unconfirmed", np.where(df["set"] == "controlled", "supplement", "main")))
        df["eligible_main"] = df["eligible_main"] & ~unconfirmed & (df["set"] != "controlled")
        df["benchmark_version"] = args.version
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    df.to_csv(os.path.splitext(args.out)[0] + ".csv", index=False)
    if len(df):
        print(df.groupby("win").agg(episodes=("episode_id", "size"), eligible=("eligible_main", "sum"), drives=("drive", "nunique"),
                                    med_offset_s=("t0_offset_s", "median"), p90_offset_s=("t0_offset_s", lambda s: s.quantile(0.9))).round(1).to_string())
        print(df.groupby(["win", "admission"]).size().unstack(fill_value=0).to_string())
        print(df[df.eligible_main].groupby(["win", "fold"]).size().unstack(fill_value=0).to_string())
    print("written", args.out)


if __name__ == "__main__":
    main()
