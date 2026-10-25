#!/usr/bin/env python3
"""m0_vehicle_params.py: public nominal vehicle parameters per PLATFORM from the logged carParams.

The released model label (`model_canon`, taken from the raw folder name) disagrees with the car's own fingerprint on
hundreds of drives, so static parameters are keyed by the platform the car reported (`carParams.carFingerprint`,
normalised), and every drive gets its platform in a public drive table. Mock fingerprints (dashcam / unrecognised car)
carry no parameters.

Inputs (private): <bench>/private/carparams_by_drive.csv (fast pass) and <native>/driver_XXX/drive_YYY/carparams.json
(full re-decode, majority non-mock fingerprint; overrides the fast pass).
Outputs (public): GPS/manifests/vehicle_params.csv (one row per platform: median wheelbase_m, steer_ratio,
center_to_front_m, mass_kg, counts, spread, opendbc nominal when the name matches) and GPS/manifests/drive_platform.csv
(driver, drive, platform, model_canon, label_agrees). No per-drive raw values beyond the platform name; no VIN.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

BENCH = os.environ.get("CANTRAJ_BENCH", "/data/backup/DriveDNA-GPS_bench")
PRIV_CSV = f"{BENCH}/private/carparams_by_drive.csv"
NATIVE = f"{BENCH}/native"
DRIVES = os.path.join(os.environ.get("CANTRAJ_DRIVEDNA", "."), "index", "drives.parquet")
DRIVES_EXTRA = "/data/backup/DriveDNA_extra/DriveDNA/index/drives.parquet"
OUT = os.path.join(os.environ.get("CANTRAJ_MANIFESTS", "manifests"), "vehicle_params.csv")
OUT_DRIVES = os.path.join(os.environ.get("CANTRAJ_MANIFESTS", "manifests"), "drive_platform.csv")
NUM = ["wheelbase", "steerRatio", "centerToFront", "mass", "steerRatioRear", "rotationalInertia", "tireStiffnessFactor"]


def norm(s):
    return re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def label_agrees(platform, model_canon):
    a, b = norm(platform), norm(model_canon)
    if a == b:
        return "exact"
    ta, tb = a.split("_"), b.split("_")
    if ta[0] == tb[0] and set(ta) & set(tb[1:] or tb):
        return "brand_token"
    return "mismatch"


def opendbc_specs():
    try:
        from opendbc.car.values import PLATFORMS
    except Exception as e:  # noqa: BLE001
        print("opendbc import failed:", e)
        return {}
    out = {}
    for name, p in PLATFORMS.items():
        s = p.config.specs
        out[norm(name)] = dict(odbc_wheelbase=s.wheelbase, odbc_steerRatio=s.steerRatio, odbc_mass=s.mass)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--out-drives", default=OUT_DRIVES)
    args = ap.parse_args()
    rows = []
    if os.path.exists(PRIV_CSV):
        cp = pd.read_csv(PRIV_CSV)
        cp = cp[cp["status"] == "ok"]
        rows.append(cp[["driver", "drive", "carFingerprint", "op_version"] + NUM].assign(src="fast_pass"))
    nat = []
    for p in glob.glob(f"{NATIVE}/driver_*/drive_*/carparams.json"):
        d = json.load(open(p))
        parts = p.split(os.sep)
        nat.append(dict(driver=parts[-3], drive=parts[-2], carFingerprint=d.get("carFingerprint"), op_version=d.get("op_version"),
                        src="redecode", **{k: d.get(k) for k in NUM}))
    if nat:
        rows.append(pd.DataFrame(nat))
    if not rows:
        raise SystemExit("no carParams inputs found")
    allp = pd.concat(rows, ignore_index=True)
    allp["src_rank"] = (allp["src"] == "redecode").astype(int)
    allp = allp.sort_values("src_rank").drop_duplicates(["driver", "drive"], keep="last")
    allp["platform"] = allp["carFingerprint"].map(norm)
    allp.loc[allp["platform"].str.lower() == "mock", NUM] = np.nan
    idx = pd.read_parquet(DRIVES, columns=["driver", "drive", "model_canon"])
    if os.path.exists(DRIVES_EXTRA):
        try:
            idx = pd.concat([idx, pd.read_parquet(DRIVES_EXTRA, columns=["driver", "drive", "model_canon"])], ignore_index=True)
        except Exception:
            pass
    allp = allp.merge(idx, on=["driver", "drive"], how="left")
    allp["label_agrees"] = [label_agrees(p, m) for p, m in zip(allp["platform"], allp["model_canon"].fillna(""))]
    drives = allp[["driver", "drive", "platform", "model_canon", "label_agrees", "src"]].sort_values(["driver", "drive"])
    os.makedirs(os.path.dirname(args.out_drives), exist_ok=True)
    drives.to_csv(args.out_drives, index=False)
    real = allp[allp["platform"].str.lower() != "mock"]
    g = real.groupby("platform")
    spread = lambda s: float(np.nanmax(s) - np.nanmin(s)) if s.notna().any() else np.nan  # noqa: E731
    out = g.agg(n_drives=("drive", "size"), n_drivers=("driver", "nunique"),
                wheelbase_m=("wheelbase", "median"), steer_ratio=("steerRatio", "median"),
                center_to_front_m=("centerToFront", "median"), mass_kg=("mass", "median"),
                steer_ratio_rear=("steerRatioRear", "median"), rotational_inertia=("rotationalInertia", "median"),
                tire_stiffness_factor=("tireStiffnessFactor", "median"),
                wheelbase_spread_m=("wheelbase", spread), steer_ratio_spread=("steerRatio", spread),
                n_versions=("op_version", "nunique"),
                model_labels=("model_canon", lambda s: ";".join(sorted(set(map(str, s.dropna()))))[:200])).reset_index()
    specs = opendbc_specs()
    for c in ("odbc_wheelbase", "odbc_steerRatio", "odbc_mass"):
        out[c] = out["platform"].map(lambda n: specs.get(n, {}).get(c))
    out["brand"] = out["platform"].str.split("_").str[0]
    out["source"] = "carparams_median"
    # training-fold calibration columns are written by m0_sign_units.py; preserve existing values on rebuild
    calib_cols = ["yawrate_unit", "yawrate_sign", "angle_offset_train_deg", "steer_ratio_eff_train", "n_train_calib_drives"]
    prev = pd.read_csv(args.out) if os.path.exists(args.out) else None
    for c in calib_cols:
        if prev is not None and c in prev.columns and "platform" in prev.columns:
            out[c] = out["platform"].map(prev.set_index("platform")[c].to_dict())
        else:
            out[c] = ""
    for c in ("wheelbase_m", "steer_ratio", "center_to_front_m", "mass_kg", "steer_ratio_rear", "rotational_inertia",
              "tire_stiffness_factor", "wheelbase_spread_m", "steer_ratio_spread"):
        out[c] = out[c].astype(float).round(4)
    out = out.sort_values("n_drives", ascending=False)
    out.to_csv(args.out, index=False)
    n_mock = int((allp["platform"].str.lower() == "mock").sum())
    print(f"drives with carParams {len(allp)} (mock {n_mock}); platforms {len(out)}; opendbc match {int(out.odbc_wheelbase.notna().sum())}")
    print("label agreement:", allp["label_agrees"].value_counts().to_dict())
    print(f"platforms with steer-ratio spread > 0.5: {int((out.steer_ratio_spread > 0.5).sum())}; wheelbase spread > 0.05 m: {int((out.wheelbase_spread_m > 0.05).sum())}")
    print(out.head(10)[["platform", "n_drives", "n_drivers", "wheelbase_m", "steer_ratio", "mass_kg", "odbc_wheelbase", "odbc_steerRatio", "steer_ratio_spread"]].to_string(index=False))
    print("written", args.out, "and", args.out_drives)


if __name__ == "__main__":
    main()
