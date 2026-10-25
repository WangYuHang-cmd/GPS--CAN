#!/usr/bin/env python3
"""m0_availability.py: per-drive, per-signal availability = valid / unsupported / undetermined.

Evidence: (1) which brand decoders assign the field (opendbc carstate.py of the local 2024-08 tree and the
sunnypilot 2024-10 tree, grep of `ret.<field> =`), (2) the car's own platform (manifests/drive_platform.csv) mapped to
a decoder brand, (3) activity on the released 10 Hz table (share of moving frames with a non-zero value, share finite),
(4) message presence. Non-zero activity alone is descriptive: a brake pedal is mostly zero while valid.
Rule: activity > 0.5 -> valid; activity == 0 and the brand decoder never assigns the field -> unsupported;
otherwise undetermined (field expected but silent, low activity, or unknown fork decoder).
Outputs: manifests/availability_manifest.parquet (long + wide G1..G8 flags per drive), reports/availability_summary.md.
"""
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths, registry  # noqa: E402

# brand decoders assigning the field (union of the two trees; grep 2026-10-22)
ASSIGNS = {
    "yaw_rate_can": {"ford", "hyundai", "volkswagen"},
    "gas": {"ford", "gm", "chrysler", "mazda", "nissan", "hyundai", "subaru", "volkswagen", "honda"},
    "brake": {"gm", "ford", "mazda", "chrysler", "volkswagen", "honda", "hyundai"},
    "steer_rate": {"chrysler", "gm", "honda", "hyundai", "mazda", "toyota", "subaru", "volkswagen"},
    "speed_cluster": {"honda", "hyundai", "toyota"},
    "gas_pressed": {"hyundai", "chrysler", "mazda", "toyota", "subaru", "ford", "gm", "honda", "nissan", "volkswagen"},
    "brake_pressed": {"ford", "gm", "honda", "mazda", "hyundai", "chrysler", "volkswagen", "nissan", "toyota", "subaru"},
}
BRAND = {"HYUNDAI": "hyundai", "KIA": "hyundai", "GENESIS": "hyundai", "VOLKSWAGEN": "volkswagen", "SKODA": "volkswagen",
         "AUDI": "volkswagen", "SEAT": "volkswagen", "CUPRA": "volkswagen", "CHEVROLET": "gm", "GMC": "gm", "CADILLAC": "gm",
         "BUICK": "gm", "HONDA": "honda", "ACURA": "honda", "TOYOTA": "toyota", "LEXUS": "toyota", "FORD": "ford", "LINCOLN": "ford",
         "CHRYSLER": "chrysler", "JEEP": "chrysler", "RAM": "chrysler", "DODGE": "chrysler", "NISSAN": "nissan", "INFINITI": "nissan",
         "SUBARU": "subaru", "MAZDA": "mazda"}
GRID_SIGNALS = {"speed": "vEgo", "accel_long": "aEgo", "steer_angle": "steeringAngleDeg", "steer_rate": "steeringRateDeg",
                "gas": "gas", "brake": "brake", "gas_pressed": "gasPressed", "brake_pressed": "brakePressed",
                "steer_pressed": "steeringPressed", "blinker_left": "leftBlinker", "blinker_right": "rightBlinker",
                "cruise_on": "cruiseState_enabled", "yaw_rate_can": "yawRate", "speed_cluster": "vEgoCluster",
                "curvature_loc": "actual_curvature", "yaw_rate_loc": "yaw_rate"}
BOOL = {"gas_pressed", "brake_pressed", "steer_pressed", "blinker_left", "blinker_right", "cruise_on"}


def classify(sig, brand, act, finite):
    if finite < 0.5:
        return "undetermined", "mostly_nan"
    if sig in BOOL:
        return ("valid", "toggles") if act > 0 else ("undetermined", "never_true")
    if act > 0.5:
        return "valid", "active"
    if sig in ASSIGNS:
        if brand is None:
            return "undetermined", "unknown_decoder"
        if brand not in ASSIGNS[sig]:
            return "unsupported", "decoder_never_assigns"
        return ("undetermined", "assigned_but_silent") if act == 0 else ("undetermined", "low_activity")
    return ("undetermined", "silent") if act == 0 else ("undetermined", "low_activity")


def main():
    man = io.load_manifest()
    ok = man[man.status == "ok"]
    idx = io.load_drive_index()[["driver", "drive", "model_canon"]]
    plat = pd.read_csv(os.path.join(paths.MANIFESTS, "drive_platform.csv"))[["driver", "drive", "platform"]]
    ok = ok.merge(idx, on=["driver", "drive"], how="left").merge(plat, on=["driver", "drive"], how="left")
    rows = []
    cols = list(GRID_SIGNALS.values())
    for i, r in enumerate(ok.itertuples(index=False)):
        try:
            t = pq.read_table(paths.grid_path(r.driver, r.drive), columns=cols).to_pandas()
        except Exception as e:  # noqa: BLE001
            print("skip", r.driver, r.drive, type(e).__name__)
            continue
        mv = (t["vEgo"].to_numpy(dtype=float) > 1.0)
        name = str(r.platform) if isinstance(r.platform, str) and r.platform.lower() != "mock" else str(r.model_canon)
        brand = BRAND.get(name.split("_")[0].upper())
        for sig, col in GRID_SIGNALS.items():
            x = t[col].to_numpy(dtype=float)
            finite = float(np.isfinite(x).mean())
            act = float(np.mean(np.nan_to_num(x[mv]) != 0)) if mv.any() else 0.0
            status, ev = classify(sig, brand, act, finite)
            rows.append(dict(driver=r.driver, drive=r.drive, signal=sig, status=status, evidence=ev, frac_finite=round(finite, 3),
                             frac_nonzero_moving=round(act, 3), brand=brand or "", platform=name))
        if i % 500 == 0:
            print(i, flush=True)
    av = pd.DataFrame(rows)
    wide = av.pivot(index=["driver", "drive"], columns="signal", values="status")
    for gid in ("G1", "G2", "G3", "G4", "G5", "G6", "G7"):
        spec = registry.group_spec(gid)
        req = [s for s in spec["requires"] if s in wide.columns]
        flag = np.ones(len(wide), dtype=bool)
        for s in req:
            flag &= (wide[s] == "valid").to_numpy()
        if gid == "G2":          # at least one pedal signal valid
            ped = (wide[["gas", "brake", "gas_pressed", "brake_pressed"]] == "valid").any(axis=1).to_numpy()
            flag &= ped
        wide[gid] = flag
    nat_path = os.path.join(paths.NATIVE, "native_manifest.csv")   # G8 = G7 and a native (rlog) re-decode with IMU >= 50 Hz
    if os.path.exists(nat_path):
        nat = pd.read_csv(nat_path)
        nat = nat[(nat.status == "ok") & (nat.imu_rate_hz.fillna(0) >= 50)]
        key = set(zip(nat.driver, nat.drive))
        wide["G8"] = wide["G7"].to_numpy() & np.array([(a, b) in key for a, b in zip(wide.index.get_level_values(0), wide.index.get_level_values(1))])
    else:
        wide["G8"] = False
    wide = wide.reset_index()
    os.makedirs(paths.MANIFESTS, exist_ok=True)
    av.to_parquet(os.path.join(paths.MANIFESTS, "availability_manifest.parquet"), index=False)
    wide.to_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"), index=False)
    wide.to_csv(os.path.join(paths.MANIFESTS, "availability_wide.csv"), index=False)
    summ = av.groupby(["signal", "status"]).size().unstack(fill_value=0)
    lines = ["# Signal availability (GPS-ok drives, released 10 Hz tables)\n", f"drives: {wide.shape[0]}\n", summ.to_markdown(), "",
             "## Group applicability (all `requires` valid)", pd.Series({g: int(wide[g].sum()) for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")}).to_frame("drives").to_markdown(), "",
             "## yaw_rate_can by brand", av[av.signal == "yaw_rate_can"].groupby(["brand", "status"]).size().unstack(fill_value=0).to_markdown(), "",
             "## gas / brake by brand", av[av.signal.isin(["gas", "brake"])].groupby(["signal", "brand", "status"]).size().unstack(fill_value=0).to_markdown()]
    os.makedirs(paths.REPORTS, exist_ok=True)
    with open(os.path.join(paths.REPORTS, "availability_summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(summ.to_string())
    print({g: int(wide[g].sum()) for g in ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G8")})
    print("written manifests/availability_manifest.parquet, availability_wide.parquet, reports/availability_summary.md")


if __name__ == "__main__":
    main()
