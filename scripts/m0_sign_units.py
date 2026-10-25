#!/usr/bin/env python3
"""m0_sign_units.py: training-fold calibration tables (never touches val/test drives).

A. Heading-rate sources vs GPS course rate (10 Hz u-blox train drives): correlation and regression slope of the
   left-positive GPS course rate on (i) steer-derived omega with nominal platform parameters, (ii) raw CAN yawRate,
   (iii) localizer yaw_rate, (iv) curvature * v. From (ii) the CAN yawRate unit and sign per brand; from (i) the
   effective steer ratio and the steering zero offset per platform (median over train drives).
B. Reference time alignment compensation Delta_src: best shift of GPS speed against CAN speed per (source, rate
   class, openpilot major) on train drives, plus a position cross-check on u-blox drives (curvature dead reckoning
   over 30 s segments against GPS positions shifted by candidate deltas).
Outputs: reports/m0_sign_units.csv (per drive), reports/m0_sign_units.md, manifests/time_alignment.json, and the
yawrate_unit / yawrate_sign / angle_offset_train_deg / steer_ratio_eff columns of manifests/vehicle_params.csv.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths  # noqa: E402
from cantraj.geo import course_rate, LocalFrame, bearing_to_theta  # noqa: E402
from cantraj.splits import drive_folds  # noqa: E402

GRID_COLS = ["time_s", "vEgo", "steeringAngleDeg", "yawRate", "actual_curvature", "yaw_rate"]
UNIT_FACTORS = {"rad_s": 1.0, "deg_s": math.pi / 180.0, "deg_s_x10": math.pi / 1800.0}


def slope_corr(x, y):
    """Regress y on x (x = clean CAN quantity, y = noisy GPS course rate): slope, intercept, corr."""
    k = np.isfinite(x) & np.isfinite(y)
    if k.sum() < 50 or np.std(x[k]) == 0:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x[k], y[k], 1)
    return float(a), float(b), float(np.corrcoef(x[k], y[k])[0, 1])


def classify_unit(factor):
    """factor = slope of course rate on raw yawRate (rad/s per raw unit)."""
    if not np.isfinite(factor) or factor == 0:
        return None, None
    sign = 1 if factor > 0 else -1
    f = abs(factor)
    best = min(UNIT_FACTORS, key=lambda u: abs(math.log(f / UNIT_FACTORS[u])))
    if abs(math.log(f / UNIT_FACTORS[best])) > math.log(2.5):
        return None, sign
    return best, sign


def part_a(train, vp, dp, max_drives=260):
    rows = []
    hires = train[(train.gps_sources == "ublox") & (train.gps_rate_hz >= 8)].head(max_drives)
    for i, r in enumerate(hires.itertuples(index=False)):
        gps = io.load_gps(r.driver, r.drive)
        gps = gps[gps.speed_mps > 5]
        if len(gps) < 300:
            continue
        cr = course_rate(gps.bearing_deg.values, gps.time_s.values)
        tm = (gps.time_s.values[:-1] + gps.time_s.values[1:]) / 2
        g = pq.read_table(paths.grid_path(r.driver, r.drive), columns=GRID_COLS).to_pandas()
        at = lambda c: np.interp(tm, g.time_s.values, np.nan_to_num(g[c].values))  # noqa: E731
        v = at("vEgo")
        k = np.isfinite(cr) & (np.abs(cr) > 0.02) & (v > 5)
        if k.sum() < 50:
            continue
        plat = dp.get((r.driver, r.drive))
        p = vp.loc[plat] if plat in vp.index else None
        out = dict(driver=r.driver, drive=r.drive, platform=plat, brand=(plat or "").split("_")[0], n=int(k.sum()))
        delta_deg = at("steeringAngleDeg")
        if p is not None and np.isfinite(p["steer_ratio"]) and np.isfinite(p["wheelbase_m"]):
            SR, L = float(p["steer_ratio"]), float(p["wheelbase_m"])
            om_steer = v * np.tan(np.radians(delta_deg) / SR) / L
            a, b, c = slope_corr(om_steer[k], cr[k])
            out.update(steer_slope=a, steer_corr=c, steer_ratio_nom=SR, steer_ratio_eff=SR / a if np.isfinite(a) and a > 0 else np.nan)
            # zero offset: delta - cr * SR * L / v (small-angle inverse), median over moving frames
            kk = k & (v > 8)
            if kk.sum() > 50:
                out["angle_offset_deg"] = float(np.median(delta_deg[kk] - np.degrees(cr[kk] * SR * L / v[kk])))
        yr = at("yawRate")
        if np.std(yr[k]) > 0:
            a, b, c = slope_corr(yr[k], cr[k])
            out.update(canyaw_factor=a, canyaw_corr=c)
        a, b, c = slope_corr(at("yaw_rate")[k], cr[k])
        out.update(locyaw_slope=a, locyaw_corr=c)
        a, b, c = slope_corr((at("actual_curvature") * v)[k], cr[k])
        out.update(curv_slope=a, curv_corr=c)
        rows.append(out)
        if i % 50 == 0:
            print("A", i, flush=True)
    return pd.DataFrame(rows)


def best_lag(gps, g, lags=np.arange(-2.0, 2.01, 0.1)):
    best = None
    for d in lags:
        v = np.interp(gps.time_s.values + d, g.time_s.values, g.vEgo.values)
        k = (gps.speed_mps.values > 2) | (v > 2)
        if k.sum() < 100:
            continue
        e = float(np.sqrt(np.mean((v[k] - gps.speed_mps.values[k]) ** 2)))
        if best is None or e < best[1]:
            best = (round(float(d), 1), e)
    return best


def part_b(train, max_per_stratum=120, pos_check_drives=40):
    rows = []
    train = train.assign(hires=train.gps_rate_hz >= 8, opmaj=train.op_version.astype(str).str.extract(r"^(\d+\.\d+)")[0].fillna("?"))
    for (src, hi), g in train.groupby(["gps_sources", "hires"]):
        for r in g.sample(min(len(g), max_per_stratum), random_state=0).itertuples(index=False):
            gps = io.load_gps(r.driver, r.drive)
            grid = pq.read_table(paths.grid_path(r.driver, r.drive), columns=["time_s", "vEgo"]).to_pandas()
            b = best_lag(gps, grid)
            if b:
                rows.append(dict(driver=r.driver, drive=r.drive, source=src, hires=hi, opmaj=r.opmaj, lag_s=b[0], rmse=b[1]))
    lag = pd.DataFrame(rows)
    # position cross-check on u-blox hires train drives: curvature DR over 30 s segments vs GPS shifted by delta
    pos_rows = []
    hires = train[(train.gps_sources == "ublox") & (train.gps_rate_hz >= 8)].head(pos_check_drives)
    deltas = np.arange(-0.8, 0.81, 0.1)
    for r in hires.itertuples(index=False):
        gps = io.load_gps(r.driver, r.drive)
        grid = pq.read_table(paths.grid_path(r.driver, r.drive), columns=["time_s", "vEgo", "actual_curvature"]).to_pandas()
        gt, gv, gk = grid.time_s.values, grid.vEgo.values, np.nan_to_num(grid.actual_curvature.values)
        errs = np.zeros(len(deltas))
        n = 0
        for t_start in np.arange(gps.time_s.min() + 5, gps.time_s.max() - 40, 60.0):
            i0 = np.searchsorted(gps.time_s.values, t_start)
            if i0 >= len(gps) or gps.speed_mps.values[i0] < 5 or not (0 < gps.bearing_acc_deg.values[i0] <= 5):
                continue
            fr = LocalFrame(gps.lat.values[i0], gps.lon.values[i0])
            th = float(bearing_to_theta(gps.bearing_deg.values[i0]))
            t_anchor = gps.time_s.values[i0]
            # dead reckon 30 s on the 10 Hz grid from the anchor (anchor time itself shifted by delta later)
            for j, d in enumerate(deltas):
                ta = t_anchor - d                      # GPS stamped t describes CAN time t - d
                sel = (gt >= ta) & (gt <= ta + 30.0)
                if sel.sum() < 200:
                    errs[j] = np.nan
                    continue
                v, kap, tt = gv[sel], gk[sel], gt[sel]
                om = -v * kap
                dth = np.cumsum(om[:-1] * np.diff(tt))
                thh = th + np.r_[0.0, dth]
                x = np.cumsum(np.r_[0.0, v[:-1] * np.cos(thh[:-1]) * np.diff(tt)])
                y = np.cumsum(np.r_[0.0, v[:-1] * np.sin(thh[:-1]) * np.diff(tt)])
                # compare with GPS positions whose stamped times are tt + d
                gx, gy = fr.to_enu(np.interp(tt + d, gps.time_s.values, gps.lat.values), np.interp(tt + d, gps.time_s.values, gps.lon.values))
                errs[j] += float(np.mean(np.hypot(x - gx, y - gy)))
            n += 1
        if n:
            pos_rows.append(dict(driver=r.driver, drive=r.drive, n_segments=n, best_delta_s=float(deltas[np.nanargmin(errs)]),
                                 err_at_0=float(errs[np.argmin(np.abs(deltas))] / n), err_best=float(np.nanmin(errs) / n)))
    return lag, pd.DataFrame(pos_rows)


def main():
    man = io.load_manifest()
    folds = drive_folds()
    ok = man[(man.status == "ok") & (man.set == "main")].copy()
    ok["fold"] = [folds.get((a, b), "") for a, b in zip(ok.driver, ok.drive)]
    train = ok[ok.fold == "train"]
    vp = pd.read_csv(os.path.join(paths.MANIFESTS, "vehicle_params.csv")).set_index("platform")
    dp = pd.read_csv(os.path.join(paths.MANIFESTS, "drive_platform.csv"))
    dp = dict(zip(zip(dp.driver, dp.drive), dp.platform))
    A = part_a(train, vp, dp)
    A.to_csv(os.path.join(paths.REPORTS, "m0_sign_units.csv"), index=False)
    lag, pos = part_b(train)
    lag.to_csv(os.path.join(paths.REPORTS, "m0_time_alignment_drives.csv"), index=False)
    pos.to_csv(os.path.join(paths.REPORTS, "m0_time_alignment_poscheck.csv"), index=False)
    # --- tables
    lines = ["# Sign, unit and time-alignment calibration (training fold only)\n", f"A: {len(A)} u-blox >= 8 Hz train drives\n"]
    agg = A.groupby("brand").agg(n=("drive", "size"), steer_corr=("steer_corr", "median"), steer_slope=("steer_slope", "median"),
                                 locyaw_slope=("locyaw_slope", "median"), curv_slope=("curv_slope", "median"),
                                 canyaw_factor=("canyaw_factor", "median"), canyaw_corr=("canyaw_corr", "median")).round(3)
    lines += ["## Per brand (medians; course rate left-positive; slopes = d(course rate)/d(signal))", agg.to_markdown(), ""]
    brand_unit = {}
    for b, g in A.dropna(subset=["canyaw_factor"]).groupby("brand"):
        g = g[g.canyaw_corr.abs() > 0.5]
        if len(g) < 2:
            continue
        unit, sign = classify_unit(float(g.canyaw_factor.median()))
        brand_unit[b] = dict(unit=unit, sign=sign, n=int(len(g)), factor=float(g.canyaw_factor.median()))
    lines += ["## CAN yawRate unit and sign per brand (train fold)", pd.DataFrame(brand_unit).T.to_markdown() if brand_unit else "(none)", ""]
    plat = A.groupby("platform").agg(n=("drive", "size"), angle_offset_train_deg=("angle_offset_deg", "median"),
                                     steer_ratio_eff=("steer_ratio_eff", "median"), steer_ratio_nom=("steer_ratio_nom", "first"),
                                     steer_corr=("steer_corr", "median")).round(3)
    lines += ["## Per platform steering calibration (train fold)", plat.sort_values("n", ascending=False).head(30).to_markdown(), ""]
    lag["stratum"] = lag.source + np.where(lag.hires, "_hires", "_lowrate")
    lag["delta_s"] = -lag.lag_s
    lt = lag.groupby(["stratum", "opmaj"]).delta_s.agg(["size", "median", lambda s: s.quantile(.1), lambda s: s.quantile(.9)]).round(2)
    lt.columns = ["n", "median", "p10", "p90"]
    lines += ["## Time alignment: best speed-matching shift (s, GPS stamped time minus CAN time; positive = GPS lags)", lt.to_markdown(), ""]
    delta = {s: float(-g.lag_s.median()) for s, g in lag.groupby("stratum")}
    lines += ["## Position cross-check (u-blox hires train drives, curvature DR over 30 s segments)",
              f"best delta distribution (s): {pos.best_delta_s.describe().round(2).to_dict() if len(pos) else 'n/a'}; mean error at delta 0: {pos.err_at_0.mean():.1f} m, at best delta: {pos.err_best.mean():.1f} m" if len(pos) else "n/a", ""]
    with open(os.path.join(paths.MANIFESTS, "time_alignment.json"), "w") as f:
        json.dump(dict(delta_s=delta, rule="time_s = time_s_raw - delta_s[stratum]; stratum = source + (_hires if gps_rate_hz >= 8 else _lowrate); estimated on the training fold by speed cross-correlation", n_drives=int(len(lag))), f, indent=1)
    # --- write calibration columns into vehicle_params.csv
    vp2 = pd.read_csv(os.path.join(paths.MANIFESTS, "vehicle_params.csv"))
    vp2["yawrate_unit"] = vp2.brand.map(lambda b: (brand_unit.get(b) or {}).get("unit") or "")
    vp2["yawrate_sign"] = vp2.brand.map(lambda b: (brand_unit.get(b) or {}).get("sign") or "")
    vp2["angle_offset_train_deg"] = vp2.platform.map(plat["angle_offset_train_deg"].to_dict())
    vp2["steer_ratio_eff_train"] = vp2.platform.map(plat["steer_ratio_eff"].to_dict())
    vp2["n_train_calib_drives"] = vp2.platform.map(plat["n"].to_dict()).fillna(0).astype(int)
    vp2.to_csv(os.path.join(paths.MANIFESTS, "vehicle_params.csv"), index=False)
    # global fallback for platforms without training-fold calibration (drive-weighted medians over train drives)
    g_off = float(A["angle_offset_deg"].median()) if "angle_offset_deg" in A and A["angle_offset_deg"].notna().any() else 0.0
    ratio = (A["steer_ratio_eff"] / A["steer_ratio_nom"]).replace([np.inf, -np.inf], np.nan) if "steer_ratio_eff" in A else pd.Series(dtype=float)
    g_fac = float(ratio.median()) if ratio.notna().any() else 1.0
    json.dump(dict(angle_offset_deg=g_off, steer_ratio_factor=g_fac, n_drives=int(len(A)), rule="unseen platform: steer_ratio_eff = steer_ratio_nom * steer_ratio_factor; angle offset = angle_offset_deg; estimated on the training fold only"),
              open(os.path.join(paths.MANIFESTS, "calibration_global.json"), "w"), indent=1)
    with open(os.path.join(paths.REPORTS, "m0_sign_units.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("delta_s:", delta)


if __name__ == "__main__":
    main()
