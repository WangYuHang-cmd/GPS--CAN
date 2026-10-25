"""Training data for the learned motion models (Plan v0.5 appendix A).

Grid: ticks at absolute multiples of GRID_DT on the drive axis (not anchored at t0) so training and online inference
see identical features. Per drive we cache the zero-order-hold view of (speed, steer_angle) with age and missing flags,
and the 1 Hz reference fixes (positions in a drive-level local frame, course heading where reliable, anchor_ok flags).
Features are produced by ONE function (`features`) shared bitwise by the trainer and the online wrapper.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from .. import io, paths
from ..episodes import WIN, view_1hz
from ..events import EventStream
from ..geo import LocalFrame, bearing_to_theta
from ..inputs import canonicalize

GRID_DT = 0.1
MAX_AGE_S = 0.3
LEARN_DIR = f"{paths.BENCH}/learn/v1"
FEATURE_NAMES = ["v", "steer", "wheel_angle", "steer_rate", "accel", "age", "missing", "moving", "rate_regime", "wheelbase_dev", "steer_ratio_dev", "static_missing", "omega_nom"]
N_FEAT = len(FEATURE_NAMES)


def grid_times(t_lo, t_hi):
    k0 = int(math.ceil(t_lo / GRID_DT))
    k1 = int(math.floor(t_hi / GRID_DT))
    return k0, GRID_DT * np.arange(k0, k1 + 1)


def features(v, steer, age, missing, prev_v, prev_steer, static, nominal_dt):
    """Feature vector(s) for one tick (or arrays of ticks). All inputs are ZOH values AT the tick; prev_* are the
    previous tick's values (zero differences at the first tick of a chunk / episode)."""
    v = np.asarray(v, dtype=np.float32)
    steer = np.asarray(steer, dtype=np.float32)
    L = static.get("wheelbase_m")
    SR = static.get("steer_ratio")
    static_missing = float(L is None or SR is None or not np.isfinite(L) or not np.isfinite(SR))
    L_ = 2.8 if static_missing else float(L)
    SR_ = 15.0 if static_missing else float(SR)
    steer_c = np.clip(np.nan_to_num(steer), -300.0, 300.0)
    v_c = np.nan_to_num(v)
    wheel = np.radians(steer_c) / SR_
    omega_nom = v_c * np.tan(np.clip(wheel, -1.2, 1.2)) / L_
    f = np.stack([
        v_c / 10.0,
        np.clip(steer_c / 10.0, -30.0, 30.0),
        wheel,
        (steer_c - np.nan_to_num(np.asarray(prev_steer, dtype=np.float32))) / GRID_DT / 100.0,
        (v_c - np.nan_to_num(np.asarray(prev_v, dtype=np.float32))) / GRID_DT / 3.0,
        np.clip(np.nan_to_num(np.asarray(age, dtype=np.float32), nan=MAX_AGE_S) / MAX_AGE_S, 0.0, 3.0),
        np.asarray(missing, dtype=np.float32),
        (v_c > 0.3).astype(np.float32),
        np.full_like(v_c, math.log10(max(nominal_dt, 0.01) / 0.01)),
        np.full_like(v_c, L_ - 2.8),
        np.full_like(v_c, SR_ / 15.0 - 1.0),
        np.full_like(v_c, static_missing),
        omega_nom / 0.3,
    ], axis=-1).astype(np.float32)
    return f


def drive_cache(driver, drive, fold, static, out_dir=LEARN_DIR):
    """Build and save the per-drive cache (grid10 + ref1hz). Returns the index row or None."""
    assert fold in ("train", "val"), "the learning cache never contains test drives"
    man = io.load_manifest().set_index(["driver", "drive"]).loc[(driver, drive)]
    delta = io.time_compensation(man["gps_sources"], man["gps_rate_hz"])
    nat = io.load_native(driver, drive, ("carstate",))
    if "carstate" not in nat:
        return None
    cs = nat["carstate"]
    cols = canonicalize(cs, "native", "carstate")
    if "speed" not in cols or "steer_angle" not in cols:
        return None
    ev = EventStream(cs["t_s"].to_numpy(dtype=float), {"speed": cols["speed"], "steer_angle": cols["steer_angle"]})
    k0, tg = grid_times(float(ev.t_s[0]), float(ev.t_s[-1]))
    if len(tg) < 600:
        return None
    g = ev.causal_grid(tg, max_age_s=MAX_AGE_S)
    ref = io.load_gps(driver, drive, delta)
    ref1 = view_1hz(ref)
    if len(ref1) < 60:
        return None
    frame = LocalFrame(float(ref1.lat.iloc[0]), float(ref1.lon.iloc[0]))
    x, y = frame.to_enu(ref1.lat.values, ref1.lon.values)
    a = WIN["anchor"]
    good_course = (ref1.speed_mps.values >= a["min_speed_mps"]) & (ref1.bearing_acc_deg.values > 0) & (ref1.bearing_acc_deg.values <= a["max_bearing_acc_deg"])
    theta = np.where(good_course, bearing_to_theta(ref1.bearing_deg.values), np.nan)
    gt = ev.t_s
    gv = ev.cols["speed"]
    can_v = np.interp(ref1.time_s.values, gt, gv)
    anchor_ok = good_course & (can_v >= 2.5)
    if "ublox" in str(man["gps_sources"]):
        anchor_ok &= (ref1.h_acc_m.values > 0) & (ref1.h_acc_m.values < a["ublox_max_h_acc_m"])
    hires = bool(man["gps_rate_hz"] >= 8 and "ublox" in str(man["gps_sources"]))
    d = os.path.join(out_dir, "grid10")
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, f"{driver}__{drive}.npz"), k0=np.int64(k0), speed=g["speed"].astype(np.float32), steer=g["steer_angle"].astype(np.float32),
                        age=g["age_s"].astype(np.float16), missing=g["missing"].astype(bool),
                        t_ref=ref1.time_s.values.astype(np.float64), x=x.astype(np.float32), y=y.astype(np.float32), lon=ref1.lon.values.astype(np.float64),
                        lat0=np.float64(frame.lat0), lon0=np.float64(frame.lon0), theta_course=theta.astype(np.float32),
                        h_acc=ref1.h_acc_m.values.astype(np.float16), anchor_ok=anchor_ok.astype(bool), hires=np.bool_(hires))
    return dict(driver=driver, drive=drive, fold=fold, platform=static.get("platform"), brand=static.get("brand"), gps_source=str(man["gps_sources"]),
                gps_rate_hz=float(man["gps_rate_hz"]), hires=hires, delta_s=delta, nominal_dt=float(ev.nominal_dt), wheelbase_m=static.get("wheelbase_m"),
                steer_ratio=static.get("steer_ratio"), angle_offset_train_deg=static.get("angle_offset_train_deg"), steer_ratio_eff_train=static.get("steer_ratio_eff_train"),
                n_ticks=int(len(tg)), n_ref=int(len(ref1)), n_anchor=int(anchor_ok.sum()), lat0=frame.lat0, lon0=frame.lon0)


def load_cache(driver, drive, out_dir=LEARN_DIR, static=None):
    """Load a drive cache; with `static` also precompute the full feature matrix once (feats_full)."""
    z = np.load(os.path.join(out_dir, "grid10", f"{driver}__{drive}.npz"))
    c = {k: z[k] for k in z.files}
    if static is not None:
        v, st = c["speed"], c["steer"]
        prev_v = np.concatenate([[v[0]], v[:-1]])
        prev_st = np.concatenate([[st[0]], st[:-1]])
        c["feats_full"] = features(v, st, c["age"].astype(np.float32), c["missing"], prev_v, prev_st, static, static.get("nominal_dt", 0.01))
    return c


def apply_speed_noise(feats, v, steer, static, sigma, rng):
    """Training augmentation: replace the speed-derived feature columns (v/10, dv/dt, moving, omega_nom) by their values for
    v + N(0, sigma); other columns untouched. Mirrors `features` column by column."""
    vn = np.where(np.isfinite(v), np.maximum(np.nan_to_num(v) + rng.normal(0.0, sigma, len(v)), 0.0), v)
    L = static.get("wheelbase_m"); SR = static.get("steer_ratio")
    static_missing = L is None or SR is None or not np.isfinite(L) or not np.isfinite(SR)
    L_ = 2.8 if static_missing else float(L)
    SR_ = 15.0 if static_missing else float(SR)
    v_c = np.nan_to_num(vn).astype(np.float32)
    steer_c = np.clip(np.nan_to_num(steer), -300.0, 300.0)
    wheel = np.radians(steer_c) / SR_
    out = feats.copy()
    out[:, 0] = v_c / 10.0
    prev_v = np.concatenate([[v_c[0]], v_c[:-1]])
    out[:, 4] = (v_c - prev_v) / GRID_DT / 3.0
    out[0, 4] = 0.0
    out[:, 7] = (v_c > 0.3).astype(np.float32)
    out[:, 12] = (v_c * np.tan(np.clip(wheel, -1.2, 1.2)) / L_) / 0.3
    return out


def chunk_targets(c, i_start, length_s, static, speed_noise=0.0, rng=None):
    """Targets and features for a chunk starting at reference fix i_start (an anchor_ok fix) lasting length_s.
    Returns dict with feats (T, N_FEAT), tick times, reference (x, y) relative to the chunk origin rotated by the
    meridian convergence to the chunk's own local frame, valid mask, theta0, dtheta/ds per second where available."""
    t0 = float(c["t_ref"][i_start])
    t_end = t0 + length_s
    k0 = int(c["k0"])
    j0 = int(math.ceil(t0 / GRID_DT)) - k0
    j1 = int(math.floor(t_end / GRID_DT)) - k0
    if j0 < 0 or j1 >= len(c["speed"]) or j1 - j0 < 10:
        return None
    sl = slice(j0, j1 + 1)
    if "feats_full" in c:                      # precomputed per-drive feature matrix: slice and zero the first-tick differences
        feats = c["feats_full"][sl].copy()
        feats[0, 3] = 0.0
        feats[0, 4] = 0.0
    else:
        v, st, age, miss = c["speed"][sl], c["steer"][sl], c["age"][sl].astype(np.float32), c["missing"][sl]
        prev_v = np.concatenate([[v[0]], v[:-1]])
        prev_st = np.concatenate([[st[0]], st[:-1]])
        feats = features(v, st, age, miss, prev_v, prev_st, static, static.get("nominal_dt", 0.01))
    if speed_noise and speed_noise > 0 and rng is not None:
        feats = apply_speed_noise(feats, c["speed"][sl], c["steer"][sl], static, float(speed_noise), rng)
    ticks = GRID_DT * np.arange(k0 + j0, k0 + j1 + 1)
    # reference inside the chunk
    m = (c["t_ref"] > t0) & (c["t_ref"] <= t_end)
    gamma = float((np.radians(c["lon"][i_start] - c["lon0"])) * math.sin(math.radians(float(c["lat0"]))))   # meridian convergence to the chunk origin
    cg, sg = math.cos(gamma), math.sin(gamma)
    dx, dy = c["x"][m] - c["x"][i_start], c["y"][m] - c["y"][i_start]
    xr, yr = cg * dx + sg * dy, -sg * dx + cg * dy
    theta0 = float(c["theta_course"][i_start]) - gamma
    th = c["theta_course"][m] - gamma
    return dict(feats=feats, ticks=ticks, t_ref=c["t_ref"][m], x=xr.astype(np.float32), y=yr.astype(np.float32), theta0=theta0,
                theta_course=th.astype(np.float32), hires=bool(c["hires"]), valid=np.isfinite(xr))
