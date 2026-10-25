"""Episode construction: reference views, anchors (t0, p0, theta0), fixed windows / full drives, whitelisted inputs.

Rules (Plan.md 2.2 items 3, 4, 8): the anchor t0 is the first `view_1hz` fix at or after the nominal window start
with speed >= 3 m/s, 0 < bearing_acc <= 5 deg (u-blox also h_acc < 10 m) and CAN speed >= 2.5 m/s at that time;
p0 = GPS(t0); theta0 = pi/2 - bearing(t0) is handed over only in H+ protocols; the episode is [t0, t0 + D]. Inputs are
the raw events (native tables) with logged time inside the episode; the released grid is the `grid_compat` layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import io, registry
from .events import EventStream
from .geo import LocalFrame, bearing_to_theta
from .inputs import EpisodeInit, EpisodeInputs, StaticParams, canonicalize

WIN = registry.windows()


def view_1hz(ref: pd.DataFrame, tol=None):
    """One real fix per integer second: the fix nearest to s within +-tol (default 0.5 s). Never interpolates."""
    tol = WIN["reference"]["view_1hz_tolerance_s"] if tol is None else tol
    if ref.empty:
        return ref.iloc[0:0]
    t = ref["time_s"].to_numpy(dtype=float)
    sec = np.round(t).astype(np.int64)
    d = np.abs(t - sec)
    ok = d <= tol
    df = ref.assign(_sec=sec, _d=d)[ok]
    df = df.sort_values(["_sec", "_d"], kind="stable").drop_duplicates("_sec", keep="first")
    return df.drop(columns=["_sec", "_d"]).sort_values("time_s", kind="stable").reset_index(drop=True)


def find_anchor(ref1: pd.DataFrame, can_speed_at, t_nominal, t_max=None):
    """First qualifying fix at/after t_nominal. can_speed_at(t) -> CAN speed (m/s) from the drive's own inputs.
    Returns dict or None."""
    a = WIN["anchor"]
    df = ref1[ref1["time_s"] >= t_nominal]
    if t_max is not None:
        df = df[df["time_s"] <= t_max]
    for r in df.itertuples(index=False):
        if r.speed_mps < a["min_speed_mps"]:
            continue
        if not (0.0 < r.bearing_acc_deg <= a["max_bearing_acc_deg"]):
            continue
        if r.source == "ublox" and not (0.0 < r.h_acc_m < a["ublox_max_h_acc_m"]):
            continue
        v = can_speed_at(r.time_s)
        if not np.isfinite(v) or v < 2.5:
            continue
        return dict(t0=float(r.time_s), p0_lat=float(r.lat), p0_lon=float(r.lon), bearing_deg=float(r.bearing_deg),
                    theta0=float(bearing_to_theta(r.bearing_deg)), anchor_speed_mps=float(r.speed_mps),
                    anchor_bearing_acc_deg=float(r.bearing_acc_deg), anchor_h_acc_m=float(r.h_acc_m), anchor_kind="moving")
    return None


@dataclass
class Episode:
    episode_id: str
    driver: str
    drive: str
    win: str
    k: int
    t0: float
    t_end: float
    init_full: dict                     # anchor dict incl. theta0 (reference side; never handed to baselines directly)
    ref: pd.DataFrame                   # scored reference points: time_s, lat, lon, x, y, speed_mps, source, moving
    frame: LocalFrame
    meta: dict = field(default_factory=dict)

    def init(self, heading_given: bool, offline: bool):
        return EpisodeInit(t0=self.t0, p0_lat=self.init_full["p0_lat"], p0_lon=self.init_full["p0_lon"],
                           theta0=self.init_full["theta0"] if heading_given else None,
                           t_end=self.t_end if offline else None)


class DriveData:
    """Everything loaded once per drive: reference fixes, released grid (compat), native events (if re-decoded)."""

    def __init__(self, driver, drive, delta_s=0.0, input_layer="events", light=False, with_imu=False):
        """light=True loads only what anchors and windows need (GPS + released grid), no native event tables."""
        self.driver, self.drive = driver, drive
        self.delta_s = float(delta_s or 0.0)
        self.ref = io.load_gps(driver, drive, self.delta_s)
        self.ref1 = view_1hz(self.ref)
        self.grid = io.load_grid(driver, drive, columns=["time_s", "vEgo"] if light else None)
        if light:
            import os
            self.native = {}
            self.has_native = os.path.exists(os.path.join(io.paths.native_dir(driver, drive), "carstate.parquet"))
            self.events_all = None
            self.layer = "events" if self.has_native else "grid_compat"
            gt = self.grid["time_s"].to_numpy(dtype=float)
            gv = self.grid["vEgo"].to_numpy(dtype=float)
            self.D = float(gt[-1])
            self._speed_at = lambda t: float(np.interp(t, gt, gv))
            return
        self.native = io.load_native(driver, drive, ("carstate", "controls") + (("imu",) if with_imu else ())) if input_layer == "events" else {}
        self.has_native = "carstate" in self.native
        self.D = float(self.grid["time_s"].iloc[-1])
        if self.has_native:
            cs = self.native["carstate"]
            cols = canonicalize(cs, "native", "carstate")
            if "controls" in self.native:
                ctl = self.native["controls"]
                ctl = ctl[ctl["topic"] == "controlsState"]
                ctl_cols = canonicalize(ctl, "native", "controls")
                if ctl_cols:        # controlsState has its own clock; attach by nearest previous carState message
                    idx = np.searchsorted(cs["t_s"].to_numpy(dtype=float), ctl["t_s"].to_numpy(dtype=float), side="right") - 1
                    for n, v in ctl_cols.items():
                        a = np.full(len(cs), np.nan)
                        good = idx >= 0
                        a[idx[good]] = v[good]
                        cols[n] = a
            if with_imu and "imu" in self.native:        # device IMU (sensorEvents) attached by nearest previous carState message
                imu = self.native["imu"]
                cs_t = cs["t_s"].to_numpy(dtype=float)
                for kind, names in (("gyro", ("imu_gyro_x", "imu_gyro_y", "imu_gyro_z")), ("accel", ("imu_acc_x", "imu_acc_y", "imu_acc_z"))):
                    rows = imu[imu["kind"] == kind]
                    if kind == "gyro" and "which" in rows and (rows["which"] == "gyro").any():
                        rows = rows[rows["which"] == "gyro"]          # calibrated stream when both are logged (sensorEvents)
                    if rows.empty and kind == "gyro":
                        rows = imu[imu["kind"] == "gyroUncalibrated"]
                    if rows.empty:
                        continue
                    idx = np.searchsorted(cs_t, rows["t_s"].to_numpy(dtype=float), side="right") - 1
                    good = idx >= 0
                    for n, c in zip(names, ("v0", "v1", "v2")):
                        a = np.full(len(cs), np.nan)
                        a[idx[good]] = rows[c].to_numpy(dtype=float)[good]
                        cols[n] = a
            self.events_all = EventStream(cs["t_s"].to_numpy(dtype=float), cols)
            self.layer = "events"
        else:   # grid_compat: every released 10 Hz row is an "event" at its grid time (interpolated values; control only)
            self.events_all = EventStream(self.grid["time_s"].to_numpy(dtype=float), canonicalize(self.grid, "grid"))
            self.layer = "grid_compat"
        gt = self.grid["time_s"].to_numpy(dtype=float)
        gv = self.grid["vEgo"].to_numpy(dtype=float)
        self._speed_at = lambda t: float(np.interp(t, gt, gv))      # CAN speed for anchor checks and strata

    def can_speed_at(self, t):
        return self._speed_at(t)

    def episodes(self, win, drive_meta=None):
        """Tile the drive into episodes of window class `win` (or one `full` episode)."""
        D_win = WIN["classes"][win]["duration_s"]
        elig = WIN["eligibility"]
        out = []
        if self.ref1.empty:
            return out
        t_first, t_last = float(self.ref1["time_s"].iloc[0]), float(self.ref1["time_s"].iloc[-1])
        t_nom, k = t_first, 0
        while True:
            dur = D_win if D_win else (t_last - t_nom)
            if dur <= 0 or (D_win and t_nom + 0.5 * D_win > t_last):
                break
            anc = find_anchor(self.ref1, self.can_speed_at, t_nom, t_max=(t_nom + D_win) if D_win else None)
            if anc is None:
                if not D_win:
                    break
                t_nom += D_win
                k += 1
                continue
            t0 = anc["t0"]
            t_end = (t0 + D_win) if D_win else t_last
            ref = self.ref1[(self.ref1["time_s"] > t0) & (self.ref1["time_s"] <= t_end)].copy()
            frame = LocalFrame(anc["p0_lat"], anc["p0_lon"])
            x, y = frame.to_enu(ref["lat"].to_numpy(), ref["lon"].to_numpy())
            ref["x"], ref["y"] = x, y
            ref["moving"] = np.array([self.can_speed_at(t) > 1.0 for t in ref["time_s"]])
            gaps = np.diff(np.r_[t0, ref["time_s"].to_numpy(dtype=float)])
            path_len = float(np.sum(np.hypot(np.diff(np.r_[0.0, x]), np.diff(np.r_[0.0, y]))))
            meta = dict(t_nominal_s=t_nom, t0_offset_s=t0 - t_nom, n_ref1hz=int(len(ref)),
                        ref_frac=float(len(ref) / max(t_end - t0, 1e-9)), max_gap_s=float(gaps.max()) if len(gaps) else np.nan,
                        has_gap30=bool(len(gaps) and gaps.max() > elig["max_ref_gap_s"]), path_len_m=path_len,
                        net_disp_m=float(np.hypot(x[-1], y[-1])) if len(x) else 0.0,
                        moving_frac=float(ref["moving"].mean()) if len(ref) else np.nan, input_layer=self.layer, delta_s=self.delta_s,
                        gps_source=";".join(sorted(ref["source"].unique())) if len(ref) else "")
            meta["eligible_main"] = bool(len(ref) >= elig["min_ref1hz_frac"] * (t_end - t0) and not meta["has_gap30"]
                                         and path_len >= elig["min_path_len_m"])
            if drive_meta:
                meta.update(drive_meta)
            eid = f"{self.driver}/{self.drive}/{win}/k{k:03d}"
            out.append(Episode(eid, self.driver, self.drive, win, k, t0, t_end, anc, ref, frame, meta))
            if not D_win:
                break
            t_nom = t_end
            k += 1
        return out

    def inputs(self, ep: Episode, gid: str, protocol: str, static: StaticParams, map_graph=None, warmup_s=0.0):
        spec = registry.protocol_spec(protocol)
        offline = spec["mode"] == "offline"
        ev = self.events_all.window(ep.t0, ep.t_end)
        init = ep.init(heading_given=(spec["heading"] == "given"), offline=offline)
        warm = None
        if warmup_s and warmup_s > 0:              # warm-start variant: the drive's own events in [t0 - W, t0)
            warm = self.events_all.window(max(ep.t0 - float(warmup_s), 0.0), ep.t0 - 1e-6)
        return EpisodeInputs(gid, ev, static, init, input_layer=self.layer, map_graph=map_graph, warmup=warm)


def ref_times(ep: Episode):
    return ep.ref["time_s"].to_numpy(dtype=float)
