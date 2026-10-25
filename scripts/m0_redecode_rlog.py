#!/usr/bin/env python3
"""m0_redecode_rlog.py: native-rate CAN / IMU / calibration / parameter EVENTS from the raw openpilot logs.

Why: the released 10 Hz tables were linearly interpolated from decimated qlogs, so a grid value can depend on a raw
sample recorded after the grid time. The online protocols of the CAN-to-GPS benchmark replay raw messages at their
logged time instead. This script re-reads every drive's logs (rlog preferred, qlog fallback), anchors them to the
released `time_s` axis with the SAME segment-verified alignment as extract_gps.py (every function reused), and writes
one row per message:

  <out>/native/driver_XXX/drive_YYY/carstate.parquet    carState (100 Hz rlog / 10 Hz qlog): speed, steering, pedals, ...
                                   controls.parquet    controlsState.curvature/enabled + carControl.enabled/latActive/longActive
                                   imu.parquet         gyroscope/accelerometer (>= 0.9.0) or sensorEvents (older), raw sensor frame
                                   calib.parquet       liveCalibration rpyCalib / status
                                   liveparams.parquet  liveParameters (audit of the fusion-derived signals)
                                   carparams.json      static vehicle parameters (no VIN)
  <out>/native/native_manifest.csv                     one row per drive (no raw paths, no device/route ids)
  <out>/private/redecode_progress.jsonl                resumable progress (private: raw paths)

Rows are kept for TRIM_S <= t_s <= D - TRIM_S exactly like the GPS export. `--carparams-only` reads only the first
segment of every drive until its carParams message (fast pass for manifests/vehicle_params.csv).

    python \
        GPS/scripts/m0_redecode_rlog.py --only driver_384/drive_015 --workers 1
    ... --carparams-only --workers 4
    ... --set all --workers 5                 # full run, resumable
"""
import argparse
import datetime as dt
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.environ.get("DRIVEDNA_CODE", "code/preprocessing"))   # DriveDNA preprocessing helpers
import extract_gps as eg  # noqa: E402  listers, iter_msgs, align, repair_and_judge, hole_mask, build_jobs

BENCH = "/data/backup/DriveDNA-GPS_bench"
OUT_DEFAULT = f"{BENCH}/native"
PRIV_DIR = f"{BENCH}/private"
PROGRESS = f"{PRIV_DIR}/redecode_progress.jsonl"
CP_PROGRESS = f"{PRIV_DIR}/carparams_progress.jsonl"
PUBLIC_MANIFEST = os.path.join(os.environ.get("CANTRAJ_MANIFESTS", "manifests"), "native_manifest.csv")

CS_COLS = ["mono", "vEgo", "vEgoRaw", "aEgo", "vEgoCluster", "steeringAngleDeg", "steeringRateDeg", "yawRate",
           "ws_fl", "ws_fr", "ws_rl", "ws_rr", "gas", "brake", "gasPressed", "brakePressed", "steeringPressed",
           "standstill", "leftBlinker", "rightBlinker", "cruiseEnabled", "cruiseAvailable", "gearShifter",
           "steeringTorque", "canValid"]
CTRL_COLS = ["mono", "topic", "curvature", "enabled", "active", "latActive", "longActive"]
IMU_COLS = ["mono", "sensor_ts", "topic", "kind", "which", "source", "sensor", "v0", "v1", "v2"]
CAL_COLS = ["mono", "roll", "pitch", "yaw", "calStatus", "calPerc", "validBlocks"]
LP_COLS = ["mono", "valid", "sensorValid", "posenetValid", "gyroBias", "angleOffsetDeg", "angleOffsetAverageDeg",
           "stiffnessFactor", "steerRatio", "roll", "steerRatioStd", "angleOffsetAverageStd"]
CP_FIELDS = ["carFingerprint", "carName", "mass", "wheelbase", "centerToFront", "steerRatio", "steerRatioRear",
             "rotationalInertia", "tireStiffnessFactor", "tireStiffnessFront", "tireStiffnessRear", "minSteerSpeed",
             "minEnableSpeed", "notCar", "dashcamOnly"]
IMU_KINDS = {"acceleration": "accel", "gyro": "gyro", "gyroUncalibrated": "gyro"}
MANIFEST_COLS = ["set", "driver", "drive", "status", "reason", "duration_s", "op_version", "n_seg_read", "n_seg_rlog",
                 "n_seg_qlog", "log_policy", "k_anchor", "align_method", "align_frac", "align_cov", "n_seg_hole",
                 "n_seg_qlog_retry", "n_seg_bad", "n_seg_misnamed", "native",
                 "cs_rows", "cs_rate_hz", "cs_dt_med_s", "cs_max_gap_s", "cs_n_gaps", "cs_gap_s",
                 "ctrl_rows", "ctrl_rate_hz", "imu_rows", "imu_rate_hz", "imu_topic", "imu_source",
                 "cal_rows", "lp_rows", "carparams", "carFingerprint", "fp_majority_share", "n_fp_distinct", "fp_mock_segs",
                 "wheelbase", "steerRatio", "centerToFront", "mass", "mb_read", "elapsed_s"]
_STASH = {}   # path -> native payload of the latest read (cleared per drive)


# ----------------------------------------------------------------------------- reader
def _vec3(v):
    v = list(v)
    return (float(v[0]), float(v[1]), float(v[2])) if len(v) >= 3 else (np.nan, np.nan, np.nan)


def _carparams(cp):
    d = {}
    for f in CP_FIELDS:
        try:
            x = getattr(cp, f)
            d[f] = str(x) if isinstance(x, (str, bytes)) or f in ("carFingerprint", "carName") else (bool(x) if f in ("notCar", "dashcamOnly") else float(x))
        except Exception:
            d[f] = None
    return d


def read_segment_native(path, kind, seg, want_gnss=False):
    """Drop-in replacement for extract_gps.read_segment: same alignment arrays, plus native-rate payloads."""
    cs, ctrl, imu, cal, lp = [], [], [], [], []
    cparams, op_version = None, None
    for w, m in eg.iter_msgs(path):
        if w == "carState":
            s = m.carState
            ws = s.wheelSpeeds
            cru = s.cruiseState
            cs.append((m.logMonoTime, s.vEgo, s.vEgoRaw, s.aEgo, s.vEgoCluster, s.steeringAngleDeg, s.steeringRateDeg,
                       s.yawRate, ws.fl, ws.fr, ws.rl, ws.rr, s.gas, s.brake, s.gasPressed, s.brakePressed,
                       s.steeringPressed, s.standstill, s.leftBlinker, s.rightBlinker, cru.enabled, cru.available,
                       str(s.gearShifter), s.steeringTorque, s.canValid))
        elif w == "controlsState":
            c = m.controlsState
            ctrl.append((m.logMonoTime, "controlsState", c.curvature, c.enabled, c.active, None, None))
        elif w == "carControl":
            c = m.carControl
            ctrl.append((m.logMonoTime, "carControl", np.nan, c.enabled, None, c.latActive, c.longActive))
        elif w in ("gyroscope", "accelerometer"):
            d = getattr(m, w)
            try:
                wh = d.which()
                if wh in IMU_KINDS:
                    v = _vec3(getattr(d, wh).v)
                    imu.append((m.logMonoTime, int(d.timestamp), w, IMU_KINDS[wh], wh, str(d.source), int(d.sensor)) + v)
            except Exception:
                pass
        elif w == "sensorEventsDEPRECATED":
            for d in getattr(m, w):
                try:
                    wh = d.which()
                    if wh in IMU_KINDS:
                        v = _vec3(getattr(d, wh).v)
                        imu.append((m.logMonoTime, int(d.timestamp), "sensorEvents", IMU_KINDS[wh], wh, str(d.source), int(d.sensor)) + v)
                except Exception:
                    pass
        elif w == "liveCalibration":
            c = m.liveCalibration
            rpy = _vec3(c.rpyCalib)
            cal.append((m.logMonoTime,) + rpy + (str(c.calStatus), int(c.calPerc), int(c.validBlocks)))
        elif w == "liveParameters":
            p = m.liveParameters
            lp.append((m.logMonoTime, bool(p.valid), bool(p.sensorValid), bool(p.posenetValid), float(p.gyroBias),
                       float(p.angleOffsetDeg), float(p.angleOffsetAverageDeg), float(p.stiffnessFactor),
                       float(p.steerRatio), float(p.roll), float(p.steerRatioStd), float(p.angleOffsetAverageStd)))
        elif w == "carParams" and cparams is None:
            cparams = _carparams(m.carParams)
        elif w == "initData" and op_version is None:
            try:
                op_version = str(m.initData.version)[:40]
            except Exception:
                pass
    if cs:
        t = np.array([r[0] for r in cs], dtype=np.int64)
        v = np.array([r[1] for r in cs], dtype=np.float64)
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
    else:
        t, v = np.zeros(0, np.int64), np.zeros(0, np.float64)
    _STASH[path] = dict(cs=cs, ctrl=ctrl, imu=imu, cal=cal, lp=lp, carparams=cparams, kind=kind, seg=seg)
    return dict(seg=seg, kind=kind, path=path, cs_t=t, cs_v=v, gps=[], llk=[], gnss=[], clk=[], op_version=op_version)


eg.read_segment = read_segment_native          # repair_and_judge re-reads replacement segments through this


def read_first_carparams(path, max_msgs=5000):
    op_version, cp = None, None
    n = 0
    for w, m in eg.iter_msgs(path):
        n += 1
        if w == "carParams" and cp is None:
            cp = _carparams(m.carParams)
        elif w == "initData" and op_version is None:
            try:
                op_version = str(m.initData.version)[:40]
            except Exception:
                pass
        if cp is not None and op_version is not None:
            break
        if n >= max_msgs:
            break
    return cp, op_version


# ----------------------------------------------------------------------------- per drive
def _frame(rows, cols, t0_ns, D, seg, kind):
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=cols)
    df["t_s"] = (df["mono"].values.astype(np.int64) - t0_ns) / 1e9
    df = df[(df["t_s"] >= eg.TRIM_S) & (df["t_s"] <= D - eg.TRIM_S)].copy()
    if not len(df):
        return None
    df["seg"] = np.int16(seg)
    df["log"] = kind
    return df


def _rate_stats(t):
    """rows, rate (rows/span), median dt, max gap, number of gaps > 3x median dt and > 0.3 s, total gap seconds."""
    if t is None or len(t) < 2:
        return dict(rows=0 if t is None else len(t), rate=None, dt_med=None, max_gap=None, n_gaps=0, gap_s=0.0)
    t = np.sort(np.asarray(t, dtype=np.float64))
    d = np.diff(t)
    med = float(np.median(d))
    thr = max(3 * med, 0.3)
    g = d[d > thr]
    return dict(rows=int(len(t)), rate=round(len(t) / max(t[-1] - t[0], 1e-9), 3), dt_med=round(med, 4),
                max_gap=round(float(d.max()), 2), n_gaps=int(len(g)), gap_s=round(float(g.sum()), 1))


def _write(df, path, sort_cols):
    df = df.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    df = df.drop(columns=["mono"])
    cols = ["t_s"] + [c for c in df.columns if c != "t_s"]
    tmp = path + ".tmp"
    pq.write_table(pa.Table.from_pandas(df[cols], preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, path)
    return len(df)


def process_drive_native(job, out_dir):
    t_start = time.time()
    row = {c: None for c in MANIFEST_COLS}
    row.update(set=job["set"], driver=job["driver"], drive=job["drive"], status="error", reason="", mb_read=0.0,
               n_seg_read=0, n_seg_rlog=0, n_seg_qlog=0, n_seg_hole=0, n_seg_qlog_retry=0, n_seg_bad=0,
               n_seg_misnamed=0, log_policy="rlog_first", native=False, carparams=False)
    priv = dict(route_dir=job["route_dir"], model_dir=job["model_dir"], dongle=job["dongle"], route=job["route"],
                person=job["person"], traceback=None)
    _STASH.clear()
    try:
        n_rows = job["n_rows"]
        if n_rows is None:
            row.update(status="error", reason="released table unreadable")
            return row, priv
        D = (n_rows - 1) / eg.RATE_HZ
        row["duration_s"] = round(D, 1)
        if D < eg.MIN_D:
            row.update(status="too_short", reason=f"duration {D:.1f} s < {eg.MIN_D:.0f} s")
            return row, priv
        if not job["route_dir"] or not os.path.isdir(job["route_dir"]):
            row.update(status="route_dir_missing")
            return row, priv
        v_rel = eg.released_vego(job["released"])
        mask = eg.hole_mask(v_rel)
        attempts, verified = [], False
        for prefer in ("rlog", "qlog"):
            logs = eg.list_segment_logs(job["route_dir"], job["seg_lo"], job["seg_hi"], prefer=prefer)
            if not logs:
                row.update(status="no_logs")
                return row, priv
            if attempts and logs == attempts[-1][0]:
                break
            raw = eg.read_drive(logs, want_gnss=False)
            row["mb_read"] = round(row["mb_read"] + raw["bytes_read"] / 1e6, 1)
            if raw["cs_t"].size < 2:
                attempts.append((logs, raw, None, False, dict(row), {}))
                continue
            extra = []
            k0 = job["k_min"] if job["k_min"] in logs else min(logs)
            alts0 = eg.list_segment_alts(job["route_dir"], logs, job["seg_lo"], job["seg_hi"])
            if k0 in alts0:
                extra.append(("k_min_alt", k0, eg.first_carstate_mono(alts0[k0][0]), alts0[k0][1]))
            al = eg.align(raw, v_rel, job["k_min"], mask, extra_cands=extra)
            ok = False
            unverifiable = (np.nanmax(v_rel) <= 2.0) or (float(np.mean(mask | ~np.isfinite(v_rel))) > 0.9)
            if (al["frac"] >= eg.ANCHOR_MIN_FRAC or unverifiable) and al["cov"] >= eg.MIN_COV:
                ok = eg.repair_and_judge(raw, al, v_rel, mask, job["route_dir"], logs, job["seg_lo"], job["seg_hi"], False, row)
            else:
                row.update(align_frac=round(al["frac"], 4), align_cov=round(al["cov"], 4))
            attempts.append((logs, raw, al, ok, dict(row), alts0))
            if ok:
                verified = True
                break
        if not attempts or all(a[2] is None for a in attempts):
            row.update(status="no_carstate")
            return row, priv
        logs, raw, al, verified, saved, alts = max(attempts, key=lambda a: (a[3], a[4].get("align_frac") or -1.0))
        row.update(saved)
        row.update(op_version=raw["op_version"], n_seg_read=len(raw["segs_read"]),
                   n_seg_rlog=sum(1 for _, k in raw["segs_read"] if k == "rlog"),
                   n_seg_qlog=sum(1 for _, k in raw["segs_read"] if k == "qlog"),
                   log_policy="rlog_first" if logs is attempts[0][0] else "qlog_fallback",
                   k_anchor=al["k_anchor"], align_method=al["method"])
        if not verified:
            row.update(status="align_failed", reason=f"best {al['method']} anchor {al['frac']:.3f} kept {row.get('align_frac')} cov {row.get('align_cov')}")
            return row, priv
        t0 = al["t0_ns"]
        # kept segments and the file each one was finally read from
        segtab = al.get("segtab", {})
        kept = []
        for seg, (path, kind) in logs.items():
            st = segtab.get(seg, {})
            if st.get("dropped"):
                continue
            if st.get("retried") and seg in alts:
                path, kind = alts[seg]
            pl = _STASH.get(path)
            if pl is None:
                continue
            kept.append((seg, kind, pl))
        kept.sort(key=lambda x: x[0])
        row["n_seg_rlog"] = sum(1 for _, k, _ in kept if k == "rlog")
        row["n_seg_qlog"] = sum(1 for _, k, _ in kept if k == "qlog")
        row["native"] = row["n_seg_rlog"] > 0 and row["n_seg_qlog"] == 0
        frames = {k: [] for k in ("cs", "ctrl", "imu", "cal", "lp")}
        cols = dict(cs=CS_COLS, ctrl=CTRL_COLS, imu=IMU_COLS, cal=CAL_COLS, lp=LP_COLS)
        # carParams: majority fingerprint over kept segments, ignoring mock (dashcam / unrecognised car) segments
        fps = [pl["carparams"] for _, _, pl in kept if pl.get("carparams")]
        real = [c for c in fps if str(c.get("carFingerprint", "")).lower() != "mock"]
        cparams = None
        if fps:
            from collections import Counter
            pool = real if real else fps
            cnt = Counter(str(c.get("carFingerprint")) for c in pool)
            top, n_top = cnt.most_common(1)[0]
            cparams = next(c for c in pool if str(c.get("carFingerprint")) == top)
            row.update(fp_majority_share=round(n_top / len(pool), 3), n_fp_distinct=len(cnt), fp_mock_segs=len(fps) - len(real))
        for seg, kind, pl in kept:
            for k in frames:
                f = _frame(pl[k], cols[k], t0, D, seg, kind)
                if f is not None:
                    frames[k].append(f)
        ddir = os.path.join(out_dir, job["driver"], job["drive"])
        os.makedirs(ddir, exist_ok=True)
        names = dict(cs="carstate", ctrl="controls", imu="imu", cal="calib", lp="liveparams")
        for k, name in names.items():
            if frames[k]:
                df = pd.concat(frames[k], ignore_index=True)
                sort_cols = ["t_s", "topic"] if k == "ctrl" else (["t_s", "topic", "kind"] if k == "imu" else ["t_s"])
                n = _write(df, os.path.join(ddir, f"{name}.parquet"), sort_cols)
                if k == "cs":
                    s = _rate_stats(df["t_s"].values)
                    row.update(cs_rows=n, cs_rate_hz=s["rate"], cs_dt_med_s=s["dt_med"], cs_max_gap_s=s["max_gap"],
                               cs_n_gaps=s["n_gaps"], cs_gap_s=s["gap_s"])
                elif k == "ctrl":
                    sub = df[df["topic"] == "controlsState"]
                    s = _rate_stats(sub["t_s"].values)
                    row.update(ctrl_rows=n, ctrl_rate_hz=s["rate"])
                elif k == "imu":
                    sub = df[df["kind"] == "gyro"]
                    s = _rate_stats(sub["t_s"].values)
                    row.update(imu_rows=n, imu_rate_hz=s["rate"], imu_topic=";".join(sorted(df["topic"].unique())),
                               imu_source=";".join(sorted(df["source"].unique())))
                elif k == "cal":
                    row["cal_rows"] = n
                elif k == "lp":
                    row["lp_rows"] = n
            else:
                p = os.path.join(ddir, f"{name}.parquet")
                if os.path.exists(p):
                    os.remove(p)
        if cparams:
            with open(os.path.join(ddir, "carparams.json"), "w") as f:
                json.dump(dict(cparams, op_version=raw["op_version"]), f, indent=1)
            row.update(carparams=True, carFingerprint=cparams.get("carFingerprint"), wheelbase=cparams.get("wheelbase"),
                       steerRatio=cparams.get("steerRatio"), centerToFront=cparams.get("centerToFront"), mass=cparams.get("mass"))
        row["status"] = "ok" if row.get("cs_rows") else "no_rows_in_window"
        return row, priv
    except Exception as e:  # noqa: BLE001
        row.update(status="error", reason=eg.sanitize(f"{type(e).__name__}: {e}"))
        priv["traceback"] = traceback.format_exc()
        return row, priv
    finally:
        row["elapsed_s"] = round(time.time() - t_start, 1)
        _STASH.clear()


def process_carparams_only(job):
    t_start = time.time()
    row = dict(set=job["set"], driver=job["driver"], drive=job["drive"], status="error", reason="", op_version=None,
               carparams=False, source_log=None)
    priv = dict(route_dir=job["route_dir"], dongle=job["dongle"], route=job["route"], traceback=None)
    try:
        if not job["route_dir"] or not os.path.isdir(job["route_dir"]):
            row.update(status="route_dir_missing")
            return row, priv
        logs = eg.list_segment_logs(job["route_dir"], job["seg_lo"], job["seg_hi"], prefer="qlog")
        if not logs:
            row.update(status="no_logs")
            return row, priv
        cp = None
        for seg in sorted(logs)[:3]:
            path, kind = logs[seg]
            cp, opv = read_first_carparams(path)
            if opv:
                row["op_version"] = opv
            if cp:
                row["source_log"] = kind
                break
        if cp is None:
            row.update(status="no_carparams")
            return row, priv
        row.update(status="ok", carparams=True, **{k: cp.get(k) for k in CP_FIELDS})
        return row, priv
    except Exception as e:  # noqa: BLE001
        row.update(status="error", reason=eg.sanitize(f"{type(e).__name__}: {e}"))
        priv["traceback"] = traceback.format_exc()
        return row, priv
    finally:
        row["elapsed_s"] = round(time.time() - t_start, 1)


def _worker(job, out_dir, cp_only):
    return process_carparams_only(job) if cp_only else process_drive_native(job, out_dir)


# ----------------------------------------------------------------------------- main
def write_manifest(progress, out_dir, cp_only):
    if cp_only:
        cols = ["set", "driver", "drive", "status", "reason", "op_version", "source_log"] + CP_FIELDS + ["elapsed_s"]
        path = os.path.join(PRIV_DIR, "carparams_by_drive.csv")   # private: per-drive values could fingerprint cars
    else:
        cols = MANIFEST_COLS
        path = os.path.join(out_dir, "native_manifest.csv")
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in progress.values()], columns=cols)
    df = df.sort_values(["set", "driver", "drive"], kind="stable")
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)
    if not cp_only:
        os.makedirs(os.path.dirname(PUBLIC_MANIFEST), exist_ok=True)
        df.to_csv(PUBLIC_MANIFEST + ".tmp", index=False)
        os.replace(PUBLIC_MANIFEST + ".tmp", PUBLIC_MANIFEST)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["main", "controlled", "all"], default="all")
    ap.add_argument("--only", action="append", default=[], help="driver_XXX or driver_XXX/drive_YYY (repeatable)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--carparams-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(PRIV_DIR, exist_ok=True)
    jobs = eg.build_jobs(args.set, args.only, args.limit)
    print(f"{len(jobs)} jobs ({args.set}){' carparams-only' if args.carparams_only else ''}", flush=True)
    prog_path = CP_PROGRESS if args.carparams_only else PROGRESS
    progress = eg.load_progress(prog_path)
    terminal = {"ok", "too_short", "route_dir_missing", "no_logs", "no_carstate", "align_failed", "no_carparams", "no_rows_in_window"}
    todo = []
    for j in jobs:
        r = progress.get((j["set"], j["driver"], j["drive"]))
        if r and not args.overwrite and r.get("status") in terminal:
            if args.carparams_only or r.get("status") != "ok" or os.path.exists(os.path.join(args.out, j["driver"], j["drive"], "carstate.parquet")):
                continue
        todo.append(j)
    print(f"{len(todo)} to process, {len(jobs) - len(todo)} already done", flush=True)
    t_run, n_done = time.time(), 0
    pf = open(prog_path, "a")

    def record(row, priv):
        nonlocal n_done
        row = dict(row)
        row["_priv"] = priv
        row["run_ts"] = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        pf.write(json.dumps(row, default=str) + "\n")
        pf.flush()
        progress[(row["set"], row["driver"], row["drive"])] = row
        n_done += 1
        if n_done % 25 == 0 or n_done == len(todo):
            el = time.time() - t_run
            rate = n_done / el * 60
            eta = (len(todo) - n_done) / max(rate, 1e-6)
            mb = sum((r.get("mb_read") or 0) for r in progress.values())
            print(f"  {n_done}/{len(todo)}  {rate:.1f} drives/min  ETA {eta:.0f} min  read {mb / 1e3:.1f} GB", flush=True)
        if n_done % 200 == 0:
            write_manifest(progress, args.out, args.carparams_only)

    if args.workers <= 1:
        eg._init_worker()
        for j in todo:
            record(*_worker(j, args.out, args.carparams_only))
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=eg._init_worker) as ex:
            futs = [ex.submit(_worker, j, args.out, args.carparams_only) for j in todo]
            for fut in as_completed(futs):
                record(*fut.result())
    pf.close()
    df = write_manifest(progress, args.out, args.carparams_only)
    keys = {(j["set"], j["driver"], j["drive"]) for j in jobs}
    sub = df[[(s, d, r) in keys for s, d, r in zip(df["set"], df["driver"], df["drive"])]]
    print("status:", sub["status"].value_counts().to_dict())
    print(f"DONE in {(time.time() - t_run) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
