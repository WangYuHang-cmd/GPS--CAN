#!/usr/bin/env python3
"""run_baseline.py: run one baseline over an episode manifest for given protocols and a signal group.

    python scripts/run_baseline.py --baseline dr_steer --group G3 --protocols O-N-Hp O-N-Hm --run full_v1 \
        --manifest manifests/episode_manifest_v1.parquet --eligible-only --workers 10 [--shard 0/4] [--resume]

Baselines: p0_hold, prior, dr_none, retrieval (G1); dr_steer, dr_steer_offfixed, dr_steer_calib, dr_steer_offonline (G3);
dr_can_yaw (G4); dr_curvature, dr_loc_yaw (G5); map_pf_{none,steer,steer_calib,can_yaw,curvature} (map protocols).
Static parameters come from manifests/vehicle_params.csv keyed by the drive's platform (manifests/drive_platform.csv),
with manifests/calibration_global.json as the unseen-platform fallback for the calibrated variants.
The reference loader is disabled while baselines run (io.load_gps raises). Prediction files are written per shard
(`<baseline>__<group>.part<k>.parquet`); --resume skips episodes already present in any part. Map protocols build one
graph per (drive, window class) around the drive's anchors and view it from each episode frame; episodes whose start
has no tiles are recorded in preds/<run>/map_missing.csv and skipped (not failures).
"""
import argparse
import glob
import math
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths, predfile, registry, replay  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
from cantraj.inputs import StaticParams  # noqa: E402
from cantraj.baselines.dead_reckoning import DeadReckoning  # noqa: E402
from cantraj.baselines.p0_hold import P0Hold  # noqa: E402
from cantraj.baselines.prior import StartPrior  # noqa: E402
from cantraj.baselines.retrieval import RouteRetrieval  # noqa: E402
from cantraj.baselines.map_pf import MapParticleFilter  # noqa: E402
from cantraj.baselines.learned import LearnedDR  # noqa: E402
from cantraj.baselines.map_beam import MapBeamSearch  # noqa: E402
from cantraj.learn.provider import MotionProvider  # noqa: E402
from cantraj.maps.graph import EpisodeGraphView, build_graph_tiles  # noqa: E402
from cantraj.perturb import episode_rng, parse_spec, perturb_stream  # noqa: E402
from cantraj.maps.osm import tiles_in_radius  # noqa: E402
from cantraj.geo import LocalFrame  # noqa: E402

BASELINES = {
    "p0_hold": lambda: P0Hold(),
    "dr_steer": lambda: DeadReckoning("steer", offset_mode="none"),
    "dr_steer_offonline": lambda: DeadReckoning("steer", offset_mode="online"),
    "dr_steer_flip": lambda: DeadReckoning("steer", sign_flip=True, offset_mode="none"),
    "dr_steer_offfixed": lambda: DeadReckoning("steer", offset_mode="fixed"),
    "dr_steer_calib": lambda: DeadReckoning("steer", offset_mode="fixed", use_eff_ratio=True),
    "dr_can_yaw": lambda: DeadReckoning("can_yaw"),
    "dr_curvature": lambda: DeadReckoning("curvature"),
    "dr_loc_yaw": lambda: DeadReckoning("loc_yaw"),
    "dr_none": lambda: DeadReckoning("none"),
    "dr_imu_yaw": lambda: DeadReckoning("imu_yaw"),                       # device gyro, standstill bias (G8)
    "dr_imu_yaw_sb": lambda: DeadReckoning("imu_yaw_sb"),                 # device gyro, steering-referenced bias (G8)
    "map_pf_imu_yaw": lambda: MapParticleFilter("imu_yaw"),
    "dr_imu_yaw_rls": lambda: DeadReckoning("imu_yaw_rls"),              # device gyro, bias + steering offset by least squares on speed (G8)
    "map_pf_imu_yaw_rls": lambda: MapParticleFilter("imu_yaw_rls"),
    "prior": lambda: StartPrior(),
    "retrieval": lambda: RouteRetrieval(),
    "map_pf_none": lambda: MapParticleFilter("none"),
    "map_pf_steer": lambda: MapParticleFilter("steer"),
    "map_pf_steer_offfixed": lambda: MapParticleFilter("steer", offset_mode="fixed"),
    "map_pf_steer_calib": lambda: MapParticleFilter("steer", offset_mode="fixed", use_eff_ratio=True),
    "map_pf_can_yaw": lambda: MapParticleFilter("can_yaw"),
    "map_pf_curvature": lambda: MapParticleFilter("curvature"),
    "map_beam_none": lambda: MapBeamSearch("none"),
    "map_beam_steer": lambda: MapBeamSearch("steer"),
    "map_beam_steer_calib": lambda: MapBeamSearch("steer", offset_mode="fixed", use_eff_ratio=True),
    "map_beam_curvature": lambda: MapBeamSearch("curvature"),
    "learned": lambda: LearnedDR(CKPT),                                    # name from the checkpoint (learned_gru / learned_resid)
    "map_pf_learned": lambda: MapParticleFilter("steer", omega_source=MotionProvider(CKPT)),
}
CKPT = None
PERTURB, PERTURB_SPEC = {}, ""
INIT_PERTURB, INIT_SPEC = {}, ""


def perturb_init(init, spec, rng):
    """Shift p0 by a random offset of fixed length pos_m (uniform direction) and theta0 by N(0, heading_deg)."""
    import dataclasses
    th = init.theta0
    dx, dy = init.dx_m, init.dy_m
    d = float(spec.get("pos_m", 0.0))
    if d > 0:                                     # shift in the episode frame: baselines start from (dx, dy) instead of the true p0
        ang = rng.uniform(0, 2 * math.pi)
        dx, dy = dx + d * math.cos(ang), dy + d * math.sin(ang)
    sd = float(spec.get("heading_deg", 0.0))
    if sd > 0 and th is not None:
        th = float(th + rng.normal(0.0, math.radians(sd)))
    return dataclasses.replace(init, theta0=th, dx_m=dx, dy_m=dy)
YAWRATE_BRAND = {"FORD": ("rad_s", 1), "VOLKSWAGEN": ("rad_s", 1), "SKODA": ("rad_s", 1), "AUDI": ("rad_s", 1),
                 "HYUNDAI": ("deg_s", 1), "KIA": ("deg_s", 1), "GENESIS": ("deg_s", 1), "HONDA": ("deg_s", 1),
                 "CHEVROLET": ("deg_s_x10", 1), "GMC": ("deg_s_x10", 1), "CADILLAC": ("deg_s_x10", 1)}
GLOBAL_CALIB = None
_gc = os.path.join(paths.MANIFESTS, "calibration_global.json")
if os.path.exists(_gc):
    GLOBAL_CALIB = json.load(open(_gc))
MAP_CAP_M = 60000.0


def static_lookup():
    vp = pd.read_csv(os.path.join(paths.MANIFESTS, "vehicle_params.csv")).set_index("platform")
    dp = pd.read_csv(os.path.join(paths.MANIFESTS, "drive_platform.csv")).set_index(["driver", "drive"])

    def _num(r, c):
        try:
            v = float(r[c])
            return v if np.isfinite(v) else None
        except (KeyError, TypeError, ValueError):
            return None

    def get(driver, drive, model_canon=None):
        plat = dp.loc[(driver, drive), "platform"] if (driver, drive) in dp.index else None
        row = vp.loc[plat] if plat in vp.index else None
        brand = (plat or model_canon or "").split("_")[0]
        unit, sign = YAWRATE_BRAND.get(brand, (None, None))
        if row is not None and isinstance(row.get("yawrate_unit"), str) and row["yawrate_unit"]:
            unit, sign = row["yawrate_unit"], int(float(row["yawrate_sign"]))
        off = _num(row, "angle_offset_train_deg") if row is not None else None
        sr_eff = _num(row, "steer_ratio_eff_train") if row is not None else None
        sr_nom = _num(row, "steer_ratio") if row is not None else None
        if (off is None or sr_eff is None) and GLOBAL_CALIB is not None and sr_nom is not None:
            off = GLOBAL_CALIB["angle_offset_deg"] if off is None else off
            sr_eff = sr_nom * GLOBAL_CALIB["steer_ratio_factor"] if sr_eff is None else sr_eff
        return StaticParams(model_canon=model_canon, brand=brand, platform=plat, angle_offset_train_deg=off or 0.0,
                            steer_ratio_eff_train=sr_eff, wheelbase_m=_num(row, "wheelbase_m") if row is not None else None,
                            steer_ratio=sr_nom, center_to_front_m=_num(row, "center_to_front_m") if row is not None else None,
                            mass_kg=_num(row, "mass_kg") if row is not None else None, yawrate_unit=unit, yawrate_sign=sign), plat
    return get


def git_hash():
    try:
        return subprocess.check_output(["git", "-C", paths.DRIVEDNA, "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _deny(*a, **k):
    raise PermissionError("reference positions are not available while baselines run")


def _drive_job(job):
    """Run every requested (protocol, baseline) on the episodes of one drive after loading it once.
    Returns ({(proto, baseline): frames}, skipped, {(proto, baseline): missing_eids}, secs)."""
    (driver, drive, eids_by_proto, wins, model_canon, protos, baselines, group, layer, time_comp, tiles_dir, ckpt, perturb_spec, warmup_s) = job
    global CKPT, PERTURB, PERTURB_SPEC, INIT_PERTURB, INIT_SPEC
    CKPT = ckpt
    PERTURB_SPEC = (perturb_spec or "").split("||")[0]     # passed through the job tuple: module globals are not shared with the workers
    PERTURB = parse_spec(PERTURB_SPEC)
    INIT_SPEC = (perturb_spec or "").split("||")[1] if perturb_spec and "||" in perturb_spec else ""
    INIT_PERTURB = {k.strip(): float(v) for k, v in (kv.split("=") for kv in INIT_SPEC.split(",") if kv.strip())} if INIT_SPEC else {}
    t0 = time.time()
    gman = io.load_manifest().set_index(["driver", "drive"])
    r = gman.loc[(driver, drive)]
    delta = io.time_compensation(r["gps_sources"], r["gps_rate_hz"]) if time_comp == "train" else 0.0
    real_load = io.load_gps
    frames, skipped, missing = {}, [], {}
    try:
        dd = DriveData(driver, drive, delta_s=delta, input_layer=layer, with_imu=(group == "G8"))
        eps = {}
        for w in wins:
            for e in dd.episodes(w):
                eps[e.episode_id] = e
    except Exception as e:  # noqa: BLE001
        return {}, [(f"{driver}/{drive}", f"drive load failed: {type(e).__name__}: {str(e)[:60]}")], {}, time.time() - t0
    static, plat = static_lookup()(driver, drive, model_canon)
    io.load_gps = _deny
    RouteRetrieval._exclude = driver
    graphs = {}
    try:
        for proto in protos:
            spec = registry.protocol_spec(proto)
            for baseline in baselines:
                key = (proto, baseline)
                eids = eids_by_proto.get(key, [])
                for eid in eids:
                    ep = eps.get(eid)
                    if ep is None:
                        skipped.append((eid, "episode not rebuilt"))
                        continue
                    graph = None
                    if spec["map"] == "osm":
                        win = ep.win
                        if win not in graphs:
                            # one graph per drive and window class = the union of the episodes' own discs (radius rule
                            # r(D) capped at MAP_CAP_M around each p0); a long drive gets a corridor, not a 90 km disc
                            w_eps = [e for e in eps.values() if e.win == win]
                            anchors = np.array([[e.init_full["p0_lat"], e.init_full["p0_lon"]] for e in w_eps])
                            lat_c, lon_c = float(anchors[:, 0].mean()), float(anchors[:, 1].mean())
                            fr = LocalFrame(lat_c, lon_c)
                            tiles = set()
                            for e in w_eps:
                                tiles.update(tiles_in_radius(e.init_full["p0_lat"], e.init_full["p0_lon"], min(registry.map_radius_m(spec, e.t_end - e.t0), MAP_CAP_M)))
                            graphs[win] = build_graph_tiles(tiles_dir, tiles, fr)
                        g_drive = graphs[win]
                        need = [f"t{i}_{j}" for i, j in tiles_in_radius(ep.init_full["p0_lat"], ep.init_full["p0_lon"], min(registry.map_radius_m(spec, ep.t_end - ep.t0), MAP_CAP_M))]
                        have = {f"t{i}_{j}" for i, j in tiles_in_radius(ep.init_full["p0_lat"], ep.init_full["p0_lon"], MAP_CAP_M) if os.path.exists(os.path.join(tiles_dir, f"t{i}_{j}", "ways.parquet"))}
                        if g_drive is None or not all(t in have for t in need):
                            missing.setdefault(key, []).append(eid)
                            continue
                        graph = EpisodeGraphView(g_drive, ep.frame)
                    try:
                        inp = dd.inputs(ep, group, proto, static, map_graph=graph, warmup_s=warmup_s)
                        if PERTURB:                     # research ablation: degrade the inputs the baseline sees (never the reference)
                            inp.events = perturb_stream(inp.events, PERTURB, episode_rng(eid, PERTURB_SPEC))
                        if INIT_PERTURB:                # research ablation: wrong start position / heading given to the baseline
                            inp.init = perturb_init(inp.init, INIT_PERTURB, episode_rng(eid, "init|" + INIT_SPEC))
                        b = BASELINES[baseline]()
                        frames.setdefault(key, []).append(replay.run_episode(b, inp, ep, spec))
                    except Exception as e:  # noqa: BLE001
                        skipped.append((eid, f"{baseline}: {type(e).__name__}: {str(e)[:60]}"))
    finally:
        io.load_gps = real_load
    return frames, skipped, missing, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", default=None, choices=sorted(BASELINES))
    ap.add_argument("--baselines", nargs="+", default=None, help="several baselines of the same group in one pass over the drives")
    ap.add_argument("--group", required=True)
    ap.add_argument("--protocols", nargs="+", required=True)
    ap.add_argument("--run", default="dev")
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest.parquet"))
    ap.add_argument("--wins", nargs="+", default=None)
    ap.add_argument("--layer", default="events", choices=["events", "grid_compat"])
    ap.add_argument("--time-comp", default="train", choices=["train", "none"])
    ap.add_argument("--eligible-only", action="store_true")
    ap.add_argument("--folds", nargs="+", default=None, help="restrict to these folds (manifest column `fold`)")
    ap.add_argument("--drives-file", default=None, help="CSV with driver,drive columns to restrict the run")
    ap.add_argument("--shard", default="0/1", help="k/N: process drives whose hash mod N == k")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tiles-dir", default=f"{paths.BENCH}/maps/tiles")
    ap.add_argument("--version", default="")
    ap.add_argument("--ckpt", default=None, help="model directory for the learned baselines (weights.npz + meta.json)")
    ap.add_argument("--tag", default=None, help="file stem override (single baseline only), e.g. learned_gru_s0")
    ap.add_argument("--perturb", default=None, help="perturbation probe spec, e.g. speed_quant=1,speed_noise=0.5,steer_hz=1 (inputs only; research ablation)")
    ap.add_argument("--warmup-s", default=None, help="warm-start variant: seconds of the drive's own CAN/IMU history before t0 given to the baseline for calibration ('inf' = all past); research rows, not the frozen main board")
    ap.add_argument("--init-perturb", default=None, help="initialisation ablation, e.g. pos_m=20,heading_deg=10: shift the given start / heading (research ablation)")
    ap.add_argument("--stem-suffix", default="", help="appended to every method stem, e.g. _pq1 -> dr_steer_pq1__G3 (use with --perturb)")
    ap.add_argument("--flush-min", type=float, default=15.0, help="rewrite the part files with everything predicted so far every N minutes (0 = only at the end); lets --resume recover a killed pass")
    args = ap.parse_args()
    global CKPT
    CKPT = args.ckpt
    baselines = args.baselines or [args.baseline]
    if not baselines or baselines == [None]:
        raise SystemExit("--baseline or --baselines required")
    for b in baselines:
        if b not in BASELINES:
            raise SystemExit(f"unknown baseline {b}")
    man = pd.read_parquet(args.manifest)
    if args.wins:
        man = man[man.win.isin(args.wins)]
    if args.eligible_only:
        man = man[man.eligible_main]
    if args.folds and "fold" in man:
        man = man[man.fold.isin(args.folds)]
    if args.drives_file:
        dl = pd.read_csv(args.drives_file)
        keys = set(zip(dl.driver, dl.drive))
        man = man[[(a, b) in keys for a, b in zip(man.driver, man.drive)]]
    k, N = (int(x) for x in args.shard.split("/"))
    if N > 1:
        man = man[[int(hashlib.md5(d.encode()).hexdigest(), 16) % N == k for d in man.driver]]
    t_run = time.time()
    stems = {b: f"{(args.tag if (args.tag and len(baselines) == 1) else b)}{args.stem_suffix}__{args.group}" for b in baselines}
    done = {}
    for proto in args.protocols:
        out_dir = os.path.join(paths.PREDS, args.run, proto)
        os.makedirs(out_dir, exist_ok=True)
        for b in baselines:
            d = set()
            if args.resume:
                for f in glob.glob(os.path.join(out_dir, f"{stems[b]}*.parquet")):
                    try:
                        d |= set(pd.read_parquet(f, columns=["episode_id"]).episode_id.unique())
                    except Exception:
                        pass
            done[(proto, b)] = d
    jobs = []
    for (driver, drive), g in man.groupby(["driver", "drive"], sort=True):
        eids_by = {}
        for proto in args.protocols:
            for b in baselines:
                e = [x for x in g.episode_id if x not in done[(proto, b)]]
                if e:
                    eids_by[(proto, b)] = e
        if not eids_by:
            continue
        jobs.append((driver, drive, eids_by, sorted(g.win.unique()), g.model_canon.iloc[0] if "model_canon" in g else None,
                     list(args.protocols), list(baselines), args.group, args.layer, args.time_comp, args.tiles_dir, args.ckpt,
                     f"{args.perturb or ''}||{args.init_perturb}" if args.init_perturb else args.perturb,
                     (float("inf") if str(args.warmup_s).lower() in ("inf", "all") else float(args.warmup_s)) if args.warmup_s else 0.0))
    print(f"{len(jobs)} drives, baselines {baselines}, protocols {args.protocols}, workers {args.workers}", flush=True)
    frames, skipped, missing, secs = {}, [], {}, []
    n_done = 0
    last_flush = [time.time()]

    def write_parts(final):
        """Write one part file per (protocol, baseline) with everything predicted so far (merged with an existing part under --resume)."""
        for proto in args.protocols:
            out_dir = os.path.join(paths.PREDS, args.run, proto)
            for b in baselines:
                key = (proto, b)
                part = os.path.join(out_dir, f"{stems[b]}.part{k}.parquet")
                fl = frames.get(key, [])
                mi = missing.get(key, [])
                if fl:
                    df = pd.concat(fl, ignore_index=True)
                    if args.resume and os.path.exists(part):
                        try:
                            df = pd.concat([pd.read_parquet(part), df], ignore_index=True).drop_duplicates(["episode_id", "candidate_id", "t"])
                        except Exception:
                            pass
                    predfile.write(df, part, dict(baseline=b, group=args.group, protocol=proto, run=args.run, layer=args.layer, time_comp=args.time_comp,
                                                  manifest=args.manifest, benchmark_version=args.version, shard=args.shard, n_episodes=int(df.episode_id.nunique()),
                                                  n_skipped=len(skipped), skipped=skipped[:50], n_map_missing=len(mi), git=git_hash(), ckpt=args.ckpt,
                                                  wall_s=round(time.time() - t_run, 1), drive_secs_p50=float(np.median(secs)) if secs else None,
                                                  partial=(not final), drives_done=n_done, drives_total=len(jobs),
                                                  pf_tick_hz=(1.0 / BASELINES[b]().tick_dt) if getattr(BASELINES[b](), "tick_dt", None) else None,
                                                  perturb=args.perturb or None, init_perturb=args.init_perturb or None, warmup_s=args.warmup_s or None))
                    if final:
                        print(f"{proto} {b} {args.group} shard {args.shard}: {df.episode_id.nunique()} episodes, {len(df)} rows, map_missing {len(mi)} -> {part}", flush=True)
                elif final:
                    print(f"{proto} {b} {args.group} shard {args.shard}: nothing new; map_missing {len(mi)}", flush=True)
                if final and mi:
                    mm_path = os.path.join(paths.PREDS, args.run, "map_missing.csv")
                    mm = pd.DataFrame(dict(episode_id=mi, protocol=proto, method=stems[b]))
                    if os.path.exists(mm_path):
                        mm = pd.concat([pd.read_csv(mm_path), mm]).drop_duplicates()
                    mm.to_csv(mm_path, index=False)
        if not final:
            print(f"  partial parts written after {n_done}/{len(jobs)} drives, {time.time() - t_run:.0f}s", flush=True)

    def collect(fr, sk, mi, sc):
        nonlocal n_done
        for key, lst in fr.items():
            frames.setdefault(key, []).extend(lst)
        for key, lst in mi.items():
            missing.setdefault(key, []).extend(lst)
        skipped.extend(sk); secs.append(sc); n_done += 1
        if n_done % 200 == 0:
            print(f"  {n_done}/{len(jobs)} drives, {time.time() - t_run:.0f}s", flush=True)
        if args.flush_min > 0 and n_done < len(jobs) and time.time() - last_flush[0] >= 60.0 * args.flush_min:
            write_parts(final=False)
            last_flush[0] = time.time()

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_drive_job, jobs, chunksize=1):
                collect(*res)
    else:
        for j in jobs:
            collect(*_drive_job(j))
    write_parts(final=True)
    if skipped:
        print("skip reasons:", pd.Series([s_[1] for s_ in skipped]).value_counts().head(6).to_dict(), flush=True)
    print(f"done in {(time.time() - t_run) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
