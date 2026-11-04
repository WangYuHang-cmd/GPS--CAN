#!/usr/bin/env python3
"""Tool cache for the agent study: run the three frozen reconstructors (map beam, map PF lineage read-out, Elastic
Pathing with start heading) once per cohort window under protocol F-M-Hp, with the drive loading, graph building,
visibility and GPS denial replicated from run_baseline.py::_drive_job, and write per-window caches (candidates,
observable diagnostics, junction tables) plus the standard prediction files:

    tool_beam__G3 / tool_pf__G3 / tool_ep__G1      every candidate of the tool (identity check against icde_v11)
    poolcand_<cid>__G3|G1                           one stem per candidate, candidate_id 0 (evaluator-side oracle)

    python scripts/run_agent_tools.py --run icde_agent_val --manifest manifests/icde_agent_val.csv --workers 3 [--resume]
    python scripts/run_agent_tools.py --run icde_agent_val --only beam --episode driver_007/drive_001/w05/k000 --no-cache

The cache holds no reference position, no score and no start coordinates; `t_abs` in candidates.npz is program-side
only (needed to write prediction files) and never reaches the LLM.
"""
import argparse
import glob
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
sys.path.insert(0, HERE)
from cantraj import io, paths, predfile, registry  # noqa: E402
from cantraj.agent.runner import CACHE_VERSION, TOOL_SPECS, TOOLS, ToolRunner  # noqa: E402
from cantraj.episodes import DriveData, ref_times  # noqa: E402
from cantraj.geo import LocalFrame  # noqa: E402
from cantraj.maps.graph import EpisodeGraphView, build_graph_tiles  # noqa: E402
from cantraj.maps.osm import tiles_in_radius  # noqa: E402
from run_baseline import MAP_CAP_M, _deny, sha_files, static_lookup  # noqa: E402

PROTO = "F-M-Hp"
WIN = "w05"
GROUP = "G3"


def cache_root(run):
    return os.path.join(paths.BENCH, "agent_toolcache", run)


def cache_dir(root, driver, drive, k):
    return os.path.join(root, driver, drive, f"k{int(k):03d}")


def sha256_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_manifest(args):
    src = glob.glob(os.path.join(paths.GPS_DIR, "src", "cantraj", "**", "*.py"), recursive=True)
    tm = os.path.join(paths.MANIFESTS, "map_tiles_v2.csv")
    return dict(cohort=os.path.basename(args.manifest), cohort_sha=sha256_file(args.manifest), tiles_dir=os.path.basename(os.path.normpath(args.tiles_dir)),
                tiles_manifest_sha=sha256_file(tm) if os.path.exists(tm) else None, code_sha=sha_files(src), cache_version=CACHE_VERSION,
                tools={k: dict(stem=v["stem"], ref_stem=v["ref_stem"], group=v["group"]) for k, v in TOOL_SPECS.items()},
                protocol=PROTO, group=GROUP, map_episodes_sha=sha256_file(args.map_episodes))


def _drive_job(job):
    driver, drive, eids, model_canon, tiles_dir, root, only, no_cache = job
    t_start = time.time()
    out = dict(drive=f"{driver}/{drive}", windows={}, load_s=None, graph_s=None, errors=[], live=None)
    gman = io.load_manifest().set_index(["driver", "drive"])
    r = gman.loc[(driver, drive)]
    delta = io.time_compensation(r["gps_sources"], r["gps_rate_hz"])
    real_load = io.load_gps
    try:
        dd = DriveData(driver, drive, delta_s=delta, input_layer="events")
        eps = {e.episode_id: e for e in dd.episodes(WIN)}           # every w05 episode of the drive, as run_baseline does
    except Exception as e:  # noqa: BLE001
        out["errors"].append((f"{driver}/{drive}", f"drive load failed: {type(e).__name__}: {str(e)[:60]}"))
        return out
    out["load_s"] = time.time() - t_start
    static, _plat = static_lookup()(driver, drive, model_canon)
    io.load_gps = _deny
    try:
        spec = registry.protocol_spec(PROTO)
        w_eps = list(eps.values())
        anchors = np.array([[e.init_full["p0_lat"], e.init_full["p0_lon"]] for e in w_eps])
        fr = LocalFrame(float(anchors[:, 0].mean()), float(anchors[:, 1].mean()))
        tiles = set()
        for e in w_eps:
            tiles.update(tiles_in_radius(e.init_full["p0_lat"], e.init_full["p0_lon"], min(registry.map_radius_m(spec, e.t_end - e.t0), MAP_CAP_M)))
        t_g = time.time()
        g_drive = build_graph_tiles(tiles_dir, tiles, fr)
        out["graph_s"] = time.time() - t_g
        for eid in eids:
            ep = eps.get(eid)
            if ep is None:
                out["errors"].append((eid, "episode not rebuilt"))
                continue
            need_t = tiles_in_radius(ep.init_full["p0_lat"], ep.init_full["p0_lon"], min(registry.map_radius_m(spec, ep.t_end - ep.t0), MAP_CAP_M))
            have = {f"t{i}_{j}" for i, j in tiles_in_radius(ep.init_full["p0_lat"], ep.init_full["p0_lon"], MAP_CAP_M)
                    if os.path.exists(os.path.join(tiles_dir, f"t{i}_{j}", "ways.parquet"))}
            if g_drive is None or not all(f"t{i}_{j}" in have for i, j in need_t):
                out["errors"].append((eid, "map_missing"))
                continue
            graph = EpisodeGraphView(g_drive, ep.frame, visible_tiles=need_t)
            inp = dd.inputs(ep, GROUP, PROTO, static, map_graph=graph)
            tr = ToolRunner(inp, ref_times(ep), graph, inp.init.theta0, ep.t0, ep.t_end)
            if only:                                     # live, uncached single-tool execution (cost verification)
                res = tr.run_tool(only)
                out["live"] = dict(episode_id=eid, tool=only, tool_s=res["secs"], graph_s=out["graph_s"], load_s=out["load_s"],
                                   n_candidates=len(res["candidates"]), error=res["error"])
                continue
            tr.run_all()
            cdir = cache_dir(root, driver, drive, ep.k)
            if not no_cache:
                diag = tr.write_cache(cdir, extra=dict(episode_id=eid, driver=driver, drive=drive, k=int(ep.k), graph_build_s=out["graph_s"], load_s=out["load_s"]))
                np.savez_compressed(os.path.join(cdir, "t_abs.npz"), t_abs=tr.t_query)      # program-side only
                with open(os.path.join(cdir, "status.json"), "w") as f:
                    json.dump(dict(episode_id=eid, tools={k: dict(n_candidates=v["n_candidates"], secs=v["logical_time_s"], error=v["error"], stats=v["stats"])
                                                            for k, v in diag["tools"].items()}), f)
            out["windows"][eid] = {k: dict(n=len(v["candidates"]), secs=v["secs"], error=v["error"]) for k, v in tr.results.items()}
    except Exception as e:  # noqa: BLE001
        out["errors"].append((f"{driver}/{drive}", f"{type(e).__name__}: {str(e)[:100]}"))
    finally:
        io.load_gps = real_load
    return out


def write_parts(run, root, cohort, args, rm):
    """Rebuild every prediction part from the caches (the caches are the source of truth; idempotent)."""
    out_dir = os.path.join(paths.PREDS, run, PROTO)
    os.makedirs(out_dir, exist_ok=True)
    frames, timing, skipped = {}, {}, {}
    n_win = 0
    for _, row in cohort.iterrows():
        cdir = cache_dir(root, row.driver, row.drive, row.k)
        if not os.path.exists(os.path.join(cdir, "diagnostics.json")):
            continue
        n_win += 1
        eid = row.episode_id
        with open(os.path.join(cdir, "status.json")) as f:
            st = json.load(f)
        arr = np.load(os.path.join(cdir, "candidates.npz"))
        t = np.load(os.path.join(cdir, "t_abs.npz"))["t_abs"]
        with open(os.path.join(cdir, "diagnostics.json")) as f:
            probs = {k: v["probs"] for k, v in json.load(f)["tools"].items()}
        for name in TOOLS:
            spec = TOOL_SPECS[name]
            stem = f"{spec['stem']}__{spec['group']}"
            s = st["tools"][name]
            timing.setdefault(stem, []).append((PROTO, stem, eid, s["secs"], int(s["n_candidates"] > 0)))
            if s["n_candidates"] == 0:
                skipped.setdefault(stem, []).append((eid, f"{stem}: {s['error']}"))
                continue
            cids = [f"{name}-{k + 1}" for k in range(s["n_candidates"])]
            pr = probs.get(name) or [1.0] * len(cids)
            th = arr[f"theta__{name}"] if f"theta__{name}" in arr else None
            fl = [predfile.new_frame(eid, t, arr[f"xy__{cids[0]}"], th, candidate_id=0, prob=float(pr[0]) if len(cids) > 1 else 1.0)]
            for k in range(1, len(cids)):
                fl.append(predfile.new_frame(eid, t, arr[f"xy__{cids[k]}"], None, candidate_id=k, prob=float(pr[k])))
            frames.setdefault(stem, []).extend(fl)
            for cid in cids:
                pstem = f"poolcand_{cid}__{spec['group']}"
                frames.setdefault(pstem, []).append(predfile.new_frame(eid, t, arr[f"xy__{cid}"], None, candidate_id=0, prob=1.0))
    stems = sorted(set(frames) | set(timing))
    for stem in stems:
        fl = frames.get(stem, [])
        base = os.path.join(out_dir, f"{stem}.part0")
        if fl:
            df = pd.concat(fl, ignore_index=True)
            predfile.write(df, base + ".parquet", dict(baseline=stem.split("__")[0], group=stem.split("__")[1], protocol=PROTO, run=run, layer="events",
                                                      time_comp="train", manifest=args.manifest, benchmark_version="v1.1", shard="0/1",
                                                      n_episodes=int(df.episode_id.nunique()), n_skipped=len(skipped.get(stem, [])),
                                                      skipped=skipped.get(stem, [])[:50], n_map_missing=0, partial=False, drives_done=n_win,
                                                      drives_total=len(cohort), tiles_dir=args.tiles_dir, map_version=rm["tiles_dir"],
                                                      tiles_manifest_sha=rm["tiles_manifest_sha"], code_sha=rm["code_sha"], map_episodes=os.path.basename(args.map_episodes),
                                                      map_episodes_sha=rm["map_episodes_sha"], episode_visibility=True, manifest_sha=rm["cohort_sha"],
                                                      agent_cache_version=CACHE_VERSION, study="agent_toolcache"))
        if stem in timing:
            pd.DataFrame(timing[stem], columns=["protocol", "baseline", "key", "secs", "ok"]).to_csv(base + ".timing.csv", index=False)
        if stem in skipped:
            pd.DataFrame(skipped[stem], columns=["episode_id", "reason"]).to_csv(base + ".skipped.csv", index=False)
    print(f"{n_win} cached windows -> {len(stems)} stems in {out_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", required=True, help="cohort csv (episode_id, driver, drive, fold, k)")
    ap.add_argument("--episode-manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    ap.add_argument("--tiles-dir", default=os.path.join(paths.BENCH, "maps", "tiles_v2"))
    ap.add_argument("--map-episodes", default=os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N windows of the cohort (interface checks)")
    ap.add_argument("--only", default=None, choices=list(TOOLS), help="live mode: run one tool for --episode, print timing JSON, write nothing")
    ap.add_argument("--episode", default=None)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--parts-only", action="store_true", help="only rebuild the prediction parts from the caches")
    args = ap.parse_args()
    cohort = pd.read_csv(args.manifest)
    man = pd.read_parquet(args.episode_manifest, columns=["episode_id", "model_canon"]).set_index("episode_id")
    root = cache_root(args.run)
    os.makedirs(root, exist_ok=True)
    rm = run_manifest(args)
    rm_path = os.path.join(root, "run_manifest.json")
    if os.path.exists(rm_path):
        old = json.load(open(rm_path))
        diff = {k: (old.get(k), rm[k]) for k in rm if old.get(k) != rm[k]}
        if diff and not args.only:
            raise SystemExit(f"run manifest differs from the existing cache ({list(diff)}); use a new --run or delete the cache: {diff}")
    elif not args.only:
        json.dump(rm, open(rm_path, "w"), indent=1)
    if args.only:
        if not args.episode:
            raise SystemExit("--only needs --episode")
        row = cohort[cohort.episode_id == args.episode]
        if row.empty:
            raise SystemExit("episode not in the cohort manifest")
        row = row.iloc[0]
        res = _drive_job((row.driver, row.drive, [args.episode], man.model_canon.get(args.episode), args.tiles_dir, root, args.only, True))
        print(json.dumps(res["live"] or dict(errors=res["errors"])))
        return
    if args.parts_only:
        write_parts(args.run, root, cohort, args, rm)
        return
    if args.limit:
        cohort = cohort.head(args.limit)
    todo = cohort
    if args.resume:
        done = [os.path.exists(os.path.join(cache_dir(root, r.driver, r.drive, r.k), "diagnostics.json")) for r in cohort.itertuples()]
        todo = cohort[~np.array(done, dtype=bool)]
    jobs = [(d, v, list(g.episode_id), man.model_canon.get(g.episode_id.iloc[0]), args.tiles_dir, root, None, args.no_cache)
            for (d, v), g in todo.groupby(["driver", "drive"], sort=True)]
    print(f"{len(cohort)} windows in cohort, {len(todo)} to run over {len(jobs)} drives, workers {args.workers}, cache {root}", flush=True)
    t_run = time.time()
    n, errors, summary = 0, [], {k: dict(ok=0, fail=0, secs=[]) for k in TOOLS}

    def collect(res):
        nonlocal n
        n += 1
        errors.extend(res["errors"])
        for eid, tools in res["windows"].items():
            for k, v in tools.items():
                summary[k]["ok" if v["n"] else "fail"] += 1
                summary[k]["secs"].append(v["secs"])
        if n % 25 == 0:
            print(f"  {n}/{len(jobs)} drives, {time.time() - t_run:.0f}s", flush=True)

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_drive_job, jobs, chunksize=1):
                collect(res)
    else:
        for j in jobs:
            collect(_drive_job(j))
    for k, s in summary.items():
        sec = np.array(s["secs"]) if s["secs"] else np.zeros(1)
        print(f"{k}: ok {s['ok']}, fail {s['fail']}, secs p50 {np.median(sec):.2f} p95 {np.quantile(sec, 0.95):.2f} max {sec.max():.1f}", flush=True)
    if errors:
        print("errors:", pd.Series([e[1][:40] for e in errors]).value_counts().head(5).to_dict(), flush=True)
        pd.DataFrame(errors, columns=["key", "reason"]).to_csv(os.path.join(root, "errors.csv"), index=False)
    if not args.no_cache:
        write_parts(args.run, root, pd.read_csv(args.manifest), args, rm)
    print(f"done in {(time.time() - t_run) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
