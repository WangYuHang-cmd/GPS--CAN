#!/usr/bin/env python3
"""Plan and cut map tiles from EVERY eligible episode's own start point and radius (the runner's rule), not from the
drive median: long drives leave the disc around their median. Writes manifests/episode_p0.parquet (episode start
lat/lon, duration), cuts the missing land tiles from the configured Geofabrik extracts, materialises EMPTY tiles for
tiles that intersect no land at all (open sea), reports tiles on unconfigured land, and writes
manifests/map_coverage_episodes_v1.csv (per-episode need / missing / kind).

    python scripts/plan_episode_tiles.py [--folds val test few_shot_heldout] [--workers 6] [--max-tiles-per-pass 150] [--dry-run]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import io, paths, registry  # noqa: E402
from cantraj.maps.osm import TILE_DEG, tiles_in_radius  # noqa: E402
import fetch_maps_all as fm  # noqa: E402

MAP_CAP_M = 60000.0
P0_PATH = os.path.join(paths.MANIFESTS, "episode_p0.parquet")
COV_PATH = os.path.join(paths.MANIFESTS, "map_coverage_episodes_v1.csv")
SCHEMAS = {
    "ways": pa.schema([("way_id", pa.int64()), ("highway", pa.string()), ("oneway", pa.int64()), ("junction", pa.string()), ("maxspeed", pa.string()), ("lanes", pa.string()), ("name_len", pa.int64()), ("n_nodes", pa.int64())]),
    "way_nodes": pa.schema([("way_id", pa.int64()), ("seq", pa.int64()), ("node_id", pa.int64())]),
    "nodes": pa.schema([("node_id", pa.int64()), ("lat", pa.float64()), ("lon", pa.float64())]),
    "restrictions": pa.schema([("from_way", pa.int64()), ("via_node", pa.int64()), ("to_way", pa.int64()), ("restriction", pa.string())]),
}


def _p0_for_drive(task):
    driver, drive, eps = task          # eps: list of (episode_id, t0_s, t_end_s)
    gman = io.load_manifest().set_index(["driver", "drive"])
    r = gman.loc[(driver, drive)]
    delta = io.time_compensation(r["gps_sources"], r["gps_rate_hz"])
    g = io.load_gps(driver, drive, delta)
    t = g["time_s"].to_numpy(dtype=float)
    lat, lon = g["lat"].to_numpy(dtype=float), g["lon"].to_numpy(dtype=float)
    out = []
    for eid, t0, t1 in eps:
        i = int(np.argmin(np.abs(t - t0)))
        out.append(dict(episode_id=eid, lat0=float(lat[i]), lon0=float(lon[i]), dt_fix_s=float(abs(t[i] - t0)), dur_s=float(t1 - t0)))
    return out


def episode_p0(man, workers):
    if os.path.exists(P0_PATH):
        p0 = pd.read_parquet(P0_PATH)
        if set(man.episode_id) <= set(p0.episode_id):
            return p0[p0.episode_id.isin(man.episode_id)]
    tasks = [(d, v, list(zip(g.episode_id, g.t0_s, g.t_end_s))) for (d, v), g in man.groupby(["driver", "drive"])]
    rows = []
    t0 = time.time()
    with Pool(workers) as pool:
        for k, out in enumerate(pool.imap_unordered(_p0_for_drive, tasks, chunksize=4)):
            rows += out
            if (k + 1) % 500 == 0:
                print(f"  p0: {k + 1}/{len(tasks)} drives, {time.time() - t0:.0f}s", flush=True)
    p0 = pd.DataFrame(rows)
    if os.path.exists(P0_PATH):
        old = pd.read_parquet(P0_PATH)
        p0 = pd.concat([old[~old.episode_id.isin(p0.episode_id)], p0], ignore_index=True)
    p0.to_parquet(P0_PATH, index=False)
    return p0[p0.episode_id.isin(man.episode_id)]


def tile_name(t):
    return f"t{t[0]}_{t[1]}"


def has_ways(t):
    return os.path.exists(os.path.join(fm.TILES, tile_name(t), "ways.parquet"))


def write_empty_tile(t, source, rows):
    d = os.path.join(fm.TILES, tile_name(t))
    os.makedirs(d, exist_ok=True)
    for name, schema in SCHEMAS.items():
        pq.write_table(schema.empty_table(), os.path.join(d, f"{name}.parquet"))
    rows.append(dict(tile=tile_name(t), lat_min=t[0] * TILE_DEG, lon_min=t[1] * TILE_DEG, lat_max=(t[0] + 1) * TILE_DEG, lon_max=(t[1] + 1) * TILE_DEG,
                     source=source, snapshot_date=time.strftime("%Y-%m-%d"), n_ways=0, n_nodes=0, n_restrictions=0, pbf_md5=""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    ap.add_argument("--folds", nargs="+", default=None, help="default: every eligible episode")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tiles-per-pass", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--finalize-only", action="store_true", help="skip downloading/cutting: only write empty tiles for planned tiles without roads and recount coverage")
    args = ap.parse_args()
    man = pd.read_parquet(args.manifest)
    man = man[man.eligible_main]
    if args.folds:
        man = man[man.fold.isin(args.folds)]
    spec = registry.protocol_spec("O-M-Hp")
    print(f"{len(man)} eligible episodes, {man.groupby(['driver', 'drive']).ngroups} drives", flush=True)
    p0 = episode_p0(man, args.workers).merge(man[["episode_id", "fold", "win", "driver", "drive"]], on="episode_id")
    print(f"p0 table: {len(p0)} episodes, max |t_fix - t0| {p0.dt_fix_s.max():.2f} s", flush=True)
    # needed tiles per episode
    need_by_ep = {}
    need_all = {}
    for r in p0.itertuples(index=False):
        rad = min(registry.map_radius_m(spec, r.dur_s), MAP_CAP_M)
        ts = tiles_in_radius(r.lat0, r.lon0, rad)
        need_by_ep[r.episode_id] = ts
        for t in ts:
            need_all.setdefault(t, 0)
            need_all[t] += 1
    missing = sorted(t for t in need_all if not has_ways(t))
    print(f"needed tiles {len(need_all)}, missing {len(missing)}", flush=True)
    # classify missing tiles: configured extracts / unconfigured land / sea
    polys = fm.polygons()
    cj = json.load(open(fm.COUNTRIES))
    land = unary_union([shape(f["geometry"]) for f in cj["features"]]).buffer(0.0)
    by_extract, sea, unmapped = {}, [], []
    for t in missing:
        b = fm.tile_box(*t)
        hits = [k for k, g in polys.items() if g.intersects(b)]
        if hits:
            for k in hits:
                by_extract.setdefault(k, set()).add(t)
        elif land.intersects(b):
            unmapped.append(t)
        else:
            sea.append(t)
    print(f"missing tiles by kind: in configured extracts {sum(len(v) for v in by_extract.values())} tile-extract pairs over {len(by_extract)} extracts; sea {len(sea)}; unconfigured land {len(unmapped)}", flush=True)
    for k, v in sorted(by_extract.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k}: {len(v)} tiles", flush=True)
    if args.dry_run:
        return
    rows = []
    for t in sea:
        write_empty_tile(t, "empty:sea", rows)
    done_sources = set(pd.read_csv(fm.TILE_MANIFEST).tile + "|" + pd.read_csv(fm.TILE_MANIFEST).source.astype(str)) if os.path.exists(fm.TILE_MANIFEST) else set()
    done_sources = {s.replace("geofabrik:", "") for s in done_sources}
    for k, tiles in sorted(by_extract.items(), key=lambda kv: -len(kv[1])):
        if args.finalize_only:
            break
        t0 = time.time()
        try:
            got = fm.tile_extract(k, tiles, args.max_tiles_per_pass, done_sources)
        except Exception as e:  # noqa: BLE001
            print(f"  {k}: FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        rows += got
        print(f"  {k}: {len(tiles)} tiles cut in {time.time() - t0:.0f}s", flush=True)
    # planned tiles that every intersecting extract left without roads: materialise them empty (no ways there)
    n_empty = 0
    for k, tiles_k in by_extract.items():
        for t in tiles_k:
            if not has_ways(t):
                write_empty_tile(t, "empty:no-roads", rows)
                n_empty += 1
    print(f"planned tiles without roads written empty: {n_empty}", flush=True)
    fm.update_tile_manifest(rows)
    # per-episode coverage after the cut
    kinds = {t: "sea" for t in sea} | {t: "unmapped_land" for t in unmapped}
    cov = []
    for r in p0.itertuples(index=False):
        ts = need_by_ep[r.episode_id]
        miss = [t for t in ts if not has_ways(t)]
        kind = ";".join(sorted({kinds.get(t, "extract") for t in miss})) if miss else ""
        cov.append(dict(episode_id=r.episode_id, fold=r.fold, win=r.win, n_need=len(ts), n_missing=len(miss), missing_kind=kind))
    cov = pd.DataFrame(cov)
    cov.to_csv(COV_PATH, index=False)
    print(cov.groupby(["fold", "win"]).apply(lambda t: pd.Series(dict(episodes=len(t), covered=int((t.n_missing == 0).sum())))).to_string(), flush=True)
    print("written", COV_PATH, flush=True)


if __name__ == "__main__":
    main()
