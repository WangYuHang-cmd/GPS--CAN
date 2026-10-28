#!/usr/bin/env python3
"""Benchmark v1.1 map eligibility per episode, decided from the map and the start point only (never from a method):
- start_edge_dist_m: distance from p0 to the nearest drivable edge of the v2 map (tiles within 400 m of p0, all of which
  belong to the episode's own visible disc);
- map_start_ok: start_edge_dist_m <= 400 m (the start-search radius every map method uses);
- n_tiles_missing: tiles of the episode's disc min(r(D), 60 km) without a v2 ways file (unconfigured land);
- map_ok = map_start_ok and n_tiles_missing == 0. Map protocols run only map_ok episodes; the others are reported.
Writes manifests/episode_map_v11.parquet and prints the counts by fold and window.

    python scripts/build_map_episodes.py [--folds val test few_shot_heldout]
"""
import argparse
import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths, registry  # noqa: E402
from cantraj.maps.osm import tiles_in_radius  # noqa: E402

MAP_CAP_M = 60000.0
START_R = 400.0
TILES_V2 = os.path.join(paths.BENCH, "maps", "tiles_v2")
OUT = os.path.join(paths.MANIFESTS, "episode_map_v11.parquet")
M_PER_DEG = 111320.0


class TileSegs:
    """Per-tile drivable segments (consecutive way nodes) in degrees with a KD-tree over midpoints; LRU cache."""

    def __init__(self, cap=96):
        self.cap, self.c = cap, OrderedDict()

    def get(self, t):
        if t in self.c:
            self.c.move_to_end(t)
            return self.c[t]
        d = os.path.join(TILES_V2, f"t{t[0]}_{t[1]}")
        seg = None
        if os.path.exists(os.path.join(d, "ways.parquet")) and pq.ParquetFile(os.path.join(d, "ways.parquet")).metadata.num_rows:
            wn = pq.read_table(os.path.join(d, "way_nodes.parquet")).to_pandas().sort_values(["way_id", "seq"])
            nodes = pq.read_table(os.path.join(d, "nodes.parquet")).to_pandas().set_index("node_id")
            wn = wn[wn.node_id.isin(nodes.index)]
            lat = nodes.lat.reindex(wn.node_id).to_numpy()
            lon = nodes.lon.reindex(wn.node_id).to_numpy()
            same = wn.way_id.to_numpy()[1:] == wn.way_id.to_numpy()[:-1]
            a = np.column_stack([lat[:-1][same], lon[:-1][same]])
            b = np.column_stack([lat[1:][same], lon[1:][same]])
            if len(a):
                mid = 0.5 * (a + b)
                cs = math.cos(math.radians((t[0] + 0.5) * 0.25))       # one longitude scale per tile, used for the tree AND the query
                half = 0.5 * np.hypot((b - a)[:, 0], (b - a)[:, 1] * cs) * M_PER_DEG
                seg = (a, b, cKDTree(mid * [1.0, cs]), float(half.max()), cs)
        self.c[t] = seg
        if len(self.c) > self.cap:
            self.c.popitem(last=False)
        return seg


def start_distance(cache, lat0, lon0):
    best = np.inf
    cl = math.cos(math.radians(lat0))
    for t in tiles_in_radius(lat0, lon0, START_R):
        seg = cache.get(t)
        if seg is None:
            continue
        a, b, tree, half, cs = seg
        r_deg = 1.05 * (START_R + half) / M_PER_DEG                    # 5% margin for the per-tile longitude scale
        idx = tree.query_ball_point([lat0, lon0 * cs], r_deg)
        if not idx:
            continue
        idx = np.asarray(idx)
        ax, ay = (a[idx, 1] - lon0) * cl * M_PER_DEG, (a[idx, 0] - lat0) * M_PER_DEG
        bx, by = (b[idx, 1] - lon0) * cl * M_PER_DEG, (b[idx, 0] - lat0) * M_PER_DEG
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        u = np.clip(-(ax * dx + ay * dy) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        d = np.hypot(ax + u * dx, ay + u * dy)
        best = min(best, float(d.min()))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["val", "test", "few_shot_heldout"])
    args = ap.parse_args()
    t0 = time.time()
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "fold", "win", "eligible_main"])
    man = man[man.eligible_main & man.fold.isin(args.folds)]
    p0 = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_p0.parquet")).merge(man[["episode_id", "fold", "win"]], on="episode_id")
    p0["ti"] = np.floor(p0.lat0 / 0.25).astype(int)
    p0["tj"] = np.floor(p0.lon0 / 0.25).astype(int)
    p0 = p0.sort_values(["ti", "tj"])
    spec = registry.protocol_spec("O-M-Hp")
    cache = TileSegs()
    have = {}
    rows = []
    for r in p0.itertuples(index=False):
        need = tiles_in_radius(r.lat0, r.lon0, min(registry.map_radius_m(spec, r.dur_s), MAP_CAP_M))
        miss = 0
        for t in need:
            if t not in have:
                have[t] = os.path.exists(os.path.join(TILES_V2, f"t{t[0]}_{t[1]}", "ways.parquet"))
            miss += not have[t]
        dist = start_distance(cache, r.lat0, r.lon0)
        rows.append(dict(episode_id=r.episode_id, fold=r.fold, win=r.win, start_edge_dist_m=dist, map_start_ok=bool(dist <= START_R),
                         n_tiles_needed=len(need), n_tiles_missing=int(miss)))
    out = pd.DataFrame(rows)
    out["map_ok"] = out.map_start_ok & (out.n_tiles_missing == 0)
    out.to_parquet(OUT, index=False)
    s = out.groupby(["fold", "win"]).agg(n=("episode_id", "size"), map_ok=("map_ok", "sum"), start_missing=("map_start_ok", lambda x: int((~x).sum())),
                                          tiles_missing=("n_tiles_missing", lambda x: int((x > 0).sum())),
                                          dist_p50=("start_edge_dist_m", "median"), dist_p99=("start_edge_dist_m", lambda x: float(np.quantile(x[np.isfinite(x)], 0.99))))
    print(s.to_string())
    print(f"start distance > 50 m: {(out.start_edge_dist_m > 50).mean():.4f}; written {OUT} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
