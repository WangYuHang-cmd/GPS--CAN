#!/usr/bin/env python3
"""CAN-Trace failure mode (RQ4; analysis only: it reads the GPS reference, which no baseline ever sees).

For every pedal-subset episode that the CAN-Trace rows were expected to predict: the number of trace nodes CAN-Trace
builds (the known start plus the stops and pedal releases left after its short-edge merge; same code as the runs) vs
the number of intersections the reference path passes (nodes of undirected degree >= 3 of the episode's visible v2 map
within 12 m of the 1 Hz reference polyline; an approximation that can also count a node of a road running alongside).
The matcher maps every trace edge to exactly one road segment between adjacent intersections, so a window whose path
passes more intersections than its trace has edges has no exact match. Output: reports/icde/tables/can_trace_failure_mode.csv.

    python scripts/can_trace_failure_mode.py [--wins w05] [--folds test few_shot_heldout]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths, registry  # noqa: E402
from cantraj.baselines.can_trace import CANTrace, _SegGraph  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
from cantraj.geo import LocalFrame  # noqa: E402
from cantraj.inputs import StaticParams  # noqa: E402
from cantraj.maps.graph import EpisodeGraphView, build_graph_tiles  # noqa: E402
from cantraj.maps.osm import tiles_in_radius  # noqa: E402

MAP_CAP_M = 60000.0
TILES = f"{paths.BENCH}/maps/tiles_v2"
OUT = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "can_trace_failure_mode.csv")


def trace_nodes(ct, inp, t_query):
    """Trace nodes as CANTrace.reconstruct builds them for the start-known variant (start + merged candidates)."""
    nodes = ct._line_graph(inp, t_query)
    sg = _SegGraph(inp.map_graph)
    merged = [t for k, t in enumerate(nodes) if not (k + 1 < len(nodes) and ct._dist(t, nodes[k + 1]) < sg.min_seg)]
    return [float(inp.init.t0)] + [t for t in merged if t > inp.init.t0]


def intersections_passed(view, ref, radius=12.0):
    """Intersections (undirected degree >= 3, visible) within `radius` m of the densified reference polyline."""
    g = view.g
    a, b = np.minimum(g.u, g.v), np.maximum(g.u, g.v)
    key = np.unique(a * (len(g.node_xy) + 1) + b)
    a, b = key // (len(g.node_xy) + 1), key % (len(g.node_xy) + 1)
    deg = np.bincount(a, minlength=len(g.node_xy)) + np.bincount(b, minlength=len(g.node_xy))
    vis_nodes = np.zeros(len(g.node_xy), dtype=bool)
    vis_nodes[g.u[view.visible]] = True
    vis_nodes[g.v[view.visible]] = True
    cand = np.flatnonzero((deg >= 3) & vis_nodes)
    xy = view._to_episode(g.node_xy[cand])
    x, y = ref.x.to_numpy(float), ref.y.to_numpy(float)
    seg = []
    for k in range(len(x) - 1):                                    # densify to <= 5 m steps
        n = max(1, int(np.ceil(np.hypot(x[k + 1] - x[k], y[k + 1] - y[k]) / 5.0)))
        s = np.linspace(0.0, 1.0, n, endpoint=False)
        seg.append(np.column_stack([x[k] + s * (x[k + 1] - x[k]), y[k] + s * (y[k + 1] - y[k])]))
    pts = np.vstack(seg + [np.array([[x[-1], y[-1]]])])
    lo, hi = pts.min(axis=0) - radius, pts.max(axis=0) + radius
    inside = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
    xy = xy[inside]
    if not len(xy):
        return 0
    from scipy.spatial import cKDTree
    d, _ = cKDTree(pts).query(xy)
    return int((d <= radius).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wins", nargs="+", default=["w05"])
    ap.add_argument("--folds", nargs="+", default=["test", "few_shot_heldout"])
    args = ap.parse_args()
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "driver", "drive", "fold", "win", "eligible_main"])
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"), columns=["episode_id", "map_ok"])
    aw = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"), columns=["driver", "drive", "gas", "G3"])
    m = man[man.eligible_main & man.fold.isin(args.folds) & man.win.isin(args.wins)].merge(me, on="episode_id").merge(aw, on=["driver", "drive"])
    m = m[m.map_ok.astype(bool) & (m.gas == "valid") & m.G3.fillna(False).astype(bool)]
    outcome = {}
    for f in glob.glob(os.path.join(paths.PREDS, "icde_v11", "F-M-Hm", "can_trace__G2.part*.parquet")):
        for e in pd.read_parquet(f, columns=["episode_id"]).episode_id.unique():
            outcome[str(e)] = "matched"
    for f in glob.glob(os.path.join(paths.PREDS, "icde_v11", "F-M-Hm", "can_trace__G2.part*.skipped.csv")):
        for e, r in pd.read_csv(f)[["episode_id", "reason"]].itertuples(index=False):
            outcome.setdefault(e, "no match" if "NoMatch" in r else r[:40])
    spec = registry.protocol_spec("F-M-Hm")
    gman = io.load_manifest().set_index(["driver", "drive"])
    rows = []
    for (driver, drive), eps_m in m.groupby(["driver", "drive"]):
        r = gman.loc[(driver, drive)]
        dd = DriveData(driver, drive, delta_s=io.time_compensation(r["gps_sources"], r["gps_rate_hz"]), input_layer="events")
        want = set(eps_m.episode_id)
        for win in eps_m.win.unique():
            eps = [e for e in dd.episodes(win) if e.episode_id in want]
            if not eps:
                continue
            tiles = set()
            for e in eps:
                tiles.update(tiles_in_radius(e.init_full["p0_lat"], e.init_full["p0_lon"], min(registry.map_radius_m(spec, e.t_end - e.t0), MAP_CAP_M)))
            fr = LocalFrame(float(np.mean([e.init_full["p0_lat"] for e in eps])), float(np.mean([e.init_full["p0_lon"] for e in eps])))
            g = build_graph_tiles(TILES, tiles, fr)
            for e in eps:
                need_t = tiles_in_radius(e.init_full["p0_lat"], e.init_full["p0_lon"], min(registry.map_radius_m(spec, e.t_end - e.t0), MAP_CAP_M))
                view = EpisodeGraphView(g, e.frame, visible_tiles=need_t)
                inp = dd.inputs(e, "G2", "F-M-Hm", StaticParams(), map_graph=view)
                ref = e.ref
                tq = ref.time_s.to_numpy(float)
                try:
                    n_nodes = len(trace_nodes(CANTrace(start="known"), inp, tq))
                except Exception as ex:  # noqa: BLE001
                    n_nodes = -1 if "pedal" in str(ex) else -2
                rows.append(dict(episode_id=e.episode_id, driver=driver, win=win, outcome=outcome.get(e.episode_id, "not run"),
                                 trace_nodes=n_nodes, intersections_passed=intersections_passed(view, ref),
                                 path_len_m=float(np.hypot(np.diff(ref.x), np.diff(ref.y)).sum())))
        del dd
        print(f"{driver}/{drive}: {len(rows)} episodes so far", flush=True)
    t = pd.DataFrame(rows)
    t["trace_edges"] = t.trace_nodes - 1
    t["fewer_edges_than_intersections"] = t.trace_edges < t.intersections_passed
    t.to_csv(OUT, index=False)
    print(t.groupby("outcome").agg(n=("episode_id", "size"), trace_nodes_med=("trace_nodes", "median"),
                                   intersections_med=("intersections_passed", "median"),
                                   share_fewer=("fewer_edges_than_intersections", "mean")).round(2).to_string())


if __name__ == "__main__":
    main()
