#!/usr/bin/env python3
"""Valhalla tiles for the OPD-II re-implementation (benchmark v1.1), built from the SAME drive network every other map
method uses: the v2 map tiles (drivable ways, nodes and turn restrictions cut from the 2026-10-22 Geofabrik snapshots).
One Valhalla tile set per connected cluster (8-neighbourhood) of the tiles needed by the evaluated episodes; every
episode's disc lies inside one cluster. pyvalhalla 3.9.1 (installed with `pip --target $CANTRAJ_BENCH/pylibs/valhalla`).

Outputs: $CANTRAJ_BENCH/maps/valhalla_v2/c{k}/{net.osm.pbf, valhalla.json, tiles/, tiles.tar} and
manifests/valhalla_v2_clusters.json ({tile: cluster}, per-cluster tiles / nodes / ways / restrictions / build seconds).

    python scripts/build_valhalla_v2.py [--folds test few_shot_heldout] [--concurrency 4] [--only c3 c7]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths  # noqa: E402
from recut_tiles_v2 import TILES_V2, needed_tiles  # noqa: E402

VLIB = os.path.join(paths.BENCH, "pylibs", "valhalla")
OUT = os.path.join(paths.BENCH, "maps", "valhalla_v2")
MAN = os.path.join(paths.MANIFESTS, "valhalla_v2_clusters.json")


def clusters(tiles):
    tiles = sorted(tiles)
    idx = {t: i for i, t in enumerate(tiles)}
    r, c = [], []
    for t in tiles:
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                u = (t[0] + di, t[1] + dj)
                if u in idx:
                    r.append(idx[t])
                    c.append(idx[u])
    _, lab = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(tiles), len(tiles))), directed=False)
    order = np.argsort(-np.bincount(lab), kind="stable")          # cluster 0 = largest
    rank = {int(o): k for k, o in enumerate(order)}
    return {t: rank[int(lab[i])] for t, i in idx.items()}


def load(tiles):
    parts = {k: [] for k in ("ways", "way_nodes", "nodes", "restrictions")}
    for t in tiles:
        d = os.path.join(TILES_V2, f"t{t[0]}_{t[1]}")
        if not os.path.exists(os.path.join(d, "ways.parquet")):
            continue
        for k in parts:
            parts[k].append(pq.read_table(os.path.join(d, f"{k}.parquet")).to_pandas())
    ways = pd.concat(parts["ways"]).drop_duplicates("way_id").sort_values("way_id")
    wn = pd.concat(parts["way_nodes"]).drop_duplicates(["way_id", "seq"]).sort_values(["way_id", "seq"])
    nodes = pd.concat(parts["nodes"]).drop_duplicates("node_id").sort_values("node_id")
    restr = pd.concat(parts["restrictions"]).drop_duplicates() if parts["restrictions"] else pd.DataFrame()
    return ways, wn, nodes, restr


def write_pbf(path, ways, wn, nodes, restr):
    import osmium
    if os.path.exists(path):
        os.remove(path)
    w = osmium.SimpleWriter(path)
    for nid, la, lo in zip(nodes.node_id.to_numpy(), nodes.lat.to_numpy(), nodes.lon.to_numpy()):
        w.add_node(osmium.osm.mutable.Node(id=int(nid), location=osmium.osm.Location(float(lo), float(la))))
    refs = wn.groupby("way_id", sort=True).node_id.apply(list)
    for r in ways.itertuples(index=False):
        tags = {"highway": str(r.highway)}
        if int(r.oneway) == 1:
            tags["oneway"] = "yes"
        elif int(r.oneway) == -1:
            tags["oneway"] = "-1"
        for k in ("junction", "maxspeed", "lanes"):
            v = getattr(r, k, "")
            if isinstance(v, str) and v:
                tags[k] = v
        nl = refs.get(r.way_id)
        if nl is None or len(nl) < 2:
            continue
        w.add_way(osmium.osm.mutable.Way(id=int(r.way_id), nodes=[int(x) for x in nl], tags=tags))
    for k, r in enumerate(restr.itertuples(index=False) if len(restr) else []):
        w.add_relation(osmium.osm.mutable.Relation(id=k + 1, members=[("w", int(r.from_way), "from"), ("n", int(r.via_node), "via"), ("w", int(r.to_way), "to")],
                                                    tags={"type": "restriction", "restriction": str(r.restriction)}))
    w.close()


def vrun(args, log):
    env = dict(os.environ, PYTHONPATH=VLIB, PATH=os.path.join(VLIB, "valhalla", "bin") + os.pathsep + os.environ.get("PATH", ""))
    with open(log, "a") as f:
        subprocess.run([sys.executable, "-m", "valhalla"] + args, check=True, env=env, stdout=f, stderr=subprocess.STDOUT)


def build(cdir, conc):
    sys.path.insert(0, VLIB)
    from valhalla import get_config
    os.makedirs(os.path.join(cdir, "tiles"), exist_ok=True)
    cfg = get_config(tile_extract="", tile_dir=os.path.join(cdir, "tiles"))   # get_config insists that paths exist
    cfg["mjolnir"]["tile_extract"] = os.path.join(cdir, "tiles.tar")
    cfg["mjolnir"]["concurrency"] = conc
    cfg["mjolnir"]["logging"] = {"type": "std_out", "color": False, "file_name": "", "long_request": 110.0}
    json.dump(cfg, open(os.path.join(cdir, "valhalla.json"), "w"), indent=1)
    log = os.path.join(cdir, "build.log")
    vrun(["valhalla_build_tiles", "-c", os.path.join(cdir, "valhalla.json"), os.path.join(cdir, "net.osm.pbf")], log)
    env = dict(os.environ, PYTHONPATH=VLIB)
    with open(log, "a") as f:
        subprocess.run([sys.executable, "-m", "valhalla.valhalla_build_extract", "-c", os.path.join(cdir, "valhalla.json"), "-v", "--overwrite"],
                       check=True, env=env, stdout=f, stderr=subprocess.STDOUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["test", "few_shot_heldout"])
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    need, n_ep = needed_tiles(args.folds)
    lab = clusters(need)
    k_max = max(lab.values()) + 1
    man = json.load(open(MAN)) if os.path.exists(MAN) else {"tile_cluster": {}, "clusters": {}}
    man["tile_cluster"].update({f"t{t[0]}_{t[1]}": f"c{c}" for t, c in lab.items()})
    print(f"{n_ep} episodes in {args.folds}: {len(need)} tiles in {k_max} clusters", flush=True)
    for c in range(k_max):
        name = f"c{c}"
        if args.only and name not in args.only:
            continue
        cdir = os.path.join(OUT, name)
        if os.path.exists(os.path.join(cdir, "tiles.tar")) and name in man["clusters"]:
            continue
        t0 = time.time()
        tiles = [t for t, cc in lab.items() if cc == c]
        os.makedirs(cdir, exist_ok=True)
        ways, wn, nodes, restr = load(tiles)
        write_pbf(os.path.join(cdir, "net.osm.pbf"), ways, wn, nodes, restr)
        t1 = time.time()
        build(cdir, args.concurrency)
        man["clusters"][name] = dict(n_tiles=len(tiles), n_nodes=int(len(nodes)), n_ways=int(len(ways)), n_restrictions=int(len(restr)),
                                     pbf_s=round(t1 - t0, 1), build_s=round(time.time() - t1, 1), valhalla="pyvalhalla 3.9.1",
                                     tar_bytes=os.path.getsize(os.path.join(cdir, "tiles.tar")))
        json.dump(man, open(MAN, "w"), indent=1)
        print(f"  {name}: {len(tiles)} tiles, {len(nodes)} nodes, {len(ways)} ways; pbf {t1 - t0:.0f}s, valhalla {time.time() - t1:.0f}s", flush=True)
    json.dump(man, open(MAN, "w"), indent=1)
    print("written", MAN, flush=True)


if __name__ == "__main__":
    main()
