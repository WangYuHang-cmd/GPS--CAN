"""OSM extraction with pyosmium: drivable ways, their node geometry and turn restrictions inside a bounding box,
written as 0.25 degree tiles (ways duplicated into every tile they touch; de-duplicated on load by way id).

Drive network policy (specs/map_policy.md): osmnx "drive" semantics: highway in DRIVE_HIGHWAYS, no area=yes, not
motor_vehicle=no / motorcar=no, access not in {private, no}, service roads excluded. oneway: yes/true/1 -> forward only,
-1/reverse -> backward only, junction=roundabout -> forward only. Connectivity is by shared node id only (grade-separated
crossings do not connect). Restrictions: relations type=restriction with from/via(node)/to ways; `no_*` forbid and
`only_*` force the (from, via, to) triple.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DRIVE_HIGHWAYS = {"motorway", "motorway_link", "trunk", "trunk_link", "primary", "primary_link", "secondary",
                  "secondary_link", "tertiary", "tertiary_link", "unclassified", "residential", "living_street", "road"}
TILE_DEG = 0.25


def tile_of(lat, lon):
    return int(math.floor(lat / TILE_DEG)), int(math.floor(lon / TILE_DEG))


def tiles_in_radius(lat0, lon0, radius_m):
    """All 0.25 deg tiles intersecting the disc of `radius_m` around (lat0, lon0)."""
    dlat = radius_m / 111320.0
    dlon = radius_m / (111320.0 * max(math.cos(math.radians(lat0)), 0.05))
    i0, j0 = tile_of(lat0 - dlat, lon0 - dlon)
    i1, j1 = tile_of(lat0 + dlat, lon0 + dlon)
    return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]


def extract(pbf_path, bbox, out_dir, source_name, snapshot_date=None, tiles=None, merge=True):
    """bbox = (lat_min, lon_min, lat_max, lon_max). Writes tiles under out_dir/t{lat_i}_{lon_j}/{nodes,ways,way_nodes,
    restrictions}.parquet and returns a manifest DataFrame (one row per tile). With `tiles` (set of (i, j)), only ways
    with a node inside one of those tiles are kept and only those tiles are written; `merge=True` appends to an existing
    tile (ways from another extract crossing the border) with de-duplication."""
    import osmium

    lat_min, lon_min, lat_max, lon_max = bbox
    tiles = set(tiles) if tiles is not None else None
    t0 = time.time()

    class H(osmium.SimpleHandler):
        def __init__(self):
            super().__init__()
            self.ways, self.way_nodes, self.nodes, self.restr = [], [], {}, []
            self.way_ids = set()

        def way(self, w):
            hw = w.tags.get("highway")
            if hw not in DRIVE_HIGHWAYS or w.tags.get("area") == "yes":
                return
            if w.tags.get("motor_vehicle") == "no" or w.tags.get("motorcar") == "no" or w.tags.get("access") in ("private", "no"):
                return
            locs = []
            inside = False
            for n in w.nodes:
                if not n.location.valid():
                    continue
                la, lo = n.location.lat, n.location.lon
                locs.append((n.ref, la, lo))
                if lat_min <= la <= lat_max and lon_min <= lo <= lon_max:
                    if tiles is None or (int(math.floor(la / TILE_DEG)), int(math.floor(lo / TILE_DEG))) in tiles:
                        inside = True
            if not inside or len(locs) < 2:
                return
            ow = w.tags.get("oneway", "")
            junction = w.tags.get("junction", "")
            oneway = 1 if ow in ("yes", "true", "1") or junction in ("roundabout", "circular") else (-1 if ow in ("-1", "reverse") else 0)
            self.ways.append(dict(way_id=w.id, highway=hw, oneway=oneway, junction=junction, maxspeed=w.tags.get("maxspeed", ""),
                                  lanes=w.tags.get("lanes", ""), name_len=len(w.tags.get("name", "")), n_nodes=len(locs)))
            self.way_ids.add(w.id)
            for k, (ref, la, lo) in enumerate(locs):
                self.way_nodes.append((w.id, k, ref))
                self.nodes[ref] = (la, lo)

        def relation(self, r):
            if r.tags.get("type") != "restriction":
                return
            rt = r.tags.get("restriction") or r.tags.get("restriction:motorcar") or ""
            if not rt:
                return
            frm = to = via = None
            via_type = None
            for m in r.members:
                if m.role == "from" and m.type == "w":
                    frm = m.ref
                elif m.role == "to" and m.type == "w":
                    to = m.ref
                elif m.role == "via":
                    via, via_type = m.ref, m.type
            if frm is None or to is None or via is None or via_type != "n":
                return
            self.restr.append(dict(from_way=frm, via_node=via, to_way=to, restriction=rt))

    h = H()
    h.apply_file(pbf_path, locations=True, idx="flex_mem")
    ways = pd.DataFrame(h.ways)
    wn = pd.DataFrame(h.way_nodes, columns=["way_id", "seq", "node_id"])
    nodes = pd.DataFrame([(k, v[0], v[1]) for k, v in h.nodes.items()], columns=["node_id", "lat", "lon"])
    restr = pd.DataFrame(h.restr) if h.restr else pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"])
    restr = restr[restr.from_way.isin(h.way_ids) & restr.to_way.isin(h.way_ids)]
    # tile assignment: a way belongs to every tile that any of its nodes falls in
    nl = wn.merge(nodes, on="node_id")
    nl["ti"] = np.floor(nl.lat / TILE_DEG).astype(int)
    nl["tj"] = np.floor(nl.lon / TILE_DEG).astype(int)
    rows = []
    os.makedirs(out_dir, exist_ok=True)
    keys = {"ways": ["way_id"], "way_nodes": ["way_id", "seq"], "nodes": ["node_id"], "restrictions": ["from_way", "via_node", "to_way", "restriction"]}
    for (ti, tj), g in nl.groupby(["ti", "tj"]):
        if tiles is not None and (int(ti), int(tj)) not in tiles:
            continue
        lat_c, lon_c = (ti + 0.5) * TILE_DEG, (tj + 0.5) * TILE_DEG
        if tiles is None and not (lat_min - TILE_DEG <= lat_c <= lat_max + TILE_DEG and lon_min - TILE_DEG <= lon_c <= lon_max + TILE_DEG):
            continue
        wids = set(g.way_id)
        tdir = os.path.join(out_dir, f"t{ti}_{tj}")
        os.makedirs(tdir, exist_ok=True)
        tw = ways[ways.way_id.isin(wids)]
        twn = wn[wn.way_id.isin(wids)]
        tn = nodes[nodes.node_id.isin(set(twn.node_id))]
        tr = restr[restr.from_way.isin(wids) | restr.to_way.isin(wids)]
        src = source_name
        for name, df in (("ways", tw), ("way_nodes", twn), ("nodes", tn), ("restrictions", tr)):
            path = os.path.join(tdir, f"{name}.parquet")
            if merge and os.path.exists(path):
                old_df = pq.read_table(path).to_pandas()
                df = pd.concat([old_df, df], ignore_index=True).drop_duplicates(keys[name])
            pq.write_table(pa.Table.from_pandas(df.reset_index(drop=True), preserve_index=False), path, compression="zstd")
        rows.append(dict(tile=f"t{ti}_{tj}", lat_min=ti * TILE_DEG, lon_min=tj * TILE_DEG, lat_max=(ti + 1) * TILE_DEG, lon_max=(tj + 1) * TILE_DEG,
                         source=src, snapshot_date=snapshot_date or "", n_ways=len(tw), n_nodes=len(tn), n_restrictions=len(tr)))
    man = pd.DataFrame(rows)
    print(f"extract {os.path.basename(pbf_path)}: {len(ways)} ways, {len(nodes)} nodes, {len(restr)} restrictions, {len(man)} tiles in {time.time() - t0:.0f}s")
    return man
