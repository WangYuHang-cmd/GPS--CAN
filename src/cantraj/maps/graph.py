"""Directed drive graph over OSM tiles, projected into an episode's ENU frame (numpy + cKDTree; no per-edge Python objects).

Edges are consecutive node pairs of drivable ways (both directions unless oneway); nodes are shared OSM node ids,
so grade-separated crossings stay disconnected. `succ(e)` lists the edges leaving the head node of e (U-turn onto the
reverse edge flagged), filtered by `no_*` / `only_*` turn restrictions. `nearest_edges` uses a KD-tree over edge
midpoints with a radius widened by half the longest edge, then exact segment distances.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial import cKDTree

from ..geo import LocalFrame
from .osm import TILE_DEG


def tile_key(ti, tj):
    """Integer key of 0.25 degree tile indices (vectorised)."""
    return (np.asarray(ti, dtype=np.int64) + 4096) * 16384 + (np.asarray(tj, dtype=np.int64) + 8192)


def parse_maxspeed(s):
    """OSM maxspeed tag -> m/s (NaN when missing or symbolic): '50' km/h, '35 mph', '50 km/h', '60;80' (first)."""
    if not isinstance(s, str) or not s:
        return np.nan
    t = s.split(";")[0].strip().lower()
    num = ""
    for ch in t:
        if ch.isdigit() or ch == ".":
            num += ch
        elif num:
            break
    if not num:
        return np.nan
    v = float(num)
    return v * 0.44704 if "mph" in t else v / 3.6


def parse_lanes(s):
    try:
        v = float(str(s).split(";")[0])
        return v if 0 < v < 20 else np.nan
    except (TypeError, ValueError):
        return np.nan


class _Out:
    def __init__(self, g):
        self.g = g

    def __getitem__(self, node):
        g = self.g
        return g._out_sorted[g._out_ptr[node]:g._out_ptr[node + 1]].tolist()


def load_tiles(tile_dir, tiles):
    ways, wn, nodes, restr = [], [], [], []
    for ti, tj in tiles:
        d = os.path.join(tile_dir, f"t{ti}_{tj}")
        if not os.path.isdir(d) or not os.path.exists(os.path.join(d, "ways.parquet")):
            continue
        ways.append(pq.read_table(os.path.join(d, "ways.parquet")).to_pandas())
        wn.append(pq.read_table(os.path.join(d, "way_nodes.parquet")).to_pandas())
        nodes.append(pq.read_table(os.path.join(d, "nodes.parquet")).to_pandas())
        restr.append(pq.read_table(os.path.join(d, "restrictions.parquet")).to_pandas())
    if not ways:
        return None
    ways = pd.concat(ways).drop_duplicates("way_id")
    wn = pd.concat(wn).drop_duplicates(["way_id", "seq"])
    nodes = pd.concat(nodes).drop_duplicates("node_id")
    restr = pd.concat(restr).drop_duplicates() if len(restr) else pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"])
    return ways, wn, nodes, restr


class DriveGraph:
    def __init__(self, ways, way_nodes, nodes, restrictions, frame: LocalFrame):
        self.frame = frame
        node_ids = nodes.node_id.values.astype(np.int64)
        order = np.argsort(node_ids, kind="stable")
        node_ids_sorted = node_ids[order]
        lat_s, lon_s = nodes.lat.values[order], nodes.lon.values[order]
        x, y = frame.to_enu(lat_s, lon_s)
        self.node_xy = np.column_stack([x, y]).astype(np.float64)
        self.node_tkey = tile_key(np.floor(lat_s / TILE_DEG), np.floor(lon_s / TILE_DEG))     # tile of every node (episode visibility)
        self.node_id = node_ids_sorted
        n_nodes = len(node_ids_sorted)
        wn = way_nodes.sort_values(["way_id", "seq"], kind="stable")
        ref = wn.node_id.values.astype(np.int64)
        pos = np.clip(np.searchsorted(node_ids_sorted, ref), 0, max(n_nodes - 1, 0))
        valid = (n_nodes > 0) & (node_ids_sorted[pos] == ref) if n_nodes else np.zeros(len(ref), bool)
        wid_all = wn.way_id.values.astype(np.int64)[valid]
        pos = pos[valid]
        same = wid_all[1:] == wid_all[:-1]
        a, b, w = pos[:-1][same], pos[1:][same], wid_all[:-1][same]
        keep = a != b
        a, b, w = a[keep], b[keep], w[keep]
        wmeta = ways.drop_duplicates("way_id").set_index("way_id")
        ow = wmeta["oneway"].reindex(w).fillna(0).to_numpy(dtype=int)
        hw = wmeta["highway"].reindex(w).fillna("road").to_numpy()
        ms = (wmeta["maxspeed"].map(parse_maxspeed) if "maxspeed" in wmeta else pd.Series(np.nan, index=wmeta.index)).reindex(w).to_numpy(dtype=float)
        ln = (wmeta["lanes"].map(parse_lanes) if "lanes" in wmeta else pd.Series(np.nan, index=wmeta.index)).reindex(w).to_numpy(dtype=float)
        fwd, bwd = ow >= 0, ow <= 0
        self.u = np.concatenate([a[fwd], b[bwd]]).astype(np.int64)
        self.v = np.concatenate([b[fwd], a[bwd]]).astype(np.int64)
        self.way_id = np.concatenate([w[fwd], w[bwd]]).astype(np.int64)
        self.highway = np.concatenate([hw[fwd], hw[bwd]])
        self.maxspeed = np.concatenate([ms[fwd], ms[bwd]])          # m/s, NaN when untagged (Elastic Pathing)
        self.lanes = np.concatenate([ln[fwd], ln[bwd]])              # total lanes of the way, NaN when untagged
        self.oneway = np.concatenate([ow[fwd], ow[bwd]]) != 0
        d = self.node_xy[self.v] - self.node_xy[self.u]
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 1], d[:, 0])
        n_e = len(self.u)
        # CSR adjacency over the tail node
        o = np.argsort(self.u, kind="stable")
        self._out_sorted = o
        self._out_ptr = np.searchsorted(self.u[o], np.arange(n_nodes + 1))
        # reverse edge (same way, swapped nodes) by sorted key lookup
        M = n_nodes + 1
        key = (self.u * M + self.v) * 1 + 0
        skey = np.lexsort((self.way_id, key))
        keys_sorted = key[skey]
        wids_sorted = self.way_id[skey]
        rkey = self.v * M + self.u
        self.reverse = np.full(n_e, -1, dtype=np.int64)
        if n_e:
            lo = np.searchsorted(keys_sorted, rkey, side="left")
            hi = np.searchsorted(keys_sorted, rkey, side="right")
            for e in np.flatnonzero(hi > lo):
                cand = skey[lo[e]:hi[e]]
                m = cand[wids_sorted[lo[e]:hi[e]] == self.way_id[e]]
                if len(m):
                    self.reverse[e] = m[0]
        # restrictions as forbidden (from_edge, to_edge) pairs and only-sets, via sorted (head, way) / (tail, way) keys
        self.forbid = set()
        self.only = {}
        if len(restrictions) and n_e:
            head_key = self.v * M + 0
            hk = np.lexsort((self.way_id, self.v))
            tk = np.lexsort((self.way_id, self.u))
            hv, hw_ = self.v[hk], self.way_id[hk]
            tu, tw_ = self.u[tk], self.way_id[tk]
            for r in restrictions.itertuples(index=False):
                p = int(np.searchsorted(node_ids_sorted, int(r.via_node)))
                if p >= n_nodes or node_ids_sorted[p] != int(r.via_node):
                    continue
                i0, i1 = np.searchsorted(hv, p, "left"), np.searchsorted(hv, p, "right")
                ins = [int(hk[i]) for i in range(i0, i1) if hw_[i] == int(r.from_way)]
                j0, j1 = np.searchsorted(tu, p, "left"), np.searchsorted(tu, p, "right")
                outs = [int(tk[j]) for j in range(j0, j1) if tw_[j] == int(r.to_way)]
                if not ins or not outs:
                    continue
                if str(r.restriction).startswith("no_"):
                    for a_ in ins:
                        for b_ in outs:
                            self.forbid.add((a_, b_))
                elif str(r.restriction).startswith("only_"):
                    for a_ in ins:
                        self.only.setdefault(a_, set()).update(outs)
        mid = 0.5 * (self.node_xy[self.u] + self.node_xy[self.v]) if n_e else np.zeros((0, 2))
        self._tree = cKDTree(mid) if n_e else None
        self._half_max = 0.5 * float(self.length.max()) if n_e else 0.0

    @property
    def out(self):
        return _Out(self)

    def n_edges(self):
        return len(self.u)

    def succ(self, e):
        """Outgoing edges at the head of e with turn angle (rad, left-positive) and a U-turn flag; sorted by edge id."""
        outs = self.out[self.v[e]]
        if e in self.only:
            outs = [o for o in outs if o in self.only[e]]
        res = []
        for o in sorted(outs):
            if (e, o) in self.forbid:
                continue
            turn = (self.bearing[o] - self.bearing[e] + math.pi) % (2 * math.pi) - math.pi
            res.append((o, turn, o == self.reverse[e]))
        return res

    def seg_distance(self, es, x, y):
        """Exact point-to-segment distances for edge ids `es`."""
        es = np.asarray(es, dtype=np.int64)
        a = self.node_xy[self.u[es]]
        b = self.node_xy[self.v[es]]
        ab = b - a
        L2 = np.einsum("ij,ij->i", ab, ab)
        p = np.array([x, y], dtype=float)
        t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.where(L2 > 0, L2, 1.0), 0.0, 1.0)
        proj = a + t[:, None] * ab
        return np.hypot(proj[:, 0] - x, proj[:, 1] - y)

    def nearest_edges(self, x, y, radius_m):
        if self._tree is None:
            return []
        idx = np.asarray(self._tree.query_ball_point([x, y], radius_m + self._half_max), dtype=np.int64)
        if len(idx) == 0:
            return []
        d = self.seg_distance(idx, x, y)
        k = idx[d <= radius_m]
        return [int(i) for i in k[np.argsort(d[d <= radius_m], kind="stable")]]

    def project(self, e, x, y):
        a, b = self.node_xy[self.u[e]], self.node_xy[self.v[e]]
        ab = b - a
        L2 = float(ab @ ab)
        t = 0.0 if L2 == 0 else float(np.clip(((np.array([x, y]) - a) @ ab) / L2, 0.0, 1.0))
        p = a + t * ab
        return t * self.length[e], float(np.hypot(x - p[0], y - p[1]))

    def point(self, e, offset_m):
        a, b = self.node_xy[self.u[e]], self.node_xy[self.v[e]]
        L = self.length[e]
        t = 0.0 if L == 0 else float(np.clip(offset_m / L, 0.0, 1.0))
        return a + t * (b - a)

    def points(self, es, offsets):
        es = np.asarray(es, dtype=np.int64)
        a, b = self.node_xy[self.u[es]], self.node_xy[self.v[es]]
        L = self.length[es]
        t = np.clip(np.where(L > 0, np.asarray(offsets, dtype=float) / np.where(L > 0, L, 1.0), 0.0), 0.0, 1.0)
        return a + t[:, None] * (b - a)


def build_graph(tile_dir, lat0, lon0, radius_m, frame=None):
    """DriveGraph from every tile intersecting the disc (lat0, lon0, radius_m), in the frame centred at (lat0, lon0)."""
    from .osm import tiles_in_radius
    frame = frame or LocalFrame(lat0, lon0)
    loaded = load_tiles(tile_dir, tiles_in_radius(lat0, lon0, radius_m))
    if loaded is None:
        return None
    ways, wn, nodes, restr = loaded
    return DriveGraph(ways, wn, nodes, restr, frame)


def build_graph_tiles(tile_dir, tiles, frame):
    """DriveGraph from an explicit tile set (e.g. the union of the episode discs of one drive), in the given frame."""
    loaded = load_tiles(tile_dir, sorted(set(tiles)))
    if loaded is None:
        return None
    ways, wn, nodes, restr = loaded
    return DriveGraph(ways, wn, nodes, restr, frame)


class EpisodeGraphView:
    """A DriveGraph built in a drive-level frame (one graph per drive and window class, the union of the episodes' discs),
    viewed from one episode: queries given in episode ENU are converted exactly (through WGS-84) into the graph frame and
    returned positions are converted back. Benchmark v1.1:
    - visibility: with `visible_tiles` (the episode's own tile set, tiles_in_radius(p0, min(r(D), cap))) only edges with at
      least one end node inside those tiles exist for the episode (`nearest_edges`, `succ`); other episodes' discs of the
      same drive are never visible, whatever the cache holds;
    - bearings are re-expressed in the episode frame: the two azimuthal frames differ by the meridian convergence between
      their origins (several degrees for episodes far from a long drive's centroid), measured once at the episode origin.
    Edge lengths, successors and highway classes are shared (length distortion < 0.1% within 500 km)."""

    def __init__(self, graph: DriveGraph, episode_frame: LocalFrame, visible_tiles=None):
        self.g = graph
        self.ef = episode_frame
        self.frame = episode_frame
        lat0, lon0 = episode_frame.to_lla(0.0, 0.0)
        latn, lonn = episode_frame.to_lla(0.0, 100.0)
        gx0, gy0 = graph.frame.to_enu(lat0, lon0)
        gxn, gyn = graph.frame.to_enu(latn, lonn)
        self.rot = float(math.atan2(float(gyn) - float(gy0), float(gxn) - float(gx0)) - math.pi / 2)   # episode north seen in the graph frame
        self.bearing = (graph.bearing - self.rot + math.pi) % (2 * math.pi) - math.pi
        if visible_tiles is not None:
            vt = list(visible_tiles)
            keys = tile_key([t[0] for t in vt], [t[1] for t in vt])
            nv = np.isin(graph.node_tkey, keys)
            self.visible = nv[graph.u] | nv[graph.v]
        else:
            self.visible = None

    def n_visible(self):
        return int(self.visible.sum()) if self.visible is not None else self.g.n_edges()

    def succ(self, e):
        res = self.g.succ(e)                                   # turn angles are bearing differences: frame-invariant
        if self.visible is None:
            return res
        return [s for s in res if self.visible[s[0]]]

    def __getattr__(self, name):
        return getattr(self.g, name)

    def _to_graph(self, x, y):
        lat, lon = self.ef.to_lla(x, y)
        gx, gy = self.g.frame.to_enu(lat, lon)
        return float(gx), float(gy)

    def _to_episode(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        lat, lon = self.g.frame.to_lla(pts[:, 0], pts[:, 1])
        x, y = self.ef.to_enu(lat, lon)
        return np.column_stack([x, y])

    def nearest_edges(self, x, y, radius_m):
        gx, gy = self._to_graph(x, y)
        es = self.g.nearest_edges(gx, gy, radius_m)
        if self.visible is None:
            return es
        return [e for e in es if self.visible[e]]

    def project(self, e, x, y):
        gx, gy = self._to_graph(x, y)
        return self.g.project(e, gx, gy)

    def point(self, e, offset_m):
        return self._to_episode(self.g.point(e, offset_m))[0]

    def points(self, es, offsets):
        return self._to_episode(self.g.points(es, offsets))
