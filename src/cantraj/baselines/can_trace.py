"""CAN-Trace re-implemented from Lin, Ma, Wang, Yu, He, Ni, Liu, "CAN-Trace Attack: Exploit CAN Messages to Uncover
Driving Trajectories", IEEE T-ITS 26(3):3223-3236, 2025 (arXiv 2507.09624). No author code or data is available.

Paper algorithm. Algorithm 1 (CAN to graph): candidate stops = speed samples equal to 0, candidate turns = accelerator
pedal samples at the trace's minimum (pedal released); the time gaps between consecutive candidates are clustered with
K-means and the largest gap of the cluster with the smallest gaps becomes the threshold Delta_s / Delta_p; a candidate
whose gap to the previous candidate is at least the threshold becomes a node (time-stamped); nodes are sorted by time; a
node is deleted when the speed-integrated distance (rectangle rule) to the next node is shorter than the shortest road
segment of the area; consecutive nodes form a line graph with those distances as edge weights. Algorithm 2: subgraphs of
the road graph (nodes = intersections, edges = road segments weighted by length, undirected) isomorphic to the line graph
whose every edge satisfies |w_trace - w_road| <= sigma * w_road are found with VF2; sigma is increased until K candidates
exist; candidates are ranked by the mean absolute edge-weight difference (Eq. 3) and the top K are returned.
Prior: the road network of a 5x5 to 30x30 km area; no start point, no heading.

Values the paper leaves open (fixed here; reports/icde/methods_repro.md): K-means with 2 clusters (initialised at the
smallest and largest gap), "speed = 0" as < 0.05 m/s, "pedal = min" as within 1 % of the pedal range above the trace
minimum, 10 Hz zero-order-hold resampling of the CAN streams (the paper logs OBD at 0.1 s), sigma = 0.05, 0.1, 0.2, 0.3,
0.5; Eq. 3 uses the true path's node count, replaced by the line graph's node count.
Variants: `start="known"` (adaptation for the benchmark protocols: the trace starts at p0, whose road segment is split at
p0 and searched in both directions) and `start="area"` (the paper's prior: every intersection of the 5 km square around p0
may start the path; the square's placement uses p0, stated as such). VF2 on a path pattern is a depth-first search over
simple paths; it is implemented directly with the same matching rule. Positions per second: the speed-integrated distance
placed along the matched segments (CAN only); before the first node (area variant) the output holds the first node,
after the last node it continues on the straightest way out. A trace without a match at the largest sigma is a failure.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .base import OfflineBaseline
from .macrotracking import _parts, junction_mask


class NoMatch(RuntimeError):
    pass


def kmeans_threshold(gaps, k=2, iters=50):
    """1-D K-means on time gaps; returns the largest gap of the cluster with the smallest centre (inf if < 2 gaps)."""
    g = np.sort(np.asarray(gaps, dtype=float))
    if len(g) < 2:
        return math.inf
    c = np.quantile(g, np.linspace(0.0, 1.0, k))
    for _ in range(iters):
        lab = np.argmin(np.abs(g[:, None] - c[None, :]), axis=1)
        c_new = np.array([g[lab == j].mean() if np.any(lab == j) else c[j] for j in range(k)])
        if np.allclose(c_new, c):
            break
        c = c_new
    lab = np.argmin(np.abs(g[:, None] - c[None, :]), axis=1)
    return float(g[lab == int(np.argmin(c))].max())


class _SegGraph:
    """Undirected intersection graph of the visible map: junction nodes (undirected degree != 2) joined by chains of
    graph edges; each segment keeps its directed edge chain in both directions for the output geometry."""

    def __init__(self, mg, centre_xy=None, half_side=None):
        g0, to_g, vis = _parts(mg)
        jm = junction_mask(g0)
        n_nodes = len(g0.node_xy)
        ok = np.ones(len(g0.u), dtype=bool) if vis is None else vis.copy()
        if centre_xy is not None:                                      # area variant: the square around the start
            cx, cy = to_g(*centre_xy)
            mid = 0.5 * (g0.node_xy[g0.u] + g0.node_xy[g0.v])
            ok &= (np.abs(mid[:, 0] - cx) <= half_side) & (np.abs(mid[:, 1] - cy) <= half_side)
        self.mg, self.g0 = mg, g0
        out = {}
        for e in np.flatnonzero(ok):
            out.setdefault(int(g0.u[e]), []).append(int(e))
        self.adj = {}                                                  # junction -> [(junction2, length, edge chain)]
        self.seg_len = []
        for a, es in out.items():
            if not jm[a]:
                continue
            for e in es:
                chain, L, cur, guard = [e], float(g0.length[e]), e, 0
                while not jm[g0.v[cur]] and guard < 5000:
                    nxt = [x for x in out.get(int(g0.v[cur]), []) if g0.v[x] != g0.u[cur]]
                    if len(nxt) != 1:
                        break
                    cur = nxt[0]
                    chain.append(cur)
                    L += float(g0.length[cur])
                    guard += 1
                b = int(g0.v[cur])
                if b == a:
                    continue
                self.adj.setdefault(a, []).append((b, L, chain))
                self.seg_len.append(L)
        self.min_seg = float(min(self.seg_len)) if self.seg_len else 0.0


class CANTrace(OfflineBaseline):
    needs = ("speed", "gas")

    def __init__(self, start="known", K=10, sigmas=(0.05, 0.1, 0.2, 0.3, 0.5), area_half_m=2500.0, k_clusters=2,
                 v_stop=0.05, pedal_tol=0.01, max_expand=200000, timeout_s=120.0):
        self.start, self.K, self.sigmas, self.half = start, K, sigmas, area_half_m
        self.kc, self.v_stop, self.pedal_tol, self.max_expand, self.timeout_s = k_clusters, v_stop, pedal_tol, max_expand, timeout_s
        self.name = "can_trace" + ("_area" if start == "area" else "")
        self.stats = {}

    # ---- Algorithm 1
    def _line_graph(self, inputs, t_query):
        ev = inputs.events
        if "gas" not in ev.cols or not np.isfinite(ev.cols["gas"]).any():
            raise NoMatch("no accelerator pedal signal on this drive")
        t0, t1 = float(inputs.init.t0), float(t_query[-1])
        tg = np.arange(t0, t1 + 1e-9, 0.1)
        idx = np.searchsorted(ev.t_s, tg, side="right") - 1
        valid = idx >= 0
        v = np.where(valid, ev.cols["speed"][np.maximum(idx, 0)], 0.0)
        p = np.where(valid, ev.cols["gas"][np.maximum(idx, 0)], np.nan)
        v = np.where(np.isfinite(v), v, 0.0)
        pf = p[np.isfinite(p)]
        if not len(pf):
            raise NoMatch("no accelerator pedal samples in the episode")
        pmin, prng = float(pf.min()), float(pf.max() - pf.min())
        stop = v < self.v_stop
        rel = np.isfinite(p) & (p <= pmin + self.pedal_tol * max(prng, 1e-9))
        nodes = []
        for mask in (stop, rel):
            ts = tg[mask]
            if len(ts) < 2:
                continue
            thr = kmeans_threshold(np.diff(ts), self.kc)
            last = ts[0]
            for t in ts[1:]:
                if t - last >= thr:
                    nodes.append(float(t))
                last = t
        nodes = sorted(set(nodes))
        s_cum = np.concatenate([[0.0], np.cumsum(v[:-1] * 0.1)])        # rectangle rule on the 10 Hz grid
        self._tg, self._s = tg, s_cum
        return nodes

    def _dist(self, ta, tb):
        return float(np.interp(tb, self._tg, self._s) - np.interp(ta, self._tg, self._s))

    # ---- Algorithm 2
    def reconstruct(self, inputs, t_query):
        mg = inputs.map_graph
        if mg is None:
            raise ValueError("CAN-Trace needs a map graph")
        t_query = np.asarray(t_query, dtype=float)
        nodes = self._line_graph(inputs, t_query)
        init = inputs.init
        sg = _SegGraph(mg, centre_xy=init.p0_xy if self.start == "area" else None, half_side=self.half if self.start == "area" else None)
        merged = []                                                       # delete a node when the next edge is shorter than the shortest segment
        for k, t in enumerate(nodes):
            if k + 1 < len(nodes) and self._dist(t, nodes[k + 1]) < sg.min_seg:
                continue
            merged.append(t)
        times = merged
        t_start = time.time()
        starts = self._starts(sg, mg, init) if self.start == "known" else [(a, 0.0, None) for a in sg.adj]
        w_trace = []
        if self.start == "known":
            times = [float(init.t0)] + [t for t in times if t > init.t0]
        elif len(times) < 2:                                              # no line-graph edge: nothing to match in the area
            raise NoMatch(f"fewer than 2 trace nodes ({len(times)})")
        w_trace = [self._dist(times[k], times[k + 1]) for k in range(len(times) - 1)]
        cands = []
        for sig in self.sigmas:
            cands = self._match(sg, starts, w_trace, sig, t_start)
            if len(cands) >= self.K:
                break
        self.stats = dict(n_nodes=len(times), sigma=sig, n_cands=len(cands), secs=round(time.time() - t_start, 2))
        if not cands:
            raise NoMatch(f"no match at sigma <= {self.sigmas[-1]} ({len(times)} trace nodes)")
        cands.sort(key=lambda c: c[0])
        best = cands[:self.K]
        paths = [self._positions(mg, sg, c, times, t_query, init) for c in best]
        th = np.array([c[0] for c in best])
        pr = np.exp(-(th - th.min()) / 20.0)
        pr /= pr.sum()
        xy = paths[0]
        if len(paths) > 1:
            return xy, None, np.stack(paths), pr
        return xy, None, None, None

    def _starts(self, sg, mg, init):
        """Known start: p0 projected on the nearest road edges; each direction to the next junction is a first segment."""
        out = []
        radius, cands = 50.0, []
        while not cands and radius <= 400:
            cands = mg.nearest_edges(*init.p0_xy, radius)
            radius *= 2
        g0 = sg.g0
        jm = junction_mask(g0)
        for e in cands[:12]:
            o, _ = mg.project(e, *init.p0_xy)
            chain, L, cur, guard = [int(e)], float(g0.length[e]) - o, int(e), 0
            while not jm[g0.v[cur]] and guard < 5000:
                nxt = [s[0] for s in mg.succ(cur) if not s[2]]
                if len(nxt) != 1:
                    break
                cur = nxt[0]
                chain.append(cur)
                L += float(g0.length[cur])
                guard += 1
            out.append((int(g0.v[cur]), L, (chain, o)))
        return out

    def _match(self, sg, starts, w_trace, sig, t_start):
        """Depth-first VF2 on a path pattern: simple paths whose k-th segment matches the k-th trace edge."""
        res, n_exp = [], 0
        Q = len(w_trace)
        for s in starts:
            if self.start == "known":
                node, L0, first = s
                if Q == 0:
                    res.append((0.0, [node], [first]))
                    continue
                if abs(w_trace[0] - L0) > sig * max(L0, 1.0):
                    continue
                stack = [(node, 1, [node], [first], abs(w_trace[0] - L0))]
            else:
                stack = [(s[0], 0, [s[0]], [], 0.0)]
            while stack:
                n_exp += 1
                if n_exp > self.max_expand or time.time() - t_start > self.timeout_s:
                    return res
                a, k, nodes_, segs, err = stack.pop()
                if k == Q:
                    res.append((err / max(Q, 1), nodes_, segs))
                    continue
                for b, L, chain in sg.adj.get(a, []):
                    if b in nodes_:
                        continue
                    d = abs(w_trace[k] - L)
                    if d <= sig * L:
                        stack.append((b, k + 1, nodes_ + [b], segs + [(chain, 0.0)], err + d))
        return res

    def _positions(self, mg, sg, cand, times, t_query, init):
        """Place the speed-integrated distance along the matched segments (scaled to each segment's length)."""
        _, nodes_, segs = cand
        g0 = sg.g0
        edges, offs = [], []                                              # piecewise path: list of (edge, start offset)
        for chain, o in segs:
            for k, e in enumerate(chain):
                edges.append(e)
                offs.append(o if k == 0 else 0.0)
        # tail: straightest continuation for 5 km
        cur = edges[-1] if edges else None
        guard = 0
        while cur is not None and guard < 400:
            outs = [s for s in mg.succ(cur) if not s[2]]
            if not outs:
                break
            cur = int(min(outs, key=lambda s: abs(s[1]))[0])
            edges.append(cur)
            offs.append(0.0)
            guard += 1
        seg_len = np.array([float(g0.length[e]) - o for e, o in zip(edges, offs)])
        cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        # distance along the path at each query: node k of the trace sits at the end of segment k
        knots_t = [times[0]]
        knots_s = [0.0]
        acc = 0.0
        for k, (chain, o) in enumerate(segs):
            acc += sum(float(g0.length[e]) for e in chain) - o
            if k + 1 < len(times):
                knots_t.append(times[k + 1])
                knots_s.append(acc)
        s_q = np.interp(t_query, knots_t, knots_s) if len(knots_t) > 1 else np.zeros(len(t_query))
        after = t_query > knots_t[-1]                                      # beyond the last node: speed-integrated distance
        s_q[after] = knots_s[-1] + np.array([self._dist(knots_t[-1], t) for t in t_query[after]])
        if self.start == "area":
            s_q[t_query < knots_t[0]] = 0.0
        s_q = np.clip(s_q, 0.0, cum[-1])
        j = np.clip(np.searchsorted(cum, s_q, side="right") - 1, 0, len(edges) - 1)
        es = np.array(edges, dtype=np.int64)[j]
        oo = np.array(offs)[j] + (s_q - cum[j])
        return mg.points(es, np.minimum(oo, g0.length[es]))
