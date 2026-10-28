"""Elastic Pathing re-implemented from Gao, Firner, Sugrim, Kaiser-Pendergrast, Yang, Lindqvist, "Transforming Speed
Sequences into Road Rays on the Map with Elastic Pathing", arXiv 1710.06932 (2017; states that it supersedes the
UbiComp 2014 paper and the 2013 report). No author code or data is available.

Paper algorithm (sec. 3.2 to 3.3, Algorithms 1 and 2): 1 Hz speed samples and a known start; best-first search over
partial paths sorted by error (accumulated stretching / compressing of the speed-integrated distance against the road);
the search stops once a complete path exists and the best partial error is not below delta = 1.1 times the best complete
error; the first complete path is the prediction, the complete paths within delta are the top-n. gotoBranch advances a path
sample by sample: a path faster than the way's speed limit + 20 mph is dropped; at a road node every outgoing edge is
tried: a turn is feasible if the current speed is at most the safe speed of the turn, v = sqrt(15 r (0.01 e + f)) mph
with e = 8 % superelevation and r (ft) from Eq. 3, r = (x + y + sqrt(2xy(1 + cos a))) / (1 - cos a) for road widths
x, y and deflection a (feasible -> pin the landmark and continue; infeasible -> compressA, rewind the speed cursor to the
latest slow-enough sample, and stretchA, advance it to the next one, the moved distance added to the error); a stop
(speed ~ 0) not at an intersection is snapped back (compressB) or forward (stretchB) to an intersection, again adding
the moved distance. The start heading is not used.

Values the paper leaves open (fixed here, the free ones chosen on validation drives only; reports/icde/methods_repro.md):
friction f (default 0.3), lane width 12 ft, lanes from OSM `lanes` (missing: 1 per direction), "~0" = speed < 0.5 m/s,
compressA / stretchA search at most `max_shift` samples and never before the last landmark, intersection tolerance
lanes x lane width / 2 + 8 m (room for a car in front), untagged speed limits are not checked, U-turns only as the
single way out of a dead end are not taken (the path is dropped), error = sum of moved distances (m).
Adaptations: the start is the projection of p0 on the nearest road edges (all directions: native heading-unknown
protocol; `heading=True` keeps edges within 45 deg of theta0, labelled as an adaptation); positions per second come
from the path and the speed-integrated distance with the same landmarks (CAN only, no reference alignment); a
state-dominance prune (same edge, sample and 10 m offset bin with no lower error) bounds the search, and a budget of
`max_expand` branch calls / `timeout_s` seconds ends it: an episode without a complete path is a failure (no output).
"""
from __future__ import annotations

import heapq
import math
import time

import numpy as np

from .base import OfflineBaseline
from .macrotracking import junction_mask, _parts

MPH = 0.44704
LANE_M = 3.6576


class NoSolution(RuntimeError):
    pass


def safe_turn_speed(alpha, x, y, friction, elev_pct=8.0):
    """Eq. 1 with Eq. 3: largest speed (m/s) for a turn with deflection alpha (rad) between road widths x, y (m)."""
    ca = math.cos(alpha)
    if ca >= 1.0 - 1e-9:
        return math.inf
    r = (x + y + math.sqrt(max(2.0 * x * y * (1.0 + ca), 0.0))) / (1.0 - ca)
    return math.sqrt(15.0 * (r / 0.3048) * (0.01 * elev_pct + friction)) * MPH


class _P:
    __slots__ = ("parent", "i0", "pos", "e", "o", "i", "err", "pin", "jn", "jn_d", "carry")

    def __init__(self, parent, i0, e, o, i, err, pin, jn, jn_d, carry=-1.0):
        self.parent, self.i0, self.pos = parent, i0, []
        self.e, self.o, self.i, self.err, self.pin, self.jn, self.jn_d = e, o, i, err, pin, jn, jn_d
        self.carry = carry                  # >= 0: distance of sample i still to be driven (the sample crossed a node)


class ElasticPathing(OfflineBaseline):
    needs = ("speed",)

    def __init__(self, heading=False, delta=1.1, friction=0.3, v_zero=0.5, max_shift=10, K=10, max_expand=None,
                 timeout_s=None, start_radius_m=50.0, prune_states=True, tie_seed=None):
        self.heading, self.delta, self.f, self.v_zero, self.max_shift = heading, delta, friction, v_zero, max_shift
        self.K, self.max_expand, self.timeout_s, self.start_radius, self.prune = K, max_expand, timeout_s, start_radius_m, prune_states
        self.tie_seed = tie_seed            # None: successors in edge-id order; an int shuffles them (tie-order sensitivity, val only)
        self._rng = None
        self.name = "elastic_pathing" + ("_hp" if heading else "")
        self.stats = {}

    # ---- road helpers
    def _width(self, e):
        ln = self.g0.lanes[e]
        if not np.isfinite(ln):
            ln = 1.0 if self.g0.oneway[e] else 2.0
        return float(ln) * LANE_M

    def _tol(self, e):
        return 0.5 * self._width(e) + 8.0

    def _outs(self, e):
        """Non-U-turn successors (edge, deflection rad)."""
        outs = [(s[0], abs(s[1])) for s in self.g.succ(e) if not s[2]]
        if self._rng is not None and len(outs) > 1:
            outs = [outs[i] for i in self._rng.permutation(len(outs))]
        return outs

    def _is_junction_head(self, e):
        return bool(self.jm[self.g0.v[e]])

    def _next_junction(self, e, o, cap=500.0):
        """Follow the single-successor chain from (e, o) to the next node with a choice; (edge, distance) or None."""
        d = float(self.g.length[e]) - o
        guard = 0
        while d <= cap and guard < 200:
            outs = self._outs(e)
            if len(outs) != 1 or self._is_junction_head(e):
                return e, d
            e = outs[0][0]
            d += float(self.g.length[e])
            guard += 1
        return None

    # ---- the branch function (Algorithm 2)
    def _goto_branch(self, P):
        S, DT, n, g = self.S, self.DT, self.n, self.g
        while P.i < n:
            i = P.i
            v = S[i]
            if P.carry < 0:
                if v < self.v_zero:                                   # match zero speeds to intersections
                    while P.i < n and S[P.i] < self.v_zero:
                        P.pos.append((P.e, P.o))
                        P.i += 1
                    at = (float(g.length[P.e]) - P.o <= self._tol(P.e) and self._is_junction_head(P.e)) or P.jn_d <= self._tol(P.e)
                    if at or P.i >= n:
                        P.pin = P.i
                        return [P]
                    kids = []
                    if P.jn is not None:                               # compressB: back to the last intersection (or the start)
                        kids.append(_P(P, P.i, P.jn[0], P.jn[1], P.i, P.err + P.jn_d, P.i, P.jn, 0.0))
                    nj = self._next_junction(P.e, P.o)
                    if nj is not None:                                 # stretchB: forward to the next intersection
                        e2, d2 = nj
                        L2 = float(g.length[e2])
                        kids.append(_P(P, P.i, e2, L2, P.i, P.err + d2, P.i, (e2, L2), 0.0))
                    return kids
                ms = self.g0.maxspeed[P.e]
                if np.isfinite(ms) and v > ms + 20.0 * MPH:            # speed limit + 20 mph
                    return []
                d = v * DT[i]
            else:
                d, P.carry = P.carry, -1.0
            L = float(g.length[P.e])
            if P.o + d <= L:
                P.o += d
                P.jn_d += d
                P.pos.append((P.e, P.o))
                P.i += 1
                continue
            # the path reaches the head node during sample i: every way out is a candidate (Algorithm 2, lines 3-17)
            rem = P.o + d - L
            P.jn_d += L - P.o
            P.o = L
            outs = self._outs(P.e)
            if not outs:
                return []
            x = self._width(P.e)
            if len(outs) == 1 and safe_turn_speed(outs[0][1], x, self._width(outs[0][0]), self.f) >= v:
                e2 = outs[0][0]                                       # plain continuation: nothing to pin
                if self._is_junction_head(P.e):
                    P.jn, P.jn_d = (e2, 0.0), 0.0
                P.e, P.o, P.carry = e2, 0.0, rem
                continue
            kids = []
            for e2, a in outs:
                vmax = safe_turn_speed(a, x, self._width(e2), self.f)
                if v <= vmax:                                         # feasible: pin the landmark and branch
                    kids.append(_P(P, i, e2, 0.0, i, P.err, i, (e2, 0.0), 0.0, carry=rem))
                    continue
                acc = S[i] * DT[i] - rem                              # compressA: rewind to the latest slow-enough sample
                for j in range(i - 1, max(P.pin, i - self.max_shift) - 1, -1):
                    acc += S[j] * DT[j]
                    if S[j] <= vmax:
                        kids.append(_P(P, j, e2, 0.0, j, P.err + acc, j, (e2, 0.0), 0.0))
                        break
                acc = rem                                             # stretchA: wait before the node until slow enough
                for j in range(i + 1, min(n, i + 1 + self.max_shift)):
                    if S[j] <= vmax:
                        c = _P(P, i, e2, 0.0, j, P.err + acc, j, (e2, 0.0), 0.0)
                        c.pos.extend([(P.e, L)] * (j - i))
                        kids.append(c)
                        break
                    acc += S[j] * DT[j]
            return kids
        return [P]

    def _positions(self, P):
        segs = []
        while P is not None:
            segs.append(P)
            P = P.parent
        pos = [None] * self.n
        for s in reversed(segs):                                     # children overwrite rewound samples of their parent
            for k, pe in enumerate(s.pos):
                if s.i0 + k < self.n:
                    pos[s.i0 + k] = pe
        last = None
        for k in range(self.n):
            if pos[k] is None:
                pos[k] = last
            last = pos[k]
        return pos

    # ---- Algorithm 1
    def reconstruct(self, inputs, t_query):
        mg = inputs.map_graph
        if mg is None:
            raise ValueError("Elastic Pathing needs a map graph")
        init = inputs.init
        if self.heading and init.theta0 is None:
            raise ValueError("the heading-known variant needs theta0")
        self.g = mg
        self._rng = np.random.default_rng(self.tie_seed) if self.tie_seed is not None else None
        self.g0, _, _ = _parts(mg)
        self.jm = junction_mask(self.g0)
        t_query = np.asarray(t_query, dtype=float)
        ev = inputs.events
        tt, vv = ev.t_s, ev.cols["speed"]
        edges_t = np.concatenate([[init.t0], t_query])
        S = np.zeros(len(t_query))
        last = 0.0
        for k in range(len(t_query)):                                 # 1 Hz samples: mean CAN speed per interval
            a, b = np.searchsorted(tt, edges_t[k], side="right"), np.searchsorted(tt, edges_t[k + 1], side="right")
            w = vv[a:b]
            w = w[np.isfinite(w)]
            if len(w):
                last = float(w.mean())
            S[k] = last
        self.S, self.DT, self.n = S, np.diff(edges_t), len(t_query)
        # budget scaled with the episode length unless given: 60 s and 21k branch calls for 5 min, at most 300 s / 200k
        self._timeout = self.timeout_s if self.timeout_s is not None else float(np.clip(0.2 * self.n, 30.0, 300.0))
        self._max_exp = self.max_expand if self.max_expand is not None else int(np.clip(70 * self.n, 20000, 200000))
        radius, cands = self.start_radius, []
        while not cands and radius <= 400:
            cands = mg.nearest_edges(*init.p0_xy, radius)
            radius *= 2
        if self.heading and cands:
            cands = [e for e in cands if abs((mg.bearing[e] - init.theta0 + math.pi) % (2 * math.pi) - math.pi) < math.radians(45)] or cands
        if not cands:
            raise NoSolution("no road within 400 m of the start")
        heap, cnt = [], 0
        for e in cands:
            o, _ = mg.project(e, *init.p0_xy)
            P = _P(None, 0, int(e), float(o), 0, 0.0, 0, (int(e), float(o)), 0.0)
            heapq.heappush(heap, (0.0, 0, cnt, P))
            cnt += 1
        complete, seen = [], {}
        t_start, n_exp = time.time(), 0
        while heap:
            if complete and not (heap[0][0] < self.delta * complete[0].err):
                break
            if n_exp >= self._max_exp or time.time() - t_start > self._timeout:
                break
            _, _, _, P = heapq.heappop(heap)
            n_exp += 1
            for c in self._goto_branch(P):
                if c.i >= self.n:
                    complete.append(c)
                    complete.sort(key=lambda z: z.err)
                    continue
                if self.prune:
                    key = (c.e, c.i, int(c.o // 10.0))
                    if seen.get(key, math.inf) <= c.err:
                        continue
                    seen[key] = c.err
                heapq.heappush(heap, (c.err, -c.i, cnt, c))
                cnt += 1
        self.stats = dict(expansions=n_exp, n_complete=len(complete), secs=round(time.time() - t_start, 2), budget_hit=not complete)
        if not complete:
            why = "search exhausted: every candidate path was dropped" if not heap else "budget reached"
            raise NoSolution(f"no complete path ({why}; {n_exp} branch calls, {time.time() - t_start:.0f} s)")
        best = complete[:self.K]
        paths = []
        for P in best:
            pos = self._positions(P)
            es = np.array([p[0] for p in pos], dtype=np.int64)
            os_ = np.array([p[1] for p in pos], dtype=float)
            paths.append(mg.points(es, os_))
        errs = np.array([P.err for P in best])
        pr = np.exp(-(errs - errs.min()) / 100.0)
        pr /= pr.sum()
        xy = paths[0]
        theta = mg.bearing[np.array([p[0] for p in self._positions(best[0])], dtype=np.int64)]
        if len(paths) > 1:
            return xy, theta, np.stack(paths), pr
        return xy, theta, None, None
