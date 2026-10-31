"""Dewri et al. re-implemented from Dewri, Annadata, Eltarjaman, Thurimella, "Inferring Trip Destinations From Driving
Habits Data", ACM WPES 2013, pp. 267-272, read in its 9-page extended version (related_work/2013WPES-Extended.pdf; it
prints no journal page numbers, so "ext. p.N" is PDF page N). No author code is available.

Paper algorithm. 1 Hz samples <t, s, d> of time, speed and distance travelled (ext. p.3). Pre-processing (Sec. 3.2,
ext. p.3-4): every sample between two zero-speed points less than half a mile apart is deleted (stop-and-go traffic),
then the zero-speed runs spanning less than 3 s; the distances of the remaining zero-speed points are the stop-points,
the last one (the parking stop) is never matched. Candidates (Sec. 4.2, ext. p.4-5): a depth-first search from the known
start node tracks the path length L and keeps a path only while (1) every stop-point it has passed has an intersection
node (3-way or more) within 500 ft; (2) the path to every node is a shortest path within 0.1 mile, the reference node
moving along highway nodes so that the check is off on highways and restarts from the exit (Fig. 3); (3) a heading
change above 60 deg (a turn) happens only where the trip speed is under 25 mph, read at the sample whose distance is
closest to L; (4) once L exceeds the trip length the path ends: it is a candidate if every stop-point is matched, and per
end node the candidate with the fewest turns is kept. Ranking (Sec. 4.3, ext. p.5-6): an ideal model gives the road-type
speed limit along the path; the augmented model M_aug slows to 15 mph (left) / 10 mph (right) at every turn node and to
0 at every stop-point with a 25 ft/s^2 deceleration and a 6.5 ft/s^2 acceleration; a sample scores the log of the
integral of f over [s - eps, s + eps], f = N(1.1 s', s'/30) for s' >= 20 mph and N(s', 1 mph) below, eps = 1e-5,
s' = M_aug at the sample's distance; candidates are ranked by the sum. Constants stay in the paper's units (mph, ft,
miles) and are converted at use.

Values the paper leaves open (fixed here; reports/icde/methods_repro.md): zero speed = 1 Hz mean below 0.1 mph (a zero
sample needs a whole second at standstill, so the 3 s span of four zero samples takes about 4 to 4.5 s of standstill);
the half-mile rule compares consecutive zero-speed runs A, B (d at the first sample of B minus d at the last of A; chains
merge; ext. p.7: "more than one stop within a small distance"); a run spans t_last - t_first; one stop-point per run (its
mean distance); the pre-processed samples feed both the turn look-up and the likelihood; intersection = at least 3
distinct visible undirected neighbours; turn angle at a node = the heading change between the nearest path edges at
least `short_m` (1 m) long before and after it, left positive (US traffic: left 15, right 10 mph), in the search and in
M_aug alike: a shorter edge (two OSM nodes at one position, sub-metre jitter) has no heading of its own and carries the
incoming one (the paper's angle is undefined for a zero-length edge, whose bearing would read east and fake two turns);
simple paths (no node revisited, no U-turn), depth first with successors in succ() (edge-id) order; a stop-point s_k is
matched at an intersection x with |L(x) - s_k| <= 500 ft and the path is dropped once an unmatched s_k + 500 ft < L (the
check waits for the slack window); a path ends at its first node beyond the trip length (literal "exceeds"); that end
node counts for matching, only the stop-points whose slack window has closed there (s_k + 500 ft < L) must be matched (a
halt just before the window end whose intersection lies beyond the end node stays unmatched, like the final stop; the
paper never meets this case: its matched stop-points lie more than half a mile before the trip end, because the
half-mile rule merges any later stop into the parking stop), and the shortest-path check holds there as at every node;
highway = {motorway, motorway_link}: a node reached on such an edge becomes the reference; shortest paths by Dijkstra
(scipy) on the visible edges within D + 1 km of p0, the shortest of parallel edges, turn restrictions ignored; equal
turn counts keep the first path found in depth-first order, and equal log-likelihoods keep that order in the ranking (a
halt at every turn hides the turn direction); speed table (the 25/35/45/60 mph plateaus of Figs. 5 and 7): motorway 60,
trunk and primary 45, secondary, tertiary and *_link 35, residential, unclassified and road 25, living_street 15 mph,
OSM maxspeed ignored as in the paper (limits by road type, ext. p.4); M is a step function and M_aug the minimum of M
and the constant-rate envelopes sqrt(v^2 + 2 a |x - d|) around turn nodes and every stop-point; the eps-integral is
taken to first order, log(2 eps) + log f(s), never as a CDF difference (it underflows to 0).
Adaptations: <t, s, d> are one sample per second of CAN, at t0 + k up to the window end t_end (k >= 1; the paper's
logger records 1 Hz, ext. p.3), independent of where reference fixes exist (they can be 30 s apart): s = mean CAN speed
over the second, d = CAN distance from t0 (trapezoid rule on the raw events), and the trip length D = d at the last
sample (at the last output time if that is later); t_query gives the output times only (and the window end when t_end
is not given); the start is p0 projected on the nearest road edges (the first 12 of at least `short_m` within 50, 100,
200 or 400 m, both directions; the paper starts at a node reached by trimming the trip with GPS, ext. p.5), the path
from p0 to the head node of its start edge is fixed and that node is the first shortest-path reference; `heading=True`
keeps the start edges within 45 deg of theta0 (F-M-Hp); our windows end while driving, so the last stop-point stays
unmatched only when its zero run reaches the window end; the paper's search is exhaustive, ours stops after `max_expand`
node expansions (default clip(70 n, 2e4, 2e5), n = len(t_query)) or `timeout_s` seconds (a safety net) in all, shared by
the start edges: their depth-first searches run one after the other, each on an equal share of the expansions and
seconds left (a share left unused passes to the following ones), so that a binding budget cannot starve a start
direction (both directions of a two-way start road are exactly as close to p0 and come in KD-tree order), though a
route that needs more than its share is lost; the paths already accepted when a share runs out are kept, and an episode
without any candidate is a failure (no output); the paper ranks destinations, so positions per second are the CAN
distance laid along each of the top 10 paths from p0's projection, probabilities = normalised exp(log-likelihood - max).
"""
from __future__ import annotations

import math
import time
from bisect import bisect_left, bisect_right
from collections import OrderedDict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .base import OfflineBaseline
from .macrotracking import _parts

MPH = 0.44704                           # m/s per mph
FT = 0.3048                             # m per foot
MILE = 1609.344                         # m per mile
LEFT_MPH, RIGHT_MPH = 15.0, 10.0        # turning speeds (ext. p.5)
DECEL_FT, ACCEL_FT = 25.0, 6.5          # ft/s^2 (ext. p.6)
EPS = 1e-5                              # half-width of the speed integral, mph (ext. p.6)
HIGHWAY = ("motorway", "motorway_link")
LIMIT_MPH = {"motorway": 60.0, "trunk": 45.0, "primary": 45.0, "secondary": 35.0, "tertiary": 35.0, "residential": 25.0,
             "unclassified": 25.0, "living_street": 15.0, "road": 25.0}          # *_link: 35 mph


class DewriNoPath(RuntimeError):
    pass


def speed_limit_mph(highway):
    """Road-type speed limit (mph); OSM maxspeed is not used (ext. p.4)."""
    return 35.0 if highway.endswith("_link") else LIMIT_MPH.get(highway, 25.0)


def one_hz(tt, vv, t0, tg, t_out):
    """(speed m/s per sample time tg, CAN distance m at tg, CAN distance m at t_out). Speed = mean of the finite CAN
    samples in (b_k, b_k+1] with b = [t0] + tg (an empty interval holds the previous mean, leading ones the first);
    distance = trapezoid rule on the raw events (NaN speeds hold the last finite value), zero at t0, held speeds beyond
    the first / last event, the same odometer for the samples and for the output times."""
    fin = np.isfinite(vv)
    idx = np.searchsorted(tt, np.concatenate([[t0], tg]), side="right")
    tot = np.concatenate([[0.0], np.cumsum(np.where(fin, vv, 0.0))])[idx]
    cnt = np.diff(np.concatenate([[0], np.cumsum(fin)])[idx])
    if not cnt.any():
        raise DewriNoPath("no speed samples in the episode")
    k = np.maximum.accumulate(np.where(cnt > 0, np.arange(len(cnt)), -1))
    k[k < 0] = np.argmax(cnt > 0)
    S = np.diff(tot)[k] / cnt[k]
    i = np.maximum.accumulate(np.where(fin, np.arange(len(vv)), -1))
    v = vv[np.where(i < 0, np.argmax(fin), i)]
    s_ev = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tt))])

    def odo(t):
        t = np.asarray(t, dtype=float)
        out = np.where(t < tt[0], (t - tt[0]) * v[0], np.interp(t, tt, s_ev))
        return np.where(t > tt[-1], s_ev[-1] + (t - tt[-1]) * v[-1], out)
    return S, odo(tg) - odo(t0), odo(t_out) - odo(t0)


def preprocess(t, s_mph, d, zero_mph=0.1, traffic_mi=0.5, stop_min_s=3.0):
    """Sec. 3.2 on the 1 Hz samples: (kept-sample mask, stop-point distances (m), number of stop-points to match).
    Step 1 deletes everything from the first sample of a zero-speed run to the last sample of the next run when they are
    less than `traffic_mi` apart (chains merge); step 2 deletes the remaining zero runs spanning less than `stop_min_s`.
    One stop-point per surviving run; the last one is left unmatched only when its run reaches the window end."""
    z = np.asarray(s_mph) < zero_mph
    edge = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    a, b = np.flatnonzero(edge == 1), np.flatnonzero(edge == -1) - 1         # zero runs [a_k, b_k]
    keep = np.ones(len(z), dtype=bool)
    alive = np.ones(len(a), dtype=bool)
    for k in range(len(a) - 1):                                                # step 1: stop-and-go traffic
        if d[a[k + 1]] - d[b[k]] < traffic_mi * MILE:
            keep[a[k]:b[k + 1] + 1] = False
            alive[k] = alive[k + 1] = False
    for k in np.flatnonzero(alive):                                            # step 2: halts shorter than 3 s
        if t[b[k]] - t[a[k]] < stop_min_s:
            keep[a[k]:b[k] + 1] = False
            alive[k] = False
    runs = np.flatnonzero(alive)
    stops = np.array([float(np.mean(d[a[k]:b[k] + 1])) for k in runs])
    n_req = len(runs) - 1 if len(runs) and b[runs[-1]] == len(z) - 1 else len(runs)
    return keep, stops, n_req


def aug_model(d, cum, lim_mph, x_turn, v_turn_mph, x_stop):
    """M_aug (ext. p.5-6) in mph at path distances d (m): the speed limit of the edge holding d (step function; cum =
    the edges' start distances plus the path end, lim_mph per edge), capped by the envelopes sqrt(v^2 + 2 a |x - d|) of
    every turn node x (v = 15 mph left, 10 mph right) and stop-point (v = 0): 25 ft/s^2 before x, 6.5 ft/s^2 after."""
    d = np.asarray(d, dtype=float)
    m = np.asarray(lim_mph, dtype=float)[np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(lim_mph) - 1)]
    x = np.concatenate([np.asarray(x_turn, dtype=float), np.asarray(x_stop, dtype=float)]) / FT
    if not len(x):
        return m
    v2 = (np.concatenate([np.asarray(v_turn_mph, dtype=float), np.zeros(len(x_stop))]) * MPH / FT) ** 2
    o = np.argsort(x, kind="stable")
    x, v2, df = x[o], v2[o], d / FT
    # min over nodes x >= d of v^2 + 2 a_dec (x - d) and over x <= d of v^2 + 2 a_acc (d - x): suffix / prefix minima
    suf = np.minimum.accumulate((v2 + 2.0 * DECEL_FT * x)[::-1])[::-1]
    pre = np.minimum.accumulate(v2 - 2.0 * ACCEL_FT * x)
    i, j = np.searchsorted(x, df, side="left"), np.searchsorted(x, df, side="right") - 1
    before = np.where(i < len(x), suf[np.minimum(i, len(x) - 1)] - 2.0 * DECEL_FT * df, np.inf)
    after = np.where(j >= 0, pre[np.maximum(j, 0)] + 2.0 * ACCEL_FT * df, np.inf)
    return np.minimum(m, np.sqrt(np.minimum(before, after)) * FT / MPH)


def log_likelihood(s, s_model):
    """Summed log-probability of the observed speeds s given the model speeds s' (mph; ext. p.6): the integral of f over
    [s - eps, s + eps] to first order, log(2 eps) + log f(s), f = N(1.1 s', s'/30) for s' >= 20 mph, N(s', 1) below."""
    hi = s_model >= 20.0
    mu = np.where(hi, 1.1 * s_model, s_model)
    sd = np.where(hi, s_model / 30.0, 1.0)
    z = (s - mu) / sd
    return float(np.sum(math.log(2.0 * EPS) - 0.5 * z * z - np.log(sd) - 0.5 * math.log(2.0 * math.pi)))


class _ShortestPaths:
    """Bounded Dijkstra (scipy csgraph) on the visible edges with both ends near the start, the shortest of parallel
    edges, turn restrictions ignored. Distances are cached per source node with the largest limit asked so far (the
    least recently used are dropped beyond about 50 MB)."""

    def __init__(self, g0, near, vis):
        ok = near[g0.u] & near[g0.v]
        if vis is not None:
            ok &= vis
        nodes = np.flatnonzero(near)
        self.loc = np.full(len(g0.node_xy), -1, dtype=np.int64)
        self.loc[nodes] = np.arange(len(nodes))
        a, b, w = self.loc[g0.u[ok]], self.loc[g0.v[ok]], np.maximum(g0.length[ok], 1e-3)
        key = a * len(nodes) + b
        o = np.lexsort((w, key))                                               # parallel (u, v): keep the shortest
        o = o[np.concatenate([[True], key[o][1:] != key[o][:-1]])] if len(o) else o
        self.csr = csr_matrix((w[o], (a[o], b[o])), shape=(len(nodes), len(nodes)))
        self.cache = OrderedDict()
        self.cap = max(4, int(5e7 // (8 * max(len(nodes), 1))))
        self.runs = 0

    def dist(self, r, x, limit):
        """Shortest path length r -> x (m), inf beyond `limit` (a path of length l from r fails the 0.1 mile check only if
        the shortest one is under l - slack, so that limit is enough)."""
        hit = self.cache.get(r)
        if hit is None or hit[0] < limit:
            hit = (limit, dijkstra(self.csr, directed=True, indices=int(self.loc[r]), limit=limit))
            self.cache[r] = hit
            self.runs += 1
        self.cache.move_to_end(r)
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        i = self.loc[x]
        return float(hit[1][i]) if i >= 0 else math.inf


class Dewri(OfflineBaseline):
    needs = ("speed",)
    K = 10                                  # candidates returned (the paper ranks all of them)

    def __init__(self, heading=False, traffic_mi=0.5, stop_min_s=3.0, stop_slack_ft=500.0, sp_slack_mi=0.1, turn_deg=60.0,
                 turn_mph=25.0, zero_mph=0.1, short_m=1.0, max_expand=None, timeout_s=None):
        self.heading, self.traffic_mi, self.stop_min_s, self.stop_slack_ft = heading, traffic_mi, stop_min_s, stop_slack_ft
        self.sp_slack_mi, self.turn_deg, self.turn_mph, self.zero_mph = sp_slack_mi, turn_deg, turn_mph, zero_mph
        self.short_m = short_m              # edges shorter than this (m) have no heading of their own (ours)
        self.max_expand, self.timeout_s = max_expand, timeout_s     # per episode, shared by the start edges (None: scaled)
        self.name = "dewri" + ("_hp" if heading else "")
        self.stats = {}

    # ---- helpers
    def _speed_at(self, L):
        """Trip speed (mph) at the kept sample whose distance is closest to L (ties to the earlier sample)."""
        dk = self.dk
        j = bisect_left(dk, L)
        if j == len(dk) or (j > 0 and L - dk[j - 1] <= dk[j] - L):
            j = bisect_left(dk, dk[j - 1])                                     # the first of equal distances (a halt)
        return self.sk[j]

    @staticmethod
    def _cum(g0, edges, o):
        """Path distance (m from p0) at the start of every edge, plus the path end."""
        seg = g0.length[edges].astype(float)
        seg[0] -= o
        return np.concatenate([[0.0], np.cumsum(seg)])

    def _score(self, g0, edges, o, d, s, stops):
        """Summed log-probability of the pre-processed samples under the path's augmented speed model."""
        cum = self._cum(g0, edges, o)
        k = np.flatnonzero(g0.length[edges] >= self.short_m)                  # edges with a heading of their own, as _dfs
        tr = (g0.bearing[edges[k[1:]]] - g0.bearing[edges[k[:-1]]] + math.pi) % (2 * math.pi) - math.pi
        turn = np.abs(tr) > math.radians(self.turn_deg)
        lim = [speed_limit_mph(h) for h in g0.highway[edges]]
        return log_likelihood(s, aug_model(d, cum, lim, cum[k[1:]][turn], np.where(tr[turn] > 0, LEFT_MPH, RIGHT_MPH), stops))

    # ---- candidate paths (Sec. 4.2)
    def _dfs(self, mg, g0, roots, D, req, inter, sp, t_start):
        """Iterative DFS over simple paths, one root (edge, offset) after the other on a shared budget: each root may use
        an equal share of the node expansions (`_max_exp`) and seconds (`_timeout`) left, so what it leaves unused passes
        to the roots after it. Returns {end node: (turns, edge ids, start offset)}, the expansions per root, the number
        of paths that reached the trip length, and the budget that first ended a root's search (None: every search
        exhausted)."""
        slack, sp_slack, turn_rad = self.stop_slack_ft * FT, self.sp_slack_mi * MILE, math.radians(self.turn_deg)
        best = {}

        def match(x, L, mask):                                                 # constraint 1: intersections near L
            if inter[x]:
                mask |= ((1 << bisect_right(req, L + slack)) - 1) ^ ((1 << bisect_left(req, L - slack)) - 1)
            return mask

        def passed(L, mask):                                                   # stop-points behind the slack window
            k = (1 << bisect_left(req, L - slack)) - 1
            return (mask & k) == k

        n_exp, n_end, why = [], 0, None
        for i, (e0, o0) in enumerate(roots):                                   # the path starts at p0 on e0
            share = (self._max_exp - sum(n_exp)) // (len(roots) - i)           # equal shares of the expansions and of the
            now = time.time() - t_start                                        # time left
            deadline = now + max(self._timeout - now, 0.0) / (len(roots) - i)
            x, L = int(g0.v[e0]), float(g0.length[e0]) - o0
            h = float(g0.bearing[e0]) if g0.length[e0] >= self.short_m else math.nan   # heading of the last edge >= short_m
            mask, stack = match(x, L, 0), []
            n_end += L > D                                                     # the trip may end on the start edge
            if passed(L, mask):                                                # (depth, node, edge in, L, ref, L_ref, matched, turns, heading)
                stack.append((0, x, e0, L, -1 if L > D else x, L, mask, 0, h))
            path_n, path_e, on, n, stop = [], [], set(), 0, None
            while stack:
                depth, c, e_in, L, r, Lr, mask, nt, h = stack.pop()
                while len(path_n) > depth:                                     # back to this node's ancestors
                    on.discard(path_n.pop())
                    path_e.pop()
                if r < 0:                                                      # a path accepted past the trip length (ref -1),
                    if c not in best or nt < best[c][0]:                       # kept in DFS order (ties keep the first path
                        best[c] = (nt, np.array(path_e + [e_in], dtype=np.int64), o0)   # found), budget spent or not
                    continue
                if stop is None and n >= share:
                    stop = "expansions"
                elif stop is None and time.time() - t_start >= deadline:
                    stop = "time"
                if stop is not None:                                           # share spent: only collect accepted paths
                    continue
                n += 1
                path_n.append(c)
                path_e.append(e_in)
                on.add(c)
                slow, kids = None, []
                for e, _, uturn in mg.succ(e_in):                              # visible, restrictions, U-turn flag (turn: below)
                    x = int(g0.v[e])
                    if uturn or x in on:                                       # simple paths only
                        continue
                    if g0.length[e] < self.short_m:                            # no heading of its own: carry h across it
                        hx, is_turn = h, False
                    else:                                                      # left positive, as succ(); no turn while h is NaN
                        hx = float(g0.bearing[e])
                        is_turn = bool(abs((hx - h + math.pi) % (2 * math.pi) - math.pi) > turn_rad)
                    if is_turn:                                                # constraint 3: under 25 mph at the closest sample
                        if slow is None:
                            slow = self._speed_at(L) < self.turn_mph
                        if not slow:
                            continue
                    Lx = L + float(g0.length[e])
                    m2, end = match(x, Lx, mask), Lx > D                       # constraint 4: past the trip length the path
                    n_end += end                                               # ends at x (ref -1), checked like any node
                    if not passed(Lx, m2):                                     # constraint 1: every stop-point behind the
                        continue                                               # slack window matched
                    if g0.highway[e] in HIGHWAY:                               # constraint 2 is off: the reference moves along
                        kids.append((depth + 1, x, e, Lx, -1 if end else x, Lx, m2, nt + is_turn, hx))
                    elif Lx - Lr <= sp.dist(r, x, max(D, Lx - sp_slack) - Lr) + sp_slack:   # constraint 2: shortest within
                        kids.append((depth + 1, x, e, Lx, -1 if end else r, Lr, m2, nt + is_turn, hx))  # 0.1 mile
                stack.extend(reversed(kids))                                   # successors in succ() order
            n_exp.append(n)
            why = why or stop
        return best, n_exp, n_end, why

    # ---- the method
    def reconstruct(self, inputs, t_query):
        mg = inputs.map_graph
        if mg is None:
            raise ValueError("Dewri needs a map graph")
        init = inputs.init
        if self.heading and init.theta0 is None:
            raise ValueError("the heading-known variant needs theta0")
        t_start = time.time()
        t_query = np.asarray(t_query, dtype=float)
        if not inputs.has("speed"):
            raise DewriNoPath("no speed signal on this drive")
        # <t, s, d> once per second of CAN from t0 to the window end, wherever the reference fixes are (t_query: the
        # output times only)
        t0 = float(init.t0)
        t_end = float(t_query[-1]) if init.t_end is None else max(float(init.t_end), float(t_query[-1]))
        tg = t0 + np.arange(1.0, max(1, math.floor(t_end - t0 + 1e-6)) + 1.0)
        v, dg, d = one_hz(inputs.t_s, inputs["speed"], t0, tg, t_query)
        s = v / MPH
        keep, stops, n_req = preprocess(tg, s, dg, self.zero_mph, self.traffic_mi, self.stop_min_s)
        if not keep.any():
            raise DewriNoPath("no samples left after the traffic pre-processing")
        self.dk, self.sk = dg[keep].tolist(), s[keep].tolist()
        D, n = max(float(dg[-1]), float(d[-1])), len(t_query)                  # trip length: d at the last sample
        # budget scaled with the episode length unless given (as Elastic Pathing): 21k expansions and 60 s for 5 min,
        # shared by the start edges
        self._timeout = self.timeout_s if self.timeout_s is not None else float(np.clip(0.2 * n, 30.0, 300.0))
        self._max_exp = self.max_expand if self.max_expand is not None else int(np.clip(70 * n, 20000, 200000))
        radius, cands = 50.0, []
        while not cands and radius <= 400:
            cands = mg.nearest_edges(*init.p0_xy, radius)
            radius *= 2
        if not cands:
            raise DewriNoPath("no road within 400 m of the start")
        cands = ([e for e in cands if mg.length[e] >= self.short_m] or cands)[:12]  # a shorter edge has no heading
        if self.heading:
            cands = [e for e in cands if abs((mg.bearing[e] - init.theta0 + math.pi) % (2 * math.pi) - math.pi) < math.radians(45)] or cands
        roots = [(int(e), float(mg.project(e, *init.p0_xy)[0])) for e in cands]
        # the visible network within D + 1 km of p0 (drive frame): intersections and shortest paths
        g0, to_g, vis = _parts(mg)
        cx, cy = to_g(*init.p0_xy)
        near = np.hypot(g0.node_xy[:, 0] - cx, g0.node_xy[:, 1] - cy) <= D + 1000.0
        touch = near[g0.u] | near[g0.v]
        if vis is not None:
            touch &= vis
        M = len(g0.node_xy) + 1
        key = np.unique(np.minimum(g0.u[touch], g0.v[touch]) * M + np.maximum(g0.u[touch], g0.v[touch]))
        inter = (np.bincount(key // M, minlength=M) + np.bincount(key % M, minlength=M)) >= 3
        sp = _ShortestPaths(g0, near, vis)
        best, root_exp, n_end, why = self._dfs(mg, g0, roots, D, stops[:n_req].tolist(), inter, sp, t_start)
        n_exp = sum(root_exp)
        self.stats = dict(n_stops=len(stops), n_required=n_req, n_kept=int(keep.sum()), trip_m=round(D, 1), n_roots=len(roots),
                          n_roots_expanded=sum(k > 0 for k in root_exp), expansions=n_exp, root_expansions=root_exp, n_end=n_end,
                          n_cands=len(best), n_dijkstra=sp.runs, budget_hit=why is not None, secs=round(time.time() - t_start, 2))
        if not best:
            if why is not None:
                raise DewriNoPath(f"budget reached ({why}) before a candidate path ({n_exp} expansions, {time.time() - t_start:.0f} s)")
            raise DewriNoPath(f"search exhausted: no path meets the constraints ({n_exp} expansions, {n_req} stop-points)")
        # ranking (Sec. 4.3): the pre-processed samples under each candidate's augmented model
        ends = list(best)
        ll = np.array([self._score(g0, best[x][1], best[x][2], dg[keep], s[keep], stops) for x in ends])
        order = np.argsort(-ll, kind="stable")[:self.K]
        paths, theta = [], None
        for k in order:
            _, edges, o = best[ends[k]]
            cum = self._cum(g0, edges, o)
            sq = np.clip(d, 0.0, cum[-1])                                      # CAN distance along the path from p0
            j = np.clip(np.searchsorted(cum, sq, side="right") - 1, 0, len(edges) - 1)
            es = edges[j]
            paths.append(mg.points(es, np.minimum(sq - cum[j] + np.where(j == 0, o, 0.0), g0.length[es])))
            if theta is None:
                theta = mg.bearing[es]
        pr = np.exp(ll[order] - ll[order].max())
        pr /= pr.sum()
        self.stats["secs"] = round(time.time() - t_start, 2)
        xy = paths[0]
        if len(paths) > 1:
            return xy, theta, np.stack(paths), pr
        return xy, theta, None, None
