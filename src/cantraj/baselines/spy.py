"""SPy / RoCuMa re-implemented from Pese, Pu, Shin, "SPy: Car Steering Reveals Your Trip Route!", PoPETs 2020(2):155-174
(doi 10.2478/popets-2020-0022). The paper releases no code or data; the DTW runs in C (cantraj/native/dtw.c; when it
cannot be built the episode is an error, unless CANTRAJ_PY_DTW=1 allows the slow Python reference).

Paper algorithm (pp. 160-164). Offline, steering wheel angle (SWA) only: no speed (p. 162), no start point or heading
(Table 1, p. 156), a rough area (pp. 156, 168). The SWA is smoothed with a moving average, |SWA| above 90 deg is a turn
stored as left or right, the trace is cut into sections between turns and the salient section is the one with the
largest |SWA| (p. 162). Ground truth: OSM ways spline-interpolated, signed curvature k = (x'y'' - y'x'')/(x'^2 +
y'^2)^1.5 (Eq. 2, p. 161), every road in both directions, cut into segments at intersections (p. 161). Both signals are
z-scored (Eq. 4); Algorithm 1 (pp. 162-163) scores the salient section against every contiguous run of 1..N segments of
every road in both directions (N(N+1)/2 runs per direction) with DTW and keeps the global argmin, with a window of 30 for
runs of fewer than 4 segments and none above 4 (p. 164). The other sections are chained greedily: at the end (start) of
the match the turn angle whose sign agrees with the detected turn type (Eq. 5, p. 163) gives the next road and
direction, runs of 1..p segments along it are scored with DTW and the argmin is kept, no backtracking (p. 164).

Values the paper leaves open (fixed here; reports/icde/methods_repro.md): SWA on the paper's 10 Hz (Table 5, p. 173),
mean of the finite samples per 0.1 s bin, empty bins held; moving average of `ma` = 5 samples, centred, shrinking at the
ends (MATLAB smooth); turn = run of |smoothed| > `turn_deg` = 90 (p. 162, every vehicle), runs closer than `merge_s` =
1 s merged, type = sign of the peak (steer_angle and Eq. 2 are both left positive), interval extended until |smoothed|
< `ramp_deg` = 20 (otherwise the ramps dominate salience and z-scores; overlaps merge); z-score with ddof = 1, std <
1e-9 -> 1 (MATLAB zscore), Eq. 3 skipped (a positive affine map: z-scores are invariant); DTW = MATLAB dtw as inferred
in the spec (|a_i - b_j|, steps (1,0), (0,1), (1,1) of equal weight, unnormalised D(m,n), band |j - i (n-1)/(m-1)| <=
w), w = `band` = 30 samples for runs of <= `band_max_segs` = 3 segments (exactly 4 is open), none above; curvature of an
interpolating cubic spline (not-a-knot, chord-length parameter, metric frame) every `ds` = 1 m (p. 166 storage), not
smoothed; ties keep the first run scored (strokes in order of their first edge id, then start segment, then length).
Adaptations: no road names in our tiles, so a road is a directed stroke: the same OSM way through its nodes, at a way
end the successor with the smallest |turn| <= `stroke_deg` = 30, links kept only when mutual (every directed edge on
exactly one stroke, U-turns never continue); segments end at intersections (>= 3 distinct visible undirected
neighbours) and stroke ends; a direction exists where its directed edges exist (one-way roads once). No driveway
trimming (p. 165 drops the first and last sections by hand): benchmark windows start and end while driving. The turn
test uses the succ() turn angle (left positive) instead of Eq. 5 and needs |turn| >= `stroke_deg` (a straight
continuation is no turn, whatever the sign of its near-zero angle); several candidates -> the one closest to 90 deg,
none -> the chain stops on that side. Runs longer than `v_max` = 50 m/s x the section duration are skipped (a physical
bound, a cost guard; a chain step always keeps its first run), as are runs of one curvature sample (no z-score); a
chain step whose section has fewer than 2 samples takes the first run unscored. `start="area"` (the paper's prior):
the database is the strokes of the visible edges whose midpoint lies in the axis-aligned square of half side
`area_half_m` around p0 in the drive frame (placed with p0, as can_trace_area; strokes end at its border); the chain
runs forward and backward on the visible map. `start="known"` (adaptation, as can_trace): the first section of at
least 2 samples is matched against the runs from p0's projection on the nearest edges (the first 12 within 50 to 400 m,
both directions, one per stroke: its edge nearest p0, as the others' projections are clamped to their ends;
`heading=True` keeps those within 45 deg of theta0) to the end of segment 1..p of their stroke, then the chain runs
forward; a window starting inside a turn (section 0 shorter) has the turns before that section at p0 and no heading
filter (theta0 is a mid-turn heading). Positions per second (an extension; the paper outputs a route): the DTW warping
path places a matched section's samples along its run (mean matched curvature index per sample); the window cuts its
first and last sections mid-road while runs end at segment boundaries, so for positions only (the route choice keeps
the closed DTW), when covering the whole run in the section's duration would need more than `v_max`, the last
section's path has a free end (argmin_j D(m-1, j), ties: the smallest j) and, in the area variant, the first section's
a free start (any cell of row 0; the known start is p0), both without band (a free end everywhere would pin straight
last sections near their turn, as a z-scored straight run is flat); samples inside a turn sit at its intersection, the position holds at the route start before the first matched section and at
its end after the last; one path. A budget of `max_cells` DP cells per episode (default 1e8 per query, clipped to
[1e10, 1e11]; deterministic) ends the search with a failure (no output), as do the `timeout_s` safety net far above it
(default 0.5 s per query, clipped to [60, 600] s), a missing or constant steering signal and a start without a road.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.interpolate import CubicSpline

from .. import native
from .base import OfflineBaseline
from .macrotracking import _parts

DT = 0.1                # s: the SWA grid (10 Hz)
MIN_SEC = 2             # samples: a shorter section or run cannot be z-scored
MAX_STROKE = 1000000    # edges: guard on stroke walks


class SPyNoMatch(RuntimeError):
    pass


def resample(tt, vv, t0, t1):
    """10 Hz grid tg = t0 + 0.1 k (k >= 1, tg <= t1) with the mean of the finite samples in (tg - 0.1, tg]; empty bins
    hold the previous value, leading ones the first (values None when no bin has a sample)."""
    K = int(math.floor((t1 - t0) / DT + 1e-6))
    edges = t0 + DT * np.arange(K + 1)
    fin = np.isfinite(vv)
    tf, vf = tt[fin], vv[fin]
    lo, hi = np.searchsorted(tf, edges[:-1], side="right"), np.searchsorted(tf, edges[1:], side="right")
    has = hi > lo
    if not has.any():
        return edges[1:], None
    cs = np.concatenate([[0.0], np.cumsum(vf)])
    x = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    src = np.maximum.accumulate(np.where(has, np.arange(K), -1))
    return edges[1:], x[np.where(src >= 0, src, int(np.argmax(has)))]


def smooth(x, span):
    """Centred moving average over `span` samples (an even span uses span - 1), shrinking symmetrically at the ends
    (MATLAB smooth, 'moving')."""
    n, h = len(x), max((int(span) - 1) // 2, 0)
    if h == 0 or n < 3:
        return np.array(x, dtype=float)
    i = np.arange(n)
    hw = np.minimum(np.minimum(i, n - 1 - i), h)
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[i + hw + 1] - cs[i - hw]) / (2 * hw + 1)


def detect_turns(sw, turn_deg=90.0, ramp_deg=20.0, merge_s=1.0):
    """Turns of a smoothed 10 Hz SWA trace (deg, left positive): runs of |sw| > turn_deg, runs closer than merge_s
    merged, each extended outward until |sw| < ramp_deg, overlapping intervals merged. [(i0, i1 exclusive, +1 left or -1
    right by the sign of the peak)]."""
    a = np.abs(sw)
    d = np.diff(np.concatenate([[0], (a > turn_deg).astype(np.int8), [0]]))
    runs = []
    for s, e in zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()):
        if runs and (s - runs[-1][1]) * DT < merge_s:
            runs[-1][1] = e
        else:
            runs.append([s, e])
    out = []
    for s, e in runs:
        while s > 0 and a[s - 1] >= ramp_deg:
            s -= 1
        while e < len(a) and a[e] >= ramp_deg:
            e += 1
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e, 1 if sw[s + int(np.argmax(a[s:e]))] > 0 else -1) for s, e in out]


def sections(n, turns):
    """Sample intervals [i0, i1) between the turns, with the pieces before the first and after the last (maybe empty)."""
    out, a = [], 0
    for s, e, _ in turns:
        out.append((a, s))
        a = e
    return out + [(a, n)]


def curvature(xy, t, ds):
    """Eq. 2 (1/m, left positive) of the interpolating cubic spline (not-a-knot) through the polyline xy with its
    chord-length parameter t, sampled every ds metres; straight (zero) with fewer than 3 distinct points."""
    n = int(math.floor(t[-1] / ds + 1e-9)) + 1
    keep = np.concatenate([[True], np.diff(t) > 1e-9])                  # repeated points dropped
    if keep.sum() < 3:
        return np.zeros(n)
    sp = CubicSpline(t[keep], xy[keep] - xy[0], bc_type="not-a-knot", axis=0)
    s = ds * np.arange(n)
    d1, d2 = sp(s, 1), sp(s, 2)
    den = (d1[:, 0] ** 2 + d1[:, 1] ** 2) ** 1.5
    return np.where(den > 1e-12, (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum(den, 1e-12), 0.0)


def _in_csr(g):
    """Incoming edges per node (CSR over the head node, ascending edge ids); visibility-independent, cached on the graph."""
    c = getattr(g, "_in_csr", None)
    if c is None:
        o = np.argsort(g.v, kind="stable")
        c = g._in_csr = (o, np.searchsorted(g.v[o], np.arange(len(g.node_xy) + 1)))
    return c


class _Stroke:
    __slots__ = ("edges", "nodes", "t", "bnd", "k")

    def __init__(self, edges, nodes, t, bnd):
        self.edges, self.nodes, self.t, self.bnd, self.k = edges, nodes, t, bnd, None   # t: distance at each node (m)


class _Roads:
    """Directed strokes of the visible map (or of the edges in `ok`), built lazily: the same OSM way through its nodes,
    at a way end the successor with the smallest |turn| <= stroke_deg; a link a -> b is kept only when b is a's best
    continuation and a is b's best predecessor, so every directed edge lies on exactly one stroke (a closed one starts
    at its smallest edge id). Segment boundaries (bnd, node indices): intersections and the stroke ends."""

    def __init__(self, mg, ds, stroke_deg, ok=None):
        self.g0, _, self.vis = _parts(mg)
        self.ok = self.vis if ok is None else ok
        self.ds, self.thr = float(ds), math.radians(stroke_deg)
        self.in_o, self.in_p = _in_csr(self.g0)
        self._nx, self._pv, self._x = {}, {}, {}
        self.of, self.strokes = {}, []

    def _use(self, e):
        return self.ok is None or bool(self.ok[e])

    def preds(self, b):
        """[(edge, turn)] entering the tail of b that may continue into b: usable, no U-turn, turn restrictions kept."""
        g, res = self.g0, []
        for i in self.in_o[self.in_p[g.u[b]]:self.in_p[g.u[b] + 1]].tolist():
            if not self._use(i) or g.reverse[i] == b or (i, b) in g.forbid or (i in g.only and b not in g.only[i]):
                continue
            res.append((i, (g.bearing[b] - g.bearing[i] + math.pi) % (2 * math.pi) - math.pi))
        return res

    def _best(self, e, cands):
        """The same way's continuation if there is one, else the straightest within stroke_deg (ties: smallest id); -1."""
        wid = self.g0.way_id
        pool = [z for z in cands if wid[z[0]] == wid[e]] or [z for z in cands if abs(z[1]) <= self.thr]
        return min(pool, key=lambda z: (abs(z[1]), z[0]))[0] if pool else -1

    def cont(self, a):
        r = self._nx.get(a)
        if r is None:
            r = self._nx[a] = self._best(a, [(o, t) for o, t, ut in self.g0.succ(a) if not ut and self._use(o)])
        return r

    def pred(self, b):
        r = self._pv.get(b)
        if r is None:
            r = self._pv[b] = self._best(b, self.preds(b))
        return r

    def is_x(self, n):
        """Intersection: >= 3 distinct visible undirected neighbours."""
        r = self._x.get(n)
        if r is None:
            g, vis = self.g0, self.vis
            nb = {int(g.v[e]) for e in g.out[n] if vis is None or vis[e]}
            nb |= {int(g.u[e]) for e in self.in_o[self.in_p[n]:self.in_p[n + 1]].tolist() if vis is None or vis[e]}
            r = self._x[n] = len(nb) >= 3
        return r

    def stroke_of(self, e):
        """(stroke of the directed edge e, index of e in it)."""
        hit = self.of.get(e)
        if hit is None:
            s, closed = e, False
            for _ in range(MAX_STROKE):                                     # back to the first edge
                p = self.pred(s)
                if p < 0 or self.cont(p) != s:
                    break
                if p == e:
                    closed = True
                    break
                s = p
            if closed:                                                      # a closed stroke starts at its smallest id
                s, c = e, self.cont(e)
                while c != e:
                    s, c = min(s, c), self.cont(c)
            es = [s]
            for _ in range(MAX_STROKE):
                c = self.cont(es[-1])
                if c < 0 or c == s or self.pred(c) != es[-1]:
                    break
                es.append(c)
            g = self.g0
            es = np.array(es, dtype=np.int64)
            nodes = np.concatenate([g.u[es[:1]], g.v[es]])
            inner = [k for k in range(1, len(es)) if self.is_x(int(nodes[k]))]
            for k, x in enumerate(es.tolist()):
                self.of[x] = (len(self.strokes), k)
            self.strokes.append(_Stroke(es, nodes, np.concatenate([[0.0], np.cumsum(g.length[es])]),
                                        np.array([0] + inner + [len(es)], dtype=np.int64)))
            hit = self.of[e]
        return self.strokes[hit[0]], hit[1]

    def curv(self, st):
        if st.k is None:
            st.k = curvature(self.g0.node_xy[st.nodes], st.t, self.ds)     # drive frame: curvature is frame-invariant
        return st.k

    def idx(self, st, d):
        """Curvature sample index of the stroke distance d (m)."""
        return np.clip(np.rint(np.asarray(d, dtype=float) / self.ds), 0, len(self.curv(st)) - 1).astype(np.int64)


class _Piece:
    """A matched run: edges[ke:k1] of stroke st from offset o on edges[ke]; section sec; DTW run (i0, i1, band) in curvature
    samples (inclusive)."""
    __slots__ = ("st", "ke", "o", "k1", "sec", "run")

    def __init__(self, st, ke, o, k1, run):
        self.st, self.ke, self.o, self.k1, self.sec, self.run = st, ke, o, k1, None, run


class SPy(OfflineBaseline):
    needs = ("steer_angle",)

    def __init__(self, start="area", heading=False, ma=5, ramp_deg=20.0, turn_deg=90.0, merge_s=1.0, ds=1.0, band=30,
                 band_max_segs=3, stroke_deg=30.0, area_half_m=2500.0, v_max=50.0, max_cells=None, timeout_s=None):
        self.start, self.heading = start, heading
        self.ma, self.ramp_deg, self.turn_deg, self.merge_s, self.ds = ma, ramp_deg, turn_deg, merge_s, ds
        self.band, self.band_max_segs, self.stroke_deg = band, band_max_segs, stroke_deg
        self.area_half_m, self.v_max, self.max_cells, self.timeout_s = area_half_m, v_max, max_cells, timeout_s
        self.name = "spy" + ("_area" if start == "area" else "") + ("_hp" if heading else "")
        self.stats = {}

    # ---- trace
    def _steering(self, inputs, t_query):
        try:
            sa = inputs["steer_angle"]
        except KeyError:                                                    # absent, or outside the group's whitelist
            raise SPyNoMatch("no steering angle signal on this drive") from None
        tg, x = resample(inputs.t_s, sa, float(inputs.init.t0), float(t_query[-1]))
        if x is None:
            raise SPyNoMatch("no finite steering angle sample in the episode")
        if np.ptp(x) == 0.0:
            raise SPyNoMatch("constant steering angle in the episode")
        return tg, smooth(x, self.ma)

    def _w(self, n_segs):
        return int(round(self.band)) if n_segs <= self.band_max_segs else -1

    def _dtw(self, a, b, runs, best=math.inf):
        """One DTW batch under the episode budget (and the wall-clock safety net, checked after every batch), cutoff
        `best` carried in; (index of the run that lowered it (ties: the first) or -1, the new best)."""
        _, best, bi, self._cells, hit = native.dtw_batch(a, b, runs, best=best, cells=self._cells, cap=self._cap)
        self._runs += len(runs)
        elapsed = time.time() - self._t
        if hit or elapsed > self._timeout:
            self.stats.update(cells=self._cells, n_runs=self._runs, budget_hit=True, secs=round(elapsed, 2),
                              n_strokes=sum(len(r.strokes) for r in self._nets))
            why = f"{self._cells:.3g} DTW cells" if hit else f"timeout after {elapsed:.0f} s"
            raise SPyNoMatch(f"budget reached ({why}, {self._runs} runs)")
        return bi, best

    # ---- Algorithm 1 (area) and the known-start adaptation
    def _salient(self, db, secs):
        """The salient section against every contiguous run of 1..N segments of every stroke in the square, one stroke
        per batch with the best score carried over (the same order and ties as a single batch, bounded memory)."""
        ok = [q for q, (i0, i1) in enumerate(secs) if i1 - i0 >= MIN_SEC]
        if not ok:
            raise SPyNoMatch("no section long enough to match")
        q = max(ok, key=lambda k: float(np.abs(self.sw[secs[k][0]:secs[k][1]]).max()))   # first of equals
        self.stats["salient"] = q
        a = self.sw[secs[q][0]:secs[q][1]]
        for e in np.flatnonzero(db.ok).tolist():                            # strokes in order of their first edge id
            if e not in db.of:
                db.stroke_of(e)
        lmax = self.v_max * len(a) * DT
        best, hit, n_runs = math.inf, None, 0
        for st in db.strokes:
            tb = st.t[st.bnd]
            ii, jj = np.triu_indices(len(tb), 1)                            # boundary pairs = contiguous runs
            keep = tb[jj] - tb[ii] <= lmax
            if not keep.any():
                continue
            sb = db.idx(st, tb)
            keep &= sb[jj] - sb[ii] + 1 >= MIN_SEC                         # one sample cannot be z-scored
            ii, jj = ii[keep], jj[keep]
            if not len(ii):
                continue
            w = np.where(jj - ii <= self.band_max_segs, int(round(self.band)), -1)
            bi, best = self._dtw(a, db.curv(st), np.column_stack([sb[ii], sb[jj] + 1, w]), best)
            st.k = None                                                     # recomputed for the matched stroke
            n_runs += len(ii)
            if bi >= 0:
                i, j = int(ii[bi]), int(jj[bi])
                hit = _Piece(st, int(st.bnd[i]), 0.0, int(st.bnd[j]), (int(sb[i]), int(sb[j]), self._w(j - i)))
        if not n_runs:
            raise SPyNoMatch("no run within v_max x the salient section's duration" if db.strokes else "no road in the area square")
        if hit is None:
            raise SPyNoMatch("no admissible run for the salient section")
        hit.sec = q
        return hit

    def _known(self, mg, init, roads, secs):
        """Known start: the first section q0 that can be z-scored (> 0 when the window starts inside a turn: the turns
        before it happen at p0) against the runs from p0's projection on the nearest edges (one per stroke) to the end of
        segment 1..p of their stroke."""
        q0 = next((q for q, (i0, i1) in enumerate(secs) if i1 - i0 >= MIN_SEC), None)
        if q0 is None:
            raise SPyNoMatch("no section long enough to match")
        radius, cands = 50.0, []
        while not cands and radius <= 400:
            cands = mg.nearest_edges(*init.p0_xy, radius)
            radius *= 2
        if not cands:
            raise SPyNoMatch("no road within 400 m of the start")
        cands = cands[:12]
        if self.heading and q0 == 0:                                        # otherwise theta0 is a mid-turn heading
            cands = [e for e in cands if abs((mg.bearing[e] - init.theta0 + math.pi) % (2 * math.pi) - math.pi) < math.radians(45)] or cands
        opts, seen = [], set()
        for e in cands:                                                     # nearest first: on a stroke's farther
            st, ke = roads.stroke_of(int(e))                                # edges p0 projects to an edge end
            if id(st) in seen:
                continue
            seen.add(id(st))
            o, _ = mg.project(e, *init.p0_xy)
            opts += self._ahead(roads, st, ke, float(o), secs[q0])
        p = self._pick(roads, opts, secs, q0)
        if p is None:
            raise SPyNoMatch("no admissible run from the start")
        return p

    # ---- chaining (p. 164)
    def _ahead(self, roads, st, ke, o, sec):
        """Runs on st from offset o on its edge ke to each later segment end (the first always, then up to v_max x the
        section duration; runs of one curvature sample skipped)."""
        d0 = st.t[ke] + o
        lmax = self.v_max * (sec[1] - sec[0]) * DT
        i0, out = int(roads.idx(st, d0)), []
        for c, b in enumerate(st.bnd[st.bnd > ke].tolist()):
            i1 = int(roads.idx(st, st.t[b]))
            if i1 - i0 + 1 < MIN_SEC:                                       # one sample cannot be z-scored
                continue
            if out and st.t[b] - d0 > lmax:
                break
            out.append(_Piece(st, ke, o, b, (i0, i1, self._w(c + 1))))
        return out

    def _behind(self, roads, st, k1, sec):
        """Runs on st from each earlier segment start (nearest first) to its node k1 (mirrors _ahead)."""
        lmax = self.v_max * (sec[1] - sec[0]) * DT
        i1, out = int(roads.idx(st, st.t[k1])), []
        for c, b in enumerate(st.bnd[st.bnd < k1][::-1].tolist()):
            i0 = int(roads.idx(st, st.t[b]))
            if i1 - i0 + 1 < MIN_SEC:
                continue
            if out and st.t[k1] - st.t[b] > lmax:
                break
            out.append(_Piece(st, b, 0.0, k1, (i0, i1, self._w(c + 1))))
        return out

    def _pick(self, roads, opts, secs, q):
        """The run with the lowest DTW against section q (ties: the first), the first one when the section is too short
        to be z-scored, None when no run is admissible."""
        if not opts:
            return None
        i0, i1 = secs[q]
        best = 0
        if i1 - i0 >= MIN_SEC:
            parts, off, runs, n = [], {}, [], 0
            for p in opts:
                o = off.get(id(p.st))
                if o is None:
                    off[id(p.st)] = o = n
                    parts.append(roads.curv(p.st))
                    n += len(parts[-1])
                runs.append((o + p.run[0], o + p.run[1] + 1, p.run[2]))
            best, _ = self._dtw(self.sw[i0:i1], np.concatenate(parts), runs)
            if best < 0:
                return None
        opts[best].sec = q
        return opts[best]

    def _chain(self, mg, roads, first, secs, turns):
        """Greedy, no backtracking: across each detected turn the out-edge (in-edge backwards) whose turn sign matches
        and whose |turn| is at least stroke_deg (no straight continuation), the one closest to 90 deg among several;
        forward always, backward in the area variant."""
        pieces = [first]
        for q in range(first.sec + 1, len(secs)):
            last, typ = pieces[-1], turns[q - 1][2]
            outs = [(o, t) for o, t, ut in mg.succ(int(last.st.edges[last.k1 - 1]))
                    if not ut and t * typ > 0 and abs(t) >= roads.thr]
            if not outs:
                break
            st, ke = roads.stroke_of(int(min(outs, key=lambda z: abs(abs(z[1]) - math.pi / 2))[0]))
            p = self._pick(roads, self._ahead(roads, st, ke, 0.0, secs[q]), secs, q)
            if p is None:
                break
            pieces.append(p)
        if self.start == "area":
            for q in range(first.sec - 1, -1, -1):
                head, typ = pieces[0], turns[q][2]
                ins = [(i, t) for i, t in roads.preds(int(head.st.edges[head.ke]))
                       if t * typ > 0 and abs(t) >= roads.thr]
                if not ins:
                    break
                st, ki = roads.stroke_of(int(min(ins, key=lambda z: abs(abs(z[1]) - math.pi / 2))[0]))
                p = self._pick(roads, self._behind(roads, st, ki + 1, secs[q]), secs, q)
                if p is None:
                    break
                pieces.insert(0, p)
        return pieces

    # ---- positions (extension)
    def _positions(self, mg, roads, pieces, secs, tg, t_query, t0):
        """Path distance s on the 10 Hz grid from the warping paths, read out at t_query along the concatenated route
        (the window's last section with a free end, its first one with a free start in the area variant, when the
        whole run is out of reach at v_max)."""
        g0 = roads.g0
        edges, offs, r0, acc = [], [], [], 0.0
        for p in pieces:
            es = p.st.edges[p.ke:p.k1].tolist()
            edges += es
            offs += [p.o] + [0.0] * (len(es) - 1)
            r0.append(acc)
            acc += p.st.t[p.k1] - p.st.t[p.ke] - p.o
        s = np.zeros(len(self.sw))
        for p, r in zip(pieces, r0):
            i0, i1 = secs[p.sec]
            s[i0:i1] = r
            if i1 - i0 < MIN_SEC:
                continue
            j0, j1, w = p.run
            fs, fe = self.start == "area" and p.sec == 0, p.sec == len(secs) - 1      # cut by the window: open, no band
            if p.st.t[p.k1] - p.st.t[p.ke] - p.o <= self.v_max * (i1 - i0) * DT:   # ... unless the whole run is reachable
                fs = fe = False
            a, b = self.sw[i0:i1], roads.curv(p.st)[j0:j1 + 1]
            _, ii, jj, cells = native.dtw_path(a, b, -1 if fs or fe else w, fs, fe)
            self._cells += cells
            if len(ii):                                                     # mean matched curvature index per sample
                jm = np.bincount(ii, weights=jj, minlength=i1 - i0) / np.bincount(ii, minlength=i1 - i0)
                d0 = p.st.t[p.ke] + p.o
                s[i0:i1] = r + np.clip((j0 + jm) * self.ds - d0, 0.0, p.st.t[p.k1] - d0)
        s[:secs[pieces[0].sec][0]] = 0.0                                    # before the first match: the route start
        s[secs[pieces[-1].sec][1]:] = acc                                   # after the last: the route end
        for p, r in zip(pieces[1:], r0[1:]):                                # inside a turn: its intersection
            s[secs[p.sec - 1][1]:secs[p.sec][0]] = r
        s_q = np.interp(t_query, np.concatenate([[t0], tg]), np.concatenate([s[:1], s]))
        cum = np.concatenate([[0.0], np.cumsum(g0.length[np.array(edges)] - np.array(offs))])
        j = np.clip(np.searchsorted(cum, s_q, side="right") - 1, 0, len(edges) - 1)
        es = np.array(edges, dtype=np.int64)[j]
        oo = np.array(offs)[j] + (s_q - cum[j])
        return mg.points(es, np.minimum(oo, g0.length[es])), mg.bearing[es]

    def reconstruct(self, inputs, t_query):
        mg = inputs.map_graph
        if mg is None:
            raise ValueError("SPy needs a map graph")
        init = inputs.init
        if self.heading and init.theta0 is None:
            raise ValueError("the heading-known variant needs theta0")
        self._t = time.time()
        t_query = np.asarray(t_query, dtype=float)
        tg, self.sw = self._steering(inputs, t_query)
        turns = detect_turns(self.sw, self.turn_deg, self.ramp_deg, self.merge_s)
        secs = sections(len(self.sw), turns)
        # budget: DP cells per episode, 1e8 per query clipped to [1e10, 1e11] (deterministic); wall clock: a safety net
        self._cap = int(self.max_cells) if self.max_cells is not None else int(np.clip(1e8 * len(t_query), 1e10, 1e11))
        self._timeout = self.timeout_s if self.timeout_s is not None else float(np.clip(0.5 * len(t_query), 60.0, 600.0))
        self._cells, self._runs = 0, 0
        self.stats = dict(n_sections=len(secs), n_turns=len(turns), salient=None, n_runs=0, cells=0, secs=0.0,
                          n_strokes=0, budget_hit=False, n_pieces=0, native=native.available())
        roads = _Roads(mg, self.ds, self.stroke_deg)                       # the visible map (chain, known start)
        self._nets = [roads]
        if self.start == "area":                                            # the database: strokes inside the square
            g0, to_g, vis = _parts(mg)
            cx, cy = to_g(*init.p0_xy)
            mid = 0.5 * (g0.node_xy[g0.u] + g0.node_xy[g0.v])
            ok = (np.abs(mid[:, 0] - cx) <= self.area_half_m) & (np.abs(mid[:, 1] - cy) <= self.area_half_m)
            self._nets.append(_Roads(mg, self.ds, self.stroke_deg, ok=ok if vis is None else ok & vis))
            first = self._salient(self._nets[1], secs)
        else:
            first = self._known(mg, init, roads, secs)
        pieces = self._chain(mg, roads, first, secs, turns)
        xy, theta = self._positions(mg, roads, pieces, secs, tg, t_query, float(init.t0))
        self.stats.update(n_runs=self._runs, cells=self._cells, secs=round(time.time() - self._t, 2), n_pieces=len(pieces),
                          n_strokes=sum(len(r.strokes) for r in self._nets))
        return xy, theta, None, None
