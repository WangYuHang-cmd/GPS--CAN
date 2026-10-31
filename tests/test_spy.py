"""SPy / RoCuMa re-implementation on synthetic maps: the native DTW kernel equals the Python reference bit for bit (band,
no band, early abandoning, budget, warping paths with closed, free start and free end alignments), non-finite path
input is an error and a missing kernel too unless CANTRAJ_PY_DTW=1; the pre-processing (10 Hz bins, MATLAB smooth,
z-score, 90 deg turns with their ramps); directed strokes (same way, mutual straight continuations, one-way roads, a
closed roundabout) and the sign of the spline curvature; on a small town with a curvy road made of arc ways (nodes
every ~7 m) the area variant (salient section against the square, chained both ways) and the known-start variant
recover the route from steering synthesised from its curvature, the latter also from a window starting inside a turn;
a window cut mid-road on a long road keeps its first and last positions near the truth instead of the segment ends;
the chain never takes a straight continuation for a turn; the speed column is never read, runs repeat bit for bit, a
view with a far drive frame keeps positions, bearings and visibility right; a small cell budget (in the search or in
the chain), the time safety net, a start without a road, no section to match and a missing or constant steering
signal are failures."""
import math

import numpy as np
import pandas as pd
import pytest

from cantraj import native
from cantraj.baselines.spy import SPy, SPyNoMatch, _Piece, _Roads, curvature, detect_turns, resample, sections, smooth
from cantraj.events import EventStream
from cantraj.geo import LocalFrame
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.maps.graph import DriveGraph, EpisodeGraphView

LAT0, LON0 = 37.005, -121.9            # 555 m north of the 37.0 N tile border
TILE = (148, -488)
WB, SR, V = 2.7, 15.0, 10.0


# ---- synthetic maps
def arc(c, r, a0, a1, step=7.0):
    """Polyline of a circular arc (centre c, radius r, angles a0 -> a1 rad), nodes every ~step m."""
    n = max(int(math.ceil(abs(a1 - a0) * r / step)), 1)
    return [(c[0] + r * math.cos(a), c[1] + r * math.sin(a)) for a in np.linspace(a0, a1, n + 1)]


def build(ways, frame=None, oneway=()):
    """DriveGraph of residential ways given as episode-frame polylines (nodes shared by coordinates), built in `frame`."""
    ef = LocalFrame(LAT0, LON0)
    ids, nodes, wn, rows = {}, [], [], []
    for w, pts in enumerate(ways):
        for s, (x, y) in enumerate(pts):
            key = (round(x, 2), round(y, 2))
            if key not in ids:
                ids[key] = len(ids)
                la, lo = ef.to_lla(np.array([x]), np.array([y]))
                nodes.append(dict(node_id=ids[key], lat=float(la[0]), lon=float(lo[0])))
            wn.append(dict(way_id=w, seq=s, node_id=ids[key]))
        rows.append(dict(way_id=w, highway="residential", oneway=int(w in oneway), junction="", maxspeed="", lanes="", name_len=0, n_nodes=len(pts)))
    return DriveGraph(pd.DataFrame(rows), pd.DataFrame(wn), pd.DataFrame(nodes),
                      pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"]), frame or ef)


def curvy(x0, y0):
    """Road C from (x0, y0) north: 50 m straight, left arc R 80 over 60 deg (way split inside it), right arc R 60 over
    60 deg, straight to y0 + 300. Returns its three ways and the end of the left arc."""
    c1 = [(x0, y0), (x0, y0 + 25.0), (x0, y0 + 50.0)] + arc((x0 - 80.0, y0 + 50.0), 80.0, 0.0, math.radians(30))[1:]
    c2 = arc((x0 - 80.0, y0 + 50.0), 80.0, math.radians(30), math.radians(60))
    p2 = c2[-1]
    c3 = [p2] + arc((p2[0] + 30.0, p2[1] + 60.0 * math.sin(math.radians(60))), 60.0, math.radians(240), math.radians(180))[1:]
    x3 = c3[-1][0]
    return [c1, c2, c3 + [(x3, y0 + 200.0), (x3, y0 + 250.0), (x3, y0 + 300.0)]], p2


C, P2 = curvy(100.0, 0.0)
X1, X2, X3 = (100.0, 0.0), C[2][-1], (230.0, 300.0)


def town(with_c=True, decoy=False):
    """Grid A (y = 0), B (y = 300), V0, V1, V2, V3 with the curvy road C from X1 to X2 and its side road D at P2;
    standalone decoys: the mirror of C (right then left) and a single left arc; `decoy` adds a copy of C 1.2 km south
    (beyond the 37.0 N tile border)."""
    ways = [
        [(-250.0, 0.0), (-10.0, 0.0), X1, (230.0, 0.0), (480.0, 0.0)],
        [(-250.0, 300.0), (-10.0, 300.0), X2, X3, (480.0, 300.0)],
        [(-250.0, -150.0), (-250.0, 0.0), (-250.0, 300.0), (-250.0, 450.0)],
        [(-10.0, -150.0), (-10.0, 0.0), (-10.0, 300.0)],
        [(230.0, -150.0), (230.0, 0.0), (230.0, P2[1]), X3, (230.0, 450.0)],
        [(480.0, -150.0), (480.0, 0.0), (480.0, 300.0), (480.0, 450.0)],
        [P2, (230.0, P2[1])],
    ]
    if with_c:
        ways += C
    f = [(-700.0, -100.0), (-700.0, -50.0)] + arc((-620.0, -50.0), 80.0, math.pi, math.pi - math.radians(60))[1:]
    q = f[-1]
    f += arc((q[0] - 30.0, q[1] + 60.0 * math.sin(math.radians(60))), 60.0, math.radians(-60), 0.0)[1:]
    ways += [f + [(f[-1][0], f[-1][1] + 128.0)], [(700.0, -200.0)] + arc((600.0, -200.0), 100.0, 0.0, math.radians(90))[1:]]
    if decoy:
        ways += curvy(100.0, -1200.0)[0]
    return ways


# ---- synthetic drives
def drive_on(path, noise=0.5, seed=0, speed=None):
    """10 m/s along the polyline: steering = degrees(atan(k L)) SR from the path's curvature (vertex heading changes
    spread over the neighbouring edges) plus a 350 deg raised-cosine turn over 3 s at every corner above 45 deg, plus
    noise; 100 Hz events. Returns (events, duration, truth(q))."""
    P = np.asarray(path, dtype=float)
    seg = np.hypot(*np.diff(P, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    T = cum[-1] / V
    t = np.arange(0.0, T, 0.01)
    s = V * t
    head = np.arctan2(np.diff(P[:, 1]), np.diff(P[:, 0]))
    dh = (np.diff(head) + math.pi) % (2 * math.pi) - math.pi
    kv, corners = np.zeros(len(P)), []
    for i in range(1, len(P) - 1):
        if abs(dh[i - 1]) > math.radians(45):
            corners.append((cum[i] / V, math.copysign(1.0, dh[i - 1])))
        else:
            kv[i] = dh[i - 1] / (0.5 * (seg[i - 1] + seg[i]))
    k = np.where((s < cum[1]) | (s > cum[-2]), 0.0, np.interp(s, cum[1:-1], kv[1:-1]))
    steer = np.degrees(np.arctan(k * WB)) * SR
    for tc, sg in corners:
        u = np.clip((t - tc) / 1.5, -1.0, 1.0)
        steer += sg * 350.0 * 0.5 * (1.0 + np.cos(np.pi * u))
    steer += np.random.default_rng(seed).normal(0.0, noise, len(t))
    sp = np.full(len(t), V) if speed is None else speed(t)
    ev = EventStream(t, {"speed": sp, "steer_angle": steer})

    def truth(q):
        sq = np.clip(V * np.asarray(q, dtype=float), 0.0, cum[-1])
        return np.column_stack([np.interp(sq, cum, P[:, 0]), np.interp(sq, cum, P[:, 1])])
    return ev, T, truth


AREA_ROUTE = [(0.0, 0.0), X1] + C[0][1:] + C[1][1:] + C[2][1:] + [X3, (230.0, P2[1]), (230.0, P2[1] - 5.0)]
KNOWN_ROUTE = [(100.0, 20.0)] + C[0][2:] + C[1][1:] + C[2][1:] + [X3, (230.0, P2[1]), (230.0, P2[1] - 5.0)]


def t_at(path, pt):
    """Time (s) at which drive_on(path) passes the vertex pt."""
    P = np.asarray(path, dtype=float)
    k = int(np.flatnonzero(np.hypot(*(P - np.asarray(pt)).T) < 1e-6)[0])
    return float(np.sum(np.hypot(*np.diff(P[:k + 1], axis=0).T))) / V


def window(ev, t0, T, p0, theta0=None, graph=None):
    """G3 inputs of the drive's window [t0, T] on the graph (built in the frame at (LAT0, LON0), which is the episode
    frame); p0 is given as the start position in that frame."""
    init = EpisodeInit(t0, LAT0, LON0, theta0, t_end=T, dx_m=float(p0[0]), dy_m=float(p0[1]))
    inp = EpisodeInputs("G3", ev.window(t0, T), StaticParams(wheelbase_m=WB, steer_ratio=SR), init, input_layer="events")
    inp.map_graph = graph if graph is not None else build(town())
    return inp


def inputs(ev, T, p0=(0.0, 0.0), theta0=None, graph=None):
    """G3 inputs; the episode frame is centred at p0 (the graph is built in the frame at (LAT0, LON0) unless given)."""
    la, lo = LocalFrame(LAT0, LON0).to_lla(np.array([p0[0]]), np.array([p0[1]]))
    inp = EpisodeInputs("G3", ev, StaticParams(wheelbase_m=WB, steer_ratio=SR), EpisodeInit(0.0, float(la[0]), float(lo[0]), theta0, t_end=T), input_layer="events")
    g = graph if graph is not None else build(town())
    inp.map_graph = g if p0 == (0.0, 0.0) and graph is None else EpisodeGraphView(g, LocalFrame(float(la[0]), float(lo[0])))
    return inp


def episode_xy(p0, pts):
    """Points of the (LAT0, LON0) frame in the episode frame centred at p0."""
    f = LocalFrame(LAT0, LON0)
    la, lo = f.to_lla(np.array([p0[0]]), np.array([p0[1]]))
    x, y = LocalFrame(float(la[0]), float(lo[0])).to_enu(*f.to_lla(pts[:, 0], pts[:, 1]))
    return np.column_stack([x, y])


def edge(g, a, b):
    na = int(np.argmin(np.hypot(*(g.node_xy - np.asarray(a)).T)))
    nb = int(np.argmin(np.hypot(*(g.node_xy - np.asarray(b)).T)))
    return [e for e in range(g.n_edges()) if g.u[e] == na and g.v[e] == nb][0]


# ---- native kernel
def test_native_dtw_equals_the_reference():
    assert native.available(), native.build_error()
    rng = np.random.default_rng(7)
    for trial in range(120):
        m, nb = int(rng.integers(1, 30)), int(rng.integers(1, 90))
        a, b = rng.normal(size=m).cumsum(), rng.normal(size=nb).cumsum()
        runs = []
        for _ in range(int(rng.integers(1, 10))):
            s = int(rng.integers(0, nb))
            runs.append((s, int(rng.integers(s + 1, nb + 1)), int(rng.choice([-1, 0, 2, 30]))))
        best = math.inf if trial % 3 else float(rng.uniform(0.0, 20.0))            # a cutoff from earlier batches
        cap = None if trial % 4 else int(rng.integers(1, 2000))                     # budget reached mid-batch
        x = native.dtw_batch(a, b, runs, best=best, cap=cap)
        y = native.dtw_batch(a, b, runs, best=best, cap=cap, native=False)
        assert np.array_equal(x[0], y[0]) and x[1:] == y[1:]
        for (s, e, w), fs, fe in zip(runs[:2] * 4, [False, True] * 4, [False, False, True, True] * 2):
            p, r = native.dtw_path(a, b[s:e], w, fs, fe), native.dtw_path(a, b[s:e], w, fs, fe, native=False)
            assert (p[0] == r[0] or math.isinf(p[0]) and math.isinf(r[0])) and p[3] == r[3]
            assert np.array_equal(p[1], r[1]) and np.array_equal(p[2], r[2])
    a, b = rng.normal(size=300).cumsum(), rng.normal(size=420).cumsum()             # several checkpoint blocks
    for w in (-1, 30):
        p, r = native.dtw_path(a, b, w), native.dtw_path(a, b, w, native=False)
        assert p[0] == r[0] == native.dtw_batch(a, b, [(0, 420, w)])[1]
        assert np.array_equal(p[1], r[1]) and np.array_equal(p[2], r[2])
        assert p[1][0] == p[2][0] == 0 and p[1][-1] == 299 and p[2][-1] == 419
        assert np.all(np.diff(p[1]) >= 0) and np.all(np.diff(p[2]) >= 0)
        for fs, fe in ((True, False), (False, True), (True, True)):                # open ends across the blocks too
            p, r = native.dtw_path(a, b, w, fs, fe), native.dtw_path(a, b, w, fs, fe, native=False)
            assert p[0] == r[0] and p[3] == r[3] and np.array_equal(p[1], r[1]) and np.array_equal(p[2], r[2])
            assert p[1][0] == 0 and p[1][-1] == 299 and (fs or p[2][0] == 0) and (fe or p[2][-1] == 419)


def test_dtw_path_free_start_and_end():
    """Open ends (the window-edge sections): with a free start any cell of row 0 begins the path, with a free end it
    stops at the first column of the last row's minimum; scores equal a plain DP and the path's cost, a pattern cut out
    of a longer series is found inside it; non-finite input raises (an overflowing finite one has no path), no crash."""
    rng = np.random.default_rng(11)
    for trial in range(40):
        m, n = int(rng.integers(1, 25)), int(rng.integers(1, 40))
        a, b = rng.normal(size=m).cumsum(), np.round(rng.normal(size=n).cumsum())   # rounded: tied columns
        za, zb = native.zscore(a), native.zscore(b)
        for fs in (False, True):
            D = np.full((m, n), np.inf)
            for i in range(m):
                for j in range(n):
                    if i == 0:
                        D[i, j] = abs(za[0] - zb[j]) + (0.0 if j == 0 or fs else D[0, j - 1])
                    else:
                        D[i, j] = abs(za[i] - zb[j]) + min(D[i - 1, j], D[i - 1, j - 1] if j else np.inf,
                                                           D[i, j - 1] if j else np.inf)
            for fe in (False, True):
                sc, pi, pj, _ = native.dtw_path(a, b, -1, fs, fe)
                je = int(np.argmin(D[-1])) if fe else n - 1                         # the first column of the minimum
                assert sc == pytest.approx(D[-1, je], abs=1e-9) and pi[-1] == m - 1 and pj[-1] == je
                assert pi[0] == 0 and (fs or pj[0] == 0)
                assert sc == pytest.approx(np.abs(za[pi] - zb[pj]).sum(), abs=1e-9)
    x = np.sin(np.linspace(0.0, 3.0 * np.pi, 60))
    flat = x.mean() + x.std(ddof=1) * (-1.0) ** np.arange(100)                     # same mean and std as x, so
    b = np.concatenate([flat, x, flat])                                             # zscore(b)[100:160] ~ zscore(x)
    sc, pi, pj, _ = native.dtw_path(x, b, -1, True, True)
    assert sc < 1e-9 and np.array_equal(pi, np.arange(60)) and np.array_equal(pj, np.arange(100, 160))
    _, _, pj, _ = native.dtw_path(x, b, -1)
    assert pj[0] == 0 and pj[-1] == 259                                             # closed: the whole series
    for bad in (np.nan, np.inf):
        for nat in (True, False):
            with pytest.raises(ValueError, match="non-finite"):
                native.dtw_path(np.arange(5.0), np.array([0.0, 1.0, bad, 3.0]), native=nat)
    for nat in (True, False):                                                       # z-scores overflow to NaN
        with np.errstate(over="ignore", invalid="ignore"):
            sc, pi, _, _ = native.dtw_path(np.array([1e308, 1e308, 0.0]), np.arange(6.0), -1, True, True, native=nat)
        assert math.isinf(sc) and len(pi) == 0


def test_dtw_semantics_abandoning_and_ties():
    a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    sc = native.dtw_batch(a, np.concatenate([a, a, a[::-1] * -1.0]), [(0, 5, -1), (5, 10, -1), (10, 15, -1)])
    assert sc[0][0] == 0.0 and sc[2] == 0 and sc[0][1] == 0.0          # tie: the first run keeps the best
    assert math.isinf(sc[0][2])                                         # abandoned once a row exceeds the best (0)
    za, zb = native.zscore(a), native.zscore(np.array([0.0, 2.0, 1.0]))
    full = native.dtw_batch(a, np.array([0.0, 2.0, 1.0]), [(0, 3, -1)])[1]
    D = np.full((6, 4), np.inf)
    D[0, 0] = 0.0
    for i in range(5):
        for j in range(3):
            D[i + 1, j + 1] = abs(za[i] - zb[j]) + min(D[i, j + 1], D[i + 1, j], D[i, j])
    assert full == pytest.approx(D[5, 3], abs=1e-12)
    assert native.band_limits(0, 5, 101, 30) == (0, 30) and native.band_limits(4, 5, 101, 30) == (70, 100)
    assert native.band_limits(2, 1, 50, 30) == (0, 49) and native.band_limits(2, 5, 50, -1) == (0, 49)


def test_zscore_matlab_semantics():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    assert np.allclose(native.zscore(x), (x - x.mean()) / x.std(ddof=1))
    assert np.array_equal(native.zscore(np.full(5, 3.2)), np.zeros(5))           # std 0 -> 1
    assert np.array_equal(native.zscore(np.array([4.0])), np.zeros(1))
    assert np.allclose(native.zscore(3.0 * x + 7.0), native.zscore(x))            # Eq. 3 (positive affine) is a no-op


# ---- pre-processing
def test_resample_and_matlab_smooth():
    tt = np.array([0.05, 0.12, 0.18, 0.35, 0.36, 0.71])
    vv = np.array([9.0, 1.0, 3.0, np.nan, 5.0, 7.0])
    tg, x = resample(tt, vv, 0.0, 0.8)
    assert np.allclose(tg, 0.1 * np.arange(1, 9))
    assert np.allclose(x, [9.0, 2.0, 2.0, 5.0, 5.0, 5.0, 5.0, 7.0])              # (t-0.1, t] means, held
    assert resample(tt, np.full(6, np.nan), 0.0, 0.8)[1] is None
    y = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0])
    assert np.allclose(smooth(y, 5), [1.0, 8 / 3, 19 / 5, 27 / 5, 26 / 5, 16 / 3, 4.0])
    assert np.allclose(smooth(y, 6), smooth(y, 5)) and np.array_equal(smooth(y, 1), y)


def test_turns_and_sections():
    t = np.arange(0.0, 30.0, 0.1)

    def bump(tc, amp, half=1.5):
        return amp * 0.5 * (1.0 + np.cos(np.pi * np.clip((t - tc) / half, -1.0, 1.0)))
    sw = bump(6.0, 300.0) + bump(14.0, 60.0, 3.0) + bump(21.4, -150.0, 0.8) + bump(22.6, -150.0, 0.8)
    turns = detect_turns(sw)
    assert [z[2] for z in turns] == [1, -1]                             # left, right; the 60 deg curve is no turn
    (a0, a1, _), (b0, b1, _) = turns
    assert abs(sw[a0]) >= 20.0 > abs(sw[a0 - 1]) and abs(sw[a1 - 1]) >= 20.0 > abs(sw[a1])
    assert 20.6 < t[b0] < 21.0 and 23.0 < t[b1 - 1] < 23.4              # two runs 0.5 s apart: one turn
    assert sections(len(sw), turns) == [(0, a0), (a1, b0), (b1, len(sw))]
    assert len(detect_turns(sw, merge_s=0.1)) == 2                      # the dip is above the ramp level: still merged
    assert detect_turns(np.zeros(50)) == []


# ---- roads
def test_strokes_follow_ways_and_mutual_straight_links():
    ways = [[(0.0, 0.0), (50.0, 0.0), (100.0, 0.0)], [(100.0, 0.0), (200.0, 10.0)], [(100.0, 0.0), (100.0, 100.0)],
            [(200.0, 10.0), (250.0, 60.0)], [(-50.0, 15.0), (50.0, 0.0)], [(0.0, -100.0), (50.0, -100.0), (50.0, -50.0)]]
    g = build(ways, oneway=(2,))
    r = _Roads(g, 1.0, 30.0)
    st, k = r.stroke_of(edge(g, (0, 0), (50, 0)))
    assert k == 0 and st.edges.tolist() == [edge(g, (0, 0), (50, 0)), edge(g, (50, 0), (100, 0)), edge(g, (100, 0), (200, 10))]
    assert st.bnd.tolist() == [0, 1, 2, 3]                              # (50, 0) and (100, 0) are intersections
    st_w, _ = r.stroke_of(edge(g, (200, 10), (100, 0)))                 # the other direction is its own stroke
    assert st_w.edges.tolist() == [edge(g, (200, 10), (100, 0)), edge(g, (100, 0), (50, 0)), edge(g, (50, 0), (0, 0))]
    e_ow = edge(g, (100, 0), (100, 100))
    assert r.stroke_of(e_ow)[0].edges.tolist() == [e_ow] and g.reverse[e_ow] == -1 and len(r.strokes) == 3
    assert len(r.stroke_of(edge(g, (200, 10), (250, 60)))[0].edges) == 1           # a 39 deg bend ends the stroke
    assert len(r.stroke_of(edge(g, (-50, 15), (50, 0)))[0].edges) == 1             # 8.5 deg, but (50, 0) continues way 0
    assert len(r.stroke_of(edge(g, (0, -100), (50, -100)))[0].edges) == 2          # the same way through a 90 deg corner
    for e in range(g.n_edges()):
        r.stroke_of(e)
    flat = np.concatenate([s.edges for s in r.strokes])
    assert sorted(flat.tolist()) == list(range(g.n_edges()))            # every directed edge on exactly one stroke


def test_closed_stroke_starts_at_its_smallest_edge():
    """A one-way roundabout (a closed way) is one closed stroke whichever edge is asked first: it starts at its smallest
    edge id, the exit road's node is a segment boundary, the entry road (its continuation into the ring is not mutual)
    is a stroke of its own, and every directed edge lies on exactly one stroke."""
    ring = arc((0.0, 0.0), 30.0, 0.0, 2 * math.pi, step=8.0)
    ring[-1] = ring[0]
    g = build([ring, [(100.0, 0.0), (30.0, 0.0)], [(0.0, 30.0), (0.0, 100.0)]], oneway=(0,))
    r = _Roads(g, 1.0, 30.0)
    st, k = r.stroke_of(10)
    assert st.edges.tolist() == list(range(24)) and k == 10 and g.u[st.edges[0]] == g.v[st.edges[-1]]
    assert np.allclose(g.node_xy[st.nodes[st.bnd]], [(30.0, 0.0), (0.0, 30.0), (30.0, 0.0)], atol=0.01)
    for e in range(g.n_edges()):
        r.stroke_of(e)
    assert len(r.stroke_of(edge(g, (100.0, 0.0), (30.0, 0.0)))[0].edges) == 1
    assert sorted(np.concatenate([s.edges for s in r.strokes]).tolist()) == list(range(g.n_edges()))


def test_curvy_road_is_one_stroke_with_signed_curvature():
    g = build(town())
    r = _Roads(g, 1.0, 30.0)
    st, k = r.stroke_of(edge(g, X1, (100.0, 25.0)))
    xy = g.node_xy[st.nodes[st.bnd]]
    assert k == 0 and np.allclose(xy, [X1, P2, X2], atol=0.01)          # three ways, segments end at P2 (road D) and X2
    kc = r.curv(st)
    assert len(kc) == int(st.t[-1]) + 1
    assert np.median(kc[70:120]) == pytest.approx(1 / 80.0, rel=0.05)    # left arc: positive (Eq. 2, left positive)
    assert np.median(kc[150:190]) == pytest.approx(-1 / 60.0, rel=0.05)  # right arc: negative
    back, _ = r.stroke_of(edge(g, X2, C[2][-2]))
    assert np.median(r.curv(back)[150:190]) == pytest.approx(1 / 60.0, rel=0.05)    # the right arc driven backwards
    t = np.array([0.0, 10.0, 10.0, 20.0])
    assert np.array_equal(curvature(np.array([[0, 0], [10, 0], [10, 0], [20, 0.0]]), t, 1.0), np.zeros(21))


# ---- end to end
def test_area_variant_recovers_the_route():
    ev, T, truth = drive_on(AREA_ROUTE)
    q = np.arange(1.0, T, 1.0)
    m = SPy(start="area")
    xy, theta, cands, probs = m.reconstruct(inputs(ev, T), q)
    err = np.hypot(*(xy - truth(q)).T)
    assert cands is None and probs is None and np.isfinite(xy).all() and xy.shape == (len(q), 2)
    assert m.stats["n_turns"] == 3 and m.stats["salient"] == 1 and m.stats["n_pieces"] == 4
    t3 = t_at(AREA_ROUTE, X3)                                           # the last turn
    assert np.median(err) < 15.0 and err[q < t3].max() < 40.0
    k_c = (q > 11.0) & (q < 40.0)                                       # on the curvy road (DTW drifts on the arcs'
    assert np.median(err[k_c]) < 10.0 and err[k_c].max() < 40.0         # constant-curvature plateaus)
    last = q >= t3                                                      # a straight last section: its free end has no
    assert (err[last] <= V * (q[last] - t3) + 15.0).all()               # curvature to time, it stays behind the car


@pytest.mark.parametrize("heading", [False, True])
def test_known_start_recovers_the_route(heading):
    ev, T, truth = drive_on(KNOWN_ROUTE)
    q = np.arange(1.0, T, 1.0)
    m = SPy(start="known", heading=heading)
    inp = inputs(ev, T, p0=(100.0, 20.0), theta0=math.pi / 2 if heading else None)
    xy, theta, _, _ = m.reconstruct(inp, q)
    tr = episode_xy((100.0, 20.0), truth(q))
    err = np.hypot(*(xy - tr).T)
    assert m.stats["n_pieces"] == 3 and m.stats["salient"] is None
    t3, last = t_at(KNOWN_ROUTE, X3), q >= t_at(KNOWN_ROUTE, X3)
    assert np.median(err) < 20.0 and err[~last].max() < 40.0 and (err[last] <= V * (q[last] - t3) + 15.0).all()
    assert abs(theta[-1] + math.pi / 2) < math.radians(1.0)            # southbound at the end (episode frame)


def test_far_drive_frame_and_visibility():
    """Graph in a drive frame 265 km west, viewed from the episode frame: positions and bearings come back in the
    episode frame (same output as the plain graph); a perfect copy of the curvy road beyond the tile border wins the
    area search only while it is visible."""
    ev, T, truth = drive_on(KNOWN_ROUTE)
    q = np.arange(1.0, T, 1.0)
    far = build(town(), frame=LocalFrame(LAT0, LON0 - 3.0))
    a = SPy(start="known").reconstruct(inputs(ev, T, p0=(100.0, 20.0)), q)
    b = SPy(start="known").reconstruct(inputs(ev, T, p0=(100.0, 20.0), graph=far), q)
    tr, early = episode_xy((100.0, 20.0), truth(q)), q < t_at(KNOWN_ROUTE, X3)
    for xy, theta in (a[:2], b[:2]):
        err = np.hypot(*(xy - tr).T)
        assert np.median(err) < 20.0 and err[early].max() < 40.0 and abs(theta[-1] + math.pi / 2) < math.radians(1.0)
    assert (np.hypot(*(a[0] - b[0]).T) < 1.0).mean() > 0.8              # warping may differ on curvature plateaus
    assert abs(far.bearing[edge(build(town()), X3, (230.0, P2[1]))] + math.pi / 2) > math.radians(1.0)  # same edge ids
    ev, T, _ = drive_on(AREA_ROUTE)
    q = np.arange(1.0, T, 1.0)
    g = build(town(with_c=False, decoy=True), frame=LocalFrame(LAT0, LON0 - 3.0))
    inp = inputs(ev, T, graph=g)
    inp.map_graph = EpisodeGraphView(g, LocalFrame(LAT0, LON0))
    xy_all, _, _, _ = SPy(start="area").reconstruct(inp, q)
    assert (xy_all[:, 1] < -800.0).mean() > 0.9                         # the copy of C, 1.2 km south
    inp.map_graph = EpisodeGraphView(g, LocalFrame(LAT0, LON0), visible_tiles={TILE})
    xy_vis, _, _, _ = SPy(start="area").reconstruct(inp, q)
    assert (np.abs(xy_vis) < 1000.0).all()                              # visible town only, in the episode frame


@pytest.mark.parametrize("heading", [False, True])
def test_known_start_inside_a_turn(heading):
    """A window starting inside the left turn at X1 (section 0 empty): the first section long enough is matched from p0,
    the turn taken at p0 and theta0 not used (a mid-turn heading), so the route and errors are those of a window
    starting on a straight (the first candidate run used to be taken unscored: a coin-flip direction, the turn a
    segment later, often one held position)."""
    ev, T, truth = drive_on(AREA_ROUTE)
    t1, t3 = t_at(AREA_ROUTE, X1), t_at(AREA_ROUTE, X3)                # the turn ramps over t1 -+ 1.5 s
    for t0, theta0 in ((t1 - 0.5, 0.0), (t1 + 0.3, math.pi / 4), (t1 + 0.9, math.pi / 2)):
        q = np.arange(math.floor(t0) + 1.0, T, 1.0)
        m = SPy(start="known", heading=heading)
        xy, _, _, _ = m.reconstruct(window(ev, t0, T, truth(np.array([t0]))[0], theta0 if heading else None), q)
        err = np.hypot(*(xy - truth(q)).T)
        secs = sections(len(m.sw), detect_turns(m.sw))
        assert secs[0][1] - secs[0][0] < 2 and m.stats["n_pieces"] == 3            # C, B and V2
        assert np.median(err) < 20.0 and err[q < t3].max() < 40.0


def test_window_edges_on_a_long_road():
    """Runs end at segment boundaries, windows do not. A window ending 15 s after a left turn onto a road whose next
    junction is 2 km away keeps its last positions within the distance driven since the turn (the closed alignment
    spread them over the whole 2 km segment); an area window starting 15 s before turning off that road starts near
    the truth (the closed alignment started it at the far junction)."""
    A = [(-600.0, -40.0), (-500.0, -40.0)] + arc((-500.0, 40.0), 80.0, -math.pi / 2, -math.pi / 2 + math.radians(30))[1:]
    A += [(-300.0, 0.0), (-150.0, 0.0), (0.0, 0.0), (150.0, 0.0)]
    g = build([A, [(0.0, 0.0), (0.0, 500.0), (0.0, 1000.0), (0.0, 1500.0), (0.0, 2000.0)],
               [(-300.0, 2000.0), (0.0, 2000.0), (300.0, 2000.0)]])
    route = A[:-1] + [(0.0, 150.0)]                                     # east on A, left at (0, 0), 150 m north
    ev, T, truth = drive_on(route)
    q, t_x = np.arange(1.0, T, 1.0), t_at(route, (0.0, 0.0))
    last = q > t_x
    for start, heading in (("known", False), ("known", True), ("area", False)):
        m = SPy(start=start, heading=heading)
        xy, _, _, _ = m.reconstruct(window(ev, 0.0, T, A[0], 0.0, g), q)
        err = np.hypot(*(xy - truth(q)).T)
        assert m.stats["n_pieces"] == 2 and (err[last] <= V * (q[last] - t_x) + 15.0).all()
        assert (xy[last, 1] < 200.0).all()                              # never towards the junction at y = 2000
    route = [(0.0, 150.0), (0.0, 0.0)] + A[1:-2][::-1] + [A[0]]         # south on that road, right at (0, 0) onto A
    ev, T, truth = drive_on(route)
    q = np.arange(1.0, T, 1.0)
    m = SPy(start="area")
    xy, _, _, _ = m.reconstruct(window(ev, 0.0, T, (0.0, 150.0), None, g), q)
    err = np.hypot(*(xy - truth(q)).T)
    assert m.stats["n_pieces"] == 2 and err[q < t_at(route, (0.0, 0.0)) - 2.0].max() < 30.0


def test_chain_needs_a_real_turn():
    """Across a detected left turn the chain only takes an out-edge turning by at least stroke_deg: a straight
    continuation bending 2 deg to the left never stands in for the turn (without a left road the chain stops), a real
    left road is taken."""
    J = (0.0, 0.0)
    ways = [[(-300.0, 0.0), (-150.0, 0.0), J, (150.0, 5.2)], [J, (0.0, -200.0)]]   # 2 deg left on; a road to the right
    for left in (False, True):
        g = build(ways + [[J, (0.0, 200.0)]] * left)
        roads = _Roads(g, 1.0, 30.0)
        st, _ = roads.stroke_of(edge(g, (-300.0, 0.0), (-150.0, 0.0)))
        k1 = int(np.flatnonzero(np.hypot(*(g.node_xy[st.nodes] - J).T) < 1e-6)[0])
        first = _Piece(st, 0, 0.0, k1, (0, int(st.t[k1]), 30))
        first.sec = 0
        pieces = SPy(start="known")._chain(g, roads, first, [(0, 30), (40, 41)], [(30, 40, 1)])  # 1 sample: no DTW
        assert len(pieces) == 1 + left
        assert not left or pieces[1].st.edges[pieces[1].ke] == edge(g, J, (0.0, 200.0))


def test_speed_is_never_read_and_runs_repeat():
    ev, T, _ = drive_on(AREA_ROUTE)
    ev2, _, _ = drive_on(AREA_ROUTE, speed=lambda t: 3.0 + 20.0 * np.abs(np.sin(t)))
    q = np.arange(1.0, T, 1.0)
    for start in ("area", "known"):
        a = SPy(start=start).reconstruct(inputs(ev, T), q)
        b = SPy(start=start).reconstruct(inputs(ev2, T), q)
        c = SPy(start=start).reconstruct(inputs(ev, T), q)
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
        assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])


def test_python_fallback_gives_the_same_route(monkeypatch):
    """Without the compiled kernel an episode is an error (the reference is ~1e3 times slower: no silent fallback),
    unless CANTRAJ_PY_DTW=1 allows the reference, which gives the same output bit for bit."""
    ev, T, _ = drive_on(KNOWN_ROUTE)
    q = np.arange(1.0, T, 1.0)
    a = SPy(start="known", heading=True).reconstruct(inputs(ev, T, p0=(100.0, 20.0), theta0=math.pi / 2), q)
    monkeypatch.setattr(native, "_load", lambda: None)                  # as without a compiler
    monkeypatch.delenv("CANTRAJ_PY_DTW", raising=False)
    with pytest.raises(RuntimeError, match="native DTW kernel unavailable") as e:
        SPy(start="known", heading=True).reconstruct(inputs(ev, T, p0=(100.0, 20.0), theta0=math.pi / 2), q)
    assert type(e.value) is RuntimeError                                # not a method failure (SPyNoMatch)
    monkeypatch.setenv("CANTRAJ_PY_DTW", "1")
    m = SPy(start="known", heading=True)
    b = m.reconstruct(inputs(ev, T, p0=(100.0, 20.0), theta0=math.pi / 2), q)
    assert not m.stats["native"] and np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_budget_and_signal_failures():
    ev, T, _ = drive_on(AREA_ROUTE)
    q = np.arange(1.0, T, 1.0)
    m = SPy(start="area", max_cells=20000)
    with pytest.raises(SPyNoMatch, match="budget reached"):
        m.reconstruct(inputs(ev, T), q)
    assert m.stats["budget_hit"] and m.stats["cells"] >= 20000
    m = SPy(start="area", timeout_s=1e-9)                               # the wall-clock safety net
    with pytest.raises(SPyNoMatch, match="budget reached .timeout"):
        m.reconstruct(inputs(ev, T), q)
    assert m.stats["budget_hit"]
    with pytest.raises(SPyNoMatch, match="no road in the area square"):
        SPy(start="area").reconstruct(inputs(ev, T, p0=(0.0, 9000.0)), q)
    with pytest.raises(SPyNoMatch, match="no road within 400 m"):
        SPy(start="known").reconstruct(inputs(ev, T, p0=(0.0, 9000.0)), q)
    ev_k, T_k, _ = drive_on(KNOWN_ROUTE)                                # a budget reached inside the chain: just above
    q_k, cells = np.arange(1.0, T_k, 1.0), []                           # what the known start takes
    m = SPy(start="known")
    known = m._known

    def known_then_count(*a):
        p = known(*a)
        cells.append(m._cells)
        return p
    m._known = known_then_count
    m.reconstruct(inputs(ev_k, T_k, p0=(100.0, 20.0)), q_k)
    m = SPy(start="known", max_cells=cells[0] + 1)
    chain = m._chain

    def chain_entered(*a):
        cells.append(None)
        return chain(*a)
    m._chain = chain_entered
    with pytest.raises(SPyNoMatch, match="budget reached"):
        m.reconstruct(inputs(ev_k, T_k, p0=(100.0, 20.0)), q_k)
    assert len(cells) == 2 and m.stats["budget_hit"]
    no_steer = EventStream(ev.t_s, {"speed": ev.cols["speed"]})
    with pytest.raises(SPyNoMatch, match="no steering angle signal"):
        SPy().reconstruct(inputs(no_steer, T), q)
    flat = EventStream(ev.t_s, {"speed": ev.cols["speed"], "steer_angle": np.full(len(ev), 1.5)})
    with pytest.raises(SPyNoMatch, match="constant steering angle"):
        SPy(start="known").reconstruct(inputs(flat, T), q)
    turning = EventStream(ev.t_s, {"speed": ev.cols["speed"], "steer_angle": 200.0 + 10.0 * np.sin(ev.t_s)})
    for start in ("area", "known"):                                     # one turn over the whole window
        with pytest.raises(SPyNoMatch, match="no section long enough"):
            SPy(start=start).reconstruct(inputs(turning, T), q)
    with pytest.raises(ValueError):
        SPy(start="known", heading=True).reconstruct(inputs(ev, T), q)
