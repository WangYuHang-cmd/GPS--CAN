"""Dewri et al. (WPES 2013) re-implementation on synthetic maps: the 1 Hz samples (CAN per second from t0 to the window
end, whatever the reference fixes), the pre-processing rules (half-mile stop-and-go filter, 3 s rule, final stop), the
augmented speed model and likelihood in closed form, the true route found and ranked first when the speeds follow its
limits, the 25 mph turn rule (edges under 1 m carry the heading), a stop just before the window end, the 0.1 mile
shortest-path slack (also at the end node) and its highway exception, the budget (shared by the start edges in equal
shares, also on a 30 x 30 grid in both travel directions; accepted paths are kept), a trip ending on its start edge, the
search against a recursive DFS, determinism, far drive frames (an invisible shortcut, the heading filter), and speed as
the only signal read."""
import heapq
import math

import numpy as np
import pandas as pd
import pytest

from cantraj.baselines.dewri import MPH, Dewri, DewriNoPath, aug_model, log_likelihood, one_hz, preprocess
from cantraj.events import EventStream
from cantraj.geo import LocalFrame
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.maps.graph import DriveGraph, EpisodeGraphView

LAT0, LON0 = 37.0, -122.0
DEC, ACC = 25.0 * 0.3048, 6.5 * 0.3048          # the paper's 25 and 6.5 ft/s^2 in m/s^2


def _graph(xy, ways, origin=(LAT0, LON0), frame=None):
    """DriveGraph of ways [(highway, [node names]) or (highway, [node names], oneway)] (two-way unless given) over nodes
    {name: (x, y)} given in metres in the frame at `origin`, built in `frame` (default: the origin frame, i.e. a raw
    graph in the episode frame)."""
    names = list(xy)
    lat, lon = LocalFrame(*origin).to_lla(np.array([float(xy[k][0]) for k in names]), np.array([float(xy[k][1]) for k in names]))
    ids = {k: i for i, k in enumerate(names)}
    rows = [dict(way_id=w, highway=wy[0], oneway=wy[2] if len(wy) > 2 else 0, junction="", maxspeed="", lanes="", name_len=0,
                 n_nodes=len(wy[1])) for w, wy in enumerate(ways)]
    wn = [dict(way_id=w, seq=s, node_id=ids[k]) for w, wy in enumerate(ways) for s, k in enumerate(wy[1])]
    return DriveGraph(pd.DataFrame(rows), pd.DataFrame(wn), pd.DataFrame(dict(node_id=np.arange(len(names)), lat=lat, lon=lon)),
                      pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"]), frame or LocalFrame(*origin))


def _grid(n=31, block=100.0, rows=None, cols=None):
    """n x n grid of two-way ways centred on the origin; rows / cols give the highway class of the way at y / x (m)."""
    h = n // 2
    xy = {(i, j): ((j - h) * block, (i - h) * block) for i in range(n) for j in range(n)}
    ways = [((rows or {}).get((i - h) * block, "residential"), [(i, j) for j in range(n)]) for i in range(n)]
    ways += [((cols or {}).get((j - h) * block, "residential"), [(i, j) for i in range(n)]) for j in range(n)]
    return _graph(xy, ways)


def _tee():
    """An east-west road (nodes every 100 m, x = -600..900) with a side road south from (300, 0) to (300, -700)."""
    xy = {("r", x): (float(x), 0.0) for x in range(-600, 1000, 100)} | {("s", y): (300.0, float(y)) for y in range(-100, -800, -100)}
    return _graph(xy, [("residential", [("r", x) for x in range(-600, 1000, 100)]),
                       ("residential", [("r", 300)] + [("s", y) for y in range(-100, -800, -100)])])


def _detour(cls):
    """A residential road W-A-B-...-H along y = 0 and a second way A-H1-H2-B of class `cls` bending north (45 deg bends,
    no turn above 60 deg), 331 m longer than A-B."""
    xy = {"W": (-100, 0), "A": (100, 0), "B": (1100, 0), "C": (1400, 0), "E": (1700, 0), "F": (2000, 0), "G": (2300, 0),
          "H": (2600, 0), "H1": (500, 400), "H2": (700, 400)}
    return _graph(xy, [("residential", ["W", "A", "B", "C", "E", "F", "G", "H"]), (cls, ["A", "H1", "H2", "B"])])


def _profile(legs, events):
    """Speed (m/s) at path distance d of a driver at 1.1 x the limit (the mean of the paper's f; legs = [(end m, limit
    mph)]) who slows at the paper's rates to the speed (mph) of every event [(d m, mph)] (turns, stops)."""
    def prof(d):
        lim = next((sp for end, sp in legs if d < end), legs[-1][1])
        v = (1.1 * lim if lim >= 20.0 else lim) * MPH
        for x, ve in events:
            v = min(v, math.sqrt((ve * MPH) ** 2 + 2.0 * (DEC if d <= x else ACC) * abs(x - d)))
        return v
    return prof


def _drive(prof, stops, d_end=math.inf, t_end=math.inf, dwell=10.0, dt=0.01):
    """100 Hz speed of a driver following prof(d), halting `dwell` s at each distance in `stops`, until d_end or t_end."""
    v, x, k, wait = [], 0.0, 0, 0.0
    while x < d_end and len(v) * dt < t_end:
        if wait > 0:
            sp, wait = 0.0, wait - dt
        else:
            sp = max(prof(x), 0.3)
            if k < len(stops) and x + sp * dt >= stops[k]:
                sp, wait, k = (stops[k] - x) / dt, dwell, k + 1
        v.append(sp)
        x += sp * dt
    return np.arange(len(v)) * dt, np.array(v)


def _dist(t, v, q):
    return np.interp(q, t, np.concatenate([[0.0], np.cumsum(v[:-1])]) * 0.01)


def _along(route, s):
    """Points at distances s (m) along the polyline route."""
    seg = np.hypot(*np.diff(route, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    k = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    return route[k] + ((np.clip(s, 0.0, cum[-1]) - cum[k]) / seg[k])[:, None] * (route[k + 1] - route[k])


def _err(xy, truth):
    return float(np.median(np.hypot(*(xy - truth).T)))


def _inputs(t, v, mg, p0=(0.0, 0.0), theta0=None, gid="G1", cols=None, origin=(LAT0, LON0)):
    ev = EventStream(t, {"speed": v} | (cols or {}))
    st = StaticParams(wheelbase_m=2.7, steer_ratio=15.0) if gid == "G3" else StaticParams()
    inp = EpisodeInputs(gid, ev, st, EpisodeInit(0.0, origin[0], origin[1], theta0, t_end=float(t[-1]), dx_m=p0[0], dy_m=p0[1]), input_layer="events")
    inp.map_graph = mg
    return inp


def _main_trip():
    """Secondary road east from p0 = (50, 0), stop at (1000, 0), left onto the primary road x = 1000, stop at (1000, 1000),
    right onto the tertiary road y = 1000: stops 1000 m apart (beyond the half-mile rule), every other way residential."""
    g = _grid(rows={0.0: "secondary", 1000.0: "tertiary"}, cols={1000.0: "primary"})
    t, v = _drive(_profile([(950.0, 35.0), (1950.0, 45.0), (math.inf, 35.0)], [(950.0, 0.0), (1950.0, 0.0)]), [950.0, 1950.0], d_end=2340.0)
    return g, t, v, np.array([(50.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (1500.0, 1000.0)])


def _samples(pieces):
    """1 Hz samples from [(seconds, mph)] pieces: t (s), speed (mph), distance (m)."""
    s = np.concatenate([np.full(n, float(sp)) for n, sp in pieces])
    return np.arange(1.0, len(s) + 1.0), s, np.cumsum(s * MPH)


def _record(monkeypatch):
    """Spy on the search and the ranking of every run: {end node xy: turns} of the kept paths and {end node xy:
    log-likelihood} of the scored ones (raw graphs: node_xy is the episode frame)."""
    rec = dict(turns={}, ll={})
    dfs, score = Dewri._dfs, Dewri._score

    def dfs_spy(self, mg, g0, *a):
        out = dfs(self, mg, g0, *a)
        rec["turns"], rec["ll"] = {tuple(np.round(g0.node_xy[x], 1)): b[0] for x, b in out[0].items()}, {}
        return out

    def score_spy(self, g0, edges, *a):
        ll = score(self, g0, edges, *a)
        rec["ll"][tuple(np.round(g0.node_xy[g0.v[edges[-1]]], 1))] = ll
        return ll
    monkeypatch.setattr(Dewri, "_dfs", dfs_spy)
    monkeypatch.setattr(Dewri, "_score", score_spy)
    return rec


def test_one_hz_samples_and_distance():
    """Speed = mean of the finite CAN samples in each interval (one without any holds the previous mean, leading ones
    the first); distance from t0 = trapezoid rule on the raw events (NaN holds the last finite speed, the held speed
    before the first and after the last event), one odometer for the sample and the output times."""
    tt = np.array([1.5, 1.8, 2.0, 2.5, 4.2, 5.0])
    vv = np.array([np.nan, 2.0, 4.0, np.nan, 6.0, np.nan])
    S, dg, d = one_hz(tt, vv, 0.0, np.arange(1.0, 7.0), np.array([1.5, 6.0]))
    assert np.allclose(S, [3.0, 3.0, 3.0, 3.0, 6.0, 6.0]) and np.allclose(d, [3.0, 25.5])
    assert np.allclose(dg, [2.0, 4.2, 8.7, 13.7, 19.5, 25.5])
    S, dg, _ = one_hz(tt, vv, 0.0, np.array([2.0, 4.0, 6.0]), [])
    assert np.allclose(S, [3.0, 3.0, 6.0]) and np.allclose(dg, [4.2, 13.7, 25.5])
    with pytest.raises(DewriNoPath, match="no speed samples"):
        one_hz(tt, np.full(len(tt), np.nan), 0.0, np.arange(1.0, 7.0), [])


def test_half_mile_rule_deletes_stop_and_go():
    """Stops 402 m apart are both deleted with everything between them (chains of such stops merge); a stop 939 m on
    survives as a stop-point."""
    t, s, d = _samples([(30, 30), (10, 0), (30, 30), (10, 0), (70, 30), (10, 0), (30, 30)])
    keep, stops, n_req = preprocess(t, s, d)
    assert not keep[30:80].any() and keep[:30].all() and keep[80:].all()
    assert len(stops) == 1 and stops[0] == pytest.approx(d[150]) and n_req == 1
    t, s, d = _samples([(30, 30), (5, 0), (20, 30), (5, 0), (20, 30), (5, 0), (80, 30)])
    keep, stops, _ = preprocess(t, s, d)
    assert not keep[30:85].any() and keep[:30].all() and keep[85:].all() and len(stops) == 0
    keep, stops, _ = preprocess(t, s, d, traffic_mi=0.1)                     # the threshold is a parameter
    assert keep.all() and len(stops) == 3


def test_three_second_rule_and_final_stop():
    """A halt spanning 2 s (3 samples) is deleted, one spanning 3 s is a stop-point; the last stop-point is left
    unmatched only when its zero run reaches the window end."""
    t, s, d = _samples([(30, 30), (3, 0), (70, 30), (4, 0), (30, 30)])
    keep, stops, n_req = preprocess(t, s, d)
    assert not keep[30:33].any() and keep[103:107].all()
    assert len(stops) == 1 and stops[0] == pytest.approx(d[103]) and n_req == 1
    t, s, d = _samples([(30, 30), (10, 0), (70, 30), (10, 0)])
    assert preprocess(t, s, d)[2] == 1 and len(preprocess(t, s, d)[1]) == 2
    t, s, d = _samples([(30, 30), (10, 0), (70, 30), (10, 0), (5, 30)])
    assert preprocess(t, s, d)[2] == 2


def test_augmented_model_and_likelihood_closed_form():
    """Braking from 60 mph to a halt takes 155 ft at 25 ft/s^2; 100 ft after a left turn the model is back to
    sqrt(22^2 + 2 * 6.5 * 100) ft/s; f is N(66, 2) on a 60 mph road and N(15, 1) on a 15 mph one (eps = 1e-5)."""
    cum, x = np.array([0.0, 1000.0]), 500.0
    d = np.array([x - 154.88 * 0.3048, x, x + 100 * 0.3048, 0.0])
    assert np.allclose(aug_model(d, cum, [60.0], [], [], [x]), [60.0, 0.0, math.sqrt(1300.0) * 0.3048 / MPH, 60.0], atol=1e-6)
    assert np.allclose(aug_model(d, cum, [60.0], [x], [15.0], []), [60.0, 15.0, math.sqrt(484.0 + 1300.0) * 0.3048 / MPH, 60.0], atol=1e-6)
    c = math.log(2e-5) - 0.5 * math.log(2 * math.pi)
    assert log_likelihood(np.array([66.0]), np.array([60.0])) == pytest.approx(c - math.log(2.0))
    assert log_likelihood(np.array([15.0, 16.0]), np.array([15.0, 15.0])) == pytest.approx(2 * c - 0.5)
    assert np.isfinite(log_likelihood(np.array([80.0]), np.array([20.0])))       # no underflow to -inf


def test_true_route_is_found_and_ranked_first():
    g, t, v, route = _main_trip()
    q = np.arange(1.0, t[-1], 1.0)
    m = Dewri()
    xy, theta, cands, probs = m.reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    truth = _along(route, _dist(t, v, q))
    assert cands is not None and 1 < len(cands) <= 10 and np.array_equal(cands[0], xy) and probs.sum() == pytest.approx(1.0)
    assert _err(xy, truth) < 5.0 and np.hypot(*(xy[-1] - truth[-1])) < 10.0
    assert np.isfinite(xy).all() and np.isfinite(theta).all()
    assert m.stats["n_stops"] == 2 and m.stats["n_required"] == 2 and not m.stats["budget_hit"]


def test_samples_do_not_depend_on_reference_gaps():
    """The <t, s, d> samples are one per second of CAN from t0 to the window end, wherever the reference fixes are. A
    20 s hole in t_query over a right turn taken at 10 mph (a sample at the hole's end would hold its mean speed, above
    25 mph, and prune the turn) or over one of two 10 s mid-road stops 0.4 mi apart (the other one would survive the
    half-mile rule as a required stop-point with no intersection near it), or the last 20 fixes missing (the trip length
    is still the CAN distance at the window end), leaves the candidates and their probabilities unchanged at the
    remaining times."""
    xs = list(range(-200, 3001, 100))
    road = _graph({("r", x): (float(x), 0.0) for x in xs} | {"a": (0.0, 300.0), "b": (2600.0, 300.0)},
                  [("residential", [("r", x) for x in xs]), ("residential", [("r", 0), "a"]), ("residential", [("r", 2600), "b"])])
    turn = (_tee(), [(50.0, 0.0), (300.0, 0.0), (300.0, -700.0)], [(250.0, 10.0)], [], 650.0, 250.0)
    stops = (road, [(50.0, 0.0), (3000.0, 0.0)], [(700.0, 0.0), (1344.0, 0.0)], [700.0, 1344.0], 2200.0, 1344.0)
    for g, route, events, halts, d_end, d_hole in (turn, stops):
        t, v = _drive(_profile([(math.inf, 25.0)], events), halts, d_end=d_end)
        q = np.arange(1.0, t[-1], 1.0)
        at = t[np.abs(_dist(t, v, t) - d_hole) < 1.0]                      # the car at the turn node / the stop
        k, mid = int(np.argmin(np.abs(q - 0.5 * (at[0] + at[-1])))), np.ones(len(q), dtype=bool)
        mid[k - 9:k + 10] = False                                          # 19 fixes missing: 20 s between fixes
        full = Dewri().reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
        assert _err(full[0], _along(np.array(route), _dist(t, v, q))) < 5.0 and full[2] is not None
        for sel in (mid, np.arange(len(q)) < len(q) - 20):
            hole = Dewri().reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q[sel])
            assert np.array_equal(hole[0], full[0][sel]) and np.array_equal(hole[1], full[1][sel])
            assert np.array_equal(hole[2], full[2][:, sel]) and np.array_equal(hole[3], full[3])


def test_turn_rule_prunes_fast_turns():
    """East from p0 = (50, 0), right turn at (300, 0), south: a candidate when the car slows to 10 mph at the turn; at
    a constant 27 mph the turn is infeasible (not under 25 mph) and the route is pruned; turn_mph = 30 lets it back."""
    g = _tee()
    route = np.array([(50.0, 0.0), (300.0, 0.0), (300.0, -700.0)])
    slow = _drive(_profile([(math.inf, 25.0)], [(250.0, 10.0)]), [], d_end=650.0)
    fast = _drive(lambda d: 12.0, [], d_end=650.0)
    for (t, v), kw, found in ((slow, {}, True), (fast, {}, False), (fast, dict(turn_mph=30.0), True)):
        q = np.arange(1.0, t[-1], 1.0)
        xy, _, cands, _ = Dewri(**kw).reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
        allc = cands if cands is not None else xy[None]
        assert (min(_err(c, _along(route, _dist(t, v, q))) for c in allc) < 5.0) == found


def test_short_edges_carry_the_heading(monkeypatch):
    """North-bound residential road a-b-b2-c-d crossed by a side road at b. A second node b2 at b's position (the
    zero-length edge b-b2 reads bearing 0, east) or 0.5 m east of it has no heading of its own, so no turn is counted at
    b or b2: at a constant 33 mph (every turn infeasible) the route is found, and at 24 mph (turns allowed) it keeps no
    turn and scores exactly as on the road without b2."""
    rec = _record(monkeypatch)
    nodes = {"a": (0.0, -500.0), "b": (0.0, 0.0), "c": (0.0, 500.0), "d": (0.0, 1000.0), "w": (-300.0, 0.0), "e": (300.0, 0.0)}
    for mph in (33.0, 24.0):
        t, v = _drive(lambda d: mph * MPH, [], d_end=1480.0)
        q = np.arange(1.0, t[-1], 1.0)
        out = []
        for b2 in (None, (0.0, 0.0), (0.5, 0.0)):
            g = _graph(nodes | ({"b2": b2} if b2 else {}), [("residential", ["a", "b"] + (["b2"] if b2 else []) + ["c", "d"]),
                                                            ("residential", ["w", "b", "e"])])
            xy, _, cands, _ = Dewri().reconstruct(_inputs(t, v, g, p0=(0.0, -490.0)), q)
            assert cands is None and rec["turns"] == {(0.0, 1000.0): 0}
            out.append((xy, rec["ll"][(0.0, 1000.0)]))
        assert _err(out[0][0], _along(np.array([(0.0, -490.0), (0.0, 1000.0)]), _dist(t, v, q))) < 1.0
        assert all(np.hypot(*(xy - out[0][0]).T).max() < 1.0 and ll == out[0][1] for xy, ll in out[1:])


def test_split_corner_is_one_turn(monkeypatch):
    """A 90 deg corner: north on a-b, then east (right) or west (left) on b-e1-e2-e3, the corner node split into a
    co-located pair b, b2 on the turning way. It counts as one turn of its side (10 or 15 mph in M_aug): a car slowing
    there keeps the route with one turn and the same log-likelihood as on the unsplit corner, and at a constant 33 mph
    the turn is pruned."""
    rec = _record(monkeypatch)
    for side, v_turn in ((1.0, 10.0), (-1.0, 15.0)):
        end, ll = (side * 900.0, 0.0), []
        slow = _drive(_profile([(math.inf, 25.0)], [(590.0, v_turn)]), [], d_end=1290.0)
        fast = _drive(lambda d: 33.0 * MPH, [], d_end=1290.0)
        for split in (False, True):
            nodes = {"a": (0.0, -600.0), "b": (0.0, 0.0), "n1": (0.0, 300.0), "n2": (0.0, 900.0), "e1": (side * 300.0, 0.0),
                     "e2": (side * 600.0, 0.0), "e3": end} | ({"b2": (0.0, 0.0)} if split else {})
            g = _graph(nodes, [("residential", ["a", "b", "n1", "n2"]), ("residential", ["b"] + (["b2"] if split else []) + ["e1", "e2", "e3"])])
            for (t, v), slows in ((slow, True), (fast, False)):
                Dewri().reconstruct(_inputs(t, v, g, p0=(0.0, -590.0)), np.arange(1.0, t[-1], 1.0))
                assert (end in rec["turns"]) == slows
                if slows:
                    assert rec["turns"][end] == 1
                    ll.append(rec["ll"][end])
        assert ll[0] == ll[1]


def test_shortest_path_slack_prunes_a_detour():
    """The car takes the residential detour A-H1-H2-B, 331 m longer than A-B: beyond the 0.1 mile slack the detour is
    pruned (only the direct road remains); with a 0.5 mile slack it is a candidate."""
    g = _detour("residential")
    route = np.array([(0.0, 0.0), (100.0, 0.0), (500.0, 400.0), (700.0, 400.0), (1100.0, 0.0), (2600.0, 0.0)])
    t, v = _drive(_profile([(math.inf, 25.0)], []), [], d_end=1890.0)
    q = np.arange(1.0, t[-1], 1.0)
    truth = _along(route, _dist(t, v, q))
    for kw, found in (({}, False), (dict(sp_slack_mi=0.5), True)):
        xy, _, cands, _ = Dewri(**kw).reconstruct(_inputs(t, v, g), q)
        allc = cands if cands is not None else xy[None]
        assert (min(_err(c, truth) for c in allc) < 5.0) == found


def test_highway_resets_the_shortest_path_reference():
    """The same detour as a motorway: the shortest-path check is off along it and restarts at the exit B, so the detour
    survives, and with speeds that follow the limits (66 mph on the motorway) it ranks above the direct road."""
    g = _detour("motorway")
    route = np.array([(0.0, 0.0), (100.0, 0.0), (500.0, 400.0), (700.0, 400.0), (1100.0, 0.0), (2600.0, 0.0)])
    t, v = _drive(_profile([(100.0, 25.0), (1431.4, 60.0), (math.inf, 25.0)], []), [], d_end=1890.0)
    q = np.arange(1.0, t[-1], 1.0)
    xy, _, cands, probs = Dewri().reconstruct(_inputs(t, v, g), q)
    assert cands is not None and len(cands) == 2 and probs[0] > probs[1]
    assert _err(xy, _along(route, _dist(t, v, q))) < 5.0


def test_shortest_path_check_at_the_end_node(monkeypatch):
    """The end node gets the 0.1 mile check like every node: on W-A-B-C-E with a road C-Z-Z2 north and a direct road
    A-Z (854 m), a trip of 996 m (turns allowed) ends W-A-B-C-Z at Z, 1100 m from A, and W-A-Z-C at C, 1154 m from A
    against 800 m along the road: both are beyond the slack, so only the paths ending at E and Z2 remain; with a
    0.5 mile slack the paths ending at Z and C are candidates too."""
    rec = _record(monkeypatch)
    g = _graph({"W": (-100, 0), "A": (100, 0), "B": (500, 0), "C": (900, 0), "E": (1300, 0), "Z": (900, 300), "Z2": (900, 700)},
               [("residential", ["W", "A", "B", "C", "E"]), ("residential", ["C", "Z", "Z2"]), ("residential", ["A", "Z"])])
    t, v = _drive(lambda d: 12.0, [], d_end=1000.0)
    q = np.arange(1.0, t[-1], 1.0)
    for kw, ends in (({}, {(1300.0, 0.0), (900.0, 700.0)}), (dict(sp_slack_mi=0.5), {(1300.0, 0.0), (900.0, 700.0), (900.0, 300.0), (900.0, 0.0)})):
        m = Dewri(turn_mph=100.0, **kw)
        m.reconstruct(_inputs(t, v, g), q)
        assert set(rec["turns"]) == ends and m.stats["n_end"] == 4


def test_stop_just_before_the_window_end():
    """East along y = 0 from p0 = (-550, 0) with a 10 s halt at x = 280, 20 m before the T-junction at x = 300 (the only
    intersection), the window ending 3 m after the halt. With a crossing node at x = 290 the true path ends there, before
    the junction: the stop's 500 ft window is still open, so it is not required yet and the true path is the top
    candidate, as without the crossing node; the same with p0 in the middle of a 500 m edge (no start edge 50 m ahead to
    fall back on)."""
    t, v = _drive(_profile([(math.inf, 25.0)], [(830.0, 0.0)]), [830.0], d_end=836.0)
    q = np.arange(1.0, t[-1], 1.0)
    truth = _along(np.array([(-550.0, 0.0), (900.0, 0.0)]), _dist(t, v, q))
    ys = list(range(-100, -800, -100))
    for xs in (sorted(list(range(-600, 1000, 100)) + [290]), [-900, -400, 100, 290, 300, 400, 900], list(range(-600, 1000, 100))):
        g = _graph({("r", x): (float(x), 0.0) for x in xs} | {("s", y): (300.0, float(y)) for y in ys},
                   [("residential", [("r", x) for x in xs]), ("residential", [("r", 300)] + [("s", y) for y in ys])])
        m = Dewri()
        xy, _, _, _ = m.reconstruct(_inputs(t, v, g, p0=(-550.0, 0.0)), q)
        assert m.stats["n_required"] == 1 and 830.0 < m.stats["trip_m"] < 840.0 and _err(xy, truth) < 5.0


def test_budget():
    """Too small a budget without a candidate is a failure (the expansions shared by the start edges, or the time
    safety net); a budget hit after candidates were found keeps them."""
    g, t, v, _ = _main_trip()
    q = np.arange(1.0, t[-1], 1.0)
    with pytest.raises(DewriNoPath, match="budget reached"):
        Dewri(max_expand=3).reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    with pytest.raises(DewriNoPath, match=r"budget reached \(time\)"):
        Dewri(timeout_s=0.0).reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    m = Dewri()
    m.reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    m2 = Dewri(max_expand=m.stats["expansions"] - 1)
    xy, _, _, _ = m2.reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    assert m2.stats["budget_hit"] and 0 < m2.stats["n_cands"] <= m.stats["n_cands"] and np.isfinite(xy).all()


def test_budget_keeps_accepted_paths_on_the_stack():
    """One-way road W-J-E-E2 east and a one-way branch J-S (400 m, 30 deg left), p0 = (0, 0), trip 340 m. The first
    expansion (at J) accepts the path to S (past the trip length) and leaves J-E above it on the stack: an allowance of
    one expansion stops the search there, and the accepted path is still collected. The full search also finds the
    straight route, ranked first (equal log-likelihoods keep the depth-first order)."""
    g = _graph({"W": (-200.0, 0.0), "J": (100.0, 0.0), "E": (300.0, 0.0), "E2": (1300.0, 0.0),
                "S": (100.0 + 400.0 * math.cos(math.radians(30.0)), 400.0 * math.sin(math.radians(30.0)))},
               [("residential", ["W", "J", "E", "E2"], 1), ("residential", ["J", "S"], 1)])
    t, v = _drive(lambda d: 12.0, [], d_end=350.0)
    q = np.arange(1.0, t[-1], 1.0)
    m = Dewri(max_expand=1)
    xy, _, cands, _ = m.reconstruct(_inputs(t, v, g), q)
    assert m.stats["budget_hit"] and m.stats["root_expansions"] == [1] and m.stats["n_end"] == 1 and cands is None
    assert xy[-1, 1] > 100.0                                                    # on J-S
    xy, _, cands, _ = Dewri().reconstruct(_inputs(t, v, g), q)
    assert len(cands) == 2 and _err(xy, _along(np.array([(0.0, 0.0), (1300.0, 0.0)]), _dist(t, v, q))) < 1.0


def test_grid_search_is_bounded_by_the_turn_rule():
    """30 x 30 grid of residential ways (100 m blocks), 300 s: east from p0 = (-1450, -1300), a 20 s stop at x = -700,
    left at x = -300 (15 mph), a 20 s stop 900 m north (1300 m after the first), right at y = 0 (10 mph), then east. At
    1.1 x the limit (27.5 mph) a turn is feasible only near the slow spots: the true route is ranked first after a few
    hundred expansions. Below 25 mph every node may turn: the staircase routes within the 0.1 mile slack exhaust the
    default budget (70 expansions per second of the window, shared by the start edges) and the candidates found so far
    are kept."""
    g = _grid(n=30)
    route = np.array([(-1450.0, -1300.0), (-300.0, -1300.0), (-300.0, 0.0), (1400.0, 0.0)])
    events = [(750.0, 0.0), (1150.0, 15.0), (2050.0, 0.0), (2450.0, 10.0)]
    for lim, bounded in ((25.0, True), (20.0, False)):
        t, v = _drive(_profile([(math.inf, lim)], events), [750.0, 2050.0], t_end=300.0, dwell=20.0)
        q = np.arange(1.0, t[-1], 1.0)
        m = Dewri()
        xy, _, cands, _ = m.reconstruct(_inputs(t, v, g, p0=(-1450.0, -1300.0)), q)
        assert np.isfinite(xy).all() and m.stats["n_required"] == 2 and cands is not None
        assert m.stats["n_roots_expanded"] == m.stats["n_roots"]
        if bounded:
            assert _err(xy, _along(route, _dist(t, v, q))) < 5.0
            assert not m.stats["budget_hit"] and m.stats["expansions"] < 1000
        else:
            assert m.stats["budget_hit"] and m.stats["expansions"] == 70 * len(q)


def test_budget_is_shared_by_the_start_edges():
    """The grid trip above turned by 180 deg about the grid centre: west-bound from p0 = (1350, 1200). The two
    directions of the start road are exactly as close to p0 and come first in KD-tree order, not travel order. The start
    edges share the budget in equal shares, so a binding budget (20 mph) cannot starve one of them: every start edge is
    searched with the same share and the candidates do not depend on the order of the tied pair; with the default
    budget times the number of start edges each one gets 70 expansions per second of the window and the true route is
    among the candidates. Without a budget hit (25 mph) the first start edge needs far fewer expansions than the second:
    the share it leaves unused passes on, so the smallest budget that lets every search finish is below the number of
    start edges times the largest need, and one expansion less stops a search."""
    g = _grid(n=30)
    route = np.array([(1350.0, 1200.0), (200.0, 1200.0), (200.0, -100.0), (-1500.0, -100.0)])
    events = [(750.0, 0.0), (1150.0, 15.0), (2050.0, 0.0), (2450.0, 10.0)]
    near = g.nearest_edges
    es = near(1350.0, 1200.0, 50.0)
    assert g.reverse[es[0]] == es[1] and g.seg_distance(es[:2], 1350.0, 1200.0)[1] == g.seg_distance(es[:2], 1350.0, 1200.0)[0]
    t, v = _drive(_profile([(math.inf, 20.0)], events), [750.0, 2050.0], t_end=300.0, dwell=20.0)
    q = np.arange(1.0, t[-1], 1.0)
    out = []
    for swap in (False, True):
        if swap:                                                                # the tied pair in the other order
            g.nearest_edges = lambda x, y, r: near(x, y, r)[1::-1] + near(x, y, r)[2:]
        m = Dewri()
        _, _, cands, _ = m.reconstruct(_inputs(t, v, g, p0=(1350.0, 1200.0)), q)
        exp = m.stats["root_expansions"]
        assert m.stats["budget_hit"] and m.stats["n_roots_expanded"] == m.stats["n_roots"] == len(exp)
        assert sum(exp) == 70 * len(q) and max(exp) - min(exp) <= 1             # equal shares of the default budget
        out.append(sorted(c.tobytes() for c in cands))
    assert out[0] == out[1]
    g.nearest_edges = near
    m = Dewri(max_expand=len(exp) * 70 * len(q))
    _, _, cands, _ = m.reconstruct(_inputs(t, v, g, p0=(1350.0, 1200.0)), q)
    assert m.stats["root_expansions"] == [70 * len(q)] * len(exp) and min(_err(c, _along(route, _dist(t, v, q))) for c in cands) < 5.0
    t, v = _drive(_profile([(math.inf, 25.0)], events), [750.0, 2050.0], t_end=300.0, dwell=20.0)
    q = np.arange(1.0, t[-1], 1.0)
    m = Dewri()
    full = m.reconstruct(_inputs(t, v, g, p0=(1350.0, 1200.0)), q)
    exp = m.stats["root_expansions"]
    need = max(sum(exp[:i]) + (len(exp) - i) * k for i, k in enumerate(exp))   # every share covers its search
    assert not m.stats["budget_hit"] and need < len(exp) * max(exp)
    m = Dewri(max_expand=need)
    out = m.reconstruct(_inputs(t, v, g, p0=(1350.0, 1200.0)), q)
    assert not m.stats["budget_hit"] and m.stats["root_expansions"] == exp and all(np.array_equal(a, b) for a, b in zip(out, full))
    m = Dewri(max_expand=need - 1)
    m.reconstruct(_inputs(t, v, g, p0=(1350.0, 1200.0)), q)
    assert m.stats["budget_hit"]


def test_trip_ending_on_the_start_edge():
    """Short windows: from p0 in the middle of a 500 m edge the car covers 55 m, or stands still, so both directions of
    the start edge pass the trip length before a node, without any expansion: two candidates, the true direction among
    them (a stationary car stays at p0)."""
    g = _graph({"a": (-250.0, 0.0), "b": (250.0, 0.0), "c": (750.0, 0.0)}, [("residential", ["a", "b", "c"])])
    t = np.arange(0.0, 12.0, 0.01)
    q = np.arange(1.0, t[-1], 1.0)
    for v in (np.full(len(t), 5.0), np.zeros(len(t))):
        m = Dewri()
        xy, _, cands, probs = m.reconstruct(_inputs(t, v, g), q)
        truth = _along(np.array([(0.0, 0.0), (750.0, 0.0)]), _dist(t, v, q))
        assert cands is not None and len(cands) == 2 and min(_err(c, truth) for c in cands) < 0.01
        assert m.stats["n_end"] == 2 and m.stats["expansions"] == 0 and np.isfinite(probs).all()


def _dfs_ref(m, g, roots, D, req):
    """Recursive depth-first search with the same constraints, written independently (node sets, linear scans, turns
    against the last path edge of at least short_m found by a backward scan, a heapq Dijkstra on the whole graph, the end
    node checked like any node): {end node: (turns, edge ids, start offset)} in the order the end nodes are found."""
    slack, sp_slack, turn = m.stop_slack_ft * 0.3048, m.sp_slack_mi * 1609.344, math.radians(m.turn_deg)
    nb, dk, sk, best, sp = {}, np.asarray(m.dk), m.sk, {}, {}
    for a, b in zip(g.u, g.v):
        nb.setdefault(int(a), set()).add(int(b))
        nb.setdefault(int(b), set()).add(int(a))

    def dist(r, x):
        if r not in sp:
            sp[r], heap = {r: 0.0}, [(0.0, r)]
            while heap:
                dd, a = heapq.heappop(heap)
                for e in g.out[a]:
                    b, w = int(g.v[e]), dd + float(g.length[e])
                    if dd <= sp[r][a] and w < sp[r].get(b, math.inf):
                        sp[r][b] = w
                        heapq.heappush(heap, (w, b))
        return sp[r].get(x, math.inf)

    def matched(x, L, ms):
        return ms | {k for k, s in enumerate(req) if abs(L - s) <= slack} if len(nb[x]) >= 3 else ms

    def ok(L, ms):
        return all(k in ms for k, s in enumerate(req) if s + slack < L)

    def visit(nodes, edges, L, r, Lr, ms, nt, o):
        x = nodes[-1]
        if L > D:
            if x not in best or nt < best[x][0]:
                best[x] = (nt, edges, o)
            return
        h = [g.bearing[e] for e in edges if g.length[e] >= m.short_m]
        for e, _, uturn in g.succ(edges[-1]):
            y = int(g.v[e])
            is_turn = bool(h) and g.length[e] >= m.short_m and abs((g.bearing[e] - h[-1] + math.pi) % (2 * math.pi) - math.pi) > turn
            if uturn or y in nodes or (is_turn and not sk[int(np.argmin(np.abs(dk - L)))] < m.turn_mph):
                continue
            Ly = L + float(g.length[e])
            m2 = matched(y, Ly, ms)
            if not ok(Ly, m2):
                continue
            if g.highway[e] in ("motorway", "motorway_link"):
                visit(nodes + [y], edges + [e], Ly, y, Ly, m2, nt + is_turn, o)
            elif Ly - Lr <= dist(r, y) + sp_slack:
                visit(nodes + [y], edges + [e], Ly, r, Lr, m2, nt + is_turn, o)

    for e0, o0 in roots:
        x, L = int(g.v[e0]), float(g.length[e0]) - o0
        ms = matched(x, L, set())
        if ok(L, ms):
            visit([x], [e0], L, x, L, ms, 0, o0)
    return best


def test_dfs_matches_a_recursive_search(monkeypatch):
    """On random jittered grids (missing, one-way and motorway ways, duplicate nodes at or 0.5 m from a way's first
    node) with random stops, slow spots and constraint values, the iterative search keeps exactly the end nodes, turn
    counts, paths and order of a recursive DFS."""
    seen = []
    dfs = Dewri._dfs

    def spy(self, mg, g0, roots, D, req, inter, sp, t_start):
        out = dfs(self, mg, g0, roots, D, req, inter, sp, t_start)
        seen.append((roots, D, req, out[0]))
        return out
    monkeypatch.setattr(Dewri, "_dfs", spy)
    rng, rng_z = np.random.default_rng(7), np.random.default_rng(3)
    n_cands = n_req = n_short = 0
    for _ in range(25):
        n = int(rng.integers(4, 8))
        xy = {(i, j): ((j - n // 2) * 100.0 + rng.uniform(-25, 25), (i - n // 2) * 100.0 + rng.uniform(-25, 25)) for i in range(n) for j in range(n)}
        cls = ["residential", "residential", "tertiary", "primary", "motorway", "motorway_link"]
        ways = [(str(rng.choice(cls)), [a, b], int(rng.choice([0, 0, 0, 1, -1])))
                for a in xy for b in ((a[0], a[1] + 1), (a[0] + 1, a[1])) if b in xy and rng.random() > 0.15]
        for _, nodes, _ in ways:                                                # 15 %: a second node at the same place,
            u, phi = rng_z.random(), rng_z.uniform(0.0, 2.0 * math.pi)          # 10 %: 0.5 m off
            if u < 0.25:
                z, r = ("z",) + tuple(nodes), 0.0 if u < 0.15 else 0.5
                xy[z] = (xy[nodes[0]][0] + r * math.cos(phi), xy[nodes[0]][1] + r * math.sin(phi))
                nodes.insert(1, z)
        g = _graph(xy, ways)
        events = [(float(x), float(rng.choice([0.0, 10.0, 15.0]))) for x in rng.uniform(50, 900, int(rng.integers(0, 4)))]
        stops = sorted(float(x) for x in rng.uniform(100, 900, int(rng.integers(0, 3))))
        t, v = _drive(_profile([(math.inf, float(rng.choice([20.0, 25.0, 35.0])))], events), stops, d_end=float(rng.uniform(300, 1000)), dwell=5.0)
        m = Dewri(traffic_mi=float(rng.choice([0.05, 0.5])), sp_slack_mi=float(rng.choice([0.05, 0.1, 0.3])),
                  stop_slack_ft=float(rng.choice([100.0, 500.0])), turn_mph=float(rng.choice([15.0, 25.0, 100.0])))
        seen.clear()
        try:
            m.reconstruct(_inputs(t, v, g, p0=(float(rng.uniform(-150, 150)), float(rng.uniform(-150, 150)))), np.arange(1.0, t[-1], 1.0))
        except DewriNoPath:
            pass
        if not seen:
            continue
        roots, D, req, best = seen[0]
        ref = _dfs_ref(m, g, roots, D, req)
        assert list(best) == list(ref)
        assert all(best[x][0] == ref[x][0] and list(best[x][1]) == ref[x][1] and best[x][2] == ref[x][2] for x in ref)
        n_cands, n_req = n_cands + len(ref), n_req + (len(req) > 0 and len(ref) > 0)
        n_short += sum(bool((g.length[ref[x][1]] < 1.0).any()) for x in ref)
    assert n_cands > 50 and n_req > 0 and n_short > 10                          # the comparison is not vacuous


def test_deterministic():
    g, t, v, _ = _main_trip()
    q = np.arange(1.0, t[-1], 1.0)
    a = Dewri().reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    b = Dewri().reconstruct(_inputs(t, v, g, p0=(50.0, 0.0)), q)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_far_drive_frame_and_invisible_shortcut():
    """Episode frame at (37.002, -121.99), drive graph built about 500 km east (bearings rotated by the meridian
    convergence). The car takes A-N1-N2-B north of the 37.0 N tile border; a shortcut A-X1-X2-B south of it is 300 m
    shorter. With only the episode's tile visible, the shortcut's middle edge does not exist for the episode: the true
    route passes the shortest-path check and is recovered in the episode frame (east-bound theta = 0); with every tile
    visible the shortcut prunes it."""
    origin = (37.002, -121.99)
    xy = {"W": (-100, 0), "P": (100, 0), "A": (300, 0), "N1": (300, 400), "N2": (1300, 400), "B": (1300, 0), "C": (1600, 0),
          "E": (2000, 0), "F": (2500, 0), "G": (3000, 0), "X1": (300, -250), "X2": (1300, -250)}
    ways = [("residential", ["W", "P", "A"]), ("residential", ["A", "N1", "N2", "B"]), ("residential", ["A", "X1", "X2", "B"]),
            ("residential", ["B", "C", "E", "F", "G"])]
    g = _graph(xy, ways, origin=origin, frame=LocalFrame(37.002, -116.4))
    ef = LocalFrame(*origin)
    route = np.array([(0.0, 0.0), (300.0, 0.0), (300.0, 400.0), (1300.0, 400.0), (1300.0, 0.0), (3000.0, 0.0)])
    t, v = _drive(lambda d: 12.0, [], d_end=2500.0)
    q = np.arange(1.0, t[-1], 1.0)
    truth = _along(route, _dist(t, v, q))
    view = EpisodeGraphView(g, ef, visible_tiles={(148, -488)})
    assert abs(math.degrees(view.rot)) > 2.0                                    # the frames really differ
    xy_v, th, _, _ = Dewri(turn_mph=100.0).reconstruct(_inputs(t, v, view, origin=origin), q)
    assert _err(xy_v, truth) < 5.0 and abs(th[5]) < math.radians(0.5)
    xy_a, _, _, _ = Dewri(turn_mph=100.0).reconstruct(_inputs(t, v, EpisodeGraphView(g, ef), origin=origin), q)
    assert _err(xy_a, truth) > 100.0


def test_heading_filter_in_the_episode_frame():
    """heading=True keeps the start edges within 45 deg of theta0 in the episode frame (drive graph built about 500 km
    east, bearings 3.4 deg apart): on a two-way road at -43 deg, theta0 = 0 keeps the two south-east start edges and
    theta0 = 180 deg the two north-west ones, and the car goes that way; on a road at +47 deg no start edge is within
    45 deg of theta0 = 0, so all four are kept. Without theta0 the variant refuses to run."""
    origin = (37.002, -121.99)
    t, v = _drive(lambda d: 12.0, [], d_end=600.0)
    q = np.arange(1.0, t[-1], 1.0)
    for ang, theta0, n_roots in ((-43.0, 0.0, 2), (-43.0, math.pi, 2), (47.0, 0.0, 4)):
        a = math.radians(ang)
        g = _graph({k: (100.0 * k * math.cos(a), 100.0 * k * math.sin(a)) for k in range(-12, 13)}, [("residential", list(range(-12, 13)))],
                   origin=origin, frame=LocalFrame(37.002, -116.4))
        view = EpisodeGraphView(g, LocalFrame(*origin))
        m = Dewri(heading=True)
        xy, th, _, _ = m.reconstruct(_inputs(t, v, view, theta0=theta0, origin=origin), q)
        assert m.stats["n_roots"] == n_roots
        if n_roots == 2:
            h = a if theta0 == 0.0 else a + math.pi
            assert abs((th[-1] - h + math.pi) % (2 * math.pi) - math.pi) < math.radians(0.5)
            assert xy[-1] @ np.array([math.cos(h), math.sin(h)]) > 500.0
    with pytest.raises(ValueError, match="theta0"):
        Dewri(heading=True).reconstruct(_inputs(t, v, view, origin=origin), q)


class _Seen(dict):
    """Column dict that records every name looked up."""

    def __init__(self, cols):
        super().__init__(cols)
        self.seen = set()

    def __getitem__(self, k):
        self.seen.add(k)
        return super().__getitem__(k)

    def __contains__(self, k):
        self.seen.add(k)
        return super().__contains__(k)

    def get(self, k, default=None):
        self.seen.add(k)
        return super().get(k, default)


def test_reads_speed_only():
    """Under G3 the steering column is present: the method looks up speed only and its output does not change when the
    steering changes."""
    g = _tee()
    t, v = _drive(_profile([(math.inf, 25.0)], [(250.0, 10.0)]), [], d_end=650.0)
    q = np.arange(1.0, t[-1], 1.0)
    out = []
    for steer in (np.zeros(len(t)), np.random.default_rng(0).normal(0.0, 90.0, len(t))):
        inp = _inputs(t, v, g, p0=(50.0, 0.0), gid="G3", cols={"steer_angle": steer})
        inp.events.cols = _Seen(inp.events.cols)
        out.append(Dewri().reconstruct(inp, q))
        assert inp.events.cols.seen == {"speed"}
    assert all(np.array_equal(a, b) for a, b in zip(*out))
