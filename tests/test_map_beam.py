"""Offline beam search on a synthetic Manhattan grid: straight, right turn, straight. H+ recovers the path; H-
resolves the start direction after the turn; two runs are byte-identical (no randomness)."""
import math

import numpy as np
import pandas as pd

from cantraj.baselines.map_beam import MapBeamSearch
from cantraj.events import EventStream
from cantraj.geo import LocalFrame
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.maps.graph import DriveGraph
from cantraj import replay, registry


def grid_graph(n=15, block=100.0, lat0=37.0, lon0=-122.0):
    """n x n grid of two-way residential ways around the origin."""
    frame = LocalFrame(lat0, lon0)
    nodes, ways, wn = [], [], []
    nid = {}
    k = 0
    for i in range(n):
        for j in range(n):
            x, y = (j - n // 2) * block, (i - n // 2) * block
            lat, lon = frame.to_lla(np.array([x]), np.array([y]))
            nid[(i, j)] = k
            nodes.append(dict(node_id=k, lat=float(lat[0]), lon=float(lon[0])))
            k += 1
    w = 0
    for i in range(n):                 # east-west ways
        ways.append(dict(way_id=w, highway="residential", oneway=0, junction="", maxspeed="", lanes="", name_len=0, n_nodes=n))
        for j in range(n):
            wn.append(dict(way_id=w, seq=j, node_id=nid[(i, j)]))
        w += 1
    for j in range(n):                 # north-south ways
        ways.append(dict(way_id=w, highway="residential", oneway=0, junction="", maxspeed="", lanes="", name_len=0, n_nodes=n))
        for i in range(n):
            wn.append(dict(way_id=w, seq=i, node_id=nid[(i, j)]))
        w += 1
    g = DriveGraph(pd.DataFrame(ways), pd.DataFrame(wn), pd.DataFrame(nodes), pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"]), frame)
    return g


def _inputs(theta0, T=60.0, v=10.0, turn_at=20.0):
    """Drive east at 10 m/s, right turn (90 deg over 3 s) starting at the junction x = 200 m, then south."""
    t = np.arange(0, T, 0.1)
    steer = np.zeros(len(t))
    k = (t >= turn_at) & (t < turn_at + 3.0)
    # omega = -pi/2 / 3 s (right turn) = v tan(delta/SR)/L -> delta
    L, SR = 2.7, 15.0
    omega = -(math.pi / 2) / 3.0
    delta = math.degrees(math.atan(omega * L / v)) * SR
    steer[k] = delta
    ev = EventStream(t, {"speed": np.full(len(t), v), "steer_angle": steer})
    st = StaticParams(wheelbase_m=L, steer_ratio=SR)
    return EpisodeInputs("G3", ev, st, EpisodeInit(0.0, 37.0, -122.0, theta0, t_end=T), input_layer="events")


def _truth(q, v=10.0, turn_at=20.0):
    """Path distance s = v t along the grid: east until the junction at v*turn_at, then south."""
    xj = v * turn_at
    s_ = v * np.asarray(q, dtype=float)
    return np.column_stack([np.minimum(s_, xj), -np.maximum(s_ - xj, 0.0)])


def test_beam_recovers_turn_with_heading():
    g = grid_graph()
    inp = _inputs(0.0)
    inp.map_graph = g
    q = np.arange(1.0, 60.0, 1.0)
    xy, th, cands, probs = replay.run_offline(MapBeamSearch("steer"), inp, q)
    err = np.hypot(*(xy - _truth(q)).T)
    assert np.median(err) < 25.0 and err[-1] < 40.0


def test_beam_heading_unknown_gives_a_rotation_of_the_path():
    """On a symmetric grid the start direction cannot be known: the Top-1 must be one of the four rotations of the
    true path and the candidate set must contain more than one distinct path."""
    g = grid_graph()
    inp = _inputs(None)
    inp.map_graph = g
    q = np.arange(1.0, 60.0, 1.0)
    xy, th, cands, probs = replay.run_offline(MapBeamSearch("steer"), inp, q)
    tr = _truth(q)
    best = min(np.hypot(*(xy - np.column_stack([c * tr[:, 0] - s_ * tr[:, 1], s_ * tr[:, 0] + c * tr[:, 1]])).T).mean()
               for c, s_ in ((1, 0), (0, 1), (-1, 0), (0, -1)))
    assert best < 30.0
    assert cands is not None and len(cands) >= 2


def test_beam_is_deterministic():
    g = grid_graph()
    inp = _inputs(0.0)
    inp.map_graph = g
    q = np.arange(1.0, 60.0, 1.0)
    a = replay.run_offline(MapBeamSearch("steer"), inp, q)
    b = replay.run_offline(MapBeamSearch("steer"), inp, q)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])
