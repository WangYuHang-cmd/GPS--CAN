"""Benchmark v1.1 corrections: per-episode map visibility and frame-corrected bearings (A3), the learned motion source
fed at logged event times with the tick's own rates inside the map filter (A4), and the two lost-road definitions (A5)."""
import math

import numpy as np
import pandas as pd
import pytest

from cantraj.geo import LocalFrame
from cantraj.maps.graph import DriveGraph, EpisodeGraphView
from cantraj import metrics as M


def _two_tile_graph(frame):
    # a north-south road crossing the tile boundary at 40.25 N, and a short east-west road fully in the northern tile
    nodes = pd.DataFrame({"node_id": [1, 2, 3, 4, 5], "lat": [40.20, 40.24, 40.26, 40.30, 40.30],
                          "lon": [-100.10, -100.10, -100.10, -100.10, -100.05]})
    ways = pd.DataFrame({"way_id": [10, 11], "highway": ["primary", "residential"], "oneway": [0, 0]})
    wn = pd.DataFrame({"way_id": [10, 10, 10, 10, 11, 11], "seq": [0, 1, 2, 3, 0, 1], "node_id": [1, 2, 3, 4, 4, 5]})
    restr = pd.DataFrame(columns=["from_way", "via_node", "to_way", "restriction"])
    return DriveGraph(ways, wn, nodes, restr, frame)


def test_view_exposes_only_the_episode_tiles():
    ef = LocalFrame(40.20, -100.10)
    g = _two_tile_graph(LocalFrame(40.25, -100.08))
    south = {(160, -401)}
    v = EpisodeGraphView(g, ef, visible_tiles=south)
    assert v.n_visible() == 4                                   # 1-2 both ways and the crossing edge 2-3 both ways
    x4, y4 = ef.to_enu(40.30, -100.10)
    assert v.nearest_edges(float(x4), float(y4), 50.0) == []    # node 4 lies in the invisible northern tile
    assert len(EpisodeGraphView(g, ef).nearest_edges(float(x4), float(y4), 50.0)) > 0
    e23 = [e for e in range(g.n_edges()) if g.u[e] == 1 and g.v[e] == 2][0]      # sorted node indices: node 2 -> 3
    assert all(v.visible[s[0]] for s in v.succ(e23))
    assert len(v.succ(e23)) < len(g.succ(e23))                 # the continuation into the northern tile is cut


def test_view_bearings_are_in_the_episode_frame():
    ef = LocalFrame(40.20, -100.10)
    g = _two_tile_graph(LocalFrame(40.25, -110.0))             # drive frame about 840 km west (a long drive's centroid)
    v = EpisodeGraphView(g, ef)
    e12 = [e for e in range(g.n_edges()) if g.u[e] == 0 and g.v[e] == 1][0]      # node 1 -> node 2: due north
    assert abs(g.bearing[e12] - math.pi / 2) > math.radians(5.0)                 # meridian convergence in the drive frame
    assert abs(v.bearing[e12] - math.pi / 2) < math.radians(0.05)


def test_terminal_and_tail_lost_definitions():
    t = np.arange(7.0)
    out = M.episode_summary(np.array([10, 20, 150, 50, 150, 200, 300.0]), t_rel=t)
    assert out["terminal_lost100"] and out["lost_onset_s"] == 4.0
    assert out["tail_lost100"] == out["lost_road"]
    out2 = M.episode_summary(np.array([10, 200, 300, 400, 500, 600, 90.0]), t_rel=t)
    assert not out2["terminal_lost100"] and np.isnan(out2["lost_onset_s"])
    out3 = M.episode_summary(np.array([10, 20, 30, 40, 50, np.nan, np.nan]), t_rel=t)   # missing counts as beyond
    assert out3["terminal_lost100"] and out3["lost_onset_s"] == 5.0


torch = pytest.importorskip("torch")
from cantraj.learn.models import MotionGRU, ResidualGRU  # noqa: E402
from cantraj.learn.provider import MotionProvider  # noqa: E402
from cantraj.baselines.map_pf import MapParticleFilter  # noqa: E402
from cantraj import replay  # noqa: E402
from test_learn import _export  # noqa: E402
from test_map_beam import grid_graph  # noqa: E402
from test_map_pf import _inputs_100hz  # noqa: E402


class _Rec(MotionProvider):
    def __init__(self, d):
        super().__init__(d)
        self.ing, self.ticks, self.sync_t, self.rate_ticks = [], {}, [], []

    def ingest(self, t, view):
        self.ing.append(t)
        return super().ingest(t, view)

    def _commit_tick(self, t_k):
        super()._commit_tick(t_k)
        self.ticks[round(t_k, 6)] = self.om

    def sync(self, t_ref):
        self.sync_t.append(t_ref)
        super().sync(t_ref)

    def rates(self):
        self.rate_ticks.append(self.t_last_tick)
        return super().rates()


def test_learned_source_sees_event_times_and_its_own_tick_in_the_filter(tmp_path):
    torch.manual_seed(3)
    m = MotionGRU(3, 3)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.5)
    d = _export(m, "gru", tmp_path)
    inp = _inputs_100hz(0.0)
    inp.map_graph = grid_graph()
    rec = _Rec(d)
    pf = MapParticleFilter("steer", tick_hz=10.0, seed=0, omega_source=rec)
    replay.run_online(pf, inp, np.arange(1.0, 60.0, 1.0))
    ev_t = inp.events.t_s
    assert len(rec.ing) == int((ev_t <= 59.0 + 1e-9).sum())      # every event up to the last query, nothing later
    assert np.array_equal(np.array(rec.ing), ev_t[:len(rec.ing)])  # at the logged times, in order
    assert np.allclose(rec.rate_ticks, rec.sync_t)              # the rates used at a filter tick are that tick's own
    alone = _Rec(d)
    alone.reset(inp.init, inp.static)
    for k in range(len(ev_t)):
        alone.ingest(float(ev_t[k]), {"speed": float(inp.events.cols["speed"][k]), "steer_angle": float(inp.events.cols["steer_angle"][k])})
    common = sorted(set(rec.ticks) & set(alone.ticks))
    assert len(common) > 500
    assert all(rec.ticks[t] == alone.ticks[t] for t in common)  # identical heading rates to the stand-alone learned model


def test_residual_rates_include_the_speed_factor(tmp_path):
    m = ResidualGRU(2, 2, use_gru=False)
    d = _export(m, "resid_nogru", tmp_path)
    w = dict(np.load(f"{d}/weights.npz"))
    w["g"] = w["g"].copy()
    w["g"][2] = math.log(1.1)
    np.savez(f"{d}/weights.npz", **w)
    inp = _inputs_100hz(0.0)
    p = MotionProvider(d)
    p.reset(inp.init, inp.static)
    for k in range(200):
        p.ingest(float(inp.events.t_s[k]), {"speed": 10.0, "steer_angle": 0.0})
    p.sync(1.5)
    om, sc = p.rates()
    assert abs(p.c_p - 1.1) < 1e-5 and sc == p.scale * p.c_p     # weights are stored as float32
