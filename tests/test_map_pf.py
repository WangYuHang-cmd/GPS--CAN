"""Online map particle filter on the synthetic Manhattan grid (see test_map_beam): the causal 10 Hz tick mode must follow
the truth and agree with the per-event mode on a 100 Hz stream."""
import math

import numpy as np

from cantraj.baselines.map_pf import MapParticleFilter
from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj import replay
from test_map_beam import grid_graph, _truth


def _inputs_100hz(theta0, T=60.0, v=10.0, turn_at=18.5):
    """Steering starts 15 m before the junction (as drivers do); the truth still turns at the junction node (path distance)."""
    t = np.arange(0, T, 0.01)
    steer = np.zeros(len(t))
    k = (t >= turn_at) & (t < turn_at + 3.0)
    L, SR = 2.7, 15.0
    omega = -(math.pi / 2) / 3.0
    steer[k] = math.degrees(math.atan(omega * L / v)) * SR
    ev = EventStream(t, {"speed": np.full(len(t), v), "steer_angle": steer})
    return EpisodeInputs("G3", ev, StaticParams(wheelbase_m=L, steer_ratio=SR), EpisodeInit(0.0, 37.0, -122.0, theta0, t_end=None), input_layer="events")


def _run(pf, inp, g):
    inp.map_graph = g
    q = np.arange(1.0, 60.0, 1.0)
    xy, th = replay.run_online(pf, inp, q)
    return q, xy


def test_pf_tick_mode_follows_truth_and_matches_per_event_mode():
    g = grid_graph()
    q, xy_tick = _run(MapParticleFilter("steer", tick_hz=10.0, seed=0), _inputs_100hz(0.0), g)
    q, xy_evt = _run(MapParticleFilter("steer", tick_hz=None, seed=0), _inputs_100hz(0.0), g)
    tr = _truth(q)
    e_tick = np.hypot(*(xy_tick - tr).T)
    e_evt = np.hypot(*(xy_evt - tr).T)
    assert np.median(e_tick) < 25.0 and e_tick[-1] < 40.0
    assert np.median(e_evt) < 25.0
    assert np.median(np.hypot(*(xy_tick - xy_evt).T)) < 15.0


def test_pf_tick_mode_is_causal():
    """Changing events after a reference time must not change the predictions up to that time."""
    g = grid_graph()
    a = _inputs_100hz(0.0)
    b = _inputs_100hz(0.0)
    b.events.cols["steer_angle"][b.events.t_s > 30.0] = 200.0      # corrupt the future only
    q, xa = _run(MapParticleFilter("steer", tick_hz=10.0, seed=0), a, g)
    q, xb = _run(MapParticleFilter("steer", tick_hz=10.0, seed=0), b, g)
    assert np.allclose(xa[q <= 30.0], xb[q <= 30.0])
