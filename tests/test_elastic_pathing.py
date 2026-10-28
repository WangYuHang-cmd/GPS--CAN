"""Elastic Pathing re-implementation on the synthetic Manhattan grid (100 m blocks, two-way residential roads):
Eq. 1/3 turn speeds, a fast straight drive stays straight, a slow-down at a junction lets the true turn into the top-n,
and an exhausted budget is a failure (no output)."""
import math

import numpy as np
import pytest

from cantraj.baselines.elastic_pathing import ElasticPathing, NoSolution, safe_turn_speed
from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from test_map_beam import grid_graph


def _inputs(speed_fn, T, theta0=0.0):
    t = np.arange(0, T, 0.01)
    ev = EventStream(t, {"speed": speed_fn(t)})
    inp = EpisodeInputs("G1", ev, StaticParams(), EpisodeInit(0.0, 37.0, -122.0, theta0, t_end=T), input_layer="events")
    inp.map_graph = grid_graph()
    return inp


def test_safe_turn_speed_eq1_eq3():
    x = y = 2 * 3.6576
    r = x + y + math.sqrt(2 * x * y)                      # 90 deg: cos a = 0
    v = math.sqrt(15 * (r / 0.3048) * (0.08 + 0.3)) * 0.44704
    assert abs(safe_turn_speed(math.pi / 2, x, y, 0.3) - v) < 1e-9
    assert safe_turn_speed(0.0, x, y, 0.3) == math.inf


def test_fast_straight_drive_stays_straight():
    inp = _inputs(lambda t: np.full(len(t), 15.0), 40.0)
    q = np.arange(1.0, 40.0, 1.0)
    xy, _, _, _ = ElasticPathing(heading=True).reconstruct(inp, q)
    assert np.median(np.hypot(xy[:, 0] - 15.0 * q, xy[:, 1])) < 5.0


def test_slow_turn_is_in_the_top_n():
    """10 m/s east, slow to 3 m/s for the junction at x = 200 m, turn right (south), back to 10 m/s."""
    def speed(t):
        return np.where((t > 18.0) & (t < 26.0), 3.0, 10.0)
    T = 45.0
    inp = _inputs(speed, T)
    q = np.arange(1.0, T, 1.0)
    tt = np.arange(0, T, 0.01)
    s_q = np.interp(q, tt, np.cumsum(speed(tt)) * 0.01)
    truth = np.where(s_q[:, None] <= 200.0, np.column_stack([s_q, 0 * s_q]), np.column_stack([np.full_like(s_q, 200.0), -(s_q - 200.0)]))
    xy, _, cands, _ = ElasticPathing(heading=True).reconstruct(inp, q)
    allc = cands if cands is not None else xy[None]
    best = min(float(np.median(np.hypot(*(c - truth).T))) for c in allc)
    assert best < 15.0


def test_budget_exhausted_is_a_failure():
    inp = _inputs(lambda t: np.where((t // 5) % 2 == 0, 3.0, 9.0), 120.0)
    with pytest.raises(NoSolution):
        ElasticPathing(max_expand=3).reconstruct(inp, np.arange(1.0, 120.0, 1.0))


def test_tie_order_shuffle_keeps_the_default_and_the_candidates():
    """tie_seed=None is the edge-id order (outputs unchanged, repeatable); a shuffled tie order still returns a path
    and keeps the true turn among the top-n candidates."""
    def speed(t):
        return np.where((t > 18.0) & (t < 26.0), 3.0, 10.0)
    T = 45.0
    q = np.arange(1.0, T, 1.0)
    a, _, _, _ = ElasticPathing(heading=True).reconstruct(_inputs(speed, T), q)
    b, _, _, _ = ElasticPathing(heading=True, tie_seed=None).reconstruct(_inputs(speed, T), q)
    assert np.array_equal(a, b)
    tt = np.arange(0, T, 0.01)
    s_q = np.interp(q, tt, np.cumsum(speed(tt)) * 0.01)
    truth = np.where(s_q[:, None] <= 200.0, np.column_stack([s_q, 0 * s_q]), np.column_stack([np.full_like(s_q, 200.0), -(s_q - 200.0)]))
    for seed in (1, 2, 3):
        xy, _, cands, _ = ElasticPathing(heading=True, tie_seed=seed).reconstruct(_inputs(speed, T), q)
        allc = cands if cands is not None else xy[None]
        assert min(float(np.median(np.hypot(*(c - truth).T))) for c in allc) < 15.0
