"""MacroTracking re-implementation on the synthetic Manhattan grid: follows straight-turn-straight with the map
correction, is causal, and the as-released angle handling runs (and differs once the heading wraps through 0 deg)."""
import numpy as np

from cantraj.baselines.macrotracking import MacroTracking
from cantraj import replay
from test_map_beam import grid_graph, _truth
from test_map_pf import _inputs_100hz


def _run(b, inp, g, T=59.0):
    inp.map_graph = g
    q = np.arange(1.0, T + 1e-9, 1.0)
    xy, _ = replay.run_online(b, inp, q)
    return q, xy


def test_follows_the_grid_and_is_causal():
    g = grid_graph()
    q, xy = _run(MacroTracking(), _inputs_100hz(0.0), g)
    e = np.hypot(*(xy - _truth(q)).T)
    assert np.median(e) < 25.0 and e[-1] < 60.0
    b = _inputs_100hz(0.0)
    b.events.cols["steer_angle"][b.events.t_s > 30.0] = 300.0
    q, xb = _run(MacroTracking(), b, g)
    assert np.allclose(xy[q <= 30.0], xb[q <= 30.0])
    assert not np.allclose(xy[q > 32.0], xb[q > 32.0])


def test_as_released_angle_handling_runs():
    """Driving east with a small right steering bias: the heading drifts just below 360 deg. The wrapped comparison keeps
    pulling it back to the road; the released raw-degree comparison flips the edge (358 vs 0 deg) and stops correcting."""
    g = grid_graph()
    inp = _inputs_100hz(0.0)
    inp.events.cols["steer_angle"][:] = -3.0
    q, xa = _run(MacroTracking(), inp, g, T=40.0)                 # as released (default)
    inp = _inputs_100hz(0.0)
    inp.events.cols["steer_angle"][:] = -3.0
    q, xm = _run(MacroTracking(as_released=False), inp, g, T=40.0)
    assert np.isfinite(xa).all()
    assert abs(xm[-1, 1]) < 30.0 and abs(xa[-1, 1]) > 2.0 * abs(xm[-1, 1])   # cross-track offset from the road (y = 0)
