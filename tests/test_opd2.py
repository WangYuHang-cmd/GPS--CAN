"""OPD-II re-implementation with a stand-in matcher (orthogonal snap to the synthetic grid): the windowed bicycle model
and the 30-window snapping follow straight-turn-straight; the online read-out is causal; the final read-out differs from
the online one only by the <= 3 s matching delay; the released bearing-wrap bug is kept unless fixed."""
import math

import numpy as np

from cantraj.baselines.opd2 import OPD2
from cantraj import replay
from test_map_beam import grid_graph, _truth
from test_map_pf import _inputs_100hz


def snap_factory(init):
    """Snap each point to the nearest grid line (100 m blocks around the origin) in the episode frame."""
    fr = init.frame()

    def match(lats, lons):
        x, y = fr.to_enu(np.asarray(lats), np.asarray(lons))
        xs, ys = np.round(x / 100.0) * 100.0, np.round(y / 100.0) * 100.0
        on_x = np.abs(y - ys) < np.abs(x - xs)            # nearer to an east-west line: keep x, snap y
        mx, my = np.where(on_x, x, xs), np.where(on_x, ys, y)
        la, lo = fr.to_lla(mx, my)
        return list(zip(la.tolist(), lo.tolist()))
    return match


def test_follows_the_grid_online_and_final():
    inp = _inputs_100hz(0.0)
    inp.map_graph = grid_graph()
    q = np.arange(1.0, 59.0, 1.0)
    xy_on, _ = replay.run_online(OPD2(readout="online", matcher_factory=snap_factory), inp, q)
    xy_fi, _, _, _ = OPD2(matcher_factory=snap_factory).reconstruct(inp, q)
    tr = _truth(q)
    e_on, e_fi = np.hypot(*(xy_on - tr).T), np.hypot(*(xy_fi - tr).T)
    assert np.median(e_fi) < 15.0 and np.median(e_on) < 20.0
    assert np.all(np.isfinite(xy_fi))


def test_online_readout_is_causal():
    q = np.arange(1.0, 59.0, 1.0)
    a = _inputs_100hz(0.0)
    b = _inputs_100hz(0.0)
    b.events.cols["steer_angle"][b.events.t_s > 30.0] = 300.0
    xa, _ = replay.run_online(OPD2(readout="online", matcher_factory=snap_factory), a, q)
    xb, _ = replay.run_online(OPD2(readout="online", matcher_factory=snap_factory), b, q)
    assert np.allclose(xa[q <= 30.0], xb[q <= 30.0])
    assert not np.allclose(xa[q > 34.0], xb[q > 34.0])


def test_released_bearing_wrap():
    """Heading north (bearing 0) with a left turn: the released code maps a bearing of -x to 360 + x (one window mirrored
    to the east), the fix to 360 - x; afterwards both keep turning normally, so the slip costs about 2x per crossing."""
    inp = _inputs_100hz(math.pi / 2)                      # start heading north
    inp.events.cols["steer_angle"][:] = 0.0
    k = (inp.events.t_s >= 5.0) & (inp.events.t_s < 6.0)
    inp.events.cols["steer_angle"][k] = 60.0              # brief left turn: bearing goes below 0
    q = np.arange(1.0, 20.0, 1.0)
    none = lambda init: (lambda la, lo: None)             # noqa: E731  no map matching: pure dead reckoning
    xr, _ = replay.run_online(OPD2(readout="online", matcher_factory=none), inp, q)
    xf, _ = replay.run_online(OPD2(readout="online", as_released=False, matcher_factory=none), inp, q)
    assert xf[-1, 0] < -5.0 and xr[-1, 0] < -5.0          # both drift west after the left turn
    assert 0.0 < np.abs(xr - xf).max() < 3.0              # the released slip is small but present
