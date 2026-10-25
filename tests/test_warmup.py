"""Warm-start variant: events before t0 calibrate the state without moving the pose; the plain protocol is untouched."""
import math

import numpy as np
import pytest

from cantraj.baselines.dead_reckoning import DeadReckoning
from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj import replay


def _drive(off_deg=1.5, warm_s=300.0, T=120.0, v=15.0, dt=0.01):
    """Straight driving for warm_s seconds, then an episode with a gentle S-curve; the sensor reads angle + off_deg."""
    L, SR = 2.7, 15.0
    t_all = np.arange(0, warm_s + T, dt)
    delta_true = np.where(t_all >= warm_s, 4.0 * np.sin(2 * math.pi * (t_all - warm_s) / 40.0), 0.0)
    omega = v * np.tan(np.radians(delta_true) / SR) / L
    th = np.concatenate([[0.0], np.cumsum(omega[:-1] * dt)])
    x = np.concatenate([[0.0], np.cumsum(v * np.cos(th[:-1]) * dt)]); y = np.concatenate([[0.0], np.cumsum(v * np.sin(th[:-1]) * dt)])
    k0 = int(round(warm_s / dt))
    cols = {"speed": np.full(len(t_all), v), "steer_angle": delta_true + off_deg}
    warm = EventStream(t_all[:k0], {k: c[:k0] for k, c in cols.items()})
    ev = EventStream(t_all[k0:], {k: c[k0:] for k, c in cols.items()})
    init = EpisodeInit(float(warm_s), 37.0, -122.0, float(th[k0]), t_end=None)
    truth = lambda q: np.column_stack([np.interp(q, t_all, x - x[k0]), np.interp(q, t_all, y - y[k0])])  # noqa: E731
    static = StaticParams(wheelbase_m=L, steer_ratio=SR)
    return ev, warm, init, static, truth


def test_warmup_calibrates_online_offset():
    ev, warm, init, static, truth = _drive()
    q = np.arange(init.t0 + 1.0, init.t0 + 120.0, 1.0)
    plain = EpisodeInputs("G3", ev, static, init)
    warmed = EpisodeInputs("G3", ev, static, init, warmup=warm)
    e_plain = np.hypot(*(replay.run_online(DeadReckoning("steer", offset_mode="online"), plain, q)[0] - truth(q)).T)
    e_warm = np.hypot(*(replay.run_online(DeadReckoning("steer", offset_mode="online"), warmed, q)[0] - truth(q)).T)
    assert e_warm[-1] < 0.3 * e_plain[-1]
    # the estimator is the time-weighted mean shrunk by a 60 s prior: 300 s of straight driving gives 1.5 * 300 / 360 deg
    dr = DeadReckoning("steer", offset_mode="online")
    dr.reset(init, static)
    dr.warmup(warm)
    assert abs(dr._offset() - 1.5 * 300.0 / 360.0) < 1e-3   # 29,999 intervals of 0.01 s
    assert (dr.x, dr.y, dr.th) == (0.0, 0.0, init.theta0)          # the pose never moves during warm-up


def test_warmup_is_a_noop_for_nominal_dr_and_absent_warmup_is_bit_identical():
    ev, warm, init, static, truth = _drive()
    q = np.arange(init.t0 + 1.0, init.t0 + 120.0, 1.0)
    a = replay.run_online(DeadReckoning("steer", offset_mode="none"), EpisodeInputs("G3", ev, static, init), q)[0]
    b = replay.run_online(DeadReckoning("steer", offset_mode="none"), EpisodeInputs("G3", ev, static, init, warmup=warm), q)[0]
    c = replay.run_online(DeadReckoning("steer", offset_mode="online"), EpisodeInputs("G3", ev, static, init, warmup=None), q)[0]
    d = replay.run_online(DeadReckoning("steer", offset_mode="online"), EpisodeInputs("G3", ev, static, init), q)[0]
    assert np.array_equal(a, b) and np.array_equal(c, d)


def test_warmup_must_be_strictly_before_t0():
    ev, warm, init, static, truth = _drive()
    bad = EventStream(np.r_[warm.t_s, init.t0 + 0.5], {k: np.r_[c, c[-1]] for k, c in warm.cols.items()})
    with pytest.raises(ValueError):
        EpisodeInputs("G3", ev, static, init, warmup=bad)
