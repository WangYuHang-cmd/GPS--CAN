"""Synthetic straight line and circle: units, signs and the integrator (tool checks, not benchmark results)."""
import math

import numpy as np
import pytest

from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.baselines.dead_reckoning import DeadReckoning
from cantraj import replay


def _inputs(gid, cols, t, theta0=0.0, static=None):
    ev = EventStream(t, cols)
    static = static or StaticParams(wheelbase_m=2.7, steer_ratio=15.0, yawrate_unit="rad_s", yawrate_sign=1)
    return EpisodeInputs(gid, ev, static, EpisodeInit(t0=0.0, p0_lat=37.0, p0_lon=-122.0, theta0=theta0))


def test_straight_line_speed_only():
    t = np.arange(0, 60.0, 0.01)
    inp = _inputs("G1", {"speed": np.full(len(t), 10.0)}, t, theta0=0.0)
    xy, th = replay.run_online(DeadReckoning("none"), inp, np.arange(1.0, 60.0, 1.0))
    assert np.allclose(xy[:, 0], np.arange(1.0, 60.0, 1.0) * 10.0, atol=1e-6)
    assert np.allclose(xy[:, 1], 0.0, atol=1e-6)


def test_circle_curvature_right_positive_is_clockwise():
    v, kappa = 10.0, 0.01                     # right-positive curvature 0.01 1/m -> radius 100 m clockwise
    T = 2 * math.pi / (v * kappa)             # one lap
    t = np.arange(0, T + 0.005, 0.01)
    inp = _inputs("G5", {"speed": np.full(len(t), v), "curvature_loc": np.full(len(t), kappa)}, t, theta0=math.pi / 2)
    xy, th = replay.run_online(DeadReckoning("curvature"), inp, np.array([T / 4, T]))
    # start heading north at origin, turning clockwise: after a quarter lap the car is at (100, 100)
    assert np.allclose(xy[0], [100.0, 100.0], atol=0.5)
    assert np.hypot(*xy[1]) < 1e-2           # back at the origin after one lap
    assert abs(((th[0] - 0.0 + math.pi) % (2 * math.pi)) - math.pi) < 1e-6   # heading east after a quarter lap


def test_positive_steering_turns_left():
    t = np.arange(0, 20.0, 0.01)
    inp = _inputs("G3", {"speed": np.full(len(t), 5.0), "steer_angle": np.full(len(t), 30.0)}, t, theta0=0.0)
    xy, th = replay.run_online(DeadReckoning("steer", offset_mode="none"), inp, np.array([5.0]))
    assert xy[0, 1] > 0 and th[0] > 0        # y grows (left of the initial eastward heading), heading increases CCW


def test_can_yaw_units_and_sign():
    t = np.arange(0, 10.0, 0.01)
    for unit, raw in (("rad_s", 0.1), ("deg_s", math.degrees(0.1))):
        st = StaticParams(yawrate_unit=unit, yawrate_sign=1)
        inp = _inputs("G4", {"speed": np.full(len(t), 10.0), "yaw_rate_can": np.full(len(t), raw)}, t, 0.0, st)
        xy, th = replay.run_online(DeadReckoning("can_yaw"), inp, np.array([10.0]))
        assert abs(th[0] - 1.0) < 1e-6
    st = StaticParams(yawrate_unit="rad_s", yawrate_sign=-1)
    inp = _inputs("G4", {"speed": np.full(len(t), 10.0), "yaw_rate_can": np.full(len(t), 0.1)}, t, 0.0, st)
    xy, th = replay.run_online(DeadReckoning("can_yaw"), inp, np.array([10.0]))
    assert abs(th[0] + 1.0) < 1e-6


def test_irregular_event_spacing_matches_regular():
    v, kappa = 10.0, 0.005
    t_reg = np.arange(0, 100.0, 0.01)
    rng = np.random.default_rng(0)
    t_irr = np.sort(rng.uniform(0, 100.0, 4000))
    outs = []
    for t in (t_reg, t_irr):
        inp = _inputs("G5", {"speed": np.full(len(t), v), "curvature_loc": np.full(len(t), kappa)}, t, 0.0)
        xy, _ = replay.run_online(DeadReckoning("curvature", min_gap_s=1.0), inp, np.array([50.0, 100.0]))
        outs.append(xy)
    assert np.allclose(outs[0], outs[1], atol=0.5)
