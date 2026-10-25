"""Gyro dead reckoning: a biased gyro with a standstill prelude is corrected by the causal bias estimate; the
steering-referenced bias works without any standstill; the uncorrected gyro drifts."""
import math

import numpy as np

from cantraj.baselines.dead_reckoning import DeadReckoning, GyroBias
from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj import replay


def _circle_inputs(bias=0.02, still_s=8.0, T=120.0, v=10.0, omega=0.1, dt=0.01, with_steer=True):
    t = np.arange(0, T, dt)
    moving = t >= still_s
    speed = np.where(moving, v, 0.0)
    gyro = np.where(moving, omega, 0.0) + bias
    L, SR = 2.7, 15.0
    delta = math.degrees(math.atan(omega * L / v)) * SR
    cols = {"speed": speed, "imu_gyro_x": gyro}
    if with_steer:
        cols["steer_angle"] = np.where(moving, delta, 0.0)
    ev = EventStream(t, cols)
    return EpisodeInputs("G8", ev, StaticParams(wheelbase_m=L, steer_ratio=SR), EpisodeInit(0.0, 37.0, -122.0, 0.0, t_end=None), input_layer="events")


def _truth(q, still_s=8.0, v=10.0, omega=0.1):
    tau = np.clip(np.asarray(q) - still_s, 0, None)
    R = v / omega
    return np.column_stack([R * np.sin(omega * tau), R * (1 - np.cos(omega * tau))])


def _uncorrected(source):
    dr = DeadReckoning(source)
    dr_reset = dr.reset

    def reset_no_bias(init, static, map_graph=None):
        dr_reset(init, static, map_graph)
        dr.gyro_bias = GyroBias(prior_s=1e12)               # effectively no correction
    dr.reset = reset_no_bias
    return dr


def test_standstill_bias_corrects_the_gyro():
    """8 s of standstill before a 112 s circle with a 0.02 rad/s gyro bias: the causal standstill estimate removes most of it."""
    inp = _circle_inputs()
    q = np.arange(1.0, 120.0, 1.0)
    xy, th = replay.run_online(DeadReckoning("imu_yaw"), inp, q)
    err = np.hypot(*(xy - _truth(q)).T)
    xy0, th0 = replay.run_online(_uncorrected("imu_yaw"), inp, q)
    err0 = np.hypot(*(xy0 - _truth(q)).T)
    assert err0[-1] > 150.0                                  # uncorrected: about 2.2 rad of heading drift
    assert err[-1] < 0.4 * err0[-1] and np.median(err) < 30.0


def test_steer_referenced_bias_without_standstill():
    """No standstill at all: the gyro minus the nominal steering heading rate while moving estimates the bias."""
    inp = _circle_inputs(still_s=0.0)
    q = np.arange(1.0, 120.0, 1.0)
    xy, th = replay.run_online(DeadReckoning("imu_yaw_sb"), inp, q)
    err = np.hypot(*(xy - _truth(q, still_s=0.0)).T)
    xy0, th0 = replay.run_online(_uncorrected("imu_yaw_sb"), inp, q)
    err0 = np.hypot(*(xy0 - _truth(q, still_s=0.0)).T)
    assert err[-1] < 0.5 * err0[-1]


def _var_speed_inputs(bias=0.05, off_deg=1.0, T=240.0, dt=0.01):
    """Varying speed (8 to 22 m/s), a steering offset of 1 deg on the sensor and a 0.05 rad/s gyro bias; true path
    integrates the true heading rate. Returns (inputs, truth function)."""
    t = np.arange(0, T, dt)
    v = 15.0 + 7.0 * np.sin(2 * math.pi * t / 60.0)
    L, SR = 2.7, 15.0
    delta_true = 6.0 * np.sin(2 * math.pi * t / 37.0)                     # degrees, the real wheel angle command
    omega_true = v * np.tan(np.radians(delta_true) / SR) / L
    gyro = omega_true + bias
    steer_meas = delta_true + off_deg                                       # sensor reads the angle plus an offset
    ev = EventStream(t, {"speed": v, "imu_gyro_x": gyro, "steer_angle": steer_meas})
    th = np.concatenate([[0.0], np.cumsum(omega_true[:-1] * dt)])
    x = np.concatenate([[0.0], np.cumsum(v[:-1] * np.cos(th[:-1]) * dt)])
    y = np.concatenate([[0.0], np.cumsum(v[:-1] * np.sin(th[:-1]) * dt)])
    truth = lambda q: np.column_stack([np.interp(q, t, x), np.interp(q, t, y)])  # noqa: E731
    return EpisodeInputs("G8", ev, StaticParams(wheelbase_m=L, steer_ratio=SR), EpisodeInit(0.0, 37.0, -122.0, 0.0, t_end=None), input_layer="events"), truth


def test_rls_bias_beats_standstill_and_steering_without_standstill():
    inp, truth = _var_speed_inputs()
    q = np.arange(1.0, 240.0, 1.0)
    xy_rls, _ = replay.run_online(DeadReckoning("imu_yaw_rls"), inp, q)
    xy_std, _ = replay.run_online(DeadReckoning("imu_yaw"), inp, q)              # no standstill: bias stays at the prior
    xy_steer, _ = replay.run_online(DeadReckoning("steer", offset_mode="none"), inp, q)
    e_rls, e_std, e_steer = (np.hypot(*(xy - truth(q)).T) for xy in (xy_rls, xy_std, xy_steer))
    assert e_rls[-1] < 0.2 * e_std[-1]
    assert e_rls[-1] < 0.5 * e_steer[-1]
    assert np.median(e_rls) < 300.0                 # the first seconds run on the plain mean before the speed varies
