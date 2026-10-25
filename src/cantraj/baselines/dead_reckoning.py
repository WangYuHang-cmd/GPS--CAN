"""Kinematic dead reckoning on the raw event stream.

Heading rate (left-positive, rad/s) per source, following configs/signal_registry.yaml `integration`:
  steer:     omega = v * tan(deg2rad(steer_angle - angle_offset) / steer_ratio) / wheelbase
  can_yaw:   omega = yawrate_sign * unit * yaw_rate_can          (unit: rad_s 1, deg_s pi/180, deg_s_x10 pi/1800)
  curvature: omega = -v * curvature_loc                           (right-positive curvature -> negate)
  loc_yaw:   omega = -yaw_rate_loc                                (grid layer only)
Between consecutive events the previous speed and heading rate are held (exact circular arc). Inside an input gap
(interval > max(3 x nominal dt, 0.3 s)) the speed is held and the heading rate is zero (counted in `n_gap`).
Online `predict(t_ref)` extrapolates from the last event with the held values and does not mutate the state.
"""
from __future__ import annotations

import math

import numpy as np

from ..registry import ForbiddenSignalError
from .base import OfflineBaseline, OnlineBaseline, Prediction
from .heading_prior import rotations

UNIT = {"rad_s": 1.0, "deg_s": math.pi / 180.0, "deg_s_x10": math.pi / 1800.0}
SOURCE_NEEDS = {"steer": ("speed", "steer_angle"), "can_yaw": ("speed", "yaw_rate_can"),
                "curvature": ("speed", "curvature_loc"), "loc_yaw": ("speed", "yaw_rate_loc"), "none": ("speed",),
                "imu_yaw": ("speed", "imu_gyro_x"), "imu_yaw_sb": ("speed", "imu_gyro_x", "steer_angle"),
                "imu_yaw_rls": ("speed", "imu_gyro_x", "steer_angle")}
IMU_SOURCES = ("imu_yaw", "imu_yaw_sb", "imu_yaw_rls")


class GyroBias:
    """Causal estimate of the device gyro yaw bias (rad/s). mode 'standstill': time-weighted mean of the gyro while the car
    stands still (speed < v_still), shrunk towards 0 with a prior of `prior_s` seconds. mode 'steer': time-weighted mean of
    (gyro - nominal steering heading rate) while moving faster than v_move (uses only CAN signals; roads are straight on
    average and the steering model is unbiased on average). Both use only past samples."""

    def __init__(self, mode="standstill", prior_s=1.0, v_still=0.3, v_move=5.0, ridge_s=2.0):
        self.mode, self.prior_s, self.v_still, self.v_move, self.ridge_s = mode, prior_s, v_still, v_move, ridge_s
        self.num = self.den = 0.0
        self.S1 = self.Sv = self.Svv = self.Sy = self.Svy = 0.0      # rls mode: time-weighted sums for y = b + c * v

    def update(self, dt, v, gyro, omega_ref=None, steer_term=None):
        """omega_ref: nominal steering heading rate (mode 'steer'); steer_term: v * delta / (SR * L) with the raw
        steering angle, no offset (mode 'rls'): gyro - steer_term = b - (offset / (SR * L)) * v, linear in v, so the
        gyro bias b and the steering offset are identified jointly once the speed varies."""
        if not (math.isfinite(gyro) and dt > 0):
            return
        if self.mode == "standstill" and v < self.v_still:
            self.num += gyro * dt
            self.den += dt
        elif self.mode == "steer" and v > self.v_move and omega_ref is not None and math.isfinite(omega_ref):
            self.num += (gyro - omega_ref) * dt
            self.den += dt
        elif self.mode == "rls" and v > 2.0 and steer_term is not None and math.isfinite(steer_term):
            y = gyro - steer_term
            self.S1 += dt; self.Sv += dt * v; self.Svv += dt * v * v; self.Sy += dt * y; self.Svy += dt * v * y

    @property
    def value(self):
        if self.mode == "rls":
            if self.S1 <= 0:
                return 0.0
            mean_v = self.Sv / self.S1
            var_v = max(self.Svv / self.S1 - mean_v * mean_v, 0.0)
            if var_v < 4.0:                        # speed has not varied enough to separate bias from steering offset:
                return self.Sy / self.S1           # use the plain mean (offset assumed 0), which converges within seconds
            r = 1e-6                               # 1 and v are nearly collinear: any real ridge would bias the intercept
            a11, a12, a22 = self.S1 + r, self.Sv, self.Svv + r
            det = a11 * a22 - a12 * a12
            if det <= 1e-12:
                return self.Sy / self.S1
            return (a22 * self.Sy - a12 * self.Svy) / det
        return self.num / (self.den + self.prior_s)


def make_gyro_bias(source):
    if source == "imu_yaw_sb":
        return GyroBias("steer", prior_s=20.0)
    if source == "imu_yaw_rls":
        return GyroBias("rls")
    return GyroBias("standstill", prior_s=1.0)


def steer_term(static, v, view):
    """v * delta / (SR * L) with the raw steering angle (small-angle form used by the rls bias estimator)."""
    if static.steer_ratio is None or static.wheelbase_m is None or not math.isfinite(view.get("steer_angle", math.nan)):
        return None
    return v * math.radians(view["steer_angle"]) / (static.steer_ratio * static.wheelbase_m)


def arc_step(x, y, th, v, omega, dt):
    dth = omega * dt
    if abs(dth) < 1e-9:
        return x + v * dt * math.cos(th), y + v * dt * math.sin(th), th
    R = v / omega
    return x + R * (math.sin(th + dth) - math.sin(th)), y - R * (math.cos(th + dth) - math.cos(th)), th + dth


class DeadReckoning(OnlineBaseline, OfflineBaseline):
    def __init__(self, source="steer", sign_flip=False, gap_factor=3.0, min_gap_s=0.3, theta_fallback=0.0, K_headings=36,
                 offset_mode="online", offset_prior_s=60.0, offset_min_speed=5.0, use_eff_ratio=False):
        """offset_mode (steer source only): 'none' = nominal zero offset; 'fixed' = static.angle_offset_train_deg
        (training-fold value); 'online' = causal time-weighted mean of the steering angle while moving (a CAN-only
        self-calibration: roads are straight on average), shrunk towards 0 with a prior of `offset_prior_s` seconds."""
        self.source = source
        self.sign = -1.0 if sign_flip else 1.0
        self.gap_factor, self.min_gap_s = gap_factor, min_gap_s
        self.theta_fallback = theta_fallback
        self.K = K_headings
        self.offset_mode, self.offset_prior_s, self.offset_min_speed = offset_mode, offset_prior_s, offset_min_speed
        self.use_eff_ratio = use_eff_ratio
        self.name = f"dr_{source}" + ("_flip" if sign_flip else "") + (f"_off{offset_mode}" if source == "steer" and offset_mode != "none" else "") + ("_sreff" if use_eff_ratio else "")
        self.needs = SOURCE_NEEDS[source]

    # ---- heading-rate model
    def _omega(self, v, view):
        s = self.static
        if self.source == "none":
            return 0.0
        if self.source == "steer":
            if s.steer_ratio is None or s.wheelbase_m is None:
                raise ValueError("steer DR needs steer_ratio and wheelbase_m")
            sr = s.steer_ratio_eff_train if (self.use_eff_ratio and s.steer_ratio_eff_train) else s.steer_ratio
            delta = math.radians(view["steer_angle"] - self._offset()) / sr
            return self.sign * v * math.tan(delta) / s.wheelbase_m
        if self.source == "can_yaw":
            u = UNIT.get(s.yawrate_unit or "rad_s", 1.0)
            return self.sign * (s.yawrate_sign or 1) * u * view["yaw_rate_can"]
        if self.source == "curvature":
            return self.sign * (-v * view["curvature_loc"])
        if self.source == "loc_yaw":
            return self.sign * (-view["yaw_rate_loc"])
        if self.source in IMU_SOURCES:                        # device gyro yaw axis (v0), left-positive, minus the causal bias
            return self.sign * (view["imu_gyro_x"] - self.gyro_bias.value)
        raise ValueError(self.source)

    def _steer_omega_nominal(self, v, view):
        s = self.static
        if s.steer_ratio is None or s.wheelbase_m is None or not math.isfinite(view.get("steer_angle", math.nan)):
            return None
        return v * math.tan(math.radians(view["steer_angle"]) / s.steer_ratio) / s.wheelbase_m

    # ---- online interface
    def reset(self, init, static, map_graph=None):
        self.static = static
        self.x, self.y = init.p0_xy
        self.blind = init.theta0 is None
        self.th = init.theta0 if init.theta0 is not None else self.theta_fallback
        self.t_last = init.t0
        self.v = 0.0
        self.om = 0.0
        self.n_gap = 0
        self.nominal_dt = None
        self.off_num, self.off_den = 0.0, 0.0           # online steering-offset accumulator (deg*s, s)
        self.gyro_bias = make_gyro_bias(self.source)
        self.path = [(init.t0, self.x, self.y, self.th)]

    def _offset(self):
        if self.source != "steer" or self.offset_mode == "none":
            return 0.0
        if self.offset_mode == "fixed":
            return float(self.static.angle_offset_train_deg or 0.0)
        return self.off_num / (self.off_den + self.offset_prior_s)

    def _thr(self):
        return max(self.gap_factor * (self.nominal_dt or 0.1), self.min_gap_s)

    def observe(self, events):
        if len(events) == 0:
            return
        if self.nominal_dt is None and events.nominal_dt:
            self.nominal_dt = events.nominal_dt
        for n in self.needs:
            if n not in events.cols:
                raise ForbiddenSignalError(f"{self.name} needs {n!r}, not in inputs {events.names()}")
        t = events.t_s
        cols = {n: events.cols[n] for n in self.needs}
        for i in range(len(t)):
            dt = float(t[i] - self.t_last)
            if dt < 0:
                continue
            om = self.om
            if dt > self._thr():
                self.n_gap += 1
                om = 0.0
            self.x, self.y, self.th = arc_step(self.x, self.y, self.th, self.v, om, dt)
            view = {n: float(cols[n][i]) for n in self.needs}
            v = view["speed"]
            if math.isfinite(v):
                self.v = v
            if self.source == "steer" and self.offset_mode == "online" and self.v > self.offset_min_speed and dt <= self._thr():
                a = view["steer_angle"]
                if math.isfinite(a):
                    self.off_num += a * dt
                    self.off_den += dt
            if self.source in IMU_SOURCES and dt <= self._thr():
                self.gyro_bias.update(dt, self.v, view["imu_gyro_x"], self._steer_omega_nominal(self.v, view) if self.source == "imu_yaw_sb" else None,
                                      steer_term(self.static, self.v, view) if self.source == "imu_yaw_rls" else None)
            try:
                om_new = self._omega(self.v, view)
            except (ValueError, KeyError):
                raise
            if math.isfinite(om_new):
                self.om = om_new
            self.t_last = float(t[i])
            self.path.append((self.t_last, self.x, self.y, self.th))

    def warmup(self, events):
        """Calibrate from the drive's own events before t0 without moving the pose: online steering offset (time-weighted
        mean of the steering angle while moving) and the gyro bias estimators. Vectorised so that hours of history cost
        milliseconds."""
        if len(events) < 2:
            return
        if self.nominal_dt is None and events.nominal_dt:
            self.nominal_dt = events.nominal_dt
        t = events.t_s
        dt = np.diff(t)
        ok = dt <= self._thr()
        v = np.nan_to_num(events.cols["speed"][1:], nan=0.0)
        if self.source == "steer" and self.offset_mode == "online":
            a = events.cols["steer_angle"][1:]
            m = ok & (v > self.offset_min_speed) & np.isfinite(a)
            self.off_num += float(np.sum(a[m] * dt[m]))
            self.off_den += float(np.sum(dt[m]))
        if self.source in IMU_SOURCES:
            g = events.cols["imu_gyro_x"][1:]
            gb = self.gyro_bias
            if gb.mode == "standstill":
                m = ok & (v < gb.v_still) & np.isfinite(g)
                gb.num += float(np.sum(g[m] * dt[m])); gb.den += float(np.sum(dt[m]))
            else:
                st = events.cols["steer_angle"][1:] if "steer_angle" in events.cols else np.full(len(v), np.nan)
                s_ = self.static
                if s_.steer_ratio and s_.wheelbase_m:
                    term = v * np.radians(np.nan_to_num(st)) / (s_.steer_ratio * s_.wheelbase_m)
                    if gb.mode == "steer":
                        ref = v * np.tan(np.radians(np.nan_to_num(st)) / s_.steer_ratio) / s_.wheelbase_m
                        m = ok & (v > gb.v_move) & np.isfinite(g) & np.isfinite(st)
                        gb.num += float(np.sum((g[m] - ref[m]) * dt[m])); gb.den += float(np.sum(dt[m]))
                    else:                                                   # rls sums
                        m = ok & (v > 2.0) & np.isfinite(g) & np.isfinite(st)
                        y = g[m] - term[m]; w = dt[m]; vv = v[m]
                        gb.S1 += float(w.sum()); gb.Sv += float((w * vv).sum()); gb.Svv += float((w * vv * vv).sum())
                        gb.Sy += float((w * y).sum()); gb.Svy += float((w * vv * y).sum())
        vf = events.cols["speed"][np.isfinite(events.cols["speed"])]
        if len(vf):
            self.v = float(vf[-1])                   # last known speed carries into the first interval of the episode

    def predict(self, t_ref):
        dt = float(t_ref - self.t_last)
        om = 0.0 if dt > self._thr() else self.om
        x, y, th = arc_step(self.x, self.y, self.th, self.v, om, max(dt, 0.0))
        return Prediction(np.array([x, y]), th)

    # ---- offline interface (DR uses no future, so offline == online replay, plus K rotations when heading unknown)
    def reconstruct(self, inputs, t_query):
        self.reset(inputs.init, inputs.static, inputs.map_graph)
        ev = inputs.events
        xy = np.empty((len(t_query), 2))
        th = np.empty(len(t_query))
        i = 0
        for q, t in enumerate(t_query):
            j = int(np.searchsorted(ev.t_s, t, side="right"))
            if j > i:
                self.observe(ev.window(ev.t_s[i], ev.t_s[j - 1]) if j - i < len(ev) else ev)
                i = j
            p = self.predict(t)
            xy[q], th[q] = p.xy, p.theta
        if self.blind:
            cands, probs = rotations(xy, self.K)
            return xy, th, cands, probs
        return xy, th, None, None
