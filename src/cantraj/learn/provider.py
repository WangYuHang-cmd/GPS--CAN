"""Online inference of the learned motion models with plain numpy (no torch at run time).

`MotionProvider` owns the causal tick grid (absolute multiples of GRID_DT), the zero-order-hold buffer of the last
event, the GRU hidden state and the integrated pose. A tick k (time t_k) is committed once an event later than t_k or a
query at or after t_k arrives; its features are the ZOH values at t_k (events <= t_k only), the model output is applied
to the interval (t_k - dt, t_k], exactly as in training. Nothing recorded after the query time is ever used.
This module imports only numpy and the shared feature function (no reference loader).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .data import GRID_DT, MAX_AGE_S, N_FEAT, features


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyGRU:
    def __init__(self, w_ih, w_hh, b_ih, b_hh):
        self.w_ih, self.w_hh, self.b_ih, self.b_hh = w_ih, w_hh, b_ih, b_hh
        self.H = w_hh.shape[1]

    def step(self, x, h):
        gi = self.w_ih @ x + self.b_ih
        gh = self.w_hh @ h + self.b_hh
        H = self.H
        r = _sigmoid(gi[:H] + gh[:H])
        z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
        n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
        return (1.0 - z) * n + z * h


class MotionProvider:
    def __init__(self, model_dir):
        self.dir = model_dir
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        w = np.load(os.path.join(model_dir, "weights.npz"))
        self.kind = meta["kind"]
        self.vocab = meta["vocab"]
        self.mean = np.array(meta["norm"]["mean"], dtype=np.float32)
        self.std = np.array(meta["norm"]["std"], dtype=np.float32)
        self.use_gru = "gru.weight_ih_l0" in w.files
        if self.use_gru:
            self.gru = NumpyGRU(w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], w["gru.bias_hh_l0"])
            self.head_w, self.head_b = w["head.weight"], w["head.bias"]
        if self.kind == "gru":
            self.plat_emb, self.brand_emb = w["plat.weight"], w["brand.weight"]
        else:
            self.g, self.plat_p, self.brand_p = w["g"], w["plat.weight"], w["brand.weight"]
        self.name = f"learned_{self.kind}"

    # ---- state
    def reset(self, init, static, theta_fallback=0.0):
        self.static = dict(wheelbase_m=static.wheelbase_m, steer_ratio=static.steer_ratio)
        self.pid = self.vocab["platform"].get(str(getattr(static, "platform", None)), 0)
        self.bid = self.vocab["brand"].get(str(getattr(static, "brand", None)), 0)
        self.x = self.y = 0.0
        self.blind = init.theta0 is None
        self.th = init.theta0 if init.theta0 is not None else theta_fallback
        self.t0 = float(init.t0)
        self.k_next = int(math.ceil(self.t0 / GRID_DT))          # first tick at or after t0
        self.t_last_tick = self.t0
        self.ev_t, self.ev_v, self.ev_s = -np.inf, 0.0, 0.0
        self.prev_v, self.prev_s = None, None
        self.h = np.zeros(self.gru.H, dtype=np.float32) if self.use_gru else None
        self.om, self.scale, self.v_hold = 0.0, 1.0, 0.0
        if self.kind != "gru":
            p = self.g + self.brand_p[self.bid] + self.plat_p[self.pid]
            self.a_p, self.b_p, self.c_p = float(np.exp(p[0])), float(p[1]), float(np.exp(p[2]))
        self.n_ticks = 0

    def _commit_tick(self, t_k):
        age = t_k - self.ev_t if np.isfinite(self.ev_t) else MAX_AGE_S * 10
        missing = float(age > MAX_AGE_S)
        v, s = (self.ev_v, self.ev_s) if np.isfinite(self.ev_t) else (0.0, 0.0)
        pv = v if self.prev_v is None else self.prev_v
        ps = s if self.prev_s is None else self.prev_s
        f = features(v, s, age, missing, pv, ps, self.static, 0.01)
        f = f[0] if np.ndim(f) == 2 else f
        f = np.asarray(f, dtype=np.float32).reshape(-1)
        fs = (f - self.mean) / self.std
        gate = f[7] * (1.0 - f[6])
        if self.kind == "gru":
            x = np.concatenate([fs, self.plat_emb[self.pid], self.brand_emb[self.bid]]).astype(np.float32)
            self.h = self.gru.step(x, self.h)
            c, u = self.head_w @ self.h + self.head_b
            om = 0.6 * math.tanh(float(u)) * gate
            sc = math.exp(0.1 * math.tanh(float(c)))
        else:
            omega_nom = float(f[12]) * 0.3
            r, sc = 0.0, 1.0
            if self.use_gru:
                self.h = self.gru.step(fs.astype(np.float32), self.h)
                u, s_ = self.head_w @ self.h + self.head_b
                r = 0.05 * math.tanh(float(u))
                sc = math.exp(0.05 * math.tanh(float(s_)))
            om = (self.a_p * omega_nom + self.b_p * v + r) * gate
        # apply to the interval (t_k - dt, t_k] with the midpoint heading (same as the torch integrator)
        ds = v * sc * GRID_DT * self.c_p if self.kind != "gru" else v * sc * GRID_DT
        th_new = self.th + om * GRID_DT
        th_mid = th_new - 0.5 * om * GRID_DT
        self.x += ds * math.cos(th_mid)
        self.y += ds * math.sin(th_mid)
        self.th = th_new
        self.om, self.scale, self.v_hold = om, sc, v
        self.prev_v, self.prev_s = v, s
        self.t_last_tick = t_k
        self.n_ticks += 1

    def _commit_until(self, t, inclusive):
        while True:
            t_k = self.k_next * GRID_DT
            if (t_k <= t) if inclusive else (t_k < t):
                self._commit_tick(t_k)
                self.k_next += 1
            else:
                break

    def warm_start(self, init, static, events_t, speed, steer, theta_fallback=0.0, max_s=3600.0):
        """Run the recurrent state over the drive's own events before t0 (at most the last max_s seconds), then rebase the
        pose to the episode start: x = y = 0, theta = theta0, ticks continue seamlessly from t0."""
        import dataclasses
        t0 = float(init.t0)
        keep = events_t >= t0 - max_s
        events_t, speed, steer = events_t[keep], speed[keep], steer[keep]
        if len(events_t) == 0:
            return
        start = float(events_t[0])
        self.reset(dataclasses.replace(init, t0=start), static, theta_fallback)
        for t, v, st in zip(events_t, speed, steer):
            self.ingest(float(t), {"speed": float(v), "steer_angle": float(st)})
        self._commit_until(t0, inclusive=False)
        self.t0 = t0
        self.x, self.y = init.p0_xy
        self.blind = init.theta0 is None
        self.th = init.theta0 if init.theta0 is not None else theta_fallback
        self.t_last_tick = t0                     # the pose is anchored at t0; the first tick >= t0 applies a full GRID_DT as in the cold start

    def ingest(self, t, view):
        """Store an event (speed, steer_angle) logged at time t; commits every tick strictly before t first."""
        self._commit_until(t, inclusive=False)
        v = view.get("speed", np.nan)
        s = view.get("steer_angle", np.nan)
        if np.isfinite(v):
            self.ev_v = float(v)
        if np.isfinite(s):
            self.ev_s = float(s)
        self.ev_t = float(t)
        return self.om, self.scale

    def sync(self, t_ref):
        self._commit_until(t_ref, inclusive=True)

    def rates(self):
        """(heading rate, speed scale) of the last committed tick; the scale includes the residual model's c_p exactly as
        this provider's own integrator applies it (used by the map particle filter as its motion source)."""
        return self.om, (self.scale if self.kind == "gru" else self.scale * self.c_p)

    def pose_at(self, t_ref):
        """Pose extrapolated from the last committed tick with the held speed and heading rate (non-mutating)."""
        dt = max(float(t_ref - self.t_last_tick), 0.0)
        om = self.om
        v = self.v_hold * (self.scale if self.kind == "gru" else self.scale * self.c_p)
        th = self.th + om * dt
        th_mid = th - 0.5 * om * dt
        return np.array([self.x + v * dt * math.cos(th_mid), self.y + v * dt * math.sin(th_mid)]), th
