"""Signal-combination study (ICDE RQ2): one learner, one training budget, one decoder; only the input signals change.

Signal sets (each feature is computed from that set's signals only; columns of absent signals are constant zeros):
  S1  speed                      S2c speed + continuous pedals (accelerator position, brake)   S2b speed + pedal-pressed flags
  S3  speed + steering angle     S4c S3 + continuous pedals                                     S4b S3 + pedal-pressed flags
Learner: the causal GRU of the main study (64 units, platform / brand embeddings, 10 Hz tick grid) with two heading
outputs: the signed heading rate mu (as in the main study) and an unsigned turn magnitude a >= 0 (rad/s). Without
steering the SIGN of a turn is not identifiable, but its occurrence and size are (decelerations, pedal releases), so
the magnitude carries what speed and pedals know. Loss = free-running position error (as in the main study) + Huber of
the 1 s signed heading change + Huber of the 1 s absolute heading change against sum(a dt) + chord-length term;
train-fold GPS only; validation chunk loss for model selection; same budget for every set.
Decoder (the same for every set): the map particle filter with a turn budget: each particle accumulates the magnitude
not explained by its signed heading, U += max(a - |mu|, 0) dt (decaying with a 6 s time constant); at a junction a way out
with turn angle phi scores (1 - q) N(bearing - theta; sigma_turn) + q N(|phi| - U; 0.35 rad) with q = U / (U + 0.3 rad);
taking a turn (|phi| > 30 deg) sets the particle heading to the new road and spends |phi| of the budget. With steering
present the budget stays near zero and this is the ordinary filter.
Continuous pedal scales differ by brand (0-1, 0-100, offset raw counts): the accelerator and brake are normalised with
the brand's train-fold 1st / 99th percentiles; brands without train data get the pedal columns marked missing.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from .data import GRID_DT, MAX_AGE_S, grid_times

F = 20
BLOCKS = {"speed": [0, 1, 2, 3, 4, 5], "steer": [6, 7, 8, 9, 10, 11, 12], "pedal": [13, 14, 15, 16], "flags": [17, 18, 19]}
GROUPS = {"S1": ("speed",), "S2c": ("speed", "pedal"), "S2b": ("speed", "flags"), "S3": ("speed", "steer"),
          "S4c": ("speed", "steer", "pedal"), "S4b": ("speed", "steer", "flags")}
SIGNALS = {"speed": ["speed"], "steer": ["steer_angle"], "pedal": ["gas", "brake"], "flags": ["gas_pressed", "brake_pressed"]}
FEATURE_NAMES = ["v", "dv", "age", "missing", "moving", "rate_regime", "steer", "wheel_angle", "steer_rate", "omega_nom",
                 "wheelbase_dev", "steer_ratio_dev", "static_missing", "gas", "gas_released", "brake", "pedal_missing",
                 "gas_pressed", "brake_pressed", "flags_missing"]
BINARY = [3, 4, 12, 14, 16, 17, 18, 19]


def group_signals(group):
    return [s for b in GROUPS[group] for s in SIGNALS[b]]


def mask(group):
    m = np.zeros(F, dtype=np.float32)
    for b in GROUPS[group]:
        m[BLOCKS[b]] = 1.0
    return m


def sig_features(v, steer, gas, brake, gp, bp, age, missing, prev_v, prev_steer, static, scales, nominal_dt, group):
    """Feature rows for ticks (arrays) or one tick (scalars). All inputs are zero-order-hold values at the tick;
    prev_* are the previous tick's values (zero differences at the first tick). Shared bitwise by trainer and provider."""
    v = np.atleast_1d(np.asarray(v, dtype=np.float32))
    n = len(v)
    out = np.zeros((n, F), dtype=np.float32)
    v_c = np.nan_to_num(v)
    out[:, 0] = v_c / 10.0
    out[:, 1] = (v_c - np.nan_to_num(np.atleast_1d(np.asarray(prev_v, dtype=np.float32)))) / GRID_DT / 3.0
    out[:, 2] = np.clip(np.nan_to_num(np.atleast_1d(np.asarray(age, dtype=np.float32)), nan=MAX_AGE_S) / MAX_AGE_S, 0.0, 3.0)
    out[:, 3] = np.atleast_1d(np.asarray(missing, dtype=np.float32))
    out[:, 4] = (v_c > 0.3).astype(np.float32)
    out[:, 5] = math.log10(max(nominal_dt, 0.01) / 0.01)
    blocks = GROUPS[group]
    if "steer" in blocks:
        L, SR = static.get("wheelbase_m"), static.get("steer_ratio")
        sm = float(L is None or SR is None or not np.isfinite(L) or not np.isfinite(SR))
        L_, SR_ = (2.8, 15.0) if sm else (float(L), float(SR))
        st = np.clip(np.nan_to_num(np.atleast_1d(np.asarray(steer, dtype=np.float32))), -300.0, 300.0)
        wheel = np.radians(st) / SR_
        out[:, 6] = np.clip(st / 10.0, -30.0, 30.0)
        out[:, 7] = wheel
        out[:, 8] = (st - np.nan_to_num(np.atleast_1d(np.asarray(prev_steer, dtype=np.float32)))) / GRID_DT / 100.0
        out[:, 9] = (v_c * np.tan(np.clip(wheel, -1.2, 1.2)) / L_) / 0.3
        out[:, 10] = L_ - 2.8
        out[:, 11] = SR_ / 15.0 - 1.0
        out[:, 12] = sm
    if "pedal" in blocks:
        sc = scales.get(str(static.get("brand"))) if scales else None
        g = np.atleast_1d(np.asarray(gas, dtype=np.float32))
        b = np.atleast_1d(np.asarray(brake, dtype=np.float32))
        if sc is None or not np.isfinite(g).any():
            out[:, 16] = 1.0
        else:
            glo, ghi, bhi = sc
            gn = np.clip((np.nan_to_num(g, nan=glo) - glo) / max(ghi - glo, 1e-6), -0.5, 1.5)
            out[:, 13] = gn
            out[:, 14] = (gn < 0.03).astype(np.float32)
            if bhi is not None and np.isfinite(bhi) and bhi > 0:
                out[:, 15] = np.clip(np.nan_to_num(b) / bhi, 0.0, 1.5)
            out[:, 16] = (~np.isfinite(g)).astype(np.float32)
    if "flags" in blocks:
        gpa = np.atleast_1d(np.asarray(gp, dtype=np.float32))
        bpa = np.atleast_1d(np.asarray(bp, dtype=np.float32))
        out[:, 17] = np.nan_to_num(gpa)
        out[:, 18] = np.nan_to_num(bpa)
        out[:, 19] = (~np.isfinite(gpa) | ~np.isfinite(bpa)).astype(np.float32)
    return out


# ---------------------------------------------------------------- numpy provider (online inference, no torch)
class SigProvider:
    """Causal tick-grid state machine like learn.provider.MotionProvider, for a signal-set model; adds turn_mag()."""

    def __init__(self, model_dir):
        from .provider import NumpyGRU
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        w = np.load(os.path.join(model_dir, "weights.npz"))
        self.group = meta["group"]
        self.name = f"sig_{self.group}_s{meta.get('seed', 0)}"
        self.vocab, self.scales = meta["vocab"], {k: tuple(v) for k, v in meta["scales"].items()}
        self.mean = np.array(meta["norm"]["mean"], dtype=np.float32)
        self.std = np.array(meta["norm"]["std"], dtype=np.float32)
        self.gru = NumpyGRU(w["gru.weight_ih_l0"], w["gru.weight_hh_l0"], w["gru.bias_ih_l0"], w["gru.bias_hh_l0"])
        self.head_w, self.head_b = w["head.weight"], w["head.bias"]
        self.plat_emb, self.brand_emb = w["plat.weight"], w["brand.weight"]
        self.needs = tuple(group_signals(self.group))

    def reset(self, init, static, theta_fallback=0.0):
        self.static = dict(wheelbase_m=static.wheelbase_m, steer_ratio=static.steer_ratio, brand=getattr(static, "brand", None))
        self.nominal_dt = 0.01
        self.pid = self.vocab["platform"].get(str(getattr(static, "platform", None)), 0)
        self.bid = self.vocab["brand"].get(str(getattr(static, "brand", None)), 0)
        self.x = self.y = 0.0
        self.th = init.theta0 if init.theta0 is not None else theta_fallback
        self.t0 = float(init.t0)
        self.k_next = int(math.ceil(self.t0 / GRID_DT))
        self.t_last_tick = self.t0
        self.ev_t = -np.inf
        self.ev = {s: np.nan for s in ("speed", "steer_angle", "gas", "brake", "gas_pressed", "brake_pressed")}
        self.prev_v = self.prev_s = None
        self.h = np.zeros(self.gru.H, dtype=np.float32)
        self.om, self.mag, self.scale, self.v_hold = 0.0, 0.0, 1.0, 0.0

    def _commit_tick(self, t_k):
        age = t_k - self.ev_t if np.isfinite(self.ev_t) else MAX_AGE_S * 10
        missing = float(age > MAX_AGE_S)
        e = self.ev
        v = e["speed"] if np.isfinite(e["speed"]) else 0.0
        s = e["steer_angle"] if np.isfinite(e["steer_angle"]) else 0.0
        pv = v if self.prev_v is None else self.prev_v
        ps = s if self.prev_s is None else self.prev_s
        f = sig_features(v, s, e["gas"], e["brake"], e["gas_pressed"], e["brake_pressed"], age, missing, pv, ps, self.static,
                         self.scales, self.nominal_dt, self.group)[0]
        fs = (f - self.mean) / self.std
        gate = f[4] * (1.0 - f[3])
        x = np.concatenate([fs, self.plat_emb[self.pid], self.brand_emb[self.bid]]).astype(np.float32)
        self.h = self.gru.step(x, self.h)
        c, u, sg = self.head_w @ self.h + self.head_b
        om = 0.6 * math.tanh(float(u)) * gate
        mag = 0.6 * gate / (1.0 + math.exp(-float(sg)))
        sc = math.exp(0.1 * math.tanh(float(c)))
        ds = v * sc * GRID_DT
        th_new = self.th + om * GRID_DT
        th_mid = th_new - 0.5 * om * GRID_DT
        self.x += ds * math.cos(th_mid)
        self.y += ds * math.sin(th_mid)
        self.th = th_new
        self.om, self.mag, self.scale, self.v_hold = om, mag, sc, v
        self.prev_v, self.prev_s = v, s
        self.t_last_tick = t_k

    def _commit_until(self, t, inclusive):
        while True:
            t_k = self.k_next * GRID_DT
            if (t_k <= t) if inclusive else (t_k < t):
                self._commit_tick(t_k)
                self.k_next += 1
            else:
                break

    def ingest(self, t, view):
        self._commit_until(t, inclusive=False)
        for k in self.ev:
            val = view.get(k, np.nan)
            if val is not None and np.isfinite(val):
                self.ev[k] = float(val)
        self.ev_t = float(t)
        return self.om, self.scale

    def sync(self, t_ref):
        self._commit_until(t_ref, inclusive=True)

    def rates(self):
        return self.om, self.scale

    def turn_mag(self):
        return self.mag

    def pose_at(self, t_ref):
        dt = max(float(t_ref - self.t_last_tick), 0.0)
        th = self.th + self.om * dt
        th_mid = th - 0.5 * self.om * dt
        v = self.v_hold * self.scale
        return np.array([self.x + v * dt * math.cos(th_mid), self.y + v * dt * math.sin(th_mid)]), th
