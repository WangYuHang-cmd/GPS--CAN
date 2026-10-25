"""Perturbation probe (research ablation, not part of the frozen main board): degrade the CAN inputs a baseline sees.

Spec string, comma separated, e.g. "speed_quant=1,speed_noise=0.5,steer_hz=1":
  speed_quant=Q   quantise speed to multiples of Q m/s (round half to even)
  speed_noise=S   add zero-mean Gaussian noise with std S m/s to speed (seeded per episode)
  steer_hz=F      hold the steering angle at the value sampled at the start of each 1/F s bin (causal zero-order hold)
Only the inputs are changed; the reference, anchors and initialisation are untouched. Every operation is causal: the value
at time t depends only on events at or before t.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .events import EventStream

KNOWN = ("speed_quant", "speed_noise", "steer_hz")


def parse_spec(spec: str | None) -> dict:
    out = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=")
        k = k.strip()
        if k not in KNOWN:
            raise ValueError(f"unknown perturbation {k!r}; known: {KNOWN}")
        out[k] = float(v)
    return out


def episode_rng(episode_id: str, spec: str) -> np.random.Generator:
    """Deterministic generator per (episode, spec): results do not depend on worker assignment or run order."""
    h = hashlib.sha256(f"{episode_id}|{spec}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def perturb_stream(ev: EventStream, spec: dict, rng: np.random.Generator) -> EventStream:
    if not spec:
        return ev
    cols = {k: v.copy() for k, v in ev.cols.items()}
    t = ev.t_s
    if "speed" in cols:
        v = cols["speed"]
        if spec.get("speed_noise", 0.0) > 0:
            fin = np.isfinite(v)
            v = v + np.where(fin, rng.normal(0.0, spec["speed_noise"], len(v)), 0.0)
            v = np.where(fin, np.maximum(v, 0.0), v)          # speed stays non-negative
        q = spec.get("speed_quant", 0.0)
        if q > 0:
            v = np.where(np.isfinite(v), np.round(v / q) * q, v)
        cols["speed"] = v
    f = spec.get("steer_hz", 0.0)
    if f > 0 and "steer_angle" in cols:
        st = cols["steer_angle"]
        period = 1.0 / f
        bins = np.floor(t / period).astype(np.int64)
        # first event of each bin (events are time-sorted): hold its value for the whole bin -> causal ZOH
        first = np.r_[True, bins[1:] != bins[:-1]]
        idx = np.maximum.accumulate(np.where(first, np.arange(len(t)), 0))
        cols["steer_angle"] = st[idx]
    return EventStream(t.copy(), cols, ev.nominal_dt)
