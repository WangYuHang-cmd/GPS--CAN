"""Perturbation probe (research ablation, not part of the frozen main board): degrade the CAN inputs a baseline sees.

Spec string, comma separated, e.g. "speed_quant=1,speed_noise=0.5,steer_hz=1":
  speed_quant=Q   quantise speed to multiples of Q m/s (round half to even)
  speed_noise=S   add zero-mean Gaussian noise with std S m/s to every speed event (seeded per episode; the v1 probe)
  speed_hz=F      hold the speed at the value sampled at the start of each 1/F s bin (causal zero-order hold)
  steer_quant=Q   quantise the steering-wheel angle to multiples of Q degrees (round half to even)
  steer_noise=S   add a zero-mean Gaussian offset with std S degrees that is constant within each 1 s bin of the DRIVE
                  time axis and drawn from (drive, bin, spec, release seed): overlapping windows of one drive see the
                  same perturbed signal, as if one degraded log had been released; at 100 Hz, per-event white noise
                  would average out in the integration and probe nothing
  steer_hz=F      hold the steering angle at the value sampled at the start of each 1/F s bin (causal zero-order hold)
  steer_delay=D   report at time t the steering angle logged at or before t - D seconds (a time-misalignment probe,
                  not a resolution probe; an offline attacker who knows D can undo it)
Per signal the operations are applied in a fixed order whatever the spec string says: noise -> quantise -> hold -> delay.
Only the inputs are changed; the reference, anchors and initialisation are untouched. Every operation is causal: the value
at time t depends only on events at or before t.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .events import EventStream

KNOWN = ("speed_quant", "speed_noise", "speed_hz", "steer_quant", "steer_noise", "steer_hz", "steer_delay")
NOISE_BIN_S = 1.0


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


def _hold(t: np.ndarray, x: np.ndarray, hz: float) -> np.ndarray:
    """Causal zero-order hold: every event of a 1/hz s bin takes the value of the bin's first event (events are
    time-sorted, so that event is at or before it)."""
    period = 1.0 / hz
    bins = np.floor(t / period).astype(np.int64)
    first = np.r_[True, bins[1:] != bins[:-1]]
    idx = np.maximum.accumulate(np.where(first, np.arange(len(t)), 0))
    return x[idx]


def _quantise(x: np.ndarray, q: float) -> np.ndarray:
    return np.where(np.isfinite(x), np.round(x / q) * q, x)


def _binned_offsets(t: np.ndarray, std: float, key: str) -> np.ndarray:
    """One Gaussian offset per NOISE_BIN_S bin of the drive time axis, a pure function of (key, bin)."""
    bins = np.floor(t / NOISE_BIN_S).astype(np.int64)
    uniq, inv = np.unique(bins, return_inverse=True)
    vals = np.empty(len(uniq))
    for i, b in enumerate(uniq):
        h = hashlib.sha256(f"{key}|{int(b)}".encode()).digest()
        vals[i] = np.random.default_rng(int.from_bytes(h[:8], "little")).normal(0.0, std)
    return vals[inv]


def _delay(t: np.ndarray, x: np.ndarray, d: float) -> np.ndarray:
    """Value logged at or before t - d; before the first such event the first value is held."""
    j = np.searchsorted(t, t - d, side="right") - 1
    return x[np.maximum(j, 0)]


def perturb_stream(ev: EventStream, spec: dict, rng: np.random.Generator, drive_key: str = "", release_seed: int = 0) -> EventStream:
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
            v = _quantise(v, q)
        f = spec.get("speed_hz", 0.0)
        if f > 0:
            v = _hold(t, v, f)
        cols["speed"] = v
    if "steer_angle" in cols:
        st = cols["steer_angle"]
        s = spec.get("steer_noise", 0.0)
        if s > 0:
            key = f"{drive_key}|steer_angle|steer_noise={s:g}|seed={release_seed}"
            st = np.where(np.isfinite(st), st + _binned_offsets(t, s, key), st)
        q = spec.get("steer_quant", 0.0)
        if q > 0:
            st = _quantise(st, q)
        f = spec.get("steer_hz", 0.0)
        if f > 0:
            st = _hold(t, st, f)
        d = spec.get("steer_delay", 0.0)
        if d > 0:
            st = _delay(t, st, d)
        cols["steer_angle"] = st
    return EventStream(t.copy(), cols, ev.nominal_dt)
