"""Whitelisted inputs handed to a baseline: EpisodeInit (t0, p0, optional theta0), StaticParams, EpisodeInputs.

The wrapper physically drops every column outside the signal group's whitelist (not merely hides it) and never
holds reference data. `__getitem__` raises ForbiddenSignalError for anything outside the whitelist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .events import EventStream
from .geo import LocalFrame
from .registry import COLUMN_MAP, ForbiddenSignalError, check_whitelist, group_signals, group_static


@dataclass(frozen=True)
class EpisodeInit:
    t0: float                   # episode start on the drive axis (s)
    p0_lat: float
    p0_lon: float
    theta0: Optional[float]     # math heading (rad, CCW from east) at t0 for H+ protocols, None for H-
    t_end: Optional[float] = None   # offline protocols only; None online (no sequence length leak)
    dx_m: float = 0.0               # start position given to the baseline, in the episode frame (non-zero only in the
    dy_m: float = 0.0               # start-shift ablation; the reference frame itself is never moved)

    @property
    def p0_xy(self):
        return (float(self.dx_m), float(self.dy_m))

    def frame(self):
        return LocalFrame(self.p0_lat, self.p0_lon)


@dataclass(frozen=True)
class StaticParams:
    model_canon: Optional[str] = None
    brand: Optional[str] = None
    platform: Optional[str] = None
    wheelbase_m: Optional[float] = None
    steer_ratio: Optional[float] = None
    center_to_front_m: Optional[float] = None
    mass_kg: Optional[float] = None
    yawrate_unit: Optional[str] = None
    yawrate_sign: Optional[int] = None
    angle_offset_train_deg: float = 0.0
    steer_ratio_eff_train: Optional[float] = None

    def restricted(self, allowed):
        """Copy with every field outside `allowed` (plus model_canon/brand) set to None. The training-fold steering
        calibration (offset, effective ratio) travels with steer_ratio."""
        keep = set(allowed) | {"model_canon", "brand", "platform"}
        if "steer_ratio" in keep:
            keep |= {"angle_offset_train_deg", "steer_ratio_eff_train"}
        out = {}
        for k, v in self.__dict__.items():
            if k in keep:
                out[k] = v
            else:
                out[k] = 0.0 if k == "angle_offset_train_deg" else None
        return StaticParams(**out)


class EpisodeInputs:
    """Whitelisted event stream (canonical names) for one episode, group and protocol."""

    def __init__(self, gid: str, events: EventStream, static: StaticParams, init: EpisodeInit, input_layer="events",
                 map_graph=None, warmup: EventStream | None = None):
        self.gid = gid
        self.allowed = list(group_signals(gid))
        extra = [n for n in events.names() if n not in self.allowed]
        if extra:                                  # physically drop anything outside the whitelist
            events = events.select([n for n in events.names() if n in self.allowed])
        check_whitelist(events.names(), gid)
        self.events = events
        # warm-start variant (research protocol rows): the drive's own CAN/IMU events before t0, same whitelist, never GPS
        self.warmup = None
        if warmup is not None and len(warmup):
            if float(np.max(warmup.t_s)) >= float(init.t0):
                raise ValueError("warmup events must be strictly before t0")
            keep = [n for n in warmup.names() if n in self.allowed]
            self.warmup = warmup.select(keep) if keep != warmup.names() else warmup
        self.static = static.restricted(group_static(gid))
        self.init = init
        self.input_layer = input_layer
        self.map_graph = map_graph

    def __getitem__(self, name):
        if name not in self.allowed:
            raise ForbiddenSignalError(f"{name!r} is not in the whitelist of {self.gid}")
        if name not in self.events.cols:
            raise KeyError(f"{name!r} allowed for {self.gid} but not present on this drive")
        return self.events.cols[name]

    @property
    def t_s(self):
        return self.events.t_s

    def has(self, name):
        return name in self.events.cols

    def available(self):
        return self.events.names()


def canonicalize(df, layer="native", table="carstate"):
    """Rename raw columns of a native table (or the released grid) to canonical signal names; drop the rest."""
    col_idx = 0 if layer == "native" else 1
    out = {}
    for canon, cols in COLUMN_MAP.items():
        c = cols[col_idx]
        if c is None:
            continue
        if ":" in c:                               # other native tables (controls:curvature, imu:gyro:v2)
            parts = c.split(":")
            if parts[0] != table:
                continue
            c = parts[-1]
        if c in df.columns:
            out[canon] = df[c].to_numpy(dtype=float)
    return out
