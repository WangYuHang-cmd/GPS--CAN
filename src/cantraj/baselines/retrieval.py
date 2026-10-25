"""Training-route retrieval WITHOUT driver identity.

Library: 1 Hz reference tracks of training-fold drives (scripts/build_library.py) with cumulative CAN distance and CAN
speed along the track. Query: p0 and the causal CAN prefix (speed -> cumulative distance s(t), speed profile).
Candidates: library tracks passing within `radius_m` of p0 (both traversal directions). The predicted position at time
t is the library point `s(t)` metres further along the candidate from its point nearest to p0. Candidates are ranked
online by the RMSE between the query's speed profile and the candidate's CAN speed over the same distance (the
"same road, same stops" signature); Top-1 = best so far. Leave-one-driver-out when the query driver is in the library
(pilot runs on training drives); no candidate -> hold p0.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from .. import paths
from ..geo import LocalFrame
from .base import OfflineBaseline, OnlineBaseline, Prediction

LIB = f"{paths.BENCH}/library/train_tracks.parquet"
CELL_DEG = 0.00225


class RouteRetrieval(OnlineBaseline, OfflineBaseline):
    name = "retrieval"
    needs = ("speed",)
    _lib = None

    def __init__(self, radius_m=100.0, max_candidates=50, rerank_every_s=5.0):
        self.radius_m, self.max_candidates, self.rerank_every_s = radius_m, max_candidates, rerank_every_s
        if RouteRetrieval._lib is None:
            lib = pd.read_parquet(LIB)
            RouteRetrieval._lib = lib
            RouteRetrieval._cells = lib.groupby(["cx", "cy"]).indices
        self.lib = RouteRetrieval._lib

    # ---- candidate construction
    def _candidates(self, lat0, lon0, exclude_driver):
        cy = int(round(lat0 / CELL_DEG))
        cx = int(round(lon0 * math.cos(math.radians(lat0)) / CELL_DEG))
        idx = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                idx.extend(RouteRetrieval._cells.get((cx + dx, cy + dy), []))
        if not idx:
            return []
        pts = self.lib.iloc[idx]
        fr = LocalFrame(lat0, lon0)
        x, y = fr.to_enu(pts.lat.values, pts.lon.values)
        pts = pts.assign(d=np.hypot(x, y))
        pts = pts[(pts.d <= self.radius_m) & (pts.driver != exclude_driver)]
        cands = []
        for (drv, drive), g in pts.groupby(["driver", "drive"]):
            j = g.d.idxmin()
            track = self.lib[(self.lib.driver == drv) & (self.lib.drive == drive)].sort_values("i")
            k0 = int(track.index.get_loc(j))
            tx, ty = fr.to_enu(track.lat.values, track.lon.values)
            s = track.s_m.values
            v = track.v_can.values
            for direction in (+1, -1):
                if direction > 0:
                    ss, xx, yy, vv = s[k0:] - s[k0], tx[k0:], ty[k0:], v[k0:]
                else:
                    ss, xx, yy, vv = (s[k0] - s[:k0 + 1])[::-1], tx[:k0 + 1][::-1], ty[:k0 + 1][::-1], v[:k0 + 1][::-1]
                if len(ss) < 5 or ss[-1] < 50:
                    continue
                cands.append(dict(key=(drv, drive, direction), s=ss, x=xx, y=yy, v=vv, score=0.0))
        cands.sort(key=lambda c: -c["s"][-1])
        return cands[: self.max_candidates]

    # ---- online interface
    def reset(self, init, static, map_graph=None):
        self.cands = self._candidates(init.p0_lat, init.p0_lon, exclude_driver=getattr(init, "exclude_driver", None) or self._exclude)
        self.s = 0.0
        self.t_last = init.t0
        self.v = 0.0
        self.prof_s, self.prof_v = [], []          # query speed profile along distance
        self.best = 0
        self.t_rank = init.t0

    _exclude = None

    def observe(self, events):
        if "speed" not in events.cols:
            return
        for t, v in zip(events.t_s, events.cols["speed"]):
            dt = float(t - self.t_last)
            if 0 <= dt <= 1.0:
                self.s += self.v * dt
            self.t_last = float(t)
            if math.isfinite(v):
                self.v = float(v)
            if not self.prof_s or self.s - self.prof_s[-1] >= 10.0:
                self.prof_s.append(self.s)
                self.prof_v.append(self.v)

    def _rank(self):
        if not self.cands or len(self.prof_s) < 3:
            return
        qs, qv = np.array(self.prof_s), np.array(self.prof_v)
        for c in self.cands:
            k = qs <= c["s"][-1]
            if k.sum() < 3:
                c["score"] = 1e9
                continue
            cv = np.interp(qs[k], c["s"], c["v"])
            c["score"] = float(np.sqrt(np.mean((cv - qv[k]) ** 2))) + 2.0 * (1 - k.mean())
        self.best = int(np.argmin([c["score"] for c in self.cands]))

    def predict(self, t_ref):
        if not self.cands:
            return Prediction(np.zeros(2), None)
        if t_ref - self.t_rank >= self.rerank_every_s:
            self._rank()
            self.t_rank = t_ref
        c = self.cands[self.best]
        s = min(self.s, c["s"][-1])
        return Prediction(np.array([np.interp(s, c["s"], c["x"]), np.interp(s, c["s"], c["y"])]), None)

    def reconstruct(self, inputs, t_query):
        self.reset(inputs.init, inputs.static)
        ev = inputs.events
        xy = np.zeros((len(t_query), 2))
        i = 0
        for q, t in enumerate(t_query):
            j = int(np.searchsorted(ev.t_s, t, side="right"))
            if j > i:
                from ..events import EventStream
                self.observe(EventStream(ev.t_s[i:j], {n: v[i:j] for n, v in ev.cols.items()}, ev.nominal_dt))
                i = j
            xy[q] = self.predict(t).xy
        return xy, None, None, None
