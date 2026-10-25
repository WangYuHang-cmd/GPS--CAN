"""Learned dead reckoning (G3): online wrapper around a trained MotionProvider. Imports no reference loader."""
from __future__ import annotations

import numpy as np

from ..learn.provider import MotionProvider
from .base import OfflineBaseline, OnlineBaseline, Prediction
from .heading_prior import rotations


class LearnedDR(OnlineBaseline, OfflineBaseline):
    needs = ("speed", "steer_angle")

    def __init__(self, model_dir, theta_fallback=0.0, K_headings=36):
        self.provider = MotionProvider(model_dir)
        self.name = self.provider.name
        self.theta_fallback = theta_fallback
        self.K = K_headings

    def reset(self, init, static, map_graph=None):
        self._init, self._static = init, static
        self.provider.reset(init, static, self.theta_fallback)
        self.blind = init.theta0 is None

    def warmup(self, events):
        if len(events) == 0:
            return
        self.provider.warm_start(self._init, self._static, events.t_s, events.cols["speed"], events.cols["steer_angle"], self.theta_fallback)

    def observe(self, events):
        if len(events) == 0:
            return
        for n in self.needs:
            if n not in events.cols:
                from ..registry import ForbiddenSignalError
                raise ForbiddenSignalError(f"{self.name} needs {n!r}, not in inputs {events.names()}")
        v, s = events.cols["speed"], events.cols["steer_angle"]
        for i, t in enumerate(events.t_s):
            self.provider.ingest(float(t), {"speed": float(v[i]), "steer_angle": float(s[i])})

    def predict(self, t_ref):
        self.provider.sync(float(t_ref))
        xy, th = self.provider.pose_at(float(t_ref))
        return Prediction(xy, th)

    def reconstruct(self, inputs, t_query):
        self.reset(inputs.init, inputs.static, inputs.map_graph)
        ev = inputs.events
        xy = np.empty((len(t_query), 2))
        th = np.empty(len(t_query))
        i = 0
        for q, t in enumerate(t_query):
            j = int(np.searchsorted(ev.t_s, t, side="right"))
            if j > i:
                from ..events import EventStream
                self.observe(EventStream(ev.t_s[i:j], {n: c[i:j] for n, c in ev.cols.items()}, ev.nominal_dt))
                i = j
            p = self.predict(float(t))
            xy[q], th[q] = p.xy, p.theta
        if self.blind:
            cands, probs = rotations(xy, self.K)
            return xy, th, cands, probs
        return xy, th, None, None
