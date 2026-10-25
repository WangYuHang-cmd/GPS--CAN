"""Start prior: p0 plus the training-fold median net-displacement ratio rho(s) times the CAN distance, along theta0
(H+ only; H- degenerates to p0-hold)."""
import json
import math
import os

import numpy as np

from .. import paths
from .base import OfflineBaseline, OnlineBaseline, Prediction

LIB = f"{paths.BENCH}/library/prior_rho.json"


class StartPrior(OnlineBaseline, OfflineBaseline):
    name = "prior"
    needs = ("speed",)

    def __init__(self):
        d = json.load(open(LIB))
        self.bins = np.array(d["bins_m"], dtype=float)
        self.rho = np.array(d["rho"], dtype=float)

    def _rho(self, s):
        k = min(int(np.searchsorted(self.bins, s, side="right") - 1), len(self.rho) - 1)
        r = self.rho[max(k, 0)]
        return float(r) if np.isfinite(r) else 0.5

    def reset(self, init, static, map_graph=None):
        self.th = init.theta0
        self.s = 0.0
        self.t_last = init.t0
        self.v = 0.0

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

    def predict(self, t_ref):
        if self.th is None:
            return Prediction(np.zeros(2), None)
        d = self._rho(self.s) * self.s
        return Prediction(np.array([d * math.cos(self.th), d * math.sin(self.th)]), self.th)

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
