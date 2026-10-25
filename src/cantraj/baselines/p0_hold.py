"""p0-hold: the trivial control that stays at the start position (all eight protocols)."""
import numpy as np

from .base import OfflineBaseline, OnlineBaseline, Prediction


class P0Hold(OnlineBaseline, OfflineBaseline):
    name = "p0_hold"

    def reset(self, init, static, map_graph=None):
        self.theta = init.theta0

    def observe(self, events):
        pass

    def predict(self, t_ref):
        return Prediction(np.zeros(2), self.theta)

    def reconstruct(self, inputs, t_query):
        n = len(t_query)
        return np.zeros((n, 2)), None, None, None
