"""Baseline interfaces. Online: reset / observe / predict (causal). Offline: reconstruct over the whole episode."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..events import EventStream
from ..inputs import EpisodeInit, EpisodeInputs, StaticParams


@dataclass
class Prediction:
    xy: np.ndarray                      # (2,) episode ENU metres
    theta: Optional[float] = None
    cands: Optional[np.ndarray] = None  # (K, 2) whole-path candidates' positions at this time (offline only)
    probs: Optional[np.ndarray] = None  # (K,)


class OnlineBaseline:
    name = "base"
    needs = ()          # canonical signals this method reads (checked against the whitelist at reset)

    def reset(self, init: EpisodeInit, static: StaticParams, map_graph=None):
        raise NotImplementedError

    def observe(self, events: EventStream):
        raise NotImplementedError

    def warmup(self, events: EventStream):
        """Warm-start variant: events of the same drive strictly before t0 (same whitelist, no GPS). Baselines may use
        them to calibrate their state (steering offset, gyro bias, recurrent state) but must not move the start pose.
        Default: ignore."""
        return None

    def predict(self, t_ref: float) -> Prediction:
        raise NotImplementedError


class OfflineBaseline:
    name = "base"

    def reconstruct(self, inputs: EpisodeInputs, t_query: np.ndarray):
        """Return (xy (T,2), theta (T,) or None, cands (K,T,2) or None, probs (K,) or None)."""
        raise NotImplementedError
