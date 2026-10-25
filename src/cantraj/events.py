"""Event layer: raw CAN messages at their logged times (the online-protocol input).

An `EventStream` holds one time axis `t_s` (seconds on the released drive axis, strictly increasing after sorting)
and named float columns. `causal_view(t_query)` returns, per column, the value of the last event whose logged time is
<= t_query (zero-order hold) and that event's time; nothing recorded after t_query is ever used. `causal_grid` builds
a regular grid of such views for learning baselines (with `age_s` = t_query - t_last and `missing` inside gaps).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quality import event_gaps


@dataclass
class EventStream:
    t_s: np.ndarray
    cols: dict = field(default_factory=dict)
    nominal_dt: float | None = None

    def __post_init__(self):
        self.t_s = np.asarray(self.t_s, dtype=float)
        order = np.argsort(self.t_s, kind="stable")
        self.t_s = self.t_s[order]
        self.cols = {k: np.asarray(v, dtype=float)[order] for k, v in self.cols.items()}
        if self.nominal_dt is None and len(self.t_s) > 1:
            self.nominal_dt = float(np.median(np.diff(self.t_s)))

    def __len__(self):
        return len(self.t_s)

    def names(self):
        return list(self.cols)

    def select(self, names):
        missing = [n for n in names if n not in self.cols]
        if missing:
            raise KeyError(f"signals not in stream: {missing}")
        return EventStream(self.t_s.copy(), {n: self.cols[n].copy() for n in names}, self.nominal_dt)

    def window(self, t_lo, t_hi):
        """Events with t_lo <= t_s <= t_hi."""
        k = (self.t_s >= t_lo) & (self.t_s <= t_hi)
        return EventStream(self.t_s[k], {n: v[k] for n, v in self.cols.items()}, self.nominal_dt)

    def gaps(self, factor=3.0, min_gap_s=0.3):
        return event_gaps(self.t_s, self.nominal_dt, factor, min_gap_s)

    def causal_index(self, t_query):
        """Index of the last event with t_s <= t_query (-1 if none)."""
        return np.searchsorted(self.t_s, np.asarray(t_query, dtype=float), side="right") - 1

    def causal_view(self, t_query):
        """{name: value} of the last event at or before t_query plus 't_last' (NaN if none)."""
        i = int(self.causal_index(t_query))
        if i < 0:
            return {n: np.nan for n in self.cols} | {"t_last": np.nan}
        return {n: float(v[i]) for n, v in self.cols.items()} | {"t_last": float(self.t_s[i])}

    def causal_grid(self, t_grid, max_age_s=None):
        """Zero-order-hold views on a grid. Returns dict of arrays: each column, 'age_s', 'missing'
        (no event yet, or age > max_age_s, default 3 x nominal_dt or 0.3 s whichever is larger)."""
        t_grid = np.asarray(t_grid, dtype=float)
        idx = self.causal_index(t_grid)
        has = idx >= 0
        out = {}
        for n, v in self.cols.items():
            a = np.full(len(t_grid), np.nan)
            a[has] = v[idx[has]]
            out[n] = a
        age = np.full(len(t_grid), np.nan)
        age[has] = t_grid[has] - self.t_s[idx[has]]
        if max_age_s is None:
            max_age_s = max(3.0 * (self.nominal_dt or 0.1), 0.3)
        out["age_s"] = age
        out["missing"] = ~has | (age > max_age_s)       # age > max_age_s <=> t_grid lies inside an event gap
        return out


def from_frame(df, t_col="t_s", names=None):
    names = names or [c for c in df.columns if c != t_col and np.issubdtype(df[c].dtype, np.number)]
    return EventStream(df[t_col].to_numpy(dtype=float), {n: df[n].to_numpy(dtype=float) for n in names})


def grid_to_events(df, t_col="time_s", names=None):
    """Treat a released 10 Hz grid table as an event stream (compatibility layer `grid_compat`): every grid row is
    an event at its grid time. Interpolated values are NOT causal in general; use only for offline / compat runs."""
    return from_frame(df, t_col, names)
