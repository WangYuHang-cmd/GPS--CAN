"""Data-quality helpers shared by the audit scripts and the episode builder.

`hole_mask` is a verbatim port of extract_gps.py:417-430 (exactly linear released speed for >= min_run samples and
not all zero = a stretch the July decode interpolated across). On the released 10 Hz table it is a DIAGNOSTIC of the
interpolation artefact; the event layer derives missingness from real message gaps instead (`event_gaps`).
"""
from __future__ import annotations

import numpy as np

HOLE_MIN_RUN = 30          # samples (3 s)


def hole_mask(v_rel, min_run=HOLE_MIN_RUN):
    v_rel = np.asarray(v_rel, dtype=float)
    n = len(v_rel)
    mask = np.zeros(n, dtype=bool)
    if n < min_run + 2:
        return mask
    lin = np.concatenate([[False], np.abs(np.diff(v_rel, 2)) < 1e-6, [False]])
    edges = np.flatnonzero(np.diff(np.concatenate([[0], lin.astype(np.int8), [0]])))
    for a, b in zip(edges[::2], edges[1::2]):
        lo, hi = max(a - 1, 0), min(b + 1, n)
        if b - a >= min_run and not np.all(v_rel[lo:hi] == 0.0):
            mask[lo:hi] = True
    return mask


def hole_mask_multi(*cols, min_run=HOLE_MIN_RUN):
    """Hole where EVERY given column is exactly linear at the same time (stricter than one column)."""
    m = None
    for c in cols:
        mc = hole_mask(c, min_run)
        m = mc if m is None else (m & mc)
    return m if m is not None else np.zeros(0, dtype=bool)


def event_gaps(t_s, nominal_dt=None, factor=3.0, min_gap_s=0.3):
    """Gaps in an event stream: intervals (t_a, t_b) between consecutive messages longer than
    max(factor * nominal_dt, min_gap_s). nominal_dt defaults to the median interval. Returns an (n, 2) array."""
    t = np.sort(np.asarray(t_s, dtype=float))
    if len(t) < 2:
        return np.zeros((0, 2))
    d = np.diff(t)
    if nominal_dt is None:
        nominal_dt = float(np.median(d))
    thr = max(factor * nominal_dt, min_gap_s)
    i = np.flatnonzero(d > thr)
    return np.column_stack([t[i], t[i + 1]])


def in_gaps(t_query, gaps):
    """Boolean mask: query time falls strictly inside one of the (t_a, t_b) gaps."""
    t = np.asarray(t_query, dtype=float)
    out = np.zeros(len(t), dtype=bool)
    for a, b in gaps:
        out |= (t > a) & (t < b)
    return out
