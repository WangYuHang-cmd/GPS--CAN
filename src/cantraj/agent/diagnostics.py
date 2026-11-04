"""Observable diagnostics of candidate paths (agent study). Everything here is computed from the protocol-allowed
inputs only: the candidate positions at the query times, the CAN-integrated distance s(t) and dead-reckoned relative
heading psi(t) (map_beam.integrate_can), the given start heading theta0 and the CAN speed. No reference enters.

Conventions: positions are episode ENU metres (x east, y north, start at the origin); math headings are radians
counter-clockwise from east; the LLM-facing layer converts to compass degrees (clockwise from north).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

TURN_DEG = 45.0        # bearing change that counts as a turn ...
TURN_WIN_M = 40.0      # ... when it happens within this path distance
UTURN_DEG = 150.0
MATCH_M = 50.0         # DR turn and path turn are the same event when their path distances differ by at most this
OVERLAP_M = 15.0       # a sampled point lies on the other path if within this distance
DIVERGE_M = 50.0
MOVING_MPS = 2.0


def wrap(a):
    return (np.asarray(a, dtype=float) + np.pi) % (2 * np.pi) - np.pi


def compass_deg(theta_math):
    """Math heading (rad, CCW from east) -> compass degrees (clockwise from north) in [0, 360)."""
    return float((90.0 - math.degrees(float(theta_math))) % 360.0)


def arclength(xy):
    d = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def step_bearings(xy, min_seg_m=1.0):
    """Bearing of each step i -> i+1 (rad), NaN for steps shorter than min_seg_m; length len(xy) - 1."""
    dx, dy = np.diff(xy[:, 0]), np.diff(xy[:, 1])
    b = np.arctan2(dy, dx)
    b[np.hypot(dx, dy) < min_seg_m] = np.nan
    return b


def _unwrap_nan(b):
    """Unwrap a bearing series with NaNs (NaNs filled by the previous valid value before unwrapping)."""
    b = np.asarray(b, dtype=float).copy()
    ok = np.isfinite(b)
    if not ok.any():
        return b
    first = int(np.argmax(ok))
    b[:first] = b[first]
    for i in range(1, len(b)):
        if not np.isfinite(b[i]):
            b[i] = b[i - 1]
    return np.unwrap(b)


def turn_events(bearing_unwrapped, s, turn_deg=TURN_DEG, win_m=TURN_WIN_M):
    """Turns = places where the unwrapped bearing changes by more than turn_deg within win_m of path distance.
    bearing_unwrapped and s are aligned (same length). Returns [(s_start, signed_angle_deg)] (left positive)."""
    b, s = np.asarray(bearing_unwrapped, dtype=float), np.asarray(s, dtype=float)
    if len(b) < 2 or not np.isfinite(b).any():
        return []
    s_end = s + win_m
    b_end = np.interp(s_end, s, b)
    d = np.degrees(b_end - b)
    mark = np.abs(d) > turn_deg
    events, i = [], 0
    while i < len(mark):
        if mark[i]:
            j = i
            while j + 1 < len(mark) and mark[j + 1] and np.sign(d[j + 1]) == np.sign(d[i]):
                j += 1
            # total change over the marked stretch plus the window
            total = float(np.degrees(np.interp(s[j] + win_m, s, b) - b[i]))
            events.append((float(s[i]), total))
            # skip until the bearing has settled: next index whose window starts after s[j] + win_m
            i = int(np.searchsorted(s, s[j] + win_m, side="right"))
            i = max(i, j + 1)
        else:
            i += 1
    return events


def match_turns(a, b, tol_m=MATCH_M):
    """Count turns of list a with no same-sign turn of list b within tol_m (and vice versa)."""
    def unmatched(x, y):
        n = 0
        for s, ang in x:
            if not any(abs(s - s2) <= tol_m and np.sign(ang) == np.sign(ang2) for s2, ang2 in y):
                n += 1
        return n
    return unmatched(a, b) + unmatched(b, a)


def densify(xy, spacing=5.0):
    s = arclength(xy)
    if s[-1] <= spacing:
        return xy
    q = np.arange(0.0, s[-1], spacing)
    return np.column_stack([np.interp(q, s, xy[:, 0]), np.interp(q, s, xy[:, 1])])


class PathSet:
    """Candidate paths of one window with their geometry caches (for pairwise diagnostics)."""

    def __init__(self, paths: dict):
        self.paths = {k: np.asarray(v, dtype=float) for k, v in paths.items()}
        self._dense = {k: densify(v) for k, v in self.paths.items()}
        self._tree = {k: cKDTree(d) for k, d in self._dense.items()}
        self._s = {k: arclength(v) for k, v in self.paths.items()}

    def overlap(self, a, b, tol_m=OVERLAP_M):
        """Share of a's densified points within tol_m of path b."""
        d, _ = self._tree[b].query(self._dense[a])
        return float(np.mean(d <= tol_m))

    def end_distance(self, a, b):
        return float(np.hypot(*(self.paths[a][-1] - self.paths[b][-1])))

    def divergence(self, a, b, tol_m=DIVERGE_M):
        """First path distance along a where a is more than tol_m from b (None if never)."""
        d, _ = self._tree[b].query(self.paths[a])
        idx = np.flatnonzero(d > tol_m)
        if len(idx) == 0:
            return None
        return int(idx[0]), float(self._s[a][idx[0]])


def candidate_diagnostics(xy, s_q, psi_q, theta0, v_q):
    """Single-candidate diagnostics (dict of floats / ints)."""
    xy = np.asarray(xy, dtype=float)
    s_path = arclength(xy)
    L = float(s_path[-1])
    s_can = float(max(s_q[-1], 0.0))
    out = dict(path_len_m=L, can_dist_m=s_can, len_ratio_dev=abs(L / s_can - 1.0) if s_can > 1.0 else float("nan"))
    # heading consistency: path step bearing vs DR heading theta0 + psi at the step midpoint, moving steps only
    b = step_bearings(xy)
    theta_dr = theta0 + 0.5 * (psi_q[:-1] + psi_q[1:])
    mv = (0.5 * (v_q[:-1] + v_q[1:]) > MOVING_MPS) & np.isfinite(b)
    if mv.sum() >= 3:
        err = wrap(b[mv] - theta_dr[mv])
        out["hd_abs_rms_deg"] = float(np.degrees(np.sqrt(np.mean(err ** 2))))
        out["hd_abs_p90_deg"] = float(np.degrees(np.quantile(np.abs(err), 0.9)))
        off = math.atan2(float(np.mean(np.sin(err))), float(np.mean(np.cos(err))))
        out["hd_offset_deg"] = float(math.degrees(off))
        drift = wrap(err - off)
        out["hd_drift_rms_deg"] = float(np.degrees(np.sqrt(np.mean(drift ** 2))))
        out["n_moving_steps"] = int(mv.sum())
    else:
        out.update(hd_abs_rms_deg=float("nan"), hd_abs_p90_deg=float("nan"), hd_offset_deg=float("nan"), hd_drift_rms_deg=float("nan"), n_moving_steps=int(mv.sum()))
    # turns along the path and along the DR heading, both over path distance
    bu = _unwrap_nan(b)
    tp = turn_events(bu, s_path[:-1]) if len(bu) else []
    td = turn_events(np.unwrap(theta0 + np.asarray(psi_q, dtype=float)), np.asarray(s_q, dtype=float))
    out["turns_path"] = len(tp)
    out["turns_dr"] = len(td)
    out["turns_unmatched"] = match_turns(tp, td)
    out["uturns"] = int(sum(1 for _, ang in tp if abs(ang) > UTURN_DEG))
    out["path_turns"] = [dict(at_m=round(s, 0), deg=round(ang, 0)) for s, ang in tp[:12]]
    end = xy[-1]
    out["end_east_m"], out["end_north_m"] = float(end[0]), float(end[1])
    out["net_disp_m"] = float(np.hypot(end[0], end[1]))
    return out


def dr_turn_list(s_q, psi_q, theta0, t_rel):
    """DR turn events with their time since t0 (seconds) and path distance."""
    td = turn_events(np.unwrap(theta0 + np.asarray(psi_q, dtype=float)), np.asarray(s_q, dtype=float))
    out = []
    for s, ang in td:
        t = float(np.interp(s, s_q, t_rel))
        out.append(dict(t_s=round(t, 0), at_m=round(s, 0), deg=round(ang, 0), dir="left" if ang > 0 else "right"))
    return out


def motion_summary(events, s_q, psi_q, theta0, t0, t_end, t_rel):
    """Window-level summary from the CAN stream only (no reference): distance, speeds, stops, DR turns, event gaps."""
    v = np.asarray(events.cols["speed"], dtype=float)
    t = np.asarray(events.t_s, dtype=float)
    fin = np.isfinite(v)
    out = dict(duration_s=round(float(t_end - t0), 0), can_distance_m=round(float(s_q[-1]), 0),
               speed_mean_mps=round(float(np.mean(v[fin])), 1) if fin.any() else None,
               speed_max_mps=round(float(np.max(v[fin])), 1) if fin.any() else None,
               start_heading_compass_deg=round(compass_deg(theta0), 0),
               end_heading_dr_compass_deg=round(compass_deg(theta0 + float(psi_q[-1])), 0),
               total_heading_change_deg=round(float(np.degrees(np.sum(np.abs(np.diff(np.asarray(psi_q, dtype=float)))))), 0),
               n_events=int(len(t)), event_rate_hz=round(1.0 / events.nominal_dt, 1) if events.nominal_dt else None)
    # stops: speed below 0.5 m/s for at least 3 s
    stops, in_stop, t_start = 0, False, 0.0
    for ti, vi in zip(t[fin], v[fin]):
        if vi < 0.5 and not in_stop:
            in_stop, t_start = True, ti
        elif vi >= 0.5 and in_stop:
            in_stop = False
            if ti - t_start >= 3.0:
                stops += 1
    if in_stop and t[fin][-1] - t_start >= 3.0:
        stops += 1
    out["n_stops"] = stops
    gaps = events.gaps() if len(t) > 1 else []
    gaps = [g for g in gaps if (g[1] - g[0]) >= 1.0] if gaps is not None and len(gaps) else []
    out["can_gaps_over_1s"] = len(gaps)
    out["can_gap_total_s"] = round(float(sum(g[1] - g[0] for g in gaps)), 1) if gaps else 0.0
    out["dr_turns"] = dr_turn_list(s_q, psi_q, theta0, t_rel)
    return out
