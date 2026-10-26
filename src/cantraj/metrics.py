"""Metrics for absolute 2-D trajectory recovery (Plan.md 2.2 item 10).

Per episode: ADE, FDE, coverage C_r (share of scored reference points within r metres), success S_(r,q) = [C_r >= q],
moving-only and input-gap strata, error binned by elapsed time and by CAN distance. Aggregation: mean/median over
episodes with cluster bootstrap CIs over parent drives; paired differences for signal-group gains; Top-K oracle over
whole-path candidates only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RADII_M = (10.0, 25.0, 50.0, 100.0, 200.0, 500.0)
PATH_RADII_M = (25.0, 50.0, 100.0)          # geometric (time-free) path recovery radii
ROUTE_OK = (50.0, 0.8)                      # route recovered: >= 80% of the reference path length within 50 m of the prediction
SUCCESS = ((50.0, 0.9), (100.0, 0.8))
TIME_BINS_S = (0, 30, 60, 120, 300, 600, 900, 1800, 3600, np.inf)
DIST_BINS_M = (0, 250, 500, 1000, 2000, 5000, 10000, 20000, 50000, np.inf)


def densify(xy, spacing):
    """Resample a polyline at most `spacing` metres apart (original vertices kept); NaN vertices split the line.
    Returns (points, total_length)."""
    xy = np.asarray(xy, dtype=float)
    ok = np.isfinite(xy).all(axis=1)
    pts, total = [], 0.0
    start = 0
    for stop in list(np.flatnonzero(~ok)) + [len(xy)]:
        seg = xy[start:stop]
        start = stop + 1
        if len(seg) == 0:
            continue
        if len(seg) == 1:
            pts.append(seg)
            continue
        d = np.hypot(*np.diff(seg, axis=0).T)
        total += float(d.sum())
        cum = np.concatenate([[0.0], np.cumsum(d)])
        n = max(int(np.ceil(cum[-1] / spacing)), 1)
        q = np.linspace(0.0, cum[-1], n + 1)
        pts.append(np.column_stack([np.interp(q, cum, seg[:, 0]), np.interp(q, cum, seg[:, 1])]))
    if not pts:
        return np.zeros((0, 2)), 0.0
    return np.vstack(pts), total


def path_recovery(pred_xy, ref_xy, radii=PATH_RADII_M, spacing=5.0, max_pts=4000):
    """Geometric, time-free recovery of the reference path. recall_r = share of the reference path LENGTH that lies within
    r metres of the predicted path; precision_r = share of the predicted path length within r of the reference path.
    Both polylines are resampled at <= `spacing` metres (coarser for very long paths) so that nearest-vertex distances
    approximate polyline distances within spacing / 2. Missing predictions (NaN) break the predicted polyline."""
    from scipy.spatial import cKDTree
    out = {}
    ref_xy = np.asarray(ref_xy, dtype=float)
    pred_xy = np.asarray(pred_xy, dtype=float)
    L_ref = float(np.nansum(np.hypot(*np.diff(ref_xy, axis=0).T)))
    L_pred = float(np.nansum(np.hypot(*np.diff(pred_xy[np.isfinite(pred_xy).all(axis=1)], axis=0).T))) if np.isfinite(pred_xy).any() else 0.0
    sp = max(spacing, max(L_ref, L_pred) / max_pts)
    R, _ = densify(ref_xy, sp)
    P, _ = densify(pred_xy, sp)
    if len(R) < 2:
        return {f"path_recall{int(r)}": np.nan for r in radii} | {f"path_prec{int(r)}": np.nan for r in radii}
    if len(P) == 0:
        return {f"path_recall{int(r)}": 0.0 for r in radii} | {f"path_prec{int(r)}": np.nan for r in radii}
    d_ref = cKDTree(P).query(R, k=1)[0]              # reference samples -> nearest predicted sample
    d_pred = cKDTree(R).query(P, k=1)[0]             # predicted samples -> nearest reference sample
    for r in radii:
        out[f"path_recall{int(r)}"] = float(np.mean(d_ref <= r))
        out[f"path_prec{int(r)}"] = float(np.mean(d_pred <= r)) if len(P) > 1 else np.nan
    return out


def coverage_dist(err, ref_xy, radii=RADII_M):
    """Distance-weighted coverage: share of the reference distance travelled during seconds whose error is <= r
    (standstill seconds carry no weight; a missing prediction counts as beyond every radius)."""
    ref_xy = np.asarray(ref_xy, dtype=float)
    ds = np.concatenate([[0.0], np.hypot(*np.diff(ref_xy, axis=0).T)])
    total = float(np.nansum(ds))
    e = np.where(np.isfinite(err), err, np.inf)
    if total <= 0:
        return {f"dist_c{int(r)}": np.nan for r in radii}
    return {f"dist_c{int(r)}": float(np.nansum(ds[e <= r]) / total) for r in radii}


def time_ok(err, t_rel, r):
    """Seconds after t0 until the error first exceeds r (a missing prediction counts as exceeding); the episode
    duration if it never does."""
    e = np.where(np.isfinite(err), err, np.inf)
    beyond = np.flatnonzero(e > r)
    return float(t_rel[beyond[0]]) if len(beyond) else float(t_rel[-1])


def point_errors(pred_xy, ref_xy):
    pred_xy = np.asarray(pred_xy, dtype=float)
    ref_xy = np.asarray(ref_xy, dtype=float)
    return np.hypot(pred_xy[:, 0] - ref_xy[:, 0], pred_xy[:, 1] - ref_xy[:, 1])


def episode_summary(err, t_rel=None, s_can=None, moving=None, in_gap=None, radii=RADII_M, success=SUCCESS, pred_xy=None, ref_xy=None):
    """err: errors at the scored reference points (NaN = no prediction -> counted as a miss for C_r, excluded from ADE
    only if `allow_missing`; here NaN counts as miss AND is reported as n_missing). With pred_xy / ref_xy the
    recovery metrics are added: distance-weighted coverage, time-free path recall / precision, the route-recovered flag,
    the time until the error first exceeds 50 / 100 m and the destination-within flags."""
    err = np.asarray(err, dtype=float)
    n = len(err)
    out = {"n_points": int(n), "n_missing": int(np.isnan(err).sum())}
    if n == 0:
        return out
    fin = np.isfinite(err)
    out["ade"] = float(err[fin].mean()) if fin.any() else np.nan
    out["fde"] = float(err[np.flatnonzero(fin)[-1]]) if fin.any() else np.nan
    out["p50"] = float(np.nanmedian(err)) if fin.any() else np.nan
    out["p90"] = float(np.nanpercentile(err, 90)) if fin.any() else np.nan
    out["max"] = float(np.nanmax(err)) if fin.any() else np.nan
    e_miss = np.where(fin, err, np.inf)
    out["fde_gt_500"] = bool(np.isfinite(out["fde"]) and out["fde"] > 500.0) if fin.any() else True
    tail = e_miss[-max(1, n // 5):]                      # lost_road: the final fifth of the episode stays beyond 100 m
    out["lost_road"] = bool(np.all(tail > 100.0))
    for r in radii:
        out[f"c{int(r)}"] = float(np.mean(e_miss <= r))
    for r, q in success:
        out[f"s{int(r)}_{int(q * 100)}"] = bool(out[f"c{int(r)}"] >= q)
    if moving is not None:
        mv = np.asarray(moving, dtype=bool)
        out["n_moving"] = int(mv.sum())
        out["ade_moving"] = float(np.nanmean(err[mv])) if (mv & fin).any() else np.nan
        for r in radii:
            out[f"c{int(r)}_moving"] = float(np.mean(e_miss[mv] <= r)) if mv.any() else np.nan
    if in_gap is not None:
        gp = np.asarray(in_gap, dtype=bool)
        out["n_in_gap"] = int(gp.sum())
        out["ade_no_gap"] = float(np.nanmean(err[~gp])) if (~gp & fin).any() else np.nan
    out["end_within_100"] = bool(e_miss[-1] <= 100.0)      # destination recovered: error at the LAST reference second (missing = no)
    out["end_within_500"] = bool(e_miss[-1] <= 500.0)
    if ref_xy is not None:
        out.update(coverage_dist(err, ref_xy, (50.0, 100.0)))
        if pred_xy is not None:
            out.update(path_recovery(pred_xy, ref_xy))
            pr = out.get(f"path_recall{int(ROUTE_OK[0])}", np.nan)
            out["route_ok"] = bool(np.isfinite(pr) and pr >= ROUTE_OK[1])
    if t_rel is not None:
        t_rel = np.asarray(t_rel, dtype=float)
        dur = float(t_rel[-1]) if len(t_rel) else np.nan
        for r in (50.0, 100.0):
            out[f"t_ok{int(r)}"] = time_ok(err, t_rel, r)
            out[f"frac_ok{int(r)}"] = out[f"t_ok{int(r)}"] / dur if dur and dur > 0 else np.nan
        out["curve_time"] = binned_median(err, t_rel, TIME_BINS_S)
    if s_can is not None:
        out["curve_dist"] = binned_median(err, np.asarray(s_can, dtype=float), DIST_BINS_M)
    return out


def binned_median(err, x, bins):
    """[(lo, hi, n, median, mean)] of err over bins of x (NaN errors count as missing, not in medians)."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (x >= lo) & (x < hi)
        e = err[k]
        f = e[np.isfinite(e)]
        rows.append((float(lo), float(hi), int(k.sum()), float(np.median(f)) if len(f) else np.nan, float(f.mean()) if len(f) else np.nan))
    return rows


def topk_oracle(err_by_candidate):
    """err_by_candidate: (K, n) errors of K whole-path candidates. Returns the candidate index minimising ADE and
    that candidate's errors (the only admissible Top-K rule: pick ONE complete path)."""
    e = np.asarray(err_by_candidate, dtype=float)
    ade = np.nanmean(np.where(np.isfinite(e), e, np.nan), axis=1)
    ade = np.where(np.isnan(ade), np.inf, ade)
    k = int(np.argmin(ade))
    return k, e[k]


def cluster_bootstrap(values, clusters, stat=np.mean, n_boot=1000, seed=20260709, ci=0.95):
    """Percentile bootstrap CI of `stat` over episodes, resampling parent drives (clusters) with replacement."""
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    fin = np.isfinite(values)
    values, clusters = values[fin], clusters[fin]
    if len(values) == 0:
        return np.nan, (np.nan, np.nan)
    point = float(stat(values))
    uniq, inv = np.unique(clusters, return_inverse=True)
    if len(uniq) < 2:
        return point, (point, point)
    rng = np.random.default_rng(seed)
    if stat is np.mean:     # exact vectorised form: the mean of the resampled episodes = sum of cluster sums / sum of cluster counts
        sums = np.bincount(inv, weights=values, minlength=len(uniq))
        cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
        picks = rng.integers(0, len(uniq), (n_boot, len(uniq)))
        boots = sums[picks].sum(axis=1) / cnts[picks].sum(axis=1)
    else:
        groups = [values[inv == i] for i in range(len(uniq))]
        boots = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.integers(0, len(uniq), len(uniq))
            boots[b] = stat(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, (float(lo), float(hi))


def aggregate(table: pd.DataFrame, metric: str, by=(), cluster="parent", n_boot=1000, seed=20260709):
    """Mean with cluster-bootstrap CI (and median) of an episode-level metric, per group of `by` columns."""
    rows = []
    groups = table.groupby(list(by)) if by else [((), table)]
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        m, (lo, hi) = cluster_bootstrap(g[metric].values, g[cluster].values, np.mean, n_boot, seed)
        med, (mlo, mhi) = cluster_bootstrap(g[metric].values, g[cluster].values, np.median, n_boot, seed)
        rows.append(dict(zip(by, key)) | {"metric": metric, "n_episodes": int(g[metric].notna().sum()),
                                          "n_parents": int(g[cluster].nunique()), "mean": m, "mean_lo": lo, "mean_hi": hi,
                                          "median": med, "median_lo": mlo, "median_hi": mhi})
    return pd.DataFrame(rows)


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, metric: str, key="episode_id", cluster="parent",
                      n_boot=1000, seed=20260709):
    """Mean of (b - a) over episodes present in both tables, with a cluster-bootstrap CI."""
    m = a[[key, cluster, metric]].merge(b[[key, metric]], on=key, suffixes=("_a", "_b"))
    d = m[f"{metric}_b"].values - m[f"{metric}_a"].values
    mean, (lo, hi) = cluster_bootstrap(d, m[cluster].values, np.mean, n_boot, seed)
    return {"metric": metric, "n_pairs": int(np.isfinite(d).sum()), "mean_diff": mean, "lo": lo, "hi": hi}
