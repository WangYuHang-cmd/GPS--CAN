"""Statistics for the ICDE tables (benchmark v1.1): failure-aware medians, driver-clustered and driver-equal bootstrap,
paired differences on common episodes.

Conventions: an episode a method was expected to predict but did not (timeout, no match, error) is a failure: its ADE is
+inf for medians and CDFs (never dropped), its coverage / success / route flags are 0. Means of errors are reported only
on episodes where every compared method has an output, next to the failure rate. The bootstrap resamples drivers (the
independent unit; one driver can own hundreds of drives) with replacement; the driver-equal aggregate first summarises
each driver and then takes the median over drivers, so prolific drivers do not dominate.
"""
from __future__ import annotations

import numpy as np

SEED = 20261028


def _groups(values, clusters):
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    v = np.asarray(values, dtype=float)[order]
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    return [v[bounds[i]:bounds[i + 1]] for i in range(len(uniq))]


def median_inf(values, clusters, n_boot=1000, ci=0.95, seed=SEED):
    """Median with failures as +inf (NaN is treated as a failure too), percentile CI over driver resamples."""
    v = np.asarray(values, dtype=float)
    v = np.where(np.isnan(v), np.inf, v)
    if len(v) == 0:
        return np.nan, (np.nan, np.nan)
    point = float(np.median(v))
    groups = _groups(v, clusters)
    if len(groups) < 2:
        return point, (point, point)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        boots[b] = np.median(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, (float(lo), float(hi))


def mean_ci(values, clusters, n_boot=1000, ci=0.95, seed=SEED):
    """Mean of finite values (rates: failures already encoded as 0), CI over driver resamples."""
    v = np.asarray(values, dtype=float)
    c = np.asarray(clusters)
    ok = np.isfinite(v)
    v, c = v[ok], c[ok]
    if len(v) == 0:
        return np.nan, (np.nan, np.nan)
    uniq, inv = np.unique(c, return_inverse=True)
    point = float(v.mean())
    if len(uniq) < 2:
        return point, (point, point)
    sums = np.bincount(inv, weights=v, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq)).astype(float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(uniq), (n_boot, len(uniq)))
    boots = sums[picks].sum(axis=1) / cnts[picks].sum(axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, (float(lo), float(hi))


def driver_equal(values, drivers, per_driver=np.median, n_boot=1000, ci=0.95, seed=SEED):
    """Summarise each driver (median by default; failures +inf), then the median over drivers with a CI over drivers."""
    v = np.asarray(values, dtype=float)
    v = np.where(np.isnan(v), np.inf, v)
    per = np.array([per_driver(g) for g in _groups(v, drivers)])
    if len(per) == 0:
        return np.nan, (np.nan, np.nan), 0
    point = float(np.median(per))
    if len(per) < 2:
        return point, (point, point), len(per)
    rng = np.random.default_rng(seed)
    boots = np.median(per[rng.integers(0, len(per), (n_boot, len(per)))], axis=1)
    lo, hi = np.percentile(boots, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return point, (float(lo), float(hi)), len(per)


def paired(a, b, clusters, n_boot=1000, ci=0.95, seed=SEED):
    """a - b on episodes where both are finite: median and mean difference with driver-clustered CIs, win rate of a."""
    a, b, c = np.asarray(a, dtype=float), np.asarray(b, dtype=float), np.asarray(clusters)
    ok = np.isfinite(a) & np.isfinite(b)
    d, c = (a - b)[ok], c[ok]
    if len(d) == 0:
        return dict(n=0)
    med, (mlo, mhi) = median_inf(d, c, n_boot, ci, seed)
    mean, (alo, ahi) = mean_ci(d, c, n_boot, ci, seed)
    return dict(n=int(len(d)), n_drivers=int(len(np.unique(c))), median=med, median_lo=mlo, median_hi=mhi,
                mean=mean, mean_lo=alo, mean_hi=ahi, win_a=float(np.mean(d < 0)))


def _pct_bounds(boots, ci):
    """Percentile bounds without interpolation (nearest-rank), so +inf resamples never produce NaN."""
    b = np.sort(np.asarray(boots, dtype=float))
    lo = b[int(np.floor((1 - ci) / 2 * (len(b) - 1)))]
    hi = b[int(np.ceil((1 + ci) / 2 * (len(b) - 1)))]
    return float(lo), float(hi)


def _cluster_groups(clusters):
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    return [np.flatnonzero(inv == i) for i in range(len(uniq))]


def seed_median(values, clusters, n_boot=1000, ci=0.95, seed=SEED):
    """Agent study: values is an (S, N) matrix (seeds x episodes; failures +inf, NaN treated as +inf). Statistic = mean
    over seeds of the per-seed failure-aware median, recomputed inside every driver resample (the seeds are never
    averaged after the bootstrap). S = 1 reduces to median_inf up to the percentile method."""
    v = np.atleast_2d(np.asarray(values, dtype=float))
    v = np.where(np.isnan(v), np.inf, v)
    if v.shape[1] == 0:
        return np.nan, (np.nan, np.nan)
    stat = lambda m: float(np.mean(np.median(m, axis=1)))  # noqa: E731
    point = stat(v)
    groups = _cluster_groups(clusters)
    if len(groups) < 2:
        return point, (point, point)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots[b] = stat(v[:, idx])
    return point, _pct_bounds(boots, ci)


def seed_paired(diff, clusters, n_boot=1000, ci=0.95, seed=SEED):
    """Agent study: diff is an (S, N) matrix of per-seed paired differences (NaN where either side has no output).
    Statistic = mean over seeds of the per-seed median of the finite differences, recomputed inside every driver
    resample. Returns dict(median, lo, hi, n_per_seed, n_drivers)."""
    d = np.atleast_2d(np.asarray(diff, dtype=float))
    c = np.asarray(clusters)

    def stat(m):
        meds = [np.median(row[np.isfinite(row)]) for row in m if np.isfinite(row).any()]
        return float(np.mean(meds)) if meds else np.nan

    point = stat(d)
    groups = _cluster_groups(c)
    n_per_seed = [int(np.isfinite(row).sum()) for row in d]
    if len(groups) < 2 or not np.isfinite(point):
        return dict(median=point, lo=point, hi=point, n_per_seed=n_per_seed, n_drivers=len(groups))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots[b] = stat(d[:, idx])
    boots = boots[np.isfinite(boots)] if np.isfinite(boots).any() else boots
    lo, hi = _pct_bounds(boots, ci)
    return dict(median=point, lo=lo, hi=hi, n_per_seed=n_per_seed, n_drivers=len(groups))
