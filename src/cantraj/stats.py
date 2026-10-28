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
