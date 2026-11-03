#!/usr/bin/env python3
"""Plan v0.9 analyses A and B, from the per-episode scores of run icde_v11 and the beam's candidate files; no new run.

A  Method complementarity and common failure modes (five-minute windows, test and few-shot folds):
   complementarity_w05.csv          per method set and fold: two oracles kept apart, ADE*_i = min_m ADE_im (lowest ADE per
                                    window; +inf = every member failed) and R*_i = max_m R_im (at least one member recovers the
                                    route); the fixed reference member's median and route share; the route difference
                                    R* - R_ref with a driver-clustered CI (failures count as not recovered); the paired ADE
                                    difference oracle - reference on windows where both are finite (n given); the share of
                                    windows no member recovers; the share some member recovers but the reference does not.
                                    The reference is fixed per set in advance (the benchmark's own beam where it is a member);
                                    the best single member is reported as a description only.
   complementarity_members_w05.csv  per set, fold and member: median ADE, route share, failures, share of windows where
                                    the member attains the oracle ADE (ties to the first listed; all-fail windows excluded).
   common_failure_strata_w05.csv    windows no member of the full map set recovers, by reference turn count (from
                                    scripts/icde_area_check.py) and by road-density tertile (the failure analysis's cuts).
   driver_route_w05.csv             per method and fold: distribution over drivers of the share of their expected windows
                                    with the route recovered; drivers with fewer than 5 windows are counted apart.
B  Candidate confidence score and selective prediction (beam Top-1 probability = normalised score among the surviving
   candidates, not a calibrated probability):
   confidence_w05_w15.csv           AUROC of the score for route recovery (driver bootstrap), Spearman with ADE, and for
                                    thresholds 0.3 to 1.0: coverage over ALL expected windows (failures in the denominator),
                                    route rate among accepted windows, high-score errors (score >= 0.9, route not recovered).
   window_taxonomy_w05.csv          no output / output but no candidate within T / a candidate within T but Top-1 not /
                                    Top-1 within T, T = 50 and 100 m whole-path ADE (single-path methods: Top-1 only).
Every share is failure-aware (an expected window without output is a failure). Memory: only the needed columns of
episodes_all are read; candidate files are read row group by row group.

    python scripts/icde_extra.py [--run icde_v11] [--n-boot 500] [--what comp strata drivers conf tax]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths, stats  # noqa: E402
import icde_tables as T  # noqa: E402

OUT = T.OUT
SLIM = ["episode_id", "method", "protocol", "win", "ade", "route_ok", "c100", "topk_K", "best_rank", "top1_ade", "top3_ade", "top10_ade"]
ROW = {k: (stem, proto, req, hp) for k, stem, proto, _, _, req, hp in T.MAIN if isinstance(stem, str)}

# (set name, members by T.MAIN key, fixed reference member or None)
SETS = [("A steering map, heading known", ["pf", "beam", "lineage", "macro", "opd2", "spy_hp"], "beam"),
        ("B speed-only map, heading known", ["pf_none", "beam_none", "ep_hp", "dewri_hp"], "beam_none"),
        ("B' speed-only map, heading unknown", ["pf_none_hm", "beam_none_hm", "ep", "dewri"], "beam_none_hm"),
        ("C published, heading known", ["macro", "opd2", "ep_hp", "dewri_hp", "spy_hp"], None),
        ("D all map, heading known", ["pf", "beam", "lineage", "macro", "opd2", "spy_hp", "pf_none", "beam_none", "ep_hp", "dewri_hp"], "beam"),
        ("P pedal subset, heading unknown", ["ct", "ep", "dewri", "beam_none_hm", "pf_none_hm", "spy", "beam_hm", "pf_hm"], "beam_hm")]
DRIVER_KEYS = ["pf", "beam", "opd2", "macro", "ep_hp", "dewri_hp", "spy_hp", "pf_none", "beam_none", "ep", "dewri"]
CONF_ROWS = [("map_beam_steer__G3", "F-M-Hp", "G3"), ("map_beam_steer__G3", "F-M-Hm", "G3"),
             ("map_beam_none__G1", "F-M-Hp", "G1"), ("map_beam_none__G1", "F-M-Hm", "G1")]
PROB_ONLY = [("dewri_hp__G1", "F-M-Hp"), ("dewri__G1", "F-M-Hm"), ("elastic_pathing_hp__G1", "F-M-Hp"), ("elastic_pathing__G1", "F-M-Hm")]
TAX_ROWS = [("map_beam_steer__G3", "F-M-Hp", "G3", False), ("map_beam_steer__G3", "F-M-Hm", "G3", False),
            ("map_beam_none__G1", "F-M-Hp", "G1", False), ("map_beam_none__G1", "F-M-Hm", "G1", False),
            ("dewri_hp__G1", "F-M-Hp", "G1", True), ("dewri__G1", "F-M-Hm", "G1", False),
            ("elastic_pathing_hp__G1", "F-M-Hp", "G1", True), ("elastic_pathing__G1", "F-M-Hm", "G1", False)]
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
TURN_BINS = [(-1, 1, "0-1"), (1, 3, "2-3"), (3, 6, "4-6"), (6, 10 ** 6, "7+")]


def load_man():
    """The expected-set manifest exactly as icde_tables.load builds it, without reading every score column."""
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "driver", "drive", "fold", "win", "eligible_main"])
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"), columns=["episode_id", "map_ok"])
    aw = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"))
    man = man[man.eligible_main & man.fold.isin(list(T.FOLDS))].merge(me, on="episode_id", how="left").merge(aw, on=["driver", "drive"], how="left")
    man["map_ok"] = man.map_ok.fillna(False).astype(bool)
    man["pedal_ok"] = man.gas == "valid"
    return man


def load_scores(run):
    return pd.read_parquet(os.path.join(paths.SCORES, run, "episodes_all.parquet"), columns=SLIM)


def values(ep, stem, proto, ids, col="ade"):
    g = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id")
    if col == "ade":
        v = g.ade.reindex(ids).to_numpy(dtype=float)
        return np.where(np.isnan(v), np.inf, v)
    return g[col].reindex(ids).astype(float).fillna(0.0).to_numpy()


def expected_ids(man, win, fold, keys):
    sets = [set(T.expected(man, win, fold, ROW[k][1], ROW[k][2], ROW[k][3]).episode_id) for k in keys]
    ids = set.intersection(*sets) if sets else set()
    d = man[man.episode_id.isin(ids) & (man.win == win) & (man.fold == fold)].sort_values("episode_id")
    return d.episode_id.to_numpy(), d.driver.to_numpy()


def complementarity_table(ep, man, n_boot):
    rows, members = [], []
    for name, keys, ref in SETS:
        for fold in T.FOLDS:
            ids, drv = expected_ids(man, "w05", fold, keys)
            if not len(ids):
                continue
            A = np.vstack([values(ep, ROW[k][0], ROW[k][1], ids) for k in keys])             # members x windows, +inf = failure
            R = np.vstack([values(ep, ROW[k][0], ROW[k][1], ids, "route_ok") for k in keys])
            ade_star, r_star = A.min(axis=0), R.max(axis=0)
            omed, (olo, ohi) = stats.median_inf(ade_star, drv, n_boot)
            orate, (orlo, orhi) = stats.mean_ci(r_star, drv, n_boot)
            afail, (aflo, afhi) = stats.mean_ci(1.0 - r_star, drv, n_boot)
            meds = [stats.median_inf(A[i], drv, 50)[0] for i in range(len(keys))]
            best_i = int(np.argmin(meds))
            row = dict(set=name, fold=fold, n=len(ids), n_drivers=int(len(np.unique(drv))), members=" ".join(keys),
                       oracle_ade_median=omed, oracle_ade_lo=olo, oracle_ade_hi=ohi, oracle_route=orate, oracle_route_lo=orlo, oracle_route_hi=orhi,
                       all_fail_share=afail, all_fail_lo=aflo, all_fail_hi=afhi, n_all_fail=int((r_star == 0).sum()),
                       n_no_output_any=int(np.isinf(ade_star).sum()), best_single=keys[best_i], best_single_median=meds[best_i], reference=ref or "")
            if ref:
                i = keys.index(ref)
                a_ref, r_ref = A[i], R[i]
                rmed, (rlo, rhi) = stats.median_inf(a_ref, drv, n_boot)
                rrate, _ = stats.mean_ci(r_ref, drv, n_boot)
                gain, (glo, ghi) = stats.mean_ci(r_star - r_ref, drv, n_boot)
                comp, (clo, chi) = stats.mean_ci(((r_star == 1) & (r_ref == 0)).astype(float), drv, n_boot)
                pr = stats.paired(ade_star, a_ref, drv, n_boot)
                row.update(ref_ade_median=rmed, ref_ade_lo=rlo, ref_ade_hi=rhi, ref_route=rrate, route_gain=gain, route_gain_lo=glo, route_gain_hi=ghi,
                           complementary_share=comp, complementary_lo=clo, complementary_hi=chi,
                           ade_gain_median=pr.get("median"), ade_gain_lo=pr.get("median_lo"), ade_gain_hi=pr.get("median_hi"),
                           ade_gain_mean=pr.get("mean"), ade_gain_mean_lo=pr.get("mean_lo"), ade_gain_mean_hi=pr.get("mean_hi"), n_both_finite=pr.get("n", 0))
            rows.append(row)
            fin = np.isfinite(ade_star)
            am = np.argmin(A, axis=0)
            for i, k in enumerate(keys):
                members.append(dict(set=name, fold=fold, member=k, n=len(ids), median_ade=meds[i], route=float(R[i].mean()),
                                    n_fail=int(np.isinf(A[i]).sum()), share_attains_oracle=float(np.mean(am[fin] == i)) if fin.any() else np.nan))
    t, mt = pd.DataFrame(rows), pd.DataFrame(members)
    t.to_csv(os.path.join(OUT, "complementarity_w05.csv"), index=False)
    mt.to_csv(os.path.join(OUT, "complementarity_members_w05.csv"), index=False)
    return t


def common_failure_strata(ep, man, n_boot):
    """Set D (all map methods, heading known) on the pooled main folds: share of windows no member recovers, by turn
    count and by density tertile (cuts as scripts/failure_analysis.py: main-fold w05 windows with a map PF output)."""
    keys, ref = SETS[4][1], SETS[4][2]
    parts = [expected_ids(man, "w05", f, keys) for f in T.FOLDS]
    ids = np.concatenate([p[0] for p in parts]); drv = np.concatenate([p[1] for p in parts])
    R = np.vstack([values(ep, ROW[k][0], ROW[k][1], ids, "route_ok") for k in keys])
    r_star, r_ref = R.max(axis=0), R[keys.index(ref)]
    d = pd.DataFrame(dict(episode_id=ids, driver=drv, all_fail=(r_star == 0).astype(float), oracle_route=r_star, ref_route=r_ref))
    ac = os.path.join(OUT, "area_check_episodes.csv")
    if os.path.exists(ac):
        d = d.merge(pd.read_csv(ac)[["episode_id", "n_turns"]], on="episode_id", how="left")
    dens = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_density_v11.parquet"))
    d = d.merge(dens[["episode_id", "nodes_per_km2"]], on="episode_id", how="left")
    pf_out = set(ep[(ep.method == "map_pf_steer__G3") & (ep.protocol == "O-M-Hp") & (ep.win == "w05")].episode_id)
    base = d[d.episode_id.isin(pf_out)].nodes_per_km2.dropna()
    q1, q2 = float(base.quantile(1 / 3)), float(base.quantile(2 / 3))
    rows = []

    def add(stratum, level, g):
        if not len(g):
            return
        af, (lo, hi) = stats.mean_ci(g.all_fail.to_numpy(), g.driver.to_numpy(), n_boot)
        rows.append(dict(stratum=stratum, level=level, n=len(g), n_drivers=int(g.driver.nunique()), all_fail_share=af, all_fail_lo=lo, all_fail_hi=hi,
                         oracle_route=float(g.oracle_route.mean()), ref_route=float(g.ref_route.mean())))
    if "n_turns" in d:
        for lo_, hi_, lab in TURN_BINS:
            add("reference turns", lab, d[(d.n_turns > lo_) & (d.n_turns <= hi_)])
    for lab, g in (("low", d[d.nodes_per_km2 <= q1]), ("mid", d[(d.nodes_per_km2 > q1) & (d.nodes_per_km2 <= q2)]), ("high", d[d.nodes_per_km2 > q2])):
        add("density", f"{lab} ({q1:.0f}/{q2:.0f} per km2 cuts)", g)
    add("all", "all", d)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "common_failure_strata_w05.csv"), index=False)
    return t


def driver_route_table(ep, man):
    rows = []
    for k in DRIVER_KEYS:
        stem, proto, req, hp = ROW[k]
        for fold in T.FOLDS:
            exp = T.expected(man, "w05", fold, proto, req, hp)
            if not len(exp):
                continue
            ids = exp.episode_id.to_numpy()
            r = values(ep, stem, proto, ids, "route_ok")
            per = pd.DataFrame(dict(driver=exp.driver.to_numpy(), r=r)).groupby("driver").r.agg(["mean", "size"])
            big = per[per["size"] >= 5]
            q = big["mean"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]) if len(big) else pd.Series(dtype=float)
            rows.append(dict(method=k, fold=fold, n_windows=len(ids), n_drivers=len(per), n_drivers_ge5=len(big), n_drivers_lt5=int((per["size"] < 5).sum()),
                             pooled_route=float(r.mean()),
                             q10=q.get(0.1, np.nan), q25=q.get(0.25, np.nan), q50=q.get(0.5, np.nan), q75=q.get(0.75, np.nan), q90=q.get(0.9, np.nan),
                             share_drivers_below_half=float((big["mean"] < 0.5).mean()) if len(big) else np.nan,
                             share_drivers_zero=float((big["mean"] == 0).mean()) if len(big) else np.nan))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "driver_route_w05.csv"), index=False)
    return t


TOP1_TABLE = os.path.join(paths.SCORES, "icde_v11", "top1_scores.csv")   # thin published copy (scripts/export_top1_scores.py)


def top1_scores(stem, proto):
    """Top-1 candidate score per episode from the prediction file (constant over t), read row group by row group;
    without the prediction files (the public package) the published thin table scores/icde_v11/top1_scores.csv."""
    f = os.path.join(paths.PREDS, "icde_v11", proto, f"{stem}.part0.parquet")
    if not os.path.exists(f):
        if os.path.exists(TOP1_TABLE):
            t = pd.read_csv(TOP1_TABLE)
            t = t[(t.stem == stem) & (t.protocol == proto)]
            return t.set_index("episode_id").prob.astype(float)
        return pd.Series(dtype=float)
    pf = pq.ParquetFile(f)
    parts = []
    for i in range(pf.num_row_groups):
        g = pf.read_row_group(i, columns=["episode_id", "candidate_id", "prob"]).to_pandas()
        g = g[g.candidate_id == 0].drop_duplicates("episode_id")
        parts.append(g[["episode_id", "prob"]])
    s = pd.concat(parts).drop_duplicates("episode_id").set_index("episode_id").prob.astype(float)
    return s


def auroc(score, label):
    pos, neg = label == 1, label == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return np.nan
    r = rankdata(score)
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def auroc_ci(score, label, drivers, n_boot, seed=stats.SEED):
    point = auroc(score, label)
    uniq, inv = np.unique(drivers, return_inverse=True)
    if len(uniq) < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    boots = []
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        boots.append(auroc(score[idx], label[idx]))
    boots = np.array([b for b in boots if np.isfinite(b)])
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def confidence_table(ep, man, n_boot):
    rows = []
    for stem, proto, req in CONF_ROWS:
        sc = top1_scores(stem, proto)
        if not len(sc):
            continue
        for win in ("w05", "w15"):
            for fold in T.FOLDS:
                exp = T.expected(man, win, fold, proto, req, False)
                if not len(exp):
                    continue
                ids, drv = exp.episode_id.to_numpy(), exp.driver.to_numpy()
                a = values(ep, stem, proto, ids)
                r = values(ep, stem, proto, ids, "route_ok")
                s = sc.reindex(ids).to_numpy(dtype=float)
                out = np.isfinite(a) & np.isfinite(s)
                if out.sum() < 10:
                    continue
                au, lo, hi = auroc_ci(s[out], r[out], drv[out], n_boot)
                rho = spearmanr(s[out], a[out]).correlation
                base = dict(stem=stem, protocol=proto, win=win, fold=fold, n_expected=len(ids), n_output=int(out.sum()), n_fail=int((~out).sum()))
                rows.append(dict(base, kind="summary", auroc=au, auroc_lo=lo, auroc_hi=hi, spearman_ade=float(rho),
                                 share_prob_eq_1=float(np.mean(s[out] >= 0.999)), prob_q25=float(np.quantile(s[out], 0.25)), prob_median=float(np.median(s[out]))))
                for tau in THRESHOLDS:
                    acc = out & (s >= tau - 1e-9)
                    n_acc = int(acc.sum())
                    rr, (rlo, rhi) = stats.mean_ci(r[acc], drv[acc], n_boot) if n_acc else (np.nan, (np.nan, np.nan))
                    rows.append(dict(base, kind="threshold", tau=tau, n_accepted=n_acc, coverage=n_acc / len(ids),
                                     route_rate_accepted=rr, route_rate_lo=rlo, route_rate_hi=rhi,
                                     error_share_accepted=float(np.mean(r[acc] == 0)) if n_acc else np.nan,
                                     median_ade_accepted=float(np.median(a[acc])) if n_acc else np.nan))
    for stem, proto in PROB_ONLY:                      # single-path or uniform candidate scores: report how degenerate
        sc = top1_scores(stem, proto)
        if len(sc):
            rows.append(dict(stem=stem, protocol=proto, win="all", fold="all", kind="prob_only", n_output=len(sc), share_prob_eq_1=float(np.mean(sc >= 0.999)),
                             prob_median=float(sc.median())))
    t = pd.DataFrame(rows)
    if len(t):                                        # no candidate scores at all: keep the published table, do not write an empty one
        t.to_csv(os.path.join(OUT, "confidence_w05_w15.csv"), index=False)
    return t


def window_taxonomy(ep, man):
    rows = []
    for stem, proto, req, hp in TAX_ROWS:
        for fold in T.FOLDS:
            exp = T.expected(man, "w05", fold, proto, req, hp)
            if not len(exp):
                continue
            ids = exp.episode_id.to_numpy()
            g = ep[(ep.method == stem) & (ep.protocol == proto)].drop_duplicates("episode_id").set_index("episode_id").reindex(ids)
            a = np.where(g.ade.isna(), np.inf, g.ade.astype(float))
            best = g.top10_ade.astype(float).to_numpy()
            best = np.where(np.isnan(best), a, best)            # single path: its own only candidate
            out = np.isfinite(a)
            for thr in (50.0, 100.0):
                no_out = ~out
                no_cand = out & (best > thr)
                rank_miss = out & (best <= thr) & (a > thr)
                top1_ok = out & (a <= thr)
                rows.append(dict(stem=stem, protocol=proto, fold=fold, threshold_m=thr, n=len(ids),
                                 no_output=float(no_out.mean()), no_candidate=float(no_cand.mean()), ranking_miss=float(rank_miss.mean()), top1_ok=float(top1_ok.mean()),
                                 n_no_output=int(no_out.sum()), n_no_candidate=int(no_cand.sum()), n_ranking_miss=int(rank_miss.sum()), n_top1_ok=int(top1_ok.sum()),
                                 n_candidates_median=float(np.nanmedian(g.topk_K.astype(float))) if g.topk_K.notna().any() else 1.0))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "window_taxonomy_w05.csv"), index=False)
    return t


# Plan v0.9 item D: the nine methods of the five-minute comparison on the Controlled supplement (run icde_supp).
SUPP_ROWS = [("dr_steer__G3", "O-N-Hp", "G3", "steering DR, no map", "Dr"),
             ("map_pf_steer__G3", "O-M-Hp", "G3", "map PF, steering", "Pf"),
             ("macrotracking__G3", "O-M-Hp", "G3", "MacroTracking", "Macro"),
             ("map_beam_steer__G3", "F-M-Hp", "G3", "map beam, steering", "Beam"),
             ("opd2__G3", "F-M-Hp", "G3", "OPD-II", "OpdTwo"),
             ("map_pf_none__G1", "O-M-Hp", "G1", "map PF, speed", "PfNone"),
             ("map_beam_none__G1", "F-M-Hp", "G1", "map beam, speed", "BeamNone"),
             ("elastic_pathing_hp__G1", "F-M-Hp", "G1", "Elastic Pathing, heading", "EpHp"),
             ("dewri_hp__G1", "F-M-Hp", "G1", "Dewri et al., heading", "DewriHp")]


def supplement_table(n_boot, run="icde_supp"):
    """External check after freezing (plan v0.9 D): the frozen methods, parameters unchanged, on the Controlled
    supplement's map-eligible five-minute windows (scripts/icde_supplement.py, scripts/run_icde_supp.sh). Expected set
    = the listed map_ok windows whose drive has the method's signal group (fixed before the outputs); a window without
    output counts as a failure (+inf); medians and route shares carry driver-clustered CIs. Descriptive only: other
    drivers, one vehicle and other recording conditions, so no paired test against the main folds is made here."""
    f = os.path.join(paths.SCORES, run, "episodes_all.parquet")
    if not os.path.exists(f):
        return pd.DataFrame()
    ep = pd.read_parquet(f, columns=["episode_id", "method", "protocol", "ade", "route_ok"])
    eps = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_supplement_episodes.csv"))
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "driver", "drive", "fold", "win"])
    aw = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"))
    d = eps[eps.map_ok].merge(man, on="episode_id").merge(aw, on=["driver", "drive"], how="left").sort_values("episode_id")
    assert (d.fold == "supplement").all() and (d.win == "w05").all()
    rows = []
    for stem, proto, req, label, tag in SUPP_ROWS:
        sub = d[d[req].fillna(False).astype(bool)]
        ids, drv = sub.episode_id.to_numpy(), sub.driver.to_numpy()
        a = values(ep, stem, proto, ids)
        rt = values(ep, stem, proto, ids, "route_ok")
        med, (lo, hi) = stats.median_inf(a, drv, n_boot)
        r, (rlo, rhi) = stats.mean_ci(rt, drv, n_boot)
        fin = a[np.isfinite(a)]
        rows.append(dict(method=label, tag=tag, stem=stem, protocol=proto, n=len(ids), n_drives=sub.drop_duplicates(["driver", "drive"]).shape[0],
                         n_drivers=int(len(set(drv))), fail=int(np.isinf(a).sum()), median=med, median_lo=lo, median_hi=hi,
                         mean_finite=float(fin.mean()) if len(fin) else np.nan, p90=float(np.quantile(a, 0.9)) if len(a) else np.nan,
                         route=r, route_lo=rlo, route_hi=rhi))
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(OUT, "supplement_summary.csv"), index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="icde_v11")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--what", nargs="+", default=["comp", "strata", "drivers", "conf", "tax"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    pd.set_option("display.width", 250)
    if "supp" in args.what:
        print(supplement_table(args.n_boot).round(3).to_string())
    if not (set(args.what) - {"supp"}):
        return
    man = load_man()
    ep = load_scores(args.run)
    if "comp" in args.what:
        print(complementarity_table(ep, man, args.n_boot).round(3).to_string())
    if "strata" in args.what:
        print(common_failure_strata(ep, man, args.n_boot).round(3).to_string())
    if "drivers" in args.what:
        print(driver_route_table(ep, man).round(3).to_string())
    if "conf" in args.what:
        t = confidence_table(ep, man, args.n_boot)
        if len(t):
            print(t[t.kind != "threshold"].round(3).to_string())
        else:
            print("confidence table: no candidate scores (neither prediction files nor scores/icde_v11/top1_scores.csv); kept as published")
    if "tax" in args.what:
        print(window_taxonomy(ep, man).round(3).to_string())


if __name__ == "__main__":
    main()
