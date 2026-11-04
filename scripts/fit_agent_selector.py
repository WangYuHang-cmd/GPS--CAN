#!/usr/bin/env python3
"""Supervised selector (agent study, configuration B): fit on the validation cohort and freeze.

Labels are the evaluator's route_ok of every pool candidate (prediction stems poolcand_<cid>__G*, scored by score.py),
so this script is evaluator-side and reads the reference indirectly; the frozen selector itself only sees observable
features. Cross-fitting by driver (GroupKFold) reports what the selector would achieve on unseen drivers next to the
beam's Top-1 and the pool oracle; the final model is fitted on every validation window.

    python scripts/score.py --run icde_agent_val --manifest manifests/episode_manifest_v1.parquet --eligible-only \
        --folds val --episodes-file manifests/icde_agent_val.csv --workers 2
    python scripts/fit_agent_selector.py --run icde_agent_val --manifest manifests/icde_agent_val.csv
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths, stats  # noqa: E402
from cantraj.agent import selector as S  # noqa: E402
from cantraj.agent.toolbox import EpisodeToolBox  # noqa: E402

TABLES = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")


def load_dataset(run, cohort):
    ep = pd.read_parquet(os.path.join(paths.SCORES, run, "episodes_all.parquet"), columns=["episode_id", "method", "protocol", "route_ok", "ade"])
    ep = ep[ep.method.str.startswith("poolcand_")]
    ep["cid"] = ep.method.str.replace("poolcand_", "", regex=False).str.split("__").str[0]
    lab = ep.set_index(["episode_id", "cid"])
    root = os.path.join(paths.BENCH, "agent_toolcache", run)
    rows = []
    for r in cohort.itertuples():
        cdir = os.path.join(root, r.driver, r.drive, f"k{int(r.k):03d}")
        if not os.path.exists(os.path.join(cdir, "diagnostics.json")):
            continue
        tb = EpisodeToolBox(cdir).call_all()
        for cid in tb.all_candidates():
            key = (r.episode_id, cid)
            if key not in lab.index:
                continue
            f = tb.features(cid)
            rows.append(dict(episode_id=r.episode_id, driver=r.driver, cid=cid, tool=f["tool"], rank=f["rank"], route_ok=float(lab.loc[key, "route_ok"]),
                             ade=float(lab.loc[key, "ade"]), **{k: v for k, v in f.items() if k not in ("tool", "rank")}))
    return pd.DataFrame(rows)


def window_outcomes(df, chosen):
    """chosen: {episode_id: cid or None}; returns per-window route_ok (0 for no choice) aligned with df's windows."""
    lab = df.set_index(["episode_id", "cid"]).route_ok
    out = {}
    for eid in df.episode_id.unique():
        c = chosen.get(eid)
        out[eid] = float(lab.get((eid, c), 0.0)) if c else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="icde_agent_val")
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "icde_agent_val.csv"))
    ap.add_argument("--out", default=os.path.join(paths.GPS_DIR, "configs", "agent", "selector_v1.json"))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    from sklearn.model_selection import GroupKFold
    cohort = pd.read_csv(args.manifest)
    df = load_dataset(args.run, cohort)
    windows = df.drop_duplicates("episode_id")[["episode_id", "driver"]].reset_index(drop=True)
    n_cohort = len(cohort)
    print(f"{len(windows)} windows with candidates (cohort {n_cohort}), {len(df)} candidates, positives {df.route_ok.mean():.3f}", flush=True)
    X = np.stack([S.feature_vector(f) for f in df.to_dict("records")])
    y = df.route_ok.to_numpy(dtype=int)
    # cross-fitting by driver: scaler + regression fitted inside every training fold
    chosen_cv = {}
    gkf = GroupKFold(n_splits=min(args.k, windows.driver.nunique()))
    for tr_idx, te_idx in gkf.split(windows, groups=windows.driver):
        tr_eids, te_eids = set(windows.episode_id.iloc[tr_idx]), set(windows.episode_id.iloc[te_idx])
        m_tr = df.episode_id.isin(tr_eids).to_numpy()
        pipe = S.fit(X[m_tr], y[m_tr], C=args.C)
        for eid in te_eids:
            sub = df[df.episode_id == eid]
            feats = {r["cid"]: r for r in sub.to_dict("records")}
            chosen_cv[eid] = S.select_with_pipeline(pipe, feats)
    refs = {"selector_cv": chosen_cv,
            "beam1": {eid: ("beam-1" if ((df.episode_id == eid) & (df.cid == "beam-1")).any() else None) for eid in windows.episode_id},
            "pf1": {eid: ("pf-1" if ((df.episode_id == eid) & (df.cid == "pf-1")).any() else None) for eid in windows.episode_id},
            "ep1": {eid: ("ep-1" if ((df.episode_id == eid) & (df.cid == "ep-1")).any() else None) for eid in windows.episode_id}}
    rows = []
    drivers = windows.set_index("episode_id").driver
    all_eids = list(cohort.episode_id)                       # windows without any candidate count as failures (route 0)
    drv_all = cohort.set_index("episode_id").driver
    for name, ch in refs.items():
        oc = window_outcomes(df, ch)
        v = np.array([oc.get(e, 0.0) for e in all_eids])
        p, (lo, hi) = stats.mean_ci(v, drv_all.loc[all_eids].to_numpy(), n_boot=args.n_boot)
        rows.append(dict(row=name, n_windows=len(all_eids), n_drivers=int(drv_all.nunique()), route_pct=100 * p, route_lo=100 * lo, route_hi=100 * hi))
    orc = df.groupby("episode_id").route_ok.max()
    v = np.array([float(orc.get(e, 0.0)) for e in all_eids])
    p, (lo, hi) = stats.mean_ci(v, drv_all.loc[all_eids].to_numpy(), n_boot=args.n_boot)
    rows.append(dict(row="oracle", n_windows=len(all_eids), n_drivers=int(drv_all.nunique()), route_pct=100 * p, route_lo=100 * lo, route_hi=100 * hi))
    tab = pd.DataFrame(rows)
    os.makedirs(TABLES, exist_ok=True)
    tab.to_csv(os.path.join(TABLES, "agent_selector_val.csv"), index=False)
    print(tab.round(1).to_string(index=False), flush=True)
    # final fit on every validation window -> frozen selector
    pipe = S.fit(X, y, C=args.C)
    params = S.export(pipe, C=args.C)
    params.update(n_windows=int(len(windows)), n_candidates=int(len(df)), positive_rate=float(y.mean()), fitted_on=f"{args.run} / {os.path.basename(args.manifest)}",
                  cohort_sha=hashlib.sha256(open(args.manifest, "rb").read()).hexdigest(), fitted_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                  cv_route_pct=float(tab.loc[tab.row == "selector_cv", "route_pct"].iloc[0]), beam1_route_pct=float(tab.loc[tab.row == "beam1", "route_pct"].iloc[0]),
                  oracle_route_pct=float(tab.loc[tab.row == "oracle", "route_pct"].iloc[0]),
                  coef_by_feature={f: round(c, 4) for f, c in zip(S.FEATURES, params["coef"])})
    with open(args.out, "w") as f:
        json.dump(params, f, indent=1)
    print(f"selector -> {args.out}", flush=True)
    print({k: round(v, 3) for k, v in params["coef_by_feature"].items()})


if __name__ == "__main__":
    main()
