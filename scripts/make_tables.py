#!/usr/bin/env python3
"""make_tables.py: first full results tables from a scored run.

    python scripts/make_tables.py --run full_v1 [--manifest manifests/episode_manifest_v1.parquet] [--pilot-ref]
Reads scores/<run>/episodes_all.csv (per-episode rows written by score.py) joined with the episode manifest (fold,
gps_source, log_native, admission) and writes scores/<run>/tables/*.csv plus reports/<run>_results.md:
  headline:  protocol x method x window x fold: n episodes / drives / drivers, ADE median / mean / P90 with 95% cluster
             bootstrap CIs (by driver primary, by drive secondary), FDE median, C_10/25/50/100, S(50,0.9), S(100,0.8),
             missing outputs, fde_gt_500 and lost_road rates
  gains:     paired per-episode differences on common episodes: G3 - G1 (dr_steer vs dr_none), map - no map
             (map_pf_steer vs dr_steer, map_pf_none vs dr_none), calibrated - nominal (dr_steer_calib vs dr_steer)
  strata:    by receiver / rate, input layer, metro density tertile (if available); never by vehicle label
  curves:    median error by elapsed-time bin per method (from the per-episode curve fields when present)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import metrics as M, paths  # noqa: E402

GAIN_PAIRS = [("G3_minus_G1", "dr_steer__G3", "dr_none__G1"), ("map_minus_nomap_steer", "map_pf_steer__G3", "dr_steer__G3"),
              ("map_minus_nomap_speed", "map_pf_none__G1", "dr_none__G1"), ("calib_minus_nominal", "dr_steer_calib__G3", "dr_steer__G3"),
              ("map_calib_minus_map_nominal", "map_pf_steer_calib__G3", "map_pf_steer__G3")]
PILOT_REF = {("O-M-Hp", "map_pf_steer__G3", "w05"): 38.3, ("O-M-Hm", "map_pf_steer__G3", "w05"): 172.8, ("O-M-Hp", "map_pf_steer__G3", "w15"): 352.7,
             ("O-N-Hp", "dr_steer__G3", "w05"): 996.1, ("O-N-Hp", "dr_steer_calib__G3", "w05"): 652.9, ("O-N-Hp", "dr_curvature__G5", "w05"): 66.6}


def ci(values, clusters, stat, n_boot):
    m, (lo, hi) = M.cluster_bootstrap(values, clusters, stat, n_boot)
    return m, lo, hi


def headline(df, n_boot):
    rows = []
    for (proto, method, win, fold), g in df.groupby(["protocol", "method", "win", "fold"]):
        row = dict(protocol=proto, method=method, win=win, fold=fold, n_episodes=len(g), n_drives=g.parent.nunique(), n_drivers=g.driver.nunique())
        for metric in ("ade", "fde"):
            v = g[metric].values
            row[f"{metric}_median"] = float(np.nanmedian(v))
            m, lo, hi = ci(v, g.driver.values, np.mean, n_boot)
            row.update({f"{metric}_mean": m, f"{metric}_mean_lo_driver": lo, f"{metric}_mean_hi_driver": hi})
            m2, lo2, hi2 = ci(v, g.parent.values, np.mean, n_boot)
            row.update({f"{metric}_mean_lo_drive": lo2, f"{metric}_mean_hi_drive": hi2})
            md, mlo, mhi = ci(v, g.driver.values, np.median, max(100, n_boot // 5))
            row.update({f"{metric}_median_lo_driver": mlo, f"{metric}_median_hi_driver": mhi})
            row[f"{metric}_p90"] = float(np.nanpercentile(v, 90))
        for c in ("c10", "c25", "c50", "c100"):
            if c in g:
                row[c] = float(g[c].mean())
        for s_ in ("s50_90", "s100_80"):
            if s_ in g:
                row[s_] = float(g[s_].astype(float).mean())
        for f_ in ("fde_gt_500", "lost_road"):
            if f_ in g:
                row[f"{f_}_rate"] = float(g[f_].astype(float).mean())
        row["n_missing_points"] = int(g.n_missing.sum()) if "n_missing" in g else 0
        if "topk_ade" in g and g["topk_ade"].notna().any():          # offline protocols: oracle over the K candidates
            row["topk_ade_median"] = float(np.nanmedian(g["topk_ade"].values))
            row["topk_c100"] = float(np.nanmean(g["topk_c100"].values)) if "topk_c100" in g else np.nan
            row["topk_K"] = int(np.nanmax(g["topk_K"].values)) if "topk_K" in g else 0
        rows.append(row)
    return pd.DataFrame(rows)


def learned_pairs(df):
    """Paired comparisons for the learned motion models: each learned_<kind>_s<k> vs the calibrated and nominal steering DR
    (same protocol), map_pf_learned_<kind>_s<k> vs map_pf_steer (O-M-Hp), and beam vs filter / beam none vs beam steer."""
    pairs = []
    for m in sorted(df.method.unique()):
        if m.startswith("learned_"):
            tag = m[len("learned_"):].split("__")[0]
            pairs += [((p, p), (f"learned_{tag}_minus_calib", m, "dr_steer_calib__G3")) for p in ("O-N-Hp", "O-N-Hm")]
            pairs += [((p, p), (f"learned_{tag}_minus_nominal", m, "dr_steer__G3")) for p in ("O-N-Hp", "O-N-Hm")]
        if m.startswith("map_pf_learned_"):
            tag = m[len("map_pf_learned_"):].split("__")[0]
            pairs.append((("O-M-Hp", "O-M-Hp"), (f"map_learned_{tag}_minus_map_nominal", m, "map_pf_steer__G3")))
            pairs.append((("O-M-Hp", "O-N-Hp"), (f"map_learned_{tag}_minus_nomap_learned", m, f"learned_{tag}__G3")))
    # perturbation probes (research ablation): perturbed input vs the clean run of the same method, same protocol
    for m in sorted(df.method.unique()):
        for base in ("dr_steer", "dr_none"):
            if m.startswith(base + "_") and m.endswith("__G3") and m != f"{base}__G3" and not m.startswith("dr_steer_off") and not m.startswith("dr_steer_calib") and not m.startswith("dr_steer_flip"):
                sfx = m[len(base) + 1:-len("__G3")]
                if sfx in ("pq1", "pn05", "s1hz") or sfx.startswith("p") or sfx.startswith("s1"):
                    pairs += [((p_, p_), (f"perturb_{sfx}_{base}", m, f"{base}__G3" if base == "dr_steer" else "dr_none__G1")) for p_ in ("O-N-Hp", "O-N-Hm")]
    # initialisation ablations (start shifted / heading noise) vs the clean run of the same method, and the learned attacker on
    # perturbed inputs vs its clean run
    for m in sorted(df.method.unique()):
        for base, ref in (("dr_steer", "dr_steer__G3"), ("map_pf_steer", "map_pf_steer__G3")):
            if m.startswith(base + "_i") and m.endswith("__G3"):
                sfx = m[len(base) + 1:-len("__G3")]
                proto = "O-N-Hp" if base == "dr_steer" else "O-M-Hp"
                pairs.append(((proto, proto), (f"init_{sfx}_{base}", m, ref)))
        if m.startswith("learned_") and "__" in m:
            tag = m[len("learned_"):].split("__")[0]
            for sfx in ("pq1", "pn05", "s1hz"):
                if tag.endswith("_" + sfx):
                    clean = f"learned_{tag[:-len(sfx) - 1]}__G3"
                    if clean in set(df.method):
                        pairs.append((("O-N-Hp", "O-N-Hp"), (f"perturb_{sfx}_learned_{tag[:-len(sfx) - 1]}", m, clean)))
    # augmented attacker vs the plain attacker on the same (perturbed or clean) inputs
    ms = set(df.method)
    for sfx in ("", "_pq1", "_pn05", "_s1hz"):
        a, b = f"learned_gru_aug05_s0{sfx}__G3", f"learned_gru_s0{sfx}__G3"
        if a in ms and b in ms:
            pairs.append((("O-N-Hp", "O-N-Hp"), (f"aug05_minus_plain{sfx or '_clean'}", a, b)))
    # warm-start variant (research rows): the same method given the drive's own CAN / IMU history before t0 vs its cold start,
    # and the warmed online-offset DR vs the training-fold calibrated DR (does own history beat calibration?)
    for base, grp in (("dr_steer_offonline", "G3"), ("dr_imu_yaw_rls", "G8"), ("learned_gru_s0", "G3")):
        for sfx in ("_wu5", "_wuall"):
            a, b = f"{base}{sfx}__{grp}", f"{base}__{grp}"
            if a in ms and b in ms:
                pairs += [((p_, p_), (f"warmstart{sfx}_{base}", a, b)) for p_ in ("O-N-Hp", "O-N-Hm")]
    for sfx in ("_wu5", "_wuall"):
        a = f"dr_steer_offonline{sfx}__G3"
        if a in ms:
            pairs += [((p_, p_), (f"warmstart{sfx}_offonline_minus_calib", a, "dr_steer_calib__G3")) for p_ in ("O-N-Hp", "O-N-Hm")]
            pairs += [((p_, p_), (f"warmstart{sfx}_offonline_minus_nominal", a, "dr_steer__G3")) for p_ in ("O-N-Hp", "O-N-Hm")]
    # supplementary signal groups vs the core steering DR on the common episodes
    if "dr_can_yaw__G4" in set(df.method):
        pairs += [((p_, p_), ("G4_canyaw_minus_G3_steer", "dr_can_yaw__G4", "dr_steer__G3")) for p_ in ("O-N-Hp", "O-N-Hm")]
    for m, tag in (("dr_imu_yaw__G8", "G8_imu"), ("dr_imu_yaw_sb__G8", "G8_imu_sb"), ("dr_imu_yaw_rls__G8", "G8_imu_rls")):
        if m in set(df.method):
            pairs += [((p_, p_), (f"{tag}_minus_G3_steer", m, "dr_steer__G3")) for p_ in ("O-N-Hp", "O-N-Hm")]
            pairs += [(("O-N-Hp", "O-N-Hp"), (f"{tag}_minus_G5_curvature", m, "dr_curvature__G5"))]
    for m, tag in (("map_pf_imu_yaw__G8", "map_G8_imu"), ("map_pf_imu_yaw_rls__G8", "map_G8_imu_rls")):
        if m in set(df.method):
            pairs += [((p_, p_), (f"{tag}_minus_map_G3_steer", m, "map_pf_steer__G3")) for p_ in ("O-M-Hp", "O-M-Hm")]
    for h in ("Hp", "Hm"):
        pairs.append(((f"F-M-{h}", f"O-M-{h}"), ("beam_minus_pf_steer", "map_beam_steer__G3", "map_pf_steer__G3")))
        pairs.append(((f"F-M-{h}", f"F-M-{h}"), ("beam_calib_minus_beam_nominal", "map_beam_steer_calib__G3", "map_beam_steer__G3")))
        pairs.append(((f"F-M-{h}", f"F-M-{h}"), ("beam_steer_minus_beam_none", "map_beam_steer__G3", "map_beam_none__G1")))
        pairs.append(((f"O-M-{h}", f"O-M-{h}"), ("pf_steer_minus_pf_none", "map_pf_steer__G3", "map_pf_none__G1")))
    return pairs


def gains(df, n_boot):
    rows = []
    for (proto_a, proto_b), (name, a, b) in [((p, p), pair) for p in df.protocol.unique() for pair in GAIN_PAIRS] + learned_pairs(df) + \
            [(("O-M-Hp", "O-N-Hp"), ("map_minus_nomap_steer", "map_pf_steer__G3", "dr_steer__G3")), (("O-M-Hp", "O-N-Hp"), ("map_minus_nomap_speed", "map_pf_none__G1", "dr_none__G1")),
             (("O-M-Hm", "O-N-Hm"), ("map_minus_nomap_steer", "map_pf_steer__G3", "dr_steer__G3")), (("O-M-Hp", "O-N-Hp"), ("map_calib_minus_nomap_calib", "map_pf_steer_calib__G3", "dr_steer_calib__G3"))]:
        A = df[(df.protocol == proto_a) & (df.method == a)]
        B = df[(df.protocol == proto_b) & (df.method == b)]
        if A.empty or B.empty:
            continue
        for (win, fold), ga in A.groupby(["win", "fold"]):
            gb = B[(B.win == win) & (B.fold == fold)]
            m = ga[["episode_id", "driver", "parent", "ade", "c100"]].merge(gb[["episode_id", "ade", "c100"]], on="episode_id", suffixes=("_a", "_b"))
            if len(m) < 5:
                continue
            d = m.ade_a.values - m.ade_b.values
            mean, lo, hi = ci(d, m.driver.values, np.mean, n_boot)
            med, mlo, mhi = ci(d, m.driver.values, np.median, max(100, n_boot // 5))
            rows.append(dict(gain=name, protocol=f"{proto_a} vs {proto_b}" if proto_a != proto_b else proto_a, method_a=a, method_b=b, win=win, fold=fold, n_pairs=len(m),
                             n_drivers=m.driver.nunique(), ade_diff_mean=mean, ade_diff_lo=lo, ade_diff_hi=hi, ade_diff_median=med, ade_diff_median_lo=mlo, ade_diff_median_hi=mhi,
                             win_rate=float(np.mean(d < 0)), c100_diff_mean=float(np.mean(m.c100_a.values - m.c100_b.values))))
    return pd.DataFrame(rows)


def strata(df, n_boot):
    rows = []
    df = df.copy()
    df["receiver"] = np.where(df.gps_source.astype(str).str.contains("qcom"), "qcom_1hz", np.where(df.gps_rate_hz >= 8, "ublox_10hz", "ublox_1hz"))
    # input layer = native carState rate of the drive (rlog-derived 100 Hz vs qlog-only 10 Hz) from manifests/native_manifest.csv;
    # the episode manifest's `log_native` only says that native event tables exist (True for every eligible episode)
    nat_path = os.path.join(paths.MANIFESTS, "native_manifest.csv")
    if os.path.exists(nat_path) and "cs_rate_hz" not in df.columns:
        nat = pd.read_csv(nat_path, usecols=["driver", "drive", "cs_rate_hz"])
        df = df.merge(nat, on=["driver", "drive"], how="left")
    if "cs_rate_hz" in df.columns:
        df["input_layer"] = np.where(df.cs_rate_hz >= 50, "native_100hz", np.where(df.cs_rate_hz.notna(), "native_10hz", "unknown"))
    else:
        df["input_layer"] = "unknown"
    for strat in ("receiver", "input_layer"):
        for (proto, method, win, fold, lvl), g in df.groupby(["protocol", "method", "win", "fold", strat]):
            if len(g) < 5:
                continue
            m, lo, hi = ci(g.ade.values, g.driver.values, np.median, max(100, n_boot // 5))
            rows.append(dict(stratum=strat, level=lvl, protocol=proto, method=method, win=win, fold=fold, n_episodes=len(g), n_drivers=g.driver.nunique(),
                             ade_median=float(np.nanmedian(g.ade.values)), ade_median_lo=lo, ade_median_hi=hi, c100=float(g.c100.mean()) if "c100" in g else np.nan))
    return pd.DataFrame(rows)


def md_table(df, cols, fmt=None):
    fmt = fmt or {}
    cols = [c for c in cols if c in df.columns]
    d = df[cols].copy()
    for c in cols:
        if c in fmt:
            d[c] = d[c].map(fmt[c])
    return d.to_markdown(index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--main-folds", nargs="+", default=["test", "few_shot_heldout"])
    args = ap.parse_args()
    pq_path = os.path.join(paths.SCORES, args.run, "episodes_all.parquet")
    ep = pd.read_parquet(pq_path) if os.path.exists(pq_path) else pd.read_csv(os.path.join(paths.SCORES, args.run, "episodes_all.csv"))
    man = pd.read_parquet(args.manifest)
    keep = [c for c in ["episode_id", "fold", "gps_source", "gps_rate_hz", "log_native", "admission", "platform", "set"] if c in man.columns and c not in ep.columns]
    df = ep.merge(man[["episode_id"] + keep], on="episode_id", how="left")
    if "fold" not in df:
        df["fold"] = "all"
    out_dir = os.path.join(paths.SCORES, args.run, "tables")
    os.makedirs(out_dir, exist_ok=True)
    H = headline(df, args.n_boot); H.to_csv(os.path.join(out_dir, "headline.csv"), index=False)
    G = gains(df, args.n_boot); G.to_csv(os.path.join(out_dir, "gains.csv"), index=False)
    S = strata(df, args.n_boot); S.to_csv(os.path.join(out_dir, "strata.csv"), index=False)
    # report
    f1 = lambda v: f"{v:,.0f}" if pd.notna(v) else ""  # noqa: E731
    f2 = lambda v: f"{v:.2f}" if pd.notna(v) else ""  # noqa: E731
    L = [f"# {args.run}: full baseline results (benchmark v1)\n",
         f"Episodes scored: {len(df)} rows over {df.episode_id.nunique()} episodes, {df.parent.nunique()} drives, {df.driver.nunique()} drivers; folds: {df.fold.value_counts().to_dict()}.",
         "CIs: 95% percentile cluster bootstrap resampling drivers (primary); drive-level CIs in the CSV. Medians are more robust than means here because dead reckoning errors have heavy tails.\n"]
    main = H[H.fold.isin(args.main_folds)]
    for win in ["w05", "w15", "full", "w01"]:
        sub = main[main.win == win].sort_values(["protocol", "ade_median"])
        if sub.empty:
            continue
        L += [f"## {win}: main folds ({', '.join(args.main_folds)})",
              md_table(sub, ["protocol", "method", "fold", "n_episodes", "n_drivers", "ade_median", "ade_median_lo_driver", "ade_median_hi_driver", "ade_mean", "ade_p90", "fde_median", "c50", "c100", "s50_90", "fde_gt_500_rate", "lost_road_rate"],
                       {k: f1 for k in ["ade_median", "ade_median_lo_driver", "ade_median_hi_driver", "ade_mean", "ade_p90", "fde_median"]} | {k: f2 for k in ["c50", "c100", "s50_90", "fde_gt_500_rate", "lost_road_rate"]}), ""]
    val = H[H.fold == "val"]
    if not val.empty:
        L += ["## val fold (model selection fold, reported for transparency)"]
        for win in ["w05", "w15", "full"]:
            sub = val[val.win == win].sort_values(["protocol", "ade_median"])
            if not sub.empty:
                L += [f"### {win}", md_table(sub, ["protocol", "method", "n_episodes", "n_drivers", "ade_median", "ade_median_lo_driver", "ade_median_hi_driver", "ade_mean", "c100", "s50_90", "fde_gt_500_rate"],
                                               {k: f1 for k in ["ade_median", "ade_median_lo_driver", "ade_median_hi_driver", "ade_mean"]} | {k: f2 for k in ["c100", "s50_90", "fde_gt_500_rate"]}), ""]
    if not G.empty:
        L += ["## Paired gains (ADE difference a minus b on common episodes; negative = a better)",
              md_table(G.sort_values(["gain", "win", "fold"]), ["gain", "protocol", "win", "fold", "n_pairs", "n_drivers", "ade_diff_median", "ade_diff_median_lo", "ade_diff_median_hi", "ade_diff_mean", "win_rate", "c100_diff_mean"],
                       {k: f1 for k in ["ade_diff_median", "ade_diff_median_lo", "ade_diff_median_hi", "ade_diff_mean"]} | {"win_rate": f2, "c100_diff_mean": f2}), ""]
    stab = []
    for (proto, method, win), ref in PILOT_REF.items():
        sub = H[(H.protocol == proto) & (H.method == method) & (H.win == win) & H.fold.isin(["val", "test"])]
        for r in sub.itertuples(index=False):
            stab.append(dict(protocol=proto, method=method, win=win, fold=r.fold, pilot_median_m=ref, fold_median_m=r.ade_median, ci_lo=r.ade_median_lo_driver, ci_hi=r.ade_median_hi_driver, pilot_inside_ci=bool(r.ade_median_lo_driver <= ref <= r.ade_median_hi_driver)))
    if stab:
        L += ["## Stability against the 20-drive pilot (train-fold drivers)", pd.DataFrame(stab).round(1).to_markdown(index=False), ""]
    if not S.empty:
        L += ["## Strata (median ADE; never by vehicle label)",
              md_table(S[S.fold.isin(args.main_folds) & S.win.isin(["w05", "w15"])].sort_values(["stratum", "protocol", "method", "win", "level"]),
                       ["stratum", "level", "protocol", "method", "win", "fold", "n_episodes", "n_drivers", "ade_median", "ade_median_lo", "ade_median_hi", "c100"],
                       {k: f1 for k in ["ade_median", "ade_median_lo", "ade_median_hi"]} | {"c100": f2}), ""]
    mm = os.path.join(paths.PREDS, args.run, "map_missing.csv")
    if os.path.exists(mm):
        m = pd.read_csv(mm)
        L += ["## Map coverage", f"Episodes recorded as map_missing (no tiles within the start radius; to be rerun after tiling): {len(m)} rows; by protocol/method: {m.groupby(['protocol', 'method']).size().to_dict()}", ""]
    with open(os.path.join(paths.REPORTS, f"{args.run}_results.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("written", os.path.join(paths.REPORTS, f"{args.run}_results.md"), "and", out_dir)


if __name__ == "__main__":
    main()
