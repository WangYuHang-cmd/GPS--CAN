#!/usr/bin/env python3
"""Paper tables from scores/<run>/tables/{headline,gains}.csv -> reports/paper/tables/*.csv and *.md (idempotent).

T1 main board (w05): no-map and map methods, heading known / unknown, test + few_shot.
T2 horizons: w01 / w05 / w15 / full for the key methods.
T3 learned 2x2: no-map (3 seeds) vs map PF with the learned omega, per fold, with paired gains vs nominal DR.
T4 supplementary groups: G4 CAN yaw rate, G5 curvature (class F), G8 gyro (least squares), vs steering DR.
T5 robustness (val): perturbation probes on DR and on the learned attacker (plain and noise-augmented), initialisation ablations.
T6 offline Top-K oracle for the beam search.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402

OUT = os.path.join(paths.GPS_DIR, "reports", "paper", "tables")
FOLD = {"test": "test", "few_shot_heldout": "few-shot", "val": "val"}
NAMES = {
    "p0_hold__G1": "hold start (G1)", "prior__G1": "start prior (G1)", "dr_none__G1": "speed only, straight line (G1)",
    "retrieval__G1": "route retrieval (G1)", "dr_steer__G3": "steering DR, nominal (G3)", "dr_steer_offonline__G3": "steering DR, online offset (G3)",
    "dr_steer_calib__G3": "steering DR, train-fold calibrated (G3)", "dr_can_yaw__G4": "CAN yaw-rate DR (G4)", "dr_curvature__G5": "curvature DR, class F reference (G5)",
    "dr_imu_yaw_rls__G8": "device gyro DR, least-squares bias (G8)", "learned_gru": "learned causal GRU (G3)", "learned_resid": "learned physics residual + GRU (G3)",
    "learned_resid_nogru": "learned physics residual, no GRU (G3)", "map_pf_steer__G3": "map particle filter, nominal steering (G3)",
    "map_pf_steer_calib__G3": "map particle filter, calibrated steering (G3)", "map_pf_none__G1": "map particle filter, speed only (G1)",
    "map_pf_curvature__G5": "map particle filter, class F curvature (G5)", "map_pf_imu_yaw_rls__G8": "map particle filter, device gyro (G8)",
    "map_beam_steer__G3": "offline beam search, nominal steering (G3)", "map_beam_steer_calib__G3": "offline beam search, calibrated steering (G3)",
    "map_beam_none__G1": "offline beam search, speed only (G1)",
}
PROTO = {"O-N-Hp": ("no map", "known"), "O-N-Hm": ("no map", "unknown"), "O-M-Hp": ("map, online", "known"), "O-M-Hm": ("map, online", "unknown"),
         "F-M-Hp": ("map, offline", "known"), "F-M-Hm": ("map, offline", "unknown")}


def fmt_ci(r, med="ade_median", lo="ade_median_lo_driver", hi="ade_median_hi_driver"):
    return f"{r[med]:,.0f} [{r[lo]:,.0f}, {r[hi]:,.0f}]"


def write(df, name, floatfmt=None):
    df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
    with open(os.path.join(OUT, f"{name}.md"), "w") as f:
        f.write(df.to_markdown(index=False, floatfmt=floatfmt or ".2f") + "\n")
    print(f"{name}: {len(df)} rows")


def seed_rows(h, kind, proto, win, fold):
    """Median ADE over seeds for learned_<kind>_s{0,1,2}; returns (mean of medians, sd, n seeds, n episodes)."""
    rows = h[(h.protocol == proto) & (h.win == win) & (h.fold == fold) & h.method.str.match(rf"^learned_{kind}_s\d__G3$")]
    if rows.empty:
        return None
    return rows.ade_median.mean(), rows.ade_median.std(ddof=0), len(rows), int(rows.n_episodes.iloc[0]), rows.c100.mean(), rows.s50_90.mean()


def t1_main(h):
    out = []
    folds = ["test", "few_shot_heldout"]
    base_methods = ["dr_curvature__G5", "dr_none__G1", "dr_steer__G3", "dr_steer_offonline__G3", "dr_steer_calib__G3", "prior__G1", "retrieval__G1", "p0_hold__G1"]
    for proto in ("O-N-Hp", "O-N-Hm"):
        setting, heading = PROTO[proto]
        for m in base_methods:
            row = dict(setting=setting, heading=heading, method=NAMES[m])
            for fold in folds:
                r = h[(h.protocol == proto) & (h.method == m) & (h.win == "w05") & (h.fold == fold)]
                if len(r):
                    r = r.iloc[0]
                    row[f"{FOLD[fold]} median ADE [CI]"] = fmt_ci(r); row[f"{FOLD[fold]} c100"] = round(r.c100, 2); row[f"{FOLD[fold]} S(50,0.9)"] = round(r.s50_90, 2)
            out.append(row)
        for kind in ("gru", "resid", "resid_nogru"):
            row = dict(setting=setting, heading=heading, method=NAMES[f"learned_{kind}"] + ", 3 seeds")
            for fold in folds:
                sr = seed_rows(h, kind, proto, "w05", fold)
                if sr:
                    mu, sd, n, ne, c100, s50 = sr
                    row[f"{FOLD[fold]} median ADE [CI]"] = f"{mu:,.0f} ± {sd:,.0f} (seeds)"; row[f"{FOLD[fold]} c100"] = round(c100, 2); row[f"{FOLD[fold]} S(50,0.9)"] = round(s50, 2)
            out.append(row)
    for proto in ("O-M-Hp", "O-M-Hm", "F-M-Hp", "F-M-Hm"):
        setting, heading = PROTO[proto]
        meths = ["map_pf_steer__G3", "map_pf_steer_calib__G3", "map_pf_none__G1"] if proto.startswith("O") else ["map_beam_steer__G3", "map_beam_steer_calib__G3", "map_beam_none__G1"]
        for m in meths:
            row = dict(setting=setting, heading=heading, method=NAMES[m])
            for fold in folds:
                r = h[(h.protocol == proto) & (h.method == m) & (h.win == "w05") & (h.fold == fold)]
                if len(r):
                    r = r.iloc[0]
                    row[f"{FOLD[fold]} median ADE [CI]"] = fmt_ci(r); row[f"{FOLD[fold]} c100"] = round(r.c100, 2); row[f"{FOLD[fold]} S(50,0.9)"] = round(r.s50_90, 2)
                    if proto.startswith("F") and "topk_ade_median" in r and pd.notna(r.topk_ade_median):
                        row[f"{FOLD[fold]} Top-10 oracle ADE"] = f"{r.topk_ade_median:,.0f}"
            out.append(row)
    n = h[(h.win == "w05") & (h.method == "dr_steer__G3") & (h.protocol == "O-N-Hp")].set_index("fold").n_episodes
    df = pd.DataFrame(out)
    df.attrs["note"] = f"5-min windows; episodes: test {n.get('test', 0)}, few-shot {n.get('few_shot_heldout', 0)} (map rows: all episodes with map predictions)"
    return df


def t2_horizons(h):
    rows = []
    for win in ("w01", "w05", "w15", "full"):
        for proto, m in (("O-M-Hp", "map_pf_steer__G3"), ("F-M-Hp", "map_beam_steer__G3"), ("O-M-Hm", "map_pf_steer__G3"), ("F-M-Hm", "map_beam_steer__G3"),
                         ("O-N-Hp", "dr_curvature__G5"), ("O-N-Hp", "dr_none__G1"), ("O-N-Hp", "dr_steer__G3")):
            for fold in ("test", "few_shot_heldout"):
                r = h[(h.protocol == proto) & (h.method == m) & (h.win == win) & (h.fold == fold)]
                if len(r):
                    r = r.iloc[0]
                    rows.append(dict(window=win, setting=PROTO[proto][0], heading=PROTO[proto][1], method=NAMES[m], fold=FOLD[fold], n=int(r.n_episodes),
                                     median_ADE_CI=fmt_ci(r), mean_ADE=round(r.ade_mean), c50=round(r.c50, 2), c100=round(r.c100, 2), S50_90=round(r.s50_90, 2), lost_road=round(r.lost_road_rate, 2)))
    return pd.DataFrame(rows)


def t3_learned(h, g):
    rows = []
    for fold in ("val", "test", "few_shot_heldout"):
        for kind, label in (("gru", "causal GRU"), ("resid", "physics residual + GRU"), ("resid_nogru", "physics residual, no GRU")):
            sr = seed_rows(h, kind, "O-N-Hp", "w05", fold)
            if not sr:
                continue
            mu, sd, n, ne, _, _ = sr
            ref = h[(h.protocol == "O-N-Hp") & (h.method == "dr_steer__G3") & (h.win == "w05") & (h.fold == fold)].iloc[0]
            gains = g[(g.gain.str.match(rf"^learned_{kind}_s\d_minus_nominal$")) & (g.protocol == "O-N-Hp") & (g.win == "w05") & (g.fold == fold)]
            mp = h[(h.protocol == "O-M-Hp") & (h.win == "w05") & (h.fold == fold) & h.method.str.match(rf"^map_pf_learned_{kind}_s\d__G3$")]
            mref = h[(h.protocol == "O-M-Hp") & (h.method == "map_pf_steer__G3") & (h.win == "w05") & (h.fold == fold)]
            mg = g[(g.gain.str.match(rf"^map_learned_{kind}_s\d_minus_map_nominal$")) & (g.win == "w05") & (g.fold == fold)]
            rows.append(dict(fold=FOLD[fold], model=label, seeds=n,
                             no_map_median_ADE=f"{mu:,.0f} ± {sd:,.0f}", nominal_DR=f"{ref.ade_median:,.0f}",
                             paired_vs_nominal=(f"{gains.ade_diff_median.mean():+,.0f} (seeds {gains.ade_diff_median.min():+,.0f} to {gains.ade_diff_median.max():+,.0f}), win {gains.win_rate.mean():.2f}" if len(gains) else ""),
                             with_map_median_ADE=(f"{mp.ade_median.mean():,.0f}" + (f" ± {mp.ade_median.std(ddof=0):,.0f}" if len(mp) > 1 else "") + f" ({len(mp)} seed{'s' if len(mp) > 1 else ''})") if len(mp) else "",
                             map_nominal_steering=f"{mref.iloc[0].ade_median:,.0f}" if len(mref) else "",
                             paired_vs_map_nominal=(f"{mg.ade_diff_median.mean():+,.0f}, win {mg.win_rate.mean():.2f}" if len(mg) else "")))
    return pd.DataFrame(rows)


def t4_supplements(h, g):
    rows = []
    for m, gain in (("dr_can_yaw__G4", "G4_canyaw_minus_G3_steer"), ("dr_imu_yaw_rls__G8", "G8_imu_rls_minus_G3_steer"), ("dr_curvature__G5", None)):
        for fold in ("val", "test", "few_shot_heldout"):
            r = h[(h.protocol == "O-N-Hp") & (h.method == m) & (h.win == "w05") & (h.fold == fold)]
            if r.empty:
                continue
            r = r.iloc[0]
            gg = g[(g.gain == gain) & (g.protocol == "O-N-Hp") & (g.win == "w05") & (g.fold == fold)] if gain else pd.DataFrame()
            rows.append(dict(method=NAMES[m], fold=FOLD[fold], n=int(r.n_episodes), median_ADE_CI=fmt_ci(r), c100=round(r.c100, 2),
                             paired_vs_steering_DR=(f"{gg.iloc[0].ade_diff_median:+,.0f} [{gg.iloc[0].ade_diff_median_lo:+,.0f}, {gg.iloc[0].ade_diff_median_hi:+,.0f}], win {gg.iloc[0].win_rate:.2f}" if len(gg) else "reference (class F, not CAN-only)")))
    return pd.DataFrame(rows)


def t5_robustness(h, g):
    rows = []
    spec = [("perturb_pq1_dr_steer", "steering DR", "speed quantised to 1 m/s"), ("perturb_pn05_dr_steer", "steering DR", "speed noise 0.5 m/s"), ("perturb_s1hz_dr_steer", "steering DR", "steering held at 1 Hz"),
            ("perturb_pq1_dr_none", "straight line", "speed quantised to 1 m/s"), ("perturb_pn05_dr_none", "straight line", "speed noise 0.5 m/s"),
            ("perturb_pq1_learned_gru_s0", "learned GRU (plain)", "speed quantised to 1 m/s"), ("perturb_pn05_learned_gru_s0", "learned GRU (plain)", "speed noise 0.5 m/s"), ("perturb_s1hz_learned_gru_s0", "learned GRU (plain)", "steering held at 1 Hz"),
            ("perturb_pq1_learned_gru_aug05_s0", "learned GRU (noise-augmented)", "speed quantised to 1 m/s"), ("perturb_pn05_learned_gru_aug05_s0", "learned GRU (noise-augmented)", "speed noise 0.5 m/s"), ("perturb_s1hz_learned_gru_aug05_s0", "learned GRU (noise-augmented)", "steering held at 1 Hz"),
            ("aug05_minus_plain_clean", "noise-augmented vs plain GRU", "clean inputs"), ("aug05_minus_plain_pn05", "noise-augmented vs plain GRU", "speed noise 0.5 m/s"),
            ("init_ip20_dr_steer", "steering DR", "start shifted 20 m"), ("init_ih10_dr_steer", "steering DR", "heading noise 10 deg"), ("init_ih30_dr_steer", "steering DR", "heading noise 30 deg"),
            ("init_ip20_map_pf_steer", "map particle filter", "start shifted 20 m"), ("init_ih10_map_pf_steer", "map particle filter", "heading noise 10 deg"), ("init_ih30_map_pf_steer", "map particle filter", "heading noise 30 deg")]
    for gain, method, pert in spec:
        gg = g[(g.gain == gain) & (g.win == "w05") & (g.fold == "val") & g.protocol.str.endswith("Hp")]
        if gg.empty:
            continue
        r = gg.iloc[0]
        rows.append(dict(method=method, perturbation=pert, n_pairs=int(r.n_pairs), paired_median_diff_m=f"{r.ade_diff_median:+,.0f} [{r.ade_diff_median_lo:+,.0f}, {r.ade_diff_median_hi:+,.0f}]",
                         paired_mean_diff_m=f"{r.ade_diff_mean:+,.0f}", win_rate=round(r.win_rate, 2)))
    return pd.DataFrame(rows)


def t8_warmstart(h, g):
    """Warm-start research rows: the same method with the drive's own pre-t0 CAN / IMU history vs its cold start (paired),
    plus the warmed online-offset DR against the nominal and the train-fold calibrated DR."""
    rows = []
    spec = [("warmstart_wu5_dr_steer_offonline", "steering DR, online offset", "5 min of own history vs cold start"),
            ("warmstart_wuall_dr_steer_offonline", "steering DR, online offset", "all own history vs cold start"),
            ("warmstart_wu5_dr_imu_yaw_rls", "device gyro DR, least-squares bias (G8)", "5 min of own history vs cold start"),
            ("warmstart_wuall_dr_imu_yaw_rls", "device gyro DR, least-squares bias (G8)", "all own history vs cold start"),
            ("warmstart_wu5_learned_gru_s0", "learned GRU (seed 0)", "5 min of own history vs cold start"),
            ("warmstart_wuall_learned_gru_s0", "learned GRU (seed 0)", "last hour of own history vs cold start"),
            ("warmstart_wu5_offonline_minus_nominal", "warmed online-offset DR vs nominal DR", "5 min of own history"),
            ("warmstart_wuall_offonline_minus_nominal", "warmed online-offset DR vs nominal DR", "all own history"),
            ("warmstart_wu5_offonline_minus_calib", "warmed online-offset DR vs train-fold calibrated DR", "5 min of own history"),
            ("warmstart_wuall_offonline_minus_calib", "warmed online-offset DR vs train-fold calibrated DR", "all own history"),
            ("warmstart_wuall_imu_rls_minus_G3_steer", "warmed device gyro DR (G8) vs nominal steering DR", "all own history"),
            ("warmstart_wuall_imu_rls_minus_offonline_wuall", "warmed device gyro DR (G8) vs warmed online-offset DR", "all own history"),
            ("warmstart_wuall_imu_rls_minus_G5_curvature", "warmed device gyro DR (G8) vs class F curvature DR", "all own history")]
    for gain, method, pert in spec:
        for proto in ("O-N-Hp", "O-N-Hm"):
            for fold in ("val", "test", "few_shot_heldout"):
                gg = g[(g.gain == gain) & (g.win == "w05") & (g.fold == fold) & (g.protocol == proto)]
                if gg.empty:
                    continue
                r = gg.iloc[0]
                rows.append(dict(method=method, comparison=pert, heading=PROTO[proto][1], fold=FOLD[fold], n_pairs=int(r.n_pairs),
                                 paired_median_diff_m=f"{r.ade_diff_median:+,.0f} [{r.ade_diff_median_lo:+,.0f}, {r.ade_diff_median_hi:+,.0f}]",
                                 paired_mean_diff_m=f"{r.ade_diff_mean:+,.0f}", win_rate=round(r.win_rate, 2)))
    return pd.DataFrame(rows)


def t6_topk(h):
    rows = []
    for proto in ("F-M-Hp", "F-M-Hm"):
        for m in ("map_beam_steer__G3", "map_beam_none__G1"):
            for win in ("w01", "w05", "w15"):
                for fold in ("val", "test", "few_shot_heldout"):
                    r = h[(h.protocol == proto) & (h.method == m) & (h.win == win) & (h.fold == fold)]
                    if r.empty or pd.isna(r.iloc[0].get("topk_ade_median", np.nan)):
                        continue
                    r = r.iloc[0]
                    rows.append(dict(heading=PROTO[proto][1], method=NAMES[m], window=win, fold=FOLD[fold], n=int(r.n_episodes), top1_median_ADE=round(r.ade_median), topK_oracle_median_ADE=round(r.topk_ade_median),
                                     top1_c100=round(r.c100, 2), topK_oracle_c100=round(r.topk_c100, 2), K=int(r.topk_K)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v1")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    h = pd.read_csv(os.path.join(paths.SCORES, args.run, "tables", "headline.csv"))
    g = pd.read_csv(os.path.join(paths.SCORES, args.run, "tables", "gains.csv"))
    t1 = t1_main(h); write(t1, "T1_main_board_w05")
    with open(os.path.join(OUT, "T1_main_board_w05.md"), "a") as f:
        f.write(f"\n{t1.attrs.get('note', '')}\n")
    write(t2_horizons(h), "T2_horizons")
    write(t3_learned(h, g), "T3_learned_2x2")
    write(t4_supplements(h, g), "T4_supplementary_groups")
    write(t5_robustness(h, g), "T5_robustness_val")
    write(t6_topk(h), "T6_topk_oracle")
    t8 = t8_warmstart(h, g)
    if not t8.empty:
        write(t8, "T8_warmstart_research")
    print("written to", OUT, "(T7 comes from scripts/failure_analysis.py)")


if __name__ == "__main__":
    main()
