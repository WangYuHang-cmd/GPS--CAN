#!/usr/bin/env python3
"""LaTeX macros for every number quoted in the ICDE paper text (paper_icde/macros_numbers.tex), computed from the
manifests and the ICDE result tables, so the text never carries a hand-typed number. Run after icde_tables.py."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402

OUT = os.path.join(paths.GPS_DIR, "paper_icde", "macros_numbers.tex")


def n(x):
    return f"{int(round(x)):,}".replace(",", "{,}")


def main():
    m = {}
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    e = man[man.eligible_main]
    dr = e.drop_duplicates(["driver", "drive"])
    m["NEligibleEpisodes"] = n(len(e))
    m["NDrives"] = n(len(dr))
    m["NDrivers"] = n(e.driver.nunique())
    m["NPlatforms"] = n(dr.platform.nunique())
    m["NBrands"] = n(dr.platform.astype(str).str.split("_").str[0].nunique())
    full = e[e.win == "full"]
    m["NHours"] = n(full.dur_s.sum() / 3600.0)
    m["NKm"] = n(full.path_len_m.sum() / 1000.0)
    m["NDrivesQcom"] = n((dr.gps_source == "qcomdiag").sum())
    m["NDrivesUblox"] = n(dr.gps_source.astype(str).str.contains("ublox").sum())
    for fold, tag in (("test", "Test"), ("few_shot_heldout", "FewShot"), ("val", "Val")):
        for win, wt in (("w01", "One"), ("w05", "Five"), ("w15", "Fifteen"), ("full", "Full")):
            s = e[(e.fold == fold) & (e.win == win)]
            m[f"N{tag}{wt}Min"] = n(len(s))
        s = e[e.fold == fold]
        m[f"N{tag}Drivers"] = n(s.driver.nunique())
        m[f"N{tag}Drives"] = n(len(s.drop_duplicates(["driver", "drive"])))
    for fold, tag in (("test", "Test"), ("few_shot_heldout", "FewShot")):  # whole drives: length distribution per fold
        f = e[(e.fold == fold) & (e.win == "full")]
        m[f"FullDurMed{tag}"] = f"{f.dur_s.median() / 60:.0f}"
        m[f"FullDurNinety{tag}"] = f"{f.dur_s.quantile(0.9) / 60:.0f}"
        m[f"FullKmMed{tag}"] = f"{f.path_len_m.median() / 1000:.0f}"
    aw = pd.read_parquet(os.path.join(paths.MANIFESTS, "availability_wide.parquet"), columns=["driver", "drive", "gas"])
    dd = dr.merge(aw, on=["driver", "drive"], how="left")
    for fold, tag in (("test", "Test"), ("few_shot_heldout", "FewShot")):
        m[f"NPedal{tag}Drives"] = n(((dd.fold == fold) & (dd.gas == "valid")).sum())
    rg = os.path.join(paths.MANIFESTS, "map_regions.csv")
    if os.path.exists(rg):                     # OpenStreetMap extract regions needed by at least one drive's map disc
        r = pd.read_csv(rg)
        r = r[r.n_drives > 0]
        us = r.extract.str.startswith("north-america/us/")
        ca = r.extract.str.startswith("north-america/canada/")
        m["NRegionUS"], m["NRegionCA"] = n(us.sum()), n(ca.sum())
        m["NRegionOther"] = n(r[~us & ~ca].extract.str.split("/").str[:2].str.join("/").nunique())
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"))
    mm = me[me.fold.isin(["test", "few_shot_heldout"])]
    m["MapStartMissingMain"] = n((~mm.map_start_ok).sum())
    m["MapStartDistMedian"] = f"{mm.start_edge_dist_m.median():.1f}"
    audit = pd.read_csv(os.path.join(paths.GPS_DIR, "reports", "icde", "map_tile_audit.csv"))
    m["NTilesVtwo"] = n(len(audit))
    m["NTilesTruncated"] = n(audit.v1_truncated.sum())
    mt = os.path.join(paths.GPS_DIR, "reports", "icde", "macrotracking_author_sample.csv")
    if os.path.exists(mt):
        t = pd.read_csv(mt).set_index("as_released")
        m["MacroSampleMedian"] = f"{t.loc[True, 'd_median_m']:.1f}"
        m["MacroSampleGtOurs"] = f"{t.loc[True, 'gt_closest_ours_m']:.1f}"
        m["MacroSampleGtAuthors"] = f"{t.loc[True, 'gt_closest_authors_m']:.1f}"
    od = os.path.join(paths.GPS_DIR, "reports", "icde", "opd2_author_samples.csv")
    if os.path.exists(od):
        t = pd.read_csv(od)
        m["OpdSampleLogs"] = n(len(t))
        m["OpdSampleMaxMedian"] = f"{t.d_median_m.max():.2f}"
    results_macros(m)
    su = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "sig_signal_use.csv")
    if os.path.exists(su):                     # validation chunks: one input block shuffled across chunks
        t = pd.read_csv(su)
        t["ratio"] = t.loss_shuffled / t.loss
        st = t[t.block == "steer"]
        m["SigUseSteerRatioMin"] = f"{st.ratio.min():.1f}"
        m["SigUseSteerRatioMax"] = f"{st.ratio.max():.1f}"
        for g, tag in (("S2c", "SpeedPedal"), ("S4c", "AllPedal")):
            v = 100 * (t[(t.group == g) & (t.block == "pedal")].ratio - 1)
            m[f"SigUsePedal{tag}Pct"] = f"{v.mean():+.1f}".replace("+", "$+$").replace("-", "$-$")
        v = 100 * (t[t.block == "flags"].ratio - 1)
        m["SigUseFlagsPctMin"] = f"{v.min():+.1f}".replace("+", "$+$").replace("-", "$-$")
        m["SigUseFlagsPctMax"] = f"{v.max():+.1f}".replace("+", "$+$").replace("-", "$-$")
    a3 = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "a3_visibility.csv")
    if os.path.exists(a3):                     # union map vs own tiles on 60 test drives; PF seed change as the noise scale
        t = pd.read_csv(a3)
        u = t[t.comparison == "union vs own tiles"]
        m["AThreeWindows"] = n(u[u.method == "map_pf_steer__G3"].n.sum())
        m["AThreeDrives"] = n(len(pd.read_csv(os.path.join(paths.MANIFESTS, "icde_a3_drives.csv"))))
        m["AThreeIdenticalPct"] = f"{100 * (u.share_abs_diff_lt_1m * u.n).sum() / u.n.sum():.0f}"
        sd = t[t.comparison.str.startswith("seed")].set_index("win")
        for win, tag in (("w05", "Five"), ("w15", "Fifteen")):
            if win in sd.index:
                m[f"AThreeSeedMovedPct{tag}"] = f"{100 * (1 - sd.loc[win, 'share_abs_diff_lt_1m']):.0f}"
    ch = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "cohort_horizon.csv")
    if os.path.exists(ch):                     # Coh<method><proto><One|Five|Fifteen><Fold>Med; online 900 s from the online run
        t = pd.read_csv(ch)
        t = t[~((t.horizon_s == 900) & (t.source == "icde_v11") & t.protocol.str.startswith("O-"))]
        for r in t.to_dict("records"):
            if r["fold"] not in FOLD_TAG or r["horizon_s"] not in (60, 300, 900):
                continue
            h = {60: "One", 300: "Five", 900: "Fifteen"}[r["horizon_s"]]
            k = f"Coh{camel(r['method'].split('__')[0])}{r['protocol'].replace('-', '')}{h}{FOLD_TAG[r['fold']]}"
            m[k + "Med"] = m_or_fail(r["ade_median"])
        cp = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "cohort_pairs.csv")
        if os.path.exists(cp):                 # CohPair<BeamVsLineage|BeamVsPf|LineageVsPf><One|Five|Fifteen><Fold><stat>
            names = {"beam vs PF lineage": "BeamVsLineage", "beam vs PF": "BeamVsPf", "PF lineage vs PF": "LineageVsPf"}
            for r in pd.read_csv(cp).to_dict("records"):
                h = {60: "One", 300: "Five", 900: "Fifteen"}[r["horizon_s"]]
                k = f"CohPair{names[r['comparison']]}{h}{FOLD_TAG[r['fold']]}"
                for stat, col in (("Med", "median"), ("Lo", "median_lo"), ("Hi", "median_hi"), ("Mean", "mean"), ("MeanLo", "mean_lo"), ("MeanHi", "mean_hi")):
                    m[k + stat] = signed(r[col])
        for fold, tag in FOLD_TAG.items():
            d = t[(t.fold == fold) & (t.method == "map_pf_steer__G3") & (t.protocol == "O-M-Hp") & (t.horizon_s == 60)]
            if len(d):
                m[f"Coh{tag}N"] = n(d.n.iloc[0])
                m[f"Coh{tag}Drivers"] = n(d.n_drivers.iloc[0])
    fa = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "T7a_lost_road_timing_v11.csv")
    if os.path.exists(fa):                     # Fail<Pf|Beam|PfNone><stat> (w15, main folds): terminal excursions vs turns
        for r in pd.read_csv(fa).to_dict("records"):
            k = {"map_pf_steer__G3": "FailPf", "map_beam_steer__G3": "FailBeam", "map_pf_none__G1": "FailPfNone"}.get(r["method"])
            if k is None or r["win"] != "w15":
                continue
            m[k + "LostPct"] = f"{100 * r['lost_rate']:.0f}"
            m[k + "OnsetMedian"] = n(r["onset_median_s"])
            m[k + "AfterTurnPct"] = f"{100 * r['within_30s_after_turn']:.0f}"
            m[k + "ChancePct"] = f"{100 * r['within_30s_chance_approx']:.0f}"
            m[k + "SinceTurn"] = f"{r['since_last_turn_median_s']:.0f}"
    fb = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "T7b_density_strata_v11.csv")
    if os.path.exists(fb):                     # Dens<method><Low|Mid|High>Med (w05, main folds) and the tertile cuts
        t = pd.read_csv(fb)
        t = t[(t.win == "w05") & (t.fold == "main")]
        for r in t.to_dict("records"):
            lvl = r["level"].split(" ")[0].capitalize()
            m[f"Dens{camel(r['method'].split('__')[0])}{lvl}Med"] = m_or_fail(r["ade_median"])
        cuts = sorted({int(x) for lv in t.level.unique() for x in __import__("re").findall(r"(\d+)/km2", lv)})
        if len(cuts) == 2:
            m["DensCutLow"], m["DensCutHigh"] = n(cuts[0]), n(cuts[1])
    cost_macros(m)
    sig_macros(m)
    tk = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "topk_w05.csv")
    if os.path.exists(tk):                     # Topk<Method><Hp|Hm><Fold>K<One|Three|Five|Ten>, BestFirst (%), Single
        for r in pd.read_csv(tk).to_dict("records"):
            if r["fold"] not in FOLD_TAG:
                continue
            k = f"Topk{camel(r['stem'].split('__')[0])}{'Hm' if r['protocol'].endswith('Hm') else 'Hp'}{FOLD_TAG[r['fold']]}"
            for K, word in ((1, "One"), (3, "Three"), (5, "Five"), (10, "Ten")):
                m[f"{k}K{word}"] = m_or_fail(r[f"top{K}_ade_median"])
            m[k + "BestFirst"] = f"{100 * r['best_rank_is_1']:.0f}" if np.isfinite(r["best_rank_is_1"]) else "--"
            m[k + "Single"] = n(r.get("n_single_path", 0))
            m[k + "Out"] = n(r["n_output"])
    ct = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "can_trace_failure_mode.csv")
    if os.path.exists(ct):
        t = pd.read_csv(ct)
        f = t[t.outcome == "no match"]
        m["CtDiagNoMatch"] = n(len(f))
        m["CtDiagTraceNodesMed"] = n(f.trace_nodes.median())
        m["CtDiagIntersectionsMed"] = n(f.intersections_passed.median())
        m["CtDiagShareFewer"] = f"{100 * f.fewer_edges_than_intersections.mean():.0f}"
    coh = os.path.join(paths.MANIFESTS, "cohort_v11.csv")
    if os.path.exists(coh):
        c = pd.read_csv(coh)
        c = c[c.cohort]
        m["NCohortDrives"] = n(len(c))
        m["NCohortDrivers"] = n(c.driver.nunique())
    with open(OUT, "w") as f:
        f.write("% generated by scripts/icde_macros.py from the manifests and reports/icde/tables; do not edit\n")
        for k, v in sorted(m.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(m)} macros -> {OUT}")


WIN_TAG = {"w01": "One", "w05": "Five", "w15": "Fifteen", "full": "Full"}
FOLD_TAG = {"test": "Test", "few_shot_heldout": "FewShot"}
PAIR_TAG = {"map gain, steering, heading known": "MapGain", "map gain, steering, heading unknown": "MapGainHm",
            "start heading, map PF steering": "HeadPf", "start heading, map beam steering": "HeadBeam",
            "offline beam vs online PF, steering": "BeamVsPf", "learned GRU vs nominal DR, no map": "GruVsDr",
            "Elastic Pathing (speed) vs hold start": "EpVsHold", "Elastic Pathing (speed) vs map PF steering, heading unknown": "EpVsPfHm",
            "Elastic Pathing (speed) vs map beam steering, heading unknown": "EpVsBeamHm",
            "MacroTracking vs map PF, steering, online": "MacroVsPf", "Elastic Pathing, start heading used vs not": "EpHead",
            "CAN yaw rate vs steering, no map": "YawVsSteer", "device gyro vs steering, no map": "GyroVsSteer",
            "learned residual vs nominal DR, no map": "ResidVsDr", "learned GRU motion in map PF vs nominal steering": "GruPfVsPf",
            "learned residual motion in map PF vs nominal steering": "ResidPfVsPf",
            "decode output: PF lineage vs PF weighted mean": "LineageVsPf", "decode output: beam vs PF lineage": "BeamVsLineage",
            "steering value, map PF, heading known": "SteerPf", "steering value, map PF, heading unknown": "SteerPfHm",
            "steering value, map beam, heading known": "SteerBeam", "steering value, map beam, heading unknown": "SteerBeamHm",
            "Elastic Pathing vs map beam, speed only, heading unknown": "EpVsBeamNoneHm",
            "OPD-II (3 s delayed) vs map PF, steering": "OpdVsPf", "OPD-II vs MacroTracking": "OpdVsMacro",
            "OPD-II calibrated vs as released": "OpdCalVsOpd", "MacroTracking calibrated vs as released": "MacroCalVsMacro"}


DIGITS = dict(zip("0123456789", ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"]))


def camel(key):
    """Row key -> letters-only macro part (opd2_on -> OpdTwoOn)."""
    s = "".join(w.capitalize() for w in str(key).replace("-", "_").split("_"))
    return "".join(DIGITS.get(ch, ch) for ch in s)


def m_or_fail(x):
    return "fail" if not np.isfinite(x) else n(x)


def signed(x):
    """Signed difference for running text: one decimal below 100 m, typographic minus."""
    body = f"{abs(x):.1f}" if abs(x) < 100 else n(abs(x))
    return ("$-$" if x < 0 else "") + body


def results_macros(m):
    """Res<Row><Win><Fold><Stat> from tables/main_board.csv and Pair<Name><Win><Fold><Stat> from tables/pairs.csv."""
    tabs = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
    p = os.path.join(tabs, "main_board.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            if r["win"] not in WIN_TAG or r["fold"] not in FOLD_TAG:
                continue
            k = f"Res{camel(r['key'])}{WIN_TAG[r['win']]}{FOLD_TAG[r['fold']]}"
            m[k + "Med"] = m_or_fail(r["ade_median"])
            m[k + "Lo"] = m_or_fail(r["ade_lo"])
            m[k + "Hi"] = m_or_fail(r["ade_hi"])
            m[k + "Drv"] = m_or_fail(r["ade_driver_median"])
            m[k + "Route"] = f"{100 * r['route_ok']:.0f}"
            m[k + "Fail"] = n(r["n_fail"])
            m[k + "N"] = n(r["n_expected"])
            m[k + "Out"] = n(r["n_output"])
    p = os.path.join(tabs, "pairs.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            name = PAIR_TAG.get(r["comparison"])
            if name is None or r["win"] not in WIN_TAG or r["fold"] not in FOLD_TAG:
                continue
            k = f"Pair{name}{WIN_TAG[r['win']]}{FOLD_TAG[r['fold']]}"
            for stat, col in (("Med", "median"), ("Lo", "median_lo"), ("Hi", "median_hi"), ("Mean", "mean"), ("MeanLo", "mean_lo"), ("MeanHi", "mean_hi")):
                if col in r and np.isfinite(r[col]):
                    m[k + stat] = signed(r[col])
            m[k + "WinA"] = f"{100 * r['win_a_with_fail']:.0f}"
            m[k + "N"] = n(r["n_common"])
            if np.isfinite(r.get("n_drivers", np.nan)):
                m[k + "Drivers"] = n(r["n_drivers"])


def cost_macros(m):
    """RQ5: Cost<tag><Five|Fifteen> (median inference s per episode), CostRss<tag> (GB), CostGraph (median s per drive),
    CostAcc<tag> (val subset w05 median ADE), Prep<step>Min, Robust<method>MaxAbs (largest |paired median| over probes)."""
    tabs = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
    p = os.path.join(tabs, "cost.csv")
    if os.path.exists(p):
        c = pd.read_csv(p)
        for r in c.to_dict("records"):
            k = f"Cost{camel(r['tag'])}"
            for win, tag in (("w05", "Five"), ("w15", "Fifteen")):
                v = r.get(f"infer_median_s_{win}")
                if v is not None and np.isfinite(v):
                    m[k + tag] = f"{v:.2f}" if v >= 0.01 else f"{v:.3f}"
            v = r.get("infer_p90_s_w05")
            if v is not None and np.isfinite(v):
                m[k + "FiveNinety"] = f"{v:.2f}"
            m[k + "Rss"] = f"{r['peak_rss_gb']:.1f}"
        m["CostGraph"] = f"{c.graph_build_median_s.median():.1f}"
        m["CostRssMax"] = f"{c.peak_rss_gb.max():.1f}"
        e = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_cost_episodes_valhalla.csv"))
        m["CostN"] = n(len(e))
        m["CostDrives"] = n(e.episode_id.str.split("/").str[:2].str.join("/").nunique())
        m["CostNFive"] = n(e.episode_id.str.contains("/w05/").sum())
    p = os.path.join(tabs, "cost_accuracy.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            if r["win"] == "w05":
                m[f"CostAcc{camel(r['method'].split('__')[0])}"] = f"{r['ade_median']:.0f}"
    p = os.path.join(tabs, "map_prep_cost.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        m["PrepTilesMin"] = f"{t.seconds.iloc[0] / 60:.0f}"
        m["PrepValhallaMin"] = f"{t.seconds.iloc[1:].sum() / 60:.0f}"
        m["PrepValhallaGb"] = f"{t.size_gb.iloc[2]:.1f}"
    p = os.path.join(tabs, "robustness_w05.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for meth, tag in (("steering DR", "Dr"), ("straight line", "Straight")):
            d = t[t.method == meth]
            if len(d):
                m[f"Robust{tag}MaxAbs"] = f"{d['median'].abs().max():.1f}"


SIG_ADDED = {"pedals | speed": "PedSpeed", "steering | speed": "SteerSpeed", "pedals | speed+steering": "PedAll",
             "steering | speed+pedals": "SteerAll", "pressed flags | speed": "FlagsSpeed", "pressed flags | speed+steering": "FlagsAll"}
SIG_SETTING = {"map": "Map", "no map": "NoMap"}


def sig_macros(m):
    """Signal study (pedal-valid subset, w05): SigSet<Setting><Group>Med/Sd over seeds, SigGain<Setting><Added>
    Med (mean of seed medians), Min / Max (seed medians), Sig (seeds whose CI excludes 0), Seeds, Win (mean %)."""
    tabs = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
    p = os.path.join(tabs, "signal_sets_w05.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for (st, g), d in t.groupby(["setting", "group"]):
            k = f"SigSet{SIG_SETTING[st]}{camel(g)}"
            m[k + "Med"] = n(d.ade_median.mean())
            m[k + "Sd"] = n(d.ade_median.std()) if len(d) > 1 else "--"
            m[k + "N"] = n(d.n.iloc[0])
    p = os.path.join(tabs, "signal_gains_w05.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for (st, a), d in t.groupby(["setting", "added"]):
            k = f"SigGain{SIG_SETTING[st]}{SIG_ADDED[a]}"
            m[k + "Med"] = signed(d["median"].mean())
            m[k + "Min"] = signed(d["median"].min())
            m[k + "Max"] = signed(d["median"].max())
            m[k + "Sig"] = n(((d.median_lo > 0) | (d.median_hi < 0)).sum())
            m[k + "Seeds"] = n(len(d))
            m[k + "Win"] = f"{100 * d.win_a.mean():.0f}"
            m[k + "WinMin"] = f"{100 * d.win_a.min():.0f}"
            m[k + "WinMax"] = f"{100 * d.win_a.max():.0f}"
        m["SigNDrivers"] = n(t.n_drivers.iloc[0])
    p = os.path.join(tabs, "signal_reference_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            m[f"SigRef{camel(r['key'])}Med"] = m_or_fail(r["ade_median"])
            m[f"SigRef{camel(r['key'])}Fail"] = n(r["fail"])


if __name__ == "__main__":
    main()
