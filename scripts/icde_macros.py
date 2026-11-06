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
    gm = pd.read_csv(paths.GPS_MANIFEST)                                  # the released paired corpus (GNSS files present)
    rel = gm[(gm.status == "ok") & (gm["set"] == "main")]
    m["NReleaseDrives"], m["NReleaseDrivers"] = n(len(rel)), n(rel.driver.nunique())
    m["NReleaseHours"] = n((rel.trim_hi_s - rel.trim_lo_s).clip(lower=0).sum() / 3600.0)
    # released drives the benchmark does not use: no admissible anchor, no eligible window, or unconfirmed platform
    key = lambda d: d.driver.astype(str) + "/" + d.drive.astype(str)
    rel_keys, any_keys, elig_keys = set(key(rel)), set(key(man)), set(key(e))
    unused = rel_keys - elig_keys
    no_anchor = unused - any_keys
    rest = man[key(man).isin(unused - no_anchor)]
    adm = rest.groupby(key(rest)).admission.agg(lambda s: set(s))
    platform_only = {k for k, a in adm.items() if a == {"platform_unconfirmed"}}
    window_only = {k for k, a in adm.items() if a == {"ineligible_window"}}
    both = set(adm.index) - platform_only - window_only
    m["NReleaseUnusedDrives"], m["NReleaseNoAnchorDrives"] = n(len(unused)), n(len(no_anchor))
    m["NReleaseNoWindowDrives"], m["NReleasePlatformDrives"] = n(len(window_only)), n(len(platform_only) + len(both))
    m["NKm"] = n(full.path_len_m.sum() / 1000.0)
    m["NDrivesQcom"] = n((dr.gps_source == "qcomdiag").sum())
    m["NDrivesUblox"] = n(dr.gps_source.astype(str).str.contains("ublox").sum())
    ho = dr[dr.fold.isin(["test", "few_shot_heldout"])]
    m["NHeldoutDrives"], m["NHeldoutDrivers"] = n(len(ho)), n(ho.driver.nunique())
    m["NHeldoutPlatforms"] = n(ho.platform.nunique())
    tv = dr[dr.fold.isin(["train", "val"])]
    m["NTrainValDrives"], m["NTrainValDrivers"] = n(len(tv)), n(tv.driver.nunique())
    ua = dr[~dr.fold.isin(["train", "val", "test", "few_shot_heldout"])]
    m["NUnassignedDrives"], m["NUnassignedDrivers"] = n(len(ua)), n(ua.driver.nunique())
    m["NEligibleInFolds"] = n(e.fold.isin(["train", "val", "test", "few_shot_heldout"]).sum())
    pu = man[man.admission == "platform_unconfirmed"].drop_duplicates(["driver", "drive"])   # out of the CAN-only board
    m["NPlatformUnconfirmedDrives"], m["NPlatformUnconfirmedDrivers"] = n(len(pu)), n(pu.driver.nunique())
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
    m["NPedalHeldoutDrives"] = n((dd.fold.isin(["test", "few_shot_heldout"]) & (dd.gas == "valid")).sum())
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
    ok_ids = set(me[me.map_ok].episode_id)
    for fold, tag in (("test", "Test"), ("few_shot_heldout", "FewShot")):   # whole drives that end beyond the 60 km visible disc
        f = e[(e.fold == fold) & (e.win == "full") & e.episode_id.isin(ok_ids)]
        m[f"FullBeyondDisc{tag}"] = n((f.net_disp_m > 60_000).sum())
        m[f"FullMapOk{tag}"] = n(len(f))
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
            m[f"SigUsePedal{tag}PctMin"] = f"{v.min():+.1f}".replace("+", "$+$").replace("-", "$-$")   # per-seed range
            m[f"SigUsePedal{tag}PctMax"] = f"{v.max():+.1f}".replace("+", "$+$").replace("-", "$-$")
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
        t = t[~((t.horizon_s == 900) & (t.source == "icde_v11") & t.protocol.str.startswith("O-") & ~t.method.str.contains("wrapfix"))]
        for r in t.to_dict("records"):
            if r["fold"] not in FOLD_TAG or r["horizon_s"] not in (60, 300, 900):
                continue
            h = {60: "One", 300: "Five", 900: "Fifteen"}[r["horizon_s"]]
            k = f"Coh{camel(r['method'].split('__')[0])}{r['protocol'].replace('-', '')}{h}{FOLD_TAG[r['fold']]}"
            m[k + "Med"] = m_or_fail(r["ade_median"])
            m[k + "Fail"] = n(r["n"] - r["n_output"])
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
            m[k + "N"] = n(r["n_episodes"])
            m[k + "OnsetMedian"] = n(r["onset_median_s"])
            m[k + "AfterTurnPct"] = f"{100 * r['within_30s_after_turn']:.0f}"
            m[k + "ChancePct"] = f"{100 * r['within_30s_chance_approx']:.0f}"
            m[k + "SinceTurn"] = f"{r['since_last_turn_median_s']:.0f}"
    fb = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "T7b_density_strata_v11.csv")
    if os.path.exists(fb):                     # Dens<method><Low|Mid|High>Med (w05, main folds) and the tertile cuts
        t_all = pd.read_csv(fb)
        t = t_all[(t_all.win == "w05") & (t_all.fold == "main")]
        for r in t.to_dict("records"):
            lvl = r["level"].split(" ")[0].capitalize()
            k = f"Dens{camel(r['method'].split('__')[0])}{lvl}"
            m[k + "Med"] = m_or_fail(r["ade_median"])
            m[k + "LostPct"] = f"{100 * r['lost_road']:.0f}"          # tail_lost100 share (final fifth beyond 100 m)
            m[k + "Drivers"] = n(r["n_drivers"])
        for r in t_all[(t_all.win == "w15") & (t_all.fold == "main")].to_dict("records"):
            lvl = r["level"].split(" ")[0].capitalize()
            m[f"Dens{camel(r['method'].split('__')[0])}{lvl}MedFifteen"] = m_or_fail(r["ade_median"])
        m["DensNFive"] = n(t[t.method == "map_pf_steer__G3"].n_episodes.min())
        t15 = t_all[(t_all.win == "w15") & (t_all.fold == "main") & (t_all.method == "map_pf_steer__G3")]
        if len(t15):
            m["DensNFifteen"] = n(t15.n_episodes.min())
        cuts = sorted({int(x) for lv in t.level.unique() for x in __import__("re").findall(r"(\d+)/km2", lv)})
        if len(cuts) == 2:
            m["DensCutLow"], m["DensCutHigh"] = n(cuts[0]), n(cuts[1])
    cost_macros(m)
    robust_map_macros(m)
    supp_macros(m)
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
    sd = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "spy_dewri_diag.csv")
    if os.path.exists(sd):                     # scripts/icde_spy_dewri_diag.py
        t = pd.read_csv(sd, keep_default_na=False)
        bins = {"0-1": "ZeroOne", "2-3": "TwoThree", "4-6": "FourSix", "7+": "SevenPlus"}
        for r in t.to_dict("records"):
            s = camel(r["stem"].split("__")[0].replace("map_beam_steer", "beam").replace("elastic_pathing", "ep"))
            if r["kind"] == "same_output" and r["fold"] in FOLD_TAG:
                m[f"Diag{s}EpSame{WIN_TAG[r['win']]}{FOLD_TAG[r['fold']]}"] = f"{100 * float(r['value']):.0f}"
            elif r["kind"] == "route_by_turns":
                m[f"DiagTurnRoute{s}{bins[r['bin']]}"] = f"{100 * float(r['value']):.0f}"
                m[f"DiagTurnN{bins[r['bin']]}"] = n(r["n"])
            elif r["kind"].startswith("lowturn_"):
                m[f"DiagLowTurn{'Recall' if 'recall' in r['kind'] else 'Prec'}{s}"] = f"{float(r['value']):.2f}"
    coh = os.path.join(paths.MANIFESTS, "cohort_v11.csv")
    if os.path.exists(coh):
        c = pd.read_csv(coh)
        c = c[c.cohort]
        m["NCohortDrives"] = n(len(c))
        m["NCohortDrivers"] = n(c.driver.nunique())
    ac = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "area_coverage.csv")
    if os.path.exists(ac):                     # scripts/icde_area_check.py: Area<Ct|Spy><Fold>OutPct / N, DiscOut<G3|G1><Win><Fold>Pct / N
        for r in pd.read_csv(ac).to_dict("records"):
            if r["fold"] not in FOLD_TAG or not r["n"]:
                continue
            if r["check"] == "out_square":
                k = f"Area{camel(r['method'])}{FOLD_TAG[r['fold']]}"
            else:
                k = f"DiscOut{camel(r['method'])}{WIN_TAG[r['win']]}{FOLD_TAG[r['fold']]}"
            m[k + "OutPct"] = f"{100 * r['share']:.0f}"
            m[k + "OutN"] = n(r["n_out"])
            m[k + "N"] = n(r["n"])
    extra_macros(m)
    agent_macros(m)
    with open(OUT, "w") as f:
        f.write("% generated by scripts/icde_macros.py from the manifests and reports/icde/tables; do not edit\n")
        for k, v in sorted(m.items()):
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"{len(m)} macros -> {OUT}")


WIN_TAG = {"w01": "One", "w05": "Five", "w15": "Fifteen", "full": "Full"}
FOLD_TAG = {"test": "Test", "few_shot_heldout": "FewShot"}
PAIR_TAG = {"map gain, steering, heading known": "MapGain", "map gain, steering, heading unknown": "MapGainHm",
            "Dewri vs Elastic Pathing, speed only, heading unknown": "DewriVsEp",
            "Dewri vs Elastic Pathing, start heading used": "DewriVsEpHp",
            "SPy (steering only) vs map beam steering, start known": "SpyVsBeam", "SPy, known start vs 5 km area": "SpyKnownVsArea",
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
            "OPD-II calibrated vs as released": "OpdCalVsOpd", "MacroTracking wrap-fixed vs as released": "MacroFix",
            "OPD-II wrap-fixed vs as released": "OpdFix", "MacroTracking calibrated vs as released": "MacroCalVsMacro"}


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
    return ("$-$" if x < 0 and body.strip("0.{,}") else "") + body      # no sign on a value that rounds to zero


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
            m[k + "TermLost"] = f"{100 * r['terminal_lost100']:.0f}"      # ends beyond 100 m (failures count as lost)
            m[k + "TailLost"] = f"{100 * r['tail_lost100']:.0f}"
            m[k + "EndWithinFiveHundred"] = f"{100 * r['end_within_500']:.0f}"
            m[k + "EndOff"] = f"{100 * (1 - r['end_within_500']):.0f}"     # ends more than 500 m off (failures count as off)
            m[k + "Fail"] = n(r["n_fail"])
            m[k + "N"] = n(r["n_expected"])
            m[k + "Out"] = n(r["n_output"])
            m[k + "Drivers"] = n(r["n_drivers"])
            sd = r.get("ade_median_seed_sd")
            if sd is not None and np.isfinite(sd):
                m[k + "SeedSd"] = n(sd)
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
            m[k + "NOut"] = n(r["n"])
            if np.isfinite(r.get("n_drivers", np.nan)):
                m[k + "Drivers"] = n(r["n_drivers"])
            if np.isfinite(r.get("n_seeds", np.nan)):
                m[k + "Seeds"] = n(r["n_seeds"])
                m[k + "SeedsSig"] = n(r["n_seeds_sig"])
            sd = r.get("median_diff_seed_sd")
            if sd is not None and np.isfinite(sd):
                m[k + "SeedSd"] = f"{sd:.1f}" if sd < 100 else n(sd)


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
            v = r.get("n_fail_w05")
            if v is not None and np.isfinite(v):
                m[k + "FailFive"] = n(v)                    # searches that gave up on this subset (their time is counted)
            m[k + "Rss"] = f"{r['peak_rss_gb']:.1f}"
        m["CostGraph"] = f"{c.graph_build_median_s.median():.1f}"
        m["CostRssMax"] = f"{c.peak_rss_gb.max():.1f}"
        e = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_cost_episodes_valhalla.csv"))
        m["CostN"] = n(len(e))
        m["CostDrives"] = n(e.episode_id.str.split("/").str[:2].str.join("/").nunique())
        m["CostNFive"] = n(e.episode_id.str.contains("/w05/").sum())
    p = os.path.join(tabs, "cost_pedal.csv")
    if os.path.exists(p):                      # CostPedal<tag><Five|Fifteen>: the pedal-valid validation subset (CAN-Trace and its comparators)
        c = pd.read_csv(p)
        for r in c.to_dict("records"):
            k = f"CostPedal{camel(r['tag'])}"
            for win, tag in (("w05", "Five"), ("w15", "Fifteen")):
                v = r.get(f"infer_median_s_{win}")
                if v is not None and np.isfinite(v):
                    m[k + tag] = f"{v:.2f}" if v >= 0.01 else f"{v:.3f}"
                v = r.get(f"n_fail_{win}")
                if v is not None and np.isfinite(v):
                    m[k + "Fail" + tag] = n(v)
            v = r.get("infer_p90_s_w05")
            if v is not None and np.isfinite(v):
                m[k + "FiveNinety"] = f"{v:.2f}"
        e = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_cost_episodes_pedal.csv"))
        m["CostPedalN"] = n(len(e))
        m["CostPedalNFive"] = n(e.episode_id.str.contains("/w05/").sum())
        m["CostPedalDrives"] = n(e.episode_id.str.split("/").str[:2].str.join("/").nunique())
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


PROBE_TAG = {"sq2": "SqTwo", "sq5": "SqFive", "sq10": "SqTen", "sq20": "SqTwenty", "s1hz": "SOneHz", "s02hz": "SFifthHz",
             "sn2": "SnTwo", "sn5": "SnFive", "sd05": "SdHalf", "pq1": "PqOne", "pq2": "PqTwo", "pq5": "PqFive", "v1hz": "VOneHz"}
PROBE_METHOD = {"map_pf_steer__G3": "Pf", "map_beam_steer__G3": "Beam", "map_beam_none__G1": "BeamNone"}
MILD_STEER = ["sq2", "sq5", "sq10", "s1hz", "sn2", "sn5"]        # the steering levels short of 20 deg and 0.2 Hz


def robust_map_macros(m):
    """RQ5 (plan v0.9 C): RobMap<Pf|Beam|BeamNone><Level><Test|FewShot><Stat> from tables/robustness_map_w05.csv
    (run icde_probe against the clean icde_v11 rows on the same expected w05 windows). Route / RouteClean in %,
    RouteDiff / Lo / Hi in percentage points (driver-clustered CI), Med / MedClean unpaired medians in m, PairMed / PairLo /
    PairHi / PairMean paired statistics in m on the windows where both have an output, Changed / Same the share of windows
    whose output differs / is identical, Lost / Gained the shares of windows whose route status flipped."""
    p = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "robustness_map_w05.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)

    def metres(x):
        return "fail" if not np.isfinite(x) else (f"{x:.1f}" if abs(x) < 10 else n(x))

    for r in t.to_dict("records"):
        k = f"RobMap{PROBE_METHOD[r['stem']]}{PROBE_TAG[r['tag']]}{FOLD_TAG[r['fold']]}"
        m[k + "Route"], m[k + "RouteClean"] = pct(r["route_pert"], 1), pct(r["route_clean"], 1)
        for s, c in (("RouteDiff", "route_diff"), ("RouteDiffLo", "route_diff_lo"), ("RouteDiffHi", "route_diff_hi")):
            m[k + s] = signed(100 * r[c])
        m[k + "Med"], m[k + "MedClean"] = metres(r["median_pert"]), metres(r["median_clean"])
        for s, c in (("PairMed", "median"), ("PairLo", "median_lo"), ("PairHi", "median_hi"), ("PairMean", "mean")):
            m[k + s] = signed(r[c])
        m[k + "Changed"], m[k + "Same"] = pct(r["changed"], 0), pct(1.0 - r["changed"], 0)
        m[k + "Lost"], m[k + "Gained"] = pct(r["route_lost"], 1), pct(r["route_gained"], 1)
        m[k + "N"] = n(r["n_expected"])
    steer = t[t.stem != "map_beam_none__G1"]
    mild = steer[steer.tag.isin(MILD_STEER)]
    m["RobMapSteerMildMaxRouteAbs"] = f"{100 * mild.route_diff.abs().max():.1f}"
    m["RobMapSteerMildMaxPairMedAbs"] = f"{mild['median'].abs().max():.1f}"
    m["RobMapSteerMildMaxLost"] = pct(mild.route_lost.max(), 1)
    pq1 = t[t.tag == "pq1"]
    m["RobMapPqOneMaxRouteAbs"] = f"{100 * pq1.route_diff.abs().max():.1f}"
    m["RobMapPqOneMaxPairMedAbs"] = f"{pq1['median'].abs().max():.1f}"
    hard = steer[steer.tag.isin(["pq5", "s02hz"])]
    m["RobMapHardMinRoute"] = pct(hard.route_pert.min(), 1)
    m["RobMapNLevels"] = n(t.tag.nunique())
    m["RobMapNRuns"] = n(len(t.drop_duplicates(["stem", "tag"])))


def supp_macros(m):
    """Plan v0.9 D: Supp<Tag><Med|Lo|Hi|Route|RouteLo|RouteHi|Fail|N|NDrives|NDrivers> from tables/supplement_summary.csv
    (run icde_supp on the Controlled supplement's five-minute windows), plus SuppN / SuppDrives / SuppDrivers (the listed
    windows) and SuppDrivesTotal / SuppDriversTotal (every supplement drive)."""
    p = os.path.join(paths.GPS_DIR, "reports", "icde", "tables", "supplement_summary.csv")
    if not os.path.exists(p):
        return
    t = pd.read_csv(p)
    for r in t.to_dict("records"):
        k = f"Supp{r['tag']}"
        m[k + "Med"], m[k + "Lo"], m[k + "Hi"] = m_or_fail(r["median"]), m_or_fail(r["median_lo"]), m_or_fail(r["median_hi"])
        m[k + "Route"], m[k + "RouteLo"], m[k + "RouteHi"] = pct(r["route"]), pct(r["route_lo"]), pct(r["route_hi"])
        m[k + "Fail"], m[k + "N"], m[k + "NDrives"], m[k + "NDrivers"] = n(r["fail"]), n(r["n"]), n(r["n_drives"]), n(r["n_drivers"])
    eps = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_supplement_episodes.csv"))
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "driver", "drive", "fold", "admission"])
    d = eps[eps.map_ok].merge(man, on="episode_id")
    m["SuppN"], m["SuppDrives"], m["SuppDrivers"] = n(len(d)), n(d.drop_duplicates(["driver", "drive"]).shape[0]), n(d.driver.nunique())
    s = man[(man.fold == "supplement") & (man.admission == "supplement")].drop_duplicates(["driver", "drive"])
    m["SuppDrivesTotal"], m["SuppDriversTotal"] = n(len(s)), n(s.driver.nunique())


def pct(x, d=0):
    return "--" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


SET_TAG = {"A": "A", "B": "B", "B'": "Bm", "C": "C", "D": "D", "P": "P"}
TAU_TAG = {0.3: "Thirty", 0.4: "Forty", 0.5: "Fifty", 0.6: "Sixty", 0.7: "Seventy", 0.8: "Eighty", 0.9: "Ninety", 1.0: "Hundred"}
BIN_TAG = {"0-1": "ZeroOne", "2-3": "TwoThree", "4-6": "FourSix", "7+": "SevenPlus", "all": "All"}


def extra_macros(m):
    """Plan v0.9 A and B (scripts/icde_extra.py): Comp<Set><Fold><stat>, CompMem<Set><Fold><Member><stat>,
    CFail<Turns|Dens|All><Bin><stat>, DrvRoute<Method><Fold><stat>, Conf<Row><Hp|Hm><Win><Fold><stat>,
    ConfThr<Row><Hp|Hm><Win><Fold><Tau><stat>, ConfProbOne<Method>Pct, Tax<Row><Hp|Hm><Fold><Fifty|Hundred><stat>."""
    tabs = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")
    p = os.path.join(tabs, "complementarity_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            s = SET_TAG.get(r["set"].split(" ")[0])
            if s is None or r["fold"] not in FOLD_TAG:
                continue
            k = f"Comp{s}{FOLD_TAG[r['fold']]}"
            m[k + "N"], m[k + "NDrivers"] = n(r["n"]), n(r["n_drivers"])
            m[k + "OracleMed"] = m_or_fail(r["oracle_ade_median"])
            m[k + "OracleRoutePct"] = pct(r["oracle_route"])
            m[k + "AllFailPct"], m[k + "AllFailLo"], m[k + "AllFailHi"] = pct(r["all_fail_share"], 1), pct(r["all_fail_lo"], 1), pct(r["all_fail_hi"], 1)
            m[k + "NAllFail"] = n(r["n_all_fail"])
            m[k + "BestSingle"] = str(r["best_single"]).replace("_", " ")
            m[k + "BestSingleMed"] = m_or_fail(r["best_single_median"])
            if isinstance(r.get("reference"), str) and r["reference"]:
                m[k + "RefMed"] = m_or_fail(r["ref_ade_median"])
                m[k + "RefRoutePct"] = pct(r["ref_route"])
                m[k + "RouteGainPct"], m[k + "RouteGainLo"], m[k + "RouteGainHi"] = pct(r["route_gain"]), pct(r["route_gain_lo"]), pct(r["route_gain_hi"])
                m[k + "ComplPct"], m[k + "ComplLo"], m[k + "ComplHi"] = pct(r["complementary_share"]), pct(r["complementary_lo"]), pct(r["complementary_hi"])
                for stat, col in (("AdeGainMed", "ade_gain_median"), ("AdeGainLo", "ade_gain_lo"), ("AdeGainHi", "ade_gain_hi"),
                                  ("AdeGainMean", "ade_gain_mean"), ("AdeGainMeanLo", "ade_gain_mean_lo"), ("AdeGainMeanHi", "ade_gain_mean_hi")):
                    if np.isfinite(r.get(col, np.nan)):
                        m[k + stat] = signed(r[col])
                m[k + "NBothFinite"] = n(r["n_both_finite"])
    p = os.path.join(tabs, "complementarity_members_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            s = SET_TAG.get(r["set"].split(" ")[0])
            if s is None or r["fold"] not in FOLD_TAG:
                continue
            k = f"CompMem{s}{FOLD_TAG[r['fold']]}{camel(r['member'])}"
            m[k + "Med"] = m_or_fail(r["median_ade"])
            m[k + "RoutePct"] = pct(r["route"])
            m[k + "AttainPct"] = pct(r["share_attains_oracle"])
            m[k + "Fail"] = n(r["n_fail"])
    p = os.path.join(tabs, "common_failure_strata_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            kind = {"reference turns": "Turns", "density": "Dens", "all": "All"}[r["stratum"]]
            b = BIN_TAG.get(r["level"], r["level"].split(" ")[0].capitalize())
            k = f"CFail{kind}{b}"
            m[k + "Pct"], m[k + "Lo"], m[k + "Hi"] = pct(r["all_fail_share"], 1), pct(r["all_fail_lo"], 1), pct(r["all_fail_hi"], 1)
            m[k + "N"], m[k + "NDrivers"] = n(r["n"]), n(r["n_drivers"])
            m[k + "OracleRoutePct"], m[k + "RefRoutePct"] = pct(r["oracle_route"]), pct(r["ref_route"])
    p = os.path.join(tabs, "driver_route_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            if r["fold"] not in FOLD_TAG:
                continue
            k = f"DrvRoute{camel(r['method'])}{FOLD_TAG[r['fold']]}"
            for stat, col in (("QTen", "q10"), ("QTwentyFive", "q25"), ("Med", "q50"), ("QSeventyFive", "q75"), ("QNinety", "q90"),
                              ("BelowHalfPct", "share_drivers_below_half"), ("ZeroPct", "share_drivers_zero"), ("PooledPct", "pooled_route")):
                m[k + stat] = pct(r[col])
            m[k + "NDrivers"], m[k + "NDriversGeFive"], m[k + "NDriversLtFive"] = n(r["n_drivers"]), n(r["n_drivers_ge5"]), n(r["n_drivers_lt5"])
    p = os.path.join(tabs, "confidence_w05_w15.csv")
    if os.path.exists(p):
        row_tag = {"map_beam_steer__G3": "Beam", "map_beam_none__G1": "BeamNone", "dewri_hp__G1": "DewriHp", "dewri__G1": "Dewri",
                   "elastic_pathing_hp__G1": "EpHp", "elastic_pathing__G1": "Ep"}
        for r in pd.read_csv(p).to_dict("records"):
            rt = row_tag.get(r["stem"])
            if rt is None:
                continue
            if r["kind"] == "prob_only":
                m[f"ConfProbOne{rt}Pct"] = pct(r["share_prob_eq_1"])
                continue
            if r["fold"] not in FOLD_TAG or r["win"] not in WIN_TAG:
                continue
            k = f"Conf{rt}{'Hm' if r['protocol'].endswith('Hm') else 'Hp'}{WIN_TAG[r['win']]}{FOLD_TAG[r['fold']]}"
            if r["kind"] == "summary":
                m[k + "Auroc"], m[k + "AurocLo"], m[k + "AurocHi"] = f"{r['auroc']:.2f}", f"{r['auroc_lo']:.2f}", f"{r['auroc_hi']:.2f}"
                m[k + "Rho"] = f"{r['spearman_ade']:.2f}".replace("-", "$-$")
                m[k + "ProbOnePct"], m[k + "ProbMed"], m[k + "ProbQTwentyFive"] = pct(r["share_prob_eq_1"]), f"{r['prob_median']:.2f}", f"{r['prob_q25']:.2f}"
                m[k + "NOut"], m[k + "NFail"], m[k + "N"] = n(r["n_output"]), n(r["n_fail"]), n(r["n_expected"])
            elif r["kind"] == "threshold" and r["tau"] in TAU_TAG:
                kk = k + TAU_TAG[r["tau"]]
                m[kk + "CovPct"], m[kk + "RoutePct"] = pct(r["coverage"]), pct(r["route_rate_accepted"])
                m[kk + "RouteLo"], m[kk + "RouteHi"] = pct(r["route_rate_lo"]), pct(r["route_rate_hi"])
                m[kk + "ErrPct"], m[kk + "NAcc"] = pct(r["error_share_accepted"]), n(r["n_accepted"])
                m[kk + "Med"] = m_or_fail(r["median_ade_accepted"])
    p = os.path.join(tabs, "window_taxonomy_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            if r["fold"] not in FOLD_TAG:
                continue
            thr = {50.0: "Fifty", 100.0: "Hundred"}[float(r["threshold_m"])]
            k = f"Tax{camel(r['stem'].split('__')[0])}{'Hm' if r['protocol'].endswith('Hm') else 'Hp'}{FOLD_TAG[r['fold']]}{thr}"
            m[k + "NoOutPct"], m[k + "NoCandPct"] = pct(r["no_output"]), pct(r["no_candidate"])
            m[k + "RankMissPct"], m[k + "TopOneOkPct"], m[k + "N"] = pct(r["ranking_miss"]), pct(r["top1_ok"]), n(r["n"])


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
            if len(d) == 1:
                m[k + "Lo"], m[k + "Hi"] = signed(d.median_lo.iloc[0]), signed(d.median_hi.iloc[0])
        m["SigNDrivers"] = n(t.n_drivers.iloc[0])
    p = os.path.join(tabs, "signal_reference_w05.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).to_dict("records"):
            m[f"SigRef{camel(r['key'])}Med"] = m_or_fail(r["ade_median"])
            m[f"SigRef{camel(r['key'])}Fail"] = n(r["fail"])




AGENT_ROW_TAG = {"beam": "Beam", "selector": "Sel", "agent": "Llm", "agent_fb": "LlmFb", "oracle": "Oracle", "agent_s0": "SeedZero", "agent_s1": "SeedOne", "agent_s2": "SeedTwo"}
AGENT_FOLD_TAG = {"test": "Test", "few_shot_heldout": "FewShot", "all": "All", "val": "Val"}


def agent_macros(m):
    """Agent study (supplementary cohort): tables agent_w05.csv, agent_pairs_w05.csv, agent_cohort_counts.csv, agent_selector_val.csv,
    agent_identity_check.csv, agent_live_check.csv -> Agent* macros."""
    tdir = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")

    def f1(x):
        return "--" if x is None or not np.isfinite(x) else f"{x:.1f}"

    p = os.path.join(tdir, "agent_cohort_counts.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).itertuples():
            k = "AgentCohort" + AGENT_FOLD_TAG.get(r.fold, camel(r.fold))
            m[k + "N"], m[k + "Drives"], m[k + "Drivers"] = n(r.n_windows), n(r.n_drives), n(r.n_drivers)
    p = os.path.join(tdir, "agent_cohort_counts_val.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).set_index("fold").loc["val"]
        m["AgentCohortValN"], m["AgentCohortValDrivers"] = n(r.n_windows), n(r.n_drivers)
    for sfx, pre in (("", "Agent"), ("_max", "AgentMax")):           # frozen study -> Agent*, ceiling exploration (C-max) -> AgentMax*
        _agent_result_macros(m, tdir, sfx, pre, f1)
    p = os.path.join(tdir, "agent_selector_val.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).itertuples():
            k = "AgentSelVal" + {"selector_cv": "Sel", "beam1": "Beam", "pf1": "Pf", "ep1": "Ep", "oracle": "Oracle"}.get(r.row, camel(r.row))
            m[k + "Route"], m[k + "RouteLo"], m[k + "RouteHi"], m[k + "N"] = f"{r.route_pct:.0f}", f"{r.route_lo:.0f}", f"{r.route_hi:.0f}", n(r.n_windows)
    p = os.path.join(tdir, "agent_selector_max_val.csv")
    if os.path.exists(p):
        for r in pd.read_csv(p).itertuples():
            k = "AgentMaxSelVal" + {"selector_cv": "Sel", "beam1": "Beam", "pf1": "Pf", "ep1": "Ep", "oracle": "Oracle"}.get(r.row, camel(r.row))
            m[k + "Route"], m[k + "RouteLo"], m[k + "RouteHi"], m[k + "N"] = f"{r.route_pct:.0f}", f"{r.route_lo:.0f}", f"{r.route_hi:.0f}", n(r.n_windows)
    p = os.path.join(tdir, "agent_identity_check.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).iloc[0]
        m["AgentIdentityN"], m["AgentIdentityMaxDiff"] = n(r.n_common), f"{r.max_abs_ade_diff:.1e}"
    p = os.path.join(tdir, "agent_live_check.csv")
    if os.path.exists(p):
        r = pd.read_csv(p).iloc[0]
        for col, suf in (("n", "N"), ("ratio_median", "Ratio"), ("live_med_s", "LiveMed"), ("reconstructed_med_s", "ReconMed")):
            if col in r and np.isfinite(r[col]):
                m["AgentLive" + suf] = n(r[col]) if suf == "N" else f1(r[col])


AGENT_MAX_ROW_TAG = {"agent_prev": "Prev", "agent_qwen3max_vote": "Vote", "agent_qwen3maxcritic_s0": "CriticSeedZero", "agent_qwen3maxcritic_s1": "CriticSeedOne",
                     "agent_qwen3maxcritic_s2": "CriticSeedTwo", "agent_qwen3maxcritic_vote": "CriticVote"}
AGENT_PAIR_TAG = {"agent_vs_selector": "LlmVsSel", "agent_vs_beam": "LlmVsBeam", "agent_vs_prev": "LlmVsPrev"}


def _agent_pair_tag(pair):
    if pair in AGENT_PAIR_TAG:
        return AGENT_PAIR_TAG[pair]
    for row, tag in AGENT_MAX_ROW_TAG.items():                       # e.g. agent_qwen3max_vote_vs_selector -> VoteVsSel
        for other, otag in (("_vs_selector", "VsSel"), ("_vs_beam", "VsBeam")):
            if pair == row + other:
                return tag + otag
    return camel(pair)


def _agent_result_macros(m, tdir, sfx, pre, f1):
    """agent{sfx}_w05.csv and agent{sfx}_pairs_w05.csv -> {pre}* macros (rows x folds, pairs x folds, 'all' = both folds pooled)."""
    p = os.path.join(tdir, f"agent{sfx}_w05.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for r in t.itertuples():
            tag = AGENT_ROW_TAG.get(r.row) or AGENT_MAX_ROW_TAG.get(r.row) or camel(r.row)
            k = f"{pre}{tag}{AGENT_FOLD_TAG.get(r.fold, camel(r.fold))}"
            m[k + "Route"], m[k + "RouteLo"], m[k + "RouteHi"] = f"{r.route_pct:.0f}", f"{r.route_lo:.0f}", f"{r.route_hi:.0f}"
            m[k + "Med"], m[k + "MedLo"], m[k + "MedHi"] = m_or_fail(r.ade_median), m_or_fail(r.ade_lo), m_or_fail(r.ade_hi)
            m[k + "NoOut"], m[k + "FdeFive"], m[k + "N"], m[k + "Drivers"] = f1(r.noout_pct), f1(r.fde500_pct), n(r.n), n(r.n_drivers)
            for col, suf in (("cost_reconstructed_med", "Cost"), ("elapsed_cached_med", "Elapsed"), ("llm_wall_med", "LlmTime"), ("tool_calls_mean", "Calls"),
                             ("decisions_mean", "Decisions"), ("truncation_pct", "Trunc"), ("fallback_pct", "Fallback"), ("tool_use_beam_pct", "UseBeam"),
                             ("tool_use_pf_pct", "UsePf"), ("tool_use_ep_pct", "UseEp"), ("route_seed_min", "RouteSeedMin"), ("route_seed_max", "RouteSeedMax"),
                             ("same_sel_pct", "SameSel"), ("same_beam1_pct", "SameBeam")):
                if col in t.columns and np.isfinite(getattr(r, col, np.nan)):
                    v = getattr(r, col)
                    m[k + suf] = f"{v:.1f}" if suf.startswith("RouteSeed") else f1(v)
            for col, suf in (("gen_tokens_mean", "Tokens"), ("reasoning_tokens_mean", "ReasonTokens")):
                if col in t.columns and np.isfinite(getattr(r, col, np.nan)):
                    m[k + suf] = n(getattr(r, col))
            for col in [c for c in t.columns if c.startswith("status_") and c.endswith("_pct")]:
                v = getattr(r, col, np.nan)
                if np.isfinite(v):
                    m[k + "Status" + camel(col[len("status_"):-len("_pct")])] = f1(v)
    p = os.path.join(tdir, f"agent{sfx}_pairs_w05.csv")
    if os.path.exists(p):
        t = pd.read_csv(p)
        for r in t.itertuples():
            k = f"{pre}Pair" + _agent_pair_tag(r.pair) + AGENT_FOLD_TAG.get(r.fold, camel(r.fold))
            m[k + "RouteDiff"], m[k + "RouteLo"], m[k + "RouteHi"] = signed(r.route_diff_pp), signed(r.route_lo_pp), signed(r.route_hi_pp)
            for col, suf in (("ade_med_diff", "MedDiff"), ("ade_lo", "MedLo"), ("ade_hi", "MedHi")):
                v = getattr(r, col, np.nan)
                m[k + suf] = signed(v) if np.isfinite(v) else "--"
            for s, st in ((0, "Zero"), (1, "One"), (2, "Two")):
                col = f"route_diff_s{s}_pp"
                if col in t.columns and np.isfinite(getattr(r, col, np.nan)):
                    m[k + "RouteDiffSeed" + st] = signed(getattr(r, col))
                col = f"n_out_s{s}"
                if col in t.columns and np.isfinite(getattr(r, col, np.nan)):
                    m[k + "NOutSeed" + st] = n(getattr(r, col))
            m[k + "N"] = n(r.n)


if __name__ == "__main__":
    main()
