"""Failure analysis and research strata for benchmark v1 (never main-board numbers, never vehicle-label conclusions).

 (a) lost-road onset timing vs the last turn of the reference path (from scripts/paper_curves.py failure_events.parquet);
 (b) road-network density around the episode start (map nodes within 2 km of p0, tertiles of the main-fold 5-minute
     episodes) vs map-filter / beam / DR errors;
 (c) vehicle platform seen vs unseen in the training fold, as a stratum only.

Outputs: reports/failure_analysis_v1.md, reports/paper/tables/T7a_lost_road_timing, T7b_density_strata,
T7c_platform_strata (.csv + .md) and the local manifests/episode_density_v1.parquet (one number per episode, no
coordinates). Episode start coordinates are read only to count map nodes and are never written.

  python scripts/failure_analysis.py [--run full_v1] [--n-boot 300]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cantraj import metrics as M, paths  # noqa: E402

GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TILES = os.path.join(paths.BENCH, "maps", "tiles")
TILE_DEG = 0.25
RADIUS_M = 2000.0
MAIN = ("test", "few_shot_heldout")
METHODS_B = [("O-M-Hp", "map_pf_steer__G3"), ("F-M-Hp", "map_beam_steer__G3"), ("O-M-Hp", "map_pf_none__G1"),
             ("O-N-Hp", "dr_steer__G3"), ("O-N-Hp", "dr_curvature__G5")]
METHODS_C = [("O-N-Hp", "dr_steer__G3"), ("O-N-Hp", "dr_steer_calib__G3"), ("O-N-Hp", "learned_gru_s0__G3"),
             ("O-M-Hp", "map_pf_steer__G3"), ("F-M-Hp", "map_beam_steer__G3")]


class TileTrees:
    """KD-trees of map nodes per 0.25-degree tile, in a tile-local equirectangular frame (metres); LRU cached."""

    def __init__(self, cap=96):
        self.cache, self.cap = OrderedDict(), cap

    @staticmethod
    def name(lat, lon):
        return f"t{math.floor(lat / TILE_DEG)}_{math.floor(lon / TILE_DEG)}"

    def get(self, name):
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        f = os.path.join(TILES, name, "nodes.parquet")
        item = None
        if os.path.exists(f):
            nd = pd.read_parquet(f, columns=["lat", "lon"])
            if len(nd):
                from scipy.spatial import cKDTree
                i, j = (int(x) for x in name[1:].split("_"))
                lat_c, lon_c = (i + 0.5) * TILE_DEG, (j + 0.5) * TILE_DEG
                kx, ky = 111320.0 * math.cos(math.radians(lat_c)), 110540.0
                item = (cKDTree(np.column_stack([(nd.lon.to_numpy() - lon_c) * kx, (nd.lat.to_numpy() - lat_c) * ky])), lat_c, lon_c, kx, ky)
        self.cache[name] = item
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
        return item

    def count(self, lat, lon, r=RADIUS_M):
        dlat = r / 110540.0
        dlon = r / (111320.0 * math.cos(math.radians(lat)))
        names = {self.name(la, lo) for la in (lat - dlat, lat, lat + dlat) for lo in (lon - dlon, lon, lon + dlon)}
        n, found = 0, False
        for nm in names:
            item = self.get(nm)
            if item is None:
                continue
            found = True
            tree, lat_c, lon_c, kx, ky = item
            n += int(tree.query_ball_point([(lon - lon_c) * kx, (lat - lat_c) * ky], r, return_length=True))
        return n if found else -1


def density_table(man):
    out_p = os.path.join(paths.MANIFESTS, "episode_density_v1.parquet")
    need = man[man.fold.isin(MAIN + ("val",)) & man.win.isin(["w05", "w15", "full"])]
    if os.path.exists(out_p):
        d = pd.read_parquet(out_p)
        if set(need.episode_id) <= set(d.episode_id):
            return d
    p0 = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_p0.parquet")).merge(need[["episode_id"]], on="episode_id")
    p0 = p0.assign(tile=[TileTrees.name(a, b) for a, b in zip(p0.lat0, p0.lon0)]).sort_values("tile")
    tt = TileTrees()
    counts = np.array([tt.count(a, b) for a, b in zip(p0.lat0, p0.lon0)])
    d = pd.DataFrame({"episode_id": p0.episode_id.to_numpy(), "map_nodes_2km": counts,
                      "nodes_per_km2": np.where(counts >= 0, counts / (math.pi * (RADIUS_M / 1000.0) ** 2), np.nan)})
    d.to_parquet(out_p, index=False)
    print(f"density: {len(d)} episodes, {int((counts < 0).sum())} without any tile, "
          f"median {np.nanmedian(d.nodes_per_km2):.0f} nodes/km2 -> {out_p}")
    return d


def ci_med(v, cl, n_boot):
    m, (lo, hi) = M.cluster_bootstrap(np.asarray(v, float), np.asarray(cl), np.median, n_boot)
    return float(m), float(lo), float(hi)


def stratum_rows(ep, methods, strat_col, wins, n_boot, min_n=30, levels=None):
    rows = []
    folds = [("main", list(MAIN)), ("val", ["val"])]
    for fold_name, fl in folds:
        e = ep[ep.fold.isin(fl)]
        for proto, method in methods:
            for win in wins:
                g0 = e[(e.protocol == proto) & (e.method == method) & (e.win == win)]
                for lvl in (levels or sorted(g0[strat_col].dropna().unique())):
                    g = g0[g0[strat_col] == lvl]
                    if len(g) < min_n:
                        continue
                    med, lo, hi = ci_med(g.ade.values, g.driver.values, n_boot)
                    rows.append(dict(stratum=strat_col, level=lvl, fold=fold_name, protocol=proto, method=method, win=win,
                                     n_episodes=len(g), n_drivers=int(g.driver.nunique()), n_platforms=int(g.platform.nunique()),
                                     ade_median=med, ade_median_lo=lo, ade_median_hi=hi, ade_mean=float(g.ade.mean()),
                                     c100=float(g.c100.mean()), s50_90=float(g.s50_90.mean()), lost_road=float(g.lost_road.mean())))
    return pd.DataFrame(rows)


def md(df, cols, digits=1):
    if df.empty:
        return "_(no rows with n >= 30)_\n"
    d = df[cols].copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else (f"{x:.{digits}f}" if abs(x) < 1e4 else f"{x:,.0f}"))
    lines = ["| " + " | ".join(d.columns) + " |", "|" + "---|" * len(d.columns)]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v1")
    ap.add_argument("--n-boot", type=int, default=300)
    args = ap.parse_args()
    tdir = os.path.join(GPS_DIR, "reports", "paper", "tables")
    os.makedirs(tdir, exist_ok=True)

    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"),
                          columns=["episode_id", "fold", "platform", "win", "eligible_main", "driver", "drive"])
    man = man[man.eligible_main]
    train_plat = set(man.platform[man.fold == "train"].dropna())
    cols = ["episode_id", "protocol", "method", "win", "eligible_main", "ade", "c100", "s50_90", "lost_road", "driver", "parent"]
    ep = pd.read_parquet(os.path.join(paths.SCORES, args.run, "episodes_all.parquet"), columns=cols)
    ep = ep[ep.eligible_main].merge(man[["episode_id", "fold", "platform"]], on="episode_id")
    ep["platform_seen"] = np.where(ep.platform.isin(train_plat), "seen in train", "unseen")

    # ---- (b) density tertiles (cut points from the main-fold 5-minute episodes, one value per episode)
    dens = density_table(man)
    ep = ep.merge(dens[["episode_id", "nodes_per_km2"]], on="episode_id", how="left")
    base = ep[(ep.fold.isin(MAIN)) & (ep.win == "w05") & (ep.method == "map_pf_steer__G3") & (ep.protocol == "O-M-Hp")].nodes_per_km2.dropna()
    q1, q2 = float(base.quantile(1 / 3)), float(base.quantile(2 / 3))
    labels = [f"low (< {q1:.0f}/km2)", f"mid ({q1:.0f} to {q2:.0f}/km2)", f"high (> {q2:.0f}/km2)"]
    ep["density"] = pd.cut(ep.nodes_per_km2, [-np.inf, q1, q2, np.inf], labels=labels).astype(object)
    tb = stratum_rows(ep, METHODS_B, "density", ["w05", "w15"], args.n_boot, levels=labels)
    tb.to_csv(os.path.join(tdir, "T7b_density_strata.csv"), index=False)

    # ---- (c) platform seen / unseen
    tc = stratum_rows(ep, METHODS_C, "platform_seen", ["w05"], args.n_boot, levels=["seen in train", "unseen"])
    tc.to_csv(os.path.join(tdir, "T7c_platform_strata.csv"), index=False)

    # ---- (a) lost-road timing
    fe_p = os.path.join(GPS_DIR, "reports", "paper", "figures", "data", "failure_events.parquet")
    ta = pd.DataFrame()
    if os.path.exists(fe_p):
        fe = pd.read_parquet(fe_p)
        rows = []
        for (proto, method, win), g in fe.groupby(["protocol", "method", "win"]):
            lost = g[g.lost]
            chance = float(np.mean(np.minimum(1.0, lost.n_turns * 30.0 / lost.dur_s))) if len(lost) else np.nan
            rows.append(dict(protocol=proto, method=method, win=win, n_episodes=len(g), lost_rate=float(g.lost.mean()),
                             onset_q25_s=float(lost.t_lost_rel_s.quantile(0.25)) if len(lost) else np.nan,
                             onset_median_s=float(lost.t_lost_rel_s.median()) if len(lost) else np.nan,
                             onset_q75_s=float(lost.t_lost_rel_s.quantile(0.75)) if len(lost) else np.nan,
                             onset_first_60s=float((lost.t_lost_rel_s <= 60).mean()) if len(lost) else np.nan,
                             since_last_turn_median_s=float(lost.dt_since_last_turn_s.median()) if len(lost) else np.nan,
                             within_30s_after_turn=float(lost.turn_within_30s_before.astype(float).mean()) if len(lost) else np.nan,
                             within_30s_chance_approx=chance,
                             turns_median_lost=float(lost.n_turns.median()) if len(lost) else np.nan,
                             turns_median_kept=float(g[~g.lost].n_turns.median()) if (~g.lost).any() else np.nan))
        ta = pd.DataFrame(rows)
        ta.to_csv(os.path.join(tdir, "T7a_lost_road_timing.csv"), index=False)

    # ---- markdown
    out = ["# Failure analysis and research strata (benchmark v1, run %s)\n" % args.run,
           "Research tables. The main board is unchanged; strata are reported as strata only (n >= 30 per cell; medians with "
           "driver-cluster bootstrap 95% CIs). `main` = test + few_shot_heldout, start and heading known.\n",
           "## (a) When does a map-constrained estimate lose the road?\n",
           "Onset = start of the terminal excursion beyond 100 m (NaN predictions count as beyond). A turn = a change of the "
           "reference heading by more than 45 degrees within 10 s. `within_30s_after_turn` = share of lost episodes whose onset "
           "lies within 30 s after a turn; `within_30s_chance_approx` = the share expected if onsets were spread uniformly over "
           "the episode (n_turns x 30 s / duration, capped at 1).\n",
           md(ta, ["protocol", "method", "win", "n_episodes", "lost_rate", "onset_q25_s", "onset_median_s", "onset_q75_s", "onset_first_60s",
                   "since_last_turn_median_s", "within_30s_after_turn", "within_30s_chance_approx", "turns_median_lost", "turns_median_kept"], 2),
           "\n## (b) Road-network density around the start\n",
           f"Density = map nodes within 2 km of the episode start per km2 (tertile cuts {q1:.0f} and {q2:.0f} nodes/km2 from the "
           "main-fold 5-minute episodes). Denser networks mean more branching decisions for the map methods.\n",
           md(tb, ["fold", "protocol", "method", "win", "level", "n_episodes", "n_drivers", "ade_median", "ade_median_lo", "ade_median_hi", "c100", "s50_90", "lost_road"], 2),
           "\n## (c) Vehicle platform seen vs unseen in the training fold (stratum only)\n",
           "Platform labels are taken as recorded in the logs (no corrections); this table describes where the episodes fall, not "
           "vehicle effects. Confounded with driver, region and receiver.\n",
           md(tc, ["fold", "protocol", "method", "win", "level", "n_episodes", "n_drivers", "n_platforms", "ade_median", "ade_median_lo", "ade_median_hi", "c100", "s50_90"], 2)]
    for name, df_, cols_ in (("T7a_lost_road_timing", ta, None), ("T7b_density_strata", tb, None), ("T7c_platform_strata", tc, None)):
        if not df_.empty:
            with open(os.path.join(tdir, name + ".md"), "w") as f:
                f.write(md(df_, list(df_.columns), 2))
    with open(os.path.join(GPS_DIR, "reports", "failure_analysis_v1.md"), "w") as f:
        f.write("\n".join(out))
    print("wrote reports/failure_analysis_v1.md and T7a/T7b/T7c; rows:", len(ta), len(tb), len(tc))


if __name__ == "__main__":
    main()
