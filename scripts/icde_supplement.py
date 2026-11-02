#!/usr/bin/env python3
"""Plan v0.9 item D: prepare the Controlled supplement check (run icde_supp).

The supplement fold (admission == "supplement") never entered the CAN-only board or any selection: every row has
eligible_main == False, the learning cache asserts fold in (train, val) (src/cantraj/learn/data.py), the calibration
and validation-selection scripts restrict themselves to the train / val folds, and plan_episode_tiles.py /
build_map_episodes.py filter on eligible_main. This script therefore builds the supplement's own inputs WITHOUT touching
the published manifests:
  manifests/episode_p0_supp.parquet         start fixes of the window-eligible supplement episodes (w05, w15, full)
  manifests/episode_map_v11_supp.parquet    map eligibility (same rule as build_map_episodes.py: road within 400 m of
                                            p0 in the v2 tiles, every tile of the disc present)
  manifests/icde_supplement_episodes.csv    the five-minute windows the check runs on (map_ok ones; the others listed)
Only counts are printed (no coordinates, no place names).

    python scripts/icde_supplement.py [--workers 4]
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths, registry  # noqa: E402
from cantraj.maps.osm import tiles_in_radius  # noqa: E402
from cantraj.maps import valhalla  # noqa: E402
import build_map_episodes as bme  # noqa: E402
import plan_episode_tiles as pet  # noqa: E402

P0_SUPP = os.path.join(paths.MANIFESTS, "episode_p0_supp.parquet")
MAP_SUPP = os.path.join(paths.MANIFESTS, "episode_map_v11_supp.parquet")
EPS_SUPP = os.path.join(paths.MANIFESTS, "icde_supplement_episodes.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wins", nargs="+", default=["w05", "w15", "full"])
    args = ap.parse_args()
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    sup = man[(man.fold == "supplement") & (man.admission == "supplement") & man.win.isin(args.wins)].copy()
    assert not sup.eligible_main.any(), "supplement episodes must be outside the CAN-only board"
    print(f"supplement episodes: {len(sup)} ({sup.groupby('win').size().to_dict()}), "
          f"{sup.groupby(['driver', 'drive']).ngroups} drives, {sup.driver.nunique()} drivers", flush=True)
    # 1. start fixes (same helper as the published p0 table; written to a separate file)
    if os.path.exists(P0_SUPP) and set(sup.episode_id) <= set(pd.read_parquet(P0_SUPP).episode_id):
        p0 = pd.read_parquet(P0_SUPP)
    else:
        tasks = [(d, v, list(zip(g.episode_id, g.t0_s, g.t_end_s))) for (d, v), g in sup.groupby(["driver", "drive"])]
        rows = []
        with Pool(args.workers) as pool:
            for out in pool.imap_unordered(pet._p0_for_drive, tasks, chunksize=2):
                rows += out
        p0 = pd.DataFrame(rows)
        p0.to_parquet(P0_SUPP, index=False)
    p0 = p0[p0.episode_id.isin(sup.episode_id)].merge(sup[["episode_id", "fold", "win"]], on="episode_id")
    print(f"p0 rows: {len(p0)}, max |t_fix - t0| {p0.dt_fix_s.max():.2f} s", flush=True)
    # 2. map eligibility with the v2 tiles (rule of build_map_episodes.py; nothing is cut or downloaded here)
    spec = registry.protocol_spec("O-M-Hp")
    cache = bme.TileSegs()
    have, rows = {}, []
    for r in p0.itertuples(index=False):
        need = tiles_in_radius(r.lat0, r.lon0, min(registry.map_radius_m(spec, r.dur_s), bme.MAP_CAP_M))
        miss = 0
        for t in need:
            if t not in have:
                have[t] = os.path.exists(os.path.join(bme.TILES_V2, f"t{t[0]}_{t[1]}", "ways.parquet"))
            miss += not have[t]
        dist = bme.start_distance(cache, r.lat0, r.lon0)
        rows.append(dict(episode_id=r.episode_id, fold=r.fold, win=r.win, start_edge_dist_m=dist, map_start_ok=bool(dist <= bme.START_R),
                         n_tiles_needed=len(need), n_tiles_missing=int(miss), valhalla_cluster=valhalla.cluster_of(r.lat0, r.lon0) is not None))
    out = pd.DataFrame(rows)
    out["map_ok"] = out.map_start_ok & (out.n_tiles_missing == 0)
    out.drop(columns=["valhalla_cluster"]).to_parquet(MAP_SUPP, index=False)
    s = out.groupby("win").agg(n=("episode_id", "size"), map_ok=("map_ok", "sum"), start_missing=("map_start_ok", lambda x: int((~x).sum())),
                               tiles_missing=("n_tiles_missing", lambda x: int((x > 0).sum())), dist_p50=("start_edge_dist_m", "median"),
                               valhalla=("valhalla_cluster", "sum"))
    print(s.to_string(), flush=True)
    # 3. the five-minute windows of the check
    w05 = out[(out.win == "w05")].merge(sup[["episode_id", "driver", "drive"]], on="episode_id")
    w05[["episode_id", "map_ok"]].to_csv(EPS_SUPP, index=False)
    ok = w05[w05.map_ok]
    print(f"w05 windows: {len(w05)} listed, {len(ok)} map_ok, {ok.drop_duplicates(['driver', 'drive']).shape[0]} drives, "
          f"{ok.driver.nunique()} drivers; OPD-II Valhalla cluster present for {int(w05.valhalla_cluster.sum())} of {len(w05)}", flush=True)
    print("written", P0_SUPP, MAP_SUPP, EPS_SUPP)


if __name__ == "__main__":
    main()
