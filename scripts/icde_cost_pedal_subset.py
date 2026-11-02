#!/usr/bin/env python3
"""Fixed validation subset for timing the pedal methods (RQ5).

CAN-Trace needs a continuous accelerator pedal, and no drive of the 92-episode cost subset
(manifests/icde_cost_episodes_valhalla.csv) has one. This subset is built by a rule fixed before any timing run and
never from results: pedal-valid validation drives are taken round-robin over their drivers (drivers and drives in id
order) until N_W05 drives are chosen; the subset is the first eligible, map-ok 5-minute window of each of them and the
first eligible, map-ok 15-minute window of the first N_W15 of them that have one. Every method timed against CAN-Trace
runs on the same windows (scripts/run_icde_cost_pedal.sh).

    python scripts/icde_cost_pedal_subset.py   -> manifests/icde_cost_episodes_pedal.csv (episode_id)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402

N_W05, N_W15 = 25, 6


def main():
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"),
                          columns=["episode_id", "driver", "drive", "win", "k", "fold", "eligible_main"])
    pv = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_pedal_val_drives.csv"))
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"), columns=["episode_id", "map_ok"])
    m = man[man.eligible_main & (man.fold == "val") & man.drive.isin(set(pv.drive))].merge(me, on="episode_id", how="left")
    m = m[m.map_ok.fillna(False).astype(bool)].sort_values(["driver", "drive", "win", "k"])
    w05 = m[m.win == "w05"].drop_duplicates(["driver", "drive"], keep="first")
    w05 = w05.assign(rank=w05.groupby("driver").cumcount()).sort_values(["rank", "driver", "drive"]).head(N_W05)   # round-robin over drivers
    drives = list(zip(w05.driver, w05.drive))
    w15 = m[(m.win == "w15")].drop_duplicates(["driver", "drive"], keep="first")
    w15 = w15[[d in drives for d in zip(w15.driver, w15.drive)]]
    w15 = w15.iloc[[drives.index(d) for d in zip(w15.driver, w15.drive)]].head(0) if False else \
        w15.assign(order=[drives.index(d) for d in zip(w15.driver, w15.drive)]).sort_values("order").head(N_W15)
    sel = pd.concat([w05, w15])
    dst = os.path.join(paths.MANIFESTS, "icde_cost_episodes_pedal.csv")
    sel[["episode_id"]].to_csv(dst, index=False)
    print(f"{len(sel)} episodes ({(sel.win == 'w05').sum()} w05, {(sel.win == 'w15').sum()} w15) of "
          f"{sel.drop_duplicates(['driver', 'drive']).shape[0]} drives, {sel.driver.nunique()} drivers -> {dst}")


if __name__ == "__main__":
    main()
