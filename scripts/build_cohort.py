#!/usr/bin/env python3
"""Same-cohort horizon study (ICDE RQ3): one start per drive, three horizons. For every test / few-shot drive the first
eligible 15-minute window whose start is on the map (map_ok) gives the start; the 60 s, 300 s and 900 s prefixes from
that start must each satisfy the v1 reference rules on their own (>= 0.8 reference fixes per second, no gap > 30 s,
path >= 100 m), decided from the reference only. Writes manifests/cohort_v11.csv (episode_id = the w15 episode)."""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402

PREFIXES = (60, 300, 900)


def _drive(task):
    driver, drive, eids = task
    r = io.load_manifest().set_index(["driver", "drive"]).loc[(driver, drive)]
    dd = DriveData(driver, drive, delta_s=io.time_compensation(r["gps_sources"], r["gps_rate_hz"]), light=True)
    eps = {e.episode_id: e for e in dd.episodes("w15")}
    for eid in eids:                                   # first eligible map_ok w15 episode of the drive
        e = eps.get(eid)
        if e is None:
            continue
        row = dict(episode_id=eid, driver=driver, drive=drive)
        for P in PREFIXES:
            ref = e.ref[e.ref.time_s <= e.t0 + P]
            t = np.concatenate([[e.t0], ref.time_s.to_numpy(dtype=float)])
            gaps = np.diff(t)
            plen = float(np.hypot(np.diff(ref.x.to_numpy()), np.diff(ref.y.to_numpy())).sum()) if len(ref) > 1 else 0.0
            row[f"ok_{P}"] = bool(len(ref) >= 0.8 * P and (len(gaps) == 0 or gaps.max() <= 30.0) and plen >= 100.0)
        row["cohort"] = all(row[f"ok_{P}"] for P in PREFIXES)
        return row
    return None


def main():
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "driver", "drive", "fold", "win", "k", "eligible_main"])
    me = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_map_v11.parquet"), columns=["episode_id", "map_ok"])
    w = man[man.eligible_main & (man.win == "w15") & man.fold.isin(["test", "few_shot_heldout"])].merge(me, on="episode_id")
    w = w[w.map_ok].sort_values(["driver", "drive", "k"])
    tasks = [(d, v, list(g.episode_id)) for (d, v), g in w.groupby(["driver", "drive"])]
    with Pool(4) as pool:
        rows = [r for r in pool.imap_unordered(_drive, tasks, chunksize=4) if r]
    out = pd.DataFrame(rows).merge(w[["episode_id", "fold"]], on="episode_id")
    out.to_csv(os.path.join(paths.MANIFESTS, "cohort_v11.csv"), index=False)
    print(out.groupby("fold")[["ok_60", "ok_300", "ok_900", "cohort"]].sum().to_string())
    print(f"cohort: {int(out.cohort.sum())} drives of {out.driver[out.cohort].nunique()} drivers -> manifests/cohort_v11.csv")


if __name__ == "__main__":
    main()
