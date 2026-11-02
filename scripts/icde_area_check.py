#!/usr/bin/env python3
"""Map-extent checks for the information-budget discussion (analysis only: this script reads the GNSS reference,
which no baseline ever sees, and writes no prediction).

For every eligible test and few-shot episode of the 5-, 15-minute and whole-drive windows it records how far the
reference track gets from the start p0, in the episode frame (origin p0):
  out_square  the track leaves the 5 km square centred on p0 (half side 2,500 m) that the CAN-Trace and SPy area
              adaptations are cut to: such a window cannot be matched by them whatever the search does
  out_disc    the track leaves the episode's visible map disc (radius 5 km + 40 m/s x D, at most 60 km): no map
              method can follow its tail
  n_turns     reference turns (heading change above 45 deg within 10 s, the failure analysis's definition), used by
              scripts/icde_extra.py for the common-failure strata
Per-episode rows -> reports/icde/tables/area_check_episodes.csv; shares over the methods' expected sets (the same
sets icde_tables.py uses) -> reports/icde/tables/area_coverage.csv.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import io, paths, registry  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
import icde_tables as T  # noqa: E402
from paper_curves import turn_times  # noqa: E402
from score import LightEpisode  # noqa: E402

HALF_SIDE_M = 2500.0
WINS = ("w05", "w15", "full")
OUT = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")


def episode_rows():
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"),
                          columns=["episode_id", "driver", "drive", "win", "fold", "eligible_main"])
    man = man[man.eligible_main & man.fold.isin(list(T.FOLDS)) & man.win.isin(WINS)]
    spec = registry.protocol_spec("F-M-Hm")
    gman = io.load_manifest().set_index(["driver", "drive"])
    rows, t0 = [], time.time()
    keys = sorted(set(zip(man.driver, man.drive)))
    for i, (driver, drive) in enumerate(keys):
        if (driver, drive) not in gman.index:
            continue
        r = gman.loc[(driver, drive)]
        dd = DriveData(driver, drive, delta_s=io.time_compensation(r["gps_sources"], r["gps_rate_hz"]), input_layer="events", light=True)
        wanted = set(man[(man.driver == driver) & (man.drive == drive)].episode_id)
        for w in WINS:
            for ep in dd.episodes(w):
                if ep.episode_id not in wanted:
                    continue
                le = LightEpisode(ep)
                x, y = le._x, le._y
                radius = min(registry.map_radius_m(spec, ep.t_end - ep.t0), 60_000.0)
                rows.append(dict(episode_id=ep.episode_id, win=w, max_abs_xy_m=float(np.nanmax(np.maximum(np.abs(x), np.abs(y)))),
                                 max_r_m=float(np.nanmax(np.hypot(x, y))), disc_radius_m=radius, n_turns=int(len(turn_times(le)))))
        del dd
        if i % 100 == 0:
            print(f"  {i}/{len(keys)} drives, {len(rows)} episodes, {time.time() - t0:.0f} s", flush=True)
    e = pd.DataFrame(rows)
    e["out_square"] = e.max_abs_xy_m > HALF_SIDE_M
    e["out_disc"] = e.max_r_m > e.disc_radius_m
    return e


def coverage(e):
    """Shares over expected sets: the area rows (CAN-Trace: pedal subset; SPy: G3) for the square, every map row's
    G3 / G1 expected set for the disc."""
    _, man, _ = T.load("icde_v11")
    rows = []
    for key, req in (("ct", "pedal"), ("spy", "G3")):
        for fold in T.FOLDS:
            exp = T.expected(man, "w05", fold, "F-M-Hm", req, False)
            d = e[e.episode_id.isin(set(exp.episode_id))]
            rows.append(dict(check="out_square", method=key, win="w05", fold=fold, n=len(d), n_out=int(d.out_square.sum()),
                             share=float(d.out_square.mean()) if len(d) else np.nan, max_m_median=float(d.max_abs_xy_m.median()) if len(d) else np.nan))
    for req in ("G3", "G1"):
        for win in WINS:
            for fold in T.FOLDS:
                exp = T.expected(man, win, fold, "O-M-Hp", req, False)
                d = e[e.episode_id.isin(set(exp.episode_id))]
                rows.append(dict(check="out_disc", method=req, win=win, fold=fold, n=len(d), n_out=int(d.out_disc.sum()),
                                 share=float(d.out_disc.mean()) if len(d) else np.nan, max_m_median=float(d.max_r_m.median()) if len(d) else np.nan))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    e = episode_rows()
    e.to_csv(os.path.join(OUT, "area_check_episodes.csv"), index=False)
    c = coverage(e)
    c.to_csv(os.path.join(OUT, "area_coverage.csv"), index=False)
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
