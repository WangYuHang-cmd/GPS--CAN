#!/usr/bin/env python3
"""Author-sample check of the MacroTracking re-implementation: run it on the sample CAN trace shipped with
CrySyS/MacroTracking (@c99063d) with the authors' car constants and coordinate convention emulated, and compare with
the trajectory the authors committed (output/SAMPLE_TRACE/location.log) and their ground-truth metric.

Emulation of the released code in a metric frame (Web Mercator units are cos(lat) metres): speed = raw * 1.1 * c,
turn radius R = 2.7 c * tan(2.8e-5 * st + pi/2) (bicycle model with L = 2.7 c and road-wheel angle -2.8e-5 * st, st the
sign-flipped raw steering value), correction every 20 c m, junction scale 150 c m, raw-degree angle comparisons.
The map is the 2026 Geofabrik Hungary extract (the authors fetched 600 m osmnx neighbourhoods in 2023), so the check is
approximate: it validates the algorithm, not the map.

Inputs in $CANTRAJ_BENCH/external/macrotracking (trace.log, ground_truth.log, location.log, hungary-latest.osm.pbf).
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths, replay  # noqa: E402
from cantraj.baselines.macrotracking import MacroTracking  # noqa: E402
from cantraj.events import EventStream  # noqa: E402
from cantraj.geo import LocalFrame  # noqa: E402
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams  # noqa: E402
from cantraj.maps.graph import build_graph_tiles  # noqa: E402
from cantraj.maps.osm import extract, tiles_in_radius  # noqa: E402

D = os.path.join(paths.BENCH, "external", "macrotracking")
LAT0, LON0, H0 = 47.47175, 19.05932, 45.0
K_ST = 2.8e-5


def read_trace():
    rows = []
    for line in open(os.path.join(D, "trace.log")):
        p = line.split()
        if len(p) < 5 or p[0].startswith("#"):
            continue
        aid = int(p[1], 16)
        if aid not in (0x180, 0x410):
            continue
        data = [int(x, 16) for x in p[4:4 + int(p[3])]]
        if aid == 0x410:
            rows.append((float(p[0]), int.from_bytes(bytes(data[1:3]), "big") / 3.6 / 100.0, np.nan))
        else:
            st = int.from_bytes(bytes(data[2:4]), "big")
            if data[2] >= 128:
                st -= 1 << 16
            st *= -1
            rows.append((float(p[0]), np.nan, math.degrees(-K_ST * st)))      # road-wheel angle in degrees (SR = 1)
    return pd.DataFrame(rows, columns=["t", "speed", "steer"]).sort_values("t", kind="stable")


def read_states(name):
    out = []
    for line in open(os.path.join(D, name)):
        f = dict(kv.split(":", 1) for kv in [x.strip() for x in line.replace("\t", " \t ").split("\t")] if ":" in kv)
        if "Lat" in f:
            out.append((float(f["Time"]), float(f["Lat"]), float(f["Long"])))
    return np.array(out)


def main():
    c = math.cos(math.radians(LAT0))
    tiles = set(tiles_in_radius(LAT0, LON0, 5000))
    tdir = os.path.join(D, "tiles")
    if not os.path.isdir(tdir):
        extract(os.path.join(D, "hungary-latest.osm.pbf"), (47.3, 18.8, 47.7, 19.4), tdir, "geofabrik:europe/hungary", tiles=tiles, merge=False)
    fr = LocalFrame(LAT0, LON0)
    g = build_graph_tiles(tdir, tiles, fr)
    tr = read_trace()
    t_first = float(tr.t.iloc[0])
    ev = EventStream(tr.t.to_numpy() - t_first, {"speed": tr.speed.to_numpy(), "steer_angle": tr.steer.to_numpy()})
    inp = EpisodeInputs("G3", ev, StaticParams(wheelbase_m=2.7 * c, steer_ratio=1.0), EpisodeInit(0.0, LAT0, LON0, math.radians(H0)), input_layer="events")
    inp.map_graph = g
    theirs = read_states("location.log")
    q = theirs[:, 0] - t_first
    res, rows = {}, []
    for asrel in (True, False):
        b = MacroTracking(as_released=asrel, corr_dist_m=20 * c, junction_m=150 * c, speed_factor=1.1 * c, wheelbase_m=2.7 * c, steer_ratio=1.0)
        xy, _ = replay.run_online(b, inp, q)
        lat, lon = fr.to_lla(xy[:, 0], xy[:, 1])
        tx, ty = fr.to_enu(theirs[:, 1], theirs[:, 2])
        d = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)
        gt = read_states("ground_truth.log")
        gx, gy = fr.to_enu(gt[:, 1], gt[:, 2])
        step = max(len(gx) // 100, 1)
        sel = np.arange(0, len(gx), step)
        closest = [float(np.min(np.hypot(xy[:, 0] - gx[i], xy[:, 1] - gy[i]))) for i in sel]
        closest_theirs = [float(np.min(np.hypot(tx - gx[i], ty - gy[i]))) for i in sel]
        res[asrel] = d
        rows.append(dict(as_released=asrel, n_states=len(d), d_median_m=round(float(np.median(d)), 2), d_p90_m=round(float(np.quantile(d, 0.9)), 2),
                         d_max_m=round(float(d.max()), 2), gt_closest_ours_m=round(float(np.mean(closest)), 2),
                         gt_closest_authors_m=round(float(np.mean(closest_theirs)), 2), authors_reported_m=round(9.08 * c, 2), corrections=b.n_corr))
        print(f"as_released={asrel}: vs authors' states: median {np.median(d):.1f} m, p90 {np.quantile(d, 0.9):.1f} m, max {d.max():.1f} m "
              f"(n={len(d)}); closest-point metric vs ground truth: ours {np.mean(closest):.1f} m, authors' trajectory {np.mean(closest_theirs):.1f} m "
              f"(authors report 9.08 Mercator units = {9.08 * c:.1f} m); corrections {b.n_corr}")
    pd.DataFrame(rows).to_csv(os.path.join(paths.GPS_DIR, "reports", "icde", "macrotracking_author_sample.csv"), index=False)


if __name__ == "__main__":
    main()
