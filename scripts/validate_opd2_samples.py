#!/usr/bin/env python3
"""Author-sample check of the OPD-II re-implementation (spritz-group/OPD-II @5fe93630).

For a few of the CAN logs shipped with the repository (examples/experiments: one or two per car), run
(a) the released script scripts/OPD_II.py, unmodified, answers fed on stdin, against a local valhalla_service on port
    8002 (as in the authors' Docker set-up), and
(b) the re-implementation (`cantraj.baselines.opd2.OPD2`, as released, authors' car constants emulated: decoded angle
    with steering ratio 1 and the per-car "wheelbase" of paths.csv, clip 35) on the same decoded log,
both against the same Valhalla tiles (drive network of the area cut from the Geofabrik Italy north-east extract with the
benchmark's tiler, pyvalhalla 3.9.1). Reports the point-by-point distance between the two outputs and, for both, the
closest-point distance to the GPX ground truth. Start = first GPX point; start bearing = integer compass bearing of the
first 20 m of the GPX track (the released tool asks for these by hand; paths.csv does not list them).
Inputs and outputs in $CANTRAJ_BENCH/external/opd2.
"""
import csv
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths  # noqa: E402
from cantraj.baselines.opd2 import OPD2  # noqa: E402
from cantraj.events import EventStream  # noqa: E402
from cantraj.geo import LocalFrame  # noqa: E402
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams  # noqa: E402
from cantraj.maps.osm import extract, tiles_in_radius  # noqa: E402
from cantraj.maps.valhalla import MeiliMatcher  # noqa: E402
import build_valhalla_v2 as BV  # noqa: E402

D = os.path.join(paths.BENCH, "external", "opd2")
DEPS = os.path.join(paths.BENCH, "pylibs", "opd2deps")
SAMPLES = [("capture_1", "Renault", 0), ("capture_4", "Renault", 3), ("duster_1", "Dacia", 19), ("duster_3", "Dacia", 21),
           ("opel_1", "Opel", 30), ("opel_7", "Opel", 36), ("peugeot_2", "Peugeot", 45), ("peugeot_7", "Peugeot", 50)]
WB = {"Renault": 3.85, "Dacia": 4.0, "Opel": 4.9, "Peugeot": 4.5}


def gpx_points(path):
    s = open(path).read()
    pts = re.findall(r'lat="([-0-9.]+)"\s+lon="([-0-9.]+)"', s)
    if not pts:
        pts = [(b, a) for a, b in re.findall(r'lon="([-0-9.]+)"\s+lat="([-0-9.]+)"', s)]
    return np.array([(float(a), float(b)) for a, b in pts])


def start_bearing(gt):
    fr = LocalFrame(*gt[0])
    x, y = fr.to_enu(gt[:, 0], gt[:, 1])
    k = int(np.argmax(np.hypot(x, y) >= 20.0))
    return int(round(math.degrees(math.atan2(x[k], y[k])) % 360.0))


def decode(log, vehicle):
    sys.path.insert(0, D)
    import OPD_lib as L                                       # the released decoder
    can_id = {"Renault": "0C6", "Dacia": "0C6", "Peugeot": "2EB", "Opel": "2F5"}[vehicle]
    rows = []
    for line in open(log):
        p = line.strip().split(" ")
        t = float(p[0][1:-1])
        cid, data = p[2].split("#")
        if cid == can_id:
            b0, b1 = L.get_bytes(data, vehicle)
            rows.append((t, np.nan, L.decode_angle(b0, b1, vehicle)))
        elif cid == "7E8":
            rows.append((t, int(data[6:8], 16) / 3.6, np.nan))
    return pd.DataFrame(rows, columns=["t", "speed", "angle"])


def tiles_and_valhalla(all_gt):
    lat0, lat1 = all_gt[:, 0].min() - 0.05, all_gt[:, 0].max() + 0.05
    lon0, lon1 = all_gt[:, 1].min() - 0.07, all_gt[:, 1].max() + 0.07
    tiles = {(i, j) for i in range(int(math.floor(lat0 / 0.25)), int(math.floor(lat1 / 0.25)) + 1)
             for j in range(int(math.floor(lon0 / 0.25)), int(math.floor(lon1 / 0.25)) + 1)}
    tdir, cdir = os.path.join(D, "tiles"), os.path.join(D, "valhalla")
    if not os.path.isdir(tdir):
        extract(os.path.join(D, "nord-est-latest.osm.pbf"), (lat0, lon0, lat1, lon1), tdir, "geofabrik:europe/italy/nord-est", tiles=tiles, merge=False)
    if not os.path.exists(os.path.join(cdir, "tiles.tar")):
        BV.TILES_V2 = tdir
        os.makedirs(cdir, exist_ok=True)
        BV.write_pbf(os.path.join(cdir, "net.osm.pbf"), *BV.load(sorted(tiles)))
        BV.build(cdir, 4)
    return os.path.join(cdir, "valhalla.json")


def wait_port(port, secs=60):
    t0 = time.time()
    while time.time() - t0 < secs:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def main():
    gts = {n: gpx_points(os.path.join(D, f"{n}.gpx")) for n, _, _ in SAMPLES}
    cfg = tiles_and_valhalla(np.concatenate(list(gts.values())))
    env = dict(os.environ, PYTHONPATH=BV.VLIB, PATH=os.path.join(BV.VLIB, "valhalla", "bin") + os.pathsep + os.environ["PATH"])
    svc = subprocess.Popen([sys.executable, "-m", "valhalla", "valhalla_service", cfg, "1"], env=env,
                           stdout=open(os.path.join(D, "service.log"), "w"), stderr=subprocess.STDOUT)
    rows = []
    try:
        assert wait_port(8002), "valhalla_service did not start"
        for name, veh, pid in SAMPLES:
            gt = gts[name]
            brg = start_bearing(gt)
            wd = os.path.join(D, "run", name)
            shutil.rmtree(wd, ignore_errors=True)
            os.makedirs(wd)
            answers = f"{os.path.join(D, name + '.log')}\n{veh}\n{gt[0, 0]}\n{gt[0, 1]}\n{brg}\n{WB[veh]}\nn\n"
            renv = dict(os.environ, PYTHONPATH=DEPS + os.pathsep + D, BROWSER="true", MPLBACKEND="Agg")
            t0 = time.time()
            r = subprocess.run([sys.executable, os.path.join(D, "OPD_II.py")], input=answers, text=True, cwd=wd, env=renv,
                               capture_output=True, timeout=1200)
            t_auth = time.time() - t0
            if not os.path.exists(os.path.join(wd, "inferred_path.gpx")):
                print(name, "released script failed:", r.stderr[-400:])
                continue
            theirs = gpx_points(os.path.join(wd, "inferred_path.gpx"))
            dec = decode(os.path.join(D, name + ".log"), veh)
            t_first = float(dec.t.iloc[0])
            ev = EventStream(dec.t.to_numpy() - t_first, {"speed": dec.speed.to_numpy(), "steer_angle": dec.angle.to_numpy()})
            init = EpisodeInit(0.0, float(gt[0, 0]), float(gt[0, 1]), math.radians(90.0 - brg))
            inp = EpisodeInputs("G3", ev, StaticParams(wheelbase_m=WB[veh], steer_ratio=1.0), init, input_layer="events")
            m = OPD2(clip_sw_deg=35.0, wheelbase_m=WB[veh], steer_ratio=1.0, matcher_factory=lambda i: MeiliMatcher(None, 200.0, config=cfg))
            t0 = time.time()
            m.reconstruct(inp, np.arange(0.1, float(ev.t_s[-1]), 0.1))
            t_ours = time.time() - t0
            fr = init.frame()
            ours_xy = np.array(m.out_xy)
            n = min(len(theirs), len(ours_xy))
            tx, ty = fr.to_enu(theirs[:n, 0], theirs[:n, 1])
            d = np.hypot(ours_xy[:n, 0] - tx, ours_xy[:n, 1] - ty)
            gx, gy = fr.to_enu(gt[:, 0], gt[:, 1])

            def closest(px, py):
                return float(np.mean([np.min(np.hypot(px - a, py - b)) for a, b in zip(gx, gy)]))
            reported = [r_ for r_ in csv.DictReader(open(os.path.join(D, "paths.csv"))) if int(r_["Path ID"]) == pid][0]["Results"]
            rows.append(dict(sample=name, vehicle=veh, bearing=brg, n_theirs=len(theirs), n_ours=len(ours_xy),
                             d_median_m=round(float(np.median(d)), 2), d_p90_m=round(float(np.quantile(d, 0.9)), 2), d_max_m=round(float(d.max()), 2),
                             gt_closest_theirs_m=round(closest(tx, ty), 1), gt_closest_ours_m=round(closest(ours_xy[:n, 0], ours_xy[:n, 1]), 1),
                             match_fail=m.n_match_fail, reported_overlap_pct=reported, secs_released=round(t_auth, 1), secs_ours=round(t_ours, 2)))
            print(rows[-1], flush=True)
    finally:
        svc.terminate()
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths.GPS_DIR, "reports", "icde", "opd2_author_samples.csv"), index=False)
    print(out.to_string())


if __name__ == "__main__":
    main()
