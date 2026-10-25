"""Per-point figure data for the paper: error by elapsed time, pooled coverage curves, lost-road timing and one
anonymised trajectory example. Recomputed from the prediction parquet files and the 1 Hz reference (the per-episode
score tables keep only the summary metrics). Outputs go to reports/paper/figures/data/ and never contain coordinates
other than episode-frame metres, and never raw paths or device identifiers.

  python scripts/paper_curves.py --run full_v1 [--folds test few_shot_heldout] [--limit-drives N]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import io, paths  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
from score import LightEpisode, read_group_for_drives  # noqa: E402

GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(GPS_DIR, "reports", "paper", "figures", "data")

# (protocol, method stem) groups read for the curves; the figure scripts map stems to display names and colours
GROUPS = [
    ("O-N-Hp", "p0_hold__G1"), ("O-N-Hp", "dr_none__G1"), ("O-N-Hp", "dr_steer__G3"), ("O-N-Hp", "dr_steer_calib__G3"),
    ("O-N-Hp", "learned_gru_s0__G3"), ("O-N-Hp", "dr_curvature__G5"),
    ("O-M-Hp", "map_pf_none__G1"), ("O-M-Hp", "map_pf_steer__G3"), ("F-M-Hp", "map_beam_steer__G3"),
    ("O-N-Hm", "dr_steer__G3"), ("O-M-Hm", "map_pf_steer__G3"), ("F-M-Hm", "map_beam_steer__G3"),
]
FAIL_GROUPS = {("O-M-Hp", "map_pf_steer__G3"), ("F-M-Hp", "map_beam_steer__G3"), ("O-M-Hp", "map_pf_none__G1")}
EXAMPLE_GROUPS = [("O-N-Hp", "dr_steer__G3"), ("O-M-Hp", "map_pf_steer__G3"), ("F-M-Hp", "map_beam_steer__G3")]
BIN_S = {"w05": 1.0, "w15": 1.0, "full": 30.0}
R_EDGES = np.r_[np.arange(0.0, 2000.0, 1.0), np.arange(2000.0, 20000.0, 50.0), np.inf]


def episode_errors(ep, pred):
    """Top-1 point errors at the reference seconds (NaN = no prediction), plus t_rel and the predicted xy."""
    t = ep._t
    top1 = pred[pred["candidate_id"] == 0]
    key = np.round(t, 6)
    idx = pd.Index(np.round(top1["t"].to_numpy(dtype=float), 6))
    pos = idx.get_indexer(key)
    xy = np.full((len(t), 2), np.nan)
    hit = pos >= 0
    xy[hit] = top1[["pred_x_m", "pred_y_m"]].to_numpy(dtype=float)[pos[hit]]
    err = np.hypot(xy[:, 0] - ep._x, xy[:, 1] - ep._y)
    return err, t - ep.t0, xy


def turn_times(ep, min_turn_deg=45.0, window_s=10.0):
    """Times (relative to t0) at which the reference heading changed by more than min_turn_deg within window_s."""
    t, x, y, mv = ep._t, ep._x, ep._y, ep._mv
    if len(t) < 3:
        return np.zeros(0)
    psi = np.full(len(t), np.nan)
    dx, dy = np.diff(x), np.diff(y)
    ok = np.hypot(dx, dy) > 1.0
    psi[1:][ok] = np.arctan2(dy[ok], dx[ok])
    psi = pd.Series(psi).ffill().bfill().to_numpy()
    out = []
    j0 = 0
    for i in range(len(t)):
        while t[i] - t[j0] > window_s:
            j0 += 1
        d = np.degrees(np.abs(np.angle(np.exp(1j * (psi[i] - psi[j0])))))
        if d > min_turn_deg and (not out or t[i] - out[-1] > window_s):
            out.append(t[i])
    return np.asarray(out) - ep.t0


def lost_onset(err):
    """Start index of the terminal excursion beyond 100 m (NaN counts as beyond), or -1 if the episode ends inside."""
    beyond = ~(np.nan_to_num(err, nan=np.inf) <= 100.0)
    if not beyond[-1]:
        return -1
    inside = np.flatnonzero(~beyond)
    return int(inside[-1] + 1) if len(inside) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v1")
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    ap.add_argument("--folds", nargs="+", default=["test", "few_shot_heldout"])
    ap.add_argument("--wins", nargs="+", default=["w05", "w15", "full"])
    ap.add_argument("--example-folds", nargs="+", default=["val"])
    ap.add_argument("--example-bbox", nargs=4, type=float, default=[41.9, 42.9, -73.6, -69.9],
                    help="lat_min lat_max lon_min lon_max for the example episode (default: Massachusetts)")
    ap.add_argument("--limit-drives", type=int, default=0)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t_start = time.time()

    man = pd.read_parquet(args.manifest)
    man = man[man.eligible_main & man.win.isin(args.wins)]
    main_eps = man[man.fold.isin(args.folds)]
    # example candidates: one 5-minute and one 15-minute episode of the requested folds whose start lies inside the bbox
    # (coordinates are used only to choose, never written); the chosen episode is the median map-PF episode of the region
    p0 = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_p0.parquet"))
    sc = pd.read_parquet(os.path.join(paths.SCORES, args.run, "O-M-Hp", "map_pf_steer__G3.episodes.parquet"), columns=["episode_id", "ade"])
    la0, la1, lo0, lo1 = args.example_bbox
    ex_ids, ex_drives = [], set()
    for w in ("w05", "w15"):
        ex = man[(man.win == w) & man.fold.isin(args.example_folds)].merge(p0[["episode_id", "lat0", "lon0"]], on="episode_id")
        ex = ex[ex.lat0.between(la0, la1) & ex.lon0.between(lo0, lo1)].merge(sc, on="episode_id").sort_values("ade")
        if len(ex):
            med = ex.iloc[len(ex) // 2]
            ex_ids.append(med.episode_id)
            ex_drives.add(f"{med.driver}/{med.drive}")
            print(f"example {w}: {len(ex)} candidates; chosen episode has map PF ADE {med.ade:.1f} m (median of the region)")

    keys = sorted(set(main_eps.driver + "/" + main_eps.drive) | ex_drives)
    if args.limit_drives:
        keys = sorted(ex_drives) + [k for k in keys if k not in ex_drives][: args.limit_drives]
    allowed = set(main_eps.episode_id) | set(ex_ids)
    print(f"{len(keys)} drives, {len(allowed)} episodes, groups {len(GROUPS)}", flush=True)

    gman = io.load_manifest().set_index(["driver", "drive"])
    eps_by = {}
    for n, key in enumerate(keys):
        driver, drive = key.split("/")
        if (driver, drive) not in gman.index:
            continue
        r = gman.loc[(driver, drive)]
        dd = DriveData(driver, drive, delta_s=io.time_compensation(r["gps_sources"], r["gps_rate_hz"]), input_layer="events", light=True)
        wins = man[(man.driver == driver) & (man.drive == drive)].win.unique()
        for w in wins:
            for e in dd.episodes(w):
                if e.episode_id in allowed:
                    eps_by[e.episode_id] = LightEpisode(e)
        del dd
        if n % 100 == 0:
            print(f"  loaded {n}/{len(keys)} drives, {len(eps_by)} episodes, {time.time() - t_start:.0f} s", flush=True)
    win_of = dict(zip(man.episode_id, man.win))
    fold_of = dict(zip(man.episode_id, man.fold))

    curve_rows, cov_rows, fail_rows, ex_rows = [], [], [], {}
    for proto, stem in GROUPS:
        flist = sorted(glob.glob(os.path.join(paths.PREDS, args.run, proto, f"{stem}.part*.parquet")))
        if not flist:
            print(f"no predictions for {proto} {stem}", flush=True)
            continue
        pred = read_group_for_drives(flist, set(keys))
        acc = {w: [] for w in args.wins}                          # (t_bin, err) pairs per window
        hist = {w: np.zeros(len(R_EDGES) - 1, dtype=np.int64) for w in args.wins}
        n_pts = {w: 0 for w in args.wins}
        n_eps = {w: 0 for w in args.wins}
        for eid, p in pred.groupby("episode_id", observed=True, sort=False):
            eid = str(eid)
            ep = eps_by.get(eid)
            if ep is None:
                continue
            err, t_rel, xy = episode_errors(ep, p)
            if eid in ex_ids and (proto, stem) in EXAMPLE_GROUPS:
                ex_rows.setdefault(eid, {"t_rel_s": t_rel, "ref_x_m": ep._x, "ref_y_m": ep._y})
                ex_rows[eid][f"{proto}__{stem}__x_m"] = xy[:, 0]
                ex_rows[eid][f"{proto}__{stem}__y_m"] = xy[:, 1]
            if fold_of.get(eid) not in args.folds:
                continue
            w = win_of[eid]
            b = BIN_S[w]
            tb = np.floor(t_rel / b) * b
            if b > 1.0:                                           # per-episode bin mean first, then quantiles across episodes
                s = pd.Series(err).groupby(tb).mean()
                acc[w].append(np.column_stack([s.index.to_numpy(), s.to_numpy()]))
            else:
                acc[w].append(np.column_stack([tb, err]))
            e_fin = np.where(np.isfinite(err), err, np.inf)
            hist[w] += np.histogram(e_fin, bins=R_EDGES)[0]
            n_pts[w] += len(err); n_eps[w] += 1
            if (proto, stem) in FAIL_GROUPS and w in ("w15", "full"):
                k = lost_onset(err)
                turns = turn_times(ep)
                row = dict(protocol=proto, method=stem, episode_id=eid, win=w, dur_s=float(t_rel[-1]), n_turns=int(len(turns)),
                           lost=bool(k >= 0), t_lost_rel_s=float(t_rel[k]) if k >= 0 else np.nan)
                if k >= 0:
                    before = turns[turns <= t_rel[k]]
                    row["dt_since_last_turn_s"] = float(t_rel[k] - before[-1]) if len(before) else np.nan
                    row["turn_within_30s_before"] = bool(len(before) and t_rel[k] - before[-1] <= 30.0)
                fail_rows.append(row)
        for w in args.wins:
            if not acc[w]:
                continue
            a = np.concatenate(acc[w])
            df = pd.DataFrame({"t_bin_s": a[:, 0], "err": a[:, 1]})
            g = df.groupby("t_bin_s")["err"]
            q = pd.DataFrame({"median_m": g.median(), "q25_m": g.quantile(0.25), "q75_m": g.quantile(0.75),
                              "n": g.size(), "n_pred": g.count()}).reset_index()
            q["protocol"], q["method"], q["win"] = proto, stem, w
            curve_rows.append(q)
            c = np.cumsum(hist[w]) / max(n_pts[w], 1)
            cov_rows.append(pd.DataFrame({"r_m": R_EDGES[1:-1], "c_r": c[:-1], "protocol": proto, "method": stem, "win": w,
                                          "n_points": n_pts[w], "n_episodes": n_eps[w]}))
        print(f"{proto} {stem}: {sum(n_eps.values())} episodes in {time.time() - t_start:.0f} s", flush=True)
        del pred

    pd.concat(curve_rows).to_parquet(os.path.join(args.out, "curve_time.parquet"), index=False)
    pd.concat(cov_rows).to_parquet(os.path.join(args.out, "coverage_curve.parquet"), index=False)
    pd.DataFrame(fail_rows).to_parquet(os.path.join(args.out, "failure_events.parquet"), index=False)
    for eid, cols in ex_rows.items():
        df = pd.DataFrame(cols)
        df.to_parquet(os.path.join(args.out, f"example_trajectory_{win_of[eid]}.parquet"), index=False)
        print(f"example {win_of[eid]} written: {len(df)} reference seconds, {len(df.columns)} columns (episode frame only)")
    print(f"done in {(time.time() - t_start) / 60:.1f} min -> {args.out}")


if __name__ == "__main__":
    main()
