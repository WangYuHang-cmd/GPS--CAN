#!/usr/bin/env python3
"""m0_gps_quality.py: drive-level reference quality, anchors, spatial-grid overlap, route families, splits.

Per GPS-ok drive: first valid fix, first qualifying anchor (t0 offset), stationary start, reference gaps, 1 Hz view
size, path length, CAN moving fraction, hole fraction of the released table, metro tile (0.25 deg), 250 m grid cells.
Across drives: within-driver route families (containment >= 0.8), cross-driver overlap (share of a drive's cells seen
in OTHER drivers' train-fold drives), and the split tables:
  split_main                      DriveDNA driver-disjoint fold (train / val / test / few_shot_heldout)
  split_seen_route                within driver: one trip of every family with >= 3 trips held out (repeat-commute leak)
  split_cross_route_within_driver within driver: whole families held out (~25% of a driver's drives) when >= 2 families
  split_cross_metro               Atlanta-area tiles held out as test (the Controlled-supplement city excluded from everything public)
Outputs: manifests/drive_quality.parquet, manifests/split_drives.parquet(.csv), reports/data_audit.md.
"""
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths  # noqa: E402
from cantraj.episodes import find_anchor, view_1hz  # noqa: E402
from cantraj.quality import hole_mask  # noqa: E402
from cantraj.splits import drive_folds  # noqa: E402

CELL_DEG = 0.00225          # ~250 m
TAMPA = dict(lat=(27.5, 28.6), lon=(-83.0, -82.0))
ATLANTA = dict(lat=(33.25, 34.75), lon=(-85.0, -83.5))


def cells_of(lat, lon):
    cy = np.round(lat / CELL_DEG).astype(np.int64)
    cx = np.round(lon * np.cos(np.radians(lat)) / CELL_DEG).astype(np.int64)
    return set(zip(cx.tolist(), cy.tolist()))


def main():
    man = io.load_manifest()
    ok = man[man.status == "ok"].reset_index(drop=True)
    folds = drive_folds()
    idx = io.load_drive_index()[["driver", "drive", "model_canon"]].set_index(["driver", "drive"])
    rows, cells = [], {}
    for i, r in enumerate(ok.itertuples(index=False)):
        ref = io.load_gps(r.driver, r.drive)
        if ref.empty:
            continue
        ref1 = view_1hz(ref)
        g = pq.read_table(paths.grid_path(r.driver, r.drive), columns=["time_s", "vEgo", "steeringAngleDeg"]).to_pandas()
        gt, gv = g["time_s"].to_numpy(dtype=float), g["vEgo"].to_numpy(dtype=float)
        speed_at = lambda t: float(np.interp(t, gt, gv))  # noqa: E731
        t_first = float(ref1["time_s"].iloc[0])
        anc = find_anchor(ref1, speed_at, t_first)
        gaps = np.diff(ref1["time_s"].to_numpy(dtype=float))
        win = (gt >= t_first) & (gt <= float(ref1["time_s"].iloc[-1]))
        hm = hole_mask(gv) & hole_mask(g["steeringAngleDeg"].to_numpy(dtype=float))
        lat, lon = ref1["lat"].to_numpy(), ref1["lon"].to_numpy()
        step = max(1, int(round(len(ref) / max(1.0, ref["time_s"].iloc[-1] - ref["time_s"].iloc[0]))))
        c = cells_of(ref["lat"].to_numpy()[::step], ref["lon"].to_numpy()[::step])
        cells[(r.driver, r.drive)] = c
        rows.append(dict(set=r.set, driver=r.driver, drive=r.drive, model_canon=idx.loc[(r.driver, r.drive), "model_canon"] if (r.driver, r.drive) in idx.index else None,
                         fold=folds.get((r.driver, r.drive), "unassigned"), gps_source=r.gps_sources, gps_rate_hz=r.gps_rate_hz,
                         n_ref=len(ref), n_ref1hz=len(ref1), first_fix_s=t_first, last_fix_s=float(ref1["time_s"].iloc[-1]),
                         stationary_start=bool(ref1["speed_mps"].iloc[0] < 1.0),
                         anchor_found=anc is not None, t0_offset_s=(anc["t0"] - t_first) if anc else np.nan,
                         anchor_bearing_acc_deg=anc["anchor_bearing_acc_deg"] if anc else np.nan,
                         n_gaps_gt30=int((gaps > 30).sum()), max_gap_s=float(gaps.max()) if len(gaps) else np.nan,
                         moving_frac_can=float(np.mean(gv[win] > 1.0)) if win.any() else np.nan,
                         hole_frac=float(hm[win].mean()) if win.any() else np.nan,
                         path_len_km=float(np.sum(np.hypot(np.diff(lat) * 111320, np.diff(lon) * 111320 * np.cos(np.radians(lat[:-1]))))) / 1000 if len(lat) > 1 else 0.0,
                         lat_med=float(np.median(lat)), lon_med=float(np.median(lon)),
                         tile_lat=int(np.floor(np.median(lat) / 0.25)), tile_lon=int(np.floor(np.median(lon) / 0.25)),
                         in_tampa=bool(TAMPA["lat"][0] <= np.median(lat) <= TAMPA["lat"][1] and TAMPA["lon"][0] <= np.median(lon) <= TAMPA["lon"][1]),
                         in_atlanta=bool(ATLANTA["lat"][0] <= np.median(lat) <= ATLANTA["lat"][1] and ATLANTA["lon"][0] <= np.median(lon) <= ATLANTA["lon"][1]),
                         n_cells=len(c)))
        if i % 500 == 0:
            print(i, flush=True)
    q = pd.DataFrame(rows)
    keys = list(cells)
    post = defaultdict(list)
    for k in keys:
        for c in cells[k]:
            post[c].append(k)
    drv = {k: k[0] for k in keys}
    fold = dict(zip(zip(q.driver, q.drive), q.fold))
    same_max, other_cov_train, other_max = {}, {}, {}
    for k in keys:
        cnt = Counter()
        for c in cells[k]:
            for j in post[c]:
                if j != k:
                    cnt[j] += 1
        n = max(1, len(cells[k]))
        same_max[k] = max([v / n for j, v in cnt.items() if drv[j] == drv[k]], default=0.0)
        other_max[k] = max([v / n for j, v in cnt.items() if drv[j] != drv[k]], default=0.0)
        covered = sum(1 for c in cells[k] if any(drv[j] != drv[k] and fold.get(j) == "train" for j in post[c]))
        other_cov_train[k] = covered / n
    q["same_driver_max_containment"] = [same_max[k] for k in zip(q.driver, q.drive)]
    q["other_driver_max_containment"] = [other_max[k] for k in zip(q.driver, q.drive)]
    q["train_overlap"] = [other_cov_train[k] for k in zip(q.driver, q.drive)]
    # within-driver route families (greedy, containment >= 0.8 either way)
    fam, nf = {}, 0
    for k in keys:
        if k in fam:
            continue
        nf += 1
        fam[k] = nf
        cnt = Counter()
        for c in cells[k]:
            for j in post[c]:
                if j != k and drv[j] == drv[k]:
                    cnt[j] += 1
        for j, v in cnt.items():
            if j not in fam and (v / max(1, len(cells[k])) >= 0.8 or v / max(1, len(cells[j])) >= 0.8):
                fam[j] = nf
    q["route_family"] = [fam[k] for k in zip(q.driver, q.drive)]
    # cross-driver families (any driver), reported only
    q["cross_driver_repeat"] = q["other_driver_max_containment"] >= 0.8
    # splits
    rng = np.random.default_rng(20260722)
    q["split_main"] = q["fold"]
    q["split_seen_route"] = ""
    q["split_cross_route_within_driver"] = ""
    for d, g in q.groupby("driver"):
        fams = g.groupby("route_family")
        for f, gg in fams:
            if len(gg) >= 3:
                test = rng.choice(gg.index, 1)
                q.loc[gg.index, "split_seen_route"] = "train"
                q.loc[test, "split_seen_route"] = "test"
        nfam = g.route_family.nunique()
        if nfam >= 2:
            fam_ids = g.route_family.unique()
            rng.shuffle(fam_ids)
            n_test = max(1, int(round(0.25 * nfam)))
            test_f = set(fam_ids[:n_test])
            q.loc[g.index, "split_cross_route_within_driver"] = ["test" if f in test_f else "train" for f in g.route_family]
    q["split_cross_metro"] = np.where(q.in_atlanta, "test", np.where(q.in_tampa, "excluded", "train"))
    os.makedirs(paths.MANIFESTS, exist_ok=True)
    q.to_parquet(os.path.join(paths.MANIFESTS, "drive_quality.parquet"), index=False)
    pub = q.drop(columns=["lat_med", "lon_med", "tile_lat", "tile_lon"])   # medians stay out of the public table
    pub.to_parquet(os.path.join(paths.MANIFESTS, "split_drives.parquet"), index=False)
    pub.round(4).to_csv(os.path.join(paths.MANIFESTS, "split_drives.csv"), index=False)
    # report
    L = ["# Data audit: reference quality, anchors, overlap, splits\n", f"GPS-ok drives: {len(q)} (main {int((q.set == 'main').sum())}, controlled {int((q.set == 'controlled').sum())}); drivers {q.driver.nunique()}\n"]
    L += ["## By source and rate", q.assign(hires=q.gps_rate_hz >= 8).groupby(["gps_source", "hires"]).agg(drives=("drive", "size"), drivers=("driver", "nunique"), hours=("last_fix_s", lambda s: 0), km=("path_len_km", "sum")).round(0).to_markdown(), ""]
    L += ["## Anchors", f"anchor found: {q.anchor_found.mean():.3f}; stationary start: {q.stationary_start.mean():.3f}; t0 offset quantiles (s): {q.t0_offset_s.quantile([.5, .75, .9, .95]).round(1).to_dict()}", ""]
    L += ["## Reference gaps and holes", f"drives with a reference gap > 30 s: {int((q.n_gaps_gt30 > 0).sum())}; hole fraction > 1%: {int((q.hole_frac > 0.01).sum())}; moving fraction (CAN) median {q.moving_frac_can.median():.2f}", ""]
    L += ["## Spatial grid overlap (250 m cells)", f"same-driver containment >= 0.8: {(q.same_driver_max_containment >= 0.8).mean():.3f}; other-driver containment >= 0.8: {(q.other_driver_max_containment >= 0.8).mean():.3f}; test-fold drives' mean train_overlap: {q[q.fold == 'test'].train_overlap.mean():.3f}", ""]
    fs = q.groupby("route_family").size()
    L += ["## Route families (within driver)", f"families {len(fs)}; singletons {int((fs == 1).sum())}; largest {int(fs.max())}; drives in families >= 3: {int(fs[fs >= 3].sum())}", ""]
    L += ["## Splits (drives)"]
    for s in ("split_main", "split_seen_route", "split_cross_route_within_driver", "split_cross_metro"):
        L += [f"- {s}: {q[s].replace('', 'n/a').value_counts().to_dict()}"]
    os.makedirs(paths.REPORTS, exist_ok=True)
    with open(os.path.join(paths.REPORTS, "data_audit.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[-12:]))
    print("written manifests/drive_quality.parquet, split_drives.parquet, reports/data_audit.md")


if __name__ == "__main__":
    main()
