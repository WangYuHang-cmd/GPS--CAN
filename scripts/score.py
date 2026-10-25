#!/usr/bin/env python3
"""score.py: score every prediction file of a run against the reference, in parallel over drive shards.

    python scripts/score.py --run full_v1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --workers 10 [--resume]

Each worker owns a shard of drives: it loads every drive once (light mode: GPS reference + released speed grid, no
event tables), re-tiles the episodes exactly as the manifest did, then scores the rows of every prediction group
(protocol x method) that belong to its drives. Prediction files are read row group by row group with a dictionary-
encoded episode_id, so a worker never holds a whole 18 M-row file. Parts are written to
<bench>/scores/<run>/_parts/<protocol>__<method>__shard<k>.parquet and merged into
<bench>/scores/<run>/<protocol>/<method>.episodes.parquet, episodes_all.parquet and summary.csv.
With --resume, a part is reused when it is newer than every prediction file of its group.
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")   # return freed Arrow buffers to the OS (jemalloc keeps them; workers grew to 2 GB)

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, metrics as M, paths, registry, scoring  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402


def drive_key(episode_id: str) -> str:
    return episode_id.rsplit("/", 2)[0]


def read_group_for_drives(flist, keys):
    """Rows of a prediction group whose episode belongs to one of the drive keys, as a DataFrame (episode_id categorical)."""
    tabs = []
    for f in flist:
        pf = pq.ParquetFile(f, read_dictionary=["episode_id"])
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg)
            col = tbl.column("episode_id")
            masks = []
            for ch in col.chunks:
                d = ch.dictionary.to_pylist()
                dm = np.fromiter((drive_key(e) in keys for e in d), dtype=bool, count=len(d))
                masks.append(dm[ch.indices.to_numpy(zero_copy_only=False)])
            mask = np.concatenate(masks) if masks else np.zeros(0, dtype=bool)
            if mask.any():
                tabs.append(tbl.filter(pa.array(mask)))
    if not tabs:
        return pd.DataFrame(columns=["episode_id", "t", "pred_x_m", "pred_y_m", "candidate_id", "prob", "pred_theta_rad"])
    t = pa.concat_tables(tabs, promote_options="default").unify_dictionaries()
    df = t.to_pandas()
    return df.drop_duplicates(["episode_id", "candidate_id", "t"])


class LightEpisode:
    """Episode reduced to numpy arrays (the pandas reference frame is rebuilt on demand); keeps worker memory small."""
    __slots__ = ("episode_id", "driver", "drive", "win", "k", "t0", "t_end", "meta", "_t", "_x", "_y", "_mv")

    def __init__(self, ep):
        self.episode_id, self.driver, self.drive, self.win, self.k = ep.episode_id, ep.driver, ep.drive, ep.win, ep.k
        self.t0, self.t_end, self.meta = ep.t0, ep.t_end, ep.meta
        r = ep.ref
        self._t = r["time_s"].to_numpy(dtype=float)
        self._x = r["x"].to_numpy(dtype=float)
        self._y = r["y"].to_numpy(dtype=float)
        self._mv = r["moving"].to_numpy(dtype=bool)

    @property
    def ref(self):
        return pd.DataFrame({"time_s": self._t, "x": self._x, "y": self._y, "moving": self._mv})


def _worker(task):
    shard_i, drives, groups, man_path, eligible_only, layer, time_comp, run, parts_dir = task
    t_start = time.time()
    man = pd.read_parquet(man_path)
    if eligible_only:
        man = man[man.eligible_main]
    man = man.assign(key=man.driver + "/" + man.drive)
    man = man[man.key.isin(drives)]
    allowed = set(man.episode_id)
    gman = io.load_manifest().set_index(["driver", "drive"])
    dd_by, eps_by = {}, {}
    for key in drives:
        driver, drive = key.split("/")
        if (driver, drive) not in gman.index:
            continue
        r = gman.loc[(driver, drive)]
        delta = io.time_compensation(r["gps_sources"], r["gps_rate_hz"]) if time_comp == "train" else 0.0
        dd = DriveData(driver, drive, delta_s=delta, input_layer=layer, light=True)
        wins = man[man.key == key].win.unique()
        eps_by[key] = {e.episode_id: LightEpisode(e) for w in wins for e in dd.episodes(w)}
        dd.ref = dd.ref1 = None            # scoring needs only the episodes' own reference arrays and the speed grid
        dd.grid = dd.grid[["time_s", "vEgo"]].reset_index(drop=True)
        dd_by[key] = dd
    t_load = time.time() - t_start
    written = []
    keys = set(drives)
    for (proto, method), flist in groups:
        spec = registry.protocol_spec(proto)
        pred = read_group_for_drives(flist, keys)
        rows = []
        if not pred.empty:
            for eid, p in pred.groupby("episode_id", observed=True, sort=False):
                eid = str(eid)
                if eid not in allowed:
                    continue
                key = drive_key(eid)
                ep = eps_by.get(key, {}).get(eid)
                if ep is None:
                    continue
                s = scoring.score_episode(ep, p, dd_by[key], topk=(spec["mode"] == "offline"))
                s.update(protocol=proto, method=method, run=run)
                rows.append(s)
        out = os.path.join(parts_dir, f"{proto}__{method}__shard{shard_i}.parquet")
        tab = scoring.table(rows) if rows else pd.DataFrame(columns=["episode_id", "protocol", "method", "run"])
        tab.to_parquet(out, index=False)
        written.append((proto, method, len(tab)))
        del pred
    return shard_i, len(drives), t_load, time.time() - t_start, written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest.parquet"))
    ap.add_argument("--eligible-only", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--time-comp", default="train", choices=["train", "none"])
    ap.add_argument("--layer", default="events", choices=["events", "grid_compat"], help="input layer used to build the episodes (anchors do not depend on it)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="reuse part files newer than their prediction files")
    ap.add_argument("--methods", nargs="*", default=None, help="only score these method stems (default: all found)")
    args = ap.parse_args()
    man = pd.read_parquet(args.manifest)
    if args.eligible_only:
        man = man[man.eligible_main]
    files = sorted(glob.glob(os.path.join(paths.PREDS, args.run, "*", "*.parquet")))
    if not files:
        raise SystemExit("no prediction files")
    groups_all, groups = {}, {}
    for f in files:
        proto = os.path.basename(os.path.dirname(f))
        stem = os.path.splitext(os.path.basename(f))[0].split(".part")[0]
        groups_all.setdefault((proto, stem), []).append(f)
        if args.methods and stem not in args.methods:
            continue
        groups.setdefault((proto, stem), []).append(f)
    # drives to score = manifest drives that appear in at least one prediction file (dictionary read only)
    pred_drives = set()
    for flist in groups.values():
        for f in flist:
            pf = pq.ParquetFile(f, read_dictionary=["episode_id"])
            for rg in range(pf.num_row_groups):
                col = pf.read_row_group(rg, columns=["episode_id"]).column("episode_id")
                for ch in col.chunks:
                    pred_drives.update(drive_key(e) for e in ch.dictionary.to_pylist())
    man_keys = sorted(set(man.driver + "/" + man.drive) & pred_drives)
    W = max(1, args.workers)
    shards = [man_keys[i::W] for i in range(W)]
    parts_dir = os.path.join(paths.SCORES, args.run, "_parts")
    os.makedirs(parts_dir, exist_ok=True)

    def fresh(proto, method, k):
        p = os.path.join(parts_dir, f"{proto}__{method}__shard{k}.parquet")
        return os.path.exists(p) and all(os.path.getmtime(p) > os.path.getmtime(f) for f in groups[(proto, method)])

    tasks = []
    for k, drives in enumerate(shards):
        todo = [(g, fl) for g, fl in sorted(groups.items()) if not (args.resume and fresh(g[0], g[1], k))]
        if todo and drives:
            tasks.append((k, drives, todo, args.manifest, args.eligible_only, args.layer, args.time_comp, args.run, parts_dir))
    print(f"{len(groups)} prediction groups, {len(man_keys)} drives, {W} shards, {len(tasks)} shard tasks to run", flush=True)
    t0 = time.time()
    if tasks:
        with Pool(W) as pool:
            for shard_i, n_dr, t_load, t_all, written in pool.imap_unordered(_worker, tasks):
                print(f"shard {shard_i}: {n_dr} drives loaded in {t_load:.0f}s, {len(written)} groups scored in {t_all:.0f}s "
                      f"({sum(n for _, _, n in written)} episode rows)", flush=True)
    # merge
    mm_path = os.path.join(paths.PREDS, args.run, "map_missing.csv")
    map_missing = pd.read_csv(mm_path) if os.path.exists(mm_path) else pd.DataFrame(columns=["episode_id", "protocol", "method"])
    all_tabs = []
    for (proto, method) in sorted(groups_all):          # the merge always covers every group with parts (--methods only limits the scoring)
        parts = sorted(glob.glob(os.path.join(parts_dir, f"{proto}__{method}__shard*.parquet")))
        tabs = [pd.read_parquet(p) for p in parts]
        tabs = [t for t in tabs if len(t)]
        if not tabs:
            print(f"{proto} {method}: no scored episodes")
            continue
        tab = pd.concat(tabs, ignore_index=True).drop_duplicates("episode_id")
        n_mm = int(((map_missing.protocol == proto) & (map_missing.method == method)).sum()) if len(map_missing) else 0
        tab["n_map_missing_run"] = n_mm
        out = os.path.join(paths.SCORES, args.run, proto, f"{method}.episodes.parquet")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tab.to_parquet(out, index=False)
        all_tabs.append(tab)
        print(f"{proto} {method}: {len(tab)} episodes scored (map_missing {n_mm})", flush=True)
    full = pd.concat(all_tabs, ignore_index=True)
    full.to_parquet(os.path.join(paths.SCORES, args.run, "episodes_all.parquet"), index=False)
    summ = []
    for (proto, method, win), g in full.groupby(["protocol", "method", "win"]):
        row = dict(protocol=proto, method=method, win=win, n_episodes=len(g), n_drives=g.parent.nunique())
        for metric in ("ade", "fde", "c25", "c50", "c100"):
            if metric in g:
                m, (lo, hi) = M.cluster_bootstrap(g[metric].values, g.parent.values, np.mean, args.n_boot)
                med, _ = M.cluster_bootstrap(g[metric].values, g.parent.values, np.median, 50)
                row.update({f"{metric}_mean": m, f"{metric}_lo": lo, f"{metric}_hi": hi, f"{metric}_median": med})
        for s in ("s50_90", "s100_80"):
            if s in g:
                row[s] = float(g[s].mean())
        if "topk_ade" in g and g["topk_ade"].notna().any():
            row["topk_ade_mean"] = float(g["topk_ade"].mean())
            row["topk_K"] = int(g["topk_K"].max())
        summ.append(row)
    summary = pd.DataFrame(summ).sort_values(["protocol", "win", "method"])
    summary.to_csv(os.path.join(paths.SCORES, args.run, "summary.csv"), index=False)
    pd.set_option("display.width", 250)
    cols = ["protocol", "method", "win", "n_episodes", "ade_mean", "ade_median", "fde_mean", "fde_median", "c50_mean", "c100_mean", "s50_90", "topk_ade_mean"]
    print(summary[[c for c in cols if c in summary]].round(1).to_string(index=False))
    print(f"written {os.path.join(paths.SCORES, args.run, 'summary.csv')} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
