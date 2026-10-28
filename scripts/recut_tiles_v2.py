#!/usr/bin/env python3
"""Map v2 for benchmark v1.1: re-cut every tile needed by the evaluated episodes (val, test, few_shot_heldout; each
episode's own disc min(r(D), 60 km) around its start) from EVERY intersecting configured Geofabrik extract into a fresh
tile directory, tiles mode only. v1 tiles stay untouched.

Why: v1 tiles were cut incrementally and a tile was only re-cut when its ways.parquet was missing, so margin tiles of
early bounding-box passes kept only the ways that crossed in from the box (e.g. 13 instead of 1,358 roads); starts in
such tiles found no road within 400 m and the map methods fell back to an arbitrary edge.

Outputs: $CANTRAJ_BENCH/maps/tiles_v2/t{i}_{j}/*.parquet, manifests/map_tiles_v2.csv (one row per tile: sources, counts,
pbf md5 and snapshot dates), reports/icde/map_tile_audit.csv (v1 vs v2 way counts per needed tile).
Restartable: finished (extract, chunk) pairs are logged in tiles_v2/_progress.jsonl; a re-run skips them (merging the
same chunk twice would also be harmless, the merge de-duplicates by key).

    python scripts/recut_tiles_v2.py [--folds val test few_shot_heldout] [--max-tiles-per-pass 150] [--dry-run]
"""
import argparse
import gc
import json
import os
import sys
import time

import pandas as pd
import pyarrow.parquet as pq
from shapely.geometry import shape
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths, registry  # noqa: E402
from cantraj.maps.osm import TILE_DEG, extract, tiles_in_radius  # noqa: E402
import fetch_maps_all as fm  # noqa: E402
from plan_episode_tiles import SCHEMAS  # noqa: E402

MAP_CAP_M = 60000.0
TILES_V2 = os.path.join(paths.BENCH, "maps", "tiles_v2")
MAN_V2 = os.path.join(paths.MANIFESTS, "map_tiles_v2.csv")
AUDIT = os.path.join(paths.GPS_DIR, "reports", "icde", "map_tile_audit.csv")
PROGRESS = os.path.join(TILES_V2, "_progress.jsonl")


def tname(t):
    return f"t{t[0]}_{t[1]}"


def n_rows(path):
    return pq.ParquetFile(path).metadata.num_rows if os.path.exists(path) else -1


def needed_tiles(folds):
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    man = man[man.eligible_main & man.fold.isin(folds)]
    p0 = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_p0.parquet")).merge(man[["episode_id", "fold", "win"]], on="episode_id")
    assert len(p0) == len(man), (len(p0), len(man))
    spec = registry.protocol_spec("O-M-Hp")
    need = {}
    for r in p0.itertuples(index=False):
        for t in tiles_in_radius(r.lat0, r.lon0, min(registry.map_radius_m(spec, r.dur_s), MAP_CAP_M)):
            need[t] = need.get(t, 0) + 1
    return need, len(p0)


def write_empty(t, source):
    d = os.path.join(TILES_V2, tname(t))
    os.makedirs(d, exist_ok=True)
    for name, schema in SCHEMAS.items():
        pq.write_table(schema.empty_table(), os.path.join(d, f"{name}.parquet"))
    return dict(tile=tname(t), source=source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["val", "test", "few_shot_heldout"])
    ap.add_argument("--max-tiles-per-pass", type=int, default=150)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    need, n_ep = needed_tiles(args.folds)
    print(f"{n_ep} eligible episodes in {args.folds}: {len(need)} needed tiles", flush=True)
    polys = fm.polygons()
    cj = json.load(open(fm.COUNTRIES))
    land = unary_union([shape(f["geometry"]) for f in cj["features"]]).buffer(0.0)
    by_ext, sea, unmapped = {}, [], []
    for t in sorted(need):
        b = fm.tile_box(*t)
        hits = sorted(k for k, g in polys.items() if g.intersects(b))
        if hits:
            for k in hits:
                by_ext.setdefault(k, []).append(t)
        elif land.intersects(b):
            unmapped.append(t)
        else:
            sea.append(t)
    print(f"tile-extract pairs {sum(len(v) for v in by_ext.values())} over {len(by_ext)} extracts; sea {len(sea)}; unconfigured land {len(unmapped)}", flush=True)
    for k, v in sorted(by_ext.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"  {k}: {len(v)} tiles", flush=True)
    if args.dry_run:
        return
    os.makedirs(TILES_V2, exist_ok=True)
    done = set()
    if os.path.exists(PROGRESS):
        for line in open(PROGRESS):
            d = json.loads(line)
            done.add((d["extract"], d["chunk"]))
    src_rows = []
    for k in sorted(by_ext, key=lambda kk: -len(by_ext[kk])):
        tiles_k = sorted(by_ext[k])
        pbf = fm.download(k)                      # local file already present for every configured extract
        info = json.load(open(pbf + ".json"))
        for c0 in range(0, len(tiles_k), args.max_tiles_per_pass):
            chunk = tiles_k[c0:c0 + args.max_tiles_per_pass]
            key = (k, c0 // args.max_tiles_per_pass)
            for t in chunk:
                src_rows.append(dict(tile=tname(t), source=f"geofabrik:{k}", pbf_md5=info["md5"], snapshot_date=info.get("fetched", "")))
            if key in done:
                continue
            t0 = time.time()
            bbox = (min(t[0] for t in chunk) * TILE_DEG, min(t[1] for t in chunk) * TILE_DEG,
                    (max(t[0] for t in chunk) + 1) * TILE_DEG, (max(t[1] for t in chunk) + 1) * TILE_DEG)
            extract(pbf, bbox, TILES_V2, source_name=f"geofabrik:{k}", snapshot_date=info.get("fetched", ""), tiles=set(chunk), merge=True)
            gc.collect()
            with open(PROGRESS, "a") as f:
                f.write(json.dumps(dict(extract=k, chunk=key[1], n_tiles=len(chunk), secs=round(time.time() - t0, 1))) + "\n")
            print(f"  {k} chunk {key[1]}: {len(chunk)} tiles in {time.time() - t0:.0f}s (elapsed {time.time() - t_start:.0f}s)", flush=True)
    for t in sea:
        src_rows.append(write_empty(t, "empty:sea") | dict(pbf_md5="", snapshot_date=""))
    # land tiles outside every configured extract: v1 inspected them (Great Lakes / coastal water inside the country
    # polygons) and wrote them empty; keep that decision, anything else stays missing (map_missing for its episodes)
    v1src = pd.read_csv(fm.TILE_MANIFEST).groupby("tile").source.agg(lambda s: ";".join(sorted(set(map(str, s)))))
    for t in unmapped:
        if "empty:water" in v1src.get(tname(t), ""):
            src_rows.append(write_empty(t, "empty:water") | dict(pbf_md5="", snapshot_date=""))
    # extract-covered tiles in which no extract had a single drivable way: empty, recorded as such
    for t in sorted(need):
        if t in sea or t in unmapped:
            continue
        if not os.path.exists(os.path.join(TILES_V2, tname(t), "ways.parquet")):
            write_empty(t, "empty:no-roads")
            src_rows.append(dict(tile=tname(t), source="empty:no-roads", pbf_md5="", snapshot_date=""))
    src = pd.DataFrame(src_rows).groupby("tile").agg(source=("source", lambda s: ";".join(sorted(set(s)))),
                                                    pbf_md5=("pbf_md5", lambda s: ";".join(sorted(set(x for x in s if x)))),
                                                    snapshot_date=("snapshot_date", lambda s: ";".join(sorted(set(x for x in s if x)))))
    rows = []
    v1 = pd.read_csv(fm.TILE_MANIFEST).groupby("tile").source.agg(lambda s: ";".join(sorted(set(map(str, s)))))
    for t in sorted(need):
        n = tname(t)
        d2, d1 = os.path.join(TILES_V2, n), os.path.join(fm.TILES, n)
        rows.append(dict(tile=n, lat_min=t[0] * TILE_DEG, lon_min=t[1] * TILE_DEG, n_episodes=need[t],
                         n_ways_v1=n_rows(os.path.join(d1, "ways.parquet")), n_ways_v2=n_rows(os.path.join(d2, "ways.parquet")),
                         n_nodes_v2=n_rows(os.path.join(d2, "nodes.parquet")), n_restrictions_v2=n_rows(os.path.join(d2, "restrictions.parquet")),
                         kind="unmapped_land" if t in unmapped else ("sea" if t in sea else "extract"),
                         source_v1=v1.get(n, ""), source_v2=src.source.get(n, ""), pbf_md5=src.pbf_md5.get(n, ""), snapshot_date=src.snapshot_date.get(n, "")))
    out = pd.DataFrame(rows)
    out[["tile", "lat_min", "lon_min", "kind", "source_v2", "snapshot_date", "pbf_md5", "n_ways_v2", "n_nodes_v2", "n_restrictions_v2"]].rename(
        columns={"source_v2": "source", "n_ways_v2": "n_ways", "n_nodes_v2": "n_nodes", "n_restrictions_v2": "n_restrictions"}).to_csv(MAN_V2, index=False)
    os.makedirs(os.path.dirname(AUDIT), exist_ok=True)
    out["changed"] = out.n_ways_v1 != out.n_ways_v2
    out["v1_truncated"] = (out.n_ways_v2 > 0) & (out.n_ways_v1 < 0.9 * out.n_ways_v2)
    out.drop(columns=["lat_min", "lon_min"]).to_csv(AUDIT, index=False)
    print(f"tiles {len(out)}: changed {int(out.changed.sum())}, v1 truncated (< 90% of v2 ways) {int(out.v1_truncated.sum())}, "
          f"episode-weighted truncated share {out[out.v1_truncated].n_episodes.sum() / max(out.n_episodes.sum(), 1):.3f}; "
          f"unmapped land {len(unmapped)}, sea {len(sea)}; {time.time() - t_start:.0f}s", flush=True)
    print("written", MAN_V2, AUDIT, flush=True)


if __name__ == "__main__":
    main()
