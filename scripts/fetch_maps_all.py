#!/usr/bin/env python3
"""fetch_maps_all.py: plan, download and tile Geofabrik extracts for every drive region; check episode coverage.

    python scripts/fetch_maps_all.py plan                       # manifests/map_regions.csv (extract per drive region, counts)
    python scripts/fetch_maps_all.py run --phase val-first      # download + tile extracts needed by val drives first
    python scripts/fetch_maps_all.py run --phase usca            # remaining US / Canada extracts (incl. border neighbours)
    python scripts/fetch_maps_all.py run --phase global
    python scripts/fetch_maps_all.py check --manifest manifests/episode_manifest_v1.parquet
Tiles needed = 0.25 deg tiles intersecting the 60 km disc around every drive median (and the episode start radius rule
at check time). A tile is cut from EVERY extract whose region polygon intersects it (border tiles merged), one extract
at a time, at most --max-tiles-per-pass tiles per pass to bound memory.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Point, box, shape
from shapely.ops import unary_union

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths, registry  # noqa: E402
from cantraj.maps.osm import TILE_DEG, extract, tiles_in_radius  # noqa: E402

PBF = f"{paths.BENCH}/maps/pbf"
TILES = f"{paths.BENCH}/maps/tiles"
TILE_MANIFEST = os.path.join(paths.MANIFESTS, "map_tiles.csv")
REGION_MANIFEST = os.path.join(paths.MANIFESTS, "map_regions.csv")
STATES = os.environ.get("CANTRAJ_NE_US_STATES_GEOJSON", "ne_us_states.geojson")   # Natural Earth polygons
COUNTRIES = os.environ.get("CANTRAJ_NE_COUNTRIES_GEOJSON", "ne_countries.geojson")   # Natural Earth polygons
CFG = yaml.safe_load(open(os.path.join(paths.GPS_DIR, "configs", "geofabrik_regions.yaml")))


def polygons():
    """{extract_path: shapely polygon} for every mapped state/province and country."""
    out = {}
    sj = json.load(open(STATES))
    for f in sj["features"]:
        admin = f["properties"].get("admin") or f["properties"].get("ADMIN")
        name = f["properties"].get("name") or f["properties"].get("NAME")
        path = CFG["states"].get(admin, {}).get(name)
        if path:
            out.setdefault(path, []).append(shape(f["geometry"]))
    cj = json.load(open(COUNTRIES))
    for f in cj["features"]:
        name = f["properties"].get("NAME") or f["properties"].get("name") or f["properties"].get("ADMIN")
        path = CFG["countries"].get(name)
        if not path or name in ("United States of America", "Canada"):
            continue
        for p in (path if isinstance(path, list) else [path]):
            out.setdefault(p, []).append(shape(f["geometry"]))
    return {k: unary_union(v).buffer(0.02) for k, v in out.items()}


def drive_regions(polys):
    q = pd.read_parquet(os.path.join(paths.MANIFESTS, "drive_quality.parquet"))
    rows = []
    for r in q.itertuples(index=False):
        p = Point(r.lon_med, r.lat_med)
        hits = [k for k, g in polys.items() if g.contains(p)]
        if not hits:   # offshore medians: nearest polygon
            hits = [min(polys, key=lambda k: polys[k].distance(p))]
        rows.append(dict(driver=r.driver, drive=r.drive, fold=r.fold, lat=r.lat_med, lon=r.lon_med, extracts=";".join(sorted(hits))))
    return pd.DataFrame(rows)


def tile_box(i, j):
    return box(j * TILE_DEG, i * TILE_DEG, (j + 1) * TILE_DEG, (i + 1) * TILE_DEG)


def plan(radius_m, polys=None):
    polys = polys or polygons()
    dr = drive_regions(polys)
    need = {}
    for r in dr.itertuples(index=False):
        for t in tiles_in_radius(r.lat, r.lon, radius_m):
            need.setdefault(t, set()).add(r.fold)
    # which extracts does each needed tile intersect
    tile_ext = {}
    for t in need:
        b = tile_box(*t)
        tile_ext[t] = [k for k, g in polys.items() if g.intersects(b)]
    rows = []
    for k in polys:
        tiles_k = [t for t, ex in tile_ext.items() if k in ex]
        n_val = sum(1 for t in tiles_k if "val" in need[t])
        n_drives = int(dr.extracts.str.contains(k, regex=False).sum())
        rows.append(dict(extract=k, n_drives=n_drives, n_tiles=len(tiles_k), n_tiles_val=n_val,
                         usca=k.startswith("north-america/us") or k.startswith("north-america/canada")))
    reg = pd.DataFrame(rows).sort_values(["usca", "n_tiles_val", "n_drives"], ascending=[False, False, False])
    return dr, need, tile_ext, reg


def have_tiles():
    return set(pd.read_csv(TILE_MANIFEST).tile) if os.path.exists(TILE_MANIFEST) else set()


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def download(extract_path, tries=3):
    name = extract_path.split("/")[-1]
    pbf = os.path.join(PBF, f"{name}-latest.osm.pbf")
    if os.path.exists(pbf) and os.path.exists(pbf + ".json"):
        return pbf
    url = f"https://download.geofabrik.de/{extract_path}-latest.osm.pbf"
    for k in range(tries):
        rc = subprocess.call(["curl", "-sL", "-m", "7200", "-C", "-", "-o", pbf, url])
        if rc == 0 and os.path.getsize(pbf) > 1_000_000:
            break
        time.sleep(10)
    else:
        raise RuntimeError(f"download failed: {url}")
    json.dump(dict(region=extract_path, bytes=os.path.getsize(pbf), md5=md5(pbf), fetched=time.strftime("%Y-%m-%d")), open(pbf + ".json", "w"), indent=1)
    return pbf


def tile_extract(extract_path, tiles_todo, max_per_pass, done_sources):
    """Cut the needed tiles from one extract in passes; returns manifest rows."""
    pbf = download(extract_path)
    info = json.load(open(pbf + ".json"))
    rows = []
    todo = sorted(t for t in tiles_todo if f"{extract_path}|t{t[0]}_{t[1]}" not in done_sources)
    for s in range(0, len(todo), max_per_pass):
        chunk = todo[s:s + max_per_pass]
        lat_min = min(t[0] for t in chunk) * TILE_DEG
        lat_max = (max(t[0] for t in chunk) + 1) * TILE_DEG
        lon_min = min(t[1] for t in chunk) * TILE_DEG
        lon_max = (max(t[1] for t in chunk) + 1) * TILE_DEG
        man = extract(pbf, (lat_min, lon_min, lat_max, lon_max), TILES, source_name=f"geofabrik:{extract_path}",
                      snapshot_date=info.get("fetched", ""), tiles=set(chunk), merge=True)
        got = set(man.tile) if len(man) else set()
        for t in chunk:                       # tiles with no ways in this extract are still recorded (empty for this source)
            name = f"t{t[0]}_{t[1]}"
            if name not in got:
                rows.append(dict(tile=name, lat_min=t[0] * TILE_DEG, lon_min=t[1] * TILE_DEG, lat_max=(t[0] + 1) * TILE_DEG, lon_max=(t[1] + 1) * TILE_DEG,
                                 source=f"geofabrik:{extract_path}", snapshot_date=info.get("fetched", ""), n_ways=0, n_nodes=0, n_restrictions=0, pbf_md5=info["md5"]))
        if len(man):
            man["pbf_md5"] = info["md5"]
            rows += man.to_dict("records")
        gc.collect()
    return rows


def update_tile_manifest(rows):
    new = pd.DataFrame(rows)
    if new.empty:
        return
    if os.path.exists(TILE_MANIFEST):
        old = pd.read_csv(TILE_MANIFEST)
        old["key"] = old.tile + "|" + old.source.astype(str)
        new["key"] = new.tile + "|" + new.source.astype(str)
        allm = pd.concat([old[~old.key.isin(new.key)], new], ignore_index=True).drop(columns=["key"])
    else:
        allm = new
    allm.to_csv(TILE_MANIFEST, index=False)


def cmd_run(args):
    polys = polygons()
    dr, need, tile_ext, reg = plan(args.radius_km * 1000, polys)
    reg.to_csv(REGION_MANIFEST, index=False)
    tm = pd.read_csv(TILE_MANIFEST) if os.path.exists(TILE_MANIFEST) else pd.DataFrame(columns=["tile", "source"])
    done_sources = set(tm.source.astype(str).str.replace("geofabrik:", "", regex=False) + "|" + tm.tile)
    if args.phase == "val-first":
        order = reg[reg.n_tiles_val > 0].sort_values("n_tiles_val", ascending=False).extract.tolist()
    elif args.phase == "usca":
        order = reg[reg.usca].extract.tolist()
    else:
        order = reg.extract.tolist()
    for k in order:
        tiles_k = {t for t, ex in tile_ext.items() if k in ex}
        if args.phase == "val-first":
            tiles_k = {t for t in tiles_k if "val" in need[t]}
        todo = sorted(t for t in tiles_k if f"{k}|t{t[0]}_{t[1]}" not in done_sources)
        if not todo:
            print(f"{k}: nothing to do", flush=True)
            continue
        t0 = time.time()
        print(f"{k}: {len(todo)} tiles to cut", flush=True)
        try:
            rows = tile_extract(k, todo, args.max_tiles_per_pass, done_sources)
        except Exception as e:  # noqa: BLE001
            print(f"{k}: FAILED {type(e).__name__}: {e}", flush=True)
            continue
        update_tile_manifest(rows)
        done_sources |= {f"{k}|{r['tile']}" for r in rows}
        print(f"{k}: done {len(rows)} tile records in {time.time() - t0:.0f}s", flush=True)


def cmd_check(args):
    man = pd.read_parquet(args.manifest)
    man = man[man.eligible_main] if "eligible_main" in man else man
    have = have_tiles()
    spec = registry.protocol_spec("O-M-Hp")
    rows = []
    # episode starts are not in the manifest; use the drive's episodes' anchors from the quality table (drive median) as proxy plus radius
    q = pd.read_parquet(os.path.join(paths.MANIFESTS, "drive_quality.parquet")).set_index(["driver", "drive"])
    for (d, dr_), g in man.groupby(["driver", "drive"]):
        lat, lon = q.loc[(d, dr_), "lat_med"], q.loc[(d, dr_), "lon_med"]
        for w, gg in g.groupby("win"):
            D = float(gg.dur_s.max())
            r = min(registry.map_radius_m(spec, D), 60000.0)
            need = {f"t{i}_{j}" for i, j in tiles_in_radius(lat, lon, r)}
            rows.append(dict(driver=d, drive=dr_, win=w, n_episodes=len(gg), n_tiles=len(need), n_missing=len(need - have)))
    cov = pd.DataFrame(rows)
    cov.to_csv(os.path.join(paths.MANIFESTS, "map_coverage_v1.csv"), index=False)
    print(cov.groupby("win").apply(lambda t: pd.Series(dict(drives=len(t), covered=int((t.n_missing == 0).sum()), episodes=int(t.n_episodes.sum()), episodes_covered=int(t.n_episodes[t.n_missing == 0].sum())))).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["plan", "run", "check"])
    ap.add_argument("--phase", default="val-first", choices=["val-first", "usca", "global"])
    ap.add_argument("--radius-km", type=float, default=60.0)
    ap.add_argument("--max-tiles-per-pass", type=int, default=150)
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"))
    args = ap.parse_args()
    if args.cmd == "plan":
        dr, need, tile_ext, reg = plan(args.radius_km * 1000)
        reg.to_csv(REGION_MANIFEST, index=False)
        print(reg.head(40).to_string(index=False))
        print(f"tiles needed: {len(need)}; present: {len(have_tiles())}; extracts: {len(reg)}; multi-extract tiles: {sum(1 for v in tile_ext.values() if len(v) > 1)}")
    elif args.cmd == "run":
        cmd_run(args)
    else:
        cmd_check(args)


if __name__ == "__main__":
    main()
