#!/usr/bin/env python3
"""fetch_maps.py: download Geofabrik extracts and cut them into 0.25 deg tiles for the benchmark map protocols.

    python scripts/fetch_maps.py download north-america/us/utah
    python scripts/fetch_maps.py tile utah --bbox 40.0 -112.6 41.3 -111.0
Records source, snapshot date and md5 in manifests/map_tiles.csv (appended, de-duplicated by tile).
"""
import argparse
import hashlib
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths  # noqa: E402
from cantraj.maps.osm import extract  # noqa: E402

PBF = f"{paths.BENCH}/maps/pbf"
TILES = f"{paths.BENCH}/maps/tiles"
MANIFEST = os.path.join(paths.MANIFESTS, "map_tiles.csv")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "tile"])
    ap.add_argument("region")
    ap.add_argument("--bbox", nargs=4, type=float, help="lat_min lon_min lat_max lon_max")
    args = ap.parse_args()
    os.makedirs(PBF, exist_ok=True)
    name = args.region.split("/")[-1]
    pbf = os.path.join(PBF, f"{name}-latest.osm.pbf")
    if args.cmd == "download":
        url = f"https://download.geofabrik.de/{args.region}-latest.osm.pbf"
        subprocess.check_call(["curl", "-sL", "-m", "3600", "-o", pbf, url])
        info = dict(region=args.region, bytes=os.path.getsize(pbf), md5=md5(pbf), fetched=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
        with open(pbf + ".json", "w") as f:
            import json
            json.dump(info, f, indent=1)
        print(info)
    else:
        import json
        info = json.load(open(pbf + ".json")) if os.path.exists(pbf + ".json") else {}
        man = extract(pbf, tuple(args.bbox), TILES, source_name=f"geofabrik:{args.region}", snapshot_date=info.get("fetched", ""))
        man["pbf_md5"] = info.get("md5", "")
        if os.path.exists(MANIFEST):
            old = pd.read_csv(MANIFEST)
            man = pd.concat([old[~old.tile.isin(man.tile)], man], ignore_index=True)
        man.to_csv(MANIFEST, index=False)
        print(f"tiles total in manifest: {len(man)}; written {MANIFEST}")


if __name__ == "__main__":
    main()
