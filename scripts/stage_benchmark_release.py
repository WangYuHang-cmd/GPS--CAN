"""Stage the benchmark v1 manifests and per-episode scores for the gated DriveDNA-GPS repository (`benchmark/` folder).

  python scripts/stage_benchmark_release.py [--run full_v1] [--out /data/backup/DriveDNA-GPS_hf/benchmark]

Files: frozen manifests (episode manifest, splits, availability, vehicle parameters per platform, calibration fallback,
time alignment, map tiles with md5, episode start points, map coverage, G4 / G8 drive lists), the frozen configuration
(yaml + spec), the score tables and the per-episode metric table of the run, a README. Everything is gated: episode start
coordinates are GPS information and belong only here. Gate: no raw log paths, device or route identifiers in any text
column or file name. Prediction parquet files are not staged (tens of GB; available on request).
"""
from __future__ import annotations

import argparse
import codecs
import hashlib
import os
import re
import shutil
import sys

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from cantraj import paths  # noqa: E402

GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFESTS = ["episode_manifest_v1.parquet", "split_drives.parquet", "availability_wide.csv", "vehicle_params.csv", "drive_platform.csv",
             "calibration_global.json", "time_alignment.json", "map_tiles.csv", "episode_p0.parquet", "map_coverage_episodes_v1.csv",
             "drives_G4.csv", "drives_G8.csv", "native_manifest.csv"]
GATE = [re.compile(r"/data/|/home/|/mnt/"), re.compile(r"(?<![0-9a-zA-Z.])(?=[0-9]*[a-f])[0-9a-f]{16}(?![0-9a-zA-Z])"),
        re.compile(r"\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}"), re.compile(r"\b" + codecs.decode("Gnzcn", "rot13") + r"\b")]


def gate_text(txt, where):
    rules = GATE if "benchmark_v1." not in os.path.basename(where) else [r for r in GATE if "[0-9a-f]{16}" not in r.pattern]  # sha256 prefixes
    hits = [(rx.pattern, rx.search(txt).group(0)) for rx in rules if rx.search(txt)]
    for pat, frag in hits:
        print(f"GATE HIT {where}: {pat} -> {frag!r}")
    return len(hits)


def gate_file(p):
    if p.endswith(".parquet"):
        t = pq.read_table(p)
        n = gate_text(" ".join(t.column_names), p + " (columns)")
        for name in t.column_names:
            col = t.column(name)
            if str(col.type) in ("string", "large_string", "dictionary<values=string, indices=int32, ordered=0>", "dictionary<values=string, indices=int8, ordered=0>",
                                 "dictionary<values=string, indices=int16, ordered=0>") or "string" in str(col.type):
                vals = col.unique().to_pylist()
                n += gate_text("\n".join(str(v) for v in vals if v is not None), f"{p} [{name}]")
        return n
    with open(p, encoding="utf-8", errors="replace") as f:
        return gate_text(f.read(), p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="full_v1")
    ap.add_argument("--out", default="/data/backup/DriveDNA-GPS_hf/benchmark")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "manifests"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "scores", args.run), exist_ok=True)
    os.makedirs(os.path.join(args.out, "config"), exist_ok=True)
    staged = []
    for name in MANIFESTS:
        src = os.path.join(paths.MANIFESTS, name)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, "manifests", name)); staged.append(os.path.join(args.out, "manifests", name))
        else:
            print("missing manifest:", name)
    for rel in ("configs/benchmark_v1.yaml", "specs/benchmark_v1.md", "configs/signal_groups.yaml", "configs/protocols.yaml", "configs/signal_registry.yaml", "configs/windows.yaml"):
        shutil.copy2(os.path.join(GPS_DIR, rel), os.path.join(args.out, "config", os.path.basename(rel))); staged.append(os.path.join(args.out, "config", os.path.basename(rel)))
    sdir = os.path.join(paths.SCORES, args.run)
    for name in sorted(os.listdir(os.path.join(sdir, "tables"))):
        shutil.copy2(os.path.join(sdir, "tables", name), os.path.join(args.out, "scores", args.run, name)); staged.append(os.path.join(args.out, "scores", args.run, name))
    for name in ("episodes_all.parquet", "summary.csv"):
        if os.path.exists(os.path.join(sdir, name)):
            shutil.copy2(os.path.join(sdir, name), os.path.join(args.out, "scores", args.run, name)); staged.append(os.path.join(args.out, "scores", args.run, name))
    for name in sorted(os.listdir(os.path.join(GPS_DIR, "reports", "paper", "tables"))):
        d = os.path.join(args.out, "scores", args.run, "paper_tables"); os.makedirs(d, exist_ok=True)
        shutil.copy2(os.path.join(GPS_DIR, "reports", "paper", "tables", name), os.path.join(d, name)); staged.append(os.path.join(d, name))
    hits = sum(gate_file(p) for p in staged)
    sha = {os.path.relpath(p, args.out): hashlib.sha256(open(p, "rb").read()).hexdigest() for p in staged}
    with open(os.path.join(args.out, "CHECKSUMS.sha256"), "w") as f:
        for k, v in sorted(sha.items()):
            f.write(f"{v}  {k}\n")
    readme = f"""# CAN-to-GPS benchmark v1 (companion to DriveDNA-GPS)

Manifests, frozen configuration and per-episode scores of the CAN-to-GPS trajectory recovery benchmark (code:
https://github.com/WangYuHang-cmd/GPS--CAN). Everything here is gated because `manifests/episode_p0.parquet` holds the
episode start coordinates (the only GPS information a method receives) and the manifests reference drives by their
DriveDNA pseudonyms.

| path | content |
|---|---|
| `manifests/episode_manifest_v1.parquet` | every episode (98,972 eligible of 110,042): ids, window, t0 and t_end on the drive's time axis, anchor quality, eligibility, fold, platform label as logged |
| `manifests/split_drives.parquet` | driver-disjoint main folds (train / val / test / few_shot_heldout) and the auxiliary splits |
| `manifests/availability_wide.csv` | per-drive signal availability (valid / unsupported / undetermined) from decoder evidence |
| `manifests/vehicle_params.csv`, `drive_platform.csv` | nominal wheelbase / steering ratio per platform from the logged carParams (no VIN), platform per drive |
| `manifests/calibration_global.json`, `time_alignment.json` | training-fold calibration fallback and GPS-vs-CAN time compensation per receiver stratum |
| `manifests/map_tiles.csv`, `map_coverage_episodes_v1.csv` | OSM tiles (source extract, md5) and per-episode map coverage |
| `manifests/episode_p0.parquet` | episode start position (lat0, lon0) and the start fix delay: the only coordinates a method sees |
| `manifests/drives_G4.csv`, `drives_G8.csv`, `native_manifest.csv` | drive subsets of the CAN yaw-rate and IMU supplements; native event-table rates |
| `config/` | frozen benchmark_v1.yaml (thresholds, sha256 of the manifests), the human-readable spec, signal groups, protocols, registry, windows |
| `scores/{args.run}/episodes_all.parquet` | one row per (protocol, method, episode): ADE, FDE, P90, coverage, success flags, failure flags, Top-K oracle; no coordinates |
| `scores/{args.run}/headline.csv`, `gains.csv`, `strata.csv`, `summary.csv` | aggregated tables with driver-cluster bootstrap CIs; paired gains |
| `scores/{args.run}/paper_tables/` | the paper tables T1 to T8 |

Identifiers are DriveDNA pseudonyms (`driver_XXX/drive_YYY`); no raw log paths, device or route identifiers appear in
any file (checked at staging). Prediction parquet files (tens of GB) are available on request.
"""
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(readme)
    total = sum(os.path.getsize(p) for p in staged) / 1e6
    print(f"staged {len(staged)} files, {total:.0f} MB -> {args.out}; gate hits: {hits}")
    if hits:
        sys.exit(1)


if __name__ == "__main__":
    main()
