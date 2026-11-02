#!/usr/bin/env python3
"""patch_card_v11.py: write the DriveDNA-GPS card with the benchmark version 1.1 section, from the v1 card.

  python scripts/patch_card_v11.py --out <new README.md> [--src <v1 card>] [--root <staging tree>] [--date YYYY-MM-DD]

The v1 card (`--src`, by default the staging README, identical to the Hub copy until the next upload) is never edited in
place. Every number of the new text is computed here from the files it describes (main board, map-episode manifest, tile
list and audit, staging sizes), so run it after the version 1.1 files are staged, or the sizes are those of the v1 tree.
Changes: the benchmark section (versions, methods incl. the six re-implemented published methods, a version 1.1 results
table, metrics, reproduction), the TL;DR bullet, the layout line, two front-matter configs for the version 1.1 files, the
code link of the events section (it pointed at the DriveDNA repository), the quick-start download (sizes, a command that
works with huggingface_hub 0.x and 1.x), the u-blox count (995 counted the 3,994 files; of the 3,988 usable tracks 990
are u-blox), "no absolute time" now says "of a drive" (the tile lists carry OpenStreetMap snapshot dates) and the footer.
The version 1.1 scores are published as `scores/v1_1/` (the code calls the run `icde_v11`). Upload stays a separate,
explicitly approved step (upload_benchmark_release.py readme).
"""
import argparse
import datetime
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_benchmark_release import gate_text  # noqa: E402

GPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGING = "/data/backup/DriveDNA-GPS_hf"
CODE = "https://github.com/WangYuHang-cmd/GPS--CAN"
RUN = "v1_1"                                    # published folder name of run icde_v11
OLD_MARK = "## 🧭 CAN-to-GPS benchmark v1 (`benchmark/`)"
NEW_MARK = "## 🧭 CAN-to-GPS benchmark (`benchmark/`), version 1.1"
NEXT = "## 🔗 Relation to DriveDNA and DriveDNA-Controlled"
MAIN = ("test", "few_shot_heldout")

# table rows: main-board key, method, signals, map, start heading; * = re-implemented published method
ROWS = [("beam", "beam search (offline)", "speed + steering", "yes", "given"),
        ("beam_hm", "beam search (offline)", "speed + steering", "yes", "unknown"),
        ("pf", "particle filter (online)", "speed + steering", "yes", "given"),
        ("pf_hm", "particle filter (online)", "speed + steering", "yes", "unknown"),
        ("opd2", "OPD-II* (online, 3 s delay)", "speed + steering", "yes", "given"),
        ("macro", "MacroTracking* (online)", "speed + steering", "yes", "given"),
        ("spy_hp", "SPy* (offline)", "steering", "yes", "given"),
        ("spy", "SPy* (offline)", "steering", "yes", "unknown"),
        ("ep_hp", "Elastic Pathing* (offline)", "speed", "yes", "given"),
        ("ep", "Elastic Pathing* (offline)", "speed", "yes", "unknown"),
        ("dewri_hp", "Dewri et al.* (offline)", "speed", "yes", "given"),
        ("dewri", "Dewri et al.* (offline)", "speed", "yes", "unknown"),
        ("ct", "CAN-Trace* (offline)", "speed + accelerator pedal", "yes", "unknown"),
        ("straight", "straight line (online)", "speed", "no", "given"),
        ("dr", "dead reckoning (online)", "speed + steering", "no", "given"),
        ("gru", "learned GRU, 3 seeds (online)", "speed + steering", "no", "given")]


def board_rows(board, win):
    b = board[(board.win == win) & board.fold.isin(MAIN)]
    return {(r.key, r.fold): r for r in b.itertuples()}


def pair(rows, key, fmt):
    return " / ".join(fmt(rows[(key, f)]) for f in MAIN)


def med(r):
    return "∞" if r.ade_median == float("inf") else f"{r.ade_median:,.0f}"


def nofail(r):
    return f"{int(r.n_fail):,}" if r.key != "ct" else f"{int(r.n_fail):,} of {int(r.n_expected):,}"


def results_table(board):
    w05 = board_rows(board, "w05")
    lines = ["| method | signals | map | start heading | median ADE, m | route recovered, % | windows without output |",
             "|---|---|---|---|---|---|---|"]
    for key, name, sig, has_map, head in ROWS:
        lines.append(f"| {name} | {sig} | {has_map} | {head} | {pair(w05, key, med)} | "
                     f"{pair(w05, key, lambda r: f'{100 * r.route_ok:.0f}')} | {pair(w05, key, nofail)} |")
    n = {k: pair(w05, k, lambda r: f"{int(r.n_expected):,}") for k in ("straight", "pf", "ct")}
    w15, full = board_rows(board, "w15"), board_rows(board, "full")
    tail = [100 * full[(k, f)].tail_lost100 for k in ("pf", "beam") for f in MAIN]
    return "\n".join(lines), n, pair(w15, "beam", med), (min(tail), max(tail))


def map_facts(map_eps, tiles, audit):
    m = map_eps[map_eps.fold.isin(MAIN) & map_eps.win.isin(["w05", "w15", "full"])]
    fail = m[~m.map_ok.astype(bool)]
    ext = tiles[tiles.kind == "extract"]
    dates = sorted(d for s in ext.snapshot_date.dropna() for d in str(s).split(";"))
    return dict(n_main=len(m), n_fail=len(fail), fail_folds=sorted(fail.fold.unique()), n_tiles=len(tiles),
                n_extract=len(ext), n_empty=len(tiles) - len(ext), n_trunc=int(audit.v1_truncated.sum()),
                d0=dates[0], d1=dates[-1])


def section(table, n, b15, tail, mf):
    only = (" (all in few_shot_heldout)" if mf["fail_folds"] == ["few_shot_heldout"] else "")
    return f'''{NEW_MARK}

Can the GPS trajectory of a drive be reconstructed from the vehicle-bus signals alone, once a single start position is
known? The `benchmark/` folder holds everything that fixes this task on DriveDNA-GPS; the code (package `cantraj`,
baselines, re-implemented published methods, scoring, tables and figures) is at [{CODE}]({CODE}).

- **Task**: given the CAN events of a drive (and optionally the device IMU) and the position at one instant t0,
  predict the position at every later reference second. Eight protocols = {{online, offline}} x {{no map, OpenStreetMap
  road graph}} x {{start heading given, start heading unknown}}; windows of 1, 5 and 15 minutes and whole drives;
  signal groups from speed only (G1) and speed + steering angle (G3, the CAN-only main board) to full CAN + IMU (G8).
- **Episodes**: 98,972 eligible episodes from 3,675 drives and 431 drivers (`manifests/episode_manifest_v1.parquet`;
  anchor = a moving GPS fix with good course and accuracy; eligibility = at least 80% of seconds with a fix, no gap
  over 30 s, path over 100 m). Driver-disjoint folds train / val / test / few_shot_heldout (`manifests/split_drives.parquet`).
  Map protocols use only the episodes whose start has a drivable road within 400 m in the episode's map (`map_ok` in
  `manifests/episode_map_v11.parquet`): {mf["n_fail"]} of the {mf["n_main"]:,} test and few-shot episodes of 5 and 15 minutes
  and whole drives{only} fail this rule; they are left out of every map protocol and stay in the no-map protocols.
- **Reference**: the 1 Hz GPS view of this repository, with a per-receiver time compensation estimated on the training fold.
- **Metrics**: ADE (mean error over the reference seconds) and FDE; coverage, the share of seconds within 10, 25, 50,
  100, 200 or 500 m of the reference; success S(50 m, 0.9), at least 90% of the seconds within 50 m; path recall, the
  share of the reference path within 50 m of the predicted path regardless of timing; route recovered, a path recall of at
  least 0.8; and two failure flags, `tail_lost100` (the error stays above 100 m over the whole last fifth of the window)
  and `terminal_lost100` (the error at the last reference second is above 100 m). A window without output counts as a
  failure (infinite ADE, route not recovered). `scores/{RUN}/episodes_all.parquet` has one row per window with output;
  `scores/{RUN}/main_board.csv` lists, per method, the expected windows and the failures. Medians with driver-cluster
  bootstrap CIs; paired differences on common episodes.
- **Methods**: dead reckoning (nominal, calibrated, learned GRU and physics-residual models), a particle filter and an
  offline beam search on the road graph, and re-implementations of six published methods, each run under the protocol
  that matches its own assumptions and scored with the same metrics: MacroTracking, OPD-II, Elastic Pathing, CAN-Trace,
  Dewri et al. (2013) and SPy / RoCuMa. These are our implementations, not the authors' code. MacroTracking and OPD-II
  follow the algorithms of the authors' released code (the code where it differs from the paper) and were checked
  against it on the authors' sample data (one sample trace; 8 of the 59 sample logs), which checks the transcription, not
  that the papers' results carry over. Elastic Pathing, CAN-Trace, Dewri et al. and SPy follow the papers; their authors
  released no code or driving data. Every adaptation is documented in the code repository.
- **Results** (version 1.1; 5-minute windows, start position known; test / few_shot_heldout, {n["straight"]} windows,
  {n["pf"]} in the map protocols, CAN-Trace only on the {n["ct"]} windows with a continuous accelerator-pedal signal;
  a window without output counts as an infinite error; * = re-implemented published method):

{table}

  With the start heading given, the offline beam search and the online particle filter stay within tens of metres
  (median) over five minutes; without a map the CAN-only methods are hundreds of metres off. SPy and CAN-Trace search an
  area rather than start from a known position in their papers; the table gives them the start, and their 5 km area
  variants, which do worse, are in the main board. On 15-minute windows the
  beam search's median ADE is {b15} m; over whole drives the filter and the beam search (start heading given) stay more
  than 100 m from the reference over the whole last fifth of {tail[0]:.0f} to {tail[1]:.0f}% of the drives. Every method and
  window class with CIs and failure counts: `scores/{RUN}/main_board.csv`.
- **Versions**: version 1.1 (2026-10-28) changes the maps and the failure accounting. All {mf["n_tiles"]:,} tiles that the
  validation and evaluation folds use were rebuilt: {mf["n_extract"]:,} re-cut from every intersecting OpenStreetMap
  extract ({mf["n_trunc"]} of them had been cut incompletely at the margins of early tiling passes in v1) and {mf["n_empty"]}
  open-sea or unmapped land tiles left empty; `manifests/map_tiles_v2.csv` lists each tile with its source extract(s)
  and, for extract tiles, the extract md5 and snapshot date ({mf["d0"]} to {mf["d1"]}). New are the `map_ok` start rule,
  a map limited to each episode's own disc (5 km + 40 m/s x window length around the start, at most 60 km) with road
  bearings in the episode's local frame, the fix of a v1 timing error in feeding the learned motion models inside the
  filter (each message now arrives at its logged time), the two failure flags above (`tail_lost100` is v1's `lost_road`,
  `terminal_lost100` is new), and a window without output counts as a failure (v1 scored only windows with output).
  Episodes, folds, reference and thresholds are identical to v1, and so are the no-map predictions and their scores (no
  no-map method has a window without output). The v1 test results were known when these changes were made; every test
  run since is logged with its reason in `scores/{RUN}/run_registry.csv`. The v1 scores stay in `scores/full_v1/` as
  history; their map rows should not be used (incomplete tiles, and starts without a road nearby fell back to an
  arbitrary edge). Version 1.1 reruns the map methods on the test and few-shot folds for 5- and 15-minute windows and
  whole drives; the other v1 map rows (1-minute windows, the validation fold, the calibrated, IMU and start-perturbation
  variants) have no version 1.1 counterpart.
- **Why gated**: `manifests/episode_p0.parquet` holds the episode start coordinates, which with the start heading (the
  GPS course at t0, given only in the heading-known protocols) are the only GPS information a method receives, and
  every manifest references drives by their DriveDNA pseudonyms. No raw log path, device or route identifier appears in
  any file (checked at staging; `benchmark/CHECKSUMS.sha256` lists every file).
- **Reproduce**: clone the code, point `CANTRAJ_GPS` at this repository, `CANTRAJ_MANIFESTS` at `benchmark/manifests`
  and `CANTRAJ_BENCH` at a cache with the `events/` tables, then follow the version 1.1 steps in the README of the code
  repository. Map-protocol numbers reproduce exactly only with the same road tiles: tiles re-cut from current
  OpenStreetMap extracts differ slightly from the {mf["d0"]} to {mf["d1"]} snapshots (the md5 column of `map_tiles_v2.csv`
  shows whether an extract matches). `scores/{RUN}/run_registry.csv` records the source, scorer and map version of every
  version 1.1 run; the no-map runs reused from v1 are marked as such. Prediction files (about 20 GB of per-second
  positions) are not in this repository.

'''


TLDR_OLD = ("- **CAN-to-GPS benchmark**: `benchmark/` fixes the trajectory-recovery task on these drives (98,972 episodes, eight protocols, "
            f"frozen manifests, per-episode scores of every baseline); code at [{CODE}]({CODE}).\n")
TLDR_NEW = ("- **CAN-to-GPS benchmark**: `benchmark/` fixes the trajectory-recovery task on these drives (98,972 episodes, eight protocols, "
            "version 1.1, per-episode scores of every baseline and of six re-implemented published methods); "
            f"code at [{CODE}]({CODE}).\n")
FIXES = [  # (what, old, new): each old text must occur exactly once in the v1 card
    ("layout line",
     "├── benchmark/                    CAN-to-GPS benchmark v1: manifests/, config/, scores/full_v1/, CHECKSUMS.sha256\n",
     f"├── benchmark/                    CAN-to-GPS benchmark: manifests/, config/, scores/{RUN}/ (version 1.1), "
     "scores/full_v1/ (v1, history), CHECKSUMS.sha256\n"),
    ("front-matter configs",
     '  - config_name: benchmark_scores\n    data_files: "benchmark/scores/full_v1/episodes_all.parquet"\n',
     '  - config_name: benchmark_scores\n    data_files: "benchmark/scores/full_v1/episodes_all.parquet"\n'
     f'  - config_name: benchmark_scores_v11\n    data_files: "benchmark/scores/{RUN}/episodes_all.parquet"\n'
     '  - config_name: benchmark_map_episodes\n    data_files: "benchmark/manifests/episode_map_v11.parquet"\n'),
    ("events code link",
     "(code and manifests on [GitHub](https://github.com/WangYuHang-cmd/DriveDNA))",
     f"(code at [GPS--CAN]({CODE}), manifests in `benchmark/`)"),
    ("u-blox count", "995 were logged by a u-blox receiver", "990 were logged by a u-blox receiver"),
    ("TL;DR absolute time",
     "- **No absolute time**: no date, wall clock or receiver timestamp appears in any file.",
     "- **No absolute time**: no date, wall clock or receiver timestamp of a drive appears in any file (the benchmark's map "
     "tile lists carry only the OpenStreetMap snapshot dates)."),
    ("privacy absolute time",
     "- No absolute time is released: no date, wall clock or receiver timestamp.",
     "- No absolute time of a drive is released: no date, wall clock or receiver timestamp."),
    ("datasets config list",
     "`events_calib`, `events_manifest`).",
     "`events_calib`, `events_manifest`, `benchmark_episodes`, `benchmark_scores` (v1; its map rows should not be used), "
     "`benchmark_scores_v11` (version 1.1), `benchmark_map_episodes`)."),
]


def tree_bytes(*paths):
    out = subprocess.run(["du", "-scb", "--exclude=.cache", *paths], capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[-1].split()[0])


def replace_once(txt, old, new, what):
    n = txt.count(old)
    if n != 1:
        raise SystemExit(f"{what}: expected the old text once, found it {n} times")
    return txt.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(STAGING, "README.md"), help="the v1 card (not modified)")
    ap.add_argument("--out", required=True, help="where to write the new card")
    ap.add_argument("--root", default=STAGING, help="staging tree whose sizes the card quotes")
    ap.add_argument("--board", default=os.path.join(GPS_DIR, "reports/icde/tables/main_board.csv"))
    ap.add_argument("--map-episodes", default=os.path.join(GPS_DIR, "manifests/episode_map_v11.parquet"))
    ap.add_argument("--tiles", default=os.path.join(GPS_DIR, "manifests/map_tiles_v2.csv"))
    ap.add_argument("--audit", default=os.path.join(GPS_DIR, "reports/icde/map_tile_audit.csv"))
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--validate", action="store_true", help="also run DatasetCard.validate (sends the card to the Hub API)")
    args = ap.parse_args()
    if os.path.abspath(args.out) == os.path.abspath(args.src):
        raise SystemExit("--out must differ from --src: the v1 card is never edited in place")

    txt = open(args.src, encoding="utf-8").read()
    if NEW_MARK in txt or OLD_MARK not in txt:
        raise SystemExit("--src is not the v1 card")
    board = pd.read_csv(args.board)
    table, n, b15, tail = results_table(board)
    mf = map_facts(pd.read_parquet(args.map_episodes, columns=["fold", "win", "map_ok"]), pd.read_csv(args.tiles),
                   pd.read_csv(args.audit))
    i, j = txt.index(OLD_MARK), txt.index(NEXT)
    assert txt.count(OLD_MARK) == 1 and i < j
    txt = txt[:i] + section(table, n, b15, tail, mf) + txt[j:]
    txt = replace_once(txt, TLDR_OLD, TLDR_NEW, "TL;DR bullet")
    for what, old, new in FIXES:
        txt = replace_once(txt, old, new, what)
    txt = replace_once(txt, "card generated 2026-10-18</sub>", f"card generated 2026-10-18, updated {args.date}</sub>", "footer")

    total = tree_bytes(args.root) / 1e9
    csvs = tree_bytes(*[os.path.join(args.root, d) for d in sorted(os.listdir(args.root))
                        if d.startswith("driver_") or d == "manifest.csv"]) / 1e9
    bench = tree_bytes(os.path.join(args.root, "benchmark")) / 1e9
    txt = replace_once(txt, "--local-dir DriveDNA-GPS      # whole set, 4.46 GB\n",
                       f"--local-dir DriveDNA-GPS      # everything, {total:.1f} GB\n"
                       "hf download HenryYHW/DriveDNA-GPS manifest.csv \"driver_*/*\" --repo-type dataset --local-dir DriveDNA-GPS"
                       f"   # CSV files only (GPS, localizer, GNSS, manifest), {csvs:.2f} GB\n", "quick-start download")
    txt = replace_once(txt, "plus the event tables under `events/` (24.0 GB of parquet).",
                       f"plus the event tables under `events/` (24.0 GB of parquet) and the benchmark files under `benchmark/` ({bench:.2f} GB).",
                       "release size sentence")

    if gate_text(txt, args.out):
        raise SystemExit("identifier gate hit: card not written")
    import yaml
    names = [c["config_name"] for c in yaml.safe_load(txt.split("---\n")[1])["configs"]]
    assert len(names) == len(set(names)), names
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"card written: {args.out} ({len(names)} configs; repository {total:.1f} GB, CSV files {csvs:.2f} GB, benchmark/ {bench:.2f} GB)")
    if not os.path.isdir(os.path.join(args.root, "benchmark", "scores", RUN)):
        print(f"NOTE: {args.root}/benchmark/scores/{RUN} is not staged yet: sizes are those of the v1 tree and the card must not be uploaded")
    if args.validate:
        from huggingface_hub import DatasetCard
        DatasetCard.load(args.out).validate(repo_type="dataset")
        print("DatasetCard.validate: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
