#!/usr/bin/env python3
"""patch_card_benchmark.py: add the CAN-to-GPS benchmark v1 (`benchmark/` folder) to the staged DriveDNA-GPS dataset card.
Idempotent: front-matter configs, a TL;DR bullet, a layout entry and a section before the relation section."""
import os
import sys

STAGING = "/data/backup/DriveDNA-GPS_hf"
README = os.path.join(STAGING, "README.md")
CODE = "https://github.com/WangYuHang-cmd/GPS--CAN"
MARK = "## 🧭 CAN-to-GPS benchmark v1 (`benchmark/`)"

SECTION = f'''{MARK}

Can the GPS trajectory of a drive be reconstructed from the vehicle-bus signals alone, once a single start position is
known? The `benchmark/` folder holds everything that fixes this task on DriveDNA-GPS; the code (package `cantraj`,
baselines, scoring, tables and figures) is at [{CODE}]({CODE}).

- **Task**: given the CAN events of a drive (and optionally the device IMU) and the position at one instant t0,
  predict the position at every later reference second. Eight protocols = {{online, offline}} x {{no map, OpenStreetMap
  road graph}} x {{start heading given, start heading unknown}}; windows of 1, 5 and 15 minutes and whole drives;
  signal groups from speed only (G1) and speed + steering angle (G3, the CAN-only main board) to full CAN + IMU (G8).
- **Episodes**: 98,972 eligible episodes from 3,675 drives and 431 drivers (`manifests/episode_manifest_v1.parquet`;
  anchor = a moving GPS fix with good course and accuracy; eligibility = at least 80% of seconds with a fix, no gap
  over 30 s, path over 100 m). Driver-disjoint folds train / val / test / few_shot_heldout (`manifests/split_drives.parquet`).
- **Reference**: the 1 Hz GPS view of this repository, with a per-receiver time compensation estimated on the training fold.
- **Metrics**: ADE, FDE, coverage within 10 / 25 / 50 / 100 m, success S(50 m, 0.9), failure flags; medians with
  driver-cluster bootstrap CIs; paired gains on common episodes (`scores/full_v1/`).
- **Headline (5-minute episodes, start and heading known, test / few_shot_heldout median ADE)**: speed-only straight
  line 735 / 825 m; steering dead reckoning 921 / 1,669 m (no better than the straight line); learned causal GRU
  about 540 / 390 m; with a road map the online particle filter reaches 44 / 27 m and the offline beam search
  38 / 21 m (coverage within 100 m about 0.85); with the heading unknown the map filter gives 147 / 94 m and the
  beam search 43 / 24 m. Over whole drives the map methods lose the road in 45 to 82% of the episodes.
- **Why gated**: `manifests/episode_p0.parquet` holds the episode start coordinates (the only GPS information a method
  receives), and every manifest references drives by their DriveDNA pseudonyms. No raw log path, device or route
  identifier appears in any file (checked at staging; `benchmark/CHECKSUMS.sha256` lists every file).
- **Reproduce**: clone the code, point `CANTRAJ_GPS` at this repository, `CANTRAJ_MANIFESTS` at `benchmark/manifests`
  and `CANTRAJ_BENCH` at a cache with the `events/` tables, then follow the README of the code repository. Prediction
  files (tens of GB) are available on request.

'''
TLDR = ("- **CAN-to-GPS benchmark**: `benchmark/` fixes the trajectory-recovery task on these drives (98,972 episodes, eight protocols, "
        f"frozen manifests, per-episode scores of every baseline); code at [{CODE}]({CODE}).\n")
LAYOUT_OLD = "├── events/\n"
LAYOUT_NEW = ("├── benchmark/                    CAN-to-GPS benchmark v1: manifests/, config/, scores/full_v1/, CHECKSUMS.sha256\n"
              "├── events/\n")
CONFIGS = ('''  - config_name: benchmark_episodes
    data_files: "benchmark/manifests/episode_manifest_v1.parquet"
  - config_name: benchmark_scores
    data_files: "benchmark/scores/full_v1/episodes_all.parquet"
''')


def main():
    txt = open(README, encoding="utf-8").read()
    changed = False
    if MARK not in txt:
        anchor = "## 🔗 Relation to DriveDNA and DriveDNA-Controlled"
        assert anchor in txt
        txt = txt.replace(anchor, SECTION + anchor, 1); changed = True
    if "CAN-to-GPS benchmark**" not in txt:
        anchor = "- **Access**: gated"
        assert anchor in txt
        txt = txt.replace(anchor, TLDR + anchor, 1); changed = True
    if "├── benchmark/" not in txt:
        assert LAYOUT_OLD in txt
        txt = txt.replace(LAYOUT_OLD, LAYOUT_NEW, 1); changed = True
    if "config_name: benchmark_episodes" not in txt:
        anchor = '  - config_name: events_manifest\n    data_files: "events/events_manifest.csv"\n'
        assert anchor in txt
        txt = txt.replace(anchor, anchor + CONFIGS, 1); changed = True
    if changed:
        open(README, "w", encoding="utf-8").write(txt)
    print("card patched" if changed else "card already up to date")
    try:
        from huggingface_hub import DatasetCard
        DatasetCard.load(README).validate(repo_type="dataset")
        print("DatasetCard.validate: ok")
    except ImportError:
        print("huggingface_hub not available in this interpreter; validate with the miniconda python")


if __name__ == "__main__":
    sys.exit(main())
