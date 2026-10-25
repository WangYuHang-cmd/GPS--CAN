#!/usr/bin/env python3
"""freeze_benchmark.py: write configs/benchmark_<v>.yaml (machine-readable, with sha256 of every frozen table) and
specs/benchmark_<v>.md (human-readable). Run once per version; any threshold change requires a new version."""
import argparse
import hashlib
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import paths, registry  # noqa: E402

TABLES = ["vehicle_params.csv", "drive_platform.csv", "availability_wide.parquet", "split_drives.parquet", "time_alignment.json",
          "calibration_global.json"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 22), b""):
            h.update(ch)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    args = ap.parse_args()
    man_path = os.path.join(paths.MANIFESTS, f"episode_manifest_{args.version}.parquet")
    man = pd.read_parquet(man_path)
    win = registry.windows()
    prot = registry.protocols()
    cfg = dict(
        benchmark_version=args.version, frozen_on="2026-10-22",
        anchor=win["anchor"] | dict(can_speed_min_mps=2.5), windows=win["classes"], eligibility=win["eligibility"], reference=win["reference"],
        time_compensation=json.load(open(os.path.join(paths.MANIFESTS, "time_alignment.json")))["delta_s"],
        metrics=dict(coverage_radii_m=prot["coverage_radii_m"], success_thresholds=prot["success_thresholds"],
                     cluster_bootstrap=dict(primary="driver", secondary="drive", B=1000, seed=20260709, ci=0.95),
                     failure_flags=dict(fde_gt_500="final error > 500 m", lost_road="final fifth of the episode beyond 100 m")),
        protocols=prot["protocols"], map_radius_rule=prot["map_radius_rule"] | dict(cap_km=60),
        input_layer_main="events", grid_compat="control only",
        main_board_admission=dict(include="GPS ok and re-decode ok", exclude=["platform mock or missing (platform_unconfirmed)", "Controlled set (supplement)", "ineligible windows"]),
        static_params=dict(source="per-drive carParams (platform medians in vehicle_params.csv)", calibrated_variants_only=["angle_offset_train_deg", "steer_ratio_eff_train", "calibration_global.json fallback"]),
        splits=dict(main="driver-disjoint fold from DriveDNA (train/val/test/few_shot_heldout)", auxiliary=["seen_route", "cross_route_within_driver", "cross_metro"],
                    model_selection="val only", test_policy="frozen methods only; no repeated selection on test"),
        signal_groups=registry.groups()["groups"], core_groups=["G1", "G3"],
        tables={t: sha(os.path.join(paths.MANIFESTS, t)) for t in TABLES if os.path.exists(os.path.join(paths.MANIFESTS, t))},
        episode_manifest=dict(file=os.path.basename(man_path), sha256=sha(man_path), n_episodes=int(len(man)), n_eligible=int(man.eligible_main.sum()),
                              by_win={w: dict(episodes=int(n), eligible=int(e)) for w, n, e in man.groupby("win").agg(n=("episode_id", "size"), e=("eligible_main", "sum")).itertuples()}),
    )
    out = os.path.join(paths.GPS_DIR, "configs", f"benchmark_{args.version}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    elig = man[man.eligible_main]
    L = [f"# Benchmark {args.version} (frozen 2026-10-22)\n",
         "Any change to a threshold below requires a new version and a rerun of every table. Test and few_shot_heldout folds are scored only for frozen methods; model selection uses val.\n",
         "## Episodes", f"- Anchor: GPS speed >= {win['anchor']['min_speed_mps']} m/s, 0 < bearing_acc <= {win['anchor']['max_bearing_acc_deg']} deg, u-blox h_acc < {win['anchor']['ublox_max_h_acc_m']} m, CAN speed >= 2.5 m/s; p0 = GPS(t0); theta0 = pi/2 - bearing(t0) (H+ only).",
         f"- Windows: {', '.join(f'{k} = {v['duration_s']} s' if v['duration_s'] else 'full = whole drive' for k, v in win['classes'].items())}; eligibility: >= {win['eligibility']['min_ref1hz_frac']:.0%} of seconds with a 1 Hz fix, no reference gap > {win['eligibility']['max_ref_gap_s']} s, path >= {win['eligibility']['min_path_len_m']} m.",
         "- Main-board admission: GPS ok, re-decode ok, platform confirmed (not mock), not the Controlled supplement.",
         f"- Counts: {len(man)} episodes, {len(elig)} eligible; by window: " + ", ".join(f"{w} {int(n)}" for w, n in elig.groupby('win').size().items()) + "; by fold: " + ", ".join(f"{k} {int(n)}" for k, n in elig.groupby('fold').size().items()) + ".",
         "## Reference and time", "- 1 Hz view (one real fix per second, never interpolated); time compensation (training fold): " + ", ".join(f"{k} {v} s" for k, v in cfg['time_compensation'].items()) + "; results reported with and without.",
         "## Inputs", "- Online main board: raw CAN events at logged times; the released interpolated table is a control only. Core groups G1 (speed) and G3 (speed + steering with per-drive carParams nominal parameters); calibrated variants are named explicitly and use training-fold values with a global fallback for unseen platforms.",
         "## Metrics", f"- ADE, FDE, P90, coverage at {cfg['metrics']['coverage_radii_m']} m, success S(50 m, 0.9) primary and S(100 m, 0.8) secondary, fde_gt_500 and lost_road failure flags; cluster bootstrap by driver (primary) and by drive; paired differences for gains.",
         "## Maps", f"- OSM drive graph from Geofabrik extracts (manifests/map_tiles.csv); region = disc of radius 5 km + 40 m/s x window duration (cap 60 km) around p0; never derived from the trajectory.",
         "## Frozen tables (sha256)"] + [f"- {k}: `{v[:16]}...`" for k, v in cfg["tables"].items()] + [f"- {cfg['episode_manifest']['file']}: `{cfg['episode_manifest']['sha256'][:16]}...`"]
    with open(os.path.join(paths.GPS_DIR, "specs", f"benchmark_{args.version}.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("written", out)


if __name__ == "__main__":
    main()
