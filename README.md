# cantraj: CAN-to-GPS trajectory recovery benchmark (DriveDNA companion)

**Question.** Given a drive's starting position (and, in some protocols, its starting heading and a public road
graph), how much of the vehicle's GPS trajectory can be reconstructed from the vehicle-bus signals alone? The benchmark
treats this as both a privacy measurement (how much location a CAN stream leaks once one anchor is known) and a
GPS-denied localisation task, with one frozen protocol set, one evaluation engine and a fixed set of baselines.

* Benchmark card and methods: `reports/paper/benchmark_card_draft.md`, `reports/paper/methods_draft.md`.
* Results: `reports/full_v1_summary.md` (digest), `reports/paper/tables/` (T1 to T7), `reports/paper/figures/`,
  `reports/failure_analysis_v1.md`.
* Frozen configuration: `configs/benchmark_v1.yaml`, `specs/benchmark_v1.md` (v1.0, 2026-10-22).
* Package `src/cantraj`: data layer (`io`, `episodes`, `events`, `inputs`), signal whitelist (`registry`), causal replay
  (`replay`), metrics and scoring, baselines (`baselines/`), map graphs (`maps/`), learned motion models (`learn/`).
* Configs `configs/`: signal registry with provenance classes, signal groups G1 to G8, eight protocols, windows.
* Data: the gated Hugging Face dataset `HenryYHW/DriveDNA-GPS` (GPS reference, native CAN / IMU event tables, the
  `benchmark/` manifests and per-episode scores) plus the DriveDNA release for the 10 Hz grid tables. See `ENVIRONMENT.md`.

## Install

```
pip install -r requirements.txt            # Python 3.12; torch only for training the learned baselines
export CANTRAJ_BENCH=/path/to/cache CANTRAJ_GPS=/path/to/DriveDNA-GPS CANTRAJ_DRIVEDNA=/path/to/DriveDNA
export CANTRAJ_MANIFESTS=/path/to/DriveDNA-GPS/benchmark/manifests   # the manifests folder of the gated dataset's benchmark/
export PYTHONPATH=$PWD/src
pytest -q                                   # 125 tests: synthetic line / circle, whitelist, raw-event causality, metrics,
                                            # map filter and beam on a synthetic grid, learned wrapper, perturbations, warm start,
                                            # the six published-method re-implementations on synthetic networks
```

## Protocols, groups, baselines

Protocols `O-N-Hp O-N-Hm O-M-Hp O-M-Hm F-N-Hp F-N-Hm F-M-Hp F-M-Hm` = {online, offline} x {no map, OSM map} x
{heading given, heading unknown}; the start position is always given. Signal groups: `G1` speed, `G3` speed + steering
(core, main board), `G4` CAN yaw rate (supplement), `G5` localizer curvature (class F reference, not CAN-only), `G8`
full CAN + device IMU (supplement); `G2`, `G6`, `G7` are defined but not run in v1.

Baselines (`scripts/run_baseline.py --baseline ...`): `p0_hold`, `prior`, `dr_none`, `retrieval` (G1); `dr_steer`,
`dr_steer_offonline`, `dr_steer_offfixed`, `dr_steer_calib` (G3); `dr_can_yaw` (G4); `dr_curvature` (G5);
`dr_imu_yaw`, `dr_imu_yaw_sb`, `dr_imu_yaw_rls` (G8); `map_pf_{none,steer,steer_calib,curvature,imu_yaw,imu_yaw_rls}`
(online particle filter, O-M protocols); `map_beam_{none,steer,steer_calib,curvature}` (offline whole-path beam
search with Top-K, F-M protocols); `learned` and `map_pf_learned` with `--ckpt` (trained motion models).

## Quick start (benchmark v1)

```
python scripts/run_baseline.py --baselines p0_hold dr_none prior --group G1 --protocols O-N-Hp O-N-Hm --run full_v1 \
    --manifest $CANTRAJ_MANIFESTS/episode_manifest_v1.parquet --eligible-only --workers 4 --resume --version v1
python scripts/run_baseline.py --baseline dr_steer --group G3 --protocols O-N-Hp O-N-Hm --run full_v1 \
    --manifest $CANTRAJ_MANIFESTS/episode_manifest_v1.parquet --eligible-only --workers 4 --resume --version v1
WORKERS=3 WINSETS="w05 w15|full|w01" scripts/run_full_maps.sh val            # map protocols (PF 10 Hz ticks + beam), resumable
WORKERS=3 scripts/run_full_maps.sh test few_shot_heldout
MODE=train scripts/run_learned.sh                                             # 9 learned models (GPU, 10 to 25 min each)
MODE=eval-val EVAL_WORKERS=4 scripts/run_learned.sh                           # no-map replay on val; MODE=eval-val-map for the map filter
MODE=eval-test EVAL_WORKERS=4 scripts/run_learned.sh                          # once, frozen configuration
python scripts/score.py --run full_v1 --manifest $CANTRAJ_MANIFESTS/episode_manifest_v1.parquet --eligible-only --workers 4 --resume
python scripts/make_tables.py --run full_v1 --n-boot 1000                     # reports/full_v1_results.md, scores/full_v1/tables/*.csv
python scripts/make_paper_tables.py && python scripts/paper_curves.py && python scripts/make_paper_figures.py
python scripts/failure_analysis.py
```

## Quick start (benchmark v1.1, the ICDE study; run icde_v11)

Benchmark v1.1 = v1 with map v2 tiles (every needed tile re-cut from all intersecting extracts), the map-start rule
(`map_start_ok`: a drivable edge within 400 m of p0), per-episode visible maps with episode-frame bearings, the learned
motion source fed at event times and the named failure flags (`specs/benchmark_v1.md`, `PLAN_ICDE_EAB.md`). Thresholds,
windows, splits and reference are v1's; no-map rows are unchanged v1 runs. Map tiles: use the published tile set
(`map_tiles_v2.csv` lists every tile's source extract, md5 and snapshot date; exact reproduction), or re-cut from
current Geofabrik extracts with `python scripts/recut_tiles_v2.py` (a different OpenStreetMap snapshot: map numbers
move; the script refuses to overwrite a published tile list without `--force`).

```
python scripts/build_map_episodes.py                      # manifests/episode_map_v11.parquet: map_start_ok, tiles per episode
scripts/run_icde_maps.sh                                  # map PF and beam, speed only / steering, H+ / H-, w05 w15 full
scripts/run_icde_methods.sh                               # MacroTracking, OPD-II (pyvalhalla 3.9.1, scripts/build_valhalla_v2.py first),
                                                          # Elastic Pathing, CAN-Trace; author-sample checks in scripts/validate_*.py
scripts/run_icde_spy_dewri.sh                             # SPy / RoCuMa (validation selection first) and Dewri et al.
scripts/run_icde_learned.sh                               # learned motion sources in the map filter; signal-set study (3 seeds)
scripts/run_icde_cohort.sh                                # same-cohort 1 / 5 / 15 minute prefixes (manifests/cohort_v11.csv)
scripts/run_icde_cost.sh; scripts/run_icde_cost_pedal.sh  # cost study, one worker (92 validation windows; pedal subset for CAN-Trace)
scripts/run_icde_probe.sh                                 # degraded-input sensitivity sweep (run icde_probe, fixed grid)
python scripts/score.py --run icde_v11 --manifest $CANTRAJ_MANIFESTS/episode_manifest_v1.parquet --eligible-only \
    --folds test few_shot_heldout --workers 2 --resume
python scripts/icde_tables.py --extras pairs topk signal cohort cost robust robust_map   # reports/icde/tables/*.csv, main_board.tex
python scripts/icde_area_check.py && python scripts/icde_extra.py                         # map-extent checks; complementarity, confidence
python scripts/icde_figures.py --figs cdf topk horizon && python scripts/icde_macros.py && python scripts/registry.py
```

`run_icde_seq.sh` is the single sequential chain that produced the published runs (memory gate, 2 to 5 workers per
step); every run writes `meta.json` (benchmark version, manifest sha256, tile list md5, code sha, seed) that
`scripts/registry.py` collects into `reports/icde/run_registry.{csv,md}`.

Runner options: `--flush-min 15` rewrites part files every 15 minutes so `--resume` recovers a killed pass;
`--perturb speed_quant=1,speed_noise=0.5,steer_hz=1 --stem-suffix _pq1` (input perturbation, causal, per-episode
seeded); `--init-perturb pos_m=20,heading_deg=10 --stem-suffix _ih10` (start shift in the episode frame, heading
noise); `--warmup-s 300|inf --stem-suffix _wu5|_wuall` (warm start on the drive's own pre-t0 CAN / IMU history,
research rows); `--group G8 --drives-file manifests/drives_G8.csv` for the IMU supplement.

Reproducing the native event tables from raw openpilot logs (`scripts/m0_*.py`) needs the openpilot tools and the raw
logs, which are not part of the release; the re-decoded tables are in the gated dataset instead. Maps:
`scripts/fetch_maps_all.py` (Geofabrik extracts by region) and `scripts/plan_episode_tiles.py` (tiles per episode).

## Rules that the code enforces

* A baseline receives only its group's whitelisted columns; anything else is dropped and a request for a forbidden
  signal raises. GPS, wall-clock time and identifiers are never readable by a baseline (the runner replaces the
  reference loader with a raising stub).
* Online baselines see an event only after its logged time and are queried at the real reference seconds; offline
  baselines see the episode's events and nothing outside it.
* Test and few_shot_heldout are scored only for frozen methods; model selection uses val. Any threshold change is a
  new benchmark version and a full rerun.
* Public tables, manifests and figures contain no raw log paths, device or route identifiers, and no coordinates
  beyond episode-frame metres (`scripts/export_release.py` gates the release).

## Licence

Research use only, see `LICENSE` (DriveDNA Research License). Data access is governed by the gated dataset's terms.
Please cite the DriveDNA paper (arXiv:2607.23822) and this benchmark when you use the code or the data.
