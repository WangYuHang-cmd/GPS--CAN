# cantraj: CANTraj-Bench, trajectory recovery from CAN signals on the CANTraj dataset (DriveDNA companion)

Code of the ICDE 2027 EAB submission "[Experiment, Analysis, and Benchmark] CANTraj: Recovering Vehicle Trajectories from CAN Signals in Naturalistic Driving". CANTraj is the released dataset of paired CAN and GNSS records; CANTraj-Bench is the benchmark defined on it (protocols, folds, metrics, baselines and the re-implemented published methods).

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
* Data: the gated Hugging Face dataset `HenryYHW/CANTraj` (formerly `HenryYHW/DriveDNA-GPS`; GPS reference, native CAN / IMU event tables, the
  `benchmark/` manifests and per-episode scores) plus the DriveDNA release for the 10 Hz grid tables. See `ENVIRONMENT.md`.

## Install

```
pip install -r requirements.txt            # Python 3.12; torch only for training the learned baselines
export CANTRAJ_BENCH=/path/to/cache CANTRAJ_GPS=/path/to/CANTraj CANTRAJ_DRIVEDNA=/path/to/DriveDNA
export CANTRAJ_MANIFESTS=/path/to/CANTraj/benchmark/manifests   # the manifests folder of the gated dataset's benchmark/
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
python scripts/icde_tables.py --extras a3 pairs topk signal cohort cost robust robust_map   # reports/icde/tables/*.csv, main_board.tex
python scripts/icde_area_check.py                         # map-extent check (reads the GNSS reference): area_coverage.csv, area_check_episodes.csv
python scripts/icde_extra.py && python scripts/icde_extra.py --what supp   # complementarity, strata, confidence, taxonomy; supplement
python scripts/icde_spy_dewri_diag.py && python scripts/failure_analysis.py --run icde_v11 --tag v11 --tiles-dir $CANTRAJ_BENCH/maps/tiles_v2
python scripts/icde_figures.py --figs cdf topk horizon robust && python scripts/icde_macros.py && python scripts/registry.py
```

## Trained models (`models/learn/`)

The weights of every learned component are released with the code (24 models, 73 files, 3.3 MB): the learned motion
sources `gru`, `resid` and `resid_nogru` (three seeds each) and `gru_aug05` (one seed; the GRU retrained with 0.5 m/s
Gaussian speed noise for the v1 robustness tables), and the signal-set models `sig/S1`, `sig/S2c`, `sig/S3`, `sig/S4c`
(three seeds each) and `sig/S2b`, `sig/S4b` (one seed; pedal-pressed flags instead of continuous pedals), with
`sig/brand_scales.json`. Each model directory holds `weights.npz` and `meta.json` (all that inference needs: the map
filter and the signal-set baselines load them with NumPy) and `model.pt` (the PyTorch state dict, saved from GPU tensors;
load it with `map_location="cpu"` on a machine without CUDA). The run scripts expect them in the cache:

```
mkdir -p "$CANTRAJ_BENCH/models" && cp -r models/learn "$CANTRAJ_BENCH/models/"   # run_icde_learned.sh: --ckpt $CANTRAJ_BENCH/models/learn/<kind>/seed<s>
```

Retraining (GPU) starts from scratch and needs the learning cache first (`python scripts/build_learn_cache.py --workers 6`,
train and val folds); `run_learned.sh` writes its log under `CANTRAJ_LOGS` (default `logs/`, create it first). The training
scripts write to the same `$CANTRAJ_BENCH/models/learn/<kind>/seed<s>` directories that the copy above fills: the motion
commands overwrite a released model in place and `train_sig.py train` skips every seed directory that already holds a
`meta.json`, so retrain into a scratch cache (another `CANTRAJ_BENCH`) or move the released `models/learn/` aside first.
The commands below use the settings the released models were trained with (30 epochs for the motion models, 15 epochs and
150 steps per epoch for the signal-set models):

```
MODE=train scripts/run_learned.sh                                                    # gru, resid, resid_nogru; seeds 0, 1, 2
python scripts/train_motion.py --model gru --seed 0 --augment-speed-noise 0.5 --out "$CANTRAJ_BENCH/models/learn/gru_aug05/seed0"
python scripts/train_sig.py cache && python scripts/train_sig.py train --epochs 15   # S1, S2c, S3, S4c; seeds 0, 1, 2
python scripts/train_sig.py train --groups S2b S4b --seeds 0 --epochs 15             # the pedal-flag variants
```

The same files are published in the data repository under `models/learn/`.

## Supplementary material (ICDE 2027 EAB)

`supplementary/supplementary.pdf` (sources in `supplementary/source/`, built by `scripts/build_supplementary.py` from the
released tables and the paper's number macros): the reproduction package and commands, the benchmark definitions, the
method reproduction details, the complete result tables and the figures the page limit left out, the agent study's
protocol, prompts and per-configuration results, and further recovered windows. Data, scores, configurations, logs and
weights: https://huggingface.co/datasets/HenryYHW/CANTraj (the weights are also in this repository, `models/learn/`)

Agent study (supplementary cohort: one map-ok five-minute window per held-out drive; `reports/icde/agent_protocol_v1.md` is the
frozen protocol, `reports/icde/agent_results.md` the results report). The LLM is served by vLLM on a GPU host and reached through
an OpenAI-compatible endpoint (`--base-url`, default `http://127.0.0.1:8000/v1`; `scripts/h100_setup.sh` installs and serves it):

```
python scripts/build_agent_cohort.py                                   # manifests/icde_agent_cohort.csv (test + few-shot), counts table
python scripts/build_agent_cohort.py --folds val --out manifests/icde_agent_val.csv --pilot 40
python scripts/run_agent_tools.py --run icde_agent_val --manifest manifests/icde_agent_val.csv --workers 3   # tool cache + tool_*/poolcand_* stems
python scripts/score.py --run icde_agent_val --manifest $CANTRAJ_MANIFESTS/episode_manifest_v1.parquet --eligible-only --folds val \
    --episodes-file manifests/icde_agent_val.csv --workers 2
python scripts/fit_agent_selector.py                                   # B: cross-fitted report + configs/agent/selector_v1.json (frozen)
python scripts/run_agent.py --run icde_agent_val --mode llm --seed 0 --manifest manifests/icde_agent_pilot.csv --concurrency 12   # pilot
python scripts/agent_pilot_report.py --run icde_agent_val --live-tag agent_qwen3_s0_live
bash scripts/run_icde_agent.sh cache && bash scripts/run_icde_agent.sh selector     # formal cohort: tools, then B
bash scripts/run_icde_agent.sh llm 0; bash scripts/run_icde_agent.sh llm 1; bash scripts/run_icde_agent.sh llm 2   # C, nominal seeds
bash scripts/run_icde_agent.sh score && bash scripts/run_icde_agent.sh tables       # scores, agent_*.csv, paper table, macros
```

Extended agent configuration (B+ and C+ in Table IV; exploratory, specified after the frozen study from validation data only and run
once on the same cohort; `reports/icde/agent_ceiling.md` is the full report, `agent_max_results.md` the tables report):

```
python scripts/run_agent_tools.py --run icde_agent_max --manifest manifests/icde_agent_cohort.csv --tools max --workers 3   # ten presets, cache v2
python scripts/fit_agent_selector.py --run icde_agent_max_val --manifest manifests/icde_agent_val_max.csv --features max       # B+ (selector_max.json)
python scripts/build_agent_max_prompt.py                                 # calibration hints from validation -> configs/agent_max/system_prompt.txt
bash scripts/run_icde_agent_max.sh pilot; for s in 0 1 2; do bash scripts/run_icde_agent_max.sh llm $s; done; bash scripts/run_icde_agent_max.sh critic 0
bash scripts/run_icde_agent_max.sh vote && bash scripts/run_icde_agent_max.sh score
python scripts/icde_agent_tables.py --run icde_agent_max --agent-tag agent_qwen3max --selector-stem selector_max__G3 --selector-results selector_max.csv \
    --extra-stems agent_qwen3max_vote__G3 agent_qwen3maxcritic_s0__G3 --suffix _max --prev-run icde_agent   # agent_max_*.csv, pair with the frozen agent
python scripts/agent_ceiling_report.py --run icde_agent_max --prev-run icde_agent --served-context 16384      # reports/icde/agent_ceiling.md
python scripts/icde_agent_table_tex.py                                   # Table IV with both configurations (frozen rows unchanged)
```

Rebuilding the tables from the published per-episode scores (no method rerun): copy the published `tables/` and `reports/`
of the package into `reports/icde/` first, because several scripts read other scripts' outputs, then run the five table
lines above. Everything is regenerated from the scores except what needs the run outputs or the training: `cost*.csv`
(the runner's timing and `/usr/bin/time` logs; machine-specific), `can_trace_failure_mode.csv` (CAN-Trace's trace graphs),
`sig_signal_use.csv` (the signal-use test of the trained learner), `T7a_lost_road_timing_v11.csv` (per-second errors) and
the run registry (`meta.json` of every run); `confidence_w05_w15.csv` reads the published `scores/icde_v11/top1_scores.csv`
when the prediction files are absent (`scripts/export_top1_scores.py` writes it from them).

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
Please cite the CANTraj paper ("[Experiment, Analysis, and Benchmark] CANTraj: Recovering Vehicle Trajectories from CAN Signals in Naturalistic Driving", under review) and the DriveDNA paper (arXiv:2607.23822) when you use the code or the data.
