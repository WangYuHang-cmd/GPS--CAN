# CAN-to-GPS trajectory recovery: benchmark card (draft v1, 2026-10-25)

Draft material for the paper and the dataset card. Numbers come from `reports/full_v1_summary.md`,
`reports/paper/tables/T1 to T7` and `reports/failure_analysis_v1.md`; the configuration is frozen as benchmark v1
(`configs/benchmark_v1.yaml`, `specs/benchmark_v1.md`). Phrasing is deliberately observational: the benchmark measures
what the baselines achieve on this corpus, not what is possible in general.

## 1. Task

Given a drive's in-vehicle signals (CAN bus messages, optionally the device IMU) and the vehicle's position at one
instant t0, reconstruct the vehicle's position at every later second of the episode. No GPS, no cellular or Wi-Fi
positioning and no information recorded after the queried instant are available to an online method; an offline
method sees the whole episode's signals but nothing outside it. The start position is always given; the start heading
is given in the "H+" protocols and withheld in the "H-" protocols. Predictions are queried at the real timestamps of
the GPS reference (one fix per second), so a method must produce a position for any requested time.

The task quantifies two things at once: how much location information an in-vehicle data stream leaks when a single
anchor position is known (a privacy question), and how far a vehicle can be tracked from its own CAN data when
positioning is unavailable (a localisation question). Both readings use the same protocols and metrics.

## 2. Protocols

Eight protocols = {online, offline} x {no map, road map} x {heading given H+, heading unknown H-}, written
O-N-Hp, O-N-Hm, O-M-Hp, O-M-Hm, F-N-Hp, F-N-Hm, F-M-Hp, F-M-Hm.

- online: an event becomes visible only after its logged time; no episode length, end point or future gap mask is
  given; the method is queried at the reference seconds as they arrive.
- offline: every whitelisted event inside [t0, t_end] is visible; nothing outside the episode is. Offline map methods
  may return up to K = 10 whole-path candidates with probabilities (Top-K oracle rows are marked as such).
- map: an OpenStreetMap drive graph cut from Geofabrik extracts; the usable region is the disc of radius 5 km + 40 m/s x
  (time since t0), capped at 60 km, around the start, never derived from the trajectory.
- For dead-reckoning methods the offline no-map rows equal the online rows (no future is used), and are not repeated.

Main board (frozen v1): O-N-Hp, O-N-Hm, O-M-Hp, O-M-Hm, F-M-Hp, F-M-Hm with the CAN-only groups G1 and G3.

Research variants (never on the main board, always labelled): warm start (the method may calibrate on the drive's own
CAN / IMU history before t0, W = 5 min or all past; still no GPS), perturbed inputs (speed quantised to 1 m/s, speed
noise 0.5 m/s, steering held at 1 Hz), initialisation errors (20 m start shift, 10 or 30 degree heading noise).

## 3. Inputs: signal groups and provenance classes

Signals are whitelisted per group; a method receives only the group's columns (anything else is physically dropped and
a request for a non-whitelisted signal raises). Static per-vehicle parameters come from the drive's own logged
`carParams` (wheelbase, steering ratio) unless a variant is named "calibrated".

| group | signals | static | role |
|---|---|---|---|
| G1 speed_only | speed | none | core (main board) |
| G3 speed_steer | speed, steering angle (steering rate optional) | wheelbase, steering ratio | core (main board) |
| G4 speed_can_yaw | speed, CAN yaw rate | unit, sign | supplement (600 drives with yawRate) |
| G5 speed_curvature_F | speed, localizer curvature (yaw rate optional) | none | class F reference, not CAN-only |
| G7 full_can | speed + every CAN signal in the registry (pedals, blinkers, cruise, wheel speeds, yaw rate) | as G3 + G4 | defined, not run in v1 |
| G8 full_can_imu | G7 + device gyroscope and accelerometer | as G7 | supplement (1,928 rlog drives with IMU >= 50 Hz) |
| G2 speed_pedals, G6 speed_accel | redundancy experiments | | defined, not run in v1 |

Provenance classes: C0 = raw CAN value (speed, steering angle, yaw rate, wheel speeds), C1 = CAN-derived inside the
vehicle software (longitudinal acceleration), I0 = device IMU, F = fusion-derived by the openpilot localizer, which
is GPS-informed in old logs; class F is therefore reported only as a reference and never as a CAN-only result.
Always forbidden: latitude, longitude, altitude, GPS speed and bearing, accuracies, wall-clock time.

Input layer: the native event tables (raw messages at their logged times, 100 Hz from rlog, 10 Hz from qlog-only
drives); the released interpolated 10 Hz table is a control only. Online methods may hold the latest event on a causal
tick grid (the map filter uses 10 Hz ticks; dead reckoning steps on every event).

## 4. Episodes, eligibility and counts

- Anchor (t0): GPS speed >= 3 m/s, 0 < bearing accuracy <= 5 degrees, u-blox horizontal accuracy < 10 m, CAN speed
  >= 2.5 m/s; p0 = GPS position at t0; theta0 = GPS course at t0 (H+ only).
- Windows: w01 = 60 s, w05 = 300 s, w15 = 900 s, full = the rest of the drive. Eligibility: at least 80% of seconds
  with a 1 Hz fix, no reference gap > 30 s, path length >= 100 m.
- Corpus: 110,042 episodes built, 98,972 eligible (w01 75,236 / w05 15,651 / w15 4,607 / full 3,478) from 3,675
  drives and 431 drivers. Main folds (test + few_shot_heldout): w05 3,880, w15 1,173, full 742 episodes from 776 drives
  and 92 drivers.
- Reference: the 1 Hz GPS view (one real fix per second, never interpolated), with a receiver-specific time
  compensation estimated on the training fold (qcom 0.7 s, u-blox 0.2 to 0.3 s); results are reported with and without it.

## 5. Splits

Main split: driver-disjoint folds train / val / test / few_shot_heldout (`split_drives.parquet`, frozen with its
sha256). Test and few_shot_heldout are evaluated only for frozen methods; model selection and every ablation use val.
Auxiliary splits (seen route, cross route within driver, cross metro) are defined for later work. Confidence
intervals are driver-cluster bootstraps (95%), gains are paired differences on common episodes.

## 6. Metrics

ADE (mean position error over the episode's reference seconds), FDE, P90, coverage c_r for r in {10, 25, 50, 100} m
(share of reference seconds within r; a missing prediction counts as a miss), success S(50 m, 0.9) primary and
S(100 m, 0.8) secondary, and two failure flags: fde_gt_500 (end error > 500 m) and lost_road (the final fifth of the
episode stays beyond 100 m). Offline map rows add the Top-K oracle (best of K whole-path candidates).

## 7. Main-board admission and exclusions

Drives whose platform is "mock" or whose `carParams` are missing are outside the CAN-only main board
(`platform_unconfirmed`). The 48-drive Controlled supplement is a separate `supplement` fold. G4 and G8 are
supplements on their own drive subsets, G5 is a class F reference. No result is stratified by vehicle label beyond the
"platform seen / unseen in training" stratum, which is confounded with driver and region and reported as such.

## 8. Maps

OSM extracts (Geofabrik) cut into 0.25 degree tiles (`manifests/map_tiles.csv` with source and md5); per-episode
tile planning guarantees that every main-fold and val episode has its disc covered (0 of 42,811 episodes without a map
prediction after planning; sea, water and road-less tiles are materialised empty). The drive graph for a method is the
union of its episodes' discs; nothing about the trajectory enters the graph.

## 9. Baselines (details in `methods_draft.md`)

Trivial: hold start, start prior, route retrieval. Dead reckoning: speed-only straight line (G1); steering (G3) with
nominal parameters, with a causal online steering-offset estimate, and with training-fold calibration; CAN yaw rate
(G4); device gyro with causal bias estimators (G8); localizer curvature (G5, class F). Learned (G3): causal GRU motion
model, physics-residual GRU, physics residual without GRU (three seeds each, trained on train, selected on val).
Map: online particle filter on the drive graph with the same heading-rate sources, offline beam search (Top-1 and
Top-K). Warm-start variants of the calibrating methods (research rows).

## 10. Headline results (median ADE in metres, 5-minute episodes, start known)

| setting | method | test | few_shot_heldout |
|---|---|---|---|
| no map, heading known | speed only, straight line (G1) | 735 [651, 884] | 825 [787, 951] |
| no map, heading known | steering DR, nominal (G3) | 921 [538, 1,737] | 1,669 [878, 2,451] |
| no map, heading known | steering DR, online offset (G3) | 869 [656, 1,351] | 1,423 [981, 1,948] |
| no map, heading known | steering DR, train-fold calibrated (G3) | 1,312 [459, 4,179] | 1,651 [918, 2,386] |
| no map, heading known | learned causal GRU (G3), 3 seeds | 543 +- 37 | 393 +- 35 |
| no map, heading known | curvature DR (G5, class F reference) | 84 [74, 98] | 80 [70, 92] |
| map, heading known | online particle filter, steering (G3) | 44 [29, 61] | 27 [19, 52] |
| map, heading known | online particle filter, speed only (G1) | 682 | 649 |
| map, heading known | offline beam search, steering (G3), Top-1 | 38 [24, 55] | 21 [17, 42] |
| map, heading known | offline beam search, Top-10 oracle | 31 | 17 |
| no map, heading unknown | every no-map method | 2,400 to 3,400 | 2,500 to 3,300 |
| map, heading unknown | online particle filter, steering | 147 [105, 182] | 94 [62, 153] |
| map, heading unknown | offline beam search, steering | 43 [26, 59] | 24 [17, 44] |

Coverage at 5 minutes: c100 about 0.25 to 0.33 for the CAN-only no-map methods, 0.81 to 0.86 with the map; success
S(50 m, 0.9) 0.37 (test) and 0.54 (few_shot) for the map particle filter. Longer horizons (steering + map, heading
known): 15 min 197 / 131 m with lost_road 0.71 / 0.57 (filter) and 158 / 109 m (beam); whole drives 1,598 / 127 m
(filter) and 1,147 / 64 m (beam) with lost_road 0.45 to 0.82. One minute: 16 m (filter), 13 m (beam), 49 m straight
line, 59 m steering DR on test.

Paired findings: steering does not beat the straight line without a map (G3 minus G1 at 5 min: +151 m test, +540 m
few_shot, -32 m val; CIs include 0); the map is worth -689 m [-1,485, -298] on test with heading known and about
-2,000 m with heading unknown; the offline search beats the online filter by -2 m (heading known) and -18 m median /
about -180 m mean (heading unknown); learned motion models cut the no-map error to 0.3 to 0.5 of nominal DR on val
and few_shot but only weakly on test (CIs span 0), and give no gain inside the map filter; coarse CAN (1 m/s speed
quantisation, 1 Hz steering) changes DR medians by 0 to 4 m; a 20 m start shift costs DR +2 m and the filter 0 m;
30 degrees of heading noise costs DR +107 m and the filter +4 m.

Failure analysis (research tables): map-filter failures start right after turns (43% of 15-minute lost-road onsets
within 30 s after a turn vs 23% expected for uniformly spread onsets; 67% vs 38% on whole drives) and lost episodes
have more turns; the speed-only map filter depends strongly on road density (88 / 678 / 856 m across density
tertiles at 5 min) while the steering filter does not at 5 min (30 / 39 / 36 m) and prefers sparse networks at 15 min
(95 vs 221 / 265 m).

## 11. Known limitations

- Class F curvature is GPS-informed in old logs; it is a reference for "what a good heading-rate source would give",
  not a CAN-only result.
- The test fold's platforms are mostly unseen in training (1,312 of 3,880 five-minute episodes are on unseen
  platforms); test CIs are wide (41 to 42 drivers) and several learned-model medians have upper CI bounds above 2 km.
- Whole-drive episodes fail often with the map (lost_road 0.45 to 0.82); the benchmark reports this rather than
  trimming drives.
- Map rows depend on OSM completeness and on the disc rule; episodes whose disc left every configured extract were
  excluded from map rows before per-episode planning (now 0 in main + val; 73 land tiles in train remain unconfigured).
- Time compensation is estimated on the training fold per receiver stratum; results are also given without it.
- Vehicle labels are taken as logged, without correction; no conclusion is drawn per vehicle model.
- One validation driver contributes 337 of 787 val drives; CIs therefore cluster by driver.
- The learned models were trained once per seed with a fixed recipe; a stronger attacker is possible and the
  augmentation result (speed-noise training removes the brittleness to speed noise) shows how quickly such gaps close.

## 12. Intended use and ethics

Research only, under the DriveDNA research licence (no re-identification or individual scoring; data via the gated
DriveDNA-GPS release). The GPS reference is released gated and without absolute time; the benchmark code never reads
raw device or route identifiers, and published figures and manifests contain no coordinates beyond episode-frame
metres and no identifiers. The intended uses are measuring the location leakage of in-vehicle data streams,
evaluating defences (quantisation, noise, signal removal) and GPS-denied localisation research.

## 13. Release contents and reproduction

Code: repository `GPS--CAN` (package `cantraj`, scripts, configs, specs, tests, this card, the methods draft, tables
and figures). Data: the gated `DriveDNA-GPS` repository holds the GPS reference and the `benchmark/` folder with
`episode_manifest_v1.parquet`, `split_drives.parquet`, `availability_wide.parquet`, `vehicle_params.csv`,
`calibration_global.json`, `time_alignment.json`, `map_tiles.csv`, `episode_p0.parquet` and the per-episode score
tables. Prediction files (tens of GB) are available on request.

```
# environment: see ENVIRONMENT.md (CANTRAJ_BENCH, CANTRAJ_GPS, CANTRAJ_DRIVEDNA)
pytest -q                                                   # unit tests, no data needed
python scripts/run_baseline.py --baseline dr_steer --group G3 --protocols O-N-Hp O-N-Hm \
    --run full_v1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --wins w05 --workers 3 --version v1
python scripts/run_baseline.py --baseline map_pf_steer --group G3 --protocols O-M-Hp --run full_v1 \
    --manifest manifests/episode_manifest_v1.parquet --eligible-only --wins w05 --workers 3 --version v1
python scripts/score.py --run full_v1 --manifest manifests/episode_manifest_v1.parquet --eligible-only --workers 4
python scripts/make_tables.py && python scripts/make_paper_tables.py && python scripts/paper_curves.py && python scripts/make_paper_figures.py
```
