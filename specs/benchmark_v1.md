# Benchmark v1 (frozen 2026-10-22)

Any change to a threshold below requires a new version and a rerun of every table. Test and few_shot_heldout folds are scored only for frozen methods; model selection uses val.

## Episodes
- Anchor: GPS speed >= 3.0 m/s, 0 < bearing_acc <= 5.0 deg, u-blox h_acc < 10.0 m, CAN speed >= 2.5 m/s; p0 = GPS(t0); theta0 = pi/2 - bearing(t0) (H+ only).
- Windows: w01 = 60 s, w05 = 300 s, w15 = 900 s, full = whole drive; eligibility: >= 80% of seconds with a 1 Hz fix, no reference gap > 30.0 s, path >= 100.0 m.
- Main-board admission: GPS ok, re-decode ok, platform confirmed (not mock), not the Controlled supplement.
- Counts: 110042 episodes, 98972 eligible; by window: full 3478, w01 75236, w05 15651, w15 4607; by fold: few_shot_heldout 12659, test 12106, train 54529, unassigned 1632, val 18046.
## Reference and time
- 1 Hz view (one real fix per second, never interpolated); time compensation (training fold): android;ublox_hires 0.3 s, qcomdiag_lowrate 0.7 s, ublox_hires 0.2 s, ublox_lowrate 0.2 s; results reported with and without.
## Inputs
- Online main board: raw CAN events at logged times; the released interpolated table is a control only. Core groups G1 (speed) and G3 (speed + steering with per-drive carParams nominal parameters); calibrated variants are named explicitly and use training-fold values with a global fallback for unseen platforms.
- Online baselines may consume the raw events on a causal tick grid (zero-order hold of the latest event at or before each tick, never a later one): the map particle filter uses 10 Hz ticks by default (recorded as `pf_tick_hz` in the prediction metadata); dead-reckoning baselines step on every event. Causality is unchanged either way.
## Metrics
- ADE, FDE, P90, coverage at [10, 25, 50, 100] m, success S(50 m, 0.9) primary and S(100 m, 0.8) secondary, fde_gt_500 and lost_road failure flags; cluster bootstrap by driver (primary) and by drive; paired differences for gains.
## Maps
- OSM drive graph from Geofabrik extracts (manifests/map_tiles.csv); region = disc of radius 5 km + 40 m/s x window duration (cap 60 km) around p0; never derived from the trajectory.
## Frozen tables (sha256)
- vehicle_params.csv: `dbbf4183ec9ff66a...`
- drive_platform.csv: `a1941d6209653e8d...`
- availability_wide.parquet: `0a4e9d1c97829504...`
- split_drives.parquet: `5de9c563124fe2b5...`
- time_alignment.json: `b032b2fdba425693...`
- calibration_global.json: `3672fcc6d4a22714...`
- episode_manifest_v1.parquet: `7c4ddad3619eca73...`
