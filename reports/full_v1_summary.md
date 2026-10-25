# CAN-to-GPS trajectory recovery, benchmark v1: results digest (2026-10-24)

Full tables: `reports/full_v1_results.md`; per-cell CSVs: `/data/backup/DriveDNA-GPS_bench/scores/full_v1/tables/`.
98,972 eligible episodes (w01 75,236 / w05 15,651 / w15 4,607 / full 3,478), 3,675 drives, 431 drivers; main folds =
test + few_shot_heldout (driver-disjoint from train), val = model-selection fold. Median ADE in metres with 95% driver-cluster
bootstrap CIs; c100 = share of reference points within 100 m; lost_road = share of episodes that leave the road (> 100 m) and
never return. Map rows use OSM tiles; 848 of 42,811 main+val episodes (2.0%) have no map prediction (tiles outside every
configured extract) and are excluded from the map rows only.

## 1. Five-minute windows, start known, heading known

| method | input | test | few_shot | val |
|---|---|---|---|---|
| curvature DR (class F: localizer curvature, GPS-informed) | G5 | 84 [74, 98] | 80 [70, 92] | 74 [66, 92] |
| straight line (speed only) | G1 | 735 [651, 884] | 825 [787, 951] | 763 [709, 824] |
| nominal steering DR | G3 | 921 [538, 1,737] | 1,669 [878, 2,451] | 731 [504, 1,301] |
| online-offset steering DR (CAN-only offset estimate) | G3 | 869 [656, 1,351] | 1,423 [981, 1,948] | 855 |
| train-fold calibrated steering DR | G3 | 1,312 [459, 4,179] | 1,651 [918, 2,386] | 694 [491, 1,248] |
| CAN yaw-rate DR (600 drives with yawRate) | G4 | 851 [151, 1,529] (n 658) | 521 [81, 1,831] (n 188) | |
| learned causal GRU (3 seeds) | G3 | 513 / 520 / 595 | 354 / 386 / 439 | 230 / 230 / 239 |
| learned physics residual + GRU (3 seeds) | G3 | 568 / 632 / 871 | 400 / 421 / 485 | 252 / 273 / 285 |
| learned physics residual, no GRU (3 seeds) | G3 | 1,126 / 1,192 / 1,292 | 1,365 / 1,454 / 1,458 | 631 / 659 / 667 |
| online map particle filter, nominal steering | G3 + map | 44 [29, 61] | 27 [19, 52] | 38 [29, 51] |
| online map particle filter, speed only | G1 + map | 682 | 649 | 844 |
| offline beam search, nominal steering (Top-1) | G3 + map | 38 [24, 55] | 21 [17, 42] | 30 [24, 42] |
| offline beam search, Top-10 oracle | G3 + map | 31 | 17 | 22 |

## 2. Five-minute windows, start known, heading unknown

| method | test | few_shot | val |
|---|---|---|---|
| every no-map method | 2,400 to 3,400 | 2,500 to 3,300 | 1,700 to 2,300 (holding the start is best) |
| online map particle filter, nominal steering | 147 [105, 182] | 94 [62, 153] | 85 [63, 138] |
| offline beam search, nominal steering | 43 [26, 59] | 24 [17, 44] | 32 [24, 46] |

## 3. Longer horizons (nominal steering with map, heading known)

| window | method | test | few_shot |
|---|---|---|---|
| 15 min | online PF | 197 [80, 509], lost_road 0.71 | 131 [49, 246], lost_road 0.57 |
| 15 min | offline beam | 158 [68, 373], lost_road 0.69 | 109 [43, 208], lost_road 0.55 |
| 15 min | curvature DR (F, no map) | 511 [453, 595] | 491 [423, 588] |
| full drive | online PF | 1,598, lost_road 0.82 | 127, lost_road 0.49 |
| full drive | offline beam | 1,147 [280, 1,701], lost_road 0.75 | 64 [34, 186], lost_road 0.45 |
| full drive | curvature DR (F, no map) | 678 | 325 |
| 1 min | online PF / offline beam / curvature DR / straight line / steering DR | 16 / 13 / 11 / 49 / 59 | 13 / 9 / 9 / 53 / 93 |

## 4. Paired findings (same episodes)

- Steering DR never beats the straight line without a map: G3 minus G1 at 5 min is +151 m on test, +540 m on few_shot, -32 m on val (CIs include 0); at 1 min the medians are 59 vs 49 m (test).
- Train-fold platform calibration helps only where the platform was seen: val -29 m [-62, -10], test +13 m [-55, +1,043] (mean +641 m).
- Map minus no-map (nominal steering, heading known, 5 min): test -689 m [-1,485, -298], win rate 0.84 on val; heading unknown -2,063 m on test.
- Offline beam minus online PF: -2 m (heading known), -18 m median and about -180 m mean (heading unknown): whole-path search resolves the unknown start heading.
- Learning: the GRU models cut the no-map error to 0.3 to 0.5 of nominal DR on val and few_shot (paired -600 to -900 m on few_shot, win 0.8) but only weakly on test (-86 to -171 m, CIs span 0): the learned motion correction transfers across drivers, not across the mostly unseen test platforms. Inside the map particle filter the learned omega gives no gain on any fold (+0 to +8 m paired).
- CAN yaw rate (G4) beats steering DR where available on few_shot and val (-338 / -169 m) but not on test (+519 m, CI [-2,290, +914]).
- Perturbation probes (inputs only): speed quantised to 1 m/s and speed noise of 0.5 m/s change the 5-min DR medians by 0 m; steering held at 1 Hz by 0 to 4 m (means +5 to +17 m). Coarse CAN is as useful as raw CAN at this horizon.
- Class-F curvature inside the map particle filter (reference only) equals nominal steering with the map: 45 vs 44 m on test, 29 vs 27 m on few_shot (heading known), 148 vs 147 m and 87 vs 94 m (heading unknown). With a map, the quality of the heading-rate source stops mattering at 5 min, exactly as for the learned omega.
- Pilot stability: the 20-drive pilot medians (PF 38 m, DR 996 m, calibrated 653 m) lie inside the val and test CIs; curvature DR 66.6 m vs test 84 m [74, 98] is the one exception.
- Input layer (100 Hz rlog vs 10 Hz qlog native tables): no difference (steering DR 928 vs 920 m on test).

## 4b. Supplements and ablations (val unless noted)
- Device gyro (G8, 1,928 rlog drives; uncalibrated stream with 0.02 to 0.12 rad/s bias): within-episode standstill or steering-referenced bias estimates fail (2,880 m on test); the least-squares estimator that identifies gyro bias and steering offset jointly from the speed dependence reaches 937 m on test / 1,337 m on few_shot, i.e. no gain over the steering angle (paired +28 m [-50, +73] test, -39 m [-110, +9] few_shot) and far from the class-F curvature (+790 to +1,170 m). With the map it equals the steering filter.
- Initialisation errors (val, 5 min): a 20 m start shift costs steering DR +2 m [1, 3] and the map filter 0 m (it re-anchors to the road); heading noise of 10 deg costs DR +13 m [7, 21] vs 0 m for the filter, 30 deg +107 m [66, 138] vs +4 m.
- Learned attacker (gru seed 0) on degraded inputs: speed noise 0.5 m/s +877 m [623, 1,331] (dead reckoning is insensitive to the same noise), speed quantised to 1 m/s +15 m, steering at 1 Hz +6 m. Retraining the same model with 0.5 m/s speed-noise augmentation removes the brittleness: noisy inputs 236 m vs 1,225 m (paired -859 m [-1,322, -567], win rate 0.88) at a cost of +5 m [-0, 11] on clean inputs (240 vs 230 m).
- Map coverage after per-episode tile planning: every evaluated main+val episode has map predictions (0 map_missing).

## 5. Caveats
- Main-board inputs are CAN-only groups G1/G3 (G4 is a 600-drive supplement; G5 curvature is class F and reported as a reference, never as a CAN-only method).
- Test-fold CIs are wide (41 to 42 drivers, heterogeneous unseen platforms); several learned-model medians have upper CI bounds above 2 km.
- Mock/unconfirmed platforms and the Controlled supplement are outside the main board; no result is stratified by vehicle label.
- Learned models: configuration frozen before any test evaluation; test evaluated once; model selection by val chunk loss.
