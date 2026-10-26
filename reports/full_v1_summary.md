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

## 1b. How much of the trajectory is recovered (5-minute windows, start known; test / few_shot means over episodes)

Recovery metrics added 2026-10-25 (same episodes and predictions, rescored). Time-aligned: c100 = share of reference seconds within
100 m; time-free: path recall@50 = share of the reference path length within 50 m of the predicted path; route recovered = recall@50
>= 0.8; continuous = share of the window before the error first exceeds 100 m; destination = final second within 100 m / 500 m.

| method | heading | c100 | path recall@50 | path precision@50 | route recovered | continuous (100 m) | destination 100 / 500 m |
|---|---|---|---|---|---|---|---|
| map particle filter, steering (G3) | known | 0.81 / 0.85 | 0.89 / 0.90 | 0.88 / 0.89 | 0.85 / 0.86 | 0.77 / 0.81 | 0.59 / 0.69 and 0.83 / 0.85 |
| offline beam search, steering (G3) | known | 0.82 / 0.86 | 0.89 / 0.91 | 0.90 / 0.91 | 0.85 / 0.88 | 0.81 / 0.86 | 0.60 / 0.71 and 0.84 / 0.86 |
| map particle filter, speed only (G1) | known | 0.54 / 0.49 | 0.69 / 0.62 | 0.55 / 0.44 | 0.54 / 0.44 | 0.43 / 0.37 | 0.31 / 0.24 and 0.50 / 0.39 |
| map particle filter, steering (G3) | unknown | 0.68 / 0.73 | 0.84 / 0.85 | 0.61 / 0.64 | 0.77 / 0.79 | 0.23 / 0.20 | 0.54 / 0.65 and 0.77 / 0.79 |
| offline beam search, steering (G3) | unknown | 0.75 / 0.81 | 0.82 / 0.85 | 0.82 / 0.85 | 0.79 / 0.83 | 0.75 / 0.81 | 0.56 / 0.67 and 0.78 / 0.81 |
| curvature DR, class F reference (G5), no map | known | 0.64 / 0.67 | 0.56 / 0.56 | 0.57 / 0.56 | 0.27 / 0.27 | 0.64 / 0.67 | 0.27 / 0.28 and 0.82 / 0.84 |
| learned GRU (G3, seed 0), no map | known | 0.31 / 0.33 | 0.26 / 0.27 | 0.26 / 0.27 | 0.07 / 0.04 | 0.31 / 0.32 | 0.08 / 0.07 and 0.30 / 0.33 |
| speed only, straight line (G1), no map | known | 0.30 / 0.28 | 0.27 / 0.24 | 0.27 / 0.24 | 0.09 / 0.07 | 0.29 / 0.27 | 0.07 / 0.06 and 0.22 / 0.19 |
| steering DR, nominal (G3), no map | known | 0.25 / 0.20 | 0.20 / 0.16 | 0.20 / 0.16 | 0.04 / 0.02 | 0.24 / 0.20 | 0.04 / 0.03 and 0.17 / 0.11 |
| steering DR, nominal (G3), no map | unknown | 0.06 / 0.05 | 0.05 / 0.04 | 0.05 / 0.04 | 0.01 / 0.00 | 0.06 / 0.05 | 0.01 / 0.00 and 0.05 / 0.03 |
| hold start (G1) | either | 0.03 / 0.03 | 0.01 / 0.01 | undefined | 0.00 / 0.00 | 0.04 / 0.03 | 0.01 / 0.00 and 0.03 / 0.02 |

Reading: with a road map and the steering angle, about 85% of the 5-minute episodes have their route recovered (at least 80% of the
path within 50 m), the predicted path covers about 90% of the true path within 50 m, and the destination is within 100 m in 59 to
71% and within 500 m in 83 to 86% of the episodes. Without a map, CAN-only methods recover 16 to 27% of the path at 50 m and the
route in at most 9% of the episodes; the class-F reference recovers 56% of the path and the route in 27%. With the start heading
unknown, the geometric recovery of the map filter stays high (recall 0.84, route 0.77) while its time-aligned coverage drops and
its continuous share is only 0.23 (the heading is resolved during the first seconds); the offline search is almost unaffected.

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
- Learning: the GRU models cut the no-map error to 0.3 to 0.5 of nominal DR on val and few_shot (paired -600 to -900 m on few_shot, win 0.8) but only weakly on test (-86 to -171 m, CIs span 0): the learned motion correction transfers across drivers, not across the mostly unseen test platforms. Inside the map particle filter the learned omega gives no gain on any fold (three seeds: paired median +0 to +8 m, win rates 0.30 to 0.45) and hurts the tail: the test median rises from 44 m (nominal steering) to 78 to 95 m and c100 falls from 0.81 to 0.63 to 0.68, because a minority of episodes leave the road more often with the learned heading rate.
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

## 4c. Warm start: the attacker may use the drive's own CAN / IMU history before t0 (research rows, 5-minute windows, heading known)
Protocol variant outside the frozen main board: the method sees the drive's own events in [t0 - W, t0) (W = 5 min or all past; the GRU
uses the last hour) to calibrate its state, never any GPS. Median ADE val / test / few_shot, cold start -> W = 5 min -> all past:
- online-offset steering DR 855 / 869 / 1,423 -> 758 / 895 / 1,034 -> 699 / 952 / 937 m (means 1,301 / 1,385 / 1,961 -> 1,128 / 1,346 / 1,427).
  Paired against its cold start the median change is about 0 on val and test (many first-window episodes have no history; the gains sit in
  the tail: means -170 / -40 / -530 m) and -126 m [-563, +0] on few_shot. With all history it beats the nominal steering DR on few_shot
  (-385 m [-922, -10], win 0.65; test -68 m [-388, +78]) and the train-fold calibrated DR on few_shot (-360 m [-865, -17]; test -196 m
  [-2,161, +151]): a drive's own straight-road history calibrates the steering offset at least as well as the training fold does.
- device gyro DR with the least-squares bias (G8 subset) 691 / 937 / 1,337 -> 494 / 818 / 1,129 -> 383 / 712 / 903 m. The estimator needs
  speed variation, so history matters: all past vs cold start -91 m [-134, -36] on val (win 0.63), -34 m [-135, +0] on test, -46 m
  [-112, -4] on few_shot. With all history the gyro overtakes the steering angle (vs nominal steering DR -84 / -64 / -196 m paired on
  val / test / few_shot, the test CI spanning 0; vs the warmed online-offset DR -132 m [-187, -67] on val, -153 m [-343, +23] on test) but
  stays far from the class-F curvature reference (+290 / +558 / +769 m).
- learned GRU (seed 0, no retraining) 230 / 513 / 386 -> 232 / 459 / 363 -> 232 / 454 / 363 m: no change on val, a tail effect on test
  (mean 1,631 -> 952 m, upper CI of the median 2,920 -> 943 m; paired median -12 m [-726, +0]) from a recurrent state that no longer
  starts from zeros on unseen platforms.
- Heading unknown: no effect (online no-map methods degenerate to holding the start). Not run: the steering map filter has no
  calibration state, so its warm start equals its cold start.
Reading: own history closes part of the calibration gap for the CAN-only methods (a few hundred metres at 5 minutes on the drivers and
platforms where cold calibration is worst) but changes none of the main-board conclusions; the map, not calibration, remains the
decisive ingredient. Tables: `reports/paper/tables/T8_warmstart_research.*`; headline rows `*_wu5__*` / `*_wuall__*`.

## 5. Caveats
- Main-board inputs are CAN-only groups G1/G3 (G4 is a 600-drive supplement; G5 curvature is class F and reported as a reference, never as a CAN-only method).
- Test-fold CIs are wide (41 to 42 drivers, heterogeneous unseen platforms); several learned-model medians have upper CI bounds above 2 km.
- Mock/unconfirmed platforms and the Controlled supplement are outside the main board; no result is stratified by vehicle label.
- Learned models: configuration frozen before any test evaluation; test evaluated once; model selection by val chunk loss.
