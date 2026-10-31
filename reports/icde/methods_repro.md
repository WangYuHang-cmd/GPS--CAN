# Published methods: sources, assumptions, adaptations, status (benchmark v1.1)

Rule for every method: no author data is available for our drives, so the original papers' numbers cannot be checked on
this benchmark; "re-implemented" never means "fully reproduced". Main rows follow the released code or the paper's
algorithm; every deviation is listed; parameters left open by the authors are fixed before any test run and chosen on
train / val only. Bugs found in released code are kept in the main row (the authors' results come from that code) and
fixed in a separately named sensitivity row.

## MacroTracking (status: implemented, author-sample check passed; formal runs done, run icde_v11)
- Source: Gazdag, Lestyan, Remeli, Acs, Holczer, Biczok, "Privacy pitfalls of releasing in-vehicle network data",
  Vehicular Communications 39 (2023) 100565, doi 10.1016/j.vehcom.2022.100565. Code CrySyS/MacroTracking @c99063de
  (2023-01-25, GPL-3.0). Re-implemented in `src/cantraj/baselines/macrotracking.py` from the released algorithm; no code
  copied (licence).
- Original assumptions: one car with hand-calibrated constants (speed factor 1.1, steering 2.8e-5 rad per raw unit,
  2.7 m), start position and heading known, osmnx road graph fetched online around the current position, online.
- Algorithm as released: update when > 1 s since the last update (checked on each speed / steering message); turn radius
  from steering, heading change = distance / radius (|change| < 1e-4 deg ignored), step along the new heading; first
  update only stores the correction anchor; after > 20 m of travel, orthogonal projection on the nearest edge's line and
  heading blend toward the edge bearing (flipped if > 90 deg apart, blended only within 45 deg), weight
  0.7 * min(1, d_junction / 150 m).
- Adaptations: (1) map = the episode's visible v2 OSM drive graph (osmnx 'drive' semantics), nearest edge up to 600 m,
  junctions = nodes with undirected degree other than 2; (2) metric frame and unscaled CAN speed: the released code moves
  in Web Mercator units (cos(lat) m) and multiplies speed by 1.1, together 0.74 of the CAN distance at the sample's
  latitude, a fit to the authors' car and city; (3) turn radius from the bicycle model with the drive's carParams;
  (4) read-out at query times by the method's own motion model from the last state (no future data).
- Rows: `macrotracking` (as released angle handling: raw-degree comparisons), `macrotracking_calib` (+ train-fold
  steering offset and effective ratio), `macrotracking_wrapfix` (wrapped angle differences; sensitivity).
- Author-sample check (`scripts/validate_macrotracking_sample.py`; sample trace, ground truth and committed output from
  the repository; 2026 Geofabrik Hungary extract; authors' constants and Mercator scaling emulated): as-released emulation
  vs the authors' committed states: median 7.1 m, p90 15.0 m, max 19.4 m over 197 states; closest-point metric vs ground
  truth 7.0 m (authors' trajectory 6.1 m = their reported 9.08 Mercator units). The wrapped-angle variant deviates
  (median 25.9 m from the authors' states, 14.4 m to ground truth): near north it pulls the heading toward the road being
  left during a turn. Therefore the as-released handling is the main row (plan v0.7 had proposed the fix as main;
  changed on this evidence).
- Protocol: online, heading known (O-M-Hp). Cost: about 0.1 s per 5-minute episode.

## OPD-II (status: implemented; author-sample check passed; benchmark Valhalla tiles built; formal runs done, run icde_v11)
- Source: Bianchi, Brighente, Conti, Valori, "Your Car Tells Me Where You Drove: A Novel Path Inference Attack via CAN
  Bus and OBD-II Data", IEEE EuroS&P 2025, pp. 113-132, doi 10.1109/EuroSP63326.2025.00016 (arXiv 2407.00585).
  Code spritz-group/OPD-II @5fe93630 (2025-04-10), no licence file: re-implemented in `src/cantraj/baselines/opd2.py`.
- Original assumptions: OBD-II PID 0x0D speed (integer km/h) and a reverse-engineered steering-angle CAN ID, start
  position and bearing known, a per-car "wheelbase", Valhalla Meili running in Docker; output = the matched points.
- Algorithm as released (`scripts/OPD_II.py`): 0.1 s windows with mean speed and angle; bicycle model with the angle
  clipped to +-35 decoded units; every 30 windows (from the first) Meili trace_attributes on the points since the last
  match (search_radius 200, map_snap, auto); jump to the last matched point; anchor = first matched point; distance
  correction |straight DR - straight matched| added to the next step; at >= 30 km/h the bearing is reset every 30
  windows (counter % 30 == 2) from the anchor to the current point. The paper text says 50 km/h; the code is followed.
- Finding on calibration: `examples/experiments/paths.csv` lists "wheelbase" 3.85 m (Renault Captur, real about 2.6 m),
  4.0 (Dacia Duster), 4.9 (Opel Crossland), 4.5 m (Peugeot 5008). With the 0.01 decoding scale (raw steering in 0.1 deg
  gives a value of steering-wheel degrees / 10) these per-car gains amount to the bicycle model with a steering ratio of
  about 15 to 19, i.e. a car-specific calibration. The faithful translation for our data is therefore the physical
  bicycle model with the drive's carParams; the clip becomes +-350 deg of steering-wheel angle.
- Adaptations: CAN speed (m/s) and steering-wheel angle; empty windows hold the last value (the released code would
  produce NaN); Valhalla tiles built from the benchmark's v2 drive network (same OSM snapshot and drive filter for every
  map method; pyvalhalla 3.9.1 in-process Actor = the engine behind the HTTP service); bearing wrap bug
  (`360 - bearing` for negative bearings) kept in the main row, fixed in `opd2_wrapfix` (it mirrors one window's
  overshoot, under 1 deg per crossing of north).
- Rows: `opd2` (final output: each window's matched point, decided at most 3 s later; offline-protocol table, annotated
  "delayed online <= 3 s"), `opd2_online` (strictly online read-out: matched only when already matched at query time),
  `opd2_calib` (train-fold steering calibration), `opd2_wrapfix`. Heading-known only.
- Author-sample check (`scripts/validate_opd2_samples.py`, output `reports/icde/opd2_author_samples.csv`): 8 of the 59
  shipped CAN logs (two per car), the released `OPD_II.py` run unmodified against a local `valhalla_service` (port 8002,
  as in the authors' Docker set-up) and the re-implementation on the same decoded logs, both on one Valhalla tile set of
  the logs' area (drive network cut from the 2026-10-28 Geofabrik Italy north-east extract with the benchmark tiler).
  Start = first GPX point, bearing = integer compass bearing of the first 20 m (the tool asks for both by hand; they are
  not published). Point-by-point distance between the two outputs: median 0.00 to 0.33 m, p90 <= 3.6 m on all 8 logs
  (one log, duster_3, has 14 Meili failures and a 48 m maximum); closest-point distance to the GPX truth identical for
  both (1.7 to 202 m). The released tool takes about 1.6 s per log, the re-implementation 0.05 s (in-process Meili).
  The residual differences come from the released window loop (it appends the first packet past each 0.1 s boundary)
  and the number of trailing windows. The authors' reported overlap scores (87 to 100 %, cmpgpx against map-matched
  GPX) are not recomputed: their start values are unpublished and the 2024 map differs.

## Elastic Pathing (status: implemented with unit tests; open values chosen on val; formal runs done, run icde_v11)
- Source: Gao, Firner, Sugrim, Kaiser-Pendergrast, Yang, Lindqvist; version used: arXiv 1710.06932 (2017), which states
  that it supersedes UbiComp 2014 (doi 10.1145/2632048.2632077) and the 2013 report. No author code or data exists.
- Original assumptions: 1 Hz speed, known start, no heading, OSM speed limits, turn restrictions; output = top-n paths
  (destination error in the paper).
- Implemented (`src/cantraj/baselines/elastic_pathing.py`): Algorithm 1 (best-first by error, delta = 1.1),
  Algorithm 2 (speed limit + 20 mph pruning, feasible turn -> pin, infeasible -> compressA / stretchA, stop not at an
  intersection -> compressB / stretchB), Eq. 1 (8 % superelevation) and Eq. 3 (turn radius from road widths and
  deflection).
- Values the paper leaves open (fixed defaults; friction and "~0" to be chosen on val): friction 0.3, lane width 12 ft,
  lanes from OSM (missing: 1 per direction), "~0" = 0.5 m/s, compress / stretch search at most 10 samples and not before
  the last landmark, intersection tolerance = half the road width + 8 m, untagged speed limits not checked, error = sum of
  moved distances; dead ends drop the path. Ties in the error are not specified by the paper and are frequent (a path
  that never needs compress / stretch has error 0): the queue takes the furthest-advanced partial path first, then the
  successor order (edge id). With a zero-error complete path the delta rule stops at once, so the output is the first
  such path in this order. Tie-order sensitivity (`scripts/run_icde_ep_tie.sh`, `reports/icde/ep_tie_sensitivity.csv`;
  val only, the 378 selection episodes, three shuffled successor orders): median ADE 137.3 / 134.7 / 135.8 m vs 135.6 m,
  route recovered 0.58 to 0.60 vs 0.595, the same 5 failures; 61 to 73 % of the episodes keep their output within 1 m
  (paired median difference 0.0 m, paired mean +9 to +33 m). The tie order changes individual paths but not the
  aggregate numbers; the edge-id order stays.
- Adaptations: start = projection of p0 on the nearest edges (all directions = native heading-unknown, F-M-Hm);
  `elastic_pathing_hp` keeps start edges within 45 deg of theta0 (F-M-Hp, adapted); per-second positions from the path
  and the speed-integrated distance with the same landmarks; a state-dominance prune (same edge, sample, 10 m bin) and a
  budget (0.2 s per episode second, 30 to 300 s; 70 branch calls per second, 20k to 200k) end the search; no complete path
  = failure (counted, no output).

- Selection (`scripts/run_icde_ep_select.sh`, `reports/icde/ep_val_selection.csv`): 60 val drives of 36 drivers (378
  w05 episodes, heading-known variant), friction {0.05, 0.10, 0.15, 0.3, 0.5, 0.7} x stop threshold {0.3, 0.5, 1.0 m/s}:
  median ADE 136 m at friction 0.15 (route recovered 0.60, 5 failures) vs 248 m at 0.7; 0.05 and 0.10 tie on the
  median with more failures; the stop threshold barely matters. Fixed: friction 0.15, stop threshold 1.0 m/s.

## CAN-Trace (status: implemented with unit tests; val probe done; formal runs done, run icde_v11)
- Implemented (`src/cantraj/baselines/can_trace.py`): Algorithm 1 (zero-speed and pedal-minimum candidates, 1-D
  K-means gap thresholds, node merge below the shortest road segment, rectangle-rule edge weights) and Algorithm 2
  (path-pattern subgraph isomorphism by depth-first search over the undirected intersection graph of the visible v2
  map, tolerance sigma = 0.05 to 0.5 until 10 candidates, ranking by mean absolute edge difference). Variants:
  `can_trace` (known start: p0's road segment split at p0, both directions) and `can_trace_area` (the paper's prior,
  5 km square around p0, every intersection a possible first node). Open values fixed without tuning: 2 K-means
  clusters, stop = speed < 0.05 m/s, pedal minimum = within 1 % of the trace's pedal range.
- Val probe (w05, 496 episodes of 115 pedal-valid val drives): known start matched 31 episodes; the rest end with no
  match at sigma 0.5 (3 to 8 trace nodes). The method needs a stop or a pedal release at every intersection the
  vehicle drives through (one trace edge = one road segment), which naturalistic driving rarely provides; the area
  variant matched 249 episodes (any start junction in 25 km^2, so matches are easy and often wrong). This is the
  algorithm's behaviour, reported as failures, not tuned away. In the probe 41 area-variant episodes ended in an
  IndexError: their traces keep fewer than two nodes after the short-edge merge, so there is no line-graph edge and the
  empty match crashed the read-out. Fixed before the formal runs (they now fail as "fewer than 2 trace nodes"; same
  episodes, same outcome, correct reason; `tests/test_can_trace.py`).
- Formal runs (icde_v11, test + few-shot, pedal subset): w05 start-known 7 matches of 407 windows, area 149; matched
  windows are kilometres off (median ADE 2.8 to 3.6 km). Failure mode (`scripts/can_trace_failure_mode.py`): unmatched
  windows have a median of 6 trace nodes vs 14 intersections passed by the reference path; 77 % have fewer trace edges
  than intersections passed, so the one-segment-per-edge matcher has no exact solution.
- Source: Lin, Ma, Wang, Yu, He, Ni, Liu, "CAN-Trace Attack: Exploit CAN Messages to Uncover Driving Trajectories",
  IEEE T-ITS 26(3):3223-3236, 2025, doi 10.1109/TITS.2025.3532455 (arXiv 2507.09624). No code or data.
- Signals: speed and continuous accelerator pedal position (OBD); only the pedal-valid subset can run it (continuous gas
  valid on 48 / 302 test and 64 / 474 few-shot drives); the boolean gas-pressed flag is a different signal and would not
  carry the name.

## Dewri et al. 2013 (status: implemented with unit tests; no value tuned; formal runs done for 5- and 15-minute windows and whole drives; run icde_v11)
- Source: Dewri, Annadata, Eltarjaman, Thurimella, "Inferring Trip Destinations from Driving Habits Data", ACM WPES
  2013, pp. 267-272, doi 10.1145/2517840.2517871, read in the 9-page extended version (`related_work/2013WPES-Extended.pdf`).
  No author code; the trips are not released (the Denver road graph was, but it is not needed: we use the v2 map).
- Implemented (`src/cantraj/baselines/dewri.py`, `tests/test_dewri.py`): 1 Hz <t, s, d> samples; the two traffic
  pre-processing rules (stops less than half a mile apart, halts shorter than 3 s); stop-points; a depth-first search
  from the known start with the paper's four constraints (an intersection within 500 ft of every passed stop-point; a
  shortest path within 0.1 mile, reset along highways; turns above 60 deg only below 25 mph; the path ends beyond the
  trip length, fewest turns per end node); ranking by the augmented ideal-speed model (15 / 10 mph at left / right
  turns, 0 at stop-points, 25 / 6.5 ft/s^2) and the paper's Gaussian speed likelihood. Constants in the paper's units.
- Open values fixed a priori (no tuning; listed in the module docstring): zero speed = 1 Hz mean below 0.1 mph; the
  half-mile rule compares consecutive zero-speed runs; intersection = >= 3 visible undirected neighbours; turn angles
  from the nearest path edges of at least 1 m (zero-length OSM edges carry the heading); highway = motorway and its
  links; speed table motorway 60, trunk / primary 45, secondary / tertiary / links 35, residential / unclassified /
  road 25, living street 15 mph (OSM maxspeed ignored, as in the paper); M step-wise, M_aug the minimum of M and
  constant-rate envelopes; the likelihood integral to first order (a CDF difference underflows).
- Adaptations: samples one per second of CAN time (the paper's logger is 1 Hz); start = p0 projected on the nearest
  road edges, both directions (the paper starts at a node reached by GPS trimming); `dewri_hp` keeps start edges within
  45 deg of theta0; windows end while driving, so the last stop-point stays unmatched only when the car is stationary at
  the window end; the exhaustive search is bounded by Elastic Pathing's per-episode budget (expansions clip(70 n, 2e4,
  2e5), seconds as a safety net), shared equally by the start edges, paths accepted before the budget fires are kept;
  the paper ranks destinations, so positions per second are the CAN distance laid along each of the top 10 paths.
- Review (independent reviewers, three lenses; each finding confirmed by a separate verifier): reference-grid samples
  (fixed: CAN-time grid), a binding budget starving later start directions (fixed: shared budget), zero-length edges
  faking turns (fixed), a stop just before the window end (fixed); lows applied (shortest-path check at the end node,
  accepted paths kept when the budget fires, tests for the heading variant and the risky paths).
- Smoke (2 val drives, 11 w05 windows): every window has output, 0.2 s per window.
- Formal runs (10/31 03:05 to 03:51, `scripts/run_icde_spy_dewri.sh`, test + few-shot, w05 + w15, 3 workers): no
  value tuned. Failures (search exhausted, no path meets the constraints): w05 8 / 26 without heading, 18 / 54 with
  it (test / few-shot). w05 median ADE 293 / 487 m without heading, 81 / 89 m with it (Elastic Pathing 129 / 177 and
  87 / 80 m); same ADE within 0.5 m as Elastic Pathing on 25 / 31 % of the test windows; paired mean +929 m [380,
  1,417] without heading (more large errors), +13 m (CI includes 0) with it. Candidates: top-1 293 m, best of ten 49 m,
  best path first in 39 % (test, no heading). Tables: `tables/main_board.csv`, `pairs.csv`, `topk_w05.csv`,
  `spy_dewri_diag.csv`.
- Whole drives (10/31 07:37 to 08:15, same chain, 3 workers): failures 25 / 48 without heading, 37 / 68 with it
  (test / few-shot; 7 and 11 of them stop at the expansion budget, the rest exhaust the search). Median ADE 5,619 /
  2,172 m without heading, 5,382 / 1,758 m with it (Elastic Pathing 3,554 / 1,644 and 3,597 / 1,430 m); route
  recovered 11 / 19 % and 13 / 22 % (Elastic Pathing 15 / 25 and 17 / 29 %). Whole-drive pairs not computed.
- Cost (run icde_cost, 1 worker, the 92 val windows of the cost subset, a failed search timed until it gives up):
  median 0.04 s per 5-minute window without heading (90th percentile 0.09 s), 0.03 s with it; 0.33 / 0.23 s per
  15-minute window; peak 1.8 GB. Failures on the subset: 0 / 1 (w05 / w15) without heading, 1 / 2 with it.

## SPy / RoCuMa (status: implemented with unit tests; open values chosen on val; formal runs done for 5- and 15-minute windows and whole drives; run icde_v11)
- Source: Pese, Pu, Shin, "SPy: Car Steering Reveals Your Trip Route!", PoPETs 2020(2):155-174, doi
  10.2478/popets-2020-0022 (RoCuMa, road curvature matching, is its matching step). No code or data (MATLAB, never
  released; checked 2026-10-30: author pages, PETS page, GitHub).
- Signal and prior: steering wheel angle only (the paper excludes speed by design, p. 162; runs under G3, never reads
  speed, tested); no start point or heading, a rough area (p. 156, 168).
- Implemented (`src/cantraj/baselines/spy.py`, C kernel `src/cantraj/native/dtw.c` with a bit-exact Python reference,
  `tests/test_spy.py`): 10 Hz SWA (Table 5), moving average, turns at |SWA| > 90 deg (p. 162, kept for every vehicle),
  sections between turns, salient section = largest |SWA|, OSM curvature by an interpolating cubic spline every 1 m (Eq.
  2), z-scores (Eq. 4; Eq. 3 is a positive affine map, so z-scores make it irrelevant), Algorithm 1 over every contiguous
  run of 1..N segments of every road and direction, DTW as MATLAB's dtw (unnormalised, window 30 for runs below 4
  segments, none above), greedy chaining with the turn-sign rule (Eq. 5), no backtracking.
- Open values: moving-average window and the level that ends a turn's ramp chosen on validation; fixed a priori: merge
  turns closer than 1 s, exactly 4 segments without window, ties keep the first run. Selection
  (`scripts/run_icde_spy_dewri.sh` step 1, `reports/icde/spy_val_selection.csv`, rule fixed before the runs: route_ok,
  then path recall at 50 m, then median ADE): 60 val drives of 36 drivers (378 w05 windows), known-start variant, window
  5 or 15 samples x ramp level 20 or 45 deg; the four settings differ by at most 0.003 in route_ok and 0.007 in path
  recall (route_ok 0.10, path recall 0.24, median ADE 2.2 km for all); selected 5 samples and 45 deg.
- Adaptations: OSM names are not in our tiles, so a road is a directed stroke (same way through its nodes, at a way end
  the straightest successor within 30 deg, mutual links); no driveway trimming (windows start and end while driving);
  a chain turn needs |angle| >= 30 deg; runs longer than 50 m/s x the section duration skipped (physical cost guard);
  `spy_area` = the paper's prior with a 5 km square placed with p0 (as CAN-Trace); `spy` / `spy_hp` = known-start
  adaptation (the first section long enough to z-score matched from p0's projection, then the chain; `_hp` keeps start
  edges within 45 deg of theta0). Positions per second are an extension (the paper outputs a route): the method's own
  DTW warping path places each section along its run; turns sit at their intersection; the window's first / last
  sections get an open-ended alignment when the whole run is out of reach at 50 m/s (positions only). Budget: 1e8 DP
  cells per query second (clipped to 1e10 to 1e11), deterministic; a wall-clock safety net above it.
- Review (three lenses, then reproduction): a window starting inside a turn (fixed), window-edge sections stretched to
  a far junction (fixed as above), NaN into the C path routine (fixed: rejected), a straight continuation taken as a
  turn (fixed), clamped start projections (fixed: one start edge per stroke); fail fast when the C kernel is missing.
- Smoke (2 val drives, 11 w05 windows): every window has output; 0.3 s per window with a known start, 3.9 s (max 14 s)
  for the area variant. Analysis-only check: steering and reference-path curvature correlate positively (sign right).
- Formal runs (10/31 03:51 to 07:33, test + few-shot, w05 + w15, 3 workers, ma = 5, ramp 45 deg): known start 2 / 2
  failures (no admissible run), with heading 5 / 4, area 227 / 107 (DTW cell budget). w05 test / few-shot: route
  recovered 20 / 22 % with start heading (median ADE 1,474 / 1,284 m), 11 / 13 % without (2,198 / 1,810 m), 0 % with
  the 5 km area (3,929 / 3,400 m, above holding the start: 3,314 / 2,709 m). Paired with the beam (speed + steering,
  same start): +1,333 m [1,156, 1,476]. Fifteen-minute windows with at most one reference turn (399, analysis only):
  median path precision 1.00, recall 0.36 (beam 0.99): right roads, path too short. Route recovery falls with the
  number of reference turns (13 % -> 1 %), as for every method (beam 98 % -> 51 %). A sample-count explanation
  (unnormalised DTW cost lowest when the run has as many 1 m samples as the trace has 10 Hz samples, i.e. recall ~
  10 m/s / mean speed) was tested and rejected (Spearman -0.19 / -0.29), so the paper states no mechanism. Not
  comparable with the paper's 70.7 % (41 of 58 own trips in Ann Arbor, a database of 236 roads / 2,776 segments,
  first and last one or two segments removed by hand, p. 164 to 165).
- Whole drives (10/31 08:15 to 08:52, same chain, 3 workers): failures 5 / 5 without heading (8 at the DTW cell
  budget, 2 no admissible run), 4 / 4 with it (4 budget, 4 no admissible run). Route recovered 4 / 6 % with the start
  heading (median ADE 4,752 / 2,651 m), 2 / 3 % without (4,932 / 3,254 m); holding the start 5,235 / 3,322 m. Whole
  drives are the closest of our windows to the paper's complete trips and do not close the gap. The area variant was
  not run on whole drives (its search grows with the trace length; w05 already exhausts the budget on 227 windows).
- Cost (run icde_cost, 1 worker, 92 val windows, a failed search timed until it gives up): median per 5-minute
  window 0.13 s from the known start (90th percentile 0.68 s), 0.08 s with the start heading (0.32 s), 4.93 s for
  the 5 km area (13.53 s; 9 of the 76 windows stop at the DTW cell budget); 15-minute windows 0.24 / 0.22 / 7.71 s;
  peak 1.8 GB.
