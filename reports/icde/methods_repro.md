# Published methods: sources, assumptions, adaptations, status (benchmark v1.1)

Rule for every method: no author data is available for our drives, so the original papers' numbers cannot be checked on
this benchmark; "re-implemented" never means "fully reproduced". Main rows follow the released code or the paper's
algorithm; every deviation is listed; parameters left open by the authors are fixed before any test run and chosen on
train / val only. Bugs found in released code are kept in the main row (the authors' results come from that code) and
fixed in a separately named sensitivity row.

## MacroTracking (status: implemented, author-sample check passed; formal runs queued)
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

## OPD-II (status: implemented; author-sample check passed; benchmark Valhalla tiles pending)
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

## Elastic Pathing (status: implemented with unit tests; open values chosen on val; formal runs started)
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

## CAN-Trace (status: implemented with unit tests; val probe done; formal runs started)
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
