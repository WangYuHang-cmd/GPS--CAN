# Baseline methods (draft for the paper, benchmark v1)

One paragraph per baseline family, written to be rewritten. Implementation: `src/cantraj/baselines/`, learned models
in `src/cantraj/learn/`. Every online baseline implements `reset(init, static, map_graph)`, `observe(events)` and
`predict(t)`; the replay engine (`cantraj.replay`) feeds events strictly in logged-time order and queries at the
reference seconds, so causality is enforced by construction rather than by convention. Baselines import no reference
loader (a unit test asserts it), and the runner replaces the GPS loader with a raising stub before any baseline runs.

## Trivial controls (G1)
- **Hold start** keeps the given position for the whole episode. It is the floor of the coverage metrics and the best
  no-map method once the heading is unknown and the horizon is long.
- **Start prior** moves along the given heading by rho(s) times the CAN distance s(t), where rho is the training-fold
  median ratio of net displacement to distance travelled (binned by distance). It encodes "roads bend" without any map.
- **Route retrieval without identity** (`retrieval`): a library of training-fold 1 Hz tracks with their CAN distance and
  speed profiles; candidates are library tracks that pass within 50 m of the start in either direction, ranked online
  by the RMSE between the query's causal speed profile and the candidate's over the same distance; the prediction is
  the library point s(t) metres along the best candidate. It never uses the driver label and is leave-one-driver-out
  when a training driver is queried.

## Dead reckoning (G1, G3, G4, G5, G8)
- **Speed only, straight line** (`dr_none`, G1) integrates the CAN speed along the given heading; it is the reference
  that any heading-rate source has to beat.
- **Steering dead reckoning** (`dr_steer`, G3) uses the bicycle model omega = v tan(delta / SR) / L with the drive's own
  logged wheelbase L and steering ratio SR, steps on every CAN event with an exact circular-arc update, holds the last
  speed across gaps and sets omega = 0 in a gap longer than a few nominal intervals. Variants: **online offset**
  (`dr_steer_offonline`) subtracts a causal time-weighted mean of the steering angle while moving faster than 5 m/s,
  shrunk towards zero with a 60 s prior ("roads are straight on average"); **train-fold calibrated**
  (`dr_steer_calib`) uses a per-platform steering offset and effective steering ratio regressed on the training
  fold, with a global fallback for unseen platforms.
- **CAN yaw rate** (`dr_can_yaw`, G4) integrates the vehicle's own yaw-rate message where it exists (600 drives), after
  the unit and sign conventions audited per platform.
- **Device gyro** (`dr_imu_yaw*`, G8): the uncalibrated phone gyro yaw axis (left-positive) minus a causal bias
  estimate: time-weighted mean while standing still (`imu_yaw`), mean difference to the nominal steering heading rate
  while moving (`imu_yaw_sb`), or a least-squares fit of gyro minus the steering term against speed that identifies the
  gyro bias and the steering offset jointly once the speed has varied enough (`imu_yaw_rls`). The raw bias is 0.02 to
  0.12 rad/s, so without an estimator the gyro is useless; with it the gyro equals the steering angle and stays far
  from the class F reference.
- **Localizer curvature** (`dr_curvature`, G5, class F) integrates the openpilot localizer's curvature times speed. It is
  GPS-informed in old logs and serves only as the "good heading-rate source" reference.
- **Heading unknown** (H- protocols): a dead-reckoned path is computed with an arbitrary heading and rotated about the
  start into K = 36 candidates; online methods output the most likely rotation under a uniform prior (the figure of
  merit degenerates to hold-start), offline no-map methods may output the K rotations as Top-K candidates.

## Online map particle filter (`map_pf_*`, O-M protocols)
2,000 particles live on the directed edges of the drive graph (an OSM graph cut from the disc around the start,
numpy arrays plus a k-d tree over edge midpoints, 5 million edges in 0.4 GB). Particles start on edges within 50 m of
p0 (radius doubled until some edge is found), all directions in H-, directions within 45 degrees of theta0 in H+. The
filter advances on a causal 10 Hz tick grid with the latest event at or before the tick (an event older than 0.3 s
is a gap, omega = 0): each particle moves v (1 +- 3% noise) dt along its edge; at a node it draws a successor with
log-probability -0.5 ((bearing(e') - theta) / 20 deg)^2, u-turns weighted 0.05; the particle heading follows the
CAN heading rate (steering, CAN yaw, gyro, class F curvature or a learned omega source, all through one hook) and is
pulled back towards the edge bearing with a 15 degree consistency term. Weights are resampled when the effective
sample size drops below half; the output is the weighted mean position. A map-only run with the speed-only group
(`map_pf_none`) uses uniform successors and no heading term, which isolates the value of the map from the value of
the steering signal.

## Offline whole-path beam search (`map_beam_*`, F-M protocols)
A deterministic beam (width 200) over hypotheses (edge, offset along edge, heading correction, score, parent). The
CAN stream is first integrated into cumulative distance s(t) and relative heading psi(t) at the reference seconds;
each step moves every hypothesis by the distance increment, spawns one child per successor edge at nodes with the
same turn and u-turn terms as the filter, adds a heading consistency term and relaxes the correction towards the
edge bearing (gain 0.3), allows an "early node crossing" child during turns to absorb speed-scale and map
simplification error (length tolerance 2% of distance, at least 10 m), penalises dead ends (weight 0.1) and merges
hypotheses on the same edge within 10 m bins (Viterbi). Top-1 is the whole-path maximum a posteriori back-traced from
the best final hypothesis; candidates 2 to 10 are the best hypotheses with distinct edge sequences, with softmax
probabilities (temperature 1). In H- the start hypotheses are every direction of every edge within the start radius,
so the unknown start heading is resolved by the whole path rather than by a rotation prior. No random numbers are
used; two runs produce byte-identical files.

## Learned motion models (G3; trained on train, selected on val, test once)
The data pipeline puts every drive on an absolute 0.1 s grid (zero-order hold of speed and steering with age and
missing flags, 0.3 s maximum age), uses the drive-level GPS positions as targets in the local frame, and samples
chunks of 60 / 120 / 300 s starting at valid anchors (curriculum: short chunks first). Features are speed, steering
angle, nominal curvature, steering rate, acceleration, age, missing, moving, sampling-rate class and the static
parameters; platform and brand are embedded with an "unknown" row that is trained with dropout (0.15 / 0.05) because
most test platforms are unseen. Loss: Huber on free-running position error (ADE) plus heading-increment and
distance-increment terms. Three variants, three seeds each (about 20k / 7k / 0.1k parameters):
- **Causal GRU** (`learned_gru`): GRU(hidden 64) over the features, emitting a speed scale and a heading rate
  (bounded, gated by moving and not-missing).
- **Physics residual + GRU** (`learned_resid`): omega = (a_p omega_nominal + b_p v + r_t) gate with hierarchical
  per-platform / brand / global parameters and a GRU(32) residual r_t; equals nominal dead reckoning when all
  parameters are zero, and first-order equals calibrated dead reckoning for the right (a_p, b_p).
- **Physics residual without GRU** (`learned_resid_nogru`): the same calibration learned from the position loss on all
  training drives (a "calibration by learning" control).
Inference is plain numpy with the same causal tick state machine as the filter, so the models run inside the online
wrapper (`learned`) and as the heading-rate source of the map particle filter (`map_pf_learned`). An attacker variant
retrained with 0.5 m/s speed-noise augmentation (`learned_gru_aug05`) is used in the robustness table.

## Warm start (research protocol rows)
The calibrating methods may consume the drive's own CAN / IMU events before t0 (W = 300 s or all past, never GPS):
dead reckoning updates its steering-offset and gyro-bias accumulators over the history in one vectorised pass
without moving the pose, and the GRU runs over the last 3,600 s of history before the pose is rebased to the start.
The steering map filter has no calibration state, so its warm start equals its cold start and is not run.

## Scoring
Predictions are matched to the reference seconds by timestamp (no interpolation of the reference); per-episode
metrics are ADE, FDE, P90, coverage at 10 / 25 / 50 / 100 / 200 / 500 m and its distance-weighted form, the
time-free path recall and precision at 25 / 50 / 100 m (polylines resampled at 5 m, nearest-sample distances), the
route-recovered flag (80% of the path within 50 m), the time until the error first exceeds 50 / 100 m, the
destination flags at the final second, success S(50 m, 0.9) and S(100 m, 0.8), the fde_gt_500 and lost_road flags
and, for offline Top-K rows, the oracle over candidates. Aggregates use medians and means with driver-cluster
bootstrap CIs; gains are paired differences on the common episodes of two methods.
