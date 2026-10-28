# Inputs, priors, information use and causality per method (A6; benchmark v1.1)

Signal classes (configs/signal_registry.yaml): C0 = read directly from the car's bus; C1 = derived from CAN only (aEgo,
a Kalman derivative inside openpilot); I0 = raw device IMU; F = fusion output of openpilot's localizer / paramsd, which
used GPS in the openpilot versions of this data, so F signals are NOT CAN-only. Static vehicle parameters (wheelbase,
steering ratio, yaw-rate unit and sign) come from each drive's own carParams (platform median); train-fold calibration
is used only by variants named `_calib` / `calibrated`.

Shared conditions: the start position p0 (GPS at t0) is given in every protocol; theta0 (GPS course at t0) only in H+
protocols; the OSM map (v2 tiles, the episode's own disc) only in M protocols. Reference positions are unreadable while
methods run (`io.load_gps` raises). The GPS-vs-CAN time offset (qcom 0.7 s, u-blox 0.2 s, estimated on the train fold) is
applied to the REFERENCE before scoring and is the same for all methods.

| method | signals (class) | priors used | map | mode / look-ahead | GPS beyond p0 / theta0 | notes |
|---|---|---|---|---|---|---|
| hold start (`p0_hold`) | none | p0 | no | online | no | lower bound |
| start prior (`prior`) | speed (C0) | p0, theta0 | no | online | train-fold displacement statistics | |
| route retrieval (`retrieval`) | speed (C0) | p0, theta0 | no | online | train-fold routes of other drivers | identity-free |
| straight line (`dr_none`) | speed (C0) | p0, theta0 | no | online | no | |
| steering DR nominal (`dr_steer`) | speed, steering (C0) | p0, theta0 | no | online | no | carParams |
| steering DR online offset (`dr_steer_offonline`) | speed, steering (C0) | p0, theta0 | no | online | no | offset estimated inside the episode |
| steering DR calibrated (`dr_steer_calib`) | speed, steering (C0) | p0, theta0 | no | online | train-fold calibration (offset, ratio) | |
| CAN yaw DR (`dr_can_yaw`, G4) | speed, yawRate (C0) | p0, theta0 | no | online | no | 600 drives with a valid yawRate |
| gyro DR (`dr_imu_yaw_rls`, G8) | speed, steering (C0), gyro (I0) | p0, theta0 | no | online | no | rlog drives with IMU |
| curvature DR (`dr_curvature`, G5) | speed (C0), curvature (F) | p0, theta0 | no | online | yes, through the localizer | reference only, not on the CAN-only board |
| learned GRU / residual (`learned_*`) | speed, steering (C0) | p0, theta0 | no | online | trained on train-fold GPS | val for model selection |
| map PF (`map_pf_steer`, `map_pf_none`) | speed (+ steering) (C0) | p0 (+ theta0 in H+) | yes | online | no | 10 Hz causal ticks |
| map PF + learned source | speed, steering (C0) | p0 (+ theta0) | yes | online | trained on train-fold GPS | v1.1 timing fix |
| map beam (`map_beam_*`) | speed (+ steering) (C0) | p0 (+ theta0) | yes | offline (whole window) | no | deterministic, top-10 |
| MacroTracking (`macrotracking*`) | speed, steering (C0) | p0, theta0 | yes | online | `_calib`: train-fold calibration | re-implemented, author sample checked |
| OPD-II (`opd2`) | speed, steering (C0) | p0, theta0 | yes (Valhalla on the same network) | delayed online, <= 3 s | `_calib`: train-fold calibration | final output of the released tool |
| OPD-II online (`opd2_online`) | speed, steering (C0) | p0, theta0 | yes | online | no | matched points only once matched |
| Elastic Pathing (`elastic_pathing`) | speed (C0), 1 Hz means | p0 | yes | offline | no | heading unused (native); `_hp` uses theta0 |
| CAN-Trace (`can_trace`, pending) | speed, accelerator pedal (C0) | area (variant: p0) | yes | offline | no | pedal-valid drives only |

Causality tests in the suite: whitelist (`tests/test_whitelist.py`), raw-event causality (`test_causality_raw.py`),
suffix perturbation for the map PF (`test_map_pf.py`), the GRU wrapper (`test_learn.py`), MacroTracking
(`test_macrotracking.py`) and the OPD-II online read-out (`test_opd2.py`); the learned source inside the filter receives
exactly the logged event times (`test_v11_fixes.py`).
