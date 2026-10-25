# Protocols v0.3 (draft for the release)

Task: absolute 2-D trajectory recovery from vehicle-bus signals given the start position p0 (always), optionally the
start heading theta0 (H+) and a public road graph (M). Eight protocols: {O online, F offline} x {N no map, M map} x {H+, H-}.

- Episode: [t0, t0 + D] with D in {60 s, 300 s, 900 s} or the full drive. t0 = the first 1 Hz reference fix at or after
  the nominal window start with GPS speed >= 3 m/s, 0 < bearing accuracy <= 5 deg (u-blox also h_acc < 10 m) and CAN
  speed >= 2.5 m/s. p0 = GPS(t0); theta0 = pi/2 - bearing(t0) (math heading, CCW from east), handed over in H+ only.
  H+ and H- episodes are identical otherwise. Windows without a qualifying anchor form the H- extension set.
- Inputs: raw CAN messages at their logged times (`events` layer), restricted to the signal group's whitelist and the
  episode interval; static parameters of the group (platform nominal values from carParams; training-fold calibration
  only in explicitly named variants). The released interpolated 10 Hz table is a separate `grid_compat` control.
- Online: a message is visible after its logged time; predictions are queried at the real reference timestamps after
  t0; no sequence length, end point or future gap mask. Offline: all whitelisted events of the episode at once.
- Map: tiles intersecting the disc of radius r = 5 km + 40 m/s * tau around p0 (tau = window duration; capped), built
  from the OSM extracts listed in manifests/map_tiles.csv (specs/map_policy.md). Never derived from the trajectory.
- Output: Top-1 position per reference time (episode ENU metres or WGS-84). Offline methods may add K whole-path
  candidates with probabilities; Top-K is scored by choosing one complete path.
- Reference: 1 Hz view (one real fix per second, never interpolated) with the training-fold time compensation Delta
  (manifests/time_alignment.json); a native 10 Hz view for drives with >= 8 Hz u-blox. Results are reported with and
  without Delta.
- Forbidden at test time: any GNSS quantity after t0, trip end point, route or driver identity, absolute wall-clock
  time, fusion-derived signals outside the flagged F groups, perception outputs.
