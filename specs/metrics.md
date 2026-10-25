# Metrics v0.3

Per episode, over the scored reference points i (1 Hz view after t0; a missing prediction counts as a miss):
- ADE = mean_i e_i, FDE = e at the last reference point, p50 / p90 of e_i;
- coverage C_r = share of points with e_i <= r for r in {10, 25, 50, 100} m;
- success S_(50 m, 0.9) = [C_50 >= 0.9] (primary), S_(100 m, 0.8) (secondary);
- strata: moving points (CAN speed > 1 m/s), points inside input gaps > 5 s; error vs elapsed time and vs CAN distance.
Top-K (offline only): ADE@K = min over the K complete candidate paths of that path's ADE; no per-point switching.
Aggregation: per-episode scores -> mean and median over episodes per (protocol, method, window class, split); 95%
percentile cluster-bootstrap CIs resampling parent drives (B = 1000); signal-group gains as paired differences on the
same episodes. H- shape metrics after one global rotation are reported separately and never on the absolute board.
