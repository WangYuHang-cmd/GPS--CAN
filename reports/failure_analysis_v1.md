# Failure analysis and research strata (benchmark v1, run full_v1)

Research tables. The main board is unchanged; strata are reported as strata only (n >= 30 per cell; medians with driver-cluster bootstrap 95% CIs). `main` = test + few_shot_heldout, start and heading known.

## (a) When does a map-constrained estimate lose the road?

Onset = start of the terminal excursion beyond 100 m (NaN predictions count as beyond). A turn = a change of the reference heading by more than 45 degrees within 10 s. `within_30s_after_turn` = share of lost episodes whose onset lies within 30 s after a turn; `within_30s_chance_approx` = the share expected if onsets were spread uniformly over the episode (n_turns x 30 s / duration, capped at 1).

| protocol | method | win | n_episodes | lost_rate | onset_q25_s | onset_median_s | onset_q75_s | onset_first_60s | since_last_turn_median_s | within_30s_after_turn | within_30s_chance_approx | turns_median_lost | turns_median_kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F-M-Hp | map_beam_steer__G3 | full | 742 | 0.68 | 141.25 | 336.53 | 592.99 | 0.15 | 10.00 | 0.55 | 0.38 | 12.00 | 4.00 |
| F-M-Hp | map_beam_steer__G3 | w15 | 1173 | 0.71 | 188.99 | 339.05 | 541.99 | 0.12 | 9.10 | 0.37 | 0.23 | 4.00 | 3.00 |
| O-M-Hp | map_pf_none__G1 | full | 742 | 0.96 | 39.00 | 99.91 | 238.93 | 0.37 | 30.01 | 0.33 | 0.38 | 9.00 | 1.00 |
| O-M-Hp | map_pf_none__G1 | w15 | 1173 | 0.93 | 75.01 | 208.06 | 429.00 | 0.21 | 31.00 | 0.23 | 0.23 | 4.00 | 0.00 |
| O-M-Hp | map_pf_steer__G3 | full | 742 | 0.74 | 135.60 | 290.99 | 536.55 | 0.14 | 8.00 | 0.67 | 0.38 | 12.00 | 3.00 |
| O-M-Hp | map_pf_steer__G3 | w15 | 1173 | 0.74 | 185.00 | 337.00 | 535.99 | 0.11 | 8.99 | 0.43 | 0.23 | 5.00 | 3.00 |


## (b) Road-network density around the start

Density = map nodes within 2 km of the episode start per km2 (tertile cuts 53 and 244 nodes/km2 from the main-fold 5-minute episodes). Denser networks mean more branching decisions for the map methods.

| fold | protocol | method | win | level | n_episodes | n_drivers | ade_median | ade_median_lo | ade_median_hi | c100 | s50_90 | lost_road |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | O-M-Hp | map_pf_steer__G3 | w05 | low (< 53/km2) | 1293 | 36 | 30.38 | 18.62 | 73.52 | 0.82 | 0.50 | 0.25 |
| main | O-M-Hp | map_pf_steer__G3 | w05 | mid (53 to 244/km2) | 1293 | 82 | 39.37 | 27.47 | 48.87 | 0.83 | 0.42 | 0.28 |
| main | O-M-Hp | map_pf_steer__G3 | w05 | high (> 244/km2) | 1294 | 80 | 36.07 | 26.45 | 44.17 | 0.84 | 0.44 | 0.26 |
| main | O-M-Hp | map_pf_steer__G3 | w15 | low (< 53/km2) | 410 | 26 | 95.37 | 48.69 | 237.90 | 0.61 | 0.13 | 0.54 |
| main | O-M-Hp | map_pf_steer__G3 | w15 | mid (53 to 244/km2) | 389 | 65 | 220.73 | 132.93 | 447.24 | 0.50 | 0.10 | 0.70 |
| main | O-M-Hp | map_pf_steer__G3 | w15 | high (> 244/km2) | 374 | 60 | 265.22 | 171.98 | 476.23 | 0.51 | 0.12 | 0.67 |
| main | F-M-Hp | map_beam_steer__G3 | w05 | low (< 53/km2) | 1293 | 36 | 26.43 | 16.39 | 67.43 | 0.83 | 0.53 | 0.25 |
| main | F-M-Hp | map_beam_steer__G3 | w05 | mid (53 to 244/km2) | 1293 | 82 | 32.27 | 22.79 | 42.22 | 0.84 | 0.48 | 0.26 |
| main | F-M-Hp | map_beam_steer__G3 | w05 | high (> 244/km2) | 1294 | 80 | 31.78 | 21.45 | 37.82 | 0.85 | 0.48 | 0.26 |
| main | F-M-Hp | map_beam_steer__G3 | w15 | low (< 53/km2) | 410 | 26 | 83.98 | 45.64 | 222.23 | 0.62 | 0.13 | 0.56 |
| main | F-M-Hp | map_beam_steer__G3 | w15 | mid (53 to 244/km2) | 389 | 65 | 166.87 | 110.05 | 334.74 | 0.52 | 0.13 | 0.69 |
| main | F-M-Hp | map_beam_steer__G3 | w15 | high (> 244/km2) | 374 | 60 | 161.90 | 96.88 | 251.64 | 0.57 | 0.16 | 0.61 |
| main | O-M-Hp | map_pf_none__G1 | w05 | low (< 53/km2) | 1293 | 36 | 88.09 | 34.81 | 610.72 | 0.66 | 0.25 | 0.43 |
| main | O-M-Hp | map_pf_none__G1 | w05 | mid (53 to 244/km2) | 1293 | 82 | 678.39 | 237.89 | 953.86 | 0.49 | 0.12 | 0.62 |
| main | O-M-Hp | map_pf_none__G1 | w05 | high (> 244/km2) | 1294 | 80 | 855.58 | 610.08 | 1003.51 | 0.40 | 0.08 | 0.70 |
| main | O-M-Hp | map_pf_none__G1 | w15 | low (< 53/km2) | 410 | 26 | 2536.15 | 112.94 | 5774.18 | 0.39 | 0.01 | 0.78 |
| main | O-M-Hp | map_pf_none__G1 | w15 | mid (53 to 244/km2) | 389 | 65 | 3632.11 | 2585.29 | 4432.57 | 0.23 | 0.00 | 0.87 |
| main | O-M-Hp | map_pf_none__G1 | w15 | high (> 244/km2) | 374 | 60 | 3449.44 | 2742.43 | 4289.98 | 0.19 | 0.01 | 0.86 |
| main | O-N-Hp | dr_steer__G3 | w05 | low (< 53/km2) | 1293 | 36 | 2093.15 | 1119.69 | 3335.59 | 0.15 | 0.00 | 0.99 |
| main | O-N-Hp | dr_steer__G3 | w05 | mid (53 to 244/km2) | 1293 | 82 | 1254.18 | 823.58 | 1836.98 | 0.22 | 0.01 | 0.95 |
| main | O-N-Hp | dr_steer__G3 | w05 | high (> 244/km2) | 1294 | 80 | 632.28 | 421.60 | 1120.75 | 0.30 | 0.04 | 0.90 |
| main | O-N-Hp | dr_steer__G3 | w15 | low (< 53/km2) | 410 | 26 | 8599.35 | 6751.36 | 12,009 | 0.06 | 0.00 | 0.99 |
| main | O-N-Hp | dr_steer__G3 | w15 | mid (53 to 244/km2) | 389 | 65 | 6386.46 | 4851.29 | 7869.84 | 0.09 | 0.00 | 0.99 |
| main | O-N-Hp | dr_steer__G3 | w15 | high (> 244/km2) | 374 | 60 | 3399.28 | 1890.19 | 4880.29 | 0.13 | 0.00 | 0.97 |
| main | O-N-Hp | dr_curvature__G5 | w05 | low (< 53/km2) | 1293 | 36 | 85.42 | 74.39 | 107.86 | 0.65 | 0.09 | 0.67 |
| main | O-N-Hp | dr_curvature__G5 | w05 | mid (53 to 244/km2) | 1293 | 82 | 86.82 | 76.26 | 103.42 | 0.63 | 0.11 | 0.64 |
| main | O-N-Hp | dr_curvature__G5 | w05 | high (> 244/km2) | 1294 | 80 | 69.56 | 63.91 | 81.17 | 0.69 | 0.20 | 0.55 |
| main | O-N-Hp | dr_curvature__G5 | w15 | low (< 53/km2) | 410 | 26 | 514.46 | 465.83 | 686.08 | 0.25 | 0.00 | 0.95 |
| main | O-N-Hp | dr_curvature__G5 | w15 | mid (53 to 244/km2) | 389 | 65 | 548.25 | 446.59 | 688.55 | 0.25 | 0.00 | 0.95 |
| main | O-N-Hp | dr_curvature__G5 | w15 | high (> 244/km2) | 374 | 60 | 420.17 | 350.08 | 509.70 | 0.31 | 0.01 | 0.89 |
| val | O-M-Hp | map_pf_steer__G3 | w05 | low (< 53/km2) | 421 | 23 | 55.22 | 30.94 | 78.56 | 0.71 | 0.28 | 0.35 |
| val | O-M-Hp | map_pf_steer__G3 | w05 | mid (53 to 244/km2) | 1068 | 38 | 36.10 | 26.05 | 56.37 | 0.83 | 0.44 | 0.27 |
| val | O-M-Hp | map_pf_steer__G3 | w05 | high (> 244/km2) | 1428 | 38 | 35.28 | 27.33 | 46.33 | 0.83 | 0.45 | 0.26 |
| val | O-M-Hp | map_pf_steer__G3 | w15 | low (< 53/km2) | 131 | 19 | 333.20 | 201.47 | 2475.83 | 0.39 | 0.04 | 0.85 |
| val | O-M-Hp | map_pf_steer__G3 | w15 | mid (53 to 244/km2) | 317 | 30 | 417.55 | 272.20 | 539.99 | 0.51 | 0.10 | 0.72 |
| val | O-M-Hp | map_pf_steer__G3 | w15 | high (> 244/km2) | 427 | 30 | 380.78 | 233.40 | 561.27 | 0.48 | 0.13 | 0.72 |
| val | F-M-Hp | map_beam_steer__G3 | w05 | low (< 53/km2) | 421 | 23 | 46.14 | 22.43 | 70.76 | 0.73 | 0.33 | 0.34 |
| val | F-M-Hp | map_beam_steer__G3 | w05 | mid (53 to 244/km2) | 1068 | 38 | 29.27 | 22.72 | 48.22 | 0.84 | 0.50 | 0.26 |
| val | F-M-Hp | map_beam_steer__G3 | w05 | high (> 244/km2) | 1428 | 38 | 27.59 | 20.90 | 35.79 | 0.85 | 0.52 | 0.23 |
| val | F-M-Hp | map_beam_steer__G3 | w15 | low (< 53/km2) | 131 | 19 | 163.99 | 103.85 | 1918.45 | 0.44 | 0.07 | 0.77 |
| val | F-M-Hp | map_beam_steer__G3 | w15 | mid (53 to 244/km2) | 317 | 30 | 170.36 | 118.91 | 285.59 | 0.59 | 0.18 | 0.62 |
| val | F-M-Hp | map_beam_steer__G3 | w15 | high (> 244/km2) | 427 | 30 | 169.12 | 110.43 | 263.52 | 0.57 | 0.17 | 0.66 |
| val | O-M-Hp | map_pf_none__G1 | w05 | low (< 53/km2) | 421 | 23 | 811.62 | 127.01 | 1784.34 | 0.51 | 0.07 | 0.57 |
| val | O-M-Hp | map_pf_none__G1 | w05 | mid (53 to 244/km2) | 1068 | 38 | 893.47 | 428.87 | 1140.94 | 0.39 | 0.04 | 0.71 |
| val | O-M-Hp | map_pf_none__G1 | w05 | high (> 244/km2) | 1428 | 38 | 774.54 | 519.16 | 851.08 | 0.35 | 0.05 | 0.75 |
| val | O-M-Hp | map_pf_none__G1 | w15 | low (< 53/km2) | 131 | 19 | 4863.07 | 3147.15 | 11,328 | 0.22 | 0.01 | 0.94 |
| val | O-M-Hp | map_pf_none__G1 | w15 | mid (53 to 244/km2) | 317 | 30 | 3294.46 | 2431.72 | 3792.53 | 0.17 | 0.01 | 0.95 |
| val | O-M-Hp | map_pf_none__G1 | w15 | high (> 244/km2) | 427 | 30 | 2839.03 | 2429.81 | 3643.27 | 0.15 | 0.00 | 0.95 |
| val | O-N-Hp | dr_steer__G3 | w05 | low (< 53/km2) | 421 | 23 | 2070.13 | 1655.97 | 3746.37 | 0.17 | 0.02 | 0.97 |
| val | O-N-Hp | dr_steer__G3 | w05 | mid (53 to 244/km2) | 1068 | 38 | 984.83 | 648.33 | 1671.11 | 0.24 | 0.02 | 0.94 |
| val | O-N-Hp | dr_steer__G3 | w05 | high (> 244/km2) | 1428 | 38 | 444.76 | 351.66 | 931.38 | 0.35 | 0.06 | 0.87 |
| val | O-N-Hp | dr_steer__G3 | w15 | low (< 53/km2) | 131 | 19 | 8796.84 | 6074.62 | 11,690 | 0.07 | 0.01 | 0.98 |
| val | O-N-Hp | dr_steer__G3 | w15 | mid (53 to 244/km2) | 317 | 30 | 4335.35 | 3360.73 | 6494.59 | 0.09 | 0.01 | 0.99 |
| val | O-N-Hp | dr_steer__G3 | w15 | high (> 244/km2) | 427 | 30 | 2621.10 | 1889.41 | 5169.50 | 0.15 | 0.01 | 0.97 |
| val | O-N-Hp | dr_curvature__G5 | w05 | low (< 53/km2) | 421 | 23 | 94.52 | 78.56 | 117.99 | 0.62 | 0.08 | 0.68 |
| val | O-N-Hp | dr_curvature__G5 | w05 | mid (53 to 244/km2) | 1068 | 38 | 78.01 | 68.36 | 98.95 | 0.65 | 0.16 | 0.60 |
| val | O-N-Hp | dr_curvature__G5 | w05 | high (> 244/km2) | 1428 | 38 | 63.30 | 57.36 | 85.04 | 0.72 | 0.21 | 0.50 |
| val | O-N-Hp | dr_curvature__G5 | w15 | low (< 53/km2) | 131 | 19 | 565.22 | 504.92 | 836.21 | 0.27 | 0.01 | 0.92 |
| val | O-N-Hp | dr_curvature__G5 | w15 | mid (53 to 244/km2) | 317 | 30 | 442.47 | 391.91 | 549.79 | 0.27 | 0.01 | 0.91 |
| val | O-N-Hp | dr_curvature__G5 | w15 | high (> 244/km2) | 427 | 30 | 339.29 | 275.94 | 571.41 | 0.34 | 0.03 | 0.86 |


## (c) Vehicle platform seen vs unseen in the training fold (stratum only)

Platform labels are taken as recorded in the logs (no corrections); this table describes where the episodes fall, not vehicle effects. Confounded with driver, region and receiver.

| fold | protocol | method | win | level | n_episodes | n_drivers | n_platforms | ade_median | ade_median_lo | ade_median_hi | c100 | s50_90 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| main | O-N-Hp | dr_steer__G3 | w05 | seen in train | 2568 | 61 | 46 | 1452.30 | 919.44 | 2144.07 | 0.21 | 0.01 |
| main | O-N-Hp | dr_steer__G3 | w05 | unseen | 1312 | 40 | 42 | 906.30 | 487.40 | 1808.49 | 0.25 | 0.02 |
| main | O-N-Hp | dr_steer_calib__G3 | w05 | seen in train | 2568 | 61 | 46 | 2075.94 | 913.01 | 3655.97 | 0.19 | 0.01 |
| main | O-N-Hp | dr_steer_calib__G3 | w05 | unseen | 1312 | 40 | 42 | 847.07 | 466.51 | 1723.78 | 0.27 | 0.03 |
| main | O-N-Hp | learned_gru_s0__G3 | w05 | seen in train | 2568 | 61 | 46 | 557.54 | 339.61 | 1440.44 | 0.29 | 0.03 |
| main | O-N-Hp | learned_gru_s0__G3 | w05 | unseen | 1312 | 40 | 42 | 302.31 | 218.74 | 416.36 | 0.39 | 0.05 |
| main | O-M-Hp | map_pf_steer__G3 | w05 | seen in train | 2568 | 61 | 46 | 26.76 | 19.87 | 45.16 | 0.86 | 0.56 |
| main | O-M-Hp | map_pf_steer__G3 | w05 | unseen | 1312 | 40 | 42 | 55.98 | 45.33 | 72.89 | 0.76 | 0.26 |
| main | F-M-Hp | map_beam_steer__G3 | w05 | seen in train | 2568 | 61 | 46 | 21.85 | 17.12 | 38.14 | 0.88 | 0.60 |
| main | F-M-Hp | map_beam_steer__G3 | w05 | unseen | 1312 | 40 | 42 | 50.95 | 38.23 | 66.24 | 0.77 | 0.29 |
| val | O-N-Hp | dr_steer__G3 | w05 | seen in train | 1557 | 33 | 32 | 873.85 | 558.79 | 1515.62 | 0.25 | 0.02 |
| val | O-N-Hp | dr_steer__G3 | w05 | unseen | 1360 | 18 | 35 | 596.66 | 395.45 | 1849.92 | 0.32 | 0.05 |
| val | O-N-Hp | dr_steer_calib__G3 | w05 | seen in train | 1557 | 33 | 32 | 820.43 | 516.85 | 1445.45 | 0.26 | 0.03 |
| val | O-N-Hp | dr_steer_calib__G3 | w05 | unseen | 1360 | 18 | 35 | 560.54 | 358.97 | 1900.29 | 0.33 | 0.06 |
| val | O-N-Hp | learned_gru_s0__G3 | w05 | seen in train | 1557 | 33 | 32 | 245.17 | 186.51 | 356.73 | 0.41 | 0.05 |
| val | O-N-Hp | learned_gru_s0__G3 | w05 | unseen | 1360 | 18 | 35 | 210.05 | 178.33 | 351.72 | 0.46 | 0.07 |
| val | O-M-Hp | map_pf_steer__G3 | w05 | seen in train | 1557 | 33 | 32 | 34.21 | 25.98 | 54.70 | 0.82 | 0.46 |
| val | O-M-Hp | map_pf_steer__G3 | w05 | unseen | 1360 | 18 | 35 | 43.56 | 34.08 | 57.47 | 0.80 | 0.38 |
| val | F-M-Hp | map_beam_steer__G3 | w05 | seen in train | 1557 | 33 | 32 | 27.18 | 21.35 | 43.89 | 0.84 | 0.53 |
| val | F-M-Hp | map_beam_steer__G3 | w05 | unseen | 1360 | 18 | 35 | 36.04 | 27.43 | 52.54 | 0.82 | 0.43 |
