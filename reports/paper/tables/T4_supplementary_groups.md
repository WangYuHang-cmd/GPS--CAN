| method                                  | fold     |    n | median_ADE_CI      |   c100 | paired_vs_steering_DR             |
|:----------------------------------------|:---------|-----:|:-------------------|-------:|:----------------------------------|
| CAN yaw-rate DR (G4)                    | val      |  342 | 728 [234, 944]     |   0.30 | -169 [-454, +7], win 0.65         |
| CAN yaw-rate DR (G4)                    | test     |  658 | 851 [151, 1,529]   |   0.28 | +519 [-2,290, +914], win 0.32     |
| CAN yaw-rate DR (G4)                    | few-shot |  188 | 521 [81, 1,831]    |   0.39 | -338 [-1,161, -69], win 0.72      |
| device gyro DR, least-squares bias (G8) | val      | 1056 | 691 [524, 1,111]   |   0.29 | +31 [-24, +67], win 0.45          |
| device gyro DR, least-squares bias (G8) | test     |  667 | 937 [687, 1,524]   |   0.24 | +28 [-50, +73], win 0.46          |
| device gyro DR, least-squares bias (G8) | few-shot |  732 | 1,337 [820, 2,464] |   0.21 | -39 [-110, +9], win 0.54          |
| curvature DR, class F reference (G5)    | val      | 2917 | 74 [66, 92]        |   0.68 | reference (class F, not CAN-only) |
| curvature DR, class F reference (G5)    | test     | 1917 | 84 [74, 98]        |   0.64 | reference (class F, not CAN-only) |
| curvature DR, class F reference (G5)    | few-shot | 1963 | 80 [70, 92]        |   0.67 | reference (class F, not CAN-only) |
