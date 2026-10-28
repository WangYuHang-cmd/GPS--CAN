# Main board (run icde_v11, benchmark v1.1; failures count as +inf / 0)

## 5-minute windows

| family | method | test n (fail) | test median ADE [CI] | test drv-eq | test c100 | test route | few-shot n (fail) | few-shot median ADE [CI] | few-shot drv-eq | few-shot c100 | few-shot route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no map, online | hold start | 1917 (0) | 3,314 [2,113, 4,313] | 2,113 | 0.03 | 0.00 | 1963 (0) | 2,709 [2,232, 3,562] | 2,095 | 0.03 | 0.00 |
| no map, online | straight line (speed) | 1917 (0) | 735 [644, 911] | 926 | 0.30 | 0.09 | 1963 (0) | 825 [781, 951] | 806 | 0.28 | 0.07 |
| no map, online | start prior (speed, train stats) | 1917 (0) | 1,068 [920, 1,119] | 1,081 | 0.12 | 0.02 | 1963 (0) | 1,062 [946, 1,162] | 1,014 | 0.13 | 0.02 |
| no map, online | route retrieval (speed, train routes) | 1917 (0) | 3,083 [2,002, 4,296] | 2,015 | 0.08 | 0.04 | 1963 (0) | 2,530 [2,016, 3,258] | 1,991 | 0.08 | 0.05 |
| no map, online | steering DR, nominal | 1917 (0) | 921 [461, 1,742] | 855 | 0.25 | 0.04 | 1963 (0) | 1,669 [906, 2,472] | 1,073 | 0.20 | 0.02 |
| no map, online | steering DR, online offset | 1917 (0) | 869 [611, 1,306] | 895 | 0.25 | 0.03 | 1963 (0) | 1,423 [964, 1,979] | 864 | 0.20 | 0.02 |
| no map, online | steering DR, train calibrated | 1917 (0) | 1,312 [409, 4,189] | 985 | 0.24 | 0.04 | 1963 (0) | 1,651 [919, 2,430] | 995 | 0.20 | 0.02 |
| no map, online | learned GRU (3 seeds) | 1917 (0) | 543 [226, 2,478] (seed SD 45) | 374 | 0.32 | 0.07 | 1963 (0) | 393 [305, 492] (seed SD 43) | 351 | 0.33 | 0.05 |
| map, online | map PF, steering | 1917 (0) | 43 [29, 59] | 41 | 0.83 | 0.87 | 1959 (0) | 27 [19, 47] | 33 | 0.88 | 0.90 |
| map, online | MacroTracking | 1917 (0) | 84 [66, 210] | 103 | 0.66 | 0.62 | 1959 (0) | 52 [40, 143] | 82 | 0.73 | 0.65 |
| map, online | MacroTracking, train calibrated | 1917 (0) | 76 [61, 172] | 72 | 0.68 | 0.65 | 1959 (0) | 48 [38, 126] | 65 | 0.74 | 0.67 |
| map, online | OPD-II, online read-out | 1917 (0) | 42 [27, 69] | 37 | 0.80 | 0.79 | 1959 (0) | 24 [15, 54] | 45 | 0.83 | 0.78 |
| map, offline | map beam, steering | 1917 (0) | 37 [24, 54] | 35 | 0.84 | 0.88 | 1959 (0) | 21 [17, 40] | 31 | 0.90 | 0.92 |
| map, offline | Elastic Pathing (start heading used) | 1917 (89) | 87 [38, 324] | 262 | 0.63 | 0.58 | 1959 (78) | 80 [54, 165] | 197 | 0.66 | 0.59 |
| map, offline | OPD-II (3 s delayed) | 1917 (0) | 42 [27, 68] | 37 | 0.80 | 0.79 | 1959 (0) | 24 [15, 54] | 45 | 0.83 | 0.78 |
| map, heading unknown | map PF, steering | 1917 (0) | 138 [99, 175] | 106 | 0.70 | 0.79 | 1959 (0) | 88 [61, 124] | 86 | 0.76 | 0.82 |
| map, heading unknown | map beam, steering | 1917 (0) | 41 [26, 59] | 35 | 0.77 | 0.81 | 1959 (0) | 23 [17, 42] | 31 | 0.84 | 0.87 |
| map, heading unknown | Elastic Pathing | 1917 (62) | 129 [42, 691] | 538 | 0.56 | 0.53 | 1959 (41) | 177 [94, 344] | 374 | 0.56 | 0.50 |
| map, heading unknown | CAN-Trace (pedal subset), start known | 245 (243) | fail [n/a, n/a] | fail | 0.00 | 0.00 | 162 (157) | fail [n/a, n/a] | fail | 0.00 | 0.00 |
| map, heading unknown | CAN-Trace (pedal subset), 5 km area | 245 (160) | fail [4,344, n/a] | fail | 0.00 | 0.00 | 162 (98) | fail [n/a, n/a] | fail | 0.00 | 0.00 |
| reference | curvature DR (class F, not CAN-only) | 1917 (0) | 84 [74, 107] | 85 | 0.64 | 0.27 | 1963 (0) | 80 [70, 92] | 85 | 0.67 | 0.27 |

## 15-minute windows

| family | method | test n (fail) | test median ADE [CI] | test drv-eq | test c100 | test route | few-shot n (fail) | few-shot median ADE [CI] | few-shot drv-eq | few-shot c100 | few-shot route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no map, online | hold start | 608 (0) | 8,759 [5,761, 12,194] | 6,109 | 0.02 | 0.00 | 565 (0) | 7,494 [5,973, 10,196] | 6,847 | 0.01 | 0.00 |
| no map, online | straight line (speed) | 608 (0) | 3,389 [3,124, 4,037] | 4,088 | 0.11 | 0.01 | 565 (0) | 3,906 [3,630, 4,335] | 4,160 | 0.10 | 0.01 |
| no map, online | start prior (speed, train stats) | 608 (0) | 3,601 [3,128, 3,841] | 3,645 | 0.05 | 0.00 | 565 (0) | 3,775 [3,545, 4,116] | 3,846 | 0.05 | 0.00 |
| no map, online | route retrieval (speed, train routes) | 608 (0) | 8,365 [5,617, 12,123] | 5,918 | 0.03 | 0.01 | 565 (0) | 7,326 [5,648, 9,704] | 6,958 | 0.04 | 0.03 |
| no map, online | steering DR, nominal | 608 (0) | 5,135 [2,276, 7,673] | 4,601 | 0.10 | 0.01 | 565 (0) | 7,174 [5,078, 9,162] | 6,093 | 0.07 | 0.00 |
| no map, online | steering DR, online offset | 608 (0) | 4,805 [3,893, 6,281] | 5,616 | 0.09 | 0.01 | 565 (0) | 6,570 [5,331, 8,340] | 5,538 | 0.07 | 0.00 |
| no map, online | steering DR, train calibrated | 608 (0) | 6,375 [2,069, 12,288] | 4,271 | 0.09 | 0.01 | 565 (0) | 7,138 [5,117, 9,181] | 5,826 | 0.07 | 0.00 |
| no map, online | learned GRU (3 seeds) | 608 (0) | 3,010 [1,193, 10,461] (seed SD 434) | 2,034 | 0.13 | 0.01 | 565 (0) | 2,569 [1,923, 3,128] (seed SD 241) | 2,574 | 0.12 | 0.00 |
| map, online | map PF, steering | 608 (0) | 178 [85, 373] | 489 | 0.50 | 0.69 | 561 (0) | 127 [52, 224] | 244 | 0.62 | 0.73 |
| map, online | MacroTracking | 608 (0) | 851 [293, 3,183] | 1,577 | 0.36 | 0.40 | 561 (0) | 1,064 [245, 4,190] | 1,031 | 0.47 | 0.39 |
| map, online | MacroTracking, train calibrated | 608 (0) | 779 [284, 2,875] | 1,384 | 0.36 | 0.39 | 561 (0) | 1,110 [155, 3,912] | 1,562 | 0.49 | 0.40 |
| map, online | OPD-II, online read-out | 608 (0) | 223 [100, 963] | 811 | 0.48 | 0.55 | 561 (0) | 497 [94, 2,073] | 634 | 0.55 | 0.47 |
| map, offline | map beam, steering | 608 (0) | 149 [68, 331] | 307 | 0.53 | 0.73 | 561 (0) | 106 [45, 174] | 119 | 0.66 | 0.76 |
| map, offline | Elastic Pathing (start heading used) | 608 (56) | 1,094 [124, 3,263] | 2,087 | 0.35 | 0.42 | 561 (47) | 1,223 [928, 1,861] | 2,110 | 0.40 | 0.42 |
| map, offline | OPD-II (3 s delayed) | 608 (0) | 223 [100, 962] | 810 | 0.48 | 0.55 | 561 (0) | 497 [94, 2,072] | 634 | 0.55 | 0.47 |
| map, heading unknown | map PF, steering | 608 (0) | 340 [201, 706] | 578 | 0.44 | 0.63 | 561 (0) | 242 [118, 481] | 276 | 0.57 | 0.68 |
| map, heading unknown | map beam, steering | 608 (0) | 155 [68, 376] | 432 | 0.51 | 0.70 | 561 (0) | 110 [46, 184] | 120 | 0.64 | 0.74 |
| map, heading unknown | Elastic Pathing | 608 (40) | 1,214 [135, 3,777] | 2,500 | 0.34 | 0.42 | 561 (30) | 1,602 [1,001, 2,136] | 2,144 | 0.37 | 0.40 |
| map, heading unknown | CAN-Trace (pedal subset), 5 km area | 79 (74) | fail [n/a, n/a] | fail | 0.00 | 0.00 | 45 (43) | fail [n/a, n/a] | fail | 0.00 | 0.00 |
| reference | curvature DR (class F, not CAN-only) | 608 (0) | 511 [445, 644] | 541 | 0.27 | 0.04 | 565 (0) | 491 [424, 595] | 545 | 0.27 | 0.02 |
