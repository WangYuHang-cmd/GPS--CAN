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
| no map, online | learned physics residual (3 seeds) | 1917 (0) | 690 [310, 2,994] (seed SD 160) | 478 | 0.29 | 0.06 | 1963 (0) | 435 [366, 536] (seed SD 45) | 441 | 0.32 | 0.04 |
| map, online | map PF, speed only | 1917 (0) | 286 [50, 868] | 662 | 0.55 | 0.56 | 1959 (0) | 618 [186, 890] | 684 | 0.52 | 0.48 |
| map, online | map PF, steering | 1917 (0) | 43 [29, 59] | 41 | 0.83 | 0.87 | 1959 (0) | 27 [19, 47] | 33 | 0.88 | 0.90 |
| map, online | map PF, learned GRU motion (3 seeds) | 1917 (0) | 89 [42, 168] (seed SD 10) | 65 | 0.65 | 0.86 | 1959 (0) | 42 [29, 79] (seed SD 5) | 40 | 0.79 | 0.86 |
| map, online | map PF, learned residual motion (3 seeds) | 1917 (0) | 87 [45, 262] (seed SD 3) | 52 | 0.62 | 0.86 | 1959 (0) | 53 [33, 84] (seed SD 4) | 45 | 0.78 | 0.86 |
| map, online | MacroTracking | 1917 (0) | 84 [66, 210] | 103 | 0.66 | 0.62 | 1959 (0) | 52 [40, 143] | 82 | 0.73 | 0.65 |
| map, online | MacroTracking, train calibrated | 1917 (0) | 76 [61, 172] | 72 | 0.68 | 0.65 | 1959 (0) | 48 [38, 126] | 65 | 0.74 | 0.67 |
| map, online | OPD-II, online read-out | 1917 (0) | 42 [27, 69] | 37 | 0.80 | 0.79 | 1959 (0) | 24 [15, 54] | 45 | 0.83 | 0.78 |
| map, offline | map beam, speed only | 1917 (0) | 165 [37, 633] | 546 | 0.57 | 0.54 | 1959 (0) | 343 [88, 637] | 443 | 0.54 | 0.47 |
| map, offline | map beam, steering | 1917 (0) | 37 [24, 54] | 35 | 0.84 | 0.88 | 1959 (0) | 21 [17, 40] | 31 | 0.90 | 0.92 |
| map, offline | map PF, steering, best-lineage read-out | 1917 (0) | 39 [28, 56] | 34 | 0.83 | 0.87 | 1959 (0) | 24 [17, 44] | 29 | 0.89 | 0.90 |
| map, offline | Elastic Pathing (start heading used) | 1917 (89) | 87 [38, 324] | 262 | 0.63 | 0.58 | 1959 (78) | 80 [54, 165] | 197 | 0.66 | 0.59 |
| map, offline | OPD-II (3 s delayed) | 1917 (0) | 42 [27, 68] | 37 | 0.80 | 0.79 | 1959 (0) | 24 [15, 54] | 45 | 0.83 | 0.78 |
| map, offline | Dewri et al. (start heading used) | 1917 (18) | 81 [38, 215] | 272 | 0.65 | 0.62 | 1959 (54) | 89 [59, 157] | 183 | 0.65 | 0.59 |
| map, offline | SPy, start known (start heading used) | 1917 (5) | 1,474 [1,283, 1,602] | 1,299 | 0.12 | 0.20 | 1959 (4) | 1,284 [1,090, 1,596] | 1,401 | 0.15 | 0.22 |
| map, heading unknown | map PF, steering | 1917 (0) | 138 [99, 175] | 106 | 0.70 | 0.79 | 1959 (0) | 88 [61, 124] | 86 | 0.76 | 0.82 |
| map, heading unknown | map PF, speed only | 1917 (0) | 1,773 [1,550, 1,987] | 1,376 | 0.35 | 0.34 | 1959 (0) | 1,868 [1,572, 2,097] | 1,770 | 0.32 | 0.29 |
| map, heading unknown | map beam, steering | 1917 (0) | 41 [26, 59] | 35 | 0.77 | 0.81 | 1959 (0) | 23 [17, 42] | 31 | 0.84 | 0.87 |
| map, heading unknown | map beam, speed only | 1917 (0) | 1,314 [866, 1,619] | 1,378 | 0.35 | 0.32 | 1959 (0) | 1,527 [1,294, 1,738] | 1,328 | 0.33 | 0.29 |
| map, heading unknown | Elastic Pathing | 1917 (62) | 129 [42, 691] | 538 | 0.56 | 0.53 | 1959 (41) | 177 [94, 344] | 374 | 0.56 | 0.50 |
| map, heading unknown | CAN-Trace (pedal subset), start known | 245 (243) | fail [n/a, n/a] | fail | 0.00 | 0.00 | 162 (157) | fail [n/a, n/a] | fail | 0.00 | 0.00 |
| map, heading unknown | CAN-Trace (pedal subset), 5 km area | 245 (160) | fail [4,344, n/a] | fail | 0.00 | 0.00 | 162 (98) | fail [n/a, n/a] | fail | 0.00 | 0.00 |
| map, heading unknown | Dewri et al. | 1917 (8) | 293 [81, 775] | 545 | 0.49 | 0.46 | 1959 (26) | 487 [272, 708] | 757 | 0.48 | 0.43 |
| map, heading unknown | SPy, start known | 1917 (2) | 2,198 [1,702, 2,636] | 1,709 | 0.08 | 0.11 | 1959 (2) | 1,810 [1,512, 2,211] | 1,755 | 0.11 | 0.13 |
| map, heading unknown | SPy, 5 km area | 1917 (227) | 3,929 [3,222, 4,406] | 3,169 | 0.01 | 0.00 | 1959 (107) | 3,400 [3,050, 3,953] | 3,207 | 0.01 | 0.00 |
| reference | curvature DR (class F, not CAN-only) | 1917 (0) | 84 [74, 107] | 85 | 0.64 | 0.27 | 1963 (0) | 80 [70, 92] | 85 | 0.67 | 0.27 |
| reference | map PF, curvature (class F) | 1917 (0) | 44 [29, 61] | 44 | 0.82 | 0.86 | 1959 (0) | 28 [19, 49] | 35 | 0.87 | 0.89 |

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
| no map, online | learned physics residual (3 seeds) | 608 (0) | 4,799 [1,785, 10,746] (seed SD 1386) | 2,850 | 0.12 | 0.02 | 565 (0) | 2,834 [2,292, 3,635] (seed SD 288) | 3,226 | 0.12 | 0.00 |
| map, online | map PF, speed only | 608 (0) | 2,326 [364, 4,401] | 3,711 | 0.31 | 0.37 | 561 (0) | 3,444 [2,231, 4,825] | 2,879 | 0.26 | 0.27 |
| map, online | map PF, steering | 608 (0) | 178 [85, 373] | 489 | 0.50 | 0.69 | 561 (0) | 127 [52, 224] | 244 | 0.62 | 0.73 |
| map, online | MacroTracking | 608 (0) | 851 [293, 3,183] | 1,577 | 0.36 | 0.40 | 561 (0) | 1,064 [245, 4,190] | 1,031 | 0.47 | 0.39 |
| map, online | MacroTracking, train calibrated | 608 (0) | 779 [284, 2,875] | 1,384 | 0.36 | 0.39 | 561 (0) | 1,110 [155, 3,912] | 1,562 | 0.49 | 0.40 |
| map, online | OPD-II, online read-out | 608 (0) | 223 [100, 963] | 811 | 0.48 | 0.55 | 561 (0) | 497 [94, 2,073] | 634 | 0.55 | 0.47 |
| map, offline | map beam, speed only | 608 (0) | 2,142 [136, 4,651] | 3,540 | 0.32 | 0.40 | 561 (0) | 3,084 [1,498, 4,609] | 2,803 | 0.29 | 0.30 |
| map, offline | map beam, steering | 608 (0) | 149 [68, 331] | 307 | 0.53 | 0.73 | 561 (0) | 106 [45, 174] | 119 | 0.66 | 0.76 |
| map, offline | map PF, steering, best-lineage read-out | 608 (0) | 165 [79, 394] | 490 | 0.50 | 0.68 | 561 (0) | 125 [46, 215] | 240 | 0.63 | 0.72 |
| map, offline | Elastic Pathing (start heading used) | 608 (56) | 1,094 [124, 3,263] | 2,087 | 0.35 | 0.42 | 561 (47) | 1,223 [928, 1,861] | 2,110 | 0.40 | 0.42 |
| map, offline | OPD-II (3 s delayed) | 608 (0) | 223 [100, 962] | 810 | 0.48 | 0.55 | 561 (0) | 497 [94, 2,072] | 634 | 0.55 | 0.47 |
| map, offline | Dewri et al. (start heading used) | 608 (32) | 1,053 [142, 2,985] | 2,380 | 0.35 | 0.44 | 561 (51) | 2,058 [1,159, 2,764] | 3,022 | 0.36 | 0.37 |
| map, offline | SPy, start known (start heading used) | 608 (3) | 5,396 [4,436, 6,077] | 5,169 | 0.05 | 0.06 | 561 (3) | 4,819 [4,176, 6,440] | 4,718 | 0.07 | 0.06 |
| map, heading unknown | map PF, steering | 608 (0) | 340 [201, 706] | 578 | 0.44 | 0.63 | 561 (0) | 242 [118, 481] | 276 | 0.57 | 0.68 |
| map, heading unknown | map PF, speed only | 608 (0) | 5,799 [4,754, 6,909] | 5,254 | 0.19 | 0.15 | 561 (0) | 5,901 [4,652, 6,912] | 4,575 | 0.16 | 0.10 |
| map, heading unknown | map beam, steering | 608 (0) | 155 [68, 376] | 432 | 0.51 | 0.70 | 561 (0) | 110 [46, 184] | 120 | 0.64 | 0.74 |
| map, heading unknown | map beam, speed only | 608 (0) | 5,099 [3,859, 6,250] | 4,886 | 0.20 | 0.25 | 561 (0) | 5,419 [3,588, 6,052] | 4,969 | 0.19 | 0.21 |
| map, heading unknown | Elastic Pathing | 608 (40) | 1,214 [135, 3,777] | 2,500 | 0.34 | 0.42 | 561 (30) | 1,602 [1,001, 2,136] | 2,144 | 0.37 | 0.40 |
| map, heading unknown | CAN-Trace (pedal subset), 5 km area | 79 (74) | fail [n/a, n/a] | fail | 0.00 | 0.00 | 45 (43) | fail [n/a, n/a] | fail | 0.00 | 0.00 |
| map, heading unknown | Dewri et al. | 608 (15) | 3,064 [1,642, 4,239] | 3,128 | 0.26 | 0.32 | 561 (27) | 3,700 [2,838, 4,731] | 3,779 | 0.30 | 0.27 |
| map, heading unknown | SPy, start known | 608 (1) | 6,199 [4,874, 7,741] | 5,207 | 0.04 | 0.04 | 561 (2) | 5,928 [4,883, 7,689] | 5,734 | 0.05 | 0.04 |
| map, heading unknown | SPy, 5 km area | 608 (40) | 9,871 [6,632, 12,216] | 7,286 | 0.00 | 0.00 | 561 (27) | 8,349 [6,967, 10,942] | 7,782 | 0.00 | 0.00 |
| reference | curvature DR (class F, not CAN-only) | 608 (0) | 511 [445, 644] | 541 | 0.27 | 0.04 | 565 (0) | 491 [424, 595] | 545 | 0.27 | 0.02 |
| reference | map PF, curvature (class F) | 608 (0) | 186 [86, 430] | 476 | 0.49 | 0.67 | 561 (0) | 138 [61, 226] | 224 | 0.61 | 0.71 |

## whole-drive windows

| family | method | test n (fail) | test median ADE [CI] | test drv-eq | test c100 | test route | few-shot n (fail) | few-shot median ADE [CI] | few-shot drv-eq | few-shot c100 | few-shot route |
|---|---|---|---|---|---|---|---|---|---|---|---|
| no map, online | hold start | 282 (0) | 5,235 [3,825, 7,854] | 4,631 | 0.03 | 0.00 | 460 (0) | 3,322 [2,580, 4,870] | 4,809 | 0.04 | 0.00 |
| no map, online | straight line (speed) | 282 (0) | 6,551 [4,548, 9,073] | 5,319 | 0.10 | 0.02 | 460 (0) | 3,327 [2,389, 4,744] | 4,402 | 0.19 | 0.08 |
| no map, online | start prior (speed, train stats) | 282 (0) | 5,771 [3,750, 7,481] | 5,391 | 0.08 | 0.02 | 460 (0) | 2,912 [2,054, 3,825] | 4,041 | 0.13 | 0.06 |
| no map, online | route retrieval (speed, train routes) | 282 (0) | 5,586 [3,843, 7,854] | 4,926 | 0.04 | 0.01 | 460 (0) | 3,236 [2,360, 4,800] | 4,137 | 0.07 | 0.03 |
| no map, online | steering DR, nominal | 282 (0) | 2,708 [1,468, 8,341] | 5,237 | 0.17 | 0.05 | 460 (0) | 2,674 [1,251, 3,916] | 3,575 | 0.22 | 0.07 |
| no map, online | steering DR, online offset | 282 (0) | 5,004 [3,266, 6,809] | 3,863 | 0.13 | 0.04 | 460 (0) | 3,118 [1,868, 4,449] | 4,458 | 0.21 | 0.08 |
| no map, online | steering DR, train calibrated | 282 (0) | 3,121 [1,917, 8,403] | 4,617 | 0.15 | 0.04 | 460 (0) | 2,656 [1,498, 3,902] | 3,579 | 0.21 | 0.07 |
| no map, online | learned GRU (3 seeds) | 282 (0) | 1,704 [1,149, 4,563] (seed SD 79) | 2,038 | 0.20 | 0.06 | 460 (0) | 887 [562, 1,547] (seed SD 12) | 1,414 | 0.30 | 0.11 |
| no map, online | learned physics residual (3 seeds) | 282 (0) | 1,944 [1,255, 5,372] (seed SD 261) | 2,620 | 0.20 | 0.06 | 460 (0) | 1,100 [559, 1,838] (seed SD 124) | 1,796 | 0.29 | 0.10 |
| map, online | map PF, speed only | 282 (0) | 5,827 [3,211, 7,147] | 4,000 | 0.13 | 0.08 | 454 (0) | 2,908 [1,996, 3,946] | 3,603 | 0.20 | 0.13 |
| map, online | map PF, steering | 282 (0) | 1,906 [799, 3,128] | 1,586 | 0.36 | 0.33 | 454 (0) | 193 [91, 741] | 616 | 0.61 | 0.57 |
| map, online | MacroTracking | 282 (0) | 1,649 [779, 4,479] | 1,852 | 0.42 | 0.33 | 454 (0) | 227 [73, 900] | 1,027 | 0.57 | 0.46 |
| map, online | OPD-II, online read-out | 282 (0) | 1,712 [913, 4,120] | 1,079 | 0.41 | 0.35 | 454 (0) | 407 [52, 1,231] | 1,020 | 0.59 | 0.50 |
| map, offline | map beam, speed only | 282 (0) | 5,930 [4,134, 8,365] | 4,303 | 0.14 | 0.08 | 454 (0) | 3,095 [2,044, 3,941] | 4,035 | 0.21 | 0.12 |
| map, offline | map beam, steering | 282 (0) | 1,179 [343, 1,729] | 756 | 0.42 | 0.41 | 454 (0) | 94 [44, 239] | 159 | 0.65 | 0.61 |
| map, offline | Elastic Pathing (start heading used) | 282 (31) | 3,597 [1,952, 5,600] | 2,628 | 0.24 | 0.17 | 454 (46) | 1,430 [957, 2,343] | 2,565 | 0.40 | 0.29 |
| map, offline | OPD-II (3 s delayed) | 282 (0) | 1,711 [912, 4,120] | 1,079 | 0.41 | 0.35 | 454 (0) | 407 [53, 1,231] | 1,020 | 0.59 | 0.50 |
| map, offline | Dewri et al. (start heading used) | 282 (37) | 5,382 [2,969, 7,392] | 3,333 | 0.21 | 0.13 | 454 (68) | 1,758 [1,119, 3,596] | 3,669 | 0.34 | 0.22 |
| map, offline | SPy, start known (start heading used) | 282 (4) | 4,752 [3,473, 7,599] | 4,473 | 0.07 | 0.04 | 454 (4) | 2,651 [1,830, 4,240] | 4,265 | 0.10 | 0.06 |
| map, heading unknown | map PF, steering | 282 (0) | 2,239 [1,076, 3,319] | 1,609 | 0.31 | 0.31 | 454 (0) | 432 [230, 938] | 617 | 0.53 | 0.51 |
| map, heading unknown | map PF, speed only | 282 (0) | 6,313 [3,834, 8,070] | 3,739 | 0.10 | 0.04 | 454 (0) | 3,530 [2,312, 4,912] | 4,466 | 0.13 | 0.09 |
| map, heading unknown | map beam, steering | 282 (0) | 1,244 [447, 2,085] | 1,200 | 0.39 | 0.39 | 454 (0) | 124 [48, 510] | 173 | 0.61 | 0.58 |
| map, heading unknown | map beam, speed only | 282 (0) | 6,958 [3,931, 9,102] | 3,457 | 0.11 | 0.06 | 454 (0) | 3,605 [2,503, 4,779] | 5,011 | 0.16 | 0.09 |
| map, heading unknown | Elastic Pathing | 282 (19) | 3,554 [2,359, 5,631] | 3,400 | 0.21 | 0.15 | 454 (36) | 1,644 [1,056, 2,585] | 2,514 | 0.35 | 0.25 |
| map, heading unknown | Dewri et al. | 282 (25) | 5,619 [3,273, 7,780] | 2,958 | 0.18 | 0.11 | 454 (48) | 2,172 [1,429, 3,600] | 3,262 | 0.30 | 0.19 |
| map, heading unknown | SPy, start known | 282 (5) | 4,932 [3,613, 7,779] | 4,485 | 0.05 | 0.02 | 454 (5) | 3,254 [2,245, 4,387] | 4,327 | 0.07 | 0.03 |
| reference | curvature DR (class F, not CAN-only) | 282 (0) | 678 [498, 1,427] | 733 | 0.28 | 0.12 | 460 (0) | 325 [204, 519] | 519 | 0.41 | 0.21 |
