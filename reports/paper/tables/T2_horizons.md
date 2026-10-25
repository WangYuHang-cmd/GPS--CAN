| window   | setting      | heading   | method                                     | fold     |    n | median_ADE_CI        |   mean_ADE |   c50 |   c100 |   S50_90 |   lost_road |
|:---------|:-------------|:----------|:-------------------------------------------|:---------|-----:|:---------------------|-----------:|------:|-------:|---------:|------------:|
| w01      | map, online  | known     | map particle filter, nominal steering (G3) | test     | 9299 | 16 [14, 18]          |       1522 |  0.93 |   0.96 |     0.86 |        0.05 |
| w01      | map, online  | known     | map particle filter, nominal steering (G3) | few-shot | 9671 | 13 [11, 16]          |       4275 |  0.93 |   0.95 |     0.89 |        0.06 |
| w01      | map, offline | known     | offline beam search, nominal steering (G3) | test     | 9299 | 12 [10, 15]          |       1520 |  0.93 |   0.96 |     0.88 |        0.05 |
| w01      | map, offline | known     | offline beam search, nominal steering (G3) | few-shot | 9671 | 9 [8, 13]            |       4270 |  0.93 |   0.95 |     0.90 |        0.06 |
| w01      | map, online  | unknown   | map particle filter, nominal steering (G3) | test     | 9299 | 96 [63, 129]         |       1832 |  0.67 |   0.71 |     0.40 |        0.17 |
| w01      | map, online  | unknown   | map particle filter, nominal steering (G3) | few-shot | 9671 | 90 [58, 131]         |       4574 |  0.67 |   0.69 |     0.39 |        0.18 |
| w01      | map, offline | unknown   | offline beam search, nominal steering (G3) | test     | 9299 | 17 [14, 19]          |       1767 |  0.74 |   0.78 |     0.69 |        0.24 |
| w01      | map, offline | unknown   | offline beam search, nominal steering (G3) | few-shot | 9671 | 13 [10, 18]          |       4508 |  0.75 |   0.77 |     0.72 |        0.24 |
| w01      | no map       | known     | curvature DR, class F reference (G5)       | test     | 9299 | 11 [10, 14]          |         17 |  0.95 |   0.99 |     0.88 |        0.02 |
| w01      | no map       | known     | curvature DR, class F reference (G5)       | few-shot | 9671 | 9 [8, 11]            |         13 |  0.97 |   0.99 |     0.92 |        0.01 |
| w01      | no map       | known     | speed only, straight line (G1)             | test     | 9299 | 49 [37, 73]          |        106 |  0.63 |   0.73 |     0.37 |        0.47 |
| w01      | no map       | known     | speed only, straight line (G1)             | few-shot | 9671 | 53 [46, 67]          |        107 |  0.62 |   0.72 |     0.36 |        0.48 |
| w01      | no map       | known     | steering DR, nominal (G3)                  | test     | 9299 | 59 [34, 107]         |        136 |  0.58 |   0.71 |     0.27 |        0.52 |
| w01      | no map       | known     | steering DR, nominal (G3)                  | few-shot | 9671 | 93 [52, 142]         |        178 |  0.50 |   0.63 |     0.20 |        0.63 |
| w05      | map, online  | known     | map particle filter, nominal steering (G3) | test     | 1917 | 44 [29, 61]          |       1726 |  0.63 |   0.81 |     0.37 |        0.30 |
| w05      | map, online  | known     | map particle filter, nominal steering (G3) | few-shot | 1963 | 27 [19, 49]          |       4363 |  0.74 |   0.85 |     0.54 |        0.23 |
| w05      | map, offline | known     | offline beam search, nominal steering (G3) | test     | 1917 | 37 [24, 55]          |       1699 |  0.66 |   0.82 |     0.41 |        0.30 |
| w05      | map, offline | known     | offline beam search, nominal steering (G3) | few-shot | 1963 | 22 [17, 41]          |       4344 |  0.76 |   0.86 |     0.59 |        0.22 |
| w05      | map, online  | unknown   | map particle filter, nominal steering (G3) | test     | 1917 | 148 [107, 182]       |       2338 |  0.52 |   0.68 |     0.22 |        0.33 |
| w05      | map, online  | unknown   | map particle filter, nominal steering (G3) | few-shot | 1963 | 94 [62, 151]         |       4860 |  0.63 |   0.73 |     0.36 |        0.26 |
| w05      | map, offline | unknown   | offline beam search, nominal steering (G3) | test     | 1917 | 43 [26, 60]          |       2122 |  0.61 |   0.75 |     0.37 |        0.35 |
| w05      | map, offline | unknown   | offline beam search, nominal steering (G3) | few-shot | 1963 | 24 [18, 43]          |       4649 |  0.72 |   0.81 |     0.56 |        0.26 |
| w05      | no map       | known     | curvature DR, class F reference (G5)       | test     | 1917 | 84 [74, 98]          |        156 |  0.46 |   0.64 |     0.13 |        0.62 |
| w05      | no map       | known     | curvature DR, class F reference (G5)       | few-shot | 1963 | 80 [70, 92]          |        117 |  0.48 |   0.67 |     0.14 |        0.61 |
| w05      | no map       | known     | speed only, straight line (G1)             | test     | 1917 | 735 [651, 884]       |       1110 |  0.22 |   0.30 |     0.04 |        0.88 |
| w05      | no map       | known     | speed only, straight line (G1)             | few-shot | 1963 | 825 [787, 951]       |       1135 |  0.21 |   0.28 |     0.04 |        0.90 |
| w05      | no map       | known     | steering DR, nominal (G3)                  | test     | 1917 | 921 [538, 1,737]     |       1528 |  0.18 |   0.25 |     0.02 |        0.94 |
| w05      | no map       | known     | steering DR, nominal (G3)                  | few-shot | 1963 | 1,669 [878, 2,451]   |       2093 |  0.15 |   0.20 |     0.01 |        0.95 |
| w15      | map, online  | known     | map particle filter, nominal steering (G3) | test     |  608 | 198 [78, 507]        |       3125 |  0.30 |   0.48 |     0.06 |        0.70 |
| w15      | map, online  | known     | map particle filter, nominal steering (G3) | few-shot |  565 | 134 [52, 251]        |       5621 |  0.46 |   0.60 |     0.18 |        0.57 |
| w15      | map, offline | known     | offline beam search, nominal steering (G3) | test     |  608 | 158 [68, 372]        |       3002 |  0.35 |   0.51 |     0.09 |        0.68 |
| w15      | map, offline | known     | offline beam search, nominal steering (G3) | few-shot |  565 | 111 [47, 199]        |       5460 |  0.50 |   0.63 |     0.20 |        0.55 |
| w15      | map, online  | unknown   | map particle filter, nominal steering (G3) | test     |  608 | 385 [206, 850]       |       3835 |  0.26 |   0.43 |     0.05 |        0.72 |
| w15      | map, online  | unknown   | map particle filter, nominal steering (G3) | few-shot |  565 | 267 [116, 742]       |       6063 |  0.43 |   0.56 |     0.13 |        0.58 |
| w15      | map, offline | unknown   | offline beam search, nominal steering (G3) | test     |  608 | 167 [68, 450]        |       3815 |  0.33 |   0.49 |     0.08 |        0.69 |
| w15      | map, offline | unknown   | offline beam search, nominal steering (G3) | few-shot |  565 | 113 [47, 206]        |       5964 |  0.48 |   0.61 |     0.20 |        0.56 |
| w15      | no map       | known     | curvature DR, class F reference (G5)       | test     |  608 | 511 [453, 595]       |       1006 |  0.17 |   0.27 |     0.01 |        0.91 |
| w15      | no map       | known     | curvature DR, class F reference (G5)       | few-shot |  565 | 491 [423, 588]       |        756 |  0.17 |   0.27 |     0.00 |        0.94 |
| w15      | no map       | known     | speed only, straight line (G1)             | test     |  608 | 3,389 [3,128, 4,028] |       4659 |  0.08 |   0.11 |     0.00 |        0.98 |
| w15      | no map       | known     | speed only, straight line (G1)             | few-shot |  565 | 3,906 [3,618, 4,523] |       5038 |  0.07 |   0.10 |     0.00 |        0.98 |
| w15      | no map       | known     | steering DR, nominal (G3)                  | test     |  608 | 5,135 [2,756, 7,909] |       6184 |  0.07 |   0.10 |     0.00 |        0.98 |
| w15      | no map       | known     | steering DR, nominal (G3)                  | few-shot |  565 | 7,174 [5,066, 8,953] |       7615 |  0.05 |   0.07 |     0.00 |        0.99 |
| full     | map, online  | known     | map particle filter, nominal steering (G3) | test     |  282 | 1,964 [870, 3,556]   |       9766 |  0.27 |   0.35 |     0.09 |        0.82 |
| full     | map, online  | known     | map particle filter, nominal steering (G3) | few-shot |  460 | 202 [93, 799]        |       6112 |  0.52 |   0.60 |     0.30 |        0.54 |
| full     | map, offline | known     | offline beam search, nominal steering (G3) | test     |  282 | 1,204 [402, 1,901]   |       9579 |  0.32 |   0.41 |     0.12 |        0.75 |
| full     | map, offline | known     | offline beam search, nominal steering (G3) | few-shot |  460 | 101 [45, 224]        |       6732 |  0.56 |   0.63 |     0.36 |        0.49 |
| full     | map, online  | unknown   | map particle filter, nominal steering (G3) | test     |  282 | 2,318 [1,191, 3,971] |       9734 |  0.22 |   0.31 |     0.06 |        0.84 |
| full     | map, online  | unknown   | map particle filter, nominal steering (G3) | few-shot |  460 | 502 [230, 1,036]     |       6460 |  0.45 |   0.52 |     0.20 |        0.57 |
| full     | map, offline | unknown   | offline beam search, nominal steering (G3) | test     |  282 | 1,265 [477, 2,116]   |       9667 |  0.29 |   0.38 |     0.10 |        0.77 |
| full     | map, offline | unknown   | offline beam search, nominal steering (G3) | few-shot |  460 | 141 [52, 510]        |       7208 |  0.53 |   0.60 |     0.35 |        0.52 |
| full     | no map       | known     | curvature DR, class F reference (G5)       | test     |  282 | 678 [485, 1,482]     |       5422 |  0.21 |   0.28 |     0.07 |        0.85 |
| full     | no map       | known     | curvature DR, class F reference (G5)       | few-shot |  460 | 325 [203, 494]       |       3364 |  0.32 |   0.41 |     0.14 |        0.75 |
| full     | no map       | known     | speed only, straight line (G1)             | test     |  282 | 6,551 [4,380, 8,865] |      15681 |  0.08 |   0.10 |     0.02 |        0.97 |
| full     | no map       | known     | speed only, straight line (G1)             | few-shot |  460 | 3,327 [2,284, 4,377] |      12349 |  0.15 |   0.19 |     0.05 |        0.92 |
| full     | no map       | known     | steering DR, nominal (G3)                  | test     |  282 | 2,708 [1,479, 7,710] |      11683 |  0.12 |   0.17 |     0.02 |        0.93 |
| full     | no map       | known     | steering DR, nominal (G3)                  | few-shot |  460 | 2,674 [1,210, 3,730] |       9695 |  0.17 |   0.22 |     0.05 |        0.90 |
