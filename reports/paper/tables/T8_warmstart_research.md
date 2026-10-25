| method                                                | comparison                             | heading   | fold     |   n_pairs | paired_median_diff_m   | paired_mean_diff_m   |   win_rate |
|:------------------------------------------------------|:---------------------------------------|:----------|:---------|----------:|:-----------------------|:---------------------|-----------:|
| steering DR, online offset                            | 5 min of own history vs cold start     | known     | val      |      2917 | -0 [-23, +0]           | -168                 |       0.51 |
| steering DR, online offset                            | 5 min of own history vs cold start     | known     | test     |      1917 | +0 [-55, +0]           | -63                  |       0.49 |
| steering DR, online offset                            | 5 min of own history vs cold start     | known     | few-shot |      1963 | -118 [-420, +0]        | -551                 |       0.60 |
| steering DR, online offset                            | 5 min of own history vs cold start     | unknown   | val      |      2917 | +0 [+0, +0]            | +14                  |       0.46 |
| steering DR, online offset                            | 5 min of own history vs cold start     | unknown   | test     |      1917 | +0 [+0, +0]            | -21                  |       0.46 |
| steering DR, online offset                            | 5 min of own history vs cold start     | unknown   | few-shot |      1963 | +0 [+0, +0]            | -3                   |       0.45 |
| steering DR, online offset                            | all own history vs cold start          | known     | val      |      2917 | -1 [-113, +0]          | -174                 |       0.54 |
| steering DR, online offset                            | all own history vs cold start          | known     | test     |      1917 | +0 [-93, +0]           | -39                  |       0.48 |
| steering DR, online offset                            | all own history vs cold start          | known     | few-shot |      1963 | -126 [-563, +0]        | -534                 |       0.59 |
| steering DR, online offset                            | all own history vs cold start          | unknown   | val      |      2917 | +0 [+0, +0]            | -15                  |       0.46 |
| steering DR, online offset                            | all own history vs cold start          | unknown   | test     |      1917 | +0 [+0, +0]            | -32                  |       0.46 |
| steering DR, online offset                            | all own history vs cold start          | unknown   | few-shot |      1963 | +0 [+0, +0]            | -39                  |       0.45 |
| device gyro DR, least-squares bias (G8)               | 5 min of own history vs cold start     | known     | val      |      1056 | -8 [-29, -0]           | -111                 |       0.54 |
| device gyro DR, least-squares bias (G8)               | 5 min of own history vs cold start     | known     | test     |       667 | -2 [-17, +0]           | -85                  |       0.51 |
| device gyro DR, least-squares bias (G8)               | 5 min of own history vs cold start     | known     | few-shot |       732 | -2 [-17, +0]           | -174                 |       0.52 |
| device gyro DR, least-squares bias (G8)               | 5 min of own history vs cold start     | unknown   | val      |      1056 | +0 [+0, +0]            | -3                   |       0.45 |
| device gyro DR, least-squares bias (G8)               | 5 min of own history vs cold start     | unknown   | test     |       667 | +3 [+0, +24]           | +0                   |       0.41 |
| device gyro DR, least-squares bias (G8)               | 5 min of own history vs cold start     | unknown   | few-shot |       732 | +0 [+0, +3]            | +112                 |       0.43 |
| device gyro DR, least-squares bias (G8)               | all own history vs cold start          | known     | val      |      1056 | -91 [-134, -36]        | -490                 |       0.63 |
| device gyro DR, least-squares bias (G8)               | all own history vs cold start          | known     | test     |       667 | -34 [-135, +0]         | -375                 |       0.55 |
| device gyro DR, least-squares bias (G8)               | all own history vs cold start          | known     | few-shot |       732 | -46 [-112, -4]         | -339                 |       0.59 |
| device gyro DR, least-squares bias (G8)               | all own history vs cold start          | unknown   | val      |      1056 | +0 [+0, +18]           | +55                  |       0.44 |
| device gyro DR, least-squares bias (G8)               | all own history vs cold start          | unknown   | test     |       667 | +6 [+0, +54]           | +173                 |       0.40 |
| device gyro DR, least-squares bias (G8)               | all own history vs cold start          | unknown   | few-shot |       732 | +0 [+0, +22]           | +160                 |       0.40 |
| learned GRU (seed 0)                                  | 5 min of own history vs cold start     | known     | val      |      2917 | +0 [+0, +0]            | -21                  |       0.46 |
| learned GRU (seed 0)                                  | 5 min of own history vs cold start     | known     | test     |      1917 | -11 [-682, +0]         | -642                 |       0.59 |
| learned GRU (seed 0)                                  | 5 min of own history vs cold start     | known     | few-shot |      1963 | +0 [-5, +0]            | -86                  |       0.50 |
| learned GRU (seed 0)                                  | 5 min of own history vs cold start     | unknown   | val      |      2917 | +0 [-0, +0]            | -8                   |       0.50 |
| learned GRU (seed 0)                                  | 5 min of own history vs cold start     | unknown   | test     |      1917 | +0 [-0, +3]            | +110                 |       0.46 |
| learned GRU (seed 0)                                  | 5 min of own history vs cold start     | unknown   | few-shot |      1963 | +0 [+0, +1]            | +45                  |       0.42 |
| learned GRU (seed 0)                                  | last hour of own history vs cold start | known     | val      |      2917 | +0 [+0, +0]            | -22                  |       0.46 |
| learned GRU (seed 0)                                  | last hour of own history vs cold start | known     | test     |      1917 | -12 [-726, +0]         | -679                 |       0.60 |
| learned GRU (seed 0)                                  | last hour of own history vs cold start | known     | few-shot |      1963 | +0 [-5, +0]            | -88                  |       0.50 |
| learned GRU (seed 0)                                  | last hour of own history vs cold start | unknown   | val      |      2917 | +0 [-0, +0]            | -7                   |       0.50 |
| learned GRU (seed 0)                                  | last hour of own history vs cold start | unknown   | test     |      1917 | +0 [-0, +4]            | +115                 |       0.46 |
| learned GRU (seed 0)                                  | last hour of own history vs cold start | unknown   | few-shot |      1963 | +0 [+0, +1]            | +44                  |       0.42 |
| warmed online-offset DR vs nominal DR                 | 5 min of own history                   | known     | val      |      2917 | -8 [-204, +21]         | -268                 |       0.52 |
| warmed online-offset DR vs nominal DR                 | 5 min of own history                   | known     | test     |      1917 | -80 [-360, +62]        | -205                 |       0.57 |
| warmed online-offset DR vs nominal DR                 | 5 min of own history                   | known     | few-shot |      1963 | -398 [-776, -10]       | -683                 |       0.65 |
| warmed online-offset DR vs nominal DR                 | 5 min of own history                   | unknown   | val      |      2917 | +6 [-0, +37]           | +38                  |       0.47 |
| warmed online-offset DR vs nominal DR                 | 5 min of own history                   | unknown   | test     |      1917 | +22 [+1, +48]          | +67                  |       0.46 |
| warmed online-offset DR vs nominal DR                 | 5 min of own history                   | unknown   | few-shot |      1963 | +46 [+4, +99]          | +138                 |       0.45 |
| warmed online-offset DR vs nominal DR                 | all own history                        | known     | val      |      2917 | -41 [-280, +16]        | -274                 |       0.55 |
| warmed online-offset DR vs nominal DR                 | all own history                        | known     | test     |      1917 | -68 [-388, +78]        | -182                 |       0.56 |
| warmed online-offset DR vs nominal DR                 | all own history                        | known     | few-shot |      1963 | -385 [-922, -10]       | -666                 |       0.65 |
| warmed online-offset DR vs nominal DR                 | all own history                        | unknown   | val      |      2917 | +2 [-6, +38]           | +9                   |       0.49 |
| warmed online-offset DR vs nominal DR                 | all own history                        | unknown   | test     |      1917 | +6 [-7, +36]           | +56                  |       0.48 |
| warmed online-offset DR vs nominal DR                 | all own history                        | unknown   | few-shot |      1963 | +31 [+7, +81]          | +102                 |       0.46 |
| warmed online-offset DR vs train-fold calibrated DR   | 5 min of own history                   | known     | val      |      2917 | -0 [-184, +42]         | -220                 |       0.50 |
| warmed online-offset DR vs train-fold calibrated DR   | 5 min of own history                   | known     | test     |      1917 | -190 [-2,214, +124]    | -846                 |       0.60 |
| warmed online-offset DR vs train-fold calibrated DR   | 5 min of own history                   | known     | few-shot |      1963 | -360 [-701, -18]       | -670                 |       0.66 |
| warmed online-offset DR vs train-fold calibrated DR   | 5 min of own history                   | unknown   | val      |      2917 | -6 [-20, +16]          | +4                   |       0.52 |
| warmed online-offset DR vs train-fold calibrated DR   | 5 min of own history                   | unknown   | test     |      1917 | +11 [-19, +154]        | +214                 |       0.47 |
| warmed online-offset DR vs train-fold calibrated DR   | 5 min of own history                   | unknown   | few-shot |      1963 | +30 [+2, +84]          | +114                 |       0.47 |
| warmed online-offset DR vs train-fold calibrated DR   | all own history                        | known     | val      |      2917 | -22 [-285, +37]        | -225                 |       0.54 |
| warmed online-offset DR vs train-fold calibrated DR   | all own history                        | known     | test     |      1917 | -196 [-2,161, +151]    | -823                 |       0.59 |
| warmed online-offset DR vs train-fold calibrated DR   | all own history                        | known     | few-shot |      1963 | -360 [-865, -17]       | -653                 |       0.65 |
| warmed online-offset DR vs train-fold calibrated DR   | all own history                        | unknown   | val      |      2917 | -14 [-33, +11]         | -25                  |       0.53 |
| warmed online-offset DR vs train-fold calibrated DR   | all own history                        | unknown   | test     |      1917 | +14 [-28, +293]        | +203                 |       0.47 |
| warmed online-offset DR vs train-fold calibrated DR   | all own history                        | unknown   | few-shot |      1963 | +21 [+0, +65]          | +78                  |       0.47 |
| warmed device gyro DR (G8) vs nominal steering DR     | all own history                        | known     | val      |      1056 | -84 [-258, -31]        | -379                 |       0.60 |
| warmed device gyro DR (G8) vs nominal steering DR     | all own history                        | known     | test     |       667 | -64 [-365, +34]        | -322                 |       0.56 |
| warmed device gyro DR (G8) vs nominal steering DR     | all own history                        | known     | few-shot |       732 | -196 [-415, -53]       | -412                 |       0.62 |
| warmed device gyro DR (G8) vs nominal steering DR     | all own history                        | unknown   | val      |      1056 | +11 [-8, +86]          | +38                  |       0.47 |
| warmed device gyro DR (G8) vs nominal steering DR     | all own history                        | unknown   | test     |       667 | +29 [-1, +211]         | +176                 |       0.46 |
| warmed device gyro DR (G8) vs nominal steering DR     | all own history                        | unknown   | few-shot |       732 | +95 [+5, +193]         | +217                 |       0.42 |
| warmed device gyro DR (G8) vs warmed online-offset DR | all own history                        | known     | val      |      1056 | -132 [-187, -67]       | -306                 |       0.63 |
| warmed device gyro DR (G8) vs warmed online-offset DR | all own history                        | known     | test     |       667 | -153 [-343, +23]       | -286                 |       0.58 |
| warmed device gyro DR (G8) vs warmed online-offset DR | all own history                        | known     | few-shot |       732 | -95 [-266, +74]        | +53                  |       0.56 |
| warmed device gyro DR (G8) vs warmed online-offset DR | all own history                        | unknown   | val      |      1056 | +37 [+14, +112]        | +144                 |       0.42 |
| warmed device gyro DR (G8) vs warmed online-offset DR | all own history                        | unknown   | test     |       667 | +66 [-6, +169]         | +268                 |       0.45 |
| warmed device gyro DR (G8) vs warmed online-offset DR | all own history                        | unknown   | few-shot |       732 | +109 [+24, +188]       | +303                 |       0.41 |
| warmed device gyro DR (G8) vs class F curvature DR    | all own history                        | known     | val      |      1056 | +290 [+184, +515]      | +672                 |       0.12 |
| warmed device gyro DR (G8) vs class F curvature DR    | all own history                        | known     | test     |       667 | +558 [+399, +952]      | +1,108               |       0.09 |
| warmed device gyro DR (G8) vs class F curvature DR    | all own history                        | known     | few-shot |       732 | +769 [+378, +1,950]    | +1,588               |       0.09 |
