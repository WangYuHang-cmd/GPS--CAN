| method                        | perturbation             |   n_pairs | paired_median_diff_m   | paired_mean_diff_m   |   win_rate |
|:------------------------------|:-------------------------|----------:|:-----------------------|:---------------------|-----------:|
| steering DR                   | speed quantised to 1 m/s |      2917 | +0 [-0, +1]            | +0                   |       0.49 |
| steering DR                   | speed noise 0.5 m/s      |      2917 | +0 [+0, +0]            | +7                   |       0.45 |
| steering DR                   | steering held at 1 Hz    |      2917 | +4 [+2, +5]            | +17                  |       0.44 |
| straight line                 | speed quantised to 1 m/s |      2917 | -0 [-0, +0]            | +0                   |       0.53 |
| straight line                 | speed noise 0.5 m/s      |      2917 | +1 [+0, +1]            | +2                   |       0.26 |
| learned GRU (plain)           | speed quantised to 1 m/s |      2917 | +15 [+6, +28]          | +41                  |       0.40 |
| learned GRU (plain)           | speed noise 0.5 m/s      |      2917 | +877 [+623, +1,331]    | +1,268               |       0.09 |
| learned GRU (plain)           | steering held at 1 Hz    |      2917 | +6 [+3, +9]            | +55                  |       0.42 |
| learned GRU (noise-augmented) | speed quantised to 1 m/s |      2917 | -0 [-0, +0]            | -0                   |       0.50 |
| learned GRU (noise-augmented) | speed noise 0.5 m/s      |      2917 | -1 [-6, +6]            | +4                   |       0.51 |
| learned GRU (noise-augmented) | steering held at 1 Hz    |      2917 | +10 [+4, +15]          | +68                  |       0.38 |
| noise-augmented vs plain GRU  | clean inputs             |      2917 | +5 [-0, +11]           | +16                  |       0.47 |
| noise-augmented vs plain GRU  | speed noise 0.5 m/s      |      2917 | -859 [-1,322, -567]    | -1,248               |       0.88 |
| steering DR                   | start shifted 20 m       |      2917 | +2 [+1, +3]            | +2                   |       0.45 |
| steering DR                   | heading noise 10 deg     |      2917 | +13 [+7, +21]          | +32                  |       0.44 |
| steering DR                   | heading noise 30 deg     |      2917 | +107 [+66, +138]       | +221                 |       0.36 |
| map particle filter           | start shifted 20 m       |      2917 | +0 [+0, +0]            | -184                 |       0.36 |
| map particle filter           | heading noise 10 deg     |      2917 | +0 [-0, +0]            | -179                 |       0.48 |
| map particle filter           | heading noise 30 deg     |      2917 | +0 [+0, +0]            | -117                 |       0.43 |
