| fold     | model                    |   seeds | no_map_median_ADE   | nominal_DR   | paired_vs_nominal                   | with_map_median_ADE   |   map_nominal_steering | paired_vs_map_nominal   |
|:---------|:-------------------------|--------:|:--------------------|:-------------|:------------------------------------|:----------------------|-----------------------:|:------------------------|
| val      | causal GRU               |       3 | 233 ± 4             | 731          | -348 (seeds -360 to -329), win 0.81 | 46 ± 1 (3 seeds)      |                     39 | +4, win 0.37            |
| val      | physics residual + GRU   |       3 | 270 ± 14            | 731          | -335 (seeds -356 to -321), win 0.80 | 43 ± 7 (3 seeds)      |                     39 | +2, win 0.40            |
| val      | physics residual, no GRU |       3 | 652 ± 16            | 731          | -50 (seeds -61 to -38), win 0.57    | 62 ± 1 (3 seeds)      |                     39 | +1, win 0.26            |
| test     | causal GRU               |       3 | 543 ± 37            | 921          | -120 (seeds -170 to -86), win 0.62  | 84 ± 8 (3 seeds)      |                     44 | +6, win 0.38            |
| test     | physics residual + GRU   |       3 | 690 ± 130           | 921          | -91 (seeds -171 to -14), win 0.59   | 62 ± 20 (3 seeds)     |                     44 | +3, win 0.45            |
| test     | physics residual, no GRU |       3 | 1,203 ± 68          | 921          | +86 (seeds +33 to +123), win 0.42   | 63 ± 4 (3 seeds)      |                     44 | +0, win 0.30            |
| few-shot | causal GRU               |       3 | 393 ± 35            | 1,669        | -714 (seeds -831 to -600), win 0.81 | 44 ± 5 (3 seeds)      |                     27 | +6, win 0.33            |
| few-shot | physics residual + GRU   |       3 | 435 ± 36            | 1,669        | -841 (seeds -900 to -755), win 0.84 | 35 ± 5 (3 seeds)      |                     27 | +2, win 0.37            |
| few-shot | physics residual, no GRU |       3 | 1,426 ± 43          | 1,669        | -93 (seeds -99 to -84), win 0.59    | 40 ± 0 (3 seeds)      |                     27 | +0, win 0.33            |
