#!/usr/bin/env bash
# Same-cohort horizon study: online methods are causal, so their 15-minute predictions also answer the 1- and 5-minute
# prefixes. Link their icde_v11 parts into icde_cohort_online_p060 / _p300 / _p900 and score each with --prefix-s
# and the cohort episode list (the 900 s prefix is the w15 episode itself).
set -u
cd "$(dirname "$0")/.."
B=${CANTRAJ_BENCH:-/data/backup/DriveDNA-GPS_bench}/preds
for P in 060 300 900; do
  for proto in O-N-Hp O-N-Hm O-M-Hp O-M-Hm; do
    mkdir -p "$B/icde_cohort_online_p$P/$proto"
    for stem in dr_none__G1 dr_steer__G3 dr_steer_offonline__G3 learned_gru_s0__G3 learned_gru_s1__G3 learned_gru_s2__G3 dr_curvature__G5 \
                map_pf_steer__G3 map_pf_none__G1 macrotracking__G3 opd2_online__G3 map_pf_curvature__G5; do
      for f in "$B/icde_v11/$proto/$stem".part*; do
        [ -e "$f" ] && ln -sf "$f" "$B/icde_cohort_online_p$P/$proto/$(basename "$f")"
      done
    done
  done
done
echo linked
