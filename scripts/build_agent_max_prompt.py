#!/usr/bin/env python3
"""C-max prompts: the controller's system prompt with calibration hints computed on the VALIDATION cohort only (route
label rates of every tool's Top-1, of agreeing tools and of diagnostic bins), and the critic's prompt. Nothing from the
test or few-shot folds enters here.

    python scripts/build_agent_max_prompt.py --run icde_agent_max_val --manifest manifests/icde_agent_val.csv
Writes configs/agent_max/system_prompt.txt, critic_prompt.txt and the hint table reports/icde/tables/agent_max_val_hints.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths  # noqa: E402
from cantraj.agent.toolbox import EpisodeToolBox  # noqa: E402

CFG = os.path.join(paths.GPS_DIR, "configs", "agent_max")
TABLES = os.path.join(paths.GPS_DIR, "reports", "icde", "tables")

BASE = """You are the controller of a vehicle-trajectory recovery system. One five-minute driving window is given. The system knows only the vehicle's CAN signals (speed and steering-wheel angle), the start position (the origin), the start heading, static vehicle parameters and the road graph within the visible disc around the start. No GPS positions, no true route and no scores are available to you or to the tools; your job is to pick the candidate path most likely to be the vehicle's true path.

Tools: run_reconstructor(name) runs one reconstructor preset (beam, beam_wide, beam_loose, pf, pf_s1 to pf_s4, ep, ep_hm; each once) and returns ranked candidates with diagnostics; inspect_candidate(id) gives a candidate's full diagnostics; compare_candidates(ids) puts 2 to 4 candidates side by side and shows, for every pair, the first point where they diverge, the turn each takes there, the dead-reckoned heading change at that moment and the nearest road junction with its exits; score_candidates(ids) gives the probability that each candidate recovers the route according to a model fitted on validation drives; query_local_graph(id, distance_m) describes the junction nearest to a distance along a candidate; submit_choice(id, reason) ends the task with a candidate you have seen.

Diagnostics (all from the CAN signals, the given start heading and the road graph): path_len_m and len_ratio_dev (|path length / CAN distance - 1|); hd_abs_rms_deg (RMS difference between the candidate's road bearing and the dead-reckoned heading over the moving seconds) and hd_drift_rms_deg (same after removing a constant offset); turns_path / turns_dr / turns_unmatched (turns above 45 degrees within 40 m, matched within 50 m of path distance); uturns; dr_dist_first60_m and dr_dist_first120_m (mean distance to the pure dead-reckoning path over the first 60 and 120 s, where dead reckoning is still accurate to tens of metres); jn_agree / jn_disagree (junctions where the candidate's turn agrees or disagrees with the dead-reckoned heading change); overlap_with_called_top1 and n_called_tools_agree (agreement with the Top-1 of other reconstructors you have run; unknown for those not run). Positions are metres east and north of the start; compass headings are degrees clockwise from north.

"""

PROCEDURE = """Suggested procedure: run several reconstructors first (the beam presets and the filter seeds are cheap; Elastic Pathing adds speed-only evidence), call score_candidates on the candidates that differ, compare the two or three strongest with compare_candidates and look at the junction where they diverge (the exit whose turn matches the dead-reckoned heading change is the plausible one), then submit. Candidates that agree across different reconstructors, stay close to dead reckoning in the first minute and have no unmatched turns are usually right; a single tool's rank is weak evidence by itself. You have up to 12 turns and 40 tool calls; several tool calls may be requested in one turn.
"""

CRITIC = """You are a reviewer checking a trajectory-recovery decision. A controller has chosen one candidate path for a five-minute window from the vehicle's CAN signals, the start position and heading and the road graph; no GPS or true route is available. You receive the chosen candidate, one or two alternatives and their side-by-side comparison, and may inspect, compare, query junctions or ask the validation-fitted score. Decide whether the chosen candidate or an alternative is more plausible: prefer the candidate whose turns at the divergence points match the dead-reckoned heading changes, that stays close to dead reckoning in the first minute and that other reconstructors agree with; do not overturn without a concrete reason. Finish with submit_choice(candidate_id, reason) naming the candidate you judge most plausible (the chosen one if you confirm it). Be concise.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="icde_agent_max_val")
    ap.add_argument("--manifest", default=os.path.join(paths.MANIFESTS, "icde_agent_val.csv"))
    args = ap.parse_args()
    ep = pd.read_parquet(os.path.join(paths.SCORES, args.run, "episodes_all.parquet"), columns=["episode_id", "method", "route_ok"])
    ep = ep[ep.method.str.startswith("poolcand_")]
    ep["cid"] = ep.method.str.replace("poolcand_", "", regex=False).str.split("__").str[0]
    lab = ep.set_index(["episode_id", "cid"]).route_ok
    cohort = pd.read_csv(args.manifest)
    root = os.path.join(paths.BENCH, "agent_toolcache", args.run)
    rows = []
    for r in cohort.itertuples():
        cdir = os.path.join(root, r.driver, r.drive, f"k{int(r.k):03d}")
        if not os.path.exists(os.path.join(cdir, "diagnostics.json")):
            continue
        tb = EpisodeToolBox(cdir).call_all()
        for cid in tb.all_candidates():
            if (r.episode_id, cid) not in lab.index:
                continue
            f = tb.features(cid)
            rows.append(dict(episode_id=r.episode_id, cid=cid, tool=f["tool"], rank=f["rank"], route_ok=float(lab.loc[(r.episode_id, cid)]), **{k: v for k, v in f.items() if k not in ("tool", "rank")}))
    df = pd.DataFrame(rows)
    hints = []
    for tool, g in df[df["rank"] == 1].groupby("tool"):
        hints.append(dict(kind="tool_top1", key=tool, n=len(g), route_pct=100 * g.route_ok.mean()))
    for k, bins in (("consensus_tools", [(0, 0), (1, 1), (2, 3), (4, 99)]), ("hd_abs_rms_deg", [(0, 10), (10, 20), (20, 40), (40, 1e9)]),
                    ("jn_disagree", [(0, 0), (1, 1), (2, 99)]), ("dr_dist_first60_m", [(0, 20), (20, 50), (50, 150), (150, 1e9)]),
                    ("turns_unmatched", [(0, 0), (1, 1), (2, 99)])):
        for lo, hi in bins:
            v = df[k].astype(float)
            m = (v >= lo) & (v <= hi) if lo == hi or hi < 1e8 else (v >= lo)
            if lo != hi and hi < 1e8:
                m = (v >= lo) & (v < hi)
            g = df[m]
            if len(g) >= 30:
                hints.append(dict(kind=k, key=f"{lo}" if lo == hi else (f">={lo}" if hi >= 1e8 else f"{lo}-{hi}"), n=len(g), route_pct=100 * g.route_ok.mean()))
    h = pd.DataFrame(hints)
    os.makedirs(TABLES, exist_ok=True)
    h.to_csv(os.path.join(TABLES, "agent_max_val_hints.csv"), index=False)
    lines = ["Calibration from the validation drives (share of candidates that recovered the route):"]
    t1 = h[h.kind == "tool_top1"].sort_values("route_pct", ascending=False)
    lines.append("  Top-1 of each reconstructor: " + ", ".join(f"{r.key} {r.route_pct:.0f} %" for r in t1.itertuples()) + ".")
    for k, label in (("consensus_tools", "number of other reconstructors whose Top-1 agrees"), ("hd_abs_rms_deg", "hd_abs_rms_deg"),
                     ("jn_disagree", "jn_disagree"), ("dr_dist_first60_m", "dr_dist_first60_m"), ("turns_unmatched", "turns_unmatched")):
        g = h[h.kind == k]
        if len(g):
            lines.append(f"  {label}: " + ", ".join(f"{r.key}: {r.route_pct:.0f} %" for r in g.itertuples()) + ".")
    cal = "\n".join(lines) + "\n\n"
    os.makedirs(CFG, exist_ok=True)
    open(os.path.join(CFG, "system_prompt.txt"), "w").write(BASE + cal + PROCEDURE)
    open(os.path.join(CFG, "critic_prompt.txt"), "w").write(CRITIC)
    print(cal)
    print(f"prompts -> {CFG}; {len(df)} validation candidates from {df.episode_id.nunique()} windows")


if __name__ == "__main__":
    main()
