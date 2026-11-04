#!/usr/bin/env python3
"""C-max vote row: majority vote of the controller's choices over the seeds of one tag (ties broken by the selector's
probability when a selector is given, else by the lowest seed), written as one prediction stem and one results row set.

    python scripts/agent_vote.py --run icde_agent_max --tag agent_qwen3max --seeds 0 1 2 --manifest manifests/icde_agent_cohort.csv \
        --selector configs/agent_max/selector_max.json
"""
import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj import paths, predfile  # noqa: E402
from cantraj.agent.selector import Selector  # noqa: E402
from cantraj.agent.toolbox import EpisodeToolBox  # noqa: E402

PROTO = "F-M-Hp"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--selector", default=None)
    args = ap.parse_args()
    res_dir = os.path.join(paths.BENCH, "agent_logs", args.run, "results")
    res = {s: pd.read_csv(os.path.join(res_dir, f"{args.tag}_s{s}.csv")).set_index("episode_id") for s in args.seeds}
    sel = Selector.load(args.selector) if args.selector else None
    cohort = pd.read_csv(args.manifest)
    root = os.path.join(paths.BENCH, "agent_toolcache", args.run)
    rows, frames, skipped = [], [], []
    for r in cohort.itertuples():
        choices = [res[s].choice.get(r.episode_id) for s in args.seeds]
        choices = [c for c in choices if isinstance(c, str) and c]
        cdir = os.path.join(root, r.driver, r.drive, f"k{int(r.k):03d}")
        if not choices:
            rows.append(dict(episode_id=r.episode_id, driver=r.driver, drive=r.drive, fold=r.fold, status="no_candidates", choice=None, votes=0))
            skipped.append((r.episode_id, "vote: no seed produced a choice"))
            continue
        cnt = Counter(choices)
        top = max(cnt.values())
        tied = sorted(c for c, n in cnt.items() if n == top)
        if len(tied) > 1 and sel is not None:
            tb = EpisodeToolBox(cdir).call_all()
            tied.sort(key=lambda c: -sel.proba(tb.features(c)))
        choice = tied[0]
        rows.append(dict(episode_id=r.episode_id, driver=r.driver, drive=r.drive, fold=r.fold, status="submitted", choice=choice, votes=top, n_seeds=len(choices),
                         cost_reconstructed_s=float(np.nansum([res[s].cost_reconstructed_s.get(r.episode_id, np.nan) for s in args.seeds]))))
        t = np.load(os.path.join(cdir, "t_abs.npz"))["t_abs"]
        xy = np.load(os.path.join(cdir, "candidates.npz"))[f"xy__{choice}"]
        frames.append(predfile.new_frame(r.episode_id, t, xy, None, candidate_id=0, prob=1.0))
    stem = f"{args.tag}_vote__G3"
    df = pd.concat(frames, ignore_index=True)
    predfile.write(df, os.path.join(paths.PREDS, args.run, PROTO, f"{stem}.part0.parquet"),
                   dict(baseline=stem.split("__")[0], group="G3", protocol=PROTO, run=args.run, study="agent_vote", seeds=args.seeds, n_episodes=int(df.episode_id.nunique())))
    pd.DataFrame(rows).to_csv(os.path.join(res_dir, f"{args.tag}_vote.csv"), index=False)
    if skipped:
        pd.DataFrame(skipped, columns=["episode_id", "reason"]).to_csv(os.path.join(paths.PREDS, args.run, PROTO, f"{stem}.part0.skipped.csv"), index=False)
    t = pd.DataFrame(rows)
    print(f"{stem}: {int((t.status == 'submitted').sum())} windows; unanimous {(t.votes == len(args.seeds)).mean() * 100:.0f} %, majority 2 {(t.votes == 2).mean() * 100:.0f} %")


if __name__ == "__main__":
    main()
