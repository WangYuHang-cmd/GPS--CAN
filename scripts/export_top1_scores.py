"""Thin published copy of the beam's (and the single-path methods') Top-1 candidate score per episode.

scripts/icde_extra.py reads the Top-1 score from the prediction files (tens of GB, not released). This script writes the
same scores, one row per (stem, protocol, episode), to scores/icde_v11/top1_scores.csv (a few MB, released with the
package), and icde_extra.py falls back to that table when the prediction files are absent, so that
confidence_w05_w15.csv is rebuilt from the package. Values are written with 17 significant digits (exact round trip).

    python scripts/export_top1_scores.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import paths  # noqa: E402
import icde_extra as X  # noqa: E402


def main():
    pairs = [(s, p) for s, p, _ in X.CONF_ROWS] + list(X.PROB_ONLY)
    rows = []
    for stem, proto in pairs:
        f = os.path.join(paths.PREDS, "icde_v11", proto, f"{stem}.part0.parquet")
        if not os.path.exists(f):
            print("missing prediction file, skipped:", f)
            continue
        s = X.top1_scores(stem, proto)                 # the prediction file exists, so this reads it, not the thin table
        rows.append(pd.DataFrame({"stem": stem, "protocol": proto, "episode_id": s.index.to_numpy(), "prob": s.to_numpy()}))
        print(f"{stem} {proto}: {len(s)} episodes")
    t = pd.concat(rows, ignore_index=True)
    out = os.path.join(paths.SCORES, "icde_v11", "top1_scores.csv")
    t.to_csv(out, index=False, float_format="%.17g")
    print(len(t), "rows ->", out)


if __name__ == "__main__":
    main()
