"""Scoring: join predictions with reference points at exact reference times, per-episode summaries."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import metrics as M
from .episodes import Episode


def s_can_at(dd, ep: Episode, t):
    """Cumulative CAN distance since t0 at reference times (reference-side stratification only)."""
    gt = dd.grid["time_s"].to_numpy(dtype=float)
    gv = dd.grid["vEgo"].to_numpy(dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (gv[1:] + gv[:-1]) * np.diff(gt))])
    return np.interp(t, gt, cum) - np.interp(ep.t0, gt, cum)


def score_episode(ep: Episode, pred: pd.DataFrame, dd=None, in_gap=None, topk=False):
    ref = ep.ref
    t = ref["time_s"].to_numpy(dtype=float)
    top1 = pred[pred["candidate_id"] == 0].set_index(np.round(pred[pred["candidate_id"] == 0]["t"].to_numpy(dtype=float), 6))
    key = np.round(t, 6)
    xy = np.full((len(t), 2), np.nan)
    hit = np.isin(key, top1.index.values)
    xy[hit] = top1.loc[key[hit], ["pred_x_m", "pred_y_m"]].to_numpy(dtype=float)
    err = M.point_errors(xy, ref[["x", "y"]].to_numpy(dtype=float))
    s_can = s_can_at(dd, ep, t) if dd is not None else None
    ref_xy = ref[["x", "y"]].to_numpy(dtype=float)
    out = M.episode_summary(err, t_rel=t - ep.t0, s_can=s_can, moving=ref["moving"].to_numpy(dtype=bool), in_gap=in_gap, pred_xy=xy, ref_xy=ref_xy)
    if topk and pred["candidate_id"].max() > 0:
        errs = []
        for k, g in pred.groupby("candidate_id"):
            g = g.set_index(np.round(g["t"].to_numpy(dtype=float), 6))
            e = np.full(len(t), np.nan)
            h = np.isin(key, g.index.values)
            e[h] = M.point_errors(g.loc[key[h], ["pred_x_m", "pred_y_m"]].to_numpy(dtype=float), ref[["x", "y"]].to_numpy(dtype=float)[h])
            errs.append(e)
        kbest, ebest = M.topk_oracle(np.array(errs))
        best = M.episode_summary(ebest)
        out.update({f"topk_{k}": v for k, v in best.items() if k in ("ade", "fde", "c25", "c50", "c100", "end_within_100")})
        out["topk_K"] = len(errs)
        out["topk_pick"] = int(kbest)
        # candidate diagnostics (ICDE RQ4), candidates in rank order (candidate_id 0 = Top-1):
        #   topK_ade = whole-path oracle among the first K; topK_cov100 = per-second union coverage (any of the first K
        #   within 100 m at that second; NOT a single admissible path)
        E = np.where(np.isfinite(np.array(errs)), np.array(errs), np.inf)
        for K in (1, 3, 5, 10):
            sub = E[:K]
            ade_k = np.where(np.isfinite(sub), sub, np.nan)
            m_k = np.nanmean(ade_k, axis=1) if np.isfinite(sub).any() else np.full(len(sub), np.nan)
            out[f"top{K}_ade"] = float(np.nanmin(m_k)) if np.isfinite(m_k).any() else np.nan
            out[f"top{K}_cov100"] = float(np.mean(np.min(sub, axis=0) <= 100.0))
        ranks = np.argsort(np.where(np.isnan(np.nanmean(np.where(np.isfinite(E), E, np.nan), axis=1)), np.inf,
                                    np.nanmean(np.where(np.isfinite(E), E, np.nan), axis=1)), kind="stable")
        out["best_rank"] = int(ranks[0]) + 1
    out.update(episode_id=ep.episode_id, driver=ep.driver, drive=ep.drive, parent=f"{ep.driver}/{ep.drive}",
               win=ep.win, k=ep.k, dur_s=ep.t_end - ep.t0, **{k: v for k, v in ep.meta.items() if not isinstance(v, (list, dict))})
    return out


def table(rows):
    df = pd.DataFrame(rows)
    return df.drop(columns=[c for c in ("curve_time", "curve_dist") if c in df.columns])
