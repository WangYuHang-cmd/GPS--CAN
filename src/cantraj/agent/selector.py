"""Supervised selector (agent study, configuration B): a logistic model over the observable candidate diagnostics,
trained on validation-fold route_ok labels and frozen before the formal run. It calls every reconstructor (its cost is
the sum of the three) and picks the candidate with the highest predicted probability; ties go to the beam's Top-1.
Standardisation and regression are fitted together inside every training fold (sklearn Pipeline), never on the whole
validation set before cross-fitting.
"""
from __future__ import annotations

import json

import numpy as np

FEATURES = ["is_pf", "is_ep", "log_rank", "hd_abs_rms_deg", "hd_abs_p90_deg", "hd_drift_rms_deg", "turns_unmatched",
            "len_ratio_dev", "uturns", "overlap_mean", "n_tools_with_output", "log_end_disagree_m"]
FILL = {"hd_abs_rms_deg": 90.0, "hd_abs_p90_deg": 150.0, "hd_drift_rms_deg": 60.0, "len_ratio_dev": 0.0}   # too few moving steps
TOOL_ORDER = {"beam": 0, "pf": 1, "ep": 2}


def feature_vector(f):
    def num(k, default=0.0):
        v = f.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return FILL.get(k, default)
        return float(v)
    return np.array([float(f["tool"] == "pf"), float(f["tool"] == "ep"), np.log(float(f["rank"])), num("hd_abs_rms_deg"), num("hd_abs_p90_deg"),
                     num("hd_drift_rms_deg"), num("turns_unmatched"), num("len_ratio_dev"), num("uturns"), num("overlap_mean"),
                     num("n_tools_with_output"), np.log1p(num("end_disagree_m"))], dtype=float)


def fit(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    pipe.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return pipe


def export(pipe, C=1.0):
    sc, lr = pipe.steps[0][1], pipe.steps[1][1]
    return dict(features=FEATURES, fill=FILL, mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                intercept=float(lr.intercept_[0]), C=C, n_train=int(getattr(lr, "n_features_in_", 0) and 0))


class Selector:
    def __init__(self, params):
        self.p = params
        self.mean, self.scale = np.asarray(params["mean"], float), np.asarray(params["scale"], float)
        self.coef, self.b = np.asarray(params["coef"], float), float(params["intercept"])

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def proba(self, feats):
        x = (feature_vector(feats) - self.mean) / np.where(self.scale > 0, self.scale, 1.0)
        z = float(x @ self.coef + self.b)
        return 1.0 / (1.0 + np.exp(-z))

    def select(self, cand_feats):
        """cand_feats: {cid: feature dict}. Highest probability; ties -> beam before pf before ep, then lower rank."""
        if not cand_feats:
            return None
        rows = [(-round(self.proba(f), 12), TOOL_ORDER.get(f["tool"], 9), int(f["rank"]), cid) for cid, f in cand_feats.items()]
        rows.sort()
        return rows[0][3]


def select_with_pipeline(pipe, cand_feats):
    if not cand_feats:
        return None
    cids = list(cand_feats)
    X = np.stack([feature_vector(cand_feats[c]) for c in cids])
    p = pipe.predict_proba(X)[:, 1]
    rows = sorted((-round(float(pi), 12), TOOL_ORDER.get(cand_feats[c]["tool"], 9), int(cand_feats[c]["rank"]), c) for pi, c in zip(p, cids))
    return rows[0][3]
