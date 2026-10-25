import numpy as np
import pandas as pd
import pytest

from cantraj import metrics as M, predfile as P


def test_single_point_ade_equals_fde_and_radius_inclusive():
    s = M.episode_summary(np.array([25.0]))
    assert s["ade"] == s["fde"] == 25.0
    assert s["c25"] == 1.0 and s["c10"] == 0.0


def test_missing_prediction_counts_as_miss():
    s = M.episode_summary(np.array([1.0, np.nan]))
    assert s["n_missing"] == 1 and s["c100"] == 0.5 and s["ade"] == 1.0


def test_topk_picks_one_whole_path():
    errs = np.array([[100.0, 100.0, 100.0], [1.0, 1.0, 500.0], [30.0, 30.0, 30.0]])
    k, e = M.topk_oracle(errs)
    assert k == 2 and np.all(e == 30.0)          # not the per-point minimum (which would be [1, 1, 30])


def test_bootstrap_single_cluster_degenerates():
    tab = pd.DataFrame({"episode_id": ["a", "b"], "parent": ["d0", "d0"], "ade": [10.0, 30.0]})
    out = M.aggregate(tab, "ade", n_boot=50)
    assert out.loc[0, "mean"] == 20.0 and out.loc[0, "mean_lo"] == out.loc[0, "mean_hi"] == 20.0


def test_success_threshold_definition():
    err = np.r_[np.full(9, 10.0), 200.0]
    s = M.episode_summary(err)
    assert s["c50"] == 0.9 and s["s50_90"] is True and s["s100_80"] is True


def test_predfile_candidate_consistency():
    df = pd.concat([P.new_frame("e", [1.0, 2.0], [[0, 0], [1, 1]], candidate_id=0, prob=0.5),
                    P.new_frame("e", [1.0], [[0, 0]], candidate_id=1, prob=0.5)])
    with pytest.raises(ValueError):
        P.validate(df, pd.DataFrame({"episode_id": ["e", "e"], "t": [1.0, 2.0]}))


def test_cluster_bootstrap_mean_fast_path_matches_loop():
    """The vectorised mean bootstrap must reproduce the resampling loop (same seed, same draws)."""
    import numpy as np
    from cantraj import metrics as M
    rng = np.random.default_rng(0)
    vals = rng.exponential(100, 3000)
    clus = rng.integers(0, 40, 3000)
    m_fast, (lo_f, hi_f) = M.cluster_bootstrap(vals, clus, np.mean, 300)
    def loop_mean(values, clusters, n_boot, seed=20260709):
        uniq, inv = np.unique(clusters, return_inverse=True)
        groups = [values[inv == i] for i in range(len(uniq))]
        r = np.random.default_rng(seed)
        boots = np.array([np.mean(np.concatenate([groups[i] for i in r.integers(0, len(uniq), len(uniq))])) for _ in range(n_boot)])
        return float(np.mean(values)), tuple(np.percentile(boots, [2.5, 97.5]))
    m_loop, (lo_l, hi_l) = loop_mean(vals, clus, 300)
    assert abs(m_fast - m_loop) < 1e-9
    assert abs(lo_f - lo_l) < 1e-6 and abs(hi_f - hi_l) < 1e-6
