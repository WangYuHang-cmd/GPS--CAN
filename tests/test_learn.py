"""Learned motion baselines: numpy GRU equals torch, online wrapper is causal and reference-free, the residual
model with zero parameters and no GRU equals nominal dead reckoning, unknown platforms run."""
import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from cantraj.events import EventStream  # noqa: E402
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams  # noqa: E402
from cantraj.learn.models import MotionGRU, ResidualGRU  # noqa: E402
from cantraj.learn.provider import NumpyGRU, MotionProvider  # noqa: E402
from cantraj.learn.train import export_numpy  # noqa: E402
from cantraj.learn.data import N_FEAT  # noqa: E402
from cantraj.baselines.learned import LearnedDR  # noqa: E402
from cantraj.baselines.dead_reckoning import DeadReckoning  # noqa: E402
from cantraj import replay  # noqa: E402


def _export(model, kind, tmp_path, vocab=None):
    d = tmp_path / kind
    d.mkdir(parents=True, exist_ok=True)
    export_numpy(model, str(d / "weights.npz"))
    vocab = vocab or {"platform": {"P1": 1}, "brand": {"B1": 1}}
    json.dump(dict(kind=kind, vocab=vocab, norm=dict(mean=[0.0] * N_FEAT, std=[1.0] * N_FEAT), config={}), open(d / "meta.json", "w"))
    return str(d)


def test_numpy_gru_matches_torch():
    torch.manual_seed(0)
    g = torch.nn.GRU(7, 5, batch_first=True)
    x = torch.randn(1, 20, 7)
    with torch.no_grad():
        out, _ = g(x)
    ng = NumpyGRU(g.weight_ih_l0.detach().numpy(), g.weight_hh_l0.detach().numpy(), g.bias_ih_l0.detach().numpy(), g.bias_hh_l0.detach().numpy())
    h = np.zeros(5, dtype=np.float32)
    outs = []
    for k in range(20):
        h = ng.step(x[0, k].numpy(), h)
        outs.append(h.copy())
    assert np.allclose(np.array(outs), out[0].numpy(), atol=1e-5)


def _circle_inputs(theta0=0.0, T=120.0):
    t = np.arange(0, T, 0.01)
    st = StaticParams(wheelbase_m=2.7, steer_ratio=15.0, platform="P1", brand="B1")
    ev = EventStream(t, {"speed": np.full(len(t), 10.0), "steer_angle": np.full(len(t), 20.0)})
    return EpisodeInputs("G3", ev, st, EpisodeInit(0.0, 37.0, -122.0, theta0)), t


def test_residual_without_gru_equals_nominal_dr(tmp_path):
    m = ResidualGRU(2, 2, use_gru=False)
    d = _export(m, "resid_nogru", tmp_path)
    inp, t = _circle_inputs()
    q = np.arange(5.0, 120.0, 5.0)
    xy_l, _ = replay.run_online(LearnedDR(d), inp, q)
    xy_d, _ = replay.run_online(DeadReckoning("steer", offset_mode="none"), inp, q)
    assert np.abs(xy_l - xy_d).max() < 2.0          # tick-grid vs event-rate integration of the same model


def test_wrapper_is_causal(tmp_path):
    torch.manual_seed(1)
    m = MotionGRU(3, 3)
    with torch.no_grad():
        m.head.weight.normal_(0, 0.5)
    d = _export(m, "gru", tmp_path, {"platform": {"P1": 1, "P2": 2}, "brand": {"B1": 1, "B2": 2}})
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 200, 2500))
    speed = 8 + 3 * np.sin(t / 15)
    steer = 6 * np.sin(t / 9)
    st = StaticParams(wheelbase_m=2.7, steer_ratio=15.0, platform="P1", brand="B1")
    init = EpisodeInit(0.0, 37.0, -122.0, 0.4)
    q = np.arange(10.0, 200.0, 10.0)
    a = EpisodeInputs("G3", EventStream(t, {"speed": speed, "steer_angle": steer}), st, init)
    xy0, _ = replay.run_online(LearnedDR(d), a, q)
    speed2, steer2 = speed.copy(), steer.copy()
    after = t > 100.0
    speed2[after] = rng.uniform(0, 30, after.sum())
    steer2[after] = rng.uniform(-100, 100, after.sum())
    b = EpisodeInputs("G3", EventStream(t, {"speed": speed2, "steer_angle": steer2}), st, init)
    xy1, _ = replay.run_online(LearnedDR(d), b, q)
    assert np.array_equal(xy0[q <= 100.0], xy1[q <= 100.0])
    assert not np.allclose(xy0[q > 100.0], xy1[q > 100.0])


def test_unknown_platform_runs(tmp_path):
    m = ResidualGRU(2, 2, use_gru=True)
    d = _export(m, "resid", tmp_path)
    inp, t = _circle_inputs()
    inp = EpisodeInputs("G3", inp.events, StaticParams(wheelbase_m=2.7, steer_ratio=15.0, platform="UNSEEN", brand="NOBRAND"), inp.init)
    xy, th = replay.run_online(LearnedDR(d), inp, np.array([30.0, 60.0]))
    assert np.isfinite(xy).all()
    p = MotionProvider(d)
    assert p.vocab["platform"].get("UNSEEN", 0) == 0


def test_learned_module_has_no_reference_loader():
    import cantraj.baselines.learned as L
    import cantraj.learn.provider as P
    src = open(L.__file__).read() + open(P.__file__).read()
    assert "load_gps" not in src and "cantraj.io" not in src and "from .. import io" not in src


def test_feats_full_fast_path_matches_slow_path():
    """The precomputed per-drive feature matrix (feats_full) must give the same chunk features as the per-chunk path."""
    import numpy as np
    from cantraj.learn import data as D
    rng = np.random.default_rng(3)
    n = 5000
    v = np.abs(rng.normal(12, 5, n)).astype(np.float32); v[rng.random(n) < 0.05] = np.nan
    st = rng.normal(0, 40, n).astype(np.float32); st[rng.random(n) < 0.03] = np.nan
    age = np.clip(rng.exponential(0.05, n), 0, 1).astype(np.float32)
    miss = (rng.random(n) < 0.02).astype(np.float32)
    static = {"wheelbase_m": 2.7, "steer_ratio": 14.2, "nominal_dt": 0.01}
    prev_v = np.concatenate([[v[0]], v[:-1]]); prev_st = np.concatenate([[st[0]], st[:-1]])
    full = D.features(v, st, age, miss, prev_v, prev_st, static, 0.01)
    for _ in range(20):
        j0 = int(rng.integers(0, n - 700)); j1 = j0 + int(rng.integers(50, 600))
        sl = slice(j0, j1 + 1)
        pv = np.concatenate([[v[sl][0]], v[sl][:-1]]); ps = np.concatenate([[st[sl][0]], st[sl][:-1]])
        slow = D.features(v[sl], st[sl], age[sl], miss[sl], pv, ps, static, 0.01)
        fast = full[sl].copy(); fast[0, 3] = 0.0; fast[0, 4] = 0.0
        assert fast.shape == slow.shape
        assert np.array_equal(np.nan_to_num(fast), np.nan_to_num(slow))


def test_speed_noise_augmentation_touches_only_speed_columns():
    import numpy as np
    from cantraj.learn import data as D
    rng = np.random.default_rng(5)
    n = 400
    v = np.abs(rng.normal(12, 4, n)).astype(np.float32); st = rng.normal(0, 30, n).astype(np.float32)
    age = np.zeros(n, np.float32); miss = np.zeros(n, np.float32)
    static = {"wheelbase_m": 2.7, "steer_ratio": 14.0, "nominal_dt": 0.01}
    prev_v = np.concatenate([[v[0]], v[:-1]]); prev_st = np.concatenate([[st[0]], st[:-1]])
    f = D.features(v, st, age, miss, prev_v, prev_st, static, 0.01)
    same = D.apply_speed_noise(f, v, st, static, 0.0, np.random.default_rng(1))
    assert np.allclose(same[:, [0, 7, 12]], f[:, [0, 7, 12]]) and np.allclose(same[1:, 4], f[1:, 4])
    noisy = D.apply_speed_noise(f, v, st, static, 0.5, np.random.default_rng(1))
    untouched = [c for c in range(f.shape[1]) if c not in (0, 4, 7, 12)]
    assert np.array_equal(noisy[:, untouched], f[:, untouched])
    assert 0.3 < (noisy[:, 0] - f[:, 0]).std() * 10.0 < 0.7          # column 0 is v / 10
    # the recomputed omega_nom column is consistent with the noisy speed column
    vn = noisy[:, 0] * 10.0
    om = vn * np.tan(np.clip(np.radians(np.clip(st, -300, 300)) / 14.0, -1.2, 1.2)) / 2.7
    assert np.allclose(noisy[:, 12], om / 0.3, atol=1e-4)
