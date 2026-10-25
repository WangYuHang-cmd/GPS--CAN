"""Causality on the raw event stream: changing messages logged after t_cut must not change predictions at or before
t_cut, through preprocessing (EpisodeInputs) and the model (replay). Also: an episode init must not depend on
reference rows after t0."""
import numpy as np
import pandas as pd

from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.baselines.dead_reckoning import DeadReckoning
from cantraj.episodes import find_anchor, view_1hz
from cantraj import replay


def _stream(seed=0, n=3000):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 300.0, n))
    speed = 10 + 3 * np.sin(t / 20)
    steer = 5 * np.sin(t / 7) + rng.normal(0, 0.5, n)
    return t, speed, steer


def test_suffix_perturbation_leaves_prefix_predictions_unchanged():
    t, speed, steer = _stream()
    t_cut = 150.0
    static = StaticParams(wheelbase_m=2.7, steer_ratio=15.0)
    init = EpisodeInit(0.0, 37.0, -122.0, 0.3)
    q = np.arange(10.0, 300.0, 10.0)
    base = EpisodeInputs("G3", EventStream(t, {"speed": speed, "steer_angle": steer}), static, init)
    xy0, th0 = replay.run_online(DeadReckoning("steer", offset_mode="online"), base, q)
    rng = np.random.default_rng(1)
    speed2, steer2 = speed.copy(), steer.copy()
    after = t > t_cut
    speed2[after] = rng.uniform(0, 30, after.sum())
    steer2[after] = rng.uniform(-200, 200, after.sum())
    pert = EpisodeInputs("G3", EventStream(t, {"speed": speed2, "steer_angle": steer2}), static, init)
    xy1, th1 = replay.run_online(DeadReckoning("steer", offset_mode="online"), pert, q)
    before = q <= t_cut
    assert np.array_equal(xy0[before], xy1[before]) and np.array_equal(th0[before], th1[before])
    assert not np.allclose(xy0[~before], xy1[~before])


def test_event_timestamps_not_grid_times_define_visibility():
    """An event logged at 10.05 must not influence the prediction at 10.0 even if a grid would place it at 10.0."""
    t = np.array([9.0, 9.5, 10.05, 10.5])
    static = StaticParams(wheelbase_m=2.7, steer_ratio=15.0)
    init = EpisodeInit(0.0, 37.0, -122.0, 0.0)
    a = EpisodeInputs("G3", EventStream(t, {"speed": [10, 10, 10, 10.0], "steer_angle": [0, 0, 0, 0.0]}), static, init)
    b = EpisodeInputs("G3", EventStream(t, {"speed": [10, 10, 10, 10.0], "steer_angle": [0, 0, 90, 90.0]}), static, init)
    xa, _ = replay.run_online(DeadReckoning("steer", offset_mode="none"), a, np.array([10.0]))
    xb, _ = replay.run_online(DeadReckoning("steer", offset_mode="none"), b, np.array([10.0]))
    assert np.array_equal(xa, xb)


def test_anchor_uses_only_the_anchor_row():
    rows = []
    for s in range(0, 40):
        rows.append(dict(time_s=100.0 + s, lat=37.0 + 1e-4 * s, lon=-122.0, speed_mps=0.5 if s < 10 else 8.0,
                         bearing_deg=45.0 + s, bearing_acc_deg=1.0, h_acc_m=2.0, source="ublox"))
    ref = pd.DataFrame(rows)
    ref1 = view_1hz(ref)
    anc = find_anchor(ref1, lambda t: 8.0, 100.0)
    assert anc["t0"] == 110.0 and anc["bearing_deg"] == 55.0
    ref2 = ref.copy()
    ref2.loc[ref2["time_s"] > 110.0, ["bearing_deg", "lat", "lon"]] = [999.0, 0.0, 0.0]
    anc2 = find_anchor(view_1hz(ref2), lambda t: 8.0, 100.0)
    assert anc2 == anc
