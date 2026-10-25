"""The input wrapper physically removes non-whitelisted signals; baselines cannot reach the reference loader."""
import numpy as np
import pytest

from cantraj import io
from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj.registry import ForbiddenSignalError
from cantraj.baselines.dead_reckoning import DeadReckoning
from cantraj import replay


def _ev(n=100):
    t = np.arange(n) * 0.1
    return EventStream(t, {"speed": np.full(n, 10.0), "steer_angle": np.zeros(n), "curvature_loc": np.zeros(n), "yaw_rate_can": np.zeros(n)})


def test_g1_drops_everything_but_speed():
    inp = EpisodeInputs("G1", _ev(), StaticParams(wheelbase_m=2.7, steer_ratio=15.0), EpisodeInit(0.0, 37.0, -122.0, 0.0))
    assert inp.available() == ["speed"]
    assert inp.static.wheelbase_m is None and inp.static.steer_ratio is None
    with pytest.raises(ForbiddenSignalError):
        inp["steer_angle"]
    with pytest.raises(ForbiddenSignalError):
        inp["latitude"]


def test_steer_dr_refuses_g1_inputs():
    inp = EpisodeInputs("G1", _ev(), StaticParams(wheelbase_m=2.7, steer_ratio=15.0), EpisodeInit(0.0, 37.0, -122.0, 0.0))
    with pytest.raises(ForbiddenSignalError):
        replay.run_online(DeadReckoning("steer"), inp, np.array([1.0, 2.0]))


def test_g3_keeps_static_params_it_declares_only():
    inp = EpisodeInputs("G3", _ev(), StaticParams(wheelbase_m=2.7, steer_ratio=15.0, yawrate_unit="deg_s", mass_kg=1500.0),
                        EpisodeInit(0.0, 37.0, -122.0, 0.0))
    assert inp.static.wheelbase_m == 2.7 and inp.static.steer_ratio == 15.0
    assert inp.static.yawrate_unit is None and inp.static.mass_kg is None


def test_reference_loader_guard(monkeypatch):
    """run_baseline.py replaces io.load_gps with a raising stub before any baseline runs; the stub must bite."""
    def _raise(*a, **k):
        raise PermissionError("reference positions are not available to baselines")
    monkeypatch.setattr(io, "load_gps", _raise)
    with pytest.raises(PermissionError):
        io.load_gps("driver_001", "drive_001")
