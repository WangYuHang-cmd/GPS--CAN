"""Perturbation probe: exact quantisation, causal steering hold, determinism per episode, untouched other columns."""
import numpy as np

from cantraj.events import EventStream
from cantraj.perturb import episode_rng, parse_spec, perturb_stream


def _ev(n=500, dt=0.01):
    t = np.arange(n) * dt
    rng = np.random.default_rng(1)
    return EventStream(t, {"speed": 10 + rng.normal(0, 1, n), "steer_angle": rng.normal(0, 20, n), "other": rng.normal(0, 1, n)})


def test_quantisation_and_noise():
    ev = _ev()
    p = perturb_stream(ev, parse_spec("speed_quant=1"), episode_rng("e", "speed_quant=1"))
    assert np.allclose(p.cols["speed"], np.round(p.cols["speed"]))
    assert np.array_equal(p.cols["other"], ev.cols["other"]) and np.array_equal(p.t_s, ev.t_s)
    q = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("e", "speed_noise=0.5"))
    d = q.cols["speed"] - ev.cols["speed"]
    assert 0.3 < d.std() < 0.7 and (q.cols["speed"] >= 0).all()


def test_steer_hold_is_causal_zoh():
    ev = _ev()
    p = perturb_stream(ev, parse_spec("steer_hz=1"), episode_rng("e", "steer_hz=1"))
    st = p.cols["steer_angle"]
    # within each 1 s bin the value equals the bin's first sample, which is an earlier-or-equal event
    bins = np.floor(ev.t_s).astype(int)
    for b in np.unique(bins):
        k = np.flatnonzero(bins == b)
        assert np.all(st[k] == ev.cols["steer_angle"][k[0]])
    # changing future events never changes the past
    ev2 = _ev(); ev2.cols["steer_angle"][ev2.t_s >= 3.0] += 100.0
    p2 = perturb_stream(ev2, parse_spec("steer_hz=1"), episode_rng("e", "steer_hz=1"))
    assert np.array_equal(p2.cols["steer_angle"][ev2.t_s < 3.0], st[ev.t_s < 3.0])


def test_determinism_per_episode():
    ev = _ev()
    a = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("drive/ep1", "speed_noise=0.5"))
    b = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("drive/ep1", "speed_noise=0.5"))
    c = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("drive/ep2", "speed_noise=0.5"))
    assert np.array_equal(a.cols["speed"], b.cols["speed"]) and not np.array_equal(a.cols["speed"], c.cols["speed"])
