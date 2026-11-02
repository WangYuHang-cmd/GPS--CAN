"""Perturbation probe: exact quantisation, causal holds and delay, drive-keyed steering noise shared by overlapping
windows, determinism per episode, untouched other columns, unknown keys rejected."""
import numpy as np
import pytest

from cantraj.events import EventStream
from cantraj.perturb import episode_rng, parse_spec, perturb_stream


def _ev(n=500, dt=0.01, t_start=0.0):
    t = t_start + np.arange(n) * dt
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


def test_steer_quantisation_exact():
    ev = _ev()
    ev.cols["steer_angle"][10] = np.nan
    p = perturb_stream(ev, parse_spec("steer_quant=5"), episode_rng("e", "steer_quant=5"))
    st = p.cols["steer_angle"]
    assert np.isnan(st[10]) and np.allclose(st[~np.isnan(st)] % 5, 0)
    assert np.abs(st[~np.isnan(st)] - ev.cols["steer_angle"][~np.isnan(st)]).max() <= 2.5 + 1e-9
    assert np.array_equal(p.cols["speed"], ev.cols["speed"])


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


def test_speed_hold_is_causal_zoh():
    ev = _ev()
    p = perturb_stream(ev, parse_spec("speed_hz=1"), episode_rng("e", "speed_hz=1"))
    v = p.cols["speed"]
    bins = np.floor(ev.t_s).astype(int)
    for b in np.unique(bins):
        k = np.flatnonzero(bins == b)
        assert np.all(v[k] == ev.cols["speed"][k[0]])
    assert np.array_equal(p.cols["steer_angle"], ev.cols["steer_angle"])


def test_steer_delay_reads_the_past_only():
    ev = _ev()
    p = perturb_stream(ev, parse_spec("steer_delay=0.5"), episode_rng("e", "steer_delay=0.5"))
    st, x, t = p.cols["steer_angle"], ev.cols["steer_angle"], ev.t_s
    for i in (0, 25, 49, 50, 200, 499):
        j = np.flatnonzero(t <= t[i] - 0.5)
        assert st[i] == (x[j[-1]] if len(j) else x[0])
    ev2 = _ev(); ev2.cols["steer_angle"][ev2.t_s >= 3.0] += 100.0
    p2 = perturb_stream(ev2, parse_spec("steer_delay=0.5"), episode_rng("e", "steer_delay=0.5"))
    assert np.array_equal(p2.cols["steer_angle"][ev2.t_s < 3.5], st[ev.t_s < 3.5])
    assert not np.array_equal(p2.cols["steer_angle"], st)


def test_steer_noise_is_binned_and_shared_by_overlapping_windows():
    spec = parse_spec("steer_noise=5")
    a = _ev(n=500)                                   # a window covering drive seconds 0 to 5
    pa = perturb_stream(a, spec, episode_rng("d/ep1", "steer_noise=5"), drive_key="d1/d")
    off = pa.cols["steer_angle"] - a.cols["steer_angle"]
    bins = np.floor(a.t_s).astype(int)
    per_bin = np.array([off[bins == b][0] for b in np.unique(bins)])
    for b in np.unique(bins):
        assert np.allclose(off[bins == b], per_bin[b])        # constant within a bin
    assert len(np.unique(np.round(per_bin, 9))) == len(per_bin)   # and different across bins
    # a window of the same drive that starts 2 s later sees the same offsets at the same drive seconds
    b_ev = _ev(n=300, t_start=2.0)
    pb = perturb_stream(b_ev, spec, episode_rng("d/ep2", "steer_noise=5"), drive_key="d1/d")
    offb = pb.cols["steer_angle"] - b_ev.cols["steer_angle"]
    assert np.allclose(offb[np.floor(b_ev.t_s).astype(int) == 3], per_bin[3])
    # another drive, or another release seed, gets other offsets; the speed column is untouched
    pc = perturb_stream(a, spec, episode_rng("d/ep1", "steer_noise=5"), drive_key="d2/d")
    assert not np.allclose(pc.cols["steer_angle"], pa.cols["steer_angle"])
    pd_ = perturb_stream(a, spec, episode_rng("d/ep1", "steer_noise=5"), drive_key="d1/d", release_seed=1)
    assert not np.allclose(pd_.cols["steer_angle"], pa.cols["steer_angle"])
    assert np.array_equal(pa.cols["speed"], a.cols["speed"])
    # many bins: the offsets have about the requested spread
    long = _ev(n=60000)
    pl = perturb_stream(long, spec, episode_rng("x", "steer_noise=5"), drive_key="d1/d")
    o = pl.cols["steer_angle"] - long.cols["steer_angle"]
    assert 3.5 < np.std([o[np.floor(long.t_s).astype(int) == b][0] for b in range(600)]) < 6.5


def test_fixed_order_noise_quantise_hold_delay():
    ev = _ev()
    spec = parse_spec("steer_delay=0.5,steer_hz=1,steer_quant=5,steer_noise=2")
    p = perturb_stream(ev, spec, episode_rng("e", "x"), drive_key="d/d")
    # whatever the spec order, the result is quantised (noise came first), held and then delayed
    st = p.cols["steer_angle"]
    assert np.allclose(st % 5, 0)
    assert np.all(st[(ev.t_s >= 0.5) & (ev.t_s < 1.5)] == st[np.flatnonzero(ev.t_s >= 0.5)[0]])


def test_determinism_per_episode_and_unknown_key():
    ev = _ev()
    a = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("drive/ep1", "speed_noise=0.5"))
    b = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("drive/ep1", "speed_noise=0.5"))
    c = perturb_stream(ev, parse_spec("speed_noise=0.5"), episode_rng("drive/ep2", "speed_noise=0.5"))
    assert np.array_equal(a.cols["speed"], b.cols["speed"]) and not np.array_equal(a.cols["speed"], c.cols["speed"])
    with pytest.raises(ValueError):
        parse_spec("steer_jitter=1")
