"""Signal-combination study: feature columns of absent signals stay zero, the online provider reproduces the training
features bitwise, and the uncertainty-aware filter explores turns when the source only knows THAT a turn is likely."""
import math

import numpy as np

from cantraj.learn.sig import F, GROUPS, mask, sig_features
from cantraj.baselines.map_pf import MapParticleFilter
from cantraj.events import EventStream
from cantraj.inputs import EpisodeInit, EpisodeInputs, StaticParams
from cantraj import replay
from test_map_beam import grid_graph

STATIC = dict(wheelbase_m=2.7, steer_ratio=15.0, brand="B1")
SCALES = {"B1": (0.0, 1.0, 50.0)}


def _feats(group, n=50):
    rng = np.random.default_rng(0)
    v = rng.uniform(0, 20, n)
    st = rng.uniform(-90, 90, n)
    gas, brk = rng.uniform(0, 1, n), rng.uniform(0, 40, n)
    gp, bp = (gas > 0.5).astype(float), (brk > 20).astype(float)
    pv, ps = np.concatenate([[v[0]], v[:-1]]), np.concatenate([[st[0]], st[:-1]])
    return sig_features(v, st, gas, brk, gp, bp, np.zeros(n), np.zeros(n), pv, ps, STATIC, SCALES, 0.01, group), (v, st, gas, brk, gp, bp, pv, ps)


def test_absent_signals_leave_their_columns_zero():
    for g in GROUPS:
        f, _ = _feats(g)
        assert np.all(f[:, mask(g) == 0] == 0.0), g
    f1, _ = _feats("S1")
    f3, _ = _feats("S3")
    assert np.array_equal(f1[:, :6], f3[:, :6]) and np.any(f3[:, 6:13] != 0)


def test_scalar_and_vector_features_agree():
    f, (v, st, gas, brk, gp, bp, pv, ps) = _feats("S4c")
    for k in (0, 7, 31):
        row = sig_features(v[k], st[k], gas[k], brk[k], gp[k], bp[k], 0.0, 0.0, pv[k], ps[k], STATIC, SCALES, 0.01, "S4c")[0]
        assert np.array_equal(row, f[k])


class _UnsignedTurnSource:
    """Knows when a turn is likely (slow speed) but not its direction: signed rate 0, turn magnitude when slow."""
    name, needs = "fake", ("speed",)

    def reset(self, init, static):
        self.v = 10.0

    def ingest(self, t, view):
        self.v = view["speed"]
        return 0.0, 1.0

    def sync(self, t):
        pass

    def rates(self):
        return 0.0, 1.0

    def turn_mag(self):
        return 0.35 if self.v < 5.0 else 0.0


def test_turn_budget_filter_explores_turns():
    T = 45.0
    t = np.arange(0, T, 0.01)
    speed = np.where((t > 17.0) & (t < 24.0), 3.0, 10.0)
    ev = EventStream(t, {"speed": speed})
    inp = EpisodeInputs("G1", ev, StaticParams(), EpisodeInit(0.0, 37.0, -122.0, 0.0), input_layer="events")
    inp.map_graph = grid_graph()
    shares = {}
    for unc in (True, False):
        pf = MapParticleFilter("steer", seed=0, omega_source=_UnsignedTurnSource())
        if not unc:
            pf.unc = False
        replay.run_online(pf, inp, np.arange(1.0, T, 1.0))
        b = pf.g.bearing[pf.e]
        shares[unc] = float(np.mean(np.abs(np.cos(b)) < 0.5))        # particles on north-south roads
    assert shares[True] > 0.2 and shares[True] > shares[False] + 0.15
