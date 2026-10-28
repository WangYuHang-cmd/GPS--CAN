"""Map-matching particle filter on the OSM drive graph (online Top-1).

Particle = (edge, offset_m, theta_p). Motion from CAN: offset += v * dt * (1 + eps_s); theta_p += omega * dt + noise,
omega from the same heading-rate sources as dead reckoning (steer / can_yaw / curvature) or none (G1: speed only).
At a node the next edge is drawn with p(e') proportional to exp(-(wrap(bearing(e') - theta_p))^2 / (2 sigma_turn^2))
(G1: uniform over successors, no U-turn), respecting turn restrictions; U-turns are down-weighted. Once per query the
weights are multiplied by the heading consistency exp(-(wrap(theta_p - bearing(e)))^2 / (2 sigma_h^2)) (G1: speed vs
road-class plausibility instead), theta_p is softly re-anchored to the edge bearing, particles are resampled when
ESS < N/2, and the output is the weighted mean position on the maximum-weight edge. H+: start edges within 50 m of p0
whose bearing is within 45 deg of theta0; H-: all directions, theta_p initialised to each edge's bearing.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .base import OfflineBaseline, OnlineBaseline, Prediction
from .dead_reckoning import SOURCE_NEEDS, UNIT, IMU_SOURCES, make_gyro_bias, steer_term

ROAD_VMAX = {"motorway": 40, "motorway_link": 30, "trunk": 35, "trunk_link": 25, "primary": 30, "primary_link": 22,
             "secondary": 27, "secondary_link": 20, "tertiary": 24, "tertiary_link": 18, "unclassified": 20,
             "residential": 18, "living_street": 10, "road": 20}


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class MapParticleFilter(OnlineBaseline, OfflineBaseline):
    def __init__(self, source="steer", n_particles=2000, sigma_speed=0.03, sigma_turn_deg=20.0, sigma_heading_deg=15.0,
                 sigma_theta_rate=0.01, uturn_weight=0.05, start_radius_m=50.0, seed=0, offset_mode="none", use_eff_ratio=False,
                 omega_source=None, tick_hz=10.0, readout="mean"):
        """tick_hz: the filter advances on a causal 1/tick_hz grid (zero-order hold of the latest event at or before each
        tick; an event older than 0.3 s at the tick is a gap -> omega = 0). None = advance on every event (10x slower at 100 Hz).
        readout: "mean" (online: weighted mean on the most probable edge at each query) or "lineage" (offline decode-output
        control: the same filter run causally, then the ancestry of the heaviest final particle is traced back and its
        edge / offset at every query time is the output; whole-path, like the beam)."""
        self.readout = readout
        self.source, self.N = source, n_particles
        self.tick_dt = (1.0 / float(tick_hz)) if tick_hz else None
        self.use_eff_ratio = use_eff_ratio
        self.omega_source = omega_source          # a MotionProvider-like object: reset(init, static), ingest(t, view) -> (omega, speed_scale)
        self.sigma_speed, self.sigma_turn, self.sigma_h = sigma_speed, math.radians(sigma_turn_deg), math.radians(sigma_heading_deg)
        self.sigma_theta_rate, self.uturn_weight, self.start_radius = sigma_theta_rate, uturn_weight, start_radius_m
        self.rng = np.random.default_rng(seed)
        self.offset_mode = offset_mode
        self.name = f"map_pf_{source}" + ("_sreff" if use_eff_ratio else "") + (f"_off{offset_mode}" if offset_mode != "none" else "") + ("_lineage" if readout == "lineage" else "")
        self.needs = SOURCE_NEEDS[source]
        self.unc = False
        if omega_source is not None:
            self.name = f"map_pf_{omega_source.name}"
            self.needs = tuple(getattr(omega_source, "needs", ("speed", "steer_angle")))
            # turn budget (signal-combination study): the source also reports an unsigned turn magnitude
            self.unc = hasattr(omega_source, "turn_mag")

    def _omega(self, v, view):
        s = self.static
        if self.source == "none":
            return 0.0
        if self.source == "steer":
            off = float(s.angle_offset_train_deg or 0.0) if self.offset_mode == "fixed" else 0.0
            sr = s.steer_ratio_eff_train if (self.use_eff_ratio and s.steer_ratio_eff_train) else s.steer_ratio
            return v * math.tan(math.radians(view["steer_angle"] - off) / sr) / s.wheelbase_m
        if self.source == "can_yaw":
            return (s.yawrate_sign or 1) * UNIT.get(s.yawrate_unit or "rad_s", 1.0) * view["yaw_rate_can"]
        if self.source == "curvature":
            return -v * view["curvature_loc"]
        if self.source == "loc_yaw":
            return -view["yaw_rate_loc"]
        if self.source in IMU_SOURCES:
            return view["imu_gyro_x"] - self.gyro_bias.value
        raise ValueError(self.source)

    def reset(self, init, static, map_graph=None):
        self._pending = None
        self.gyro_bias = make_gyro_bias(self.source)
        if map_graph is None:
            raise ValueError("map_pf needs a map graph")
        self.g, self.static = map_graph, static
        g = self.g
        radius = self.start_radius
        cands = []
        while not cands and radius <= 400:
            cands = g.nearest_edges(*init.p0_xy, radius)
            if init.theta0 is not None and cands:
                cands = [e for e in cands if abs(wrap(g.bearing[e] - init.theta0)) < math.radians(45)] or cands
            radius *= 2
        if not cands:   # v1 fell back to the globally shortest edge here; v1.1 never runs such episodes (map_start_ok rule)
            raise ValueError("no drivable edge within 400 m of the start (excluded by the v1.1 map_start_ok rule)")
        self.blind = init.theta0 is None
        per = max(1, self.N // len(cands))
        edges, offs, thetas = [], [], []
        for e in cands:
            off, _ = g.project(e, 0.0, 0.0)
            edges += [e] * per
            offs += [off] * per
            thetas += [g.bearing[e] if self.blind else init.theta0] * per
        self.e = np.array(edges, dtype=np.int64)
        self.off = np.array(offs, dtype=float) + self.rng.normal(0, 3.0, len(edges))
        self.th = np.array(thetas, dtype=float) + self.rng.normal(0, math.radians(3.0), len(edges))
        self.w = np.full(len(edges), 1.0 / len(edges))
        self.t_last, self.v, self.om = init.t0, 0.0, 0.0
        self.v_scale = 1.0
        self.n_gap = 0
        self._unfed = deque()
        self.U = np.zeros(len(self.e))            # turn budget per particle (rad): turn magnitude not explained by the signed rate
        self.mag = 0.0
        if self.omega_source is not None:
            self.omega_source.reset(init, static)

    def _feed_until(self, t):
        """Give the learned source every not yet ingested event logged at or before t, at its own logged time (its age /
        missing features and tick commits then match training exactly)."""
        q = self._unfed
        while q and q[0][0] <= t:
            te, *vals = q.popleft()
            self.omega_source.ingest(te, dict(zip(self.needs, vals)))

    def _advance(self, dt):
        g = self.g
        n = len(self.e)
        ds = self.v * self.v_scale * dt * (1.0 + self.rng.normal(0, self.sigma_speed, n))
        self.th = self.th + self.om * dt + self.rng.normal(0, self.sigma_theta_rate * math.sqrt(max(dt, 1e-3)), n)
        self.off = self.off + np.maximum(ds, 0.0)
        if self.unc:
            self.U = self.U * math.exp(-dt / 6.0) + max(self.mag - abs(self.om), 0.0) * dt
        # particles that ran past the end of their edge choose a successor; grouped by edge so each edge's successor list
        # is fetched once and all its particles are sampled in one vectorised draw (rounds handle multi-edge hops)
        for _round in range(20):
            over = np.flatnonzero(self.off > g.length[self.e])
            if len(over) == 0:
                break
            for e_cur in np.unique(self.e[over]):
                idx = over[self.e[over] == e_cur]
                succ = g.succ(int(e_cur))
                if not succ:
                    self.off[idx] = g.length[e_cur]
                    self.w[idx] *= 0.1
                    continue
                rem = self.off[idx] - g.length[e_cur]
                if self.source == "none":
                    opts = [s_ for s_ in succ if not s_[2]] or succ
                    k = self.rng.integers(len(opts), size=len(idx))
                    e2 = np.array([opts[j][0] for j in k])
                else:
                    bear = np.array([g.bearing[s_[0]] for s_ in succ])
                    utw = np.array([self.uturn_weight if s_[2] else 1.0 for s_ in succ])
                    p = np.exp(-0.5 * (wrap(bear[None, :] - self.th[idx, None]) / self.sigma_turn) ** 2)
                    if self.unc:                          # unsigned evidence: |turn| close to the particle's turn budget
                        phi = np.abs(np.array([s_[1] for s_ in succ]))
                        q = (self.U[idx] / (self.U[idx] + 0.3))[:, None]
                        p = (1.0 - q) * p + q * np.exp(-0.5 * ((phi[None, :] - self.U[idx, None]) / 0.35) ** 2)
                    p = p * utw[None, :]
                    psum = p.sum(axis=1, keepdims=True)
                    p = np.where(psum > 0, p / np.where(psum > 0, psum, 1.0), 1.0 / len(succ))
                    cum = np.cumsum(p, axis=1)
                    u = self.rng.random(len(idx))
                    k = np.minimum((u[:, None] > cum).sum(axis=1), len(succ) - 1)
                    e2 = np.array([succ[j][0] for j in k])
                    if self.unc:                          # a taken turn resolves the heading and spends the budget
                        phi_k = np.abs(np.array([succ[j][1] for j in k]))
                        turned = phi_k > math.radians(30)
                        if turned.any():
                            ii = idx[turned]
                            self.th[ii] = g.bearing[e2[turned]]
                            self.U[ii] = np.maximum(self.U[ii] - phi_k[turned], 0.0)
                self.e[idx] = e2
                self.off[idx] = rem

    def warmup(self, events):
        if self.source not in IMU_SOURCES or len(events) < 2:
            return
        from .dead_reckoning import DeadReckoning
        helper = DeadReckoning(self.source)
        helper.static, helper.gyro_bias, helper.nominal_dt = self.static, self.gyro_bias, events.nominal_dt
        helper.v = 0.0
        helper.warmup(events)
        self.gyro_bias = helper.gyro_bias

    def _step(self, t, view, gap):
        dt = float(t - self.t_last)
        if dt < 0:
            return
        if gap:
            self.n_gap += 1
            self.om = 0.0
        if dt > 0:
            self._advance(dt)
        if math.isfinite(view["speed"]):
            self.v = view["speed"]
        if self.source in IMU_SOURCES and not gap and dt > 0:
            ref = None
            if self.source == "imu_yaw_sb" and self.static.steer_ratio and self.static.wheelbase_m and math.isfinite(view.get("steer_angle", math.nan)):
                ref = self.v * math.tan(math.radians(view["steer_angle"]) / self.static.steer_ratio) / self.static.wheelbase_m
            self.gyro_bias.update(dt, self.v, view["imu_gyro_x"], ref, steer_term(self.static, self.v, view) if self.source == "imu_yaw_rls" else None)
        if self.omega_source is not None:
            # v1.1: the source has seen every event <= t at its logged time; commit its tick at t and use that tick's rates
            # for the next interval, the same timing as the nominal sources (v1 passed the tick time as the event time and
            # read the previous tick: zero age features, one extra tick of lag, and no c_p for the residual model)
            self._feed_until(float(t))
            self.omega_source.sync(float(t))
            om, sc = self.omega_source.rates()
            self.v_scale = sc
            if self.unc:
                self.mag = float(self.omega_source.turn_mag())
        else:
            om = self._omega(self.v, view)
        if math.isfinite(om):
            self.om = om
        self.t_last = float(t)

    def observe(self, events):
        if len(events) == 0:
            return
        cols = {n: events.cols[n] for n in self.needs}
        ts = events.t_s
        if self.omega_source is not None:
            self._unfed.extend(zip(ts.tolist(), *[cols[n].tolist() for n in self.needs]))
        if self.tick_dt is None:                       # per-event mode
            for i, t in enumerate(ts):
                self._step(float(t), {n: float(cols[n][i]) for n in self.needs}, gap=(float(t) - self.t_last) > 0.3)
            return
        k0 = math.floor(self.t_last / self.tick_dt) + 1
        k1 = math.floor(float(ts[-1]) / self.tick_dt)
        if k1 < k0:                                    # no tick inside this batch: remember the latest event for the next tick
            self._pending = (float(ts[-1]), {n: float(cols[n][-1]) for n in self.needs})
            return
        ticks = self.tick_dt * np.arange(k0, k1 + 1)
        idx = np.searchsorted(ts, ticks, side="right") - 1      # latest event at or before each tick
        pend = getattr(self, "_pending", None)
        for tk, i in zip(ticks, idx):
            tk = float(tk)
            if i >= 0:
                t_ev, view = float(ts[i]), {n: float(cols[n][i]) for n in self.needs}
            elif pend is not None:
                t_ev, view = pend
            else:
                t_ev, view = -math.inf, {n: math.nan for n in self.needs}
            self._step(tk, view, gap=(tk - t_ev) > 0.3)
        self._pending = (float(ts[-1]), {n: float(cols[n][-1]) for n in self.needs})

    def _reweight(self):
        g = self.g
        if self.source == "none":
            vmax = np.array([ROAD_VMAX.get(h, 25) for h in g.highway[self.e]])
            pen = np.exp(-0.5 * (np.maximum(self.v - vmax, 0.0) / 5.0) ** 2)
            self.w *= pen
        else:
            dth = wrap(self.th - g.bearing[self.e])
            self.w *= np.exp(-0.5 * (dth / self.sigma_h) ** 2) + 1e-12
            self.th = self.th - 0.3 * dth              # soft re-anchoring of the dead-reckoned heading to the road
        s = self.w.sum()
        if not np.isfinite(s) or s <= 0:
            self.w[:] = 1.0 / len(self.w)
        else:
            self.w /= s
        ess = 1.0 / np.sum(self.w ** 2)
        if ess < 0.5 * len(self.w):
            idx = self._systematic(self.w)
            self.e, self.off, self.th, self.U = self.e[idx], self.off[idx], self.th[idx], self.U[idx]
            self.w = np.full(len(idx), 1.0 / len(idx))
            self._last_parents = idx

    def _systematic(self, w):
        n = len(w)
        pos = (self.rng.random() + np.arange(n)) / n
        return np.searchsorted(np.cumsum(w), pos).clip(0, n - 1)

    def reconstruct(self, inputs, t_query):
        """Lineage read-out (offline): run the filter causally over the episode, keep every query step's particle edges,
        offsets and resampling parents, then trace the heaviest final particle back."""
        from .. import replay
        self.reset(inputs.init, inputs.static, inputs.map_graph)
        ev = inputs.events
        hist_e, hist_o, hist_p = [], [], []
        i = 0
        for t in np.asarray(t_query, dtype=float):
            j = int(np.searchsorted(ev.t_s, t, side="right"))
            if j > i:
                self.observe(replay._slice(ev, i, j))
                i = j
            self._last_parents = None
            self.predict(float(t))
            hist_e.append(self.e.copy())
            hist_o.append(self.off.copy())
            hist_p.append(self._last_parents)
        k = int(np.argmax(self.w))
        es, os_ = np.empty(len(hist_e), dtype=np.int64), np.empty(len(hist_e))
        for q in range(len(hist_e) - 1, -1, -1):
            es[q], os_[q] = hist_e[q][k], hist_o[q][k]
            if hist_p[q] is not None:
                k = int(hist_p[q][k])
        xy = self.g.points(es, np.minimum(os_, self.g.length[es]))
        return xy, self.g.bearing[es], None, None

    def predict(self, t_ref):
        dt = float(t_ref - self.t_last)
        if dt > 0:
            self._advance(min(dt, 1.0))
            self.t_last = float(t_ref)
        self._reweight()
        g = self.g
        uniq, inv = np.unique(self.e, return_inverse=True)
        wsum = np.bincount(inv, weights=self.w, minlength=len(uniq))
        e_map = int(uniq[int(np.argmax(wsum))])
        on = self.e == e_map
        pts = g.points(np.full(int(on.sum()), e_map, dtype=np.int64), self.off[on])     # one projection for all particles on the MAP edge
        ww = self.w[on] / self.w[on].sum()
        xy = (pts * ww[:, None]).sum(axis=0)
        return Prediction(xy, float(g.bearing[e_map]))
