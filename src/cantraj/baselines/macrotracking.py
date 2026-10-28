"""MacroTracking re-implemented from the released algorithm (Gazdag et al., "Privacy pitfalls of releasing in-vehicle
network data", Vehicular Communications 2023, doi 10.1016/j.vehcom.2022.100565; code CrySyS/MacroTracking @c99063d,
GPL-3.0, not copied).

Released algorithm (src/vehicle/vehicle_model.py, src/map/map.py): whenever more than 1 s has passed since the last
update (checked on every speed / steering message), dead reckoning with the latest speed and the turn radius of the latest
steering position: heading change = distance / radius (changes below 1e-4 deg ignored), then a straight step along the
new heading; the very first update only stores the correction anchor. After more than 20 m of travel since the last
correction the position is pulled towards the nearest road edge (orthogonal projection on the edge's line) and the
heading towards the edge bearing (taken in the travel direction, blended only within 45 deg), both with the weight
0.7 * min(1, d_junction / 150 m), d_junction = distance to the nearest intersection node. Online and causal.

Adaptations (documented in reports/icde/methods_repro.md):
- map: the episode's visible OSM drive graph (osmnx 'drive' semantics) instead of osmnx downloads of 600 m network-distance
  neighbourhoods; nearest edge searched up to 600 m; junctions = nodes whose undirected degree is not 2 (what osmnx
  simplification keeps);
- motion: metric episode frame instead of Web Mercator units and CAN speed without the released car-specific factor 1.1
  (together 0.74 of the CAN distance at the sample's latitude, a fit to the authors' car and city); turn radius from the
  bicycle model with the drive's carParams (wheelbase, steering ratio) instead of the authors' car constants;
  `calib=True` uses the train-fold steering offset and effective ratio (never the episode's GPS);
- angles: the released code compares raw degrees (359 vs 1 deg counts as 358 deg apart, the edge is flipped and the
  heading is not blended). This is the main row (`as_released=True`, default): emulated on the authors' sample trace it
  reproduces their committed trajectory (median 7 m apart), while wrapped comparisons do not (near north they pull the
  heading towards the road being left during a turn); `as_released=False` (wrapped differences) is a sensitivity row;
- read-out: at a query time the last state is propagated with the method's own motion model (no future data; the
  released code emits states only at its update times).
Heading-known protocols only (the method needs a start heading).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial import cKDTree

from .base import OnlineBaseline, Prediction


def _wrapdeg(a):
    return (a + 180.0) % 360.0 - 180.0


def _parts(mg):
    """(underlying DriveGraph, episode->graph point transform, visible-edge mask or None)."""
    if hasattr(mg, "_to_graph") and hasattr(mg, "g"):
        return mg.g, mg._to_graph, mg.visible
    return mg, (lambda x, y: (float(x), float(y))), None


def junction_mask(g):
    """Nodes whose undirected degree is not 2 (intersections and dead ends); cached on the graph."""
    m = getattr(g, "_junction_mask", None)
    if m is None:
        a, b = np.minimum(g.u, g.v), np.maximum(g.u, g.v)
        key = np.unique(a * (len(g.node_xy) + 1) + b)
        a, b = key // (len(g.node_xy) + 1), key % (len(g.node_xy) + 1)
        deg = np.bincount(a, minlength=len(g.node_xy)) + np.bincount(b, minlength=len(g.node_xy))
        m = (deg > 0) & (deg != 2)
        g._junction_mask = m
    return m


class MacroTracking(OnlineBaseline):
    needs = ("speed", "steer_angle")

    def __init__(self, calib=False, as_released=True, update_s=1.0, corr_dist_m=20.0, max_w=0.7, junction_m=150.0,
                 heading_gate_deg=45.0, search_m=600.0, speed_factor=1.0, wheelbase_m=None, steer_ratio=None):
        """speed_factor / wheelbase_m / steer_ratio override the drive's values; only used to emulate the authors' car on
        their sample trace (scripts/validate_macrotracking_sample.py), never in the benchmark runs."""
        self.calib, self.as_released = calib, as_released
        self.speed_factor, self.L_over, self.SR_over = speed_factor, wheelbase_m, steer_ratio
        self.update_s, self.corr_dist, self.max_w, self.junction_m = update_s, corr_dist_m, max_w, junction_m
        self.gate, self.search_m = heading_gate_deg, search_m
        self.name = "macrotracking" + ("_calib" if calib else "") + ("" if as_released else "_wrapfix")

    # ---- state
    def reset(self, init, static, map_graph=None):
        if init.theta0 is None:
            raise ValueError("MacroTracking needs a start heading (heading-known protocols only)")
        if map_graph is None:
            raise ValueError("MacroTracking needs a map graph")
        self.mg = map_graph
        g, self.to_g, vis = _parts(map_graph)
        jm = junction_mask(g)
        if vis is not None:
            nv = np.zeros(len(g.node_xy), dtype=bool)
            nv[g.u[vis]] = True
            nv[g.v[vis]] = True
            jm = jm & nv
        pts = g.node_xy[jm]
        self.jtree = cKDTree(pts) if len(pts) else None
        self.L = float(self.L_over or static.wheelbase_m or 2.7)
        if self.calib and static.steer_ratio_eff_train:
            self.SR, self.off = float(static.steer_ratio_eff_train), float(static.angle_offset_train_deg or 0.0)
        else:
            self.SR, self.off = float(static.steer_ratio or 15.0), 0.0
        if self.SR_over:
            self.SR, self.off = float(self.SR_over), 0.0
        self.x, self.y = init.p0_xy
        self.h = math.degrees(init.theta0) % 360.0       # degrees, counter-clockwise from east (as the released code)
        self.v, self.delta = 0.0, 0.0                     # latest speed (m/s) and road-wheel angle (rad)
        self.t_upd = None                                 # time of the last update (first message initialises it)
        self.t_state = float(init.t0)
        self.anchor = None                                # position of the last map correction
        self.n_corr = 0

    def _dheading(self, dist):
        if abs(self.delta) < 1e-12:
            return 0.0
        dh = math.degrees(dist * math.tan(self.delta) / self.L)   # = dist * 360 / (2 pi R), R = L / tan(delta)
        return 0.0 if abs(dh) < 1e-4 else dh

    def _update(self, t):
        if self.v == 0:
            return
        if self.anchor is None:                           # released behaviour: the first update only stores the anchor
            self.anchor = (self.x, self.y)
            self.t_state = t
            return
        d = self.v * (t - self.t_upd)
        self.h = (self.h + self._dheading(d)) % 360.0
        self.x += d * math.cos(math.radians(self.h))
        self.y += d * math.sin(math.radians(self.h))
        self.t_state = t
        if math.hypot(self.x - self.anchor[0], self.y - self.anchor[1]) > self.corr_dist:
            self._correct()
            self.anchor = (self.x, self.y)

    def _correct(self):
        mg = self.mg
        es = []
        r = 50.0
        while not es and r <= self.search_m:
            es = mg.nearest_edges(self.x, self.y, r)
            r *= 2.0
        if not es:
            return
        e = int(es[0])
        a, b = mg.points(np.array([e, e], dtype=np.int64), np.array([0.0, float(mg.length[e])]))
        eb = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0
        diff = abs(self.h - eb) if self.as_released else abs(_wrapdeg(self.h - eb))
        if diff > 90.0:
            eb = (eb + 180.0) % 360.0
        nx, ny = b[0] - a[0], b[1] - a[1]
        n2 = nx * nx + ny * ny
        if n2 <= 0:
            return
        u = ((self.x - a[0]) * nx + (self.y - a[1]) * ny) / n2          # orthogonal projection on the edge's (infinite) line
        px, py = a[0] + u * nx, a[1] + u * ny
        w = self.max_w
        if self.jtree is not None:
            dj, _ = self.jtree.query(self.to_g(self.x, self.y))
            if dj <= self.junction_m:
                w = (dj / self.junction_m) * self.max_w
        self.x = (1.0 - w) * self.x + w * px
        self.y = (1.0 - w) * self.y + w * py
        if self.as_released:
            if abs(self.h - eb) < self.gate:
                self.h = (1.0 - w) * self.h + w * eb
        else:
            dh = _wrapdeg(eb - self.h)
            if abs(dh) < self.gate:
                self.h = (self.h + w * dh) % 360.0
        self.n_corr += 1

    def observe(self, events):
        if len(events) == 0:
            return
        ts = events.t_s
        v, s = events.cols["speed"], events.cols["steer_angle"]
        for i in range(len(ts)):
            t = float(ts[i])
            if self.t_upd is None:
                self.t_upd = t
            if math.isfinite(s[i]):
                self.delta = math.radians(float(s[i]) - self.off) / self.SR
            if math.isfinite(v[i]):
                self.v = float(v[i]) * self.speed_factor
            if t - self.t_upd > self.update_s:
                self._update(t)
                self.t_upd = t

    def predict(self, t_ref):
        dt = max(float(t_ref) - self.t_state, 0.0) if self.anchor is not None else 0.0
        if dt <= 0 or self.v == 0:
            return Prediction(np.array([self.x, self.y]), math.radians(self.h))
        d = self.v * dt
        h = self.h + self._dheading(d)
        return Prediction(np.array([self.x + d * math.cos(math.radians(h)), self.y + d * math.sin(math.radians(h))]), math.radians(h))
