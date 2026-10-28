"""OPD-II re-implemented from the released tool (Bianchi, Brighente, Conti, Valori, "OPD-II", IEEE EuroS&P 2025,
arXiv 2407.00585; code spritz-group/OPD-II @5fe9363, scripts/OPD_II.py; no licence file, nothing copied).

Released algorithm: 0.1 s windows; mean speed and mean decoded steering angle per window; bicycle model
omega = v tan(angle) / wheelbase with the angle clipped to +-35 decoded units; compass bearing -= omega dt; a geodesic
step of v dt (+ the carried distance correction) along the bearing. Every 30 windows (counter % 30 == 0, starting with
the first window) the dead-reckoned points accumulated since the last successful match go to Valhalla Meili
trace_attributes (search_radius 200 m, shape_match map_snap, costing auto); on success the position jumps to the last
matched point, the first matched point becomes the bearing anchor, the matched points are appended to the output, and
|straight distance(DR first, DR last) - straight distance(matched first, matched last)| is added to the next step. At
>= 30 km/h the bearing is reset every 30 windows (counter % 30 == 2) to the bearing from the anchor to the current point.

Adaptations (reports/icde/methods_repro.md):
- signals: CAN speed (m/s) and the steering-wheel angle with the drive's carParams instead of OBD PID 0x0D and the authors'
  reverse-engineered angle; the released per-car "wheelbase" values (3.85 to 4.9 m for 2.6 to 2.8 m cars) together with
  the 0.01 decoding scale amount to the bicycle model with a realistic steering ratio, so the faithful translation is
  omega = v tan(rad(steer) / SR) / L; the +-35 decoded-unit clip becomes +-350 deg of steering-wheel angle;
  `calib=True` uses the train-fold offset and effective ratio; empty windows hold the last value (the released code would
  produce NaN);
- map: Valhalla tiles built from the benchmark's v2 drive network (same OSM snapshot and drive filter as all map methods);
- angle wrap: the released code maps a negative bearing b to 360 - b (should be 360 + b), i.e. the window that crosses
  north is mirrored by its overshoot (under a degree per crossing at 0.1 s windows); `as_released=True` keeps it,
  `as_released=False` fixes it (sensitivity row);
- read-out: `readout="final"` = the tool's output (each window's matched point, decided at most 3 s later; offline-protocol
  row annotated "delayed online, <= 3 s"); `readout="online"` = at a query time the matched point if the window was
  matched by then, else the current dead-reckoned point (strictly online row). Heading-known protocols only.
"""
from __future__ import annotations

import math

import numpy as np

from .base import OfflineBaseline, OnlineBaseline, Prediction


class OPD2(OnlineBaseline, OfflineBaseline):
    needs = ("speed", "steer_angle")

    def __init__(self, readout="final", as_released=True, calib=False, window_s=0.1, batch=30, clip_sw_deg=350.0,
                 straight_kmh=30.0, search_radius=200.0, matcher_factory=None, wheelbase_m=None, steer_ratio=None):
        """wheelbase_m / steer_ratio override the drive's values; only used to emulate the authors' cars on their sample
        logs (scripts/validate_opd2_samples.py), never in the benchmark runs."""
        self.readout, self.as_released, self.calib = readout, as_released, calib
        self.L_over, self.SR_over = wheelbase_m, steer_ratio
        self.dt, self.batch, self.clip_sw, self.v_straight = window_s, batch, clip_sw_deg, straight_kmh / 3.6
        self.search_radius = search_radius
        self.matcher_factory = matcher_factory           # tests inject a matcher; default: Valhalla cluster of the start
        self.name = "opd2" + ("_online" if readout == "online" else "") + ("_calib" if calib else "") + ("" if as_released else "_wrapfix")

    # ---- engine
    def reset(self, init, static, map_graph=None):
        if init.theta0 is None:
            raise ValueError("OPD-II needs a start bearing (heading-known protocols only)")
        self.fr = init.frame()
        if self.matcher_factory is not None:
            self.match = self.matcher_factory(init)
        else:
            from ..maps.valhalla import MeiliMatcher, cluster_of
            c = cluster_of(init.p0_lat, init.p0_lon)
            if c is None:
                raise ValueError("start outside the Valhalla clusters")
            self.match = MeiliMatcher(c, self.search_radius)
        self.L = float(self.L_over or static.wheelbase_m or 2.7)
        if self.calib and static.steer_ratio_eff_train:
            self.SR, self.off = float(static.steer_ratio_eff_train), float(static.angle_offset_train_deg or 0.0)
        else:
            self.SR, self.off = float(static.steer_ratio or 15.0), 0.0
        if self.SR_over:
            self.SR, self.off = float(self.SR_over), 0.0
        self.t0 = float(init.t0)
        self.x, self.y = init.p0_xy
        self.bearing = (90.0 - math.degrees(init.theta0)) % 360.0     # compass degrees (released input convention)
        self.k = 0                                                     # next window index (window k ends at t0 + (k+1) dt)
        self.counter = 0                                               # released `interpolation_counter`
        self.pend = [(self.x, self.y, self.t0)]                        # DR points waiting for the next match (start included)
        self.anchor = None                                             # first matched point of the last batch
        self.remaining = 0.0
        self.v = 0.0
        self.sw = 0.0
        self.buf_v, self.buf_s = [], []
        self.out_t, self.out_xy, self.out_matched_at = [self.t0], [(self.x, self.y)], [np.inf]   # one entry per DR point
        self.n_match_fail = 0
        self._ev = []                                                  # observed events (t, v, s); _p = first not yet in a closed window
        self._p = 0

    def _window(self, t_end, v_list, s_list):
        if v_list:
            self.v = float(np.mean(v_list))
        if s_list:
            self.sw = float(np.mean(s_list))
        v = self.v
        sw = max(-self.clip_sw, min(self.clip_sw, self.sw - self.off))
        d = v * self.dt + self.remaining
        self.remaining = 0.0
        omega = v * math.tan(math.radians(sw) / self.SR) / self.L
        dth = math.degrees(math.atan2(math.sin(omega * self.dt), math.cos(omega * self.dt)))
        self.bearing = self.bearing - dth
        if v >= self.v_straight and self.counter % self.batch == 2 and self.anchor is not None:
            ax, ay = self.anchor
            self.bearing = math.degrees(math.atan2(self.x - ax, self.y - ay)) % 360.0
        if self.bearing < 0:
            self.bearing = (360.0 - self.bearing) if self.as_released else (360.0 + self.bearing)
        elif self.bearing > 360:
            self.bearing = self.bearing - 360.0
        b = math.radians(self.bearing)
        self.x += d * math.sin(b)
        self.y += d * math.cos(b)
        self.pend.append((self.x, self.y, t_end))
        self.out_t.append(t_end)
        self.out_xy.append((self.x, self.y))
        self.out_matched_at.append(np.inf)
        if self.counter % self.batch == 0:
            self._match(t_end)
        self.counter += 1

    def _match(self, t_now):
        pts = np.array([(p[0], p[1]) for p in self.pend])
        lat, lon = self.fr.to_lla(pts[:, 0], pts[:, 1])
        res = self.match(lat, lon)
        if res is None:
            self.n_match_fail += 1
            return                                   # released behaviour: keep the points, retry with the next batch
        mx, my = self.fr.to_enu(np.array([r[0] for r in res]), np.array([r[1] for r in res]))
        n = len(self.pend)
        i0 = len(self.out_t) - n                      # the pending points are the last n output entries
        for j in range(n):
            self.out_xy[i0 + j] = (float(mx[j]), float(my[j]))
            self.out_matched_at[i0 + j] = t_now
        d_dr = math.hypot(pts[-1, 0] - pts[0, 0], pts[-1, 1] - pts[0, 1])
        d_mm = math.hypot(mx[-1] - mx[0], my[-1] - my[0])
        self.remaining = abs(d_dr - d_mm)
        self.x, self.y = float(mx[-1]), float(my[-1])
        self.anchor = (float(mx[0]), float(my[0]))
        self.pend = []

    def _advance_to(self, t):
        """Close every window that ends at or before t (its events are all observed by then)."""
        ev = self._ev
        while self.t0 + (self.k + 1) * self.dt <= t + 1e-9:
            t_end = self.t0 + (self.k + 1) * self.dt
            j = self._p
            while j < len(ev) and ev[j][0] <= t_end + 1e-9:
                j += 1
            vl = [e[1] for e in ev[self._p:j] if math.isfinite(e[1])]
            sl = [e[2] for e in ev[self._p:j] if math.isfinite(e[2])]
            self._p = j
            self._window(t_end, vl, sl)
            self.k += 1

    # ---- online interface
    def observe(self, events):
        if len(events):
            self._ev.extend(zip(events.t_s.tolist(), events.cols["speed"].tolist(), events.cols["steer_angle"].tolist()))

    def predict(self, t_ref):
        self._advance_to(float(t_ref))
        i = int(np.searchsorted(np.asarray(self.out_t), float(t_ref), side="right")) - 1
        x, y = self.out_xy[i]                         # online: matched only if that happened by now (out_xy is updated in place)
        return Prediction(np.array([x, y]), math.radians(90.0 - self.bearing))

    # ---- offline interface (the tool's final output)
    def reconstruct(self, inputs, t_query):
        self.reset(inputs.init, inputs.static, inputs.map_graph)
        ev = inputs.events
        self.observe(ev)
        t_query = np.asarray(t_query, dtype=float)
        self._advance_to(float(t_query[-1]))
        out_t = np.asarray(self.out_t)
        idx = np.searchsorted(out_t, t_query, side="right") - 1
        xy = np.array([self.out_xy[i] for i in idx])
        return xy, None, None, None
