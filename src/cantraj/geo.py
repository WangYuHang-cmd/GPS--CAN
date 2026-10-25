"""Coordinate helpers: WGS-84 <-> local ENU (azimuthal equidistant about the episode origin), bearings, headings.

Conventions (Plan.md 2.2): ENU x east, y north; theta measured counter-clockwise from east; a GPS compass bearing b
(clockwise from north, degrees) maps to theta = pi/2 - radians(b).
"""
from __future__ import annotations

import math

import numpy as np
from pyproj import CRS, Transformer

WGS84 = CRS.from_epsg(4326)
EARTH_R_M = 6371000.0


class LocalFrame:
    """Local ENU frame centred at (lat0, lon0) using an azimuthal equidistant projection (metre-exact at the origin,
    sub-metre within ~50 km)."""

    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = float(lat0), float(lon0)
        aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={self.lat0} +lon_0={self.lon0} +datum=WGS84 +units=m +no_defs")
        self._fwd = Transformer.from_crs(WGS84, aeqd, always_xy=True)
        self._inv = Transformer.from_crs(aeqd, WGS84, always_xy=True)

    def to_enu(self, lat, lon):
        x, y = self._fwd.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def to_lla(self, x, y):
        lon, lat = self._inv.transform(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
        return np.asarray(lat, dtype=float), np.asarray(lon, dtype=float)


def bearing_to_theta(bearing_deg):
    """Compass bearing (deg, clockwise from north) -> math heading (rad, CCW from east), wrapped to (-pi, pi]."""
    return wrap_angle(math.pi / 2.0 - np.radians(np.asarray(bearing_deg, dtype=float)))


def theta_to_bearing(theta_rad):
    b = np.degrees(math.pi / 2.0 - np.asarray(theta_rad, dtype=float))
    return np.mod(b, 360.0)


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    return -((-a + math.pi) % (2.0 * math.pi) - math.pi)


def haversine_m(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(lambda v: np.radians(np.asarray(v, dtype=float)), (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def course_rate(bearing_deg, t_s, max_dt=0.3):
    """Left-positive heading rate (rad/s) from consecutive GPS courses; NaN where dt > max_dt or dt <= 0."""
    th = np.unwrap(bearing_to_theta(bearing_deg))
    dt = np.diff(np.asarray(t_s, dtype=float))
    r = np.full(len(dt), np.nan)
    ok = (dt > 0) & (dt <= max_dt)
    r[ok] = np.diff(th)[ok] / dt[ok]
    return r
