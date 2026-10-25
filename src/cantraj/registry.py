"""Signal registry, groups, protocols and window configs (configs/*.yaml) plus the canonical <-> column maps."""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs")

# canonical signal -> column in the native event tables (carstate.parquet unless noted) and in the released grid table
COLUMN_MAP = {
    "speed":          ("vEgo", "vEgo"),
    "speed_cluster":  ("vEgoCluster", "vEgoCluster"),
    "accel_long":     ("aEgo", "aEgo"),
    "steer_angle":    ("steeringAngleDeg", "steeringAngleDeg"),
    "steer_rate":     ("steeringRateDeg", "steeringRateDeg"),
    "gas":            ("gas", "gas"),
    "brake":          ("brake", "brake"),
    "gas_pressed":    ("gasPressed", "gasPressed"),
    "brake_pressed":  ("brakePressed", "brakePressed"),
    "steer_pressed":  ("steeringPressed", "steeringPressed"),
    "blinker_left":   ("leftBlinker", "leftBlinker"),
    "blinker_right":  ("rightBlinker", "rightBlinker"),
    "cruise_on":      ("cruiseEnabled", "cruiseState_enabled"),
    "yaw_rate_can":   ("yawRate", "yawRate"),
    "wheel_speed_fl": ("ws_fl", None),
    "wheel_speed_fr": ("ws_fr", None),
    "wheel_speed_rl": ("ws_rl", None),
    "wheel_speed_rr": ("ws_rr", None),
    "curvature_loc":  ("controls:curvature", "actual_curvature"),
    "yaw_rate_loc":   (None, "yaw_rate"),
    "imu_gyro_x": ("imu:gyro:v0", None), "imu_gyro_y": ("imu:gyro:v1", None), "imu_gyro_z": ("imu:gyro:v2", None),
    "imu_acc_x": ("imu:accel:v0", None), "imu_acc_y": ("imu:accel:v1", None), "imu_acc_z": ("imu:accel:v2", None),
}


class ForbiddenSignalError(KeyError):
    pass


def _load(name):
    with open(os.path.join(CONFIG_DIR, name)) as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=None)
def registry():
    return _load("signal_registry.yaml")


@lru_cache(maxsize=None)
def groups():
    return _load("signal_groups.yaml")


@lru_cache(maxsize=None)
def protocols():
    return _load("protocols.yaml")


@lru_cache(maxsize=None)
def windows():
    return _load("windows.yaml")


def forbidden_names():
    r = registry()
    return set(r["forbidden_always"]) | set(r.get("meta_only", []))


def signal_class(name):
    return registry()["signals"][name]["class"]


def group_spec(gid):
    """Resolved group: requires, optional, static (with `extends` expanded)."""
    g = dict(groups()["groups"][gid])
    if "extends" in g:
        base = group_spec(g["extends"])
        g["requires"] = base["requires"]
        g["optional"] = list(base["optional"]) + list(g.get("optional_extra", []))
        g["static"] = list(base["static"]) + list(g.get("static_extra", []))
    g.setdefault("requires", [])
    g.setdefault("optional", [])
    g.setdefault("static", [])
    return g


def group_signals(gid):
    g = group_spec(gid)
    return list(g["requires"]) + [s for s in g["optional"] if s not in g["requires"]]


def group_static(gid):
    return list(group_spec(gid)["static"])


def check_whitelist(names, gid):
    """Raise ForbiddenSignalError if any requested name is outside the group's signals or is forbidden."""
    allowed = set(group_signals(gid))
    bad = [n for n in names if n not in allowed or n in forbidden_names()]
    if bad:
        raise ForbiddenSignalError(f"signals not allowed for {gid}: {bad}")


def protocol_spec(pid):
    p = protocols()
    spec = dict(p["protocols"][pid])
    spec["id"] = pid
    spec["input_layer"] = spec.get("input_layer", p.get("input_layer_default", "events"))
    spec["reference_view"] = spec.get("reference_view", p.get("reference_view", "view_1hz"))
    spec["map_radius_rule"] = p.get("map_radius_rule", {"base_km": 5.0, "speed_mps": 40.0})
    return spec


def map_radius_m(spec, tau_s):
    r = spec["map_radius_rule"]
    return 1000.0 * float(r["base_km"]) + float(r["speed_mps"]) * max(float(tau_s), 0.0)
