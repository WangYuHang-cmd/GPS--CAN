"""Valhalla Meili map matching for the OPD-II re-implementation: one in-process pyvalhalla Actor (the engine behind
the Valhalla HTTP service the authors run in Docker) per cluster tile set built by scripts/build_valhalla_v2.py from
the benchmark's own v2 drive network; an episode uses the cluster that contains its start tile."""
from __future__ import annotations

import json
import math
import os
import sys

from .. import paths

VLIB = os.path.join(paths.BENCH, "pylibs", "valhalla")
ROOT = os.path.join(paths.BENCH, "maps", "valhalla_v2")
CLUSTERS = os.path.join(paths.MANIFESTS, "valhalla_v2_clusters.json")
_actors = {}
_tile_cluster = None


def cluster_of(lat, lon):
    global _tile_cluster
    if _tile_cluster is None:
        _tile_cluster = json.load(open(CLUSTERS))["tile_cluster"]
    return _tile_cluster.get(f"t{int(math.floor(lat / 0.25))}_{int(math.floor(lon / 0.25))}")


def actor(cluster):
    if cluster not in _actors:
        if VLIB not in sys.path:
            sys.path.insert(0, VLIB)
        from valhalla import Actor
        _actors[cluster] = Actor(os.path.join(ROOT, cluster, "valhalla.json"))
    return _actors[cluster]


class MeiliMatcher:
    """Callable(lats, lons) -> list of (lat, lon) matched points (Meili's matched_points, one per input point; unmatched
    points keep their input position as in the released tool), or None when Valhalla returns an error (the released
    tool then keeps its points and retries with the next batch)."""

    def __init__(self, cluster, search_radius=200.0, config=None):
        """config: an explicit valhalla.json (author-sample checks); default: the cluster's tile set."""
        self.cluster, self.search_radius = cluster, search_radius
        self.n_calls = self.n_fail = 0
        if config is not None:
            if VLIB not in sys.path:
                sys.path.insert(0, VLIB)
            from valhalla import Actor
            self.cluster = f"cfg:{config}"
            _actors.setdefault(self.cluster, Actor(config))

    def __call__(self, lats, lons):
        self.n_calls += 1
        req = {"shape": [{"lon": float(lo), "lat": float(la), "time": i} for i, (la, lo) in enumerate(zip(lats, lons))],
               "search_radius": self.search_radius, "shape_match": "map_snap", "costing": "auto", "format": "osrm"}
        try:
            res = json.loads(actor(self.cluster).trace_attributes(json.dumps(req)))
        except Exception:  # noqa: BLE001  (Valhalla raises on "no suitable edges" and similar)
            self.n_fail += 1
            return None
        mp = res.get("matched_points")
        if not mp or len(mp) != len(lats):
            self.n_fail += 1
            return None
        return [(float(p["lat"]), float(p["lon"])) for p in mp]
