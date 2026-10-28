#!/usr/bin/env python3
"""A3 real-data check: for sampled test episodes, the per-drive union graph seen through EpisodeGraphView(visible_tiles)
must expose exactly the directed edges (tail node, head node, way) of a graph built from the episode's own tiles only,
and its bearings must equal those of an episode-frame graph to within 0.05 deg. Writes reports/icde/a3_edge_check.csv."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from cantraj import io, paths, registry  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
from cantraj.geo import LocalFrame  # noqa: E402
from cantraj.maps.graph import EpisodeGraphView, build_graph_tiles  # noqa: E402
from cantraj.maps.osm import tiles_in_radius  # noqa: E402

TILES = os.path.join(paths.BENCH, "maps", "tiles_v2")
CAP = 60000.0


def keyset(g, mask=None):
    idx = np.flatnonzero(mask) if mask is not None else np.arange(g.n_edges())
    return set(zip(g.node_id[g.u[idx]].tolist(), g.node_id[g.v[idx]].tolist(), g.way_id[idx].tolist())), idx


def main():
    drives = pd.read_csv(os.path.join(paths.MANIFESTS, "icde_a3_drives.csv")).head(12)
    spec = registry.protocol_spec("O-M-Hp")
    rows = []
    for r in drives.itertuples(index=False):
        g_man = io.load_manifest().set_index(["driver", "drive"]).loc[(r.driver, r.drive)]
        dd = DriveData(r.driver, r.drive, delta_s=io.time_compensation(g_man["gps_sources"], g_man["gps_rate_hz"]), light=True)
        for win in ("w05", "w15"):
            eps = [e for e in dd.episodes(win) if e.meta.get("eligible_main")]
            if not eps:
                continue
            anchors = np.array([[e.init_full["p0_lat"], e.init_full["p0_lon"]] for e in eps])
            fr = LocalFrame(float(anchors[:, 0].mean()), float(anchors[:, 1].mean()))
            tiles = set()
            for e in eps:
                tiles.update(tiles_in_radius(e.init_full["p0_lat"], e.init_full["p0_lon"], min(registry.map_radius_m(spec, e.t_end - e.t0), CAP)))
            g_union = build_graph_tiles(TILES, tiles, fr)
            for e in eps[:: max(1, len(eps) // 3)][:3]:
                need = tiles_in_radius(e.init_full["p0_lat"], e.init_full["p0_lon"], min(registry.map_radius_m(spec, e.t_end - e.t0), CAP))
                view = EpisodeGraphView(g_union, e.frame, visible_tiles=need)
                g_own = build_graph_tiles(TILES, need, e.frame)
                ku, iu = keyset(g_union, view.visible)
                ko, io_ = keyset(g_own)
                # bearings of the common edges: view (rotated drive frame) vs own graph (episode frame)
                common = ku & ko
                pos_u = {k: i for k, i in zip(zip(g_union.node_id[g_union.u[iu]].tolist(), g_union.node_id[g_union.v[iu]].tolist(), g_union.way_id[iu].tolist()), iu)}
                pos_o = {k: i for k, i in zip(zip(g_own.node_id[g_own.u[io_]].tolist(), g_own.node_id[g_own.v[io_]].tolist(), g_own.way_id[io_].tolist()), io_)}
                ks = list(common)[:20000]
                db = np.array([abs(((view.bearing[pos_u[k]] - g_own.bearing[pos_o[k]] + np.pi) % (2 * np.pi)) - np.pi) for k in ks])
                rows.append(dict(episode_id=e.episode_id, win=win, union_edges=g_union.n_edges(), visible_edges=len(ku), own_edges=len(ko),
                                 only_in_view=len(ku - ko), only_in_own=len(ko - ku), bearing_max_diff_deg=float(np.degrees(db.max())) if len(db) else np.nan,
                                 frame_offset_km=float(np.hypot(*fr.to_enu(e.init_full["p0_lat"], e.init_full["p0_lon"])) / 1000.0)))
                print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(paths.GPS_DIR, "reports", "icde", "a3_edge_check.csv"), index=False)
    print(out[["union_edges", "visible_edges", "own_edges", "only_in_view", "only_in_own", "bearing_max_diff_deg"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
