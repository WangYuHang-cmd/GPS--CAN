#!/usr/bin/env python3
"""Two figures on real held-out windows, drawn in the episode frame (metres from the given start; no coordinates, no
place names, no identifiers beyond the pseudonymous episode ids in the data file).

fig_examples_w05: six five-minute windows, each with the forward camera frame at mid-window above the GNSS reference and
    the recovered paths of the offline beam and the online filter (both speed and steering, start and heading known).
    The windows were chosen among held-out five-minute windows that both methods recover (route recovered, beam ADE
    below 12 m, path longer than 2.5 km): six drivers, road densities from rural to dense urban, 0 to 14 turns. They
    show accurate recoveries, not typical ones (the medians are in the main table).
fig_method: the pipeline on one of those windows: camera frame and CAN signals, start and visible map, the beam's
    candidate paths and the filter's path, the reference and the error over time.

Frames come from the released 1-minute forward-video segments (segment floor(t / 60), checked against the reference's
segment column); without CANTRAJ_VIDEOS the frame panels stay empty. Outputs paper_icde/figures/{fig_examples_w05,
fig_method}.{pdf,png} and reports/icde/figures_data/fig_examples{,_meta}.csv.

    CANTRAJ_VIDEOS=/path/to/videos python scripts/icde_fig_examples.py
"""
import argparse
import glob
import math
import os
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cantraj import io, paths  # noqa: E402
from cantraj.episodes import DriveData  # noqa: E402
from cantraj.maps.osm import DRIVE_HIGHWAYS, tile_of  # noqa: E402
import icde_figures as F  # noqa: E402
from score import read_group_for_drives  # noqa: E402

VIDEOS = os.environ.get("CANTRAJ_VIDEOS")          # <driver>/<drive>/segN.ts, the released forward video
TILES = os.path.join(paths.BENCH, "maps", "tiles_v2")
ROAD, REF = "#d3d2cd", F.INK
BEAM = ("map_beam_steer__G3", "F-M-Hp")
PF = ("map_pf_steer__G3", "O-M-Hp")
# (episode, short label): chosen as described in the module docstring; the label is the road density class (nodes per
# km2 around the start: rural < 30, suburban < 200, urban otherwise) and the path length
EXAMPLES = [("driver_266/drive_002/w05/k001", "suburb, 5.1 km"),
            ("driver_099/drive_007/w05/k114", "rural, 9.3 km"),
            ("driver_238/drive_001/w05/k001", "suburb, 4.1 km"),
            ("driver_104/drive_002/w05/k024", "urban, 2.9 km"),
            ("driver_250/drive_014/w05/k002", "rural, 4.3 km"),
            ("driver_096/drive_001/w05/k000", "urban, 4.8 km")]
METHOD_EXAMPLE = "driver_104/drive_002/w05/k024"


def load_episode(gman, eid):
    driver, drive = eid.split("/")[:2]
    r = gman.loc[(driver, drive)]
    dd = DriveData(driver, drive, delta_s=io.time_compensation(r["gps_sources"], r["gps_rate_hz"]), input_layer="events", light=True)
    for e in dd.episodes("w05"):
        if e.episode_id == eid:
            return e
    raise KeyError(eid)


def xy_at(ep, p):
    """Predicted position at the reference seconds (NaN where the method has none)."""
    t = np.round(ep.ref["time_s"].to_numpy(dtype=float), 6)
    idx = pd.Index(np.round(p["t"].to_numpy(dtype=float), 6))
    pos = idx.get_indexer(t)
    xy = np.full((len(t), 2), np.nan)
    hit = pos >= 0
    xy[hit] = p[["pred_x_m", "pred_y_m"]].to_numpy(dtype=float)[pos[hit]]
    return xy


def roads(frame, x0, x1, y0, y1):
    """Drivable ways of the published tiles that touch the box, as polylines in the episode frame."""
    lat, lon = frame.to_lla(np.array([x0, x1, x0, x1]), np.array([y0, y0, y1, y1]))
    i0, j0 = tile_of(lat.min(), lon.min())
    i1, j1 = tile_of(lat.max(), lon.max())
    out = []
    for i in range(i0, i1 + 1):
        for j in range(j0, j1 + 1):
            d = os.path.join(TILES, f"t{i}_{j}")
            if not os.path.isdir(d):
                continue
            ways = pd.read_parquet(os.path.join(d, "ways.parquet"), columns=["way_id", "highway"])
            ways = ways[ways.highway.isin(DRIVE_HIGHWAYS)]
            wn = pd.read_parquet(os.path.join(d, "way_nodes.parquet"))
            wn = wn[wn.way_id.isin(ways.way_id)].merge(pd.read_parquet(os.path.join(d, "nodes.parquet")), on="node_id")
            if not len(wn):
                continue
            wn = wn.sort_values(["way_id", "seq"])
            wn["x"], wn["y"] = frame.to_enu(wn.lat.to_numpy(), wn.lon.to_numpy())
            inside = (wn.x.between(x0, x1) & wn.y.between(y0, y1)).groupby(wn.way_id).any()
            for wid, g in wn[wn.way_id.isin(inside[inside].index)].groupby("way_id", sort=False):
                out.append(np.column_stack([g.x.to_numpy(), g.y.to_numpy()]))
    return out


def frame_png(driver, drive, t, out_dir):
    """Forward camera frame at drive time t from the released 1-minute segments; None when unavailable."""
    if not VIDEOS:
        return None
    seg = int(math.floor(t / 60.0))
    f = os.path.join(VIDEOS, driver, drive, f"seg{seg}.ts")
    if not os.path.exists(f):
        print(f"  no video segment for {driver}/{drive} at {t:.0f} s")
        return None
    g = pd.read_csv(paths.gps_path(driver, drive), usecols=["time_s", "seg"])
    near = g.iloc[(g.time_s - t).abs().argmin()]
    if int(near.seg) != seg:                          # the segment numbering of this drive is not the time axis / 60
        print(f"  segment mismatch for {driver}/{drive}: reference seg {int(near.seg)} vs {seg}; frame skipped")
        return None
    out = os.path.join(out_dir, f"{driver}_{drive}_{int(t)}.png")
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{t - 60.0 * seg:.3f}", "-i", f, "-frames:v", "1", out], check=True)
    return out


def box(xs, ys, pad=0.10):
    x0, x1, y0, y1 = np.nanmin(xs), np.nanmax(xs), np.nanmin(ys), np.nanmax(ys)
    w = max(x1 - x0, y1 - y0, 400.0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = w * (1 + 2 * pad) / 2
    return cx - h, cx + h, cy - h, cy + h


def draw_roads(ax, segs):
    if segs:
        ax.add_collection(LineCollection(segs, colors=ROAD, linewidths=0.6, zorder=1))


def empty_corners(x, y, x0, x1, y0, y1):
    """Corners of the box ordered by how few path points fall in their quarter (emptiest first)."""
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    quads = {"tl": ((x < cx) & (y > cy)).sum(), "tr": ((x > cx) & (y > cy)).sum(), "bl": ((x < cx) & (y < cy)).sum(), "br": ((x > cx) & (y < cy)).sum()}
    return sorted(quads, key=quads.get)


def scale_bar(ax, x0, x1, y0, y1, corner="br"):
    span = x1 - x0
    L = 1000.0 if span > 3200 else (500.0 if span > 2000 else 200.0)
    xa = x1 - 0.06 * span - L if corner.endswith("r") else x0 + 0.06 * span
    ya = y0 + 0.07 * span if corner.startswith("b") else y1 - 0.10 * span
    ax.plot([xa, xa + L], [ya, ya], color=F.INK, lw=1.6, solid_capstyle="butt", zorder=6)
    ax.text(xa + L / 2, ya + 0.025 * span, f"{L / 1000:g} km" if L >= 1000 else f"{L:.0f} m", ha="center", va="bottom", fontsize=9.5, color=F.INK)


def corner_text(ax, corner, text):
    pos = {"tl": (0.04, 0.97, "left", "top"), "tr": (0.96, 0.97, "right", "top"), "bl": (0.04, 0.03, "left", "bottom"), "br": (0.96, 0.03, "right", "bottom")}[corner]
    ax.text(pos[0], pos[1], text, ha=pos[2], va=pos[3], fontsize=9.5, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9), zorder=8)


def fig_examples(items, out_dir):
    F.rc()
    fig = plt.figure(figsize=(7.2, 2.5))
    gs = fig.add_gridspec(2, len(items), height_ratios=[0.72, 1.4], hspace=0.02, wspace=0.05, left=0.005, right=0.995, top=0.92, bottom=0.09)
    rows, meta = [], []
    for k, it in enumerate(items):
        ep, label, img, segs, beam, pf, ade_b, ade_p, info = it["ep"], it["label"], it["frame"], it["roads"], it["beam"], it["pf"], it["ade_beam"], it["ade_pf"], it["info"]
        axf = fig.add_subplot(gs[0, k])
        if img is not None:
            axf.imshow(mpimg.imread(img))
        else:
            axf.text(0.5, 0.5, "camera frame", ha="center", va="center", fontsize=8, color=F.INK2, transform=axf.transAxes)
        axf.set_title(label, fontsize=10, pad=2.5)
        axf.axis("off")
        ax = fig.add_subplot(gs[1, k])
        x, y = ep.ref["x"].to_numpy(dtype=float), ep.ref["y"].to_numpy(dtype=float)
        x0, x1, y0, y1 = it["box"]
        draw_roads(ax, segs)
        ax.plot(x, y, color=REF, lw=2.6, solid_capstyle="round", zorder=3)
        ax.plot(pf[:, 0], pf[:, 1], color=F.STEER, lw=1.3, ls="-", zorder=4)
        ax.plot(beam[:, 0], beam[:, 1], color=F.SPEED, lw=1.3, ls="--", zorder=5)
        ax.plot([0], [0], marker="o", ms=5.5, mfc="white", mec=F.INK, mew=1.4, ls="none", zorder=7)
        corners = empty_corners(x, y, x0, x1, y0, y1)
        corner_text(ax, corners[0], f"beam {ade_b:.1f} m\nfilter {ade_p:.1f} m")
        scale_bar(ax, x0, x1, y0, y1, next((c for c in corners[1:] if c[0] != corners[0][0]), corners[1]))
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
        t_rel = ep.ref["time_s"].to_numpy(dtype=float) - ep.t0
        for name, arr in (("reference", np.column_stack([x, y])), ("beam", beam), ("filter", pf)):
            rows.append(pd.DataFrame({"example": k + 1, "series": name, "t_rel_s": np.round(t_rel, 3), "x_m": np.round(arr[:, 0], 2), "y_m": np.round(arr[:, 1], 2)}))
        meta.append(dict(example=k + 1, episode_id=ep.episode_id, label=label, frame_t_rel_s=150.0, ade_beam_m=round(ade_b, 2), ade_filter_m=round(ade_p, 2), **info))
    handles = [Line2D([], [], color=REF, lw=2.6, label="GNSS reference"),
               Line2D([], [], color=F.SPEED, lw=1.3, ls="--", label="offline beam"),
               Line2D([], [], color=F.STEER, lw=1.3, label="online filter"),
               Line2D([], [], marker="o", ms=5.5, mfc="white", mec=F.INK, mew=1.4, ls="none", label="given start")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10.5, bbox_to_anchor=(0.5, -0.015), handlelength=2.2, columnspacing=2.0)
    F.save(fig, "fig_examples_w05")
    os.makedirs(F.DATA, exist_ok=True)
    data = pd.concat(rows, ignore_index=True)
    data.to_csv(os.path.join(F.DATA, "fig_examples.csv"), index=False)
    pd.DataFrame(meta).to_csv(os.path.join(F.DATA, "fig_examples_meta.csv"), index=False)
    for name in ("fig_examples.csv", "fig_examples_meta.csv"):
        txt = open(os.path.join(F.DATA, name)).read()
        assert not F.FORBIDDEN.search(txt), name


def box_wide(x, y, aspect=1.45, pad=0.10):
    """Box around the path with a fixed width-to-height ratio (wide panels for the pipeline figure)."""
    x0, x1, y0, y1 = np.nanmin(x), np.nanmax(x), np.nanmin(y), np.nanmax(y)
    w = max((x1 - x0) * (1 + 2 * pad), (y1 - y0) * (1 + 2 * pad) * aspect, 500.0)
    h = w / aspect
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2


def fig_method(it, cand, out_dir):
    """Pipeline on one window: camera frame and CAN signals; start, heading and roads; candidate paths; reference."""
    F.rc()
    ep, segs, beam, pf = it["ep"], it["roads"], it["beam"], it["pf"]
    x, y = ep.ref["x"].to_numpy(dtype=float), ep.ref["y"].to_numpy(dtype=float)
    x0, x1, y0, y1 = box_wide(x, y)
    segs = roads(ep.frame, x0, x1, y0, y1)
    fig = plt.figure(figsize=(7.2, 2.15))
    outer = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.2, 1.2, 1.2], wspace=0.10, left=0.065, right=0.995, top=0.90, bottom=0.25)
    ga = outer[0].subgridspec(3, 1, height_ratios=[1.15, 0.7, 0.7], hspace=0.7)
    axf = fig.add_subplot(ga[0])
    if it["frame"] is not None:
        axf.imshow(mpimg.imread(it["frame"]))
    axf.axis("off")
    axf.set_title("1. CAN signals", fontsize=11, pad=3)
    cs = pd.read_parquet(os.path.join(paths.native_dir(ep.driver, ep.drive), "carstate.parquet"), columns=["t_s", "vEgo", "steeringAngleDeg"])
    cs = cs[(cs.t_s >= ep.t0) & (cs.t_s <= ep.t_end)]
    for g, col, lab, color in ((ga[1], "vEgo", "speed (m/s)", F.SPEED), (ga[2], "steeringAngleDeg", "steering (deg)", F.STEER)):
        pass
    for g, col, lab, color in ((ga[1], "vEgo", "speed, m/s", F.SPEED), (ga[2], "steeringAngleDeg", "steer, deg", F.STEER)):
        a = fig.add_subplot(g)
        a.plot(cs.t_s - ep.t0, cs[col], color=color, lw=0.9)
        a.set_xlim(0, 300); a.set_xticks([0, 150, 300]); a.tick_params(labelsize=8, length=2, pad=1)
        a.set_ylabel(lab, fontsize=8.5, labelpad=2); a.grid(False)
        if col == "vEgo":
            a.set_xticklabels([])
        else:
            a.set_xlabel("time (s)", fontsize=8.5, labelpad=1)
    axb = fig.add_subplot(outer[1]); draw_roads(axb, segs)
    th = ep.init_full.get("theta0")
    if th is not None and np.isfinite(th):
        L = 0.10 * (x1 - x0)
        axb.annotate("", xy=(L * math.cos(th), L * math.sin(th)), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=F.INK, lw=1.8), zorder=6)
    axb.plot([0], [0], marker="o", ms=6.5, mfc="white", mec=F.INK, mew=1.6, ls="none", zorder=7)
    axb.set_title("2. start, heading, map", fontsize=11, pad=3)
    axc = fig.add_subplot(outer[2]); draw_roads(axc, segs)
    for cid, p in cand.items():
        if cid != 0:
            axc.plot(p[:, 0], p[:, 1], color=F.SPEED, lw=0.9, alpha=0.45, zorder=3)
    axc.plot(pf[:, 0], pf[:, 1], color=F.STEER, lw=1.4, zorder=4)
    axc.plot(beam[:, 0], beam[:, 1], color=F.SPEED, lw=1.4, ls="--", zorder=5)
    axc.plot([0], [0], marker="o", ms=6.5, mfc="white", mec=F.INK, mew=1.6, ls="none", zorder=7)
    axc.set_title("3. candidate paths", fontsize=11, pad=3)
    axd = fig.add_subplot(outer[3]); draw_roads(axd, segs)
    axd.plot(x, y, color=REF, lw=2.6, zorder=3)
    axd.plot(pf[:, 0], pf[:, 1], color=F.STEER, lw=1.3, zorder=4)
    axd.plot(beam[:, 0], beam[:, 1], color=F.SPEED, lw=1.3, ls="--", zorder=5)
    axd.set_title("4. score vs reference", fontsize=11, pad=3)
    corners = empty_corners(x, y, x0, x1, y0, y1)
    corner_text(axd, corners[0], f"beam {it['ade_beam']:.1f} m\nfilter {it['ade_pf']:.1f} m")
    scale_bar(axd, x0, x1, y0, y1, next((c for c in corners[1:] if c[0] != corners[0][0]), corners[1]))
    for a in (axb, axc, axd):
        a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect("equal"); a.axis("off")
    handles = [Line2D([], [], color=REF, lw=2.6, label="GNSS reference"),
               Line2D([], [], color=F.SPEED, lw=1.3, ls="--", label="beam, best path"),
               Line2D([], [], color=F.SPEED, lw=0.9, alpha=0.45, label="beam, other candidates"),
               Line2D([], [], color=F.STEER, lw=1.3, label="online filter"),
               Line2D([], [], marker="o", ms=6.5, mfc="white", mec=F.INK, mew=1.6, ls="none", label="given start")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10.5, bbox_to_anchor=(0.5, -0.015), handlelength=2.2, columnspacing=1.4)
    F.save(fig, "fig_method")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="icde_v11")
    ap.add_argument("--figs", nargs="+", default=["examples", "method"])
    args = ap.parse_args()
    gman = io.load_manifest().set_index(["driver", "drive"])
    ids = [e for e, _ in EXAMPLES]
    if METHOD_EXAMPLE not in ids:
        ids.append(METHOD_EXAMPLE)
    keys = sorted({"/".join(e.split("/")[:2]) for e in ids})
    scores = pd.read_parquet(os.path.join(paths.SCORES, args.run, "episodes_all.parquet"), columns=["episode_id", "protocol", "method", "ade", "route_ok"])
    scores = scores[scores.episode_id.isin(ids)]
    man = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_manifest_v1.parquet"), columns=["episode_id", "fold", "path_len_m"]).set_index("episode_id")
    dens = pd.read_parquet(os.path.join(paths.MANIFESTS, "episode_density_v11.parquet")).set_index("episode_id")
    turns = pd.read_csv(os.path.join(F.DATA, "..", "tables", "area_check_episodes.csv")).set_index("episode_id") if os.path.exists(os.path.join(F.DATA, "..", "tables", "area_check_episodes.csv")) else None
    preds = {}
    for stem, proto in (BEAM, PF):
        flist = sorted(glob.glob(os.path.join(paths.PREDS, args.run, proto, f"{stem}.part*.parquet")))
        preds[(stem, proto)] = read_group_for_drives(flist, set(keys))
        print(f"{stem} {proto}: {len(preds[(stem, proto)])} prediction rows for {len(keys)} drives")
    tmp = tempfile.mkdtemp(prefix="icde_frames_", dir=os.environ.get("TMPDIR"))
    items, cand = {}, {}
    for eid, label in EXAMPLES + ([(METHOD_EXAMPLE, "")] if METHOD_EXAMPLE not in ids[:len(EXAMPLES)] else []):
        if eid in items:
            continue
        ep = load_episode(gman, eid)
        x, y = ep.ref["x"].to_numpy(dtype=float), ep.ref["y"].to_numpy(dtype=float)
        bx = box(x, y)
        pb = preds[BEAM]; pb = pb[pb.episode_id.astype(str) == eid]
        pp = preds[PF]; pp = pp[pp.episode_id.astype(str) == eid]
        beam = xy_at(ep, pb[pb.candidate_id == 0])
        pf = xy_at(ep, pp[pp.candidate_id == 0])
        if eid == METHOD_EXAMPLE:
            cand = {int(c): xy_at(ep, g) for c, g in pb.groupby("candidate_id")}
        sc = scores[scores.episode_id == eid]
        ade_b = float(sc[(sc.method == BEAM[0]) & (sc.protocol == BEAM[1])].ade.iloc[0])
        ade_p = float(sc[(sc.method == PF[0]) & (sc.protocol == PF[1])].ade.iloc[0])
        info = dict(fold=man.loc[eid, "fold"], path_km=round(float(man.loc[eid, "path_len_m"]) / 1000, 2),
                    nodes_per_km2=round(float(dens.loc[eid, "nodes_per_km2"]), 1) if eid in dens.index else np.nan,
                    n_turns=int(turns.loc[eid, "n_turns"]) if turns is not None and eid in turns.index else -1)
        frame = frame_png(ep.driver, ep.drive, ep.t0 + 150.0, tmp)
        items[eid] = dict(ep=ep, label=label, frame=frame, roads=roads(ep.frame, *bx), box=bx, beam=beam, pf=pf, ade_beam=ade_b, ade_pf=ade_p, info=info)
        print(f"{eid}: beam {ade_b:.1f} m, filter {ade_p:.1f} m, {len(items[eid]['roads'])} roads, frame {'yes' if frame else 'no'}")
    if "examples" in args.figs:
        fig_examples([items[e] for e, _ in EXAMPLES], tmp)
    if "method" in args.figs:
        fig_method(items[METHOD_EXAMPLE], cand, tmp)
    print("done ->", F.FIG)


if __name__ == "__main__":
    main()
