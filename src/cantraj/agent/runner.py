"""Tool runner (agent study): runs the three existing reconstructors once on one window, computes every observable
diagnostic and the junction tables, and writes a per-window cache. Imports the graph and the reconstructors, so it
runs only in the tool-cache process, where `io.load_gps` is denied exactly as in run_baseline.py. The cache carries no
reference position, no score, no p0 coordinates and no reference time stamps (only relative times).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

from ..baselines.elastic_pathing import ElasticPathing
from ..baselines.map_beam import MapBeamSearch, integrate_can
from ..baselines.map_pf import MapParticleFilter
from . import diagnostics as D

TOOLS = ("beam", "pf", "ep")                                 # frozen study (protocol v1)
TOOLS_MAX = ("beam", "pf", "ep", "beam_wide", "beam_loose", "pf_s1", "pf_s2", "pf_s3", "pf_s4", "ep_hm")   # ceiling exploration (C-max)
TOOL_SPECS = {
    # frozen configurations = the published benchmark defaults (icde_v11 stems in `ref_stem`)
    "beam": dict(make=lambda: MapBeamSearch("steer"), stem="tool_beam", ref_stem="map_beam_steer__G3", group="G3",
                 info="offline map beam search over the visible road graph using speed and steering angle; returns up to 10 ranked whole-path candidates"),
    "pf": dict(make=lambda: MapParticleFilter("steer", readout="lineage"), stem="tool_pf", ref_stem="map_pf_steer_lineage__G3", group="G3",
               info="map particle filter using speed and steering angle, run causally, best final lineage read out; returns 1 candidate"),
    "ep": dict(make=lambda: ElasticPathing(heading=True, friction=0.15, v_zero=1.0), stem="tool_ep", ref_stem="elastic_pathing_hp__G1", group="G1",
               info="Elastic Pathing search using speed only (plus the given start and heading); returns up to 10 ranked candidates"),
    # C-max presets: the same algorithms with other fixed settings (more or looser search, other filter seeds, no heading)
    "beam_wide": dict(make=lambda: MapBeamSearch("steer", beam_width=500, K=20), stem="tool_beam_wide", ref_stem=None, group="G3",
                      info="the beam search with a wider beam (500) returning up to 20 candidates"),
    "beam_loose": dict(make=lambda: MapBeamSearch("steer", sigma_heading_deg=25.0, sigma_turn_deg=30.0), stem="tool_beam_loose", ref_stem=None, group="G3",
                       info="the beam search with looser heading and turn tolerances (25 and 30 degrees); up to 10 candidates"),
    "pf_s1": dict(make=lambda: MapParticleFilter("steer", readout="lineage", seed=1), stem="tool_pf_s1", ref_stem=None, group="G3", info="the particle filter with another random seed (1); 1 candidate"),
    "pf_s2": dict(make=lambda: MapParticleFilter("steer", readout="lineage", seed=2), stem="tool_pf_s2", ref_stem=None, group="G3", info="the particle filter with another random seed (2); 1 candidate"),
    "pf_s3": dict(make=lambda: MapParticleFilter("steer", readout="lineage", seed=3), stem="tool_pf_s3", ref_stem=None, group="G3", info="the particle filter with another random seed (3); 1 candidate"),
    "pf_s4": dict(make=lambda: MapParticleFilter("steer", readout="lineage", seed=4), stem="tool_pf_s4", ref_stem=None, group="G3", info="the particle filter with another random seed (4); 1 candidate"),
    "ep_hm": dict(make=lambda: ElasticPathing(heading=False, friction=0.15, v_zero=1.0), stem="tool_ep_hm", ref_stem=None, group="G1",
                  info="Elastic Pathing without the start heading (speed only); up to 10 candidates"),
}
MAX_JUNCTIONS = 60
MAX_OPTIONS = 6
SNAP_M = 30.0
CACHE_VERSION = 2


def _ffill(v):
    v = np.asarray(v, dtype=float).copy()
    last = np.nan
    for i in range(len(v)):
        if np.isfinite(v[i]):
            last = v[i]
        else:
            v[i] = last
    return v


class ToolRunner:
    """One window: protocol-allowed inputs in, candidates + diagnostics out. Never sees the reference."""

    def __init__(self, inputs, t_query, graph, theta0, t0, t_end, tools=TOOLS):
        self.tools = tuple(tools)
        if theta0 is None:
            raise ValueError("the agent study runs the heading-known protocol (F-M-Hp)")
        self.inputs, self.graph = inputs, graph
        if getattr(inputs, "map_graph", None) is None:      # the reconstructors read the graph from the inputs
            inputs.map_graph = graph
        self.t_query = np.asarray(t_query, dtype=float)
        self.theta0, self.t0, self.t_end = float(theta0), float(t0), float(t_end)
        self.t_rel = self.t_query - self.t0
        self.s_q, self.psi_q = integrate_can(inputs.events, inputs.static, "steer", self.t0, self.t_query)
        ev = inputs.events
        idx = ev.causal_index(self.t_query)
        sp = _ffill(ev.cols["speed"])
        self.v_q = np.where(idx >= 0, sp[np.clip(idx, 0, len(sp) - 1)], np.nan) if len(sp) else np.full(len(self.t_query), np.nan)
        self.dr_xy = D.dr_path(self.s_q, self.psi_q, self.theta0)
        self.results = {}
        self._junctions = None

    # ---- tools
    def run_tool(self, name):
        if name in self.results:
            return self.results[name]
        spec = TOOL_SPECS[name]
        tool = spec["make"]()                       # fresh instance per window (the PF seeds its rng in __init__)
        t = time.perf_counter()
        res = dict(tool=name, candidates=[], probs=[], theta_top1=None, secs=0.0, error=None, stats=None)
        try:
            xy, th, cands, probs = tool.reconstruct(self.inputs, self.t_query)
            res["candidates"] = [np.asarray(xy, dtype=float)] if cands is None else [np.asarray(c, dtype=float) for c in cands]
            res["probs"] = [1.0] if cands is None else [float(p) for p in probs]
            res["theta_top1"] = None if th is None else np.asarray(th, dtype=float)
        except Exception as e:  # noqa: BLE001  (a failed search is a tool failure: no candidates, time still counted)
            res["error"] = f"{type(e).__name__}: {str(e)[:100]}"
        res["secs"] = time.perf_counter() - t
        st = getattr(tool, "stats", None)
        if isinstance(st, dict):
            res["stats"] = {k: (float(v) if isinstance(v, (int, float, np.floating)) else bool(v)) for k, v in st.items()}
        self.results[name] = res
        return res

    def run_all(self):
        for name in self.tools:
            self.run_tool(name)
        return self.results

    # ---- diagnostics
    def candidate_ids(self):
        out = []
        for name in self.tools:
            r = self.results.get(name)
            if r:
                out += [f"{name}-{k + 1}" for k in range(len(r["candidates"]))]
        return out

    def candidate_xy(self, cid):
        name, k = cid.rsplit("-", 1)
        return self.results[name]["candidates"][int(k) - 1]

    def diagnostics(self):
        cids = self.candidate_ids()
        single = {c: D.candidate_diagnostics(self.candidate_xy(c), self.s_q, self.psi_q, self.theta0, self.v_q) for c in cids}
        jt = self.junction_tables()
        for c in cids:                                              # C-max evidence: DR shape distance and junction consistency
            single[c].update(D.dr_distances(self.candidate_xy(c), self.dr_xy, self.t_rel))
            single[c].update(D.junction_consistency(jt.get(c, [])))
        ps = D.PathSet({c: self.candidate_xy(c) for c in cids})
        pair = {}
        for a in cids:
            pair[a] = {}
            for b in cids:
                if a == b:
                    continue
                dv = ps.divergence(a, b)
                rec = dict(overlap=round(ps.overlap(a, b), 3), end_dist_m=round(ps.end_distance(a, b), 0))
                if dv is not None:
                    i, s_div = dv
                    rec.update(diverge_at_m=round(s_div, 0), **self._turn_context(a, b, i, s_div))
                pair[a][b] = rec
        for a in cids:                                              # consensus over every tool's Top-1 (selector feature; the toolbox
            own = a.rsplit("-", 1)[0]                               # shows only the called tools' share of it)
            single[a]["consensus_tools"] = sum(1 for n in self.tools if n != own and f"{n}-1" in pair[a] and pair[a][f"{n}-1"]["overlap"] >= 0.8)
        tools = {}
        for name in self.tools:
            r = self.results.get(name, {})
            tools[name] = dict(n_candidates=len(r.get("candidates", [])), logical_time_s=round(float(r.get("secs", 0.0)), 3),
                               error=r.get("error"), stats=r.get("stats"), probs=[float(p) for p in r.get("probs", [])],
                               info=TOOL_SPECS[name]["info"], group=TOOL_SPECS[name]["group"])
        motion = D.motion_summary(self.inputs.events, self.s_q, self.psi_q, self.theta0, self.t0, self.t_end, self.t_rel)
        init = dict(duration_s=round(self.t_end - self.t0, 1), start_heading_compass_deg=round(D.compass_deg(self.theta0), 1),
                    t0_rel_s=0.0, n_query_times=int(len(self.t_query)))
        return dict(cache_version=CACHE_VERSION, tools=tools, candidates=single, pairwise=pair, motion=motion, init=init)

    def _turn_context(self, a, b, i, s_div):
        """Bearing changes of a and b over the 40 m after a's divergence point, and the DR heading change there."""
        xa, xb = self.candidate_xy(a), self.candidate_xy(b)
        sa = D.arclength(xa)
        ba = D._unwrap_nan(D.step_bearings(xa))
        ta = float(np.degrees(np.interp(min(s_div + D.TURN_WIN_M, sa[-2] if len(sa) > 1 else 0), sa[:-1], ba) - np.interp(s_div, sa[:-1], ba))) if len(ba) else float("nan")
        # b's closest point to the divergence point and its bearing change from there
        d = np.hypot(xb[:, 0] - xa[i, 0], xb[:, 1] - xa[i, 1])
        j = int(np.argmin(d))
        sb = D.arclength(xb)
        bb = D._unwrap_nan(D.step_bearings(xb))
        tb = float(np.degrees(np.interp(min(sb[j] + D.TURN_WIN_M, sb[-2] if len(sb) > 1 else 0), sb[:-1], bb) - np.interp(sb[j], sb[:-1], bb))) if len(bb) else float("nan")
        i0, i1 = max(i - 5, 0), min(i + 5, len(self.psi_q) - 1)
        dr = float(np.degrees(self.psi_q[i1] - self.psi_q[i0]))
        return dict(a_turn_after_deg=round(ta, 0), b_turn_after_deg=round(tb, 0), dr_change_deg=round(dr, 0), at_t_s=round(float(self.t_rel[i]), 0))

    def junction_tables(self):
        """Per candidate: junctions passed (nodes with at least two non-U-turn exits) with the alternatives. Only
        through nearest_edges / succ / bearing / length / highway / oneway of the visible graph; no OSM ids."""
        if self._junctions is not None:
            return self._junctions
        g = self.graph
        out = {}
        for cid in self.candidate_ids():
            xy = self.candidate_xy(cid)
            s_path = D.arclength(xy)
            seq = []                                  # (edge, first index)
            for i in range(len(xy)):
                es = g.nearest_edges(float(xy[i, 0]), float(xy[i, 1]), SNAP_M)
                if not es:
                    continue
                e = int(es[0])
                if not seq or seq[-1][0] != e:
                    seq.append((e, i))
            rows = []
            for (e, _), (e2, i2) in zip(seq[:-1], seq[1:]):
                opts = g.succ(e)
                if not any(o == e2 for o, _, _ in opts):
                    continue                          # the snapped sequence jumped (not a graph step): no junction record
                non_u = [(o, turn, u) for o, turn, u in opts if not u]
                if len(non_u) < 2:
                    continue
                taken = next(turn for o, turn, _ in opts if o == e2)
                i0, i1 = max(i2 - 5, 0), min(i2 + 5, len(self.psi_q) - 1)
                alts = sorted(non_u, key=lambda z: z[1])[:MAX_OPTIONS]
                rows.append(dict(at_m=round(float(s_path[i2]), 0), at_t_s=round(float(self.t_rel[i2]), 0),
                                 arrival_compass_deg=round(D.compass_deg(float(g.bearing[e])), 0),
                                 taken_turn_deg=round(float(math.degrees(taken)), 0),
                                 dr_change_deg=round(float(np.degrees(self.psi_q[i1] - self.psi_q[i0])), 0),
                                 n_exits=len(non_u),
                                 exits=[dict(turn_deg=round(float(math.degrees(turn)), 0), length_m=round(float(g.length[o]), 0),
                                             road=str(g.highway[o]), oneway=bool(g.oneway[o]), taken=(o == e2)) for o, turn, _ in alts]))
                if len(rows) >= MAX_JUNCTIONS:
                    break
            out[cid] = rows
        self._junctions = out
        return out

    # ---- cache
    def write_cache(self, cache_dir, extra=None):
        os.makedirs(cache_dir, exist_ok=True)
        diag = self.diagnostics()
        if extra:
            diag["run"] = extra
        arrays = {"t_rel": self.t_rel, "psi_q": self.psi_q, "s_q": self.s_q, "v_q": self.v_q, "dr_xy": self.dr_xy}   # CAN-derived series only
        for cid in self.candidate_ids():
            arrays[f"xy__{cid}"] = self.candidate_xy(cid)
        for name in self.tools:
            th = self.results.get(name, {}).get("theta_top1")
            if th is not None:
                arrays[f"theta__{name}"] = th
        np.savez_compressed(os.path.join(cache_dir, "candidates.npz"), **arrays)
        with open(os.path.join(cache_dir, "diagnostics.json"), "w") as f:
            json.dump(diag, f, default=_json_default)
        with open(os.path.join(cache_dir, "junctions.json"), "w") as f:
            json.dump(self.junction_tables(), f, default=_json_default)
        return diag


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(str(type(o)))
