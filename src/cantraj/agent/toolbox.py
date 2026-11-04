"""Read-only, visibility-enforcing tool box for the LLM agent (agent study).

Reads one window's cache written by `runner.ToolRunner` and exposes four tools. Runtime state decides what the agent
may see: `called_tools` (reconstructors the agent has run) and `visible_candidates`. Any field that depends on a
candidate of a tool the agent has not run is reported as unknown; inspection, graph queries and the final choice accept
visible candidates only. This module imports neither the graph nor the reconstructors nor the reference loaders.
"""
from __future__ import annotations

import json
import os

import numpy as np

TOOLS = ("beam", "pf", "ep")
UNKNOWN = "unknown (tool not run)"
AGREE_OVERLAP = 0.8
TOOL_INFO = {
    "beam": "offline map beam search over the visible road graph using speed and steering angle; returns up to 10 ranked whole-path candidates",
    "pf": "map particle filter using speed and steering angle, run causally, best final lineage read out; returns 1 candidate",
    "ep": "Elastic Pathing search using speed only (plus the given start and heading); returns up to 10 ranked candidates",
}
SINGLE_FIELDS = ("path_len_m", "can_dist_m", "len_ratio_dev", "hd_abs_rms_deg", "hd_abs_p90_deg", "hd_drift_rms_deg",
                 "turns_path", "turns_dr", "turns_unmatched", "uturns", "end_east_m", "end_north_m", "net_disp_m")
NEVER = ("lat", "lon", "latitude", "longitude", "p0_lat", "p0_lon", "ref", "ade", "route_ok", "score", "way_id", "node_id", "tile")


class ToolError(Exception):
    pass


def _r(x, nd=1):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return x
    if not np.isfinite(xf):
        return None
    return round(xf, nd) if nd else int(round(xf))


class EpisodeToolBox:
    def __init__(self, cache_dir):
        self.dir = cache_dir
        with open(os.path.join(cache_dir, "diagnostics.json")) as f:
            self.cache = json.load(f)
        with open(os.path.join(cache_dir, "junctions.json")) as f:
            self.junctions = json.load(f)
        self._arrays = None
        self.called_tools = []
        self.visible = []
        self.n_calls = 0
        self.tool_logical_s = 0.0

    # ---- program-side helpers (never shown to the LLM)
    def _arr(self):
        if self._arrays is None:
            self._arrays = np.load(os.path.join(self.dir, "candidates.npz"))
        return self._arrays

    def candidate_xy(self, cid):
        return np.asarray(self._arr()[f"xy__{cid}"], dtype=float)

    def t_rel(self):
        return np.asarray(self._arr()["t_rel"], dtype=float)

    def all_candidates(self):
        return list(self.cache["candidates"])

    def tool_of(self, cid):
        return cid.rsplit("-", 1)[0]

    def tool_candidates(self, name):
        return [c for c in self.cache["candidates"] if self.tool_of(c) == name]

    def logical_time(self, name):
        return float(self.cache["tools"][name]["logical_time_s"])

    # ---- LLM-facing
    def motion_summary(self):
        m = dict(self.cache["motion"])
        m["start_position"] = "origin (0 m east, 0 m north)"
        m["coordinate_note"] = "positions in metres relative to the start; compass headings in degrees clockwise from north"
        return m

    def tool_descriptions(self):
        return dict(TOOL_INFO)

    def run_reconstructor(self, name):
        if name not in TOOLS:
            raise ToolError(f"unknown reconstructor {name!r}; choose one of {list(TOOLS)}")
        first = name not in self.called_tools
        if first:
            self.called_tools.append(name)
            self.tool_logical_s += self.logical_time(name)
            self.visible += self.tool_candidates(name)
        t = self.cache["tools"][name]
        out = dict(tool=name, already_run=not first, n_candidates=t["n_candidates"], logical_time_s=t["logical_time_s"])
        if t.get("error"):
            out["error"] = "no solution within the tool's budget" if t["n_candidates"] == 0 else t["error"]
        out["candidates"] = [self._brief(c) for c in self.tool_candidates(name)]
        return out

    def _cross(self, cid):
        """Cross-tool fields restricted to the tools the agent has run."""
        own = self.tool_of(cid)
        pw = self.cache["pairwise"].get(cid, {})
        overlap, end_d, agree, n_other = {}, {}, 0, 0
        for name in TOOLS:
            if name == own:
                continue
            top = f"{name}-1"
            if name not in self.called_tools:
                overlap[name], end_d[name] = UNKNOWN, UNKNOWN
            elif top in pw:
                overlap[name], end_d[name] = _r(pw[top]["overlap"], 2), _r(pw[top]["end_dist_m"], 0)
                n_other += 1
                agree += int(pw[top]["overlap"] >= AGREE_OVERLAP)
            else:
                overlap[name], end_d[name] = "no candidate from this tool", "no candidate from this tool"
        out = dict(overlap_with_called_top1=overlap, end_dist_to_called_top1_m=end_d, n_called_tools_agree=agree, n_called_tools=n_other)
        own_top = f"{own}-1"
        if cid != own_top and own_top in pw:
            out["overlap_with_own_top1"] = _r(pw[own_top]["overlap"], 2)
        return out

    def _brief(self, cid):
        d = self.cache["candidates"][cid]
        b = dict(id=cid, rank_in_tool=int(cid.rsplit("-", 1)[1]), path_len_m=_r(d["path_len_m"], 0), len_ratio_dev=_r(d["len_ratio_dev"], 3),
                 hd_abs_rms_deg=_r(d["hd_abs_rms_deg"], 1), hd_drift_rms_deg=_r(d["hd_drift_rms_deg"], 1), turns_path=d["turns_path"],
                 turns_unmatched=d["turns_unmatched"], uturns=d["uturns"])
        b.update(self._cross(cid))
        return b

    def _check_visible(self, cid):
        if cid not in self.visible:
            raise ToolError(f"candidate {cid!r} is not available: run its reconstructor first (visible: {self.visible})")

    def inspect_candidate(self, cid):
        self._check_visible(cid)
        d = self.cache["candidates"][cid]
        out = dict(id=cid, rank_in_tool=int(cid.rsplit("-", 1)[1]))
        for k in SINGLE_FIELDS:
            out[k] = _r(d.get(k), 3 if k == "len_ratio_dev" else (1 if "deg" in k else 0)) if isinstance(d.get(k), (int, float)) else d.get(k)
        out["path_turns"] = d.get("path_turns", [])
        out["dr_turns"] = self.cache["motion"].get("dr_turns", [])
        out.update(self._cross(cid))
        div = []
        pw = self.cache["pairwise"].get(cid, {})
        others = [f"{n}-1" for n in self.called_tools if n != self.tool_of(cid)] + ([f"{self.tool_of(cid)}-1"] if not cid.endswith("-1") else [])
        for o in others:
            rec = pw.get(o)
            if not rec:
                continue
            if "diverge_at_m" in rec:
                div.append(dict(other=o, diverge_at_m=_r(rec["diverge_at_m"], 0), at_t_s=_r(rec.get("at_t_s"), 0), this_turn_after_deg=_r(rec.get("a_turn_after_deg"), 0),
                                other_turn_after_deg=_r(rec.get("b_turn_after_deg"), 0), dr_change_deg=_r(rec.get("dr_change_deg"), 0)))
            else:
                div.append(dict(other=o, diverge_at_m=None, note="never more than 50 m apart"))
        out["divergences"] = div[:3]
        return out

    def query_local_graph(self, cid, distance_m):
        self._check_visible(cid)
        rows = self.junctions.get(cid, [])
        if not rows:
            return dict(candidate=cid, note="no junction with alternatives along this candidate")
        try:
            dist = float(distance_m)
        except (TypeError, ValueError):
            raise ToolError("distance_m must be a number (metres along the candidate path)")
        j = min(rows, key=lambda r: abs(r["at_m"] - dist))
        return dict(candidate=cid, requested_m=dist, junction=j, n_junctions_total=len(rows),
                    junction_positions_m=[r["at_m"] for r in rows[:20]])

    def submit(self, cid):
        """Program-side validation of the final choice. Returns (ok, reason)."""
        if cid not in self.cache["candidates"]:
            return False, "invalid_choice"
        if cid not in self.visible:
            return False, "invalid_choice"
        return True, "submitted"

    # ---- selector view (B runs every tool, so it sees every field)
    def call_all(self):
        for name in TOOLS:
            self.run_reconstructor(name)
        return self

    def features(self, cid):
        """Numeric feature dict for the supervised selector (all tools called)."""
        d = self.cache["candidates"][cid]
        pw = self.cache["pairwise"].get(cid, {})
        own = self.tool_of(cid)
        ov = [pw[f"{n}-1"]["overlap"] for n in TOOLS if n != own and f"{n}-1" in pw]
        ed = [pw[f"{n}-1"]["end_dist_m"] for n in TOOLS if n != own and f"{n}-1" in pw]
        return dict(tool=own, rank=int(cid.rsplit("-", 1)[1]), hd_abs_rms_deg=d["hd_abs_rms_deg"], hd_abs_p90_deg=d["hd_abs_p90_deg"],
                    hd_drift_rms_deg=d["hd_drift_rms_deg"], turns_unmatched=d["turns_unmatched"], len_ratio_dev=d["len_ratio_dev"],
                    uturns=d["uturns"], overlap_mean=float(np.mean(ov)) if ov else 0.0, n_tools_with_output=len(ov),
                    end_disagree_m=float(np.median(ed)) if ed else 0.0)


def assert_safe(obj, path="root"):
    """Raise if a key that must never reach the LLM appears anywhere in a tool result."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in NEVER:
                raise AssertionError(f"forbidden key {k!r} at {path}")
            assert_safe(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            assert_safe(v, f"{path}[{i}]")
