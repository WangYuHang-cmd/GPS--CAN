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

TOOLS = ("beam", "pf", "ep")                 # frozen study; a cache may carry more tools (C-max presets), read from the cache
UNKNOWN = "unknown (tool not run)"
AGREE_OVERLAP = 0.8
TOOL_INFO = {
    "beam": "offline map beam search over the visible road graph using speed and steering angle; returns up to 10 ranked whole-path candidates",
    "pf": "map particle filter using speed and steering angle, run causally, best final lineage read out; returns 1 candidate",
    "ep": "Elastic Pathing search using speed only (plus the given start and heading); returns up to 10 ranked candidates",
}
SINGLE_FIELDS = ("path_len_m", "can_dist_m", "len_ratio_dev", "hd_abs_rms_deg", "hd_abs_p90_deg", "hd_drift_rms_deg",
                 "turns_path", "turns_dr", "turns_unmatched", "uturns", "end_east_m", "end_north_m", "net_disp_m",
                 "dr_dist_first60_m", "dr_dist_first120_m", "dr_mean_dist_m", "jn_agree", "jn_disagree")   # the last five: C-max caches
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
    def __init__(self, cache_dir, selector=None):
        self.dir = cache_dir
        with open(os.path.join(cache_dir, "diagnostics.json")) as f:
            self.cache = json.load(f)
        with open(os.path.join(cache_dir, "junctions.json")) as f:
            self.junctions = json.load(f)
        self.tools = tuple(self.cache["tools"])             # base tools, or the C-max presets when the cache carries them
        self.selector = selector                            # optional frozen selector behind score_candidates (C-max)
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
        return {n: self.cache["tools"][n].get("info", TOOL_INFO.get(n, "")) for n in self.tools}

    def run_reconstructor(self, name):
        if name not in self.tools:
            raise ToolError(f"unknown reconstructor {name!r}; choose one of {list(self.tools)}")
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
        for name in self.tools:
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
        """Compact per-candidate line for run_reconstructor (the full record is in inspect_candidate / compare_candidates)."""
        d = self.cache["candidates"][cid]
        b = dict(id=cid, len_ratio_dev=_r(d["len_ratio_dev"], 2), hd_abs_rms_deg=_r(d["hd_abs_rms_deg"], 0), hd_drift_rms_deg=_r(d["hd_drift_rms_deg"], 0),
                 turns_path=d["turns_path"], turns_unmatched=d["turns_unmatched"], uturns=d["uturns"])
        for k, kk in (("dr_dist_first60_m", "dr60_m"), ("dr_dist_first120_m", "dr120_m"), ("jn_agree", "jn_agree"), ("jn_disagree", "jn_disagree")):   # C-max caches
            if k in d:
                b[kk] = _r(d[k], 0)
        cross = self._cross(cid)
        if cross["n_called_tools"] > 0:                       # agreement with the Top-1 of the tools already run (others are unknown)
            b["overlap_with_called_top1"] = {k: v for k, v in cross["overlap_with_called_top1"].items() if not isinstance(v, str)}
            b["n_called_tools_agree"] = cross["n_called_tools_agree"]
        if "overlap_with_own_top1" in cross:
            b["overlap_with_own_top1"] = cross["overlap_with_own_top1"]
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

    # ---- C-max tools
    def compare_candidates(self, cids):
        """Side-by-side diagnostics of 2 to 4 visible candidates, with the first divergence of every pair and the road
        junction nearest to it (exits, which one each candidate took, the dead-reckoned heading change there)."""
        if not isinstance(cids, (list, tuple)) or not 2 <= len(cids) <= 4:
            raise ToolError("compare_candidates takes a list of 2 to 4 candidate ids")
        cids = [str(c).strip() for c in cids]
        for c in cids:
            self._check_visible(c)
        table = {c: dict(self._brief(c), end_east_m=_r(self.cache["candidates"][c]["end_east_m"], 0), end_north_m=_r(self.cache["candidates"][c]["end_north_m"], 0)) for c in cids}
        pairs = []
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                rec = self.cache["pairwise"].get(a, {}).get(b)
                if rec is None:
                    rec = self._pair_runtime(a, b)
                item = dict(a=a, b=b, overlap=_r(rec.get("overlap"), 2), end_dist_m=_r(rec.get("end_dist_m"), 0))
                if rec.get("diverge_at_m") is not None:
                    item.update(diverge_at_m=_r(rec["diverge_at_m"], 0), at_t_s=_r(rec.get("at_t_s"), 0),
                                a_turn_after_deg=_r(rec.get("a_turn_after_deg"), 0), b_turn_after_deg=_r(rec.get("b_turn_after_deg"), 0),
                                dr_change_deg=_r(rec.get("dr_change_deg"), 0))
                    ja = self._nearest_junction(a, float(rec["diverge_at_m"]))
                    jb = self._nearest_junction(b, float(rec["diverge_at_m"]))
                    if ja is not None:
                        item["junction_on_a"] = ja
                    if jb is not None and (ja is None or jb["at_m"] != ja["at_m"]):
                        item["junction_on_b"] = jb
                else:
                    item["note"] = "never more than 50 m apart"
                pairs.append(item)
        return dict(candidates=table, pairs=pairs)

    def _pair_runtime(self, a, b):
        from .diagnostics import PathSet
        ps = PathSet({a: self.candidate_xy(a), b: self.candidate_xy(b)})
        rec = dict(overlap=ps.overlap(a, b), end_dist_m=ps.end_distance(a, b))
        dv = ps.divergence(a, b)
        if dv is not None:
            i, s_div = dv
            arr = self._arr()
            psi = np.asarray(arr["psi_q"], dtype=float) if "psi_q" in arr else None
            rec.update(diverge_at_m=s_div, at_t_s=float(self.t_rel()[i]))
            if psi is not None:
                i0, i1 = max(i - 5, 0), min(i + 5, len(psi) - 1)
                rec["dr_change_deg"] = float(np.degrees(psi[i1] - psi[i0]))
        return rec

    def _nearest_junction(self, cid, dist_m):
        rows = self.junctions.get(cid, [])
        if not rows:
            return None
        j = min(rows, key=lambda r: abs(r["at_m"] - dist_m))
        if abs(j["at_m"] - dist_m) > 150.0:
            return None
        return j

    def score_candidates(self, cids):
        """Probabilities of a frozen selector (trained on validation labels) for visible candidates."""
        if self.selector is None:
            raise ToolError("no selector is available in this configuration")
        if not isinstance(cids, (list, tuple)) or not cids:
            raise ToolError("score_candidates takes a non-empty list of candidate ids")
        out = {}
        for c in cids:
            c = str(c).strip()
            self._check_visible(c)
            out[c] = round(float(self.selector.proba(self.features(c, called_only=True))), 3)
        return dict(selector_probability=out, note="probability that the candidate recovers the route, from a model fitted on validation drives")

    # ---- selector view (B runs every tool, so it sees every field)
    def call_all(self):
        for name in self.tools:
            self.run_reconstructor(name)
        return self

    def features(self, cid, called_only=False):
        """Numeric feature dict for the supervised selector. With called_only (the agent's score_candidates tool) the
        cross-tool fields use only the tools the agent has run; the selector itself (B) calls every tool first."""
        d = self.cache["candidates"][cid]
        pw = self.cache["pairwise"].get(cid, {})
        own = self.tool_of(cid)
        names = [n for n in self.tools if n != own and (not called_only or n in self.called_tools)]
        ov = [pw[f"{n}-1"]["overlap"] for n in names if f"{n}-1" in pw]
        ed = [pw[f"{n}-1"]["end_dist_m"] for n in names if f"{n}-1" in pw]
        f = dict(tool=own, rank=int(cid.rsplit("-", 1)[1]), hd_abs_rms_deg=d["hd_abs_rms_deg"], hd_abs_p90_deg=d["hd_abs_p90_deg"],
                 hd_drift_rms_deg=d["hd_drift_rms_deg"], turns_unmatched=d["turns_unmatched"], len_ratio_dev=d["len_ratio_dev"],
                 uturns=d["uturns"], overlap_mean=float(np.mean(ov)) if ov else 0.0, n_tools_with_output=len(ov),
                 end_disagree_m=float(np.median(ed)) if ed else 0.0)
        for k in ("jn_agree", "jn_disagree", "dr_dist_first60_m", "dr_dist_first120_m", "dr_mean_dist_m"):
            if k in d:
                f[k] = d[k]
        f["consensus_tools"] = sum(1 for o in ov if o >= AGREE_OVERLAP)
        return f


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
