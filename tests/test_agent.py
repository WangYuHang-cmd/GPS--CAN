"""Agent study: tool runner and read-only tool box on the synthetic grid (visibility rules, safe fields), the budgeted
control loop with a scripted fake client (decision / call / token / timeout semantics), the supervised selector's
determinism and the seed-aware statistics."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from cantraj import io, stats
from cantraj.agent import selector as S
from cantraj.agent.loop import Budget, run_agent_episode
from cantraj.agent.runner import ToolRunner
from cantraj.agent.toolbox import EpisodeToolBox, ToolError, assert_safe
from cantraj.geo import LocalFrame
from cantraj.maps.graph import EpisodeGraphView
from test_map_beam import _inputs, _truth, grid_graph

SYSTEM = "test system prompt"
SCHEMA = []


def _deny(*a, **k):
    raise PermissionError("no reference")


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    g = grid_graph()
    view = EpisodeGraphView(g, LocalFrame(37.0, -122.0))
    inp = _inputs(theta0=0.0, T=60.0)
    tq = np.arange(1.0, 60.0, 1.0)
    real = io.load_gps
    io.load_gps = _deny
    try:
        tr = ToolRunner(inp, tq, view, theta0=0.0, t0=0.0, t_end=60.0)
        tr.run_all()
        d = tmp_path_factory.mktemp("agentcache")
        diag = tr.write_cache(str(d), extra=dict(episode_id="synthetic"))
    finally:
        io.load_gps = real
    return str(d), diag, tq


def test_runner_beam_recovers_truth_and_cache_is_safe(cache):
    d, diag, tq = cache
    assert diag["tools"]["beam"]["n_candidates"] >= 1
    tb = EpisodeToolBox(d)
    xy = tb.candidate_xy("beam-1")
    err = np.hypot(*(xy - _truth(tq)).T)
    assert err.mean() < 5.0
    assert_safe(diag["candidates"])
    assert_safe(diag["motion"])
    txt = json.dumps(diag)
    for bad in ("p0_lat", "p0_lon", "latitude", "route_ok", "ade"):
        assert bad not in txt


def test_toolbox_visibility_rules(cache):
    d, diag, _ = cache
    tb = EpisodeToolBox(d)
    r = tb.run_reconstructor("beam")
    assert_safe(r)
    json.dumps(r)
    assert all(c["id"].startswith("beam-") for c in r["candidates"])
    c1 = r["candidates"][0]
    assert c1["overlap_with_called_top1"]["pf"].startswith("unknown")
    assert c1["overlap_with_called_top1"]["ep"].startswith("unknown")
    assert c1["n_called_tools"] == 0
    with pytest.raises(ToolError):
        tb.inspect_candidate("pf-1")
    with pytest.raises(ToolError):
        tb.query_local_graph("pf-1", 100.0)
    assert tb.submit("pf-1") == (False, "invalid_choice")
    assert tb.submit("beam-99") == (False, "invalid_choice")
    assert tb.submit("beam-1") == (True, "submitted")
    ins = tb.inspect_candidate("beam-1")
    assert_safe(ins)
    assert all(dv["other"].startswith("beam-") for dv in ins["divergences"])      # no pf / ep divergence before they run
    if diag["tools"]["pf"]["n_candidates"]:
        r2 = tb.run_reconstructor("pf")
        assert tb.tool_logical_s == pytest.approx(tb.logical_time("beam") + tb.logical_time("pf"))
        c1 = tb.run_reconstructor("beam")["candidates"][0]
        assert isinstance(c1["overlap_with_called_top1"]["pf"], float)
        assert c1["overlap_with_called_top1"]["ep"].startswith("unknown")
        assert tb.submit("pf-1") == (True, "submitted")
        assert r2["already_run"] is False
    q = tb.query_local_graph("beam-1", 200.0)
    assert_safe(q)
    assert "junction" in q or "note" in q


def test_toolbox_import_footprint():
    code = ("import sys, cantraj.agent.toolbox, cantraj.agent.loop, cantraj.agent.llm, cantraj.agent.selector; "
            "bad=[m for m in ('cantraj.io','cantraj.episodes','cantraj.maps.graph','cantraj.baselines.map_beam') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    env = dict(os.environ, PYTHONPATH=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


class FakeClient:
    def __init__(self, script):
        self.script, self.calls = list(script), []

    def chat(self, messages, tools, seed, max_tokens, tool_choice="auto", enable_thinking=True):
        self.calls.append(dict(max_tokens=max_tokens, tool_choice=tool_choice, enable_thinking=enable_thinking, n_msgs=len(messages)))
        step = self.script.pop(0)
        calls = [dict(id=f"c{i}", name=n, arguments=json.dumps(a)) for i, (n, a) in enumerate(step.get("calls", []))]
        return dict(content=step.get("content", ""), reasoning=step.get("reasoning"), tool_calls=calls, finish_reason=step.get("finish", "tool_calls" if calls else "stop"),
                    prompt_tokens=100, completion_tokens=step.get("tokens", 50), wall_s=step.get("wall", 0.1), retries=0,
                    assistant_message={"role": "assistant", "content": step.get("content", ""), "tool_calls": [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]})

    def count_tokens(self, text):
        return len(text.split())


def test_loop_call_budget_and_forced_submit(cache):
    d, _, _ = cache
    tb = EpisodeToolBox(d)
    many = [("run_reconstructor", {"name": "beam"}), ("run_reconstructor", {"name": "pf"}), ("run_reconstructor", {"name": "ep"}),
            ("inspect_candidate", {"candidate_id": "beam-1"}), ("query_local_graph", {"candidate_id": "beam-1", "distance_m": 150}),
            ("inspect_candidate", {"candidate_id": "beam-1"}), ("inspect_candidate", {"candidate_id": "beam-1"}), ("inspect_candidate", {"candidate_id": "beam-1"})]
    fc = FakeClient([dict(calls=many, tokens=300, reasoning="a b c"), dict(calls=[("submit_choice", {"candidate_id": "beam-1", "reason": "x"})], tokens=40)])
    res = run_agent_episode(tb, fc, Budget(), SYSTEM, SCHEMA, seed=0, enable_thinking=True, episode_id="e")
    assert res.status == "submitted" and res.choice == "beam-1"
    assert res.n_tool_calls == 7 and res.n_decisions == 2 and res.forced_submit
    assert fc.calls[0]["tool_choice"] == "required" and fc.calls[0]["max_tokens"] == 2048 - 0 and fc.calls[0]["enable_thinking"] is True
    assert fc.calls[1]["tool_choice"]["function"]["name"] == "submit_choice" and fc.calls[1]["enable_thinking"] is False
    assert fc.calls[1]["max_tokens"] == min(2048, 4096 - 300)
    assert res.completion_tokens == 340 and res.reasoning_tokens == 3
    assert "budget exhausted" in res.messages[-3]["content"] or any("budget exhausted" in m.get("content", "") for m in res.messages if m["role"] == "tool")
    assert res.tool_logical_s == pytest.approx(sum(tb.logical_time(t) for t in ("beam", "pf", "ep")), abs=1e-9)
    assert res.virtual_clock_s >= res.llm_wall_s + res.tool_logical_s


def test_loop_invalid_and_hidden_choice(cache):
    d, _, _ = cache
    tb = EpisodeToolBox(d)
    fc = FakeClient([dict(calls=[("run_reconstructor", {"name": "beam"})]), dict(calls=[("submit_choice", {"candidate_id": "pf-1", "reason": "hidden"})])])
    res = run_agent_episode(tb, fc, Budget(), SYSTEM, SCHEMA, seed=1, enable_thinking=False)
    assert res.status == "invalid_choice" and res.choice is None


def test_loop_truncation_vs_total_budget(cache):
    d, _, _ = cache
    tb = EpisodeToolBox(d)
    fc = FakeClient([dict(calls=[], finish="length", tokens=500), dict(calls=[("run_reconstructor", {"name": "beam"})], tokens=100),
                     dict(calls=[("submit_choice", {"candidate_id": "beam-1", "reason": "ok"})], tokens=10)])
    res = run_agent_episode(tb, fc, Budget(), SYSTEM, SCHEMA, seed=0, enable_thinking=True)
    assert res.status == "submitted" and res.truncations == 1 and res.n_decisions == 3
    assert fc.calls[1]["max_tokens"] == min(2048, 4096 - 500 - 256)
    fc2 = FakeClient([dict(calls=[], finish="length", tokens=4096)])
    res2 = run_agent_episode(EpisodeToolBox(d), fc2, Budget(), SYSTEM, SCHEMA, seed=0, enable_thinking=True)
    assert res2.status == "total_token_budget"


def test_loop_no_submit_and_timeout(cache):
    d, _, _ = cache
    fc = FakeClient([dict(calls=[("run_reconstructor", {"name": "beam"})])] * 3 + [dict(calls=[], finish="stop")])
    res = run_agent_episode(EpisodeToolBox(d), fc, Budget(), SYSTEM, SCHEMA, seed=0, enable_thinking=True)
    assert res.status == "no_submit" and res.n_decisions == 4 and res.n_tool_calls == 3
    fc = FakeClient([dict(calls=[("run_reconstructor", {"name": "beam"})], wall=700.0)])
    res = run_agent_episode(EpisodeToolBox(d), fc, Budget(), SYSTEM, SCHEMA, seed=0, enable_thinking=True)
    assert res.status == "timeout"


def test_selector_deterministic_and_tie_to_beam():
    rng = np.random.default_rng(0)
    feats = []
    for i in range(200):
        tool = ["beam", "pf", "ep"][i % 3]
        f = dict(tool=tool, rank=1 + i % 4, hd_abs_rms_deg=rng.uniform(2, 60), hd_abs_p90_deg=rng.uniform(5, 120), hd_drift_rms_deg=rng.uniform(1, 40),
                 turns_unmatched=int(rng.integers(0, 4)), len_ratio_dev=rng.uniform(0, 0.3), uturns=0, overlap_mean=rng.uniform(0, 1),
                 n_tools_with_output=2, end_disagree_m=rng.uniform(0, 500))
        feats.append(f)
    X = np.stack([S.feature_vector(f) for f in feats])
    y = (X[:, 3] < 25).astype(int)
    pipe = S.fit(X, y)
    sel = S.Selector(S.export(pipe))
    cands = {f"beam-{i + 1}": feats[i] for i in range(3)} | {"pf-1": feats[5], "ep-1": feats[7]}
    c1 = sel.select(cands)
    assert c1 == S.select_with_pipeline(pipe, cands) == sel.select(dict(reversed(list(cands.items()))))
    same = dict(tool="beam", rank=1, hd_abs_rms_deg=10.0, hd_abs_p90_deg=20.0, hd_drift_rms_deg=5.0, turns_unmatched=0, len_ratio_dev=0.0, uturns=0,
                overlap_mean=0.5, n_tools_with_output=2, end_disagree_m=10.0)
    assert sel.select({"ep-1": dict(same, tool="ep"), "beam-1": same, "pf-1": dict(same, tool="pf")}) in ("beam-1", "ep-1", "pf-1")
    tie = {"beam-1": same, "beam-2": dict(same, rank=1)}
    assert sel.select(tie) == "beam-1"


def test_seed_statistics_with_inf():
    v = np.array([[1.0, 2.0, np.inf, 4.0, np.nan], [2.0, 2.0, 3.0, np.inf, 5.0]])
    c = ["a", "a", "b", "b", "c"]
    p, (lo, hi) = stats.seed_median(v, c, n_boot=100)
    assert np.isfinite(p) and np.isfinite(lo) and not np.isnan(hi)
    p2, (lo2, hi2) = stats.seed_median(np.full((3, 4), np.inf), ["a", "a", "b", "b"], n_boot=20)
    assert p2 == np.inf and lo2 == np.inf and hi2 == np.inf
    p3, _ = stats.seed_median(v[:1], c, n_boot=50)
    assert p3 == stats.median_inf(v[0], c, n_boot=50)[0]
    r = stats.seed_paired(np.array([[1.0, -1.0, np.nan, 2.0], [0.5, np.nan, np.nan, 1.0]]), ["a", "a", "b", "b"], n_boot=50)
    assert r["n_per_seed"] == [3, 2] and np.isfinite(r["median"]) and not np.isnan(r["lo"]) and not np.isnan(r["hi"])


# ---- C-max additions: dead-reckoning shape evidence, junction consistency, compare / score tools on a max cache
from cantraj.agent import diagnostics as D  # noqa: E402
from cantraj.agent.runner import TOOLS_MAX  # noqa: E402


def test_dr_path_and_distances():
    s = np.array([0.0, 10.0, 20.0, 30.0])
    psi = np.zeros(4)
    xy = D.dr_path(s, psi, theta0=0.0)                       # straight east at heading 0
    assert np.allclose(xy[:, 0], s) and np.allclose(xy[:, 1], 0.0)
    d = D.dr_distances(xy + np.array([0.0, 5.0]), xy, t_rel=np.array([1.0, 2.0, 61.0, 121.0]))
    assert d["dr_dist_first60_m"] == pytest.approx(5.0) and d["dr_mean_dist_m"] == pytest.approx(5.0) and d["dr_end_dist_m"] == pytest.approx(5.0)
    jc = D.junction_consistency([dict(taken_turn_deg=90, dr_change_deg=80), dict(taken_turn_deg=-90, dr_change_deg=85), dict(taken_turn_deg=0, dr_change_deg=10)])
    assert jc == dict(jn_agree=2, jn_disagree=1, jn_total=3)


@pytest.fixture(scope="module")
def cache_max(tmp_path_factory):
    g = grid_graph()
    view = EpisodeGraphView(g, LocalFrame(37.0, -122.0))
    inp = _inputs(theta0=0.0, T=60.0)
    tq = np.arange(1.0, 60.0, 1.0)
    tr = ToolRunner(inp, tq, view, theta0=0.0, t0=0.0, t_end=60.0, tools=TOOLS_MAX)
    tr.run_all()
    d = tmp_path_factory.mktemp("agentcache_max")
    diag = tr.write_cache(str(d), extra=dict(episode_id="synthetic"))
    return str(d), diag


def test_max_cache_tools_compare_and_score(cache_max):
    d, diag = cache_max
    assert set(diag["tools"]) == set(TOOLS_MAX)
    c1 = diag["candidates"]["beam-1"]
    for k in ("dr_dist_first60_m", "jn_agree", "jn_disagree", "consensus_tools"):
        assert k in c1
    sel = S.Selector(dict(features=S.FEATURES_MAX, mean=[0.0] * len(S.FEATURES_MAX), scale=[1.0] * len(S.FEATURES_MAX), coef=[0.0] * len(S.FEATURES_MAX), intercept=0.0))
    tb = EpisodeToolBox(d, selector=sel)
    assert tb.tools == tuple(TOOLS_MAX)
    with pytest.raises(ToolError):
        tb.compare_candidates(["beam-1", "pf-1"])                 # nothing visible yet
    r = tb.run_reconstructor("beam")
    assert "dr_dist_first60_m" in r["candidates"][0] and "consensus_tools" not in r["candidates"][0]   # consensus over unrun tools stays hidden
    if diag["tools"]["beam_wide"]["n_candidates"]:
        tb.run_reconstructor("beam_wide")
        cmp_ = tb.compare_candidates(["beam-1", "beam_wide-1"])
        assert_safe(cmp_)
        json.dumps(cmp_)
        assert set(cmp_["candidates"]) == {"beam-1", "beam_wide-1"} and cmp_["pairs"][0]["a"] == "beam-1"
    sc = tb.score_candidates(["beam-1"])
    assert sc["selector_probability"]["beam-1"] == pytest.approx(0.5)
    with pytest.raises(ToolError):
        tb.score_candidates(["pf-1"])                             # pf not run: not visible
    f = tb.features("beam-1")
    assert all(k in f for k in ("jn_disagree", "dr_dist_first60_m", "consensus_tools"))
    x = S.feature_vector(f, S.FEATURES_MAX)
    assert x.shape == (len(S.FEATURES_MAX),) and np.isfinite(x).all()
