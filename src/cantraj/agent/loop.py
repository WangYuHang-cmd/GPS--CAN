"""Budgeted control loop of the single LLM controller (agent study, configuration C).

Semantics (frozen in configs/agent/budget.json):
- at most `max_decisions` assistant turns; at most `max_tool_calls` non-submit tool calls, one `submit_choice` call is
  reserved on top of them; every reconstructor at most once (a repeat returns the same result and still counts);
- completion tokens (thinking included, as the server counts them) at most `total_completion_tokens`; non-final turns
  get max_tokens = min(per_turn_max_tokens, remaining - reserve_tokens); the reserve is spent only by the final turn;
- the final turn (the last allowed decision, or no non-submit call left) forces `submit_choice` through tool_choice and
  runs without thinking; a reply cut by max_tokens with budget left is a `per_turn_truncation` (one decision used, the
  model is told to answer briefly); cut with no budget left it is the failure `total_token_budget`;
- a virtual clock = LLM wall time + logical tool time (measured when the cache was built) + tool execution time; the
  window fails with `timeout` when it exceeds `timeout_virtual_s`; the reported cost uses the same clock;
- the program re-validates every candidate id against the visible set; the constrained output format never replaces it.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .llm import LLMError, parse_arguments
from .toolbox import ToolError, assert_safe

SUBMIT = "submit_choice"
NON_SUBMIT = ("run_reconstructor", "inspect_candidate", "query_local_graph")


@dataclass
class Budget:
    max_decisions: int = 4
    max_tool_calls: int = 7
    total_completion_tokens: int = 4096
    reserve_tokens: int = 256
    per_turn_max_tokens: int = 2048
    timeout_virtual_s: float = 600.0
    forced_submit_disable_thinking: bool = True
    tool_choice_non_final: str = "required"

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AgentResult:
    episode_id: str
    seed: int
    status: str = "running"
    choice: str | None = None
    submit_reason: str | None = None
    n_decisions: int = 0
    n_tool_calls: int = 0
    tools_run: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int | None = 0
    llm_wall_s: float = 0.0
    tool_logical_s: float = 0.0
    tool_exec_s: float = 0.0
    virtual_clock_s: float = 0.0
    elapsed_s: float = 0.0
    truncations: int = 0
    retries: int = 0
    forced_submit: bool = False
    error: str | None = None
    messages: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def initial_user_message(toolbox):
    body = dict(task="Recover the five-minute trajectory of this window by choosing one candidate path produced by the reconstructors.",
                motion_summary=toolbox.motion_summary(), reconstructors=toolbox.tool_descriptions(),
                procedure=["call run_reconstructor for one or more reconstructors (each at most once)",
                           "optionally inspect_candidate or query_local_graph on candidates you have seen",
                           "finish with submit_choice(candidate_id) for a candidate you have seen"])
    return json.dumps(body, ensure_ascii=False)


def run_agent_episode(toolbox, client, budget: Budget, system_prompt, tools_schema, seed, enable_thinking, episode_id="",
                      live_runner=None):
    """toolbox: EpisodeToolBox; client: QwenClient-like with .chat(...) and .count_tokens(...); live_runner: optional
    callable(tool_name) -> seconds that executes the reconstructor for real (uncached verification)."""
    res = AgentResult(episode_id=episode_id, seed=int(seed))
    t_start = time.perf_counter()
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": initial_user_message(toolbox)}]
    remaining = int(budget.total_completion_tokens)
    try:
        while res.status == "running":
            res.n_decisions += 1
            final = res.n_decisions >= budget.max_decisions or res.n_tool_calls >= budget.max_tool_calls
            if final:
                max_tokens = min(budget.per_turn_max_tokens, remaining)
                tool_choice = {"type": "function", "function": {"name": SUBMIT}}
                thinking = enable_thinking and not budget.forced_submit_disable_thinking
                res.forced_submit = True
            else:
                max_tokens = min(budget.per_turn_max_tokens, remaining - budget.reserve_tokens)
                tool_choice = budget.tool_choice_non_final
                thinking = enable_thinking
            if max_tokens <= 0:
                res.status = "total_token_budget"
                break
            out = client.chat(messages, tools_schema, seed=seed, max_tokens=max_tokens, tool_choice=tool_choice, enable_thinking=thinking)
            res.llm_wall_s += out["wall_s"]
            res.virtual_clock_s += out["wall_s"]
            res.prompt_tokens += out["prompt_tokens"]
            res.completion_tokens += out["completion_tokens"]
            res.retries += out.get("retries", 0)
            remaining -= out["completion_tokens"]
            if out.get("reasoning"):
                n_r = client.count_tokens(out["reasoning"])
                res.reasoning_tokens = None if (n_r is None or res.reasoning_tokens is None) else res.reasoning_tokens + n_r
            messages.append(dict(out["assistant_message"], reasoning_content=out.get("reasoning") or None, finish_reason=out["finish_reason"]))
            calls = out["tool_calls"]
            if not calls:
                if out["finish_reason"] == "length":
                    if remaining <= 0 or final:
                        res.status = "total_token_budget" if remaining <= 0 else "no_submit"
                        break
                    res.truncations += 1
                    messages.append({"role": "user", "content": "Your previous reply was cut off before any tool call. Answer briefly and call a tool."})
                    continue
                if final:
                    res.status = "no_submit"
                    break
                messages.append({"role": "user", "content": "You must call one of the tools."})
                continue
            for tc in calls:
                name = tc["name"]
                try:
                    args = parse_arguments(tc["arguments"])
                except LLMError as e:
                    if name == SUBMIT:
                        res.status, res.error = "parse_error", str(e)
                        break
                    _tool_reply(messages, tc, {"error": str(e)})
                    continue
                if name == SUBMIT:
                    cid = str(args.get("candidate_id", "")).strip()
                    ok, why = toolbox.submit(cid)
                    res.submit_reason = str(args.get("reason", ""))[:500]
                    if ok:
                        res.choice, res.status = cid, "submitted"
                    else:
                        res.status, res.error = "invalid_choice", f"submitted {cid!r}; visible {toolbox.visible}"
                    break
                if name not in NON_SUBMIT:
                    _tool_reply(messages, tc, {"error": f"unknown tool {name!r}"})
                    continue
                if res.n_tool_calls >= budget.max_tool_calls:
                    _tool_reply(messages, tc, {"error": "tool-call budget exhausted: call submit_choice now"})
                    continue
                res.n_tool_calls += 1
                t_e = time.perf_counter()
                live_s = 0.0
                try:
                    if name == "run_reconstructor":
                        tool = str(args.get("name", "")).strip()
                        first = tool not in toolbox.called_tools
                        reply = toolbox.run_reconstructor(tool)
                        if first:
                            res.tools_run.append(tool)
                            if live_runner is not None:            # uncached check: the real execution replaces the logical time
                                t_l = time.perf_counter()
                                secs = live_runner(tool)
                                live_s = time.perf_counter() - t_l   # excluded from the exec time below (counted once)
                            else:
                                secs = toolbox.logical_time(tool)
                            res.tool_logical_s += secs
                            res.virtual_clock_s += secs
                    elif name == "inspect_candidate":
                        reply = toolbox.inspect_candidate(str(args.get("candidate_id", "")).strip())
                    else:
                        reply = toolbox.query_local_graph(str(args.get("candidate_id", "")).strip(), args.get("distance_m"))
                    assert_safe(reply)
                except ToolError as e:
                    reply = {"error": str(e)}
                exec_s = max(time.perf_counter() - t_e - live_s, 0.0)
                res.tool_exec_s += exec_s
                res.virtual_clock_s += exec_s
                res.calls.append(dict(decision=res.n_decisions, tool=name, args=args, ok="error" not in reply))
                _tool_reply(messages, tc, reply)
                if res.virtual_clock_s > budget.timeout_virtual_s:
                    res.status = "timeout"
                    break
            if res.status == "running" and res.virtual_clock_s > budget.timeout_virtual_s:
                res.status = "timeout"
            if res.status == "running" and final:
                res.status = "no_submit"
    except LLMError as e:
        res.status, res.error = "llm_error", str(e)
    res.elapsed_s = time.perf_counter() - t_start
    res.messages = messages
    return res


def _tool_reply(messages, tc, payload):
    messages.append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": json.dumps(payload, ensure_ascii=False, default=str)})
