#!/usr/bin/env python3
"""Smoke test of the vLLM endpoint for the agent study: model list, one tool-calling request in thinking mode (hermes
parser + qwen3 reasoning parser), one forced-function request without thinking (constrained output), and /tokenize.
Prints what the protocol needs to record (fields present, token counts, latency). No benchmark data involved.

    python scripts/agent_smoke_llm.py [--base-url http://127.0.0.1:8000/v1]
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
from cantraj.agent.llm import QwenClient, parse_arguments  # noqa: E402

CFG = os.path.join(os.path.dirname(HERE), "configs", "agent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()
    sampling = json.load(open(os.path.join(CFG, "sampling.json")))
    schema = json.load(open(os.path.join(CFG, "tools_schema.json")))
    client = QwenClient(args.base_url, sampling["model"], sampling)
    models = client.models()
    print("models:", [m["id"] for m in models.get("data", [])])
    sys_p = "You control a trajectory-recovery system. Use the tools. Candidate ids look like beam-1."
    user = json.dumps(dict(task="pick the most plausible candidate", motion_summary=dict(duration_s=300, can_distance_m=4100, dr_turns=[dict(t_s=80, deg=-88)]),
                           procedure=["run a reconstructor", "then submit_choice"]))
    msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
    t = time.perf_counter()
    out = client.chat(msgs, schema, seed=0, max_tokens=1024, tool_choice="required", enable_thinking=True)
    print(f"thinking+required: finish={out['finish_reason']} completion_tokens={out['completion_tokens']} prompt_tokens={out['prompt_tokens']} wall={out['wall_s']:.1f}s")
    print("  reasoning present:", bool(out["reasoning"]), "| reasoning tokens via /tokenize:", client.count_tokens(out["reasoning"] or ""))
    print("  tool_calls:", [(c["name"], c["arguments"][:80]) for c in out["tool_calls"]])
    for c in out["tool_calls"]:
        parse_arguments(c["arguments"])
    # forced submit without thinking (the final-turn configuration)
    msgs2 = msgs + [out["assistant_message"]] + [{"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                                 "content": json.dumps(dict(tool="beam", n_candidates=2, candidates=[dict(id="beam-1", hd_abs_rms_deg=6.1, turns_unmatched=0), dict(id="beam-2", hd_abs_rms_deg=31.0, turns_unmatched=2)]))}
                                                for c in out["tool_calls"]]
    out2 = client.chat(msgs2, schema, seed=0, max_tokens=256, tool_choice={"type": "function", "function": {"name": "submit_choice"}}, enable_thinking=False)
    print(f"forced submit, no thinking: finish={out2['finish_reason']} completion_tokens={out2['completion_tokens']} wall={out2['wall_s']:.1f}s reasoning_present={bool(out2['reasoning'])}")
    print("  tool_calls:", [(c["name"], c["arguments"][:120]) for c in out2["tool_calls"]])
    args2 = parse_arguments(out2["tool_calls"][0]["arguments"]) if out2["tool_calls"] else None
    print("  parsed:", args2)
    # seed repeatability (nominal): same request twice
    a = client.chat(msgs, schema, seed=1, max_tokens=512, tool_choice="required", enable_thinking=False)
    b = client.chat(msgs, schema, seed=1, max_tokens=512, tool_choice="required", enable_thinking=False)
    print("same seed twice, identical tool call:", [c["arguments"] for c in a["tool_calls"]] == [c["arguments"] for c in b["tool_calls"]])
    print(f"total {time.perf_counter() - t:.1f}s")


if __name__ == "__main__":
    main()
