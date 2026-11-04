"""OpenAI-compatible chat client for the vLLM-served Qwen3 model (agent study). Records everything the protocol
needs: completion tokens (thinking included, as vLLM counts them), the reasoning text (counted separately through the
server's /tokenize endpoint), tool calls, finish reason and wall time. Retries only network-level errors.
"""
from __future__ import annotations

import json
import time

import requests


class LLMError(Exception):
    pass


class QwenClient:
    def __init__(self, base_url, model, sampling, api_key="EMPTY", timeout_s=600.0, max_retries=2):
        from openai import OpenAI
        self.base_url, self.model, self.sampling = base_url.rstrip("/"), model, dict(sampling)
        self.client = OpenAI(base_url=self.base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        self.max_retries = max_retries
        self.root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url

    def chat(self, messages, tools, seed, max_tokens, tool_choice="auto", enable_thinking=True):
        extra = {"top_k": int(self.sampling.get("top_k", 20)), "min_p": float(self.sampling.get("min_p", 0.0)),
                 "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
        if self.sampling.get("presence_penalty"):
            extra["presence_penalty"] = float(self.sampling["presence_penalty"])
        last = None
        for attempt in range(self.max_retries + 1):
            t = time.perf_counter()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, messages=messages, tools=tools, tool_choice=tool_choice, seed=int(seed),
                    temperature=float(self.sampling["temperature"]), top_p=float(self.sampling["top_p"]),
                    max_tokens=int(max_tokens), extra_body=extra)
                wall = time.perf_counter() - t
                return self._unpack(resp, wall, attempt)
            except Exception as e:  # noqa: BLE001
                last = e
                name = type(e).__name__
                if attempt < self.max_retries and name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "ConnectionError"):
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise LLMError(f"{name}: {str(e)[:200]}") from e
        raise LLMError(str(last))

    @staticmethod
    def _unpack(resp, wall, attempt):
        ch = resp.choices[0]
        msg = ch.message
        extra = getattr(msg, "model_extra", None) or {}
        reasoning = extra.get("reasoning_content") or getattr(msg, "reasoning_content", None) or extra.get("reasoning")
        calls = []
        for tc in (msg.tool_calls or []):
            calls.append(dict(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or ""))
        usage = resp.usage
        return dict(content=msg.content, reasoning=reasoning, tool_calls=calls, finish_reason=ch.finish_reason,
                    prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0), completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                    wall_s=float(wall), retries=int(attempt),
                    assistant_message=_assistant_message(msg, calls))

    def count_tokens(self, text):
        """Token count through vLLM's /tokenize (None if the endpoint is unavailable)."""
        if not text:
            return 0
        try:
            r = requests.post(f"{self.root}/tokenize", json={"model": self.model, "prompt": text}, timeout=30)
            r.raise_for_status()
            return int(r.json()["count"])
        except Exception:  # noqa: BLE001
            return None

    def models(self):
        r = requests.get(f"{self.base_url}/models", timeout=30)
        r.raise_for_status()
        return r.json()


def _assistant_message(msg, calls):
    """The assistant turn as it is sent back in the next request (reasoning is not resent: Qwen3's template drops
    earlier thinking blocks anyway)."""
    out = {"role": "assistant", "content": msg.content or ""}
    if calls:
        out["tool_calls"] = [{"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
    return out


def parse_arguments(s):
    if isinstance(s, dict):
        return s
    try:
        v = json.loads(s or "{}")
    except json.JSONDecodeError as e:
        raise LLMError(f"tool arguments are not valid JSON: {str(e)[:80]}") from e
    if not isinstance(v, dict):
        raise LLMError("tool arguments must be a JSON object")
    return v
