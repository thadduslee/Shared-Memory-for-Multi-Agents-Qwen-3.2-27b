"""Mock LLM transport with the same interface as `llm.client.AsyncLLMClient`.

Used when `EVAL_TRANSPORT` / `JUDGE_TRANSPORT` are set to `http` while
`MOCK_MODE` is on.  Responses are routed by role and produced by the same
canned-response functions the mock harness uses, so the two transports agree --
which is the point: switching transport must not change the numbers.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm.client import ChatResult

_ROLE_BY_MODEL_HINT = (
    ("judge", "judge"),
    ("gpt-4", "judge"),
    ("qwen", "evaluator"),
    ("deepseek", "architect"),
)


class MockLLMClient:
    """Deterministic, offline, and awaitable."""

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.calls: list[dict[str, Any]] = []
        self._usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _role_for(route: str, model: str, messages: list[dict[str, str]]) -> str:
        """Fallback role inference, used only when the caller passed no `role`.

        Substring matching on the system prompt is NOT reliable -- the
        Developer's prompt says "the Architect's work order", which made every
        Developer call dispatch to the Architect responder and hung the
        Developer's loop until it ran out of turns. The `role` argument exists so production
        paths never rely on this; it remains for ad-hoc calls in tests.
        """
        system = (messages[0].get("content", "") if messages else "").lower()
        for marker in ("architect", "developer", "critic", "judge", "evaluator"):
            if f"you are the {marker}" in system[:200]:
                return marker
        if route == "vllm":
            return "evaluator"
        if route == "openai":
            return "judge"
        for hint, role in _ROLE_BY_MODEL_HINT:
            if hint in model.lower():
                return role
        return "architect"

    async def chat(
        self,
        *,
        route: str,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        role: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> ChatResult:
        from mocks.dsh_responses import canned_response

        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        role = role or self._role_for(route, model, messages)
        task = "\n\n".join(_text_of(m) for m in messages)
        self.calls.append({"route": route, "model": model, "role": role, "chars": len(task),
                           "tools": sorted(_advertised(tools))})

        text, ok = canned_response(role, task)
        usage = {"input_tokens": max(1, len(task) // 4), "output_tokens": max(1, len(text) // 4)}
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        for key, value in usage.items():
            self._usage[key] += value
        # A caller that advertised tools gets NATIVE tool calls back, exactly as
        # a real provider would answer. Without this the mock would only ever
        # exercise the fenced-block fallback, so the path the live Developer
        # actually takes -- and the tool_call_id bookkeeping it depends on --
        # would be untested offline, which is the one thing this mock is for.
        tool_calls: list[dict[str, Any]] = []
        content = text
        if tools:
            action = _fenced_action(text)
            if action and action.get("tool") in _advertised(tools):
                tool_calls = [{
                    "id": f"call_{len(self.calls)}",
                    "name": str(action["tool"]),
                    "arguments": dict(action.get("args") or {}),
                }]
                content = _thought_of(text)
        return ChatResult(
            text=text, model=model, route=route, usage=usage,
            error=None if ok else "mock failure injected",
            content=content, tool_calls=tool_calls,
        )

    async def aclose(self) -> None:
        return None

    @property
    def usage_total(self) -> dict[str, int]:
        return dict(self._usage)


def _text_of(message: dict[str, Any]) -> str:
    """One message's text, whatever shape it is in.

    Tool results arrive as `{"role": "tool", "content": "..."}` and matter as
    much as the user turns: the Developer's `<STATUS>` block rides on them, and
    the canned Developer policy reads that block to decide what to do next.
    """
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item.get("text", "") if isinstance(item, dict) else str(item) for item in content
        )
    return ""


def _advertised(tools: list[dict[str, Any]] | None) -> set[str]:
    return {
        str((tool.get("function") or {}).get("name") or "")
        for tool in (tools or [])
        if isinstance(tool, dict)
    } - {""}


def _fenced_action(text: str) -> dict[str, Any]:
    """The `{"tool": ..., "args": ...}` block out of a canned Developer reply."""
    from harness.dsh_client import extract_json_block

    block = extract_json_block(text) or {}
    return block if isinstance(block, dict) and "tool" in block else {}


def _thought_of(text: str) -> str:
    """The prose before the fenced block, which is what a real turn's content is."""
    head, _, _rest = text.partition("```")
    return head.strip()
