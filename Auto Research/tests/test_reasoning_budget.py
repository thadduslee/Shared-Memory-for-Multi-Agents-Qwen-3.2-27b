"""Capping the reasoning budget on OpenRouter reasoning models.

WHAT BROKE (2026-08-25, reproduced directly against DeepSeek's own endpoint).
"deepseek/deepseek-v4-flash-0731" is a REASONING model, and OpenRouter bills its
reasoning tokens against `max_tokens` alongside the visible answer. The
Architect's design prompt is open-ended enough that the model never stopped
thinking: the call returned `finish_reason="length"` with
`reasoning_tokens=12288` -- the entire allowance -- and `content=None`. The node
saw an empty string, reported `architect invocation failed: empty response`, and
the run halted at MGS=0 having paid full price for the turn.

This is NOT the provider roulette `test_provider_pinning.py` covers. It
reproduces on DeepSeek's own 99.99%-uptime endpoint, and no amount of pinning
fixes it. The fix is `payload["reasoning"]`: cap the effort so the model commits.
With effort="low" the identical prompt returned `finish_reason="stop"`, 3932
reasoning tokens and a complete 13k-character design document.

THE CAP MUST BE MODEL-SCOPED, for the same reason the pin is: the Judge shares
the openrouter route with "openai/gpt-4.1", which is not a reasoning model and
has no business receiving a reasoning block.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import config
from llm.client import AsyncLLMClient

# ----------------------------------------------------------------------
# the policy: config.openrouter_reasoning
# ----------------------------------------------------------------------


def test_deepseek_models_get_a_reasoning_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setattr(config, "OPENROUTER_REASONING_MODEL_PREFIXES", ("deepseek/",))
    assert config.openrouter_reasoning("deepseek/deepseek-v4-flash-0731") == {"effort": "low"}


def test_the_judge_model_gets_no_reasoning_block(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-4.1 is not a reasoning model; sending it the block is a request bug."""
    monkeypatch.setattr(config, "OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setattr(config, "OPENROUTER_REASONING_MODEL_PREFIXES", ("deepseek/",))
    assert config.openrouter_reasoning("openai/gpt-4.1") is None


def test_an_empty_effort_disables_the_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENROUTER_REASONING_EFFORT="" is the documented off switch."""
    monkeypatch.setattr(config, "OPENROUTER_REASONING_EFFORT", "")
    assert config.openrouter_reasoning("deepseek/deepseek-v4-flash-0731") is None


def test_default_policy_caps_deepseek() -> None:
    """The shipped default, exactly -- a regression fence around the value.

    If this ever reverts to None, every agentic node goes back to burning its
    whole completion budget in the reasoning channel and the loop halts on
    iteration 1.
    """
    assert config.openrouter_reasoning("deepseek/deepseek-v4-flash-0731") == {"effort": "low"}


# ----------------------------------------------------------------------
# the wiring: AsyncLLMClient puts the block on the request
# ----------------------------------------------------------------------


class _CapturingHTTP:
    """Stands in for httpx.AsyncClient; records the payload, answers 200."""

    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._body = body

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str],
                   timeout: Any) -> _FakeResponse:
        self.payloads.append(json)
        return _FakeResponse(self._body)

    async def aclose(self) -> None:  # pragma: no cover - lifecycle only
        pass


class _FakeResponse:
    status_code = 200

    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self._body = body

    def json(self) -> dict[str, Any]:
        if self._body is not None:
            return self._body
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _chat(client: AsyncLLMClient, *, route: str, model: str) -> Any:
    return asyncio.run(client.chat(route=route, model=model,
                                   messages=[{"role": "user", "content": "hi"}]))


def test_cap_is_sent_on_the_openrouter_route() -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="openrouter", model="deepseek/deepseek-v4-flash-0731")
    assert http.payloads[0]["reasoning"] == {"effort": "low"}


def test_uncapped_model_gets_no_reasoning_key_on_openrouter() -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="openrouter", model="openai/gpt-4.1")
    assert "reasoning" not in http.payloads[0]


def test_other_routes_never_get_the_reasoning_key() -> None:
    """vLLM would reject the unknown body key; the cap must not leak there."""
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="vllm", model="deepseek/deepseek-v4-flash-0731")
    assert "reasoning" not in http.payloads[0]


# ----------------------------------------------------------------------
# the diagnosis: an empty reply must say WHY it was empty
# ----------------------------------------------------------------------


_BUDGET_EXHAUSTED = {
    "choices": [{"message": {"content": None, "reasoning": "thinking..."},
                 "finish_reason": "length"}],
    "usage": {"prompt_tokens": 758, "completion_tokens": 12288, "total_tokens": 13046,
              "completion_tokens_details": {"reasoning_tokens": 12288}},
}


def test_finish_reason_is_carried_on_the_result() -> None:
    """Without this, "empty response" and "dead endpoint" look identical."""
    client = AsyncLLMClient()
    client._client = _CapturingHTTP(_BUDGET_EXHAUSTED)  # type: ignore[assignment]
    result = _chat(client, route="openrouter", model="deepseek/deepseek-v4-flash-0731")
    assert result.text == ""
    assert result.finish_reason == "length"


def test_agent_call_names_the_finish_reason_in_its_error() -> None:
    """The transport seam is what the nodes actually read; assert its wording."""
    from harness.profiles import ARCHITECT_PROFILE
    from llm import client as client_module
    from nodes import _transport

    client = AsyncLLMClient()
    client._client = _CapturingHTTP(_BUDGET_EXHAUSTED)  # type: ignore[assignment]
    client_module._CLIENT = client
    try:
        result = asyncio.run(
            _transport.agent_call(ARCHITECT_PROFILE, "design something", config.PROJECT_ROOT)
        )
    finally:
        client_module.reset_llm_client()

    assert not result.ok
    assert "length" in str(result.error), result.error
