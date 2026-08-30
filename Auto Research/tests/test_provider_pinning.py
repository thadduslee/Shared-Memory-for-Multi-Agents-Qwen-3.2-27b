"""OpenRouter provider pinning.

WHAT BROKE (runs_iter3, 2026-08-25). "deepseek/deepseek-v4-flash-0731" on
OpenRouter is served by ~28 competing hosts and OpenRouter picks one per
request. Several hosts were down that evening (30-minute uptimes of 11-58%),
and every request that landed on one failed in a shape the orchestrator could
not repair: an SSE stream that emitted a single token and closed without
[DONE] -- which halted the run with "architect failed: empty response
(error)" -- or a connection that returned no bytes until the 150s wall-clock
abandon. The fix is `payload["provider"]`: name the trusted hosts, refuse the
rest.

THE PIN MUST BE MODEL-SCOPED, NOT ROUTE-SCOPED. The Judge can share the
openrouter route with "openai/gpt-4.1", which no DeepSeek host serves; a
route-wide pin with fallbacks off would 404 every Judge call. That near-miss
is why half these tests are about NOT pinning.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import config
from llm.client import AsyncLLMClient


# ----------------------------------------------------------------------
# the policy: config.openrouter_provider_preferences
# ----------------------------------------------------------------------


def test_deepseek_models_are_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENROUTER_PROVIDER_ORDER", ("DeepSeek", "Fireworks"))
    monkeypatch.setattr(config, "OPENROUTER_ALLOW_FALLBACKS", False)
    monkeypatch.setattr(config, "OPENROUTER_PIN_MODEL_PREFIXES", ("deepseek/",))
    assert config.openrouter_provider_preferences("deepseek/deepseek-v4-flash-0731") == {
        "order": ["DeepSeek", "Fireworks"],
        "allow_fallbacks": False,
    }


def test_the_judge_model_is_not_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """gpt-4.1 through OpenRouter must keep OpenRouter's own routing."""
    monkeypatch.setattr(config, "OPENROUTER_PROVIDER_ORDER", ("DeepSeek",))
    monkeypatch.setattr(config, "OPENROUTER_PIN_MODEL_PREFIXES", ("deepseek/",))
    assert config.openrouter_provider_preferences("openai/gpt-4.1") is None


def test_an_empty_order_disables_pinning(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENROUTER_PROVIDER_ORDER="" is the documented off switch."""
    monkeypatch.setattr(config, "OPENROUTER_PROVIDER_ORDER", ())
    assert config.openrouter_provider_preferences("deepseek/deepseek-v4-flash-0731") is None


def test_allow_fallbacks_is_carried_through(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "OPENROUTER_PROVIDER_ORDER", ("DeepSeek",))
    monkeypatch.setattr(config, "OPENROUTER_ALLOW_FALLBACKS", True)
    monkeypatch.setattr(config, "OPENROUTER_PIN_MODEL_PREFIXES", ("deepseek/",))
    preferences = config.openrouter_provider_preferences("deepseek/deepseek-v4-flash-0731")
    assert preferences is not None and preferences["allow_fallbacks"] is True


def test_default_policy_pins_deepseek_and_refuses_fallbacks() -> None:
    """The shipped defaults, exactly -- a regression fence around the values."""
    preferences = config.openrouter_provider_preferences("deepseek/deepseek-v4-flash-0731")
    assert preferences == {
        "order": ["DeepSeek", "Fireworks", "Novita"],
        "allow_fallbacks": False,
    }


# ----------------------------------------------------------------------
# the wiring: AsyncLLMClient puts the block on the request
# ----------------------------------------------------------------------


class _CapturingHTTP:
    """Stands in for httpx.AsyncClient; records the payload, answers 200."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str],
                   timeout: Any) -> "_FakeResponse":
        self.payloads.append(json)
        return _FakeResponse()

    async def aclose(self) -> None:  # pragma: no cover - lifecycle only
        pass


class _FakeResponse:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _chat(client: AsyncLLMClient, *, route: str, model: str) -> None:
    asyncio.run(client.chat(route=route, model=model,
                            messages=[{"role": "user", "content": "hi"}]))


def test_pin_is_sent_on_the_openrouter_route() -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="openrouter", model="deepseek/deepseek-v4-flash-0731")
    assert http.payloads[0]["provider"] == {
        "order": ["DeepSeek", "Fireworks", "Novita"],
        "allow_fallbacks": False,
    }


def test_unpinned_model_gets_no_provider_key_on_openrouter() -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="openrouter", model="openai/gpt-4.1")
    assert "provider" not in http.payloads[0]


def test_other_routes_never_get_the_provider_key() -> None:
    """vLLM would reject the unknown body key; the pin must not leak there."""
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="vllm", model="deepseek/deepseek-v4-flash-0731")
    assert "provider" not in http.payloads[0]
