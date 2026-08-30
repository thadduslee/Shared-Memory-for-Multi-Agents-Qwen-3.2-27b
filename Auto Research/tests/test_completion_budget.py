"""Uncapped completion budgets: `max_tokens=None` means OMIT, not zero.

WHY THIS FILE EXISTS. The five node profiles used to carry hard ceilings
(architect 12288, developer 16384, evaluator/judge 2048, critic 8192) and a
reply that ran past one came back truncated -- an unterminated ```json fence
that `_parse_action` could not read, or a design document that stopped
mid-sentence. The ceilings are now `None` by default and a node runs until it
emits a stop token.

The failure this fences off is the obvious implementation of that change:
setting `payload["max_tokens"] = None`. OpenAI and OpenRouter tolerate an
explicit null, but vLLM's OpenAI-compatible server validates the body against a
typed schema and 400s on a null for an `int` field -- so the Evaluator, the one
node on the vLLM route, would have been the only one to break. "Uncapped" has
to mean the same thing on all three routes, and the only spelling every
provider agrees on is the ABSENCE of the key.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import config
from llm.client import AsyncLLMClient
from harness import profiles


# ----------------------------------------------------------------------
# the policy: config._env_opt_int
# ----------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["", "none", "None", "null", "off", "0", "-1"])
def test_absent_spellings_mean_uncapped(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Zero must NOT survive as a number: it would mean 'generate nothing'."""
    monkeypatch.setenv("SOME_MAX_TOKENS", raw)
    assert config._env_opt_int("SOME_MAX_TOKENS", 4096) is None


def test_a_positive_integer_restores_a_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_MAX_TOKENS", "12288")
    assert config._env_opt_int("SOME_MAX_TOKENS", None) == 12288


def test_garbage_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOME_MAX_TOKENS", "lots")
    assert config._env_opt_int("SOME_MAX_TOKENS", 2048) == 2048


def test_unset_uses_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SOME_MAX_TOKENS", raising=False)
    assert config._env_opt_int("SOME_MAX_TOKENS", None) is None


# ----------------------------------------------------------------------
# the shipped defaults
# ----------------------------------------------------------------------


def test_every_node_profile_ships_uncapped() -> None:
    """A regression fence around the value, like test_reasoning_budget's."""
    assert set(profiles.ALL_PROFILES) == {
        "architect", "developer", "evaluator", "judge", "critic"
    }
    for name, profile in profiles.ALL_PROFILES.items():
        assert profile.max_tokens is None, f"{name} still carries a ceiling"


# ----------------------------------------------------------------------
# the wiring: the key is omitted, on every route
# ----------------------------------------------------------------------


class _CapturingHTTP:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self.payloads: list[dict[str, Any]] = []
        self._body = body

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str],
                   timeout: Any) -> "_FakeResponse":
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


def _chat(client: AsyncLLMClient, *, route: str, **kwargs: Any) -> Any:
    return asyncio.run(client.chat(route=route, model="m",
                                   messages=[{"role": "user", "content": "hi"}], **kwargs))


@pytest.mark.parametrize("route", ["openrouter", "vllm", "openai"])
def test_no_max_tokens_key_when_uncapped(route: str) -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route=route)
    assert "max_tokens" not in http.payloads[0], http.payloads[0]


@pytest.mark.parametrize("route", ["openrouter", "vllm", "openai"])
def test_an_explicit_ceiling_is_still_sent(route: str) -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route=route, max_tokens=512)
    assert http.payloads[0]["max_tokens"] == 512


def test_agent_call_omits_the_key_for_an_uncapped_profile() -> None:
    """The seam the nodes actually use, not just the client directly."""
    from llm import client as client_module
    from nodes import _transport

    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    client_module._CLIENT = client
    try:
        asyncio.run(_transport.agent_call(
            profiles.ARCHITECT_PROFILE, "design something", config.PROJECT_ROOT
        ))
    finally:
        client_module.reset_llm_client()

    assert http.payloads, "no request was made"
    assert "max_tokens" not in http.payloads[0], http.payloads[0]


# ----------------------------------------------------------------------
# the diagnosis still works with no ceiling to blame
# ----------------------------------------------------------------------


_BUDGET_EXHAUSTED = {
    "choices": [{"message": {"content": None}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 758, "completion_tokens": 99000, "total_tokens": 99758,
              "completion_tokens_details": {"reasoning_tokens": 99000}},
}


def test_an_empty_uncapped_reply_blames_the_context_window(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`finish_reason=length` on an uncapped call means the CONTEXT filled.

    Telling the reader to "raise max_tokens" there would send them to look for
    a ceiling that no longer exists.
    """
    client = AsyncLLMClient()
    client._client = _CapturingHTTP(_BUDGET_EXHAUSTED)  # type: ignore[assignment]
    with caplog.at_level("ERROR"):
        result = _chat(client, route="openrouter")
    assert result.text == ""
    assert result.finish_reason == "length"
    assert "UNCAPPED" in caplog.text
    assert "no token ceiling to raise" in caplog.text
    assert "OPENROUTER_REASONING_EFFORT" in caplog.text


# ----------------------------------------------------------------------
# ...and the CAPPED case still gets the opposite advice
# ----------------------------------------------------------------------
#
# REGRESSION (observed in a real `--preflight`, 2026-08-25). The probe in
# main.py pings each endpoint with max_tokens=16. A reasoning model cannot
# answer inside 16 tokens, so it returns empty content with
# finish_reason="length" -- and this log line fired telling the reader to lower
# OPENROUTER_REASONING_EFFORT because "on an uncapped call a token ceiling is
# not the lever", one sentence after printing "the 16-token completion budget".
# The call was capped, the ceiling WAS the lever, and on the vLLM probe the
# suggested knob does not exist. The advice has to follow the numbers.

_CAPPED_EXHAUSTED = {
    "choices": [{"message": {"content": None}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 20, "completion_tokens": 16, "total_tokens": 36,
              "completion_tokens_details": {"reasoning_tokens": 16}},
}

_CAPPED_NO_REASONING = {
    "choices": [{"message": {"content": None}, "finish_reason": "length"}],
    "usage": {"prompt_tokens": 59, "completion_tokens": 16, "total_tokens": 75},
}


def test_a_capped_reply_says_raise_the_ceiling(caplog: pytest.LogCaptureFixture) -> None:
    client = AsyncLLMClient()
    client._client = _CapturingHTTP(_CAPPED_EXHAUSTED)  # type: ignore[assignment]
    with caplog.at_level("ERROR"):
        _chat(client, route="openrouter", max_tokens=16)
    assert "16-token completion budget" in caplog.text
    assert "Raise this node's *_MAX_TOKENS" in caplog.text
    # The self-contradiction that started this: never claim a capped call was
    # uncapped, and never tell its reader a ceiling is not the lever.
    assert "UNCAPPED" not in caplog.text
    assert "no token ceiling to raise" not in caplog.text


def test_the_vllm_probe_is_not_told_to_tune_an_openrouter_knob(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """OPENROUTER_REASONING_EFFORT does nothing on the vLLM route."""
    client = AsyncLLMClient()
    client._client = _CapturingHTTP(_CAPPED_NO_REASONING)  # type: ignore[assignment]
    with caplog.at_level("ERROR"):
        _chat(client, route="vllm", max_tokens=16)
    assert "Raise this node's *_MAX_TOKENS" in caplog.text
    assert "OPENROUTER_REASONING_EFFORT" not in caplog.text


def test_no_reasoning_tokens_means_no_reasoning_advice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A model that reported zero reasoning tokens did not think its budget away."""
    client = AsyncLLMClient()
    client._client = _CapturingHTTP(_CAPPED_NO_REASONING)  # type: ignore[assignment]
    with caplog.at_level("ERROR"):
        _chat(client, route="openrouter", max_tokens=16)
    assert "OPENROUTER_REASONING_EFFORT" not in caplog.text
