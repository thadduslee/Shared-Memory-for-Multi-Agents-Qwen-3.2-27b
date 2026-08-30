"""Turning the thinking chat template off on the vLLM route.

WHAT BROKE (2026-08-29, runs_3iter). "Qwen/Qwen3.8-27B" is a hybrid reasoning
model and vLLM serves its thinking chat template by default, so the Evaluator's
answer-rendering call reasoned before emitting the ```json fence. On the medical
prompts that ran past HTTP_TIMEOUT_S: the dev round burned 12 ReadTimeouts (4
attempts x 3 checkpoints, 1440s of a 1460s shard) and `_render_answer` fell back
to `answer: ""` for all three -- which were the ONLY three checkpoints in the
round whose retrieval had succeeded (8, 3 and 7 records allowed). U scored
0.0000 with a working retrieval layer, and the Critic then blamed retrieval.

This is the vLLM twin of `test_reasoning_budget.py`. The fix there was
`payload["reasoning"]`; the fix here is `payload["chat_template_kwargs"]`,
because vLLM has no `reasoning` field.

THE FIX IS NOT A `max_tokens` CEILING. The reply has to carry a COMPLETE fenced
block, and truncating it yields the same empty answer by another path -- which
is exactly what `test_completion_budget.py` fences off. Measured against the
live server: thinking on = 447 completion tokens in 16.8s; thinking off = 110
tokens in 4.2s, and the non-thinking reply was the more complete of the two.

LIKE THE REASONING CAP, IT MUST BE MODEL-SCOPED: a non-thinking model served on
the same route would reject the unknown template kwarg.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import config
from llm.client import AsyncLLMClient

# ----------------------------------------------------------------------
# the policy: config.vllm_chat_template_kwargs
# ----------------------------------------------------------------------


def test_qwen_models_get_thinking_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "VLLM_DISABLE_THINKING", True)
    monkeypatch.setattr(config, "VLLM_THINKING_MODEL_PREFIXES", ("Qwen/",))
    assert config.vllm_chat_template_kwargs("Qwen/Qwen3.8-27B") == {"enable_thinking": False}


def test_a_non_thinking_model_gets_no_template_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unknown template kwarg is a 400 from vLLM, not a silent no-op."""
    monkeypatch.setattr(config, "VLLM_DISABLE_THINKING", True)
    monkeypatch.setattr(config, "VLLM_THINKING_MODEL_PREFIXES", ("Qwen/",))
    assert config.vllm_chat_template_kwargs("meta-llama/Llama-3-70B") is None


def test_the_flag_is_the_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """VLLM_DISABLE_THINKING=0 restores the model's own default template."""
    monkeypatch.setattr(config, "VLLM_DISABLE_THINKING", False)
    assert config.vllm_chat_template_kwargs("Qwen/Qwen3.8-27B") is None


def test_default_policy_disables_thinking_for_the_evaluator() -> None:
    """The shipped default, exactly -- a regression fence around the value.

    If this reverts to None the Evaluator goes back to reasoning past the read
    timeout and recording an empty answer for every checkpoint that retrieved
    successfully, which reads downstream as a total retrieval failure.
    """
    assert config.vllm_chat_template_kwargs(config.EVALUATOR_MODEL) == {"enable_thinking": False}


# ----------------------------------------------------------------------
# the wiring: AsyncLLMClient puts the kwargs on the request
# ----------------------------------------------------------------------


class _FakeResponse:
    status_code = 200

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


class _CapturingHTTP:
    """Stands in for httpx.AsyncClient; records the payload, answers 200."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str],
                   timeout: Any) -> _FakeResponse:
        self.payloads.append(json)
        return _FakeResponse()

    async def aclose(self) -> None:  # pragma: no cover - lifecycle only
        pass


def _chat(client: AsyncLLMClient, *, route: str, model: str) -> Any:
    return asyncio.run(client.chat(route=route, model=model,
                                   messages=[{"role": "user", "content": "hi"}]))


def test_kwargs_are_sent_on_the_vllm_route() -> None:
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="vllm", model="Qwen/Qwen3.8-27B")
    assert http.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_max_tokens_stays_absent_when_uncapped() -> None:
    """The fix must not smuggle a ceiling back in; see test_completion_budget."""
    client = AsyncLLMClient()
    http = _CapturingHTTP()
    client._client = http  # type: ignore[assignment]
    _chat(client, route="vllm", model="Qwen/Qwen3.8-27B")
    assert "max_tokens" not in http.payloads[0]


def test_other_routes_never_get_the_template_kwargs() -> None:
    """OpenRouter and OpenAI would reject the unknown body key."""
    for route in ("openrouter", "openai"):
        client = AsyncLLMClient()
        http = _CapturingHTTP()
        client._client = http  # type: ignore[assignment]
        _chat(client, route=route, model="Qwen/Qwen3.8-27B")
        assert "chat_template_kwargs" not in http.payloads[0], route
