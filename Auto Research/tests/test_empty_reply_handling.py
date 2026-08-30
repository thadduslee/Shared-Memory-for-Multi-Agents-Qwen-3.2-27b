"""A whitespace-only model reply is an empty turn, not a successful one.

WHAT BROKE (runs_smoke/iter_1). A Developer turn came back as a single space
character. `bool(" ")` is True, so the transport reported a SUCCESSFUL call
carrying an unparseable reply. `dev_think` then spent one of its few retries on
the no-parseable-action fallback, when what the turn actually deserved was a
transport retry. The distinction matters because `DSH_MAX_RETRIES` covers a
flaky endpoint and `MAX_DEV_RETRIES` covers an unbuildable design; charging one
to the other hides both.
"""

from __future__ import annotations

import asyncio

import config
import nodes._transport as transport
from harness.dsh_client import DSHResult
from harness.profiles import DEVELOPER_PROFILE
from llm.client import ChatResult


def _run(coro):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


class _StubClient:
    def __init__(self, text: str) -> None:
        self.text = text

    async def chat(self, **kwargs):
        return ChatResult(text=self.text, model="m", route="openrouter",
                          usage={"total_tokens": 5})

    async def aclose(self) -> None:
        pass


def _call_over_http(text: str, tmp_path) -> DSHResult:
    import llm.client as lc

    previous_client, previous_transport = lc._CLIENT, config.AGENT_TRANSPORT
    lc._CLIENT = _StubClient(text)
    config.AGENT_TRANSPORT = "http"
    try:
        return _run(transport.agent_call(DEVELOPER_PROFILE, "task", tmp_path))
    finally:
        lc._CLIENT, config.AGENT_TRANSPORT = previous_client, previous_transport


def test_a_single_space_reply_is_not_ok(tmp_path) -> None:
    result = _call_over_http(" ", tmp_path)
    assert not result.ok
    assert result.error == "empty response"


def test_a_newline_only_reply_is_not_ok(tmp_path) -> None:
    assert not _call_over_http("\n\n\t  \n", tmp_path).ok


def test_a_real_reply_is_still_ok(tmp_path) -> None:
    result = _call_over_http('```json\n{"tool": "finish", "args": {}}\n```', tmp_path)
    assert result.ok
    assert result.error is None
    assert result.json_block() == {"tool": "finish", "args": {}}


def test_leading_whitespace_does_not_disqualify_a_real_reply(tmp_path) -> None:
    """The live failure's sibling: real content that merely opens with a space."""
    result = _call_over_http("   \n\nThought: proceed.\n", tmp_path)
    assert result.ok


# ======================================================================
# `timeout_s` must be a real wall-clock bound on the http transport
# ======================================================================


def test_a_hanging_http_call_is_abandoned_at_the_timeout(tmp_path) -> None:
    """OBSERVED: a think turn budgeted at 150s held one connection for 17 minutes.

    Two things defeat `timeout_s` on its own: httpx's read timeout is per-READ
    rather than total, so a response whose bytes keep arriving never trips it;
    and `AsyncLLMClient.chat` retries a timed-out request internally up to
    HTTP_MAX_RETRIES times. The dsh path has always had a hard timeout -- this
    pins that `timeout_s` means the same thing on both transports.
    """
    import llm.client as lc

    class _HangingClient:
        async def chat(self, **kwargs):
            await asyncio.sleep(30)  # never returns within the budget
            raise AssertionError("should have been abandoned")

        async def aclose(self) -> None:
            pass

    previous_client, previous_transport = lc._CLIENT, config.AGENT_TRANSPORT
    lc._CLIENT = _HangingClient()
    config.AGENT_TRANSPORT = "http"
    try:
        result = _run(transport.agent_call(DEVELOPER_PROFILE, "task", tmp_path, timeout_s=1))
    finally:
        lc._CLIENT, config.AGENT_TRANSPORT = previous_client, previous_transport

    assert not result.ok
    assert result.finish_reason == "timeout"
    assert "timed out after 1s" in (result.error or "")


def test_a_prompt_reply_is_unaffected_by_the_wall_clock_bound(tmp_path) -> None:
    result = _call_over_http('```json\n{"tool": "finish", "args": {}}\n```', tmp_path)
    assert result.ok
    assert result.finish_reason == "completed"
