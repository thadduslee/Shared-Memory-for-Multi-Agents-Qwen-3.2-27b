"""A model that emits a tool call it cannot make must not stall the loop.

WHAT BROKE (runs_5iter, 2026-08-29). The Critic's task opened with "READ THE
SOURCE FIRST ... Open the files that own the mechanisms", and the Critic runs
over `AGENT_TRANSPORT=http`, which advertises no tools at all. A model told to
open a file it has no tool for writes the call out as text and stops. All three
critiques in that run were four lines of `<|DSML|>tool_calls` markup:

  * `result.ok` was True -- markup is not an empty reply,
  * `json_block()` was None, so `component`, `mechanisms` and `proposals` were
    empty for every iteration of a loop whose entire purpose is that feedback,
  * and the markup was pasted verbatim into the next Architect's prompt, where
    in iteration 3 the Architect imitated it: 181 characters of `OpenFile`, no
    design, an empty work order, and an iteration that changed zero bytes.

Three defences are pinned here: detect it, strip it before it crosses a node
boundary, and re-ask once for the deliverable that was actually requested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.dsh_client import (
    DSHResult,
    looks_like_tool_call_markup,
    strip_tool_call_markup,
)
from harness.profiles import CRITIC_PROFILE
import nodes._transport as transport

# The reply DeepSeek actually produced, byte for byte.
DSML_REPLY = (
    '<｜｜DSML｜｜tool_calls>\n'
    '<｜｜DSML｜｜invoke name="OpenFile">\n'
    '<｜｜DSML｜｜parameter name="file" string="true">memory_system/store.py'
    '</｜｜DSML｜｜parameter>\n'
    '</｜｜DSML｜｜invoke>\n'
    '</｜｜DSML｜｜tool_calls>'
)

REAL_CRITIQUE = (
    "## Critique\n"
    "`sanitize_and_decide` returns `answer_redacted` whenever anything was denied,\n"
    "which fails every utility checkpoint whose answer was otherwise complete.\n"
    '```json\n{"dominant_term": "U", "component": "agent", "proposals": []}\n```'
)


# ----------------------------------------------------------------------
# detection
# ----------------------------------------------------------------------


@pytest.mark.parametrize("artifact", [
    DSML_REPLY,
    '<tool_call>\n{"name": "read_file"}\n</tool_call>',
    '<invoke name="Read">\n<parameter name="path">store.py</parameter>\n</invoke>',
])
def test_a_reply_that_is_mostly_a_tool_call_is_recognised(artifact: str) -> None:
    assert looks_like_tool_call_markup(artifact)


@pytest.mark.parametrize("artifact", [
    REAL_CRITIQUE,
    # A document that MENTIONS the syntax is not made of it.
    "The Developer's schema advertises a tool_call surface; see `<tool_call>` in "
    "dev_tools.py. The rest of this critique is 200 characters of genuine prose "
    "about the RBAC gates, the tombstone ordering and the retrieval loop.",
    "CREATE INDEX idx ON records(patient_id, seq); -- WHERE seq <= :as_of",
    "",
])
def test_ordinary_documents_are_not_flagged(artifact: str) -> None:
    assert not looks_like_tool_call_markup(artifact)


def test_stripping_keeps_the_prose_and_drops_the_tags() -> None:
    mixed = "Here is the design.\n" + DSML_REPLY + "\nAnd here is the tradeoff."
    cleaned = strip_tool_call_markup(mixed)
    assert "Here is the design." in cleaned
    assert "And here is the tradeoff." in cleaned
    assert "DSML" not in cleaned and "invoke" not in cleaned


def test_a_real_critique_survives_stripping_unchanged() -> None:
    assert strip_tool_call_markup(REAL_CRITIQUE).strip() == REAL_CRITIQUE.strip()


# ----------------------------------------------------------------------
# the repair pass
# ----------------------------------------------------------------------


def _reply(text: str, ok: bool = True) -> DSHResult:
    return DSHResult(ok=ok, text=text, profile="critic", finish_reason="completed",
                     usage={"total_tokens": 10, "input_tokens": 7, "output_tokens": 3})


async def test_a_markup_reply_is_re_asked_and_the_repair_is_used(monkeypatch) -> None:
    calls: list[list[dict]] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        calls.append(kwargs.get("messages") or [])
        return _reply(DSML_REPLY) if len(calls) == 1 else _reply(REAL_CRITIQUE)

    monkeypatch.setattr(transport, "agent_call", scripted)
    result, block = await transport.agent_call_json(CRITIC_PROFILE, "task", Path("."))

    assert len(calls) == 2, "a reply with no JSON block must be re-asked exactly once"
    assert block.get("component") == "agent"
    assert "DSML" not in result.text
    # The repair turn has to show the model what it did and why it cannot work.
    repair = "\n".join(str(m.get("content") or "") for m in calls[1])
    assert "YOU HAVE NO TOOLS" in repair
    # Both attempts were billed, so both are reported.
    assert result.usage["total_tokens"] == 20


async def test_a_good_first_reply_is_not_re_asked(monkeypatch) -> None:
    calls: list[int] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        calls.append(1)
        return _reply(REAL_CRITIQUE)

    monkeypatch.setattr(transport, "agent_call", scripted)
    _, block = await transport.agent_call_json(CRITIC_PROFILE, "task", Path("."))

    assert len(calls) == 1, "a usable reply must not cost a second sample"
    assert block["dominant_term"] == "U"


async def test_a_repair_that_fails_too_leaves_the_caller_a_clean_empty_block(
    monkeypatch,
) -> None:
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return _reply(DSML_REPLY)

    monkeypatch.setattr(transport, "agent_call", scripted)
    result, block = await transport.agent_call_json(CRITIC_PROFILE, "task", Path("."))

    assert block == {}, "callers branch on an empty block to reach their fallback"
    assert result.usage["total_tokens"] == 20


async def test_a_dead_endpoint_is_not_re_asked(monkeypatch) -> None:
    """Repairing a transport failure just spends the caller's wall clock."""
    calls: list[int] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        calls.append(1)
        return DSHResult(ok=False, text="", profile=profile.name,
                         finish_reason="error", error="connection refused")

    monkeypatch.setattr(transport, "agent_call", scripted)
    result, block = await transport.agent_call_json(CRITIC_PROFILE, "task", Path("."))

    assert len(calls) == 1
    assert block == {} and not result.ok
