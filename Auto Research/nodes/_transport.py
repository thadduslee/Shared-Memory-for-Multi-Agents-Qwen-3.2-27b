"""One call seam for the three agentic nodes (Architect, Developer, Critic).

WHY THIS EXISTS
---------------
The brief requires every node to run on the `dsh` harness, and `dsh` is the
default here. But routing OpenRouter traffic through the harness means setting
`DEEPSEEK_BASE_URL` to OpenRouter and trusting the `llm-deepseek` adapter to
treat it as a generic OpenAI-compatible endpoint. That is a documented pattern
("callers can... point those variables at a local proxy") but it is NOT verified
against OpenRouter specifically, and it is the single assumption that would take
the whole loop down if it were wrong.

So there is a second path. It is cheap to offer because of how this orchestrator
is built: the Developer drives its own loop and executes its tools in
`nodes/dev_tools.py`, so the harness is only ever asked for the next turn.
Nothing in any of the three nodes depends on dsh's own tool loop, which means
plain `/chat/completions` is a complete substitute.

    AGENT_TRANSPORT=dsh    (default)  full harness, per-node Cordis composition
    AGENT_TRANSPORT=http              direct OpenAI-compatible calls

Graph topology is identical either way; only this function changes behaviour.

TWO SHAPES OF CALL
------------------
The Architect and the Critic ask for one document and are done: one `task`
string in, one reply out. The Developer runs a CONVERSATION -- it calls tools,
reads their output, and calls more -- so it passes `messages` and `tools`
instead, and the reply carries `tool_calls` back. `tools` is only meaningful on
the http path: the dsh harness runs its own tool loop over mounted Cordis
plugins and has no seam that would accept a schema of ours, so a call that
advertises tools is routed over http regardless of configuration rather than
silently losing the toolbox.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import config
from harness.dsh_client import (
    DSHProfile,
    DSHResult,
    looks_like_tool_call_markup,
    run_dsh,
    strip_tool_call_markup,
)

log = logging.getLogger("orchestrator.transport")

def route_for(profile: DSHProfile) -> str:
    """The HTTP serving route this profile declares."""
    return profile.route


def transport_for(profile: DSHProfile, *, tools: bool = False) -> str:
    """Which transport this node actually uses.

    Per-node overrides exist because the three agentic nodes do not behave the
    same way on the same harness: see `config.DEVELOPER_TRANSPORT` for the
    measured reason the Developer defaults to `http` while the Architect and
    Critic stay on `dsh`. An empty override means "inherit AGENT_TRANSPORT",
    so the global switch still works as the single lever it was.

    `tools=True` overrides both. A tool schema has nowhere to go on the dsh
    path, and the failure mode of pretending otherwise is the silent one: the
    call would succeed, the model would be handed no tools, and it would fall
    back to emitting tool-call syntax as prose -- which is the exact regression
    the schema exists to end (see nodes/dev_tools.py).
    """
    override = str(getattr(config, f"{profile.name.upper()}_TRANSPORT", "") or "")
    configured = override or config.AGENT_TRANSPORT
    if tools and configured != "http":
        log.info("%s: tool schema requested; using the http transport instead of %r",
                 profile.name, configured)
        return "http"
    return configured


async def agent_call(
    profile: DSHProfile,
    task: str,
    workdir: Path,
    timeout_s: int | None = None,
    *,
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> DSHResult:
    """Invoke one agentic node, over whichever transport is configured.

    Returns a `DSHResult` in both cases so callers -- and their `.json_block()`
    parsing -- are transport-blind.

    `messages` replaces `task` for a caller that keeps a conversation rather than
    asking a single question; `task` is then used only as the prompt text a dsh
    fallback would need. `tools` advertises a schema and forces the http path.
    """
    timeout_s = int(timeout_s or config.DSH_DEFAULT_TIMEOUT_S)

    if transport_for(profile, tools=bool(tools)) != "http":
        return await run_dsh(profile, task, workdir, timeout_s)

    from llm import get_llm_client

    wire = messages if messages is not None else [
        {"role": "system", "content": profile.system_prompt},
        {"role": "user", "content": task},
    ]

    call = get_llm_client().chat(
        route=route_for(profile),
        model=profile.model,
        messages=wire,
        temperature=profile.temperature,
        max_tokens=profile.max_tokens,
        timeout_s=float(timeout_s),
        role=profile.name,
        tools=tools,
        tool_choice=tool_choice if tools else None,
    )
    # `asyncio.wait_for`, because `timeout_s` alone is NOT a wall-clock bound
    # here and the caller needs one. Two things defeat it:
    #
    #   1. httpx's read timeout is per-READ, not total. A response whose bytes
    #      keep arriving never trips it, so a model streaming a degenerate
    #      16384-token completion runs to completion however long that takes.
    #   2. `AsyncLLMClient.chat` retries a timed-out request internally up to
    #      HTTP_MAX_RETRIES times, so even a clean timeout costs a multiple of
    #      the budget.
    #
    # OBSERVED: a Developer think turn budgeted at 150s sat on one established
    # connection for 17 minutes at 0.1% CPU. The dsh path has always had a hard
    # timeout; this gives `timeout_s` the same meaning on both transports, which
    # is what every caller already assumes it means.
    try:
        result = await asyncio.wait_for(call, timeout=float(timeout_s))
    except asyncio.TimeoutError:
        log.warning("%s: http transport exceeded %ss; abandoning the call",
                    profile.name, timeout_s)
        return DSHResult(
            ok=False, text="", profile=profile.name, finish_reason="timeout",
            error=f"http transport timed out after {timeout_s}s",
        )
    # `.strip()` rather than truthiness, for the same reason as the dsh path:
    # a reply of one whitespace character is an empty turn, not a successful one.
    # A turn that is PURE tool calls has no prose at all and is the most
    # substantive turn there is, so it counts too.
    substantive = bool((result.text or "").strip()) or bool(result.tool_calls)
    # Name the upstream finish_reason in the error. A bare "empty response" reads
    # as a dead endpoint and sends you to check connectivity, when the actual
    # cause is usually `length` -- the model talked its whole budget away in the
    # reasoning channel and never emitted content.
    empty_error = "empty response"
    if getattr(result, "finish_reason", ""):
        empty_error = f"empty response (finish_reason={result.finish_reason})"
    return DSHResult(
        ok=result.ok and substantive,
        text=result.text,
        profile=profile.name,
        finish_reason="completed" if result.ok else "error",
        usage=result.usage,
        error=result.error or (None if substantive else empty_error),
        content=result.content,
        tool_calls=list(result.tool_calls),
    )


# ======================================================================
# Calls that owe a JSON deliverable
# ======================================================================

_REPAIR_INSTRUCTION = (
    "Your previous reply could not be used: it contained no parseable ```json "
    "block.\n\n"
    "YOU HAVE NO TOOLS ON THIS CALL. There is no file system, no `Read`, no "
    "`OpenFile`, and nothing will execute a tool call you write out -- emitting "
    "one only ends your turn with an empty deliverable. Everything you are "
    "allowed to look at is already inlined in the task above; if something you "
    "want is not there, reason from what is and say what you could not check.\n\n"
    "Reply now with the document the task asked for, ending in exactly one "
    "```json fenced block with the required keys."
)


async def agent_call_json(
    profile: DSHProfile,
    task: str,
    workdir: Path,
    timeout_s: int | None = None,
    *,
    repair_attempts: int = 1,
) -> tuple[DSHResult, dict[str, Any]]:
    """Invoke a node that owes a ```json block, and insist on actually getting one.

    Returns `(result, block)`; `block` is `{}` when nothing parseable survived.

    WHY A RETRY AND NOT A SHRUG. Both callers here already had a fallback for an
    unparseable reply -- the Architect salvages DDL out of the prose, the Critic
    has a deterministic attribution-only critique -- and both fallbacks are
    strictly worse than the document that was asked for. runs_5iter spent a
    whole iteration on one: the Architect's reply was four lines of tool-call
    markup, the salvage produced an empty work order, and the Developer read the
    workspace, ran the gates and changed nothing. One more sampled call is a
    rounding error against the cost of an iteration that builds nothing.

    A reply that is MOSTLY tool-call markup is treated as a failed reply even
    though it parsed, arrived and has a `finish_reason` of `completed`: it is a
    model waiting for a tool result that is never coming, and the useful move is
    to tell it that and ask again -- not to paste it into the next node's prompt.
    """
    result = await agent_call(profile, task, workdir, timeout_s)
    block = result.json_block() if result.ok else None

    for attempt in range(max(0, repair_attempts)):
        markup = looks_like_tool_call_markup(result.text)
        if block and not markup:
            break
        if not result.ok and not result.text:
            # Transport failure: nothing to repair, and re-asking a dead
            # endpoint just spends the caller's wall clock.
            break
        log.warning(
            "%s: reply has %s; asking again (repair %d/%d)",
            profile.name,
            "tool-call markup and no usable JSON" if markup else "no parseable json block",
            attempt + 1, max(0, repair_attempts),
        )
        prior = strip_tool_call_markup(result.text)[:2000] or "(the reply was tool-call markup only)"
        retry = await agent_call(
            profile, task, workdir, timeout_s,
            messages=[
                {"role": "system", "content": profile.system_prompt},
                {"role": "user", "content": task},
                {"role": "assistant", "content": prior},
                {"role": "user", "content": _REPAIR_INSTRUCTION},
            ],
        )
        # Keep the better of the two attempts rather than the later one: a
        # repair that comes back worse must not overwrite a usable first reply.
        retry_block = retry.json_block() if retry.ok else None
        retry.usage = _merged_usage(result.usage, retry.usage)
        if retry_block:
            return retry, retry_block
        if retry.ok and not looks_like_tool_call_markup(retry.text) and not result.ok:
            result = retry
        else:
            result.usage = retry.usage
    return result, block or {}


def _merged_usage(first: dict[str, Any], second: dict[str, Any]) -> dict[str, int]:
    """Both attempts were billed, so both are reported."""
    keys = ("input_tokens", "output_tokens", "total_tokens")
    return {k: int((first or {}).get(k, 0) or 0) + int((second or {}).get(k, 0) or 0) for k in keys}
