"""Token accounting off the harness event stream.

WHAT BROKE (runs_smoke/iter_1). `_usage_from_events` looked for OpenAI-style
snake_case keys, but the harness emits CAMEL CASE on `assistant/message`:

    {"inputTokens": 692, "outputTokens": 4898,
     "cacheReadTokens": 0, "reasoningTokens": 2353}

So every dsh call accounted as ZERO tokens. `MAX_TOTAL_TOKENS` was therefore
unreachable for the Architect, Developer and Critic -- a cost cap that can never
fire is not a cap -- and every node span reported `tokens=0`. The key names in
these fixtures are copied from real session logs.
"""

from __future__ import annotations

from harness.dsh_client import _usage_from_events


def usage_event(usage: dict) -> dict:
    return {"type": "assistant/message", "data": {"usage": usage}}


# The Architect's real turn from runs_smoke/iter_1.
REAL_ARCHITECT_USAGE = {
    "inputTokens": 692, "outputTokens": 4898,
    "cacheReadTokens": 0, "reasoningTokens": 2353,
}


def test_camel_case_usage_is_counted() -> None:
    assert _usage_from_events([usage_event(REAL_ARCHITECT_USAGE)]) == {
        "input_tokens": 692, "output_tokens": 4898, "total_tokens": 5590}


def test_reasoning_tokens_are_not_added_on_top_of_output() -> None:
    """They are a SUBSET of `outputTokens`, not a sibling.

    The Architect reported 4898 output and 2353 reasoning, and its captured text
    was ~2545 tokens: 2353 + 2545 = 4898 exactly. Adding them would inflate that
    node's billed output by 92% and trip the budget guard early.
    """
    totals = _usage_from_events([usage_event(REAL_ARCHITECT_USAGE)])
    assert totals["output_tokens"] == 4898


def test_cache_reads_are_not_added_to_input() -> None:
    """Providers already count cached reads inside `inputTokens`."""
    totals = _usage_from_events([usage_event(
        {"inputTokens": 100, "outputTokens": 10, "cacheReadTokens": 90})])
    assert totals["input_tokens"] == 100


def test_snake_case_usage_still_works() -> None:
    """The http-shaped payload must keep reading correctly."""
    assert _usage_from_events([usage_event(
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12})]) == {
        "input_tokens": 5, "output_tokens": 7, "total_tokens": 12}


def test_usage_is_summed_across_a_multi_step_turn() -> None:
    events = [usage_event({"inputTokens": 10, "outputTokens": 1}),
              usage_event({"inputTokens": 20, "outputTokens": 2})]
    assert _usage_from_events(events) == {
        "input_tokens": 30, "output_tokens": 3, "total_tokens": 33}


def test_an_explicit_total_is_preferred_over_the_sum() -> None:
    assert _usage_from_events([usage_event(
        {"inputTokens": 5, "outputTokens": 7, "totalTokens": 99})])["total_tokens"] == 99


def test_events_without_usage_contribute_nothing() -> None:
    events = [{"type": "turn/end", "data": {"reason": {"kind": "completed"}}},
              {"type": "assistant/chunk", "data": {"chunk": {"type": "text-delta"}}},
              usage_event({"inputTokens": 4, "outputTokens": 6})]
    assert _usage_from_events(events) == {
        "input_tokens": 4, "output_tokens": 6, "total_tokens": 10}


def test_malformed_events_and_values_do_not_raise() -> None:
    events = [None, "string", 42, {}, {"data": None}, {"data": {"usage": "nope"}},
              usage_event({"inputTokens": None, "outputTokens": "abc"}),
              usage_event({"inputTokens": 3, "outputTokens": 4})]
    assert _usage_from_events(events) == {
        "input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


def test_no_events_is_all_zeroes() -> None:
    assert _usage_from_events([]) == {
        "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
