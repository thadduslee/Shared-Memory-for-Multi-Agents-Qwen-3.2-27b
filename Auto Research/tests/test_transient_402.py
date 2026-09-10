"""One rate limit must not end a 23-iteration run.

WHAT HAPPENED. `run-c993a6e93050` had been going for three hours and 12.5
million tokens when its 23rd Architect call came back with a single HTTP 402:

    {"error":{"message":"This request would exceed your available credits given
     your current in-flight requests. Retry after in-flight requests settle, or
     add credits.","code":402,"metadata":{"reason":"in_flight_budget_exhausted",
     ... "headers":{"Retry-After":"120"}}}}

The provider said, in the response body, exactly when to come back. Two things
between it and the run:

  1. `llm/client.RETRYABLE_STATUS` did not contain 402, so the client returned
     an error on the first attempt without waiting at all. 402 is two different
     errors sharing a status code -- "you are out of money" (terminal) and "too
     many requests in flight for your balance" (transient) -- and the client
     treated both as the first.
  2. `route_after_architect` read "the Architect only fails fatally; there is
     nothing to build without it" as fatal to the RUN rather than to the
     ITERATION, for every cause including a two-minute wait.

Waiting 120 seconds would have saved the run. These tests pin both layers.
"""

from __future__ import annotations

import time

import config
import routers
from llm.client import RETRYABLE_STATUS, AsyncLLMClient, _is_transient_payment_required

# ======================================================================
# layer 1: the client tells the two 402s apart
# ======================================================================


IN_FLIGHT_BODY = (
    '{"error":{"message":"This request would exceed your available credits given '
    'your current in-flight requests. Retry after in-flight requests settle, or '
    'add credits.","code":402,"metadata":{"reason":"in_flight_budget_exhausted"}}}'
)
OUT_OF_MONEY_BODY = (
    '{"error":{"message":"Insufficient credits. Add more at '
    'https://openrouter.ai/settings/credits","code":402}}'
)


def test_the_exact_402_that_ended_the_run_is_retryable() -> None:
    assert _is_transient_payment_required(402, IN_FLIGHT_BODY, None) is True


def test_a_402_with_an_explicit_retry_after_is_retryable() -> None:
    """A provider with no credit left has no reason to say WHEN to return."""
    assert _is_transient_payment_required(402, OUT_OF_MONEY_BODY, "120") is True


def test_a_plain_out_of_credits_402_is_terminal() -> None:
    """Retrying this spends the wall clock to be told the same thing."""
    assert _is_transient_payment_required(402, OUT_OF_MONEY_BODY, None) is False


def test_other_statuses_are_not_affected() -> None:
    assert _is_transient_payment_required(403, IN_FLIGHT_BODY, "120") is False
    assert _is_transient_payment_required(200, "", None) is False


def test_402_is_still_not_blanket_retryable() -> None:
    """The status set stays as it was; only the classified case is added.

    A blanket entry would make a genuinely empty account retry four times on
    every call for the rest of the run.
    """
    assert 402 not in RETRYABLE_STATUS


# ======================================================================
# ...and honours the stated wait rather than its own guess
# ======================================================================


def test_a_stated_retry_after_is_honoured_in_full() -> None:
    """`Retry-After: 120` clamped to the 30s guess-ceiling meant every retry
    arrived while the same requests were still in flight."""
    assert AsyncLLMClient._backoff_delay(0, "120", max_delay=180.0) == 120.0


def test_the_ordinary_backoff_ceiling_is_unchanged() -> None:
    """Only the explicitly-stated wait gets the longer ceiling."""
    assert AsyncLLMClient._backoff_delay(0, "120") == config.HTTP_BACKOFF_MAX_S


def test_an_absurd_retry_after_is_still_bounded() -> None:
    assert AsyncLLMClient._backoff_delay(0, "99999", max_delay=180.0) == 180.0


def test_a_malformed_retry_after_falls_back_to_jitter() -> None:
    delay = AsyncLLMClient._backoff_delay(0, "soon", max_delay=180.0)
    assert 0.0 <= delay <= config.HTTP_BACKOFF_MAX_S


# ======================================================================
# layer 2: the router retries the iteration instead of ending the run
# ======================================================================


def architect_state(reason, **overrides):
    state = {
        "iteration_count": 22,
        "halt_reason": reason,
        "architect_retry_count": 0,
        "started_at": time.monotonic(),
        "token_usage": {"total_tokens": 0},
    }
    state.update(overrides)
    return state


THE_REAL_FAILURE = (
    'architect invocation failed: HTTP 402: {"error":{"message":"This request '
    'would exceed your available credits given your current in-flight requests. '
    'Retry after in-flight requests settle, or add credits.","code":402,'
    '"metadata":{"reason":"in_flight_budget_exhausted"'
)


def test_the_run_ending_failure_now_retries_instead() -> None:
    """THE FIX, on the verbatim `halt_reason` that ended the run."""
    assert routers.route_after_architect(architect_state(THE_REAL_FAILURE)) == "retry_architect"


def test_timeouts_and_connection_errors_retry_too() -> None:
    for reason in (
        "architect invocation failed: http transport timed out after 900s",
        "architect invocation failed: ConnectError: connection refused",
        "architect invocation failed: empty response (finish_reason=length)",
        "architect invocation failed: HTTP 503: upstream unavailable",
    ):
        assert routers.route_after_architect(architect_state(reason)) == "retry_architect", reason


def test_a_non_transport_architect_failure_still_halts() -> None:
    """A design the model could not produce is not fixed by asking again."""
    assert routers.route_after_architect(
        architect_state("architect invocation failed: HTTP 400: bad request")
    ) == "halt"


def test_an_unrelated_halt_reason_is_untouched() -> None:
    """Only the Architect's OWN failure qualifies; a curriculum halt does not."""
    assert routers.route_after_architect(
        architect_state("curriculum phase failed")
    ) == "halt"


def test_the_architect_retry_budget_is_bounded() -> None:
    """A permanently dead provider must not spin one iteration forever."""
    assert routers.route_after_architect(architect_state(
        THE_REAL_FAILURE, architect_retry_count=config.MAX_INFRA_RETRIES,
    )) == "halt"


def test_the_budget_guard_still_wins_over_an_architect_retry() -> None:
    assert routers.route_after_architect(architect_state(
        THE_REAL_FAILURE,
        token_usage={"total_tokens": config.MAX_TOTAL_TOKENS + 1},
    )) == "halt"


def test_a_healthy_architect_goes_to_the_developer() -> None:
    assert routers.route_after_architect(architect_state(None)) == "developer"


# ======================================================================
# the counters do not clear each other
# ======================================================================


def test_the_architect_and_developer_retry_budgets_are_separate() -> None:
    """`architect_node` resets `infra_retry_count` at the top of its own turn.

    Sharing one counter would mean an Architect retry clearing its own budget on
    every attempt and looping until the wall clock -- so the Architect's budget
    is reset by the DEVELOPER instead, once an iteration has got as far as a
    build. This asserts the two are genuinely distinct keys.
    """
    state = architect_state(THE_REAL_FAILURE, infra_retry_count=config.MAX_INFRA_RETRIES)
    assert routers.route_after_architect(state) == "retry_architect"
