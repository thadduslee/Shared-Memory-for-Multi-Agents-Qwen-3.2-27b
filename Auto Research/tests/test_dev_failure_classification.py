"""A gate that never ran is not a gate that failed.

THE FAILURE THIS PINS. run-8cf58d33b311 iteration 2 spent its entire episode --
60 turns, 1.22 million tokens, 16 minutes -- on OpenRouter transport timeouts.
The log for it is a wall of `http transport exceeded 150s`, `no action in reply
(attempt 3/4)`, and finally `turn ceiling (60) reached`. It never called
`run_tests`. It never called `sql_exec`. It wrote zero bytes.

What reached the Architect was:

    reason: developer hit the 60-turn ceiling after 1 retry/ies
            (compile_ok=True tests_ok=False migration_ok=False)
    unmet mandatory gates: tests_ok (set by `run_tests`),
                           migration_ok (set by `sql_exec`)

...under the heading "Read this as a critique of the design". `n_build_failures`
was 0, `n_loop_failures` was 0, and `last_error` was empty -- the report
contained no evidence at all, and said nothing about the transport. The
Architect duly redesigned, and the work order it produced ("remove the top_k
stop") is the change that cost the run 0.095 MGS when iteration 3 built it.

Three things had to be true for that to happen, and each has a test here:
the report could not distinguish `failed` from `never_ran`; nothing counted
transport failures separately from build failures; and the graph had no edge
that re-runs an iteration instead of redesigning it.
"""

from __future__ import annotations

import time

import config
import routers
from nodes.architect import _developer_failure_block
from nodes.developer import (
    CLASSIFY_DESIGN,
    CLASSIFY_INCONCLUSIVE,
    CLASSIFY_INFRASTRUCTURE,
    _failure_report,
    classify_failure,
    dev_observe,
    gate_status,
)


def final(**overrides):
    """A finished episode state; tests override only what they are testing."""
    base = {
        "compile_ok": False, "tests_ok": False, "migration_ok": False,
        "gate_attempts": {},
        "transport_failures": 0,
        "transport_errors": [],
        "retry_count": 0,
        "pass_rate": 0.0,
        "last_stack_trace": "",
        "failures": [],
    }
    base.update(overrides)
    return base


def report(**overrides):
    return _failure_report(
        final(**overrides), iteration=2, provenance="iter_1",
        signature="435b8549c2fbcf02",
        reason="developer hit the 60-turn ceiling after 1 retry/ies",
        files_written=[],
    )


# ======================================================================
# gate_status
# ======================================================================


def test_a_gate_that_ran_and_failed_is_reported_as_failed() -> None:
    status = gate_status(final(
        gate_attempts={"tests_ok": {"ran": 2, "ok": False}}, tests_ok=False))
    assert status["tests_ok"] == "failed"


def test_a_gate_whose_tool_was_never_called_is_reported_as_never_ran() -> None:
    """THE DISTINCTION. Both of these have `tests_ok is False`."""
    assert gate_status(final())["tests_ok"] == "never_ran"
    assert gate_status(final())["migration_ok"] == "never_ran"


def test_a_true_flag_is_itself_proof_the_tool_ran() -> None:
    """`dev_observe` is the only writer and only ever sets one on completion.

    So a state that predates `gate_attempts` -- a resumed run rehydrated from an
    older checkpointer snapshot -- is still classified correctly.
    """
    assert gate_status(final(compile_ok=True))["compile_ok"] == "passed"


async def test_dev_observe_records_the_attempt_for_every_gate_tool() -> None:
    """The provenance is captured where the observation exists, not rebuilt later."""
    update = await dev_observe({
        "action": {"tool": "run_tests", "args": {}},
        "observation": {"tool": "run_tests", "ok": False, "text": "FAILED",
                        "data": {"pass_rate": 0.5, "full_suite": True},
                        "stack_trace": "AssertionError: x", "signature": "s"},
        "gate_attempts": {},
    })
    assert update["gate_attempts"]["tests_ok"] == {"ran": 1, "ok": False}
    assert update["tests_ok"] is False


async def test_dev_observe_counts_repeated_attempts() -> None:
    update = await dev_observe({
        "action": {"tool": "compile_check", "args": {}},
        "observation": {"tool": "compile_check", "ok": True, "text": "ok",
                        "data": {}, "stack_trace": None, "signature": None},
        "gate_attempts": {"compile_ok": {"ran": 2, "ok": False}},
    })
    assert update["gate_attempts"]["compile_ok"] == {"ran": 3, "ok": True}


async def test_a_non_gate_tool_leaves_the_attempt_record_untouched() -> None:
    update = await dev_observe({
        "action": {"tool": "read_file", "args": {}},
        "observation": {"tool": "read_file", "ok": True, "text": "...",
                        "data": {}, "stack_trace": None, "signature": None},
        "gate_attempts": {},
    })
    assert "gate_attempts" not in update


# ======================================================================
# classify_failure
# ======================================================================


def test_iteration_two_of_the_real_run_is_classified_as_infrastructure() -> None:
    """Its exact shape: one charged retry, all of it transport, no gate run."""
    classification, reason = classify_failure(final(
        compile_ok=True,
        gate_attempts={"compile_ok": {"ran": 1, "ok": True}},
        transport_failures=1, retry_count=1,
    ))
    assert classification == CLASSIFY_INFRASTRUCTURE
    assert "transport" in reason


def test_a_gate_that_ran_and_failed_is_a_design_failure() -> None:
    classification, reason = classify_failure(final(
        gate_attempts={"tests_ok": {"ran": 3, "ok": False}},
        retry_count=3,
    ))
    assert classification == CLASSIFY_DESIGN
    assert "tests_ok" in reason


def test_transport_failures_do_not_overrule_a_gate_that_actually_failed() -> None:
    """One flaky turn in an episode that also found a real test failure is a
    design failure -- re-running it would just find the same red test."""
    classification, _ = classify_failure(final(
        gate_attempts={"tests_ok": {"ran": 4, "ok": False}},
        transport_failures=1, retry_count=5,
    ))
    assert classification == CLASSIFY_DESIGN


def test_an_episode_that_read_for_sixty_turns_and_built_nothing_is_inconclusive() -> None:
    """No transport failure, no gate failure, no gate ever run.

    This is a work-order problem, and it must NOT be retried: re-running the
    identical work order reproduces it exactly.
    """
    classification, reason = classify_failure(final(
        compile_ok=True, gate_attempts={"compile_ok": {"ran": 1, "ok": True}},
    ))
    assert classification == CLASSIFY_INCONCLUSIVE
    assert "`run_tests` never ran" in reason
    assert "`sql_exec` never ran" in reason


def test_all_gates_green_but_no_finish_is_still_a_design_failure() -> None:
    classification, reason = classify_failure(final(
        compile_ok=True, tests_ok=True, migration_ok=True,
        gate_attempts={g: {"ran": 1, "ok": True}
                       for g in ("compile_ok", "tests_ok", "migration_ok")},
    ))
    assert classification == CLASSIFY_DESIGN
    assert "did not finish" in reason


# ======================================================================
# the report
# ======================================================================


def test_the_report_separates_failed_gates_from_never_run_gates() -> None:
    entry = report(gate_attempts={"tests_ok": {"ran": 1, "ok": False}}, retry_count=1)
    assert entry["gates_failed"] == ["tests_ok"]
    assert entry["gates_never_ran"] == ["compile_ok", "migration_ok"]
    # ...and the old undifferentiated list is still there for anything that
    # reads it, so this is additive rather than a rename.
    assert entry["missing_gates"] == ["compile_ok", "tests_ok", "migration_ok"]


def test_the_report_carries_the_transport_errors_deduplicated() -> None:
    """A flapping endpoint emits the same string every turn."""
    entry = report(
        transport_failures=4,
        transport_errors=["http transport timed out after 150s"] * 3 + ["empty response"],
        retry_count=4,
    )
    assert entry["n_transport_failures"] == 4
    assert entry["transport_errors"] == [
        "http transport timed out after 150s", "empty response",
    ]


def test_the_report_names_its_classification_and_why() -> None:
    entry = report(compile_ok=True, gate_attempts={"compile_ok": {"ran": 1, "ok": True}},
                   transport_failures=1, retry_count=1)
    assert entry["classification"] == CLASSIFY_INFRASTRUCTURE
    assert entry["classification_reason"]


# ======================================================================
# what the Architect is told
# ======================================================================


def test_an_infrastructure_failure_tells_the_architect_not_to_redesign() -> None:
    """THE SENTENCE THAT WAS WRONG. It used to read, for every classification,
    "Read this as a critique of the design, not as a status report"."""
    block = _developer_failure_block({"dev_failure_report": report(
        compile_ok=True, gate_attempts={"compile_ok": {"ran": 1, "ok": True}},
        transport_failures=1, retry_count=1,
        transport_errors=["http transport timed out after 150s"],
    )})
    assert "DO NOT REDESIGN IN RESPONSE TO THIS" in block
    assert "Read this as a critique of the design" not in block
    assert "TRANSPORT failures, not build failures" in block
    assert "Changing the schema will not make the network work" in block
    # And the gates it never reached are named as never reached.
    assert "migration_ok (`sql_exec` was never called)" in block


def test_an_inconclusive_failure_asks_for_a_better_work_order_not_a_new_design() -> None:
    block = _developer_failure_block({"dev_failure_report": report(
        compile_ok=True, gate_attempts={"compile_ok": {"ran": 1, "ok": True}},
    )})
    assert "WORK-ORDER PROBLEM, NOT A SCHEMA PROBLEM" in block
    assert "Rewrite the WORK" in block
    assert "Read this as a critique of the design" not in block


def test_a_design_failure_still_asks_for_a_redesign() -> None:
    """The original behaviour, preserved for the one case it was right about."""
    block = _developer_failure_block({"dev_failure_report": report(
        gate_attempts={"tests_ok": {"ran": 2, "ok": False}}, retry_count=2,
        failures=[{"step": 1, "tool": "run_tests", "args": {}, "kind": "build",
                   "signature": "s", "error": "AssertionError: boom",
                   "observation": "FAILED"}],
    )})
    assert "Read this as a critique of the design" in block
    assert "Only the design can fix this" in block
    assert "gates that RAN AND FAILED: tests_ok" in block


def test_a_report_without_gate_status_still_renders() -> None:
    """A resumed run can carry a report written before this field existed."""
    legacy = {"reason": "old", "missing_gates": ["tests_ok"],
              "gate_tools": {"tests_ok": "run_tests"}, "classification": "design"}
    block = _developer_failure_block({"dev_failure_report": legacy})
    assert "unmet mandatory gates: tests_ok (set by `run_tests`)" in block


# ======================================================================
# the routing decision
# ======================================================================


def base_state(**overrides):
    state = {
        "iteration_count": 2,
        "halt_reason": "developer hit the 60-turn ceiling after 1 retry/ies",
        "infra_retry_count": 0,
        "started_at": time.monotonic(),
        "token_usage": {"total_tokens": 0},
        "dev_failure_report": {"classification": CLASSIFY_INFRASTRUCTURE,
                               "classification_reason": "1 of 1 were transport failures"},
    }
    state.update(overrides)
    return state


def test_an_infrastructure_failure_re_runs_the_same_iteration() -> None:
    """THE FIX. The old edge sent this to the Architect and got a redesign."""
    assert routers.route_after_developer(base_state()) == "retry_developer"


def test_a_design_failure_still_goes_to_the_architect() -> None:
    assert routers.route_after_developer(base_state(
        dev_failure_report={"classification": CLASSIFY_DESIGN})) == "architect"


def test_an_inconclusive_failure_goes_to_the_architect_not_back_to_the_developer() -> None:
    """Re-running an identical work order reproduces an identical stall."""
    assert routers.route_after_developer(base_state(
        dev_failure_report={"classification": CLASSIFY_INCONCLUSIVE})) == "architect"


def test_the_infrastructure_retry_budget_is_bounded() -> None:
    """A permanently dead endpoint must not spin one iteration forever."""
    spent = base_state(infra_retry_count=config.MAX_INFRA_RETRIES)
    assert routers.route_after_developer(spent) == "architect"


def test_the_retry_budget_can_be_switched_off(monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_INFRA_RETRIES", 0)
    assert routers.route_after_developer(base_state()) == "architect"


def test_the_iteration_cap_still_wins_over_an_infrastructure_retry() -> None:
    """MAX_ITERATIONS is the outermost bound and nothing may defeat it."""
    assert routers.route_after_developer(
        base_state(iteration_count=config.MAX_ITERATIONS)) == "halt"


def test_the_budget_guard_still_wins_over_an_infrastructure_retry() -> None:
    over = base_state(token_usage={"total_tokens": config.MAX_TOTAL_TOKENS + 1})
    assert routers.route_after_developer(over) == "halt"


def test_a_green_build_is_unaffected() -> None:
    assert routers.route_after_developer(
        base_state(halt_reason=None, dev_failure_report={})) == "evaluate"
