"""Scriptable sandbox executor for the Developer's tools.

The Developer's tools are REAL in mock mode -- `compile_check` really parses,
`run_tests` really runs pytest, `sql_exec` really applies the migration to a
scratch SQLite database.  This module exists to *inject failure* on top of
those real tools, so a reviewer can force the retry-exhaustion path without
first having to write broken code.

`MOCK_SCENARIO=dev_retry_exhaustion` makes `run_tests` report red no matter
what the suite actually does; everything else passes through untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mocks.scripted import active_scenario


@dataclass
class SandboxVerdict:
    """An override, or `None` to let the real tool result stand."""

    ok: bool
    stdout: str = ""
    stderr: str = ""
    stack_trace: str = ""


_SCRIPTED_TEST_FAILURE = (
    "============================= test session starts ==============================\n"
    "collected 18 items\n\n"
    "tests/test_forgetting.py::test_tombstone_beats_authorization FAILED\n\n"
    "=================================== FAILURES ===================================\n"
    "Traceback (most recent call last):\n"
    '  File "tests/test_forgetting.py", line 61, in test_tombstone_beats_authorization\n'
    "    assert decision.touched_deleted\n"
    "AssertionError: tombstone gate did not fire for an authorized clinician\n"
    "========================= 1 failed, 17 passed in 0.09s =========================\n"
)


def override(tool: str, iteration: int, *, real_ok: bool) -> SandboxVerdict | None:
    """Return a scripted verdict for `tool`, or None to keep the real result."""
    scenario = active_scenario()
    if tool == "run_tests" and scenario.developer_fails(iteration):
        return SandboxVerdict(
            ok=False,
            stdout=_SCRIPTED_TEST_FAILURE,
            stderr="",
            stack_trace=_SCRIPTED_TEST_FAILURE.split("=================================== FAILURES ===================================")[-1],
        )
    return None


def worker_failure(iteration: int) -> str | None:
    """The stack trace every evaluation shard should raise this iteration.

    Returns None outside the `failfast_signature` scenario.
    """
    return active_scenario().worker_stack_trace(iteration)


def scripted_scores(iteration: int, stage: str) -> dict[str, Any]:
    """The (U, A, F) the Judge should report for this round."""
    spec = active_scenario().scores(iteration, stage)
    return {
        "utility": spec.utility,
        "access": spec.access,
        "forgetting": spec.forgetting,
        "dev_pass_rate": spec.dev_pass_rate,
    }


def scripted_phase_score(iteration: int) -> float | None:
    """Force the curriculum phase score, or None to keep the measured one.

    Returned as a score rather than a boolean so it flows through the same
    `phase_score < CURRICULUM_PASS_THRESHOLD` comparison the real path uses --
    the router never learns that it is being driven by a script.
    """
    if active_scenario().curriculum_fails(iteration):
        return 0.0
    return None


def scripted_shard_tokens() -> int:
    """Tokens to charge per evaluation shard.

    The `budget_exhausted` scenario sets this high enough to trip the token
    guard within a couple of shards, so the guard is reachable in a two-second
    offline run instead of only after a real eight-hour one.
    """
    return int(active_scenario().tokens_per_shard)
