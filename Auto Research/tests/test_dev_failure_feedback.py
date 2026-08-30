"""The Developer's build failure, carried back to the Architect as a critique.

WHAT WAS MISSING. `routers.route_after_developer` sends an unbuildable design
back to the Architect, on the stated grounds that a build which never compiled
is evidence about the DESIGN. But the only things that travelled that edge were
`failure_signature` -- a hash -- and a `halt_reason` naming which booleans were
false, and neither appeared anywhere in the Architect's prompt. The Architect
therefore re-derived iteration N+1 from exactly the inputs it had used for
iteration N, with nothing in front of it saying the last design had been
rejected by the toolbox. The edge existed; the information did not cross it.

Worse, this is the one path with NO Critic feedback at all: a failed build never
reaches the Judge, so the Critic never runs and `critique` still describes some
earlier iteration that did build.

These tests pin both halves: the Developer records where it failed, and the
Architect is shown it.
"""

from __future__ import annotations

import config
from nodes.architect import _developer_failure_block, _failure_lines, _repeat_warning
from nodes.developer import (
    _compact_args,
    _error_line,
    _failure_history_entry,
    _failure_report,
    dev_observe,
)

TRACE = (
    "Traceback (most recent call last):\n"
    '  File "/w/tests/test_rbac.py", line 41, in test_scoped_retrieve\n'
    "    assert store.retrieve(principal, query) == []\n"
    "AssertionError: leaked 2 records to an unauthorized principal"
)


def _observation(**overrides):
    base = {
        "tool": "run_tests",
        "ok": False,
        "text": f"FAILED run_tests: {TRACE}",
        "data": {"pass_rate": 0.83},
        "stack_trace": TRACE,
        "signature": "sig-rbac",
    }
    base.update(overrides)
    return base


# ======================================================================
# the Developer records the steps that spent its retry budget
# ======================================================================


async def test_a_charged_failure_is_recorded_for_the_architect() -> None:
    """The scratchpad keeps the rendered text; only this keeps the diagnosis.

    `signature` and `stack_trace` exist on the observation and nowhere else, so
    if they are not captured in `dev_observe` they cannot be recovered later.
    """
    update = await dev_observe({
        "scratchpad": [{"ok": True}, {"ok": True}],
        "action": {"tool": "run_tests", "args": {}},
        "observation": _observation(),
        "retry_count": 0,
    })
    (entry,) = update["failures"]
    assert entry["tool"] == "run_tests"
    assert entry["kind"] == "build"
    assert entry["signature"] == "sig-rbac"
    assert entry["error"] == "AssertionError: leaked 2 records to an unauthorized principal"
    assert entry["step"] == 3


async def test_a_redirect_is_not_recorded_because_it_is_not_charged() -> None:
    """A toolbox refusing an action says nothing about whether the design builds.

    It is already exempt from `retry_count` (see the note in `dev_observe`); the
    failure record has to draw the same line or the Architect would redesign
    around the `write_file` rewrite guard.
    """
    update = await dev_observe({
        "scratchpad": [],
        "action": {"tool": "write_file", "args": {"path": "store.py"}},
        "observation": _observation(tool="write_file", redirect=True),
        "retry_count": 0,
    })
    assert "failures" not in update
    assert "retry_count" not in update


async def test_a_repeated_idempotent_call_is_tagged_as_a_loop_stall() -> None:
    """Charged, so it is recorded -- but it is not a design fault.

    Every real run in runs_multi/ produced these. Presenting them to the
    Architect as build evidence would invite a schema redesign in response to
    the model calling `compile_check` twice.
    """
    update = await dev_observe({
        "scratchpad": [],
        "action": {"tool": "compile_check", "args": {}},
        "observation": {
            "tool": "compile_check", "ok": False,
            "text": "skipped: `compile_check` already ran with these arguments",
            "data": {"skipped_duplicate": True, "repeats": 2},
            "stack_trace": None, "signature": None,
        },
        "retry_count": 0,
    })
    (entry,) = update["failures"]
    assert entry["kind"] == "loop"


async def test_a_rejected_finish_is_recorded_as_a_loop_stall() -> None:
    """Declaring victory without running the tools is an episode fault too."""
    update = await dev_observe({
        "scratchpad": [],
        "action": {"tool": "finish"},
        "observation": {"tool": "finish", "ok": True, "text": "all green"},
        "retry_count": 0,
        "compile_ok": True, "tests_ok": False, "migration_ok": False,
    })
    (entry,) = update["failures"]
    assert entry["kind"] == "loop"
    assert entry["tool"] == "finish"
    assert "tests_ok" in entry["error"] and "migration_ok" in entry["error"]


PYTEST_OUTPUT = (
    "collected 18 items\n"
    "tests/test_forgetting.py::test_tombstone_beats_authorization FAILED\n"
    "Traceback (most recent call last):\n"
    '  File "tests/test_forgetting.py", line 61, in test_tombstone_beats_authorization\n'
    "    assert decision.touched_deleted\n"
    "AssertionError: tombstone gate did not fire for an authorized clinician\n"
    "========================= 1 failed, 17 passed in 0.09s =========================\n"
)


def test_the_error_line_is_the_assertion_not_pytests_scoreboard() -> None:
    """The last line of a pytest run names no cause at all.

    `_last_error_line` (which the Developer's own status block uses) returns
    that trailing summary, and the Developer can simply re-read the output. The
    Architect cannot: this one line is the summary it changes a design against.
    """
    assert _error_line(PYTEST_OUTPUT) == (
        "AssertionError: tombstone gate did not fire for an authorized clinician")


def test_the_error_line_falls_back_to_the_last_line_when_nothing_matches() -> None:
    assert _error_line("build failed\nno exception here") == "no exception here"
    assert _error_line("") == ""
    assert _error_line("   \n  ") == ""


def test_a_file_body_argument_is_summarized_by_length() -> None:
    """A 20k-char `write_file` payload would crowd out every other failure."""
    compact = _compact_args({"args": {"path": "store.py", "content": "x" * 5000, "n": 3}})
    assert compact["path"] == "store.py"
    assert compact["content"] == "<5000 chars>"
    assert compact["n"] == 3


# ======================================================================
# the report the macro-graph carries
# ======================================================================


def _final(**overrides):
    base = {
        "compile_ok": True, "tests_ok": False, "migration_ok": False,
        "lint_ok": True, "smoke_ok": False,
        "pass_rate": 0.83,
        "retry_count": config.MAX_DEV_RETRIES,
        "last_stack_trace": TRACE,
        "failures": [
            {"step": 3, "tool": "run_tests", "args": {}, "kind": "build",
             "signature": "sig-rbac", "error": "AssertionError: leaked 2 records",
             "observation": "FAILED run_tests: ..."},
            {"step": 5, "tool": "compile_check", "args": {}, "kind": "loop",
             "signature": None, "error": "", "observation": "skipped: already ran"},
        ],
    }
    base.update(overrides)
    return base


def _report(**overrides):
    return _failure_report(
        _final(**overrides), iteration=3, provenance="iter_2",
        signature="sig-rbac", reason="developer exhausted 5 retries",
        files_written=["memory_system/store.py"],
    )


def test_the_report_names_the_unmet_gates_and_the_tool_that_sets_each() -> None:
    report = _report()
    assert report["missing_gates"] == ["tests_ok", "migration_ok"]
    assert report["gate_tools"]["tests_ok"] == "run_tests"
    assert report["exhausted"] is True


def test_advisory_gates_are_never_reported_as_unmet() -> None:
    """`smoke_ok` is false here and must not appear.

    A style or smoke finding never blocked the build (see the exit-condition
    note at the top of nodes/developer.py), so listing it as a reason the build
    failed would send the Architect after a problem that does not exist.
    """
    assert "smoke_ok" not in _report()["missing_gates"]
    assert "lint_ok" not in _report()["missing_gates"]


def test_build_and_loop_failures_are_counted_separately() -> None:
    report = _report()
    assert report["n_build_failures"] == 1
    assert report["n_loop_failures"] == 1


def test_the_history_entry_prefers_the_build_error_over_the_last_trace() -> None:
    """The last thing to fail is often the loop stall, not the real fault."""
    entry = _failure_history_entry(_report())
    assert entry["error"] == "AssertionError: leaked 2 records"
    assert entry["signature"] == "sig-rbac"
    assert entry["iteration"] == 3


# ======================================================================
# what the Architect actually sees
# ======================================================================


def test_a_green_build_produces_no_block_at_all() -> None:
    """The Developer writes `{}` on success; nothing must be rendered from it."""
    assert _developer_failure_block({"dev_failure_report": {}}) == ""
    assert _developer_failure_block({}) == ""


def test_the_block_carries_the_gates_the_error_and_the_trace() -> None:
    block = _developer_failure_block({"dev_failure_report": _report()})
    assert "COULD NOT BUILD" in block
    assert "tests_ok (set by `run_tests`)" in block
    assert "migration_ok (set by `sql_exec`)" in block
    assert "AssertionError: leaked 2 records" in block
    assert "memory_system/store.py" in block
    assert "test_scoped_retrieve" in block          # the trace
    assert "dev_failure_mitigations" in block       # what to do about it


def test_build_failures_are_ordered_ahead_of_loop_stalls() -> None:
    """The design can only be changed in response to the build failures."""
    lines = _failure_lines(_report())
    assert "run_tests" in lines[0]
    assert "compile_check" in lines[1]
    assert "loop stall" in lines[1]


def test_identical_failures_collapse_into_one_stanza_with_a_count() -> None:
    """A stuck loop re-runs the same tool until its retries are gone.

    Printed once per attempt, the same 900-character pytest report appears five
    times and crowds out every other failure. The repetition carries exactly one
    extra fact -- that nothing the Developer tried moved it -- and a count says
    that in a clause. Observed shape: the `dev_retry_exhaustion` run produces
    five byte-identical `run_tests` failures.
    """
    repeated = [
        {"step": step, "tool": "run_tests", "args": {"path": "tests"}, "kind": "build",
         "signature": "sig-rbac", "error": "AssertionError: leaked 2 records",
         "observation": "FAILED run_tests: 1 failed, 17 passed\n"
                        "AssertionError: leaked 2 records"}
        for step in (3, 4, 5, 6, 7)
    ]
    lines = _failure_lines(_report(failures=repeated))
    assert len(lines) == 1
    assert "steps 3, 4, 5, 6, 7 -- 5 attempts, identical result" in lines[0]
    assert lines[0].count("AssertionError: leaked 2 records") == 2  # summary + excerpt


def test_failures_that_differ_are_not_collapsed_together() -> None:
    """Two different errors from the same tool are two different diagnoses."""
    distinct = [
        {"step": 3, "tool": "run_tests", "args": {}, "kind": "build",
         "signature": "sig-a", "error": "AssertionError: leaked 2 records", "observation": ""},
        {"step": 5, "tool": "run_tests", "args": {}, "kind": "build",
         "signature": "sig-b", "error": "OperationalError: no such column: purpose",
         "observation": ""},
    ]
    lines = _failure_lines(_report(failures=distinct))
    assert len(lines) == 2


def test_a_first_failure_carries_no_repeat_warning() -> None:
    state = {"dev_failure_history": [{"iteration": 3, "signature": "sig-rbac"}]}
    assert _repeat_warning(state, _report()) == ""


def test_the_same_signature_twice_escalates() -> None:
    """The case the loop most needs pointed out.

    A recurring signature means the previous redesign never touched what
    actually broke -- and read in isolation, each report looks like a first
    occurrence.
    """
    state = {"dev_failure_history": [
        {"iteration": 2, "signature": "sig-rbac"},
        {"iteration": 3, "signature": "sig-rbac"},
    ]}
    warning = _repeat_warning(state, _report())
    assert "REPEATED" in warning
    assert "2 iterations (2, 3)" in warning


def test_a_different_signature_does_not_count_as_a_repeat() -> None:
    state = {"dev_failure_history": [
        {"iteration": 2, "signature": "sig-migration"},
        {"iteration": 3, "signature": "sig-rbac"},
    ]}
    assert _repeat_warning(state, _report()) == ""


# ======================================================================
# size bound -- the Architect's prompt is already large
# ======================================================================


def _twenty_distinct_failures() -> list[dict[str, object]]:
    """Twenty failures that cannot be collapsed -- each a different diagnosis."""
    return [
        {"step": i, "tool": "run_tests", "args": {}, "kind": "build",
         "signature": f"sig-{i}", "error": f"AssertionError: boom {i}",
         "observation": "x" * 4000}
        for i in range(20)
    ]


def test_the_block_stays_within_its_budget_on_a_huge_failure() -> None:
    block = _developer_failure_block({"dev_failure_report": _report(
        failures=_twenty_distinct_failures(), last_stack_trace="y" * 9000)})
    assert len(block) <= config.ARCHITECT_DEV_FAILURE_MAX_CHARS


def test_truncation_never_drops_the_instruction_or_the_gates() -> None:
    """The parts that change what the Architect DOES survive; detail is what goes."""
    block = _developer_failure_block(
        {"dev_failure_report": _report(failures=_twenty_distinct_failures())})
    assert "tests_ok (set by `run_tests`)" in block
    assert "dev_failure_mitigations" in block
    assert "omitted for length" in block
