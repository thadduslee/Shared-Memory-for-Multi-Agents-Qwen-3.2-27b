"""The Developer's exit gates, and the two ways iteration 1 of run-6b4bb1167f72
burned 4h53m without producing a build.

Both failures were loops that cost a full harness call per turn while carrying
no new information:

1.  `finish` was claimed with `migration_ok=False` because `sql_exec` had never
    run. The rejection named the flag but not the tool, so the model re-asserted
    in prose that the migration was applied until its retries ran out.
2.  Eleven consecutive turns produced no parseable action block, and the
    fallback re-anchored on `compile_check` every time, for free.

These tests pin the fixes: the rejection names the tool, an uninformative repeat
is skipped rather than replayed, and neither loop can run unbounded.
"""

from __future__ import annotations

import pytest

from nodes.developer import (
    _consecutive_fallbacks,
    _consecutive_repeats,
    dev_act,
    dev_observe,
)


def _finish_state(**gates):
    return {
        "scratchpad": [],
        "retry_count": 0,
        "thought": "all gates green",
        "action": {"tool": "finish"},
        "observation": {"tool": "finish", "ok": True, "text": "All gates verified green"},
        "compile_ok": True, "tests_ok": True, "migration_ok": True, **gates,
    }


# ======================================================================
# 1. the gate rejection names the tool, not just the flag
# ======================================================================


@pytest.mark.parametrize(
    ("flag", "tool"),
    [("compile_ok", "compile_check"), ("tests_ok", "run_tests"), ("migration_ok", "sql_exec")],
)
async def test_rejection_names_the_missing_tool(flag, tool):
    out = await dev_observe(_finish_state(**{flag: False}))

    assert out["done"] is False
    assert out["retry_count"] == 1
    rejection = out["scratchpad"][-1]["observation"]
    assert f"`{tool}`" in rejection, "the model cannot act on a flag name alone"
    assert flag in rejection


async def test_rejection_lists_every_missing_tool():
    out = await dev_observe(_finish_state(tests_ok=False, migration_ok=False))

    rejection = out["scratchpad"][-1]["observation"]
    assert "`run_tests`" in rejection
    assert "`sql_exec`" in rejection
    assert "`compile_check`" not in rejection, "a satisfied gate must not be re-demanded"


async def test_finish_survives_when_all_three_gates_are_green():
    out = await dev_observe(_finish_state())

    assert out["done"] is True
    assert out.get("retry_count", 0) == 0


# ======================================================================
# 2. an uninformative repeat is skipped, not replayed
# ======================================================================


def _repeat_state(tmp_path, pad, **gates):
    return {
        "workspace": str(tmp_path),
        "migration_sql": "",
        "iteration": 1,
        "scratchpad": pad,
        "action": {"tool": "compile_check", "args": {}},
        "compile_ok": True, "tests_ok": False, "migration_ok": False, **gates,
    }


GREEN_COMPILE = {
    "action": {"tool": "compile_check", "args": {}},
    "ok": True,
    "observation": "OK compile_check: parsed 10 file(s) cleanly",
}


async def test_repeat_is_skipped_and_steers_toward_the_unset_gates(tmp_path):
    out = await dev_act(_repeat_state(tmp_path, [GREEN_COMPILE]))

    observation = out["observation"]
    assert observation["data"]["skipped_duplicate"] is True
    assert observation["ok"] is True, "the first repeat is free steering"
    assert "`run_tests`" in observation["text"]
    assert "`sql_exec`" in observation["text"]


async def test_second_consecutive_repeat_costs_a_retry(tmp_path):
    pad = [GREEN_COMPILE, {**GREEN_COMPILE, "observation": "skipped"}]

    out = await dev_act(_repeat_state(tmp_path, pad))

    assert out["observation"]["ok"] is False, "a loop that will not progress must be bounded"


async def test_repeat_after_a_write_really_runs(tmp_path):
    """Re-checking a file you just changed is the point of the tool."""
    (tmp_path / "memory_system").mkdir()
    (tmp_path / "memory_system" / "store.py").write_text("x = 1\n", encoding="utf-8")
    pad = [GREEN_COMPILE, {"action": {"tool": "write_file", "args": {"path": "store.py"}},
                           "ok": True, "observation": "wrote"}]

    out = await dev_act(_repeat_state(tmp_path, pad))

    assert not (out["observation"].get("data") or {}).get("skipped_duplicate")


async def test_repeat_of_a_failed_call_really_runs(tmp_path):
    """After a red observation, re-running is how a fix gets confirmed."""
    (tmp_path / "memory_system").mkdir()
    (tmp_path / "memory_system" / "store.py").write_text("x = 1\n", encoding="utf-8")
    pad = [{**GREEN_COMPILE, "ok": False, "observation": "SyntaxError"}]

    out = await dev_act(_repeat_state(tmp_path, pad))

    assert not (out["observation"].get("data") or {}).get("skipped_duplicate")


def test_write_tools_are_never_suppressed():
    pad = [{"action": {"tool": "write_file", "args": {"path": "a.py"}}, "ok": True,
            "observation": "wrote"}]
    state = {"scratchpad": pad}

    # The guard only consults idempotent tools; write_file is not one of them,
    # so dev_act never asks. Assert the policy at its source.
    from nodes.developer import _IDEMPOTENT_TOOLS

    assert "write_file" not in _IDEMPOTENT_TOOLS
    assert "apply_patch" not in _IDEMPOTENT_TOOLS
    assert "finish" not in _IDEMPOTENT_TOOLS
    assert _consecutive_repeats(state, "write_file", {"path": "a.py"}) == 1


# ======================================================================
# 3. the prose-answer streak is counted, so it can be bounded
# ======================================================================


def test_fallback_streak_counts_only_the_unbroken_run():
    fallback = {"action": {"tool": "compile_check", "args": {}, "_fallback": True}, "ok": True}
    real = {"action": {"tool": "run_tests", "args": {}}, "ok": True}

    assert _consecutive_fallbacks({"scratchpad": []}) == 0
    assert _consecutive_fallbacks({"scratchpad": [fallback] * 3}) == 3
    assert _consecutive_fallbacks({"scratchpad": [fallback, fallback, real]}) == 0
    assert _consecutive_fallbacks({"scratchpad": [real, fallback]}) == 1


# ======================================================================
# The status block must never raise (regression)
# ======================================================================


def test_a_whitespace_only_stack_trace_does_not_crash_the_status_block() -> None:
    """`trace.strip().splitlines()[-1]` raises on a whitespace-only trace.

    The string is truthy so the old guard passed, but stripping it leaves ""
    and `"".splitlines()` is the empty list. This is reachable in a real run:
    `run_sandbox_smoke_test` stores the child's stderr as the trace, and a child
    that writes a single newline before failing produces exactly this. The
    IndexError would land inside `dev_think` and take down the whole graph.
    """
    from nodes.developer import _last_error_line

    for trace in (None, "", "   ", "\n", "\n\n", "\t \n  \n"):
        assert _last_error_line(trace) == ""


def test_the_last_error_line_is_the_exception_line() -> None:
    from nodes.developer import _last_error_line

    trace = 'Traceback (most recent call last):\n  File "a.py", line 1\nValueError: boom\n'
    assert _last_error_line(trace) == "ValueError: boom"


def test_the_status_block_renders_with_a_whitespace_trace(tmp_path) -> None:
    """End to end: the node-level call that would actually have crashed."""
    from nodes.dev_tools import DevToolbox
    from nodes.developer import _status_block

    box = DevToolbox(tmp_path, migration_sql="", iteration=1)
    block = _status_block({"iteration": 1, "last_stack_trace": "  \n "}, box)
    assert "last_error=" in block
    assert "</STATUS>" in block
