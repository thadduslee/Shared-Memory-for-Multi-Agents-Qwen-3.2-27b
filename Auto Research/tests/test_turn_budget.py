"""The turn ceiling must not be a cliff the model cannot see.

WHAT HAPPENED. run-b3275eb7e373 -- the first real run with the self-correction
fixes in -- lost two of its five iterations to the 60-turn ceiling, and both
were two tool calls from a green build:

    iteration 3: 70 steps, 53 of them `read_file`. All three source files
                 written. `compile_check`, `run_tests` and `sql_exec` never
                 called even once.
    iteration 5: 69 steps, 60 of them `read_file`. `run_tests` green at
                 pass_rate 1.0, two source files written, and `compile_check`
                 and `sql_exec` never called.

Both were classified `inconclusive`, which is exactly right and was no help at
the time. The cause is that the status block the model reacts to reported

    retry=0/5

and said nothing whatsoever about turns. An episode could therefore arrive at
turn 59 with the work finished, no idea it was about to be cut off, and spend
its last turn on another `read_file`.

An episode that ends without running its gates is scored as a FAILED BUILD: it
is never evaluated, the Judge never sees it, and every byte it wrote is thrown
away. So the fix is to show the budget, and once it is nearly gone, to say
plainly that reading is now the wrong move.

NOT a bigger ceiling. Those episodes were 76% and 87% `read_file` -- they were
over-reading, not under-working, and more budget is more room to over-read.
"""

from __future__ import annotations

import config
import nodes.developer as dev
from nodes.architect import _turn_budget_note


class _Box:
    """The two attributes `_status_block` touches."""

    def __init__(self, tmp_path):
        self.workspace = tmp_path
        (tmp_path / "memory_system").mkdir(parents=True, exist_ok=True)
        (tmp_path / "memory_system" / "store.py").write_text("x = 1\n", encoding="utf-8")


def state(**overrides):
    base = {
        "iteration": 2, "workspace_from": "iter_1", "retry_count": 0,
        "compile_ok": False, "tests_ok": False, "migration_ok": False,
        "lint_ok": False, "smoke_ok": False, "last_stack_trace": None,
    }
    base.update(overrides)
    return base


# ======================================================================
# the budget is visible at all
# ======================================================================


def test_the_status_block_reports_the_turn_budget(tmp_path, monkeypatch) -> None:
    """THE OMISSION, AS AN ASSERTION. This line did not exist."""
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 60)
    block = dev._status_block(state(), _Box(tmp_path), turns_used=7)
    assert "turn=7/60 (53 left)" in block


def test_the_budget_line_sits_beside_the_retry_line(tmp_path) -> None:
    """`retry=0/5` was the only budget the model could see; now there are two."""
    block = dev._status_block(state(), _Box(tmp_path), turns_used=1)
    assert f"retry=0/{config.MAX_DEV_RETRIES}" in block
    assert "turn=1/" in block


def test_early_turns_get_the_count_and_no_nagging(tmp_path, monkeypatch) -> None:
    """A directive that fires on turn 3 is a directive the model learns to skip."""
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 60)
    monkeypatch.setattr(config, "DEVELOPER_LANDING_TURNS", 12)
    block = dev._status_block(state(), _Box(tmp_path), turns_used=5)
    assert "turn=5/60" in block
    assert "LAND IT" not in block
    assert "TURN BUDGET NEARLY SPENT" not in block


# ======================================================================
# the landing directive
# ======================================================================


def test_the_landing_directive_fires_inside_the_window(tmp_path, monkeypatch) -> None:
    """Iteration 5's exact situation, one turn before it lost the iteration."""
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 60)
    monkeypatch.setattr(config, "DEVELOPER_LANDING_TURNS", 12)
    block = dev._status_block(
        state(tests_ok=True), _Box(tmp_path), turns_used=52,
    )
    assert "TURN BUDGET NEARLY SPENT" in block
    assert "STOP READING AND STOP EDITING" in block
    # It names the FIRST gate still unset, so the instruction is a single
    # unambiguous next action rather than a list to choose from.
    assert "Call compile_check NOW" in block
    assert "`sql_exec`" in block
    assert "scored as a FAILED BUILD" in block


def test_the_directive_names_the_first_unset_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 20)
    monkeypatch.setattr(config, "DEVELOPER_LANDING_TURNS", 5)
    block = dev._status_block(
        state(compile_ok=True, tests_ok=True), _Box(tmp_path), turns_used=17,
    )
    assert "Call sql_exec NOW" in block


def test_all_gates_green_inside_the_window_says_finish(tmp_path, monkeypatch) -> None:
    """Nothing left to run, so the only useful instruction is `finish`."""
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 20)
    monkeypatch.setattr(config, "DEVELOPER_LANDING_TURNS", 5)
    block = dev._status_block(
        state(compile_ok=True, tests_ok=True, migration_ok=True),
        _Box(tmp_path), turns_used=17,
    )
    assert "LAND IT" in block
    assert "call `finish` NOW" in block


def test_the_landing_window_is_configurable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 60)
    monkeypatch.setattr(config, "DEVELOPER_LANDING_TURNS", 30)
    block = dev._status_block(state(), _Box(tmp_path), turns_used=31)
    assert "TURN BUDGET NEARLY SPENT" in block


def test_the_landing_call_is_ordered_cheapest_first() -> None:
    """No point running the suite against code that does not parse, and no
    point applying DDL against a suite that is red."""
    assert dev._landing_call(state()) == "compile_check"
    assert dev._landing_call(state(compile_ok=True)) == "run_tests"
    assert dev._landing_call(state(compile_ok=True, tests_ok=True)) == "sql_exec"
    assert dev._landing_call(state(compile_ok=True, tests_ok=True, migration_ok=True)) == ""


# ======================================================================
# the ceiling is configurable
# ======================================================================


def test_the_ceiling_is_read_from_config_at_call_time(monkeypatch) -> None:
    """It was a module constant, so a run could not lower or raise it."""
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 25)
    assert dev._max_turns() == 25


# ======================================================================
# what the Architect is told about it
# ======================================================================


def test_the_architect_is_told_how_much_of_the_budget_the_last_order_cost() -> None:
    note = _turn_budget_note({
        "dev_turns_used": 19, "dev_turn_ceiling": 60,
        "dev_instructions": "1. one\n2. two\n3. three\n4. four\n5. five",
    })
    assert "used 19/60 turns" in note
    assert "5-step work order" in note
    assert "!!" not in note, "a comfortable episode must not be flagged"


def test_an_episode_that_hit_the_ceiling_is_flagged_as_a_work_order_problem() -> None:
    """The lever the Architect actually controls is work-order SIZE."""
    note = _turn_budget_note({
        "dev_turns_used": 60, "dev_turn_ceiling": 60,
        "dev_instructions": "\n".join(f"{i}. step" for i in range(1, 11)),
    })
    assert "IT RAN OUT OF TURNS" in note
    assert "YOUR WORK ORDER WAS TOO BIG" in note
    assert "every edit it made is discarded" in note
    assert "10-step work order" in note


def test_a_near_miss_is_warned_about_before_it_becomes_a_failure() -> None:
    """43/60 is the iteration that worked and nearly did not.

    Warning only on the actual failure would teach the Architect one iteration
    too late, every time.
    """
    note = _turn_budget_note({
        "dev_turns_used": 47, "dev_turn_ceiling": 60, "dev_instructions": "1. a",
    })
    assert "78% of the budget" in note
    assert "would have hit the ceiling" in note


def test_iteration_one_is_not_handed_a_fabricated_budget() -> None:
    """There is no previous episode, so there is nothing honest to report."""
    assert _turn_budget_note({}) == ""
    assert _turn_budget_note({"dev_turns_used": 0, "dev_turn_ceiling": 60}) == ""


# ======================================================================
# it actually reaches the model, mid-episode
# ======================================================================


async def test_the_directive_reaches_the_model_before_the_ceiling(
    monkeypatch, tmp_path
) -> None:
    """THE WHOLE POINT, end to end.

    A block that renders correctly and never lands in the conversation is a
    block that changes nothing. This drives a real `DeveloperSession` whose
    model does nothing but `list_dir` -- iteration 5's shape, green busywork
    that moves no gate -- and asserts the landing directive appears in the
    messages the model was actually sent, with turns to spare.
    """
    from harness.dsh_client import DSHResult

    workspace = tmp_path / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")

    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 8)
    monkeypatch.setattr(config, "DEVELOPER_LANDING_TURNS", 3)

    sent: list[list[dict]] = []
    paths = [".", "memory_system", "tests"]
    turn = {"n": 0}

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        sent.append(list(kwargs.get("messages") or []))
        index = turn["n"]
        turn["n"] += 1
        return DSHResult(
            ok=True, text="", profile="developer", finish_reason="completed",
            usage={"total_tokens": 10}, content="",
            tool_calls=[{"id": f"c{index}", "name": "list_dir",
                         "arguments": {"path": paths[index % len(paths)]}}],
        )

    monkeypatch.setattr(dev, "agent_call", scripted)

    session = dev.DeveloperSession({
        "instructions": "build it", "sql_schema": "", "migration_sql": "",
        "workspace": str(workspace), "iteration": 1, "workspace_from": "template",
        "scratchpad": [], "failures": [], "retry_count": 0,
        "compile_ok": False, "tests_ok": False, "migration_ok": False,
        "lint_ok": False, "smoke_ok": False, "done": False,
        "pass_rate": 0.0, "last_stack_trace": None,
        "gate_attempts": {}, "transport_failures": 0, "transport_errors": [],
    })
    await session.run()

    def flatten(messages):
        return "\n".join(str(m.get("content") or "") for m in messages)

    # Early turns: the count, and no nagging.
    assert "turn=0/8" in flatten(sent[0])
    assert "TURN BUDGET NEARLY SPENT" not in flatten(sent[1])

    # The directive arrives while there are still turns left to act on it --
    # a warning delivered on the final turn is not a warning.
    warned = [i for i, messages in enumerate(sent)
              if "TURN BUDGET NEARLY SPENT" in flatten(messages)]
    assert warned, "the model was never told its budget was nearly gone"
    assert warned[0] <= config.DEVELOPER_MAX_TURNS - 2, warned
    assert "Call compile_check NOW" in flatten(sent[warned[0]])
