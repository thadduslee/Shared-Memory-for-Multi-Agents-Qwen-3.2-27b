"""The Developer drives its own loop: it calls tools, reads the real output, retries.

WHAT THIS REPLACED. `think`, `act` and `observe` used to be three LangGraph
nodes, and the graph -- not the Developer -- drove the loop. The model was handed
no tool schema at all: its persona described ten tools in prose, it emitted one
fenced JSON action, and the orchestrator executed it and re-prompted from
scratch. Two costs, both measured:

  * A tool-calling model given no tools does not fall back to prose. It emits
    its NATIVE tool-call syntax as text. runs_multi/run-280411c75b99 produced 27
    of those, parsed zero, and wrote zero bytes across four iterations.
  * Every turn started from nothing. The model never saw its own transcript,
    only a six-entry rendering of it, so it could not tell "I ran the tests and
    they failed" from "someone told me the tests failed".

These tests pin the shape of the loop that replaced it. The transcript ones are
not cosmetic: an assistant tool call with no answering `tool` message, or a
`tool` message quoting an id that was never issued, is a 400 from the provider
on the very next request -- an episode that dies of bookkeeping, not of the work.
"""

from __future__ import annotations

import json

import config
import nodes.developer as dev
from harness.dsh_client import DSHResult


def _call(name: str, call_id: str, **args) -> dict:
    return {"id": call_id, "name": name, "arguments": args}


def _native(calls: list[dict], content: str = "") -> DSHResult:
    """A reply that used the tool schema, as a real provider returns one."""
    return DSHResult(
        ok=True, text=content, profile="developer", finish_reason="completed",
        usage={"total_tokens": 10}, content=content, tool_calls=calls,
    )


def _prose(text: str) -> DSHResult:
    """A reply that did not, which is the fallback path."""
    return DSHResult(ok=True, text=text, profile="developer", finish_reason="completed",
                     usage={"total_tokens": 10}, content=text)


def _workspace(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "memory_system" / "schema.sql").write_text(
        "CREATE TABLE records (id INTEGER PRIMARY KEY);", encoding="utf-8")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8")
    return workspace


def _state(workspace, **overrides):
    return {
        "instructions": "build it", "sql_schema": "", "migration_sql": "",
        "workspace": str(workspace), "iteration": 1, "workspace_from": "template",
        "scratchpad": [], "failures": [], "retry_count": 0,
        "compile_ok": False, "tests_ok": False, "migration_ok": False,
        "lint_ok": False, "smoke_ok": False, "done": False,
        "pass_rate": 0.0, "last_stack_trace": None, **overrides,
    }


def _script(monkeypatch, replies):
    """Answer each turn from `replies`, repeating the last one forever."""
    seen = {"n": 0}
    sent: list[list[dict]] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        sent.append(list(kwargs.get("messages") or []))
        reply = replies[min(seen["n"], len(replies) - 1)]
        seen["n"] += 1
        return reply

    monkeypatch.setattr(dev, "agent_call", scripted)
    return sent


# ======================================================================
# 1. the model is actually handed its tools
# ======================================================================


async def test_the_schema_is_advertised_on_every_turn(monkeypatch, tmp_path) -> None:
    """The regression in one line: a tool-calling model with no tools."""
    advertised: list[list[str]] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        advertised.append(
            [t["function"]["name"] for t in (kwargs.get("tools") or [])])
        return _native([_call("finish", "c1", summary="done")])

    monkeypatch.setattr(dev, "agent_call", scripted)
    await dev.DeveloperSession(_state(_workspace(tmp_path))).run()

    from nodes.dev_tools import TOOL_NAMES

    assert advertised, "the model was never called"
    for names in advertised:
        assert set(names) == set(TOOL_NAMES)


async def test_a_native_tool_call_is_executed_without_parsing(monkeypatch, tmp_path) -> None:
    workspace = _workspace(tmp_path)
    _script(monkeypatch, [_native([_call("write_file", "c1",
                                         path="memory_system/new.py", content="x = 1\n")])])

    turn = await dev.dev_think(_state(workspace), [])

    assert turn.actions == [{"tool": "write_file", "_id": "c1",
                             "args": {"path": "memory_system/new.py", "content": "x = 1\n"}}]


async def test_several_tools_in_one_turn_all_run(monkeypatch, tmp_path) -> None:
    """A turn is allowed to ask for more than one thing, and each is observed."""
    workspace = _workspace(tmp_path)
    _script(monkeypatch, [
        _native([_call("compile_check", "c1"), _call("sql_exec", "c2")]),
        _native([_call("run_tests", "c3", path="tests")]),
        # An episode has to build something before `finish` is accepted, so the
        # script does what a real one does. See test_no_op_episode.py.
        _native([_call("write_file", "c4", path="memory_system/new.py", content="x = 1\n")]),
        _native([_call("finish", "c5", summary="green")]),
    ])

    final = await dev.DeveloperSession(_state(workspace)).run()

    tools = [step["action"]["tool"] for step in final["scratchpad"]]
    assert tools[:2] == ["compile_check", "sql_exec"], "both calls in the first turn ran"
    assert final["done"] is True


async def test_a_double_wrapped_arguments_object_is_unwrapped(monkeypatch, tmp_path) -> None:
    """args={"args": {...}} is the tool-calling twin of the flat-shape drift."""
    workspace = _workspace(tmp_path)
    _script(monkeypatch, [_native([{"id": "c1", "name": "read_file",
                                    "arguments": {"args": {"path": "memory_system/schema.sql"}}}])])

    turn = await dev.dev_think(_state(workspace), [])

    assert turn.actions[0]["args"] == {"path": "memory_system/schema.sql"}


# ======================================================================
# 2. the Developer sees its own work
# ======================================================================


async def test_each_tool_result_comes_back_in_the_conversation(monkeypatch, tmp_path) -> None:
    """The real output, in the model's own transcript -- not a re-rendered summary."""
    workspace = _workspace(tmp_path)
    sent = _script(monkeypatch, [
        _native([_call("compile_check", "c1")]),
        _native([_call("finish", "c2", summary="x")]),
    ])

    await dev.DeveloperSession(_state(workspace)).run()

    second_turn = sent[1]
    answers = [m for m in second_turn if m.get("role") == "tool"]
    assert answers, "the second turn was asked without the first turn's result"
    assert "compile_check" in answers[-1]["content"]
    assert answers[-1]["tool_call_id"] == "c1"


async def test_every_tool_call_is_answered_exactly_once(monkeypatch, tmp_path) -> None:
    """A stranded tool call is a 400 on the next request, not a degraded prompt."""
    workspace = _workspace(tmp_path)
    sent = _script(monkeypatch, [
        _native([_call("compile_check", "c1"), _call("list_dir", "c2", path=".")]),
        _native([_call("read_file", "c3", path="memory_system/schema.sql")]),
        _native([_call("finish", "c4", summary="x")]),
    ])

    await dev.DeveloperSession(_state(workspace)).run()

    for messages in sent:
        issued = [
            call["id"]
            for message in messages if message.get("role") == "assistant"
            for call in (message.get("tool_calls") or [])
        ]
        answered = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        assert issued == answered, f"transcript is malformed: {issued} vs {answered}"


async def test_a_prose_reply_is_answered_as_a_user_message(monkeypatch, tmp_path) -> None:
    """The fallback path must not invent a tool_call_id it was never given."""
    workspace = _workspace(tmp_path)
    sent = _script(monkeypatch, [
        _prose('Thought: compile.\n```json\n{"tool": "compile_check", "args": {}}\n```'),
        _native([_call("finish", "c9", summary="x")]),
    ])

    await dev.DeveloperSession(_state(workspace)).run()

    second_turn = sent[1]
    assert not [m for m in second_turn if m.get("role") == "tool"]
    assert "compile_check" in second_turn[-1]["content"]
    assert second_turn[-1]["role"] == "user"


async def test_the_status_block_is_restated_after_every_turn(monkeypatch, tmp_path) -> None:
    """It is the ground truth that survives pruning, so it cannot go stale."""
    workspace = _workspace(tmp_path)
    sent = _script(monkeypatch, [
        _native([_call("compile_check", "c1")]),
        _native([_call("finish", "c2", summary="x")]),
    ])

    await dev.DeveloperSession(_state(workspace)).run()

    latest = sent[1][-1]["content"]
    assert "<STATUS>" in latest
    assert "compile_ok=true" in latest, "the status must reflect what just happened"


# ======================================================================
# 3. it loops on its own failures -- which is the point
# ======================================================================


async def test_a_failed_tool_is_seen_and_retried(monkeypatch, tmp_path) -> None:
    """Fail, read the error, fix it, succeed -- without leaving the node."""
    workspace = _workspace(tmp_path)
    broken = workspace / "memory_system" / "broken.py"
    broken.write_text("def f(:\n", encoding="utf-8")

    sent = _script(monkeypatch, [
        _native([_call("compile_check", "c1")]),                       # red
        _native([_call("write_file", "c2", path="memory_system/broken.py",
                       content="def f():\n    return 1\n")]),
        _native([_call("compile_check", "c3")]),                       # green
        _native([_call("run_tests", "c4", path="tests")]),
        _native([_call("sql_exec", "c5")]),
        _native([_call("finish", "c6", summary="fixed")]),
    ])

    final = await dev.DeveloperSession(_state(workspace)).run()

    assert final["done"] is True
    assert final["compile_ok"] is True
    assert final["retry_count"] == 1, "exactly the one real failure was charged"
    # The model was shown the SyntaxError before it wrote the fix.
    before_the_fix = sent[1][-1]["content"]
    assert "SyntaxError" in before_the_fix


async def test_the_retry_budget_still_ends_the_episode(monkeypatch, tmp_path) -> None:
    """Looping on failure is the point; looping forever is not."""
    workspace = _workspace(tmp_path)
    (workspace / "memory_system" / "broken.py").write_text("def f(:\n", encoding="utf-8")
    monkeypatch.setattr(config, "MAX_DEV_RETRIES", 3)
    _script(monkeypatch, [_native([_call("compile_check", "c1")])])

    final = await dev.DeveloperSession(_state(workspace)).run()

    assert final["retry_count"] == 3
    assert final["halt_reason"] == "exhausted"
    assert not final.get("done")


# ======================================================================
# 4. the bounds that replaced `recursion_limit`
# ======================================================================


async def test_the_turn_ceiling_ends_a_loop_that_never_fails(monkeypatch, tmp_path) -> None:
    """MAX_DEV_RETRIES counts FAILURES, so green busywork is bounded by this."""
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 5)
    # `list_dir` is green, informative-looking, and moves no gate. The path
    # CHANGES every turn so the duplicate-call guard never fires -- otherwise
    # this would be testing that guard rather than the ceiling.
    paths = [".", "memory_system", "tests"]
    replies = [
        _native([_call("list_dir", f"c{i}", path=paths[i % len(paths)])])
        for i in range(10)
    ]
    session_replies = _script(monkeypatch, replies)
    assert session_replies is not None

    session = dev.DeveloperSession(_state(workspace))
    final = await session.run()

    assert session.turns == 5, "the ceiling, not something else, ended this"
    assert final["halt_reason"] == "turn_cap"
    assert int(final.get("retry_count", 0)) == 0, "nothing here failed"


async def test_the_wall_clock_ends_the_episode(monkeypatch, tmp_path) -> None:
    """`DSH_DEVELOPER_TIMEOUT_S` was documented as the outer stop and read by
    nothing at all. The loop enforces it now."""
    import time

    workspace = _workspace(tmp_path)
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(config, "DSH_DEVELOPER_TIMEOUT_S", 100.0)

    async def slow(profile, task, workdir, timeout_s=None, **kwargs):
        clock["now"] += 60.0
        return _native([_call("list_dir", f"c{clock['now']}", path=".")])

    monkeypatch.setattr(dev, "agent_call", slow)
    final = await dev.DeveloperSession(_state(workspace)).run()

    assert final["halt_reason"] == "timeout"


async def test_a_wall_clock_stop_is_not_reported_as_exhausted_retries(
    monkeypatch, tmp_path
) -> None:
    """The Architect redesigns from this text; it has to be true.

    An episode killed by the clock has retries to spare, and telling the
    Architect the build failed five times asks it to fix a problem that did not
    happen.
    """
    import time

    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MOCK_MODE", False)
    monkeypatch.setattr(config, "DSH_DEVELOPER_TIMEOUT_S", 100.0)
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    async def slow(profile, task, workdir, timeout_s=None, **kwargs):
        clock["now"] += 60.0
        return _native([_call("list_dir", f"c{clock['now']}", path=".")])

    monkeypatch.setattr(dev, "agent_call", slow)
    out = await dev.developer_node({"iteration_count": 1, "dev_instructions": "x"})

    assert "wall clock" in out["halt_reason"], out["halt_reason"]
    assert "exhausted" not in out["halt_reason"]
    assert out["dev_failure_report"]["retries_used"] < config.MAX_DEV_RETRIES


# ======================================================================
# 5. the conversation stays sendable
# ======================================================================


async def test_pruning_keeps_whole_turns(monkeypatch, tmp_path) -> None:
    """Cutting mid-turn strands a `tool` message whose call is gone."""
    workspace = _workspace(tmp_path)
    monkeypatch.setattr(dev, "_KEEP_TURNS", 2)
    monkeypatch.setattr(config, "DEVELOPER_MAX_TURNS", 6)
    paths = [".", "memory_system", "tests"]
    _script(monkeypatch, [
        _native([_call("list_dir", f"c{i}", path=paths[i % len(paths)])]) for i in range(8)
    ])

    session = dev.DeveloperSession(_state(workspace))
    await session.run()

    assert session.turns > 2, "nothing was pruned, so this asserts nothing"
    assert len(session.messages) < 2 + 2 * session.turns, "the transcript was never trimmed"
    assert session.messages[0]["role"] == "system"
    assert session.messages[1]["role"] == "user", "the work order is never pruned"
    assert session.messages[2]["role"] == "assistant", "a turn was cut in half"
    issued = [
        call["id"]
        for message in session.messages if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    ]
    answered = [m["tool_call_id"] for m in session.messages if m.get("role") == "tool"]
    assert issued == answered


async def test_the_conversation_is_json_serializable(monkeypatch, tmp_path) -> None:
    """It goes on the wire as JSON; a non-serializable value is a dead episode."""
    workspace = _workspace(tmp_path)
    _script(monkeypatch, [
        _native([_call("write_file", "c1", path="a.py", content="x = 1\n")]),
        _native([_call("finish", "c2", summary="x")]),
    ])

    session = dev.DeveloperSession(_state(workspace))
    await session.run()

    json.dumps(session.messages)


# ======================================================================
# 6. the reducers the sub-graph used to apply
# ======================================================================


def test_apply_accumulates_and_sums_like_the_reducers_did(tmp_path) -> None:
    """`scratchpad` and `failures` append; `tokens_used` sums; the rest replace.

    These were `DeveloperState`'s Annotated reducers, applied by LangGraph
    between supersteps. There are no supersteps now, so reading an update of
    `{"scratchpad": [entry]}` as "replace" would silently throw the episode away
    one step at a time.
    """
    session = dev.DeveloperSession(_state(_workspace(tmp_path)))

    session._apply({"scratchpad": [{"a": 1}], "failures": [{"f": 1}],
                    "tokens_used": {"total_tokens": 10}, "retry_count": 1})
    session._apply({"scratchpad": [{"a": 2}], "failures": [{"f": 2}],
                    "tokens_used": {"total_tokens": 5}, "retry_count": 2})

    assert session.state["scratchpad"] == [{"a": 1}, {"a": 2}]
    assert session.state["failures"] == [{"f": 1}, {"f": 2}]
    assert session.state["tokens_used"]["total_tokens"] == 15
    assert session.state["retry_count"] == 2
