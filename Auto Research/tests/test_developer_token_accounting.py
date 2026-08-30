"""The Developer's token spend must reach the budget guard.

WHAT BROKE. `dev_think` recorded `tokens_used` into `DeveloperState`, but the
field had no reducer -- so each turn OVERWROTE the last and a twenty-turn
episode reported the cost of one turn. `developer_node` then hardcoded
`span["tokens"] = 0` and returned no `token_usage` at all, so none of it reached
`OrchestratorState` either.

The Developer is the loop's biggest consumer: ~20 turns per iteration, each up
to `DEVELOPER_PROFILE.max_tokens`, plus every degenerate resample billed in
full. All of it was invisible to `MAX_TOTAL_TOKENS`. A cost cap that cannot see
the biggest spender is not a cap.

The accumulation moved when the micro-graph did -- `DeveloperSession._apply`
sums `tokens_used` where the LangGraph reducer used to -- so the reducer test
below and the end-to-end node test are now BOTH load-bearing: the reducer alone
no longer proves the episode adds up.
"""

from __future__ import annotations

import config
import nodes.developer as dev
from harness.dsh_client import DSHResult
from state import merge_dicts

GOOD = 'Thought: go.\n```json\n{"tool": "compile_check", "args": {}}\n```'
DEGENERATE = "<thought\nLet's read the files. response\n" * 60


def _messages():
    return [{"role": "system", "content": "dev"}, {"role": "user", "content": "work order"}]


def _state(tmp_path):
    return {"instructions": "x", "migration_sql": "", "workspace": str(tmp_path),
            "iteration": 1, "workspace_from": "template", "scratchpad": [], "retry_count": 0,
            "compile_ok": False, "tests_ok": False, "migration_ok": False, "lint_ok": False,
            "smoke_ok": False, "done": False, "pass_rate": 0.0, "last_stack_trace": None}


def test_the_reducer_sums_turns_instead_of_overwriting() -> None:
    """`tokens_used` is an accumulator, not a status."""
    total = merge_dicts({"total_tokens": 100, "input_tokens": 60},
                        {"total_tokens": 250, "input_tokens": 200})
    assert total["total_tokens"] == 350
    assert total["input_tokens"] == 260


async def test_a_degenerate_sample_is_still_billed(monkeypatch, tmp_path) -> None:
    """It ran to the token cap and the provider charged for it.

    Dropping its cost because the text was unusable under-reports the single
    most expensive thing this loop does.
    """
    replies = [(DEGENERATE, 16000), (GOOD, 40)]
    seen = {"n": 0}

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        text, tokens = replies[min(seen["n"], len(replies) - 1)]
        seen["n"] += 1
        return DSHResult(ok=True, text=text, profile=profile.name, finish_reason="stop",
                         usage={"total_tokens": tokens, "output_tokens": tokens})

    monkeypatch.setattr(dev, "agent_call", scripted)
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert turn.actions[0]["tool"] == "compile_check"
    assert turn.update["tokens_used"]["total_tokens"] == 16040, (
        "the discarded sample was billed too")


async def test_non_numeric_usage_values_do_not_raise(monkeypatch, tmp_path) -> None:
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return DSHResult(ok=True, text=GOOD, profile=profile.name, finish_reason="stop",
                         usage={"total_tokens": None, "model": "x", "flag": True,
                                "output_tokens": 7})

    monkeypatch.setattr(dev, "agent_call", scripted)
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert turn.update["tokens_used"] == {"output_tokens": 7}


def test_add_usage_ignores_booleans_and_strings() -> None:
    assert dev._add_usage({"total_tokens": 5}, {"total_tokens": 3, "cached": True, "m": "x"}) == {
        "total_tokens": 8}
    assert dev._add_usage({}, None) == {}


async def test_the_node_returns_token_usage_to_the_macro_graph(monkeypatch, tmp_path) -> None:
    """`token_usage` is the field the budget guard actually reads."""
    workspace = tmp_path / "runs" / "iter_1" / "workspace"
    workspace.mkdir(parents=True)
    (workspace / "memory_system").mkdir()
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MOCK_MODE", False)

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return DSHResult(ok=True, text=GOOD, profile=profile.name, finish_reason="stop",
                         usage={"total_tokens": 123, "input_tokens": 100, "output_tokens": 23})

    monkeypatch.setattr(dev, "agent_call", scripted)
    out = await dev.developer_node({"iteration_count": 1, "dev_instructions": "x"})

    assert "token_usage" in out, "the Developer's spend never reached the budget guard"
    assert out["token_usage"]["total_tokens"] > 0
    assert out["node_timings"][0]["tokens"] == out["token_usage"]["total_tokens"]
    assert out["node_timings"][0]["tokens"] > 0, "the span reported a hardcoded zero"


async def test_every_turn_of_an_episode_is_billed(monkeypatch, tmp_path) -> None:
    """The episode total must be the SUM of its turns, not the last one.

    This is the regression in its new home. `tokens_used` used to accumulate
    because `DeveloperState` declared a summing reducer and LangGraph applied it
    between supersteps; there are no supersteps any more, so the loop does it
    itself and this is what proves it still happens.
    """
    workspace = tmp_path / "runs" / "iter_1" / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "MOCK_MODE", False)

    # compile_check is green, so the episode keeps going until the retry budget
    # is spent on the repeat guard -- several turns, each billed 10.
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return DSHResult(ok=True, text=GOOD, profile=profile.name, finish_reason="stop",
                         usage={"total_tokens": 10, "output_tokens": 10})

    monkeypatch.setattr(dev, "agent_call", scripted)
    out = await dev.developer_node({"iteration_count": 1, "dev_instructions": "x"})

    turns = out["node_timings"][0]["turns"]
    assert turns > 1, "the episode should have taken several turns"
    assert out["token_usage"]["total_tokens"] == 10 * turns
