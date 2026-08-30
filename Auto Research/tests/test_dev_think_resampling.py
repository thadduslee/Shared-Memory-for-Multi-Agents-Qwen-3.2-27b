"""The Developer resamples a malfunctioning completion instead of spending a retry.

WHAT THIS IS FOR (measured against the live DEVELOPER_MODEL). Roughly half of
the model's replies to the Developer prompt degenerate into a repetition loop --
`<thought ... response` emitted until the token cap -- and never reach an action
block. Two other shapes appear: a completely empty reply, and a lone space.
None of them say anything about the work; they are sampling malfunctions, and an
immediate re-ask usually succeeds.

Charging that to MAX_DEV_RETRIES conflates two different things. That budget
means "this design cannot be built" and routes back to the Architect when spent;
retiring an episode because the decoder stuttered asks the Architect to redesign
something that was never attempted.
"""

from __future__ import annotations

import config
import nodes.developer as dev
from harness.dsh_client import DSHResult
from harness.profiles import DEVELOPER_PROFILE

# The three real degenerate shapes, verbatim in form.
REPETITION_LOOP = "<thought\nLet's read the files. response\n" * 140
EMPTY_REPLY = ""
LONE_SPACE = " "
GOOD_REPLY = 'Thought: compile.\n```json\n{"tool": "compile_check", "args": {}}\n```'


def _install(monkeypatch, replies):
    """Feed `dev_think` a scripted sequence of transport replies."""
    seen = []

    async def fake_agent_call(profile, task, workdir, timeout_s=None, **kwargs):
        text = replies[min(len(seen), len(replies) - 1)]
        seen.append(profile.temperature)
        return DSHResult(ok=bool(text.strip()), text=text, profile=profile.name,
                         finish_reason="stop", usage={"total_tokens": 10},
                         error=None if text.strip() else "empty response (stop)")

    monkeypatch.setattr(dev, "agent_call", fake_agent_call)
    return seen


def _messages():
    """The Developer's conversation, as `dev_think` now receives it."""
    return [{"role": "system", "content": "dev"}, {"role": "user", "content": "work order"}]


def _state(tmp_path):
    return {"instructions": "do the thing", "migration_sql": "", "workspace": str(tmp_path),
            "iteration": 1, "workspace_from": "template", "scratchpad": [], "retry_count": 0,
            "compile_ok": False, "tests_ok": False, "migration_ok": False, "lint_ok": False,
            "smoke_ok": False, "done": False, "pass_rate": 0.0, "last_stack_trace": None}


async def test_a_repetition_loop_is_resampled_and_recovers(monkeypatch, tmp_path) -> None:
    temperatures = _install(monkeypatch, [REPETITION_LOOP, GOOD_REPLY])
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert turn.actions[0]["tool"] == "compile_check"
    assert not turn.actions[0].get("_fallback"), "a recovered turn must not be a fallback"
    assert "retry_count" not in turn.update, "a resampled turn must NOT cost a ReAct retry"
    assert len(temperatures) == 2, "exactly one resample was needed"


async def test_an_empty_reply_is_resampled(monkeypatch, tmp_path) -> None:
    _install(monkeypatch, [EMPTY_REPLY, GOOD_REPLY])
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert turn.actions[0]["tool"] == "compile_check"
    assert "retry_count" not in turn.update


async def test_a_lone_space_reply_is_resampled(monkeypatch, tmp_path) -> None:
    _install(monkeypatch, [LONE_SPACE, GOOD_REPLY])
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert turn.actions[0]["tool"] == "compile_check"


async def test_each_resample_raises_the_temperature(monkeypatch, tmp_path) -> None:
    """A near-greedy redraw is the draw least likely to differ from the first."""
    temperatures = _install(monkeypatch, [REPETITION_LOOP, REPETITION_LOOP, GOOD_REPLY])
    await dev.dev_think(_state(tmp_path), _messages())
    assert temperatures[0] == DEVELOPER_PROFILE.temperature, "attempt 1 stays reproducible"
    assert temperatures[1] > temperatures[0]
    assert temperatures[2] > temperatures[1]


async def test_the_temperature_is_capped(monkeypatch, tmp_path) -> None:
    assert dev._warmer(DEVELOPER_PROFILE, 99).temperature == 0.8


async def test_warming_does_not_mutate_the_shared_profile() -> None:
    """The module-level profiles are frozen and shared; a copy is mandatory."""
    before = DEVELOPER_PROFILE.temperature
    warmed = dev._warmer(DEVELOPER_PROFILE, 2)
    assert DEVELOPER_PROFILE.temperature == before
    assert warmed.temperature != before
    assert warmed.name == DEVELOPER_PROFILE.name
    assert warmed.system_prompt == DEVELOPER_PROFILE.system_prompt
    assert warmed.capabilities == DEVELOPER_PROFILE.capabilities


async def test_exhausting_every_resample_falls_back_exactly_once(monkeypatch, tmp_path) -> None:
    """When every draw degenerates, the old behaviour must still apply."""
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", 2)
    temperatures = _install(monkeypatch, [REPETITION_LOOP])
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert len(temperatures) == 3, "1 initial attempt + 2 resamples"
    assert turn.actions[0] == {"tool": "compile_check", "args": {}, "_fallback": True}


async def test_resampling_can_be_switched_off(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", 0)
    temperatures = _install(monkeypatch, [REPETITION_LOOP])
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert len(temperatures) == 1
    assert turn.actions[0].get("_fallback")


async def test_a_negative_resample_setting_is_treated_as_zero(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", -5)
    temperatures = _install(monkeypatch, [REPETITION_LOOP])
    await dev.dev_think(_state(tmp_path), _messages())
    assert len(temperatures) == 1


async def test_a_good_first_reply_costs_no_extra_calls(monkeypatch, tmp_path) -> None:
    """Resampling must be free when nothing is wrong."""
    temperatures = _install(monkeypatch, [GOOD_REPLY])
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert len(temperatures) == 1
    assert temperatures[0] == DEVELOPER_PROFILE.temperature
    assert turn.actions[0]["tool"] == "compile_check"


async def test_a_native_tool_call_reply_is_accepted_without_resampling(
    monkeypatch, tmp_path
) -> None:
    """The XML/tool-call recovery path still short-circuits the resample loop."""
    temperatures = _install(monkeypatch, [
        ('<thought>read it</thought>\n<invoke name="read_file">'
         '<parameter name="path">memory_system/store.py</parameter></invoke>')])  # noqa: E501
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert len(temperatures) == 1
    assert turn.actions[0]["tool"] == "read_file"
    assert turn.actions[0]["args"]["path"] == "memory_system/store.py"


# ======================================================================
# The turn's time budget must not multiply with the resample count
# ======================================================================


def test_the_samples_share_one_think_budget() -> None:
    """Resampling must not turn a 300s turn into a 1200s one.

    MAX_DEV_RETRIES only bounds the episode if a single turn is bounded first,
    so the whole turn still costs at most DSH_DEVELOPER_THINK_TIMEOUT_S.
    """
    budget = float(config.DSH_DEVELOPER_THINK_TIMEOUT_S)
    resamples = 3
    slices = [dev._sample_timeout(i, resamples) for i in range(resamples + 1)]
    assert sum(slices) <= budget


def test_the_first_sample_gets_the_largest_slice() -> None:
    """It is the one most likely to be a legitimate long answer.

    A `write_file` action embeds a whole source file in its arguments, and the
    template's `store.py` alone is 22KB.
    """
    slices = [dev._sample_timeout(i, 3) for i in range(4)]
    assert slices[0] > max(slices[1:])


def test_disabling_resampling_preserves_the_original_contract() -> None:
    assert dev._sample_timeout(0, 0) == int(config.DSH_DEVELOPER_THINK_TIMEOUT_S)


def test_a_tiny_budget_never_starves_a_sample_below_the_floor() -> None:
    """Slicing a small budget too thin guarantees timeouts on every sample,
    which is strictly worse than not resampling."""
    assert dev._sample_timeout(1, 50) >= 45
    assert dev._sample_timeout(0, 50) >= 45


async def test_each_sample_is_called_with_its_own_slice(monkeypatch, tmp_path) -> None:
    seen: list[int] = []

    async def fake_agent_call(profile, task, workdir, timeout_s=None, **kwargs):
        seen.append(timeout_s)
        text = GOOD_REPLY if len(seen) == 3 else REPETITION_LOOP
        return DSHResult(ok=True, text=text, profile=profile.name,
                         finish_reason="stop", usage={})

    monkeypatch.setattr(dev, "agent_call", fake_agent_call)
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", 3)
    await dev.dev_think(_state(tmp_path), _messages())
    assert seen == [dev._sample_timeout(i, 3) for i in range(3)]
    assert sum(seen) <= float(config.DSH_DEVELOPER_THINK_TIMEOUT_S)


# ======================================================================
# The wall-clock deadline (the per-sample slice is not a bound on its own)
# ======================================================================


async def test_resampling_stops_once_the_think_budget_is_spent(
    monkeypatch, tmp_path
) -> None:
    """The per-sample slice alone does not bound the turn.

    On the http transport a timed-out request is retried inside
    `AsyncLLMClient.chat` up to HTTP_MAX_RETRIES times, so one "sample" can
    quietly cost several times its slice. Without a wall-clock deadline a turn
    budgeted at 300s could run for twenty minutes.
    """
    import time

    calls = []
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    async def slow_agent_call(profile, task, workdir, timeout_s=None, **kwargs):
        calls.append(timeout_s)
        clock["now"] += float(config.DSH_DEVELOPER_THINK_TIMEOUT_S)  # burn the budget
        return DSHResult(ok=True, text=REPETITION_LOOP, profile=profile.name,
                         finish_reason="stop", usage={})

    monkeypatch.setattr(dev, "agent_call", slow_agent_call)
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", 3)
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert len(calls) == 1, "the budget was gone after the first sample"
    assert turn.actions[0].get("_fallback")


async def test_a_resample_never_outlives_the_remaining_budget(
    monkeypatch, tmp_path
) -> None:
    import time

    calls: list[int] = []
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    async def agent_call(profile, task, workdir, timeout_s=None, **kwargs):
        calls.append(timeout_s)
        clock["now"] += 120.0
        return DSHResult(ok=True, text=REPETITION_LOOP, profile=profile.name,
                         finish_reason="stop", usage={})

    monkeypatch.setattr(dev, "agent_call", agent_call)
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", 3)
    await dev.dev_think(_state(tmp_path), _messages())
    budget = float(config.DSH_DEVELOPER_THINK_TIMEOUT_S)
    assert sum(calls) <= budget, f"samples asked for {sum(calls)}s of a {budget}s budget"


async def test_fast_samples_still_get_all_their_resamples(monkeypatch, tmp_path) -> None:
    """The deadline must not cost resamples when samples return quickly."""
    import time

    calls = []
    clock = {"now": 0.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["now"])

    async def quick_agent_call(profile, task, workdir, timeout_s=None, **kwargs):
        calls.append(timeout_s)
        clock["now"] += 2.0
        text = GOOD_REPLY if len(calls) == 4 else REPETITION_LOOP
        return DSHResult(ok=True, text=text, profile=profile.name,
                         finish_reason="stop", usage={})

    monkeypatch.setattr(dev, "agent_call", quick_agent_call)
    monkeypatch.setattr(config, "DEV_THINK_RESAMPLES", 3)
    turn = await dev.dev_think(_state(tmp_path), _messages())
    assert len(calls) == 4, "all three resamples should have been available"
    assert turn.actions[0]["tool"] == "compile_check"
    assert "retry_count" not in turn.update
