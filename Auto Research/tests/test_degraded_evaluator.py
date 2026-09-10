"""A dead answerer must not be able to produce a clean-looking score.

WHAT HAPPENED. `run-c993a6e93050` ran 23 iterations over three hours and spent
12.5 million tokens with the local vLLM evaluator unreachable the entire time --
7,496 `ConnectError`s in the log. Every render call failed. Every one of them
fell back to `_render_answer`'s safe default: the gated record bodies, joined,
served as the answer.

    predictions total:        1000
    answering actions:         357
    LLM-RENDERED answers:        0
    fell back to raw bodies:   357  (100.0%)

The loop then worked perfectly on the fiction. It scored, it ranked, it rolled
back, it critiqued, and it signed off with `best MGS=0.8366` -- a real number,
honestly computed, about a system nobody was trying to measure: the retrieval
and gating layer with raw evidence pasted in where the answer should be. For
comparison, `runs_real5_fixed` rendered 54 of 54 and `runs_real5_fixed2` 94 of
94, so the pipeline is perfectly capable of telling you -- it just never did.

THE FALLBACK IS RIGHT AND THE SILENCE IS NOT. For one checkpoint, serving the
gated bodies beats scoring a transport blip as a design failure. For a whole
stage it is a different experiment. These tests pin the difference.
"""

from __future__ import annotations

import time

import config
import routers
import scoreboard


def base_state(**overrides):
    state = {
        "iteration_count": 3,
        "eval_stage": "dev",
        "n_checkpoints_evaluated": 50,
        "n_worker_failures": 0,
        "render_degraded": False,
        "render_degraded_rate": 0.0,
        "n_render_degraded": 0,
        "halt_reason": None,
        "failure_signature": None,
        "started_at": time.monotonic(),
        "token_usage": {"total_tokens": 0},
    }
    state.update(overrides)
    return state


def row(iteration, mgs, *, degraded=False, stage="dev"):
    return scoreboard.score_row(
        iteration=iteration, stage=stage, phase="p",
        utility=mgs, access=0.0, forgetting=0.0, mgs=mgs, degraded=degraded,
    )


# ======================================================================
# the run stops instead of scoring fiction
# ======================================================================


def test_a_degraded_stage_halts_before_the_judge_sees_it() -> None:
    """THE 12.5-MILLION-TOKEN LESSON, as an assertion.

    Every mechanism downstream of this router works correctly on whatever it is
    handed, which is exactly why the check has to be here -- before the Judge
    turns the fallback into a score the rest of the loop will optimise against.
    """
    assert routers.route_after_collect(
        base_state(render_degraded=True, render_degraded_rate=1.0, n_render_degraded=357)
    ) == "halt"


def test_a_healthy_stage_is_scored_as_usual() -> None:
    assert routers.route_after_collect(base_state()) == "judge"


def test_the_halt_can_be_switched_off_for_a_deliberate_degraded_run(monkeypatch) -> None:
    """The numbers are labelled either way; this only decides whether to stop."""
    monkeypatch.setattr(config, "HALT_ON_DEGRADED_EVAL", False)
    assert routers.route_after_collect(base_state(render_degraded=True)) == "judge"


def test_the_halt_reason_says_what_actually_went_wrong() -> None:
    """`architect invocation failed: HTTP 402...` was the only clue the last
    run left, and it was about a different failure entirely."""
    reason = routers.halt_reason_for(
        base_state(render_degraded=True, render_degraded_rate=1.0, n_render_degraded=357)
    )
    assert "evaluator degraded" in reason
    assert "357" in reason
    assert "100%" in reason
    assert "would not describe the system under test" in reason


def test_an_empty_batch_still_takes_priority_over_the_degraded_check() -> None:
    """Nothing to score at all is the older and more specific diagnosis."""
    assert routers.route_after_collect(
        base_state(n_checkpoints_evaluated=0, render_degraded=True, iteration_count=1)
    ) == "architect"


def test_the_budget_guard_still_wins() -> None:
    assert routers.route_after_collect(base_state(
        render_degraded=True,
        token_usage={"total_tokens": config.MAX_TOTAL_TOKENS + 1},
    )) == "halt"


# ======================================================================
# a degraded score is never compared against a healthy one
# ======================================================================


def test_a_degraded_row_cannot_become_the_champion() -> None:
    """It is a real measurement of a DIFFERENT system.

    With the renderer down the answer is the raw evidence, which contains every
    required string the gated records hold -- so a degraded row tends to score
    HIGHER than a healthy one. Ranking them together would make the dead
    endpoint look like the best design the run ever produced, and hand its
    workspace to the next iteration.
    """
    history = [row(1, 0.40), row(2, 0.95, degraded=True), row(3, 0.45)]
    assert scoreboard.best_of({"score_history": history}) == (0.45, 3)
    assert scoreboard.champion_iteration({"score_history": history}, fallback=3) == 3


def test_a_degraded_row_is_still_recorded_and_reported() -> None:
    """Dropping it would hide that the run happened at all."""
    history = [row(1, 0.40), row(2, 0.95, degraded=True)]
    rows = scoreboard.rows_for_stage(history)
    assert [r["iteration"] for r in rows] == [1, 2]
    assert rows[1]["degraded"] is True


def test_a_degraded_row_gets_its_own_verdict_rather_than_a_comparison() -> None:
    history = [row(1, 0.40), row(2, 0.95, degraded=True)]
    assert scoreboard.verdict_for(history, 2) == scoreboard.VERDICT_DEGRADED


def test_a_healthy_row_is_not_compared_against_a_degraded_predecessor() -> None:
    """Iteration 3 beats every honest measurement before it, so it is a new
    best -- even though a degraded row in between scored higher."""
    history = [row(1, 0.40), row(2, 0.95, degraded=True), row(3, 0.60)]
    assert scoreboard.verdict_for(history, 3) == scoreboard.VERDICT_IMPROVED


def test_the_trend_table_marks_a_degraded_row_as_not_comparable() -> None:
    table = scoreboard.trend_table(
        {"score_history": [row(1, 0.40), row(2, 0.95, degraded=True)]}
    )
    assert "NOT COMPARABLE" in table
    assert "answerer was down" in table


def test_rows_default_to_healthy() -> None:
    """Every row written before this field existed must keep counting."""
    legacy = {"iteration": 1, "stage": "dev", "MGS": 0.5, "U": 0.5, "A": 0.0, "F": 0.0}
    assert scoreboard.best_of([legacy]) == (0.5, 1)


# ======================================================================
# the worker and the collector actually count it
# ======================================================================


def test_a_failed_render_marks_itself_degraded() -> None:
    """`_render_answer`'s fallback is correct; being silent about it was not."""
    import asyncio

    from nodes import medical_evaluator as ev

    async def dead_transport(*args, **kwargs):
        from harness.dsh_client import DSHResult

        return DSHResult(ok=False, text="", profile="evaluator",
                         finish_reason="error", error="ConnectError")

    record = {"checkpoint_id": "c1", "action": "answer",
              "answer": "joined record bodies", "used_record_ids": ["r1"],
              "evidence": [{"record_id": "r1", "text": "joined record bodies"}],
              "query_text": "q", "asker": {}}

    async def run():
        import llm

        class _Client:
            async def chat(self, **kwargs):
                from llm.client import ChatResult

                return ChatResult(text="", model="m", route="vllm",
                                  error="ConnectError: [Errno 111] Connection refused")

        original = llm.get_llm_client
        llm.get_llm_client = lambda: _Client()
        try:
            return await ev._render_answer(record, "standard_retrieval")
        finally:
            llm.get_llm_client = original

    output, _usage = asyncio.run(run())
    assert output["degraded"] is True
    # ...and it still serves the gated bodies, which is the half that was right.
    assert output["answer"] == "joined record bodies"
    assert output["action"] == "answer"
    # No `answer_structured`: that absence is what identified all 357 of
    # run-c993a6e93050's fallbacks after the fact.
    assert "answer_structured" not in output
