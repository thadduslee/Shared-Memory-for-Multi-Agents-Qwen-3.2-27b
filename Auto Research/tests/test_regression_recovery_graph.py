"""End to end: a collapsing run, driven through the REAL compiled graph.

`test_scoreboard.py` proves the arithmetic. `test_champion_lineage.py` proves
the file copy. `test_regression_verdict.py` proves the rendered text. None of
those would catch the mistake this file exists for: a mechanism that is correct
in every part and wired to the wrong node, or wired correctly and never reached
because some other guard fires first.

THE SCENARIO IS run-8cf58d33b311. `MOCK_SCENARIO=regression` scripts its real
dev-slice scores -- 0.3172, 0.2222, 0.1830, 0.1190 -- and asserts the loop does
now what it could not do then: notice, roll back, say so, and report the truth
at the end.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import config
import scoreboard


@pytest.fixture(autouse=True)
def isolated_run(tmp_path, monkeypatch):
    """Fresh artifacts and fresh process-level singletons per test.

    Identical to `tests/test_graph_paths.py`'s fixture and duplicated rather
    than shared for the same reason that file gives: the orchestrator caches the
    dataset, the harness client and the circuit breakers at module level, which
    is right for a real run and poisonous for a suite that switches scenarios
    between cases.
    """
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "DSH_CORDIS_DIR", tmp_path / "cordis")
    monkeypatch.setattr(config, "MOCK_MODE", True)

    import harness.dsh_client as dsh
    import llm.client as llm_client
    import nodes.medical_evaluator as evaluator
    import websearch

    dsh.reset_dsh_client()
    llm_client.reset_llm_client()
    evaluator.reset_dataset()
    evaluator.reset_breakers()
    evaluator._COUNT_CHECKED = False
    websearch.reset_search_client()
    yield
    dsh.reset_dsh_client()
    evaluator.reset_dataset()
    evaluator.reset_breakers()


async def run_regression(monkeypatch, iterations: int = 4, **overrides) -> dict:
    from graph import build_graph
    from state import initial_state

    monkeypatch.setattr(config, "MOCK_SCENARIO", "regression")
    monkeypatch.setattr(config, "MAX_ITERATIONS", iterations)
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)

    graph = build_graph()
    state = initial_state(
        workspace=str(config.iteration_dir(1) / "workspace"), started_at=time.monotonic()
    )
    return await graph.ainvoke(state, config={"recursion_limit": config.RECURSION_LIMIT})


def spans(final: dict, node: str) -> list[dict]:
    return [s for s in final.get("node_timings") or [] if s.get("node") == node]


# ======================================================================
# the history exists at all
# ======================================================================


async def test_every_judged_iteration_leaves_a_score_row(monkeypatch) -> None:
    final = await run_regression(monkeypatch)
    rows = scoreboard.rows_for_stage(final["score_history"])
    assert [r["iteration"] for r in rows] == [1, 2, 3, 4]
    assert [round(r["MGS"], 4) for r in rows] == [0.3172, 0.2222, 0.1830, 0.1190]


async def test_each_row_records_the_workspace_that_produced_it(monkeypatch) -> None:
    """This is what makes a rollback executable rather than reconstructed."""
    final = await run_regression(monkeypatch)
    for row in scoreboard.rows_for_stage(final["score_history"]):
        assert row["workspace"], row
        assert Path(row["workspace"]).is_dir(), row


# ======================================================================
# the rollback actually happens, on disk
# ======================================================================


async def test_the_workspace_rolls_back_to_the_champion(monkeypatch) -> None:
    """THE HEADLINE. Iterations 3 and 4 inherit iteration 1, not their predecessor.

    Under the old rule every one of them inherited the iteration just measured
    as worse, which is how a run loses 0.2 MGS in four steps while every design
    predicts a rise.
    """
    final = await run_regression(monkeypatch)
    provenance = {s["iteration"]: s.get("workspace_from") for s in spans(final, "developer")}
    assert provenance[1] == "template"
    assert provenance[2] == "iter_1"          # iteration 1 was still the best
    assert provenance[3] == "iter_1", provenance   # NOT iter_2
    assert provenance[4] == "iter_1", provenance   # NOT iter_3


async def test_the_rollback_is_flagged_on_the_span(monkeypatch) -> None:
    """`node_timings` is the run's audit trail; a silent rollback is unauditable."""
    final = await run_regression(monkeypatch)
    rolled = {s["iteration"]: s.get("rolled_back") for s in spans(final, "developer")}
    assert rolled[2] is False
    assert rolled[3] is True
    assert rolled[4] is True


async def test_rollback_can_be_disabled_and_the_old_lineage_returns(monkeypatch) -> None:
    """The knob is real, so "always inherit N-1" stays available as an experiment."""
    final = await run_regression(monkeypatch, ROLLBACK_TO_BEST=False)
    provenance = {s["iteration"]: s.get("workspace_from") for s in spans(final, "developer")}
    assert provenance[3] == "iter_2"
    assert provenance[4] == "iter_3"


# ======================================================================
# the Architect is told
# ======================================================================


async def test_the_architect_prompt_carries_the_trend_and_the_rollback(monkeypatch) -> None:
    """Wired end to end: the Judge's rows reach the Architect's task text."""
    tasks: list[tuple[str, str]] = []
    from nodes import _transport as transport

    real = transport.agent_call

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        tasks.append((profile.name, task))
        return await real(profile, task, workdir, timeout_s, **kwargs)

    from nodes import developer

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)

    await run_regression(monkeypatch)

    architect_tasks = [task for name, task in tasks if name == "architect"]
    assert len(architect_tasks) >= 4

    # Iteration 1 has nothing to show yet and must not pretend otherwise.
    assert "nothing has been judged yet" in architect_tasks[0]

    # By iteration 4 the whole trajectory is in front of it, with the verdict.
    fourth = architect_tasks[3]
    assert "MEASURED PERFORMANCE -- EVERY ITERATION" in fourth
    assert "0.3172" in fourth and "0.1830" in fourth
    assert "BEST SO FAR: iteration 1" in fourth
    assert "REGRESSION" in fourth
    assert "ITERATIONS IN A ROW" in fourth
    # ...and it is told its workspace was rolled back, so it does not spend the
    # iteration writing a work order to revert changes that are already gone.
    assert "THE LOSING CHANGES ARE ALREADY GONE" in fourth
    assert "rolled back to iteration 1's code" in fourth


async def test_the_critic_prompt_leads_with_the_regression(monkeypatch) -> None:
    tasks: list[tuple[str, str]] = []
    from nodes import _transport as transport

    real = transport.agent_call

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        tasks.append((profile.name, task))
        return await real(profile, task, workdir, timeout_s, **kwargs)

    from nodes import developer

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)

    await run_regression(monkeypatch)

    critic_tasks = [task for name, task in tasks if name == "critic"]
    assert len(critic_tasks) >= 4

    # Iteration 1 is the first measurement: no regression to report.
    assert "!! REGRESSION" not in critic_tasks[0]

    # Iteration 2 onwards: the regression is the FIRST thing in the prompt,
    # ahead of the marginal ranking that would otherwise say "U" every time.
    second = critic_tasks[1]
    assert second.startswith("## !! REGRESSION -- THIS IS THE FINDING")
    assert "START WITH THE REGRESSION" in second
    assert "REVERT" in second
    assert "regression_verdict" in second
    # ...and the marginal ranking is still there, explicitly demoted so the
    # model does not read "dominant_term = U" as this round's finding.
    assert "dominant_term       = U" in second
    assert "naming it again is not a finding" in second
    assert "If there is a" in second and "REGRESSION section above" in second


async def test_the_attribution_artifact_records_the_regression(monkeypatch) -> None:
    """`attribution.json` is what `check_learning.py` reads afterwards."""
    await run_regression(monkeypatch)
    attribution = json.loads(
        (config.RUNS_DIR / "iter_3" / "attribution.json").read_text(encoding="utf-8")
    )
    assert attribution["regression"]["champion_iteration"] == 1
    assert attribution["regression"]["streak"] >= 2
    assert attribution["regression"]["terms"]["worst_term"] == "U"
    assert attribution["verdict_vs_best"] == scoreboard.VERDICT_REGRESSION


async def test_the_judge_report_records_the_verdict_and_the_best(monkeypatch) -> None:
    await run_regression(monkeypatch)
    report = json.loads(
        (config.RUNS_DIR / "iter_4" / "judge_report.json").read_text(encoding="utf-8")
    )
    assert report["verdict_vs_best"] == scoreboard.VERDICT_REGRESSION
    assert report["best_iteration_so_far"] == 1
    assert round(report["best_mgs_so_far"], 4) == 0.3172


# ======================================================================
# the run reports the truth about itself
# ======================================================================


async def test_the_halt_reason_names_the_best_not_the_last(monkeypatch) -> None:
    """THE LIE, AS AN ASSERTION.

    The string this replaces was `iteration budget exhausted after 5; best
    MGS=0.1190` on a run whose best was 0.3172.
    """
    final = await run_regression(monkeypatch)
    reason = str(final["halt_reason"])
    assert "best MGS=0.3172" in reason, reason
    assert "iteration 1" in reason
    assert "final MGS=0.1190" in reason, reason
    assert "best MGS=0.1190" not in reason


async def test_the_scoreboard_summary_says_the_run_regressed(monkeypatch) -> None:
    final = await run_regression(monkeypatch)
    board = scoreboard.summary(final)
    assert board["best_iteration"] == 1
    assert board["final_iteration"] == 4
    assert board["regressed_from_best"] is True
    assert board["regression_streak"] == 3


async def test_a_healthy_run_reports_no_regression_and_no_rollback(monkeypatch) -> None:
    """THE CONTROL. None of this may fire on a run that is working.

    `happy_path` improves and reaches the target; its lineage must still be the
    plain N-1 chain and its halt reason must not mention a regression.
    """
    from graph import build_graph
    from state import initial_state

    monkeypatch.setattr(config, "MOCK_SCENARIO", "happy_path")
    monkeypatch.setattr(config, "MAX_ITERATIONS", 4)
    graph = build_graph()
    final = await graph.ainvoke(
        initial_state(workspace=str(config.iteration_dir(1) / "workspace"),
                      started_at=time.monotonic()),
        config={"recursion_limit": config.RECURSION_LIMIT},
    )

    assert "target reached" in str(final["halt_reason"])
    board = scoreboard.summary(final)
    assert board["regressed_from_best"] is False
    assert all(s.get("rolled_back") is False for s in spans(final, "developer"))
    assert not spans(final, "infra_retry")
