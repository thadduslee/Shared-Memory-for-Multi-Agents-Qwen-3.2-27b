"""End to end: linear lineage that steps over a failed build, through the REAL graph.

WHAT THIS FILE EXISTS FOR. `ROLLBACK_TO_BEST=false` makes iteration N inherit
iteration N-1 so the lineage compounds instead of re-seeding from a frozen
champion every round -- which is what run-b5d7565ddb4a did for fifteen straight
iterations. Strict N-1 has one hole the champion rule did not: a build that
FAILED has a workspace on disk full of code whose tests do not pass, and
unconditional N-1 would make it the foundation of everything after it.
Iteration 14 of that run ended `2 failed, 38 passed`.

`LINEAGE_SKIP_FAILED_BUILDS` closes it, and the closing has two halves that are
easy to get half-right:

  1. THE WORKSPACE steps over the failed iteration -- 4 <- 2 when 3 failed.
  2. THE KNOWLEDGE does not. Everything iteration 3 learned by failing has to
     reach iteration 4's Architect, or the loop simply re-proposes the work
     order that just broke and fails the same way again.

Half 1 without half 2 is not an improvement; it is amnesia with a tidier
lineage. Both halves are asserted here against the compiled graph rather than
against the functions, because either could be correct in isolation and wired
to the wrong node.

`MOCK_SCENARIO=failed_build_midrun` scripts exactly that shape: iterations 1, 2
and 4 build and improve, iteration 3's `run_tests` never goes green.
"""

from __future__ import annotations

import time

import pytest

import config
import scoreboard


@pytest.fixture(autouse=True)
def isolated_run(tmp_path, monkeypatch):
    """Fresh artifacts and fresh process-level singletons per test.

    Same fixture as `tests/test_regression_recovery_graph.py`, duplicated for
    the same reason: the orchestrator caches the dataset, the harness client and
    the circuit breakers at module level, which is right for a real run and
    poisonous for a suite that switches scenarios between cases.
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


async def run_linear(monkeypatch, iterations: int = 4, **overrides) -> dict:
    """The `failed_build_midrun` scenario under linear, failure-skipping lineage."""
    from graph import build_graph
    from state import initial_state

    monkeypatch.setattr(config, "MOCK_SCENARIO", "failed_build_midrun")
    monkeypatch.setattr(config, "MAX_ITERATIONS", iterations)
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    monkeypatch.setattr(config, "LINEAGE_SKIP_FAILED_BUILDS", True)
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)

    graph = build_graph()
    state = initial_state(
        workspace=str(config.iteration_dir(1) / "workspace"), started_at=time.monotonic()
    )
    return await graph.ainvoke(state, config={"recursion_limit": config.RECURSION_LIMIT})


def spans(final: dict, node: str) -> list[dict]:
    return [s for s in final.get("node_timings") or [] if s.get("node") == node]


def capture_prompts(monkeypatch) -> list[tuple[str, str]]:
    """Install a spy on `agent_call` and return the list it appends to.

    The LIST is returned rather than its Architect subset, because the spy has
    to be installed before the run and read after it; snapshotting the subset
    here would only ever capture the empty list.
    """
    captured: list[tuple[str, str]] = []
    from nodes import _transport as transport
    from nodes import developer

    real = transport.agent_call

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        captured.append((profile.name, task))
        return await real(profile, task, workdir, timeout_s, **kwargs)

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)
    return captured


def architect_tasks(captured: list[tuple[str, str]]) -> list[str]:
    """The Architect task texts out of a completed capture, in order."""
    return [task for name, task in captured if name == "architect"]


# ======================================================================
# half 1: the workspace steps over the failed build
# ======================================================================


async def test_the_lineage_is_linear_while_every_build_is_green(monkeypatch) -> None:
    """THE POINT OF THE SETTING. Iteration 2 inherits 1, not a frozen champion.

    Iterations 1 and 2 both build here, so this is the case that must look like
    ordinary N-1 lineage -- if it did not, the setting would be doing something
    other than what it says.
    """
    final = await run_linear(monkeypatch)
    provenance = {s["iteration"]: s.get("workspace_from") for s in spans(final, "developer")}
    assert provenance[1] == "template"
    assert provenance[2] == "iter_1"
    assert provenance[3] == "iter_2"


async def test_a_failed_build_is_stepped_over_as_a_parent(monkeypatch) -> None:
    """4 <- 2, not 4 <- 3. Iteration 3 built a tree whose tests do not pass."""
    final = await run_linear(monkeypatch)
    provenance = {s["iteration"]: s.get("workspace_from") for s in spans(final, "developer")}
    assert provenance[4] == "iter_2", provenance


async def test_the_skip_is_distinguishable_from_a_rollback_on_the_span(monkeypatch) -> None:
    """`node_timings` is the audit trail, and the two reasons are not the same fact.

    A run whose spans say only `rolled_back: true` cannot be read afterwards to
    tell "the last iteration scored worse" from "the last iteration could not
    build" -- and those call for opposite conclusions about the run.
    """
    final = await run_linear(monkeypatch)
    reasons = {s["iteration"]: s.get("lineage_reason") for s in spans(final, "developer")}
    assert reasons[2] == scoreboard.LINEAGE_DIRECT
    assert reasons[3] == scoreboard.LINEAGE_DIRECT
    assert reasons[4] == scoreboard.LINEAGE_SKIPPED_FAILED, reasons


async def test_unconditional_lineage_inherits_the_broken_tree(monkeypatch) -> None:
    """THE HOLE THE SETTING CLOSES, pinned so it stays closed.

    With the skip disabled, iteration 4 is seeded from iteration 3 -- the build
    that never went green. This is the behaviour `LINEAGE_SKIP_FAILED_BUILDS`
    exists to prevent, and asserting it here is what proves the default is
    actually doing something.
    """
    final = await run_linear(monkeypatch, LINEAGE_SKIP_FAILED_BUILDS=False)
    provenance = {s["iteration"]: s.get("workspace_from") for s in spans(final, "developer")}
    assert provenance[4] == "iter_3", provenance


# ======================================================================
# half 2: the knowledge does not step over anything
# ======================================================================


async def test_the_next_architect_is_told_why_the_build_failed(monkeypatch) -> None:
    """THE HALF THAT MATTERS. Skipping the workspace must not skip the lesson.

    Iteration 4 designs against iteration 2's code, so without this it would be
    designing as though iteration 3 had never happened -- and the obvious design
    against iteration 2's code is the one iteration 3 already tried.
    """
    captured = capture_prompts(monkeypatch)
    await run_linear(monkeypatch)

    prompts = architect_tasks(captured)
    assert len(prompts) >= 4, f"only {len(prompts)} Architect turns"
    fourth = prompts[3]
    assert "THE DEVELOPER COULD NOT BUILD YOUR PREVIOUS DESIGN" in fourth
    assert "failure_signature" in fourth
    # The specific gate that failed, not just the fact that something did.
    assert "tests_ok" in fourth


async def test_the_next_architect_is_told_whose_code_it_is_looking_at(monkeypatch) -> None:
    """And that the failed iteration's edits are gone, so it does not revert them."""
    captured = capture_prompts(monkeypatch)
    await run_linear(monkeypatch)

    prompts = architect_tasks(captured)
    assert len(prompts) >= 4, f"only {len(prompts)} Architect turns"
    fourth = prompts[3]
    assert "Your workspace is iteration 2's code" in fourth
    assert "PASSED ITS GATES" in fourth
    # It must NOT be told this was a score rollback: nothing scored worse.
    assert "THE LOSING CHANGES ARE ALREADY GONE" not in fourth
    assert "the current champion" not in fourth


async def test_the_failed_iteration_is_visible_in_the_trend_table(monkeypatch) -> None:
    """A gap in a numbered history reads as a lost record, not as a failed build."""
    captured = capture_prompts(monkeypatch)
    await run_linear(monkeypatch)

    prompts = architect_tasks(captured)
    assert len(prompts) >= 4, f"only {len(prompts)} Architect turns"
    fourth = prompts[3]
    assert "3    " in fourth
    assert "build failed, never evaluated" in fourth


async def test_the_failure_survives_in_the_run_history(monkeypatch) -> None:
    """`dev_failure_history` is what `_repeat_warning` reads on a SECOND failure.

    The report itself is cleared by the next green build; the history entry is
    not, and it is the only thing that can tell the loop the same signature has
    now broken two iterations.
    """
    final = await run_linear(monkeypatch)
    history = final.get("dev_failure_history") or []
    assert [entry["iteration"] for entry in history] == [3]
    assert history[0]["gates_failed"] or history[0]["gates_never_ran"]
    assert history[0]["signature"]


# ======================================================================
# what must NOT change
# ======================================================================


async def test_the_best_iteration_is_still_tracked_and_reported(monkeypatch) -> None:
    """The lineage rule decides what is INHERITED, never what is REPORTED."""
    final = await run_linear(monkeypatch)
    summary = scoreboard.summary(final)
    rows = {int(r["iteration"]): r for r in summary["history"]}
    assert set(rows) == {1, 2, 4}, "iteration 3 never reached the Judge"
    assert summary["best_iteration"] == 4
    assert summary["best_mgs"] == pytest.approx(rows[4]["MGS"])


async def test_a_failed_build_never_becomes_a_parent_under_the_champion_rule(
    monkeypatch,
) -> None:
    """The default is untouched: it never had this hole to begin with."""
    final = await run_linear(monkeypatch, ROLLBACK_TO_BEST=True)
    provenance = {s["iteration"]: s.get("workspace_from") for s in spans(final, "developer")}
    assert provenance[4] == "iter_2", provenance
    reasons = {s["iteration"]: s.get("lineage_reason") for s in spans(final, "developer")}
    assert reasons[4] == scoreboard.LINEAGE_CHAMPION
