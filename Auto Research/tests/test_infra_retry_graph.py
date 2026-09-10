"""A network outage must cost a retry, not a redesign.

THE FAILURE, IN FULL. run-8cf58d33b311, iteration 2, from the run log:

    16:19:15 WARNING developer: http transport exceeded 150s; abandoning the call
    16:19:15 WARNING developer: no action in reply (attempt 1/4, ok=False, 0 chars)
    16:20:05 WARNING developer: http transport exceeded 50s; abandoning the call
    ...
    16:32:08 WARNING developer: turn ceiling (60) reached
    16:32:08 WARNING developer exhausted at iteration 2: missing=tests_ok,migration_ok
    16:32:08 WARNING developer -> architect: developer hit the 60-turn ceiling

Sixteen minutes and 1.22 million tokens on a flapping OpenRouter endpoint. The
episode never called `run_tests`, never called `sql_exec`, and wrote no bytes.
The graph's only failure edge went to the Architect, so the Architect was asked
to redesign -- and the work order it wrote ("remove the `top_k` stop") is the
change that cost the run 0.095 MGS the moment iteration 3 built it.

These tests drive the real compiled graph with a transport that fails for a
bounded number of calls and assert that the SAME iteration is re-run instead.
"""

from __future__ import annotations

import time

import pytest

import config
from harness.dsh_client import DSHResult


@pytest.fixture(autouse=True)
def isolated_run(tmp_path, monkeypatch):
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


def calls_to_exhaust_an_episode() -> int:
    """How many failing completions it takes to burn one episode's retry budget.

    One `dev_think` TURN costs `DEV_THINK_RESAMPLES + 1` completions -- the
    first sample plus the resamples that exist precisely to survive a decoder
    stutter -- and only the whole turn failing charges one retry. So exhausting
    MAX_DEV_RETRIES needs the product, not MAX_DEV_RETRIES calls. Getting this
    wrong makes the test pass for the wrong reason: the episode recovers, goes
    green, and asserts nothing about the retry path at all.
    """
    return config.MAX_DEV_RETRIES * (config.DEV_THINK_RESAMPLES + 1)


def flapping_developer_transport(monkeypatch, *, fail_first: int | None = None):
    """Make the Developer's model calls time out for the first `fail_first` calls.

    Only the Developer: the Architect, Critic and Judge stay healthy, which is
    what the real outage looked like -- the Developer is the node that makes
    dozens of calls per episode, so it is the node a flaky endpoint kills.

    Returns a counter dict the test can read afterwards.
    """
    fail_first = calls_to_exhaust_an_episode() if fail_first is None else fail_first
    from nodes import _transport as transport
    from nodes import developer

    real = transport.agent_call
    counter = {"developer_calls": 0, "failed": 0}

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        if profile.name == "developer":
            counter["developer_calls"] += 1
            if counter["failed"] < fail_first:
                counter["failed"] += 1
                return DSHResult(
                    ok=False, text="", profile=profile.name, finish_reason="timeout",
                    error=f"http transport timed out after {timeout_s}s",
                )
        return await real(profile, task, workdir, timeout_s, **kwargs)

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)
    return counter


async def run(monkeypatch, scenario: str = "happy_path", **overrides) -> dict:
    from graph import build_graph
    from state import initial_state

    monkeypatch.setattr(config, "MOCK_SCENARIO", scenario)
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)
    return await build_graph().ainvoke(
        initial_state(workspace=str(config.iteration_dir(1) / "workspace"),
                      started_at=time.monotonic()),
        config={"recursion_limit": config.RECURSION_LIMIT},
    )


def spans(final: dict, node: str) -> list[dict]:
    return [s for s in final.get("node_timings") or [] if s.get("node") == node]


# ======================================================================


async def test_a_transient_outage_re_runs_the_iteration_and_the_run_recovers(
    monkeypatch,
) -> None:
    """THE FIX, end to end.

    The transport eats enough of iteration 1's episode to exhaust its retries;
    the graph re-runs iteration 1 rather than asking the Architect for a new
    design, the retry succeeds, and the run goes on to reach its target.
    """
    flapping_developer_transport(monkeypatch)
    final = await run(monkeypatch, "happy_path", MAX_ITERATIONS=4)

    retries = spans(final, "infra_retry")
    assert retries, "the infrastructure retry edge was never taken"
    assert retries[0]["iteration"] == 1
    assert retries[0]["attempt"] == 1

    # The SAME iteration was rebuilt -- two developer episodes, one iteration.
    developer_spans = [s for s in spans(final, "developer") if s["iteration"] == 1]
    assert len(developer_spans) == 2, developer_spans

    # ...and no new design was written for it. `architect` ran once for
    # iteration 1, not twice.
    architect_spans = [s for s in spans(final, "architect") if s["iteration"] == 1]
    assert len(architect_spans) == 1, architect_spans

    assert "target reached" in str(final["halt_reason"])


async def test_the_failure_report_is_cleared_by_the_retry(monkeypatch) -> None:
    """A recovered outage must not follow the run around.

    `dev_failure_report` is the Architect's only feedback channel for a failed
    build. Leaving an OpenRouter timeout in it would make the NEXT Architect
    turn redesign against a failure that has already been recovered from --
    which is the same bug one node further along.
    """
    flapping_developer_transport(monkeypatch)
    final = await run(monkeypatch, "happy_path", MAX_ITERATIONS=4)
    assert final.get("dev_failure_report") == {}


async def test_a_permanent_outage_gives_up_after_the_budget_and_tells_the_truth(
    monkeypatch,
) -> None:
    """The bound is real: a dead endpoint cannot spin one iteration forever.

    And when the graph does finally hand it to the Architect, it hands it over
    labelled -- `infrastructure`, with the transport errors attached -- rather
    than as the phantom test failure the old report described.
    """
    tasks: list[tuple[str, str]] = []
    from nodes import _transport as transport
    from nodes import developer

    real = transport.agent_call

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        if profile.name == "developer":
            return DSHResult(
                ok=False, text="", profile=profile.name, finish_reason="timeout",
                error="http transport timed out after 150s",
            )
        tasks.append((profile.name, task))
        return await real(profile, task, workdir, timeout_s, **kwargs)

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)

    final = await run(monkeypatch, "happy_path", MAX_ITERATIONS=3, MAX_INFRA_RETRIES=2)

    # The budget is PER ITERATION, so a run of three iterations against a dead
    # endpoint spends it three times over rather than twice in total -- and
    # every one of those iterations is bounded, which is the property that
    # matters.
    retries = spans(final, "infra_retry")
    assert retries, "the infrastructure retry edge was never taken"
    by_iteration: dict[int, list[int]] = {}
    for span in retries:
        by_iteration.setdefault(span["iteration"], []).append(span["attempt"])
    for iteration, attempts in by_iteration.items():
        assert attempts == list(range(1, len(attempts) + 1)), (iteration, attempts)
        assert len(attempts) <= config.MAX_INFRA_RETRIES, (iteration, attempts)
    assert max(len(a) for a in by_iteration.values()) == config.MAX_INFRA_RETRIES

    report = final["dev_failure_report"]
    assert report["classification"] == "infrastructure"
    assert report["n_transport_failures"] >= 1
    assert report["transport_errors"] == ["http transport timed out after 150s"]
    # `run_tests` and `sql_exec` never ran -- and the report says so, instead of
    # reporting them as unmet gates the design failed.
    assert "tests_ok" in report["gates_never_ran"]
    assert "migration_ok" in report["gates_never_ran"]
    assert report["gates_failed"] == []

    # The Architect that finally sees it is told not to redesign around it.
    architect_tasks = [task for name, task in tasks if name == "architect"]
    assert len(architect_tasks) >= 2
    assert "DO NOT REDESIGN IN RESPONSE TO THIS" in architect_tasks[1]
    assert "TRANSPORT failures, not build failures" in architect_tasks[1]
    assert "was never called" in architect_tasks[1]


async def test_a_real_test_failure_is_never_retried_as_infrastructure(
    monkeypatch,
) -> None:
    """THE CONTROL. `dev_retry_exhaustion` scripts a red test with a healthy
    transport; it must still go straight to the Architect for a redesign."""
    final = await run(monkeypatch, "dev_retry_exhaustion", MAX_ITERATIONS=2)
    assert not spans(final, "infra_retry")
    architect_spans = [s for s in spans(final, "architect")]
    assert len(architect_spans) >= 2, "the Architect should have been asked to redesign"


async def test_the_retry_budget_is_per_iteration(monkeypatch) -> None:
    """Two blips in iteration 1 must not leave iteration 2 with none left.

    The Architect resets the counter at the top of every iteration, so the
    budget absorbs two outages INSIDE one iteration rather than two spread
    across the run.
    """
    calls = {"n": 0}
    from nodes import _transport as transport
    from nodes import developer

    real = transport.agent_call

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        if profile.name == "developer":
            calls["n"] += 1
            # Fail the whole of iteration 1's first episode, and the whole of
            # iteration 2's first episode, with healthy turns in between.
            if calls["n"] <= calls_to_exhaust_an_episode():
                return DSHResult(ok=False, text="", profile=profile.name,
                                 finish_reason="timeout", error="http transport timed out")
        return await real(profile, task, workdir, timeout_s, **kwargs)

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)

    final = await run(monkeypatch, "happy_path", MAX_ITERATIONS=4)
    # Iteration 1 spent one retry; iteration 2 starts from zero again.
    assert final["infra_retry_count"] == 0, "the counter must reset per iteration"
    assert spans(final, "infra_retry")[0]["iteration"] == 1
