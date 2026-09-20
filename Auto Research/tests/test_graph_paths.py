"""Integration: run each scripted scenario through the REAL compiled graph.

The unit tests in `test_routing.py` prove each router returns the right label.
These prove the labels are wired to the right nodes in `graph.py` -- a
distinction that matters, because a router can be perfectly correct and still be
mapped to the wrong destination in `add_conditional_edges`.

Every test here runs fully offline: no network, no GPUs, no Node runtime.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import config
from harness.profiles import DEVELOPER_PROFILE


@pytest.fixture(autouse=True)
def isolated_run(tmp_path, monkeypatch):
    """Fresh artifacts and fresh process-level singletons per test.

    The orchestrator caches the dataset, the harness client and the circuit
    breakers at module level -- correct for a real run, poisonous for a test
    suite that switches scenarios between cases.
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


async def run_scenario(scenario: str, monkeypatch, **overrides) -> dict:
    from graph import build_graph
    from state import initial_state

    monkeypatch.setattr(config, "MOCK_SCENARIO", scenario)
    for key, value in overrides.items():
        monkeypatch.setattr(config, key, value)

    graph = build_graph()
    state = initial_state(
        workspace=str(config.iteration_dir(1) / "workspace"), started_at=time.monotonic()
    )
    return await graph.ainvoke(state, config={"recursion_limit": config.RECURSION_LIMIT})


def stages_run(runs_dir: Path) -> set[str]:
    """Which evaluation stages actually produced predictions on disk."""
    return {
        path.parent.name
        for path in runs_dir.rglob("predictions.jsonl")
        if path.stat().st_size > 0
    }


# ======================================================================


async def test_happy_path_passes_the_gate_runs_full_and_hits_the_target(monkeypatch) -> None:
    final = await run_scenario("happy_path", monkeypatch, MAX_ITERATIONS=4)
    assert "target reached" in str(final["halt_reason"])
    assert final["mgs_score"] >= config.MGS_TARGET
    assert stages_run(config.RUNS_DIR) == {"dev", "full"}, "the full run should have executed"


async def test_immediate_success_terminates_in_one_iteration(monkeypatch) -> None:
    final = await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=5)
    assert final["iteration_count"] == 1
    assert "target reached" in str(final["halt_reason"])


async def test_dev_gate_failure_never_runs_the_full_evaluation(monkeypatch) -> None:
    """The expensive 579-checkpoint run must be SKIPPED, not merely scored low."""
    final = await run_scenario("dev_gate_fail", monkeypatch, MAX_ITERATIONS=2)
    assert final["mgs_score"] < config.DEV_GATE_MGS
    assert final["proceed_to_full"] is False
    assert stages_run(config.RUNS_DIR) == {"dev"}, "the full run must never have been entered"


async def test_developer_retry_exhaustion_routes_back_to_the_architect(monkeypatch) -> None:
    final = await run_scenario("dev_retry_exhaustion", monkeypatch, MAX_ITERATIONS=2)
    # Iteration 1's Developer gave up and the loop re-entered the Architect,
    # which is why the run reached a second iteration at all.
    assert final["iteration_count"] >= 2
    assert (config.RUNS_DIR / "iter_1" / "developer_scratchpad.json").is_file()
    assert (config.RUNS_DIR / "iter_2" / "design.md").is_file()


async def test_the_next_architect_is_shown_where_the_developer_failed(monkeypatch) -> None:
    """The failure edge must carry the failure, not just the fact of one.

    Routing back to the Architect is only useful if the Architect can tell what
    to change. This scenario fails `run_tests` with a real traceback
    (mocks/sandbox.py), so iteration 2's design turn must be able to quote the
    assertion that killed iteration 1 -- and no other node can supply it, since
    a failed build never reaches the Judge or the Critic.
    """
    import json

    import nodes._transport as transport

    tasks: list[tuple[str, str]] = []
    real_agent_call = transport.agent_call

    async def spy(profile, task, workdir, timeout_s=None, **kwargs):
        # The Developer passes a conversation rather than a task string, so the
        # prompt text this spy is looking for is in `messages` for that node and
        # in `task` for the Architect. Flatten both.
        text = task or "\n\n".join(
            str(m.get("content") or "") for m in (kwargs.get("messages") or [])
        )
        tasks.append((profile.name, text))
        return await real_agent_call(profile, task, workdir, timeout_s, **kwargs)

    # The Developer bound `agent_call` at import time, so it is patched at its
    # own import site. The Architect and the Critic go through
    # `_transport.agent_call_json` -- which owes them a ```json block and will
    # re-ask once to get one -- and that wrapper resolves `agent_call` as a
    # module global at call time, so patching it on `_transport` covers both.
    from nodes import developer

    monkeypatch.setattr(transport, "agent_call", spy)
    monkeypatch.setattr(developer, "agent_call", spy)

    await run_scenario("dev_retry_exhaustion", monkeypatch, MAX_ITERATIONS=2)

    report = json.loads((config.RUNS_DIR / "iter_1" / "developer_failure.json").read_text())
    # `migration_ok` is unmet as well as `tests_ok`: the scripted Developer never
    # gets past the red test to call `sql_exec`, which is what a stuck real loop
    # looks like too. Both belong in the report -- the Architect needs to know
    # the migration was never even attempted.
    assert report["missing_gates"] == ["tests_ok", "migration_ok"], report["missing_gates"]
    # The assertion, not pytest's trailing "1 failed, 17 passed" scoreboard.
    assert report["last_error"].startswith("AssertionError:")
    assert "tombstone gate did not fire" in report["last_error"]

    architect_tasks = [task for name, task in tasks if name == "architect"]
    assert len(architect_tasks) >= 2, "the loop should have re-entered the Architect"
    second = architect_tasks[1]
    assert "COULD NOT BUILD" in second
    # The gate that RAN and failed, distinguished from the one that never ran.
    assert "gates that RAN AND FAILED: tests_ok (`run_tests`)" in second
    assert "migration_ok (`sql_exec` was never called)" in second
    # And the episode is classified as a DESIGN failure, so the Architect is
    # told to redesign -- which is right here, and is exactly what must NOT
    # happen when the classification is `infrastructure`.
    assert report["classification"] == "design", report["classification_reason"]
    assert "Read this as a critique of the design" in second
    assert "tombstone gate did not fire" in second, "the assertion never crossed the edge"

    # ...and the first Architect turn, which had no failure to answer for, must
    # not be carrying one.
    assert "COULD NOT BUILD" not in architect_tasks[0]


async def test_circuit_breaker_aborts_the_batch_and_skips_remaining_shards(monkeypatch) -> None:
    await run_scenario("failfast_signature", monkeypatch, MAX_ITERATIONS=2)
    report_path = config.RUNS_DIR / "iter_1" / "dev" / "shard_report.json"
    assert report_path.is_file()

    import json

    report = json.loads(report_path.read_text())
    assert report["circuit_breaker_signature"], "the breaker should have tripped"
    assert report["n_skipped"] > 0, "shards after the trip must be skipped, not run"
    assert report["n_predictions"] == 0
    # A tripped breaker must not be scored: the Judge is skipped entirely.
    assert not (config.RUNS_DIR / "iter_1" / "dev" / "judge_report.json").is_file()


async def test_curriculum_failure_repeats_the_phase_instead_of_advancing(monkeypatch) -> None:
    final = await run_scenario("curriculum_fail", monkeypatch, MAX_ITERATIONS=2)
    history = final.get("curriculum_history") or []
    advanced = [h for h in history if h.get("advanced_to")]
    # Iteration 1's phase failed, so the first advance decision keeps the phase.
    assert advanced and advanced[0]["advanced_to"] == config.CURRICULUM_PHASES[0]


async def test_budget_guard_halts_with_a_populated_reason(monkeypatch) -> None:
    final = await run_scenario("budget_exhausted", monkeypatch, MAX_ITERATIONS=5)
    assert "budget exhausted" in str(final["halt_reason"])
    assert final["iteration_count"] <= 2, "the guard should fire before the loop runs away"


async def test_max_iterations_stops_a_run_that_never_converges(monkeypatch) -> None:
    final = await run_scenario("max_iterations", monkeypatch, MAX_ITERATIONS=3)
    assert final["iteration_count"] == 3
    assert "iteration budget" in str(final["halt_reason"])
    assert final["mgs_score"] < config.MGS_TARGET


# ======================================================================
# Artifacts and the fan-out contract
# ======================================================================


async def test_every_required_artifact_is_written(monkeypatch) -> None:
    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    iter_dir = config.RUNS_DIR / "iter_1"
    for name in ("design.md", "migration.sql", "judge_report.json", "critique.md"):
        assert (iter_dir / name).is_file(), f"missing artifact {name}"
    assert (iter_dir / "full" / "predictions.jsonl").is_file()


async def test_the_notebook_accumulates_through_the_real_graph(monkeypatch) -> None:
    """The Architect's notebook, through the real graph. See tests/test_critique_recap.py.

    It is REWRITTEN each turn from the previous version plus what the round
    found, so what must hold end to end is that the earlier notes survive the
    rewrites rather than each turn starting over. The last iteration's own
    lesson is absent because the turn that would have folded it in is the one
    that never ran.
    """
    await run_scenario("max_iterations", monkeypatch, MAX_ITERATIONS=4)

    from nodes._recap import notes_from_notebook

    notebook = (config.RUNS_DIR / "critique_summary.md").read_text(encoding="utf-8")
    notes = notes_from_notebook(notebook)
    assert len(notes) >= 3, f"earlier notes were lost across rewrites: {notes}"
    assert len(notes) == len(set(notes)), "a rewrite duplicated a note"

    history = (config.RUNS_DIR / "notebook_history.md").read_text(encoding="utf-8")
    assert history.count("## after iteration") >= 3, "every version must be logged"


async def test_the_critique_files_carry_only_their_own_iteration(monkeypatch) -> None:
    """`critique.md` is what the next Architect opens for iteration i, and the
    notebook is what it opens for 1..i-1. Mixing them would have the Architect
    summarising a file that already contains the earlier summaries."""
    await run_scenario("max_iterations", monkeypatch, MAX_ITERATIONS=4)

    for n in (1, 4):
        written = (config.RUNS_DIR / f"iter_{n}" / "critique.md").read_text(encoding="utf-8")
        assert "**iteration" not in written, f"iter_{n}/critique.md carries a recap"


async def test_an_unbuilt_iteration_is_recapped_even_though_it_has_no_critique(
    monkeypatch,
) -> None:
    """A failed build never reaches the Judge, so no Critic runs and no critique
    exists. Without a row of its own that iteration would simply be missing from
    a numbered history, and a gap reads as a lost record."""
    await run_scenario("dev_retry_exhaustion", monkeypatch, MAX_ITERATIONS=3)

    assert not (config.RUNS_DIR / "iter_1" / "critique.md").exists()
    # The DIGEST is the numbered record, and it is what must show the gap-free
    # history; the notebook is curated prose and no longer carries row labels.
    final = await run_scenario("dev_retry_exhaustion", monkeypatch, MAX_ITERATIONS=3)
    kinds = {(e["iteration"], e["kind"]) for e in final["critique_digest"]}
    assert (1, "dev_failure") in kinds, "an unbuilt iteration must still be recorded"


async def test_the_same_critique_is_never_recapped_twice(monkeypatch) -> None:
    """Iteration 1 fails to build, so iteration 2's Architect is handed the same
    (empty) critique iteration 1's was. Only the build failure may be recorded."""
    final = await run_scenario("dev_retry_exhaustion", monkeypatch, MAX_ITERATIONS=3)

    recapped = [entry["iteration"] for entry in final["critique_digest"]]
    assert recapped == sorted(set(recapped)), f"an iteration was recapped twice: {recapped}"


async def test_no_notebook_is_written_before_there_is_anything_to_put_in_it(
    monkeypatch,
) -> None:
    """Iteration 1 has nothing before it. An empty file under a header reads as
    'this was checked and there was nothing', which is a different claim."""
    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    notebook = config.RUNS_DIR / "critique_summary.md"
    if notebook.exists():
        from nodes._recap import notes_from_notebook

        assert notes_from_notebook(notebook.read_text(encoding="utf-8")), (
            "a notebook that exists must have notes in it; an empty file under a "
            "header reads as 'this was checked and there was nothing'"
        )


async def test_the_manifest_the_evaluator_read_contains_no_hidden_fields(monkeypatch) -> None:
    """The wall, verified against the file that was actually on disk."""
    from gatemem_adapter import HIDDEN_ANNOTATION_FIELDS

    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    manifests = list(config.RUNS_DIR.rglob("checkpoints.stripped.jsonl"))
    assert manifests, "the dispatcher must write the stripped manifest"
    for manifest in manifests:
        blob = manifest.read_text()
        for field in HIDDEN_ANNOTATION_FIELDS:
            assert f'"{field}"' not in blob, f"{field} leaked into {manifest}"


async def test_fan_out_produced_more_than_one_shard(monkeypatch) -> None:
    """A `Send` fan-out that degenerates to one shard proves nothing."""
    import json

    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    report = json.loads((config.RUNS_DIR / "iter_1" / "dev" / "shard_report.json").read_text())
    assert report["n_shards"] > 1
    assert report["n_failed"] == 0
    assert report["n_predictions"] == config.EXPECTED_DEV_CHECKPOINTS


async def test_predictions_are_deterministic_across_two_identical_runs(monkeypatch) -> None:
    """Shards finish out of order; the collector must still produce one file."""
    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    first = (config.RUNS_DIR / "iter_1" / "dev" / "predictions.jsonl").read_text()

    import harness.dsh_client as dsh
    import nodes.medical_evaluator as evaluator

    dsh.reset_dsh_client()
    evaluator.reset_dataset()
    evaluator.reset_breakers()
    monkeypatch.setattr(config, "RUNS_DIR", config.RUNS_DIR.parent / "runs2")

    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    second = (config.RUNS_DIR / "iter_1" / "dev" / "predictions.jsonl").read_text()
    assert first == second, "the fan-in must be order-independent"


async def test_developer_reactloop_actually_ran_its_tools(monkeypatch) -> None:
    """The micro-graph must be a loop, not a single-shot call."""
    import json

    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    steps = json.loads((config.RUNS_DIR / "iter_1" / "developer_scratchpad.json").read_text())
    tools = [s["action"].get("tool") for s in steps]
    assert len(steps) > 5, "a single-shot call is not a ReAct loop"
    for required in ("write_file", "compile_check", "sql_exec", "run_tests"):
        assert required in tools, f"the Developer never called {required}"


def test_least_privilege_configs_are_materialized(tmp_path) -> None:
    """A profile that cannot render a valid composition fails offline, not live.

    Asserted at the SOURCE, like the Developer test below, and for the same
    reason: this used to fish `architect.cordis.yml` out of a completed mock
    run, which silently stopped asserting anything the day AGENT_TRANSPORT
    moved the Architect over http (2026-08-25, the provider-pinning change) --
    a config the node never renders is a config no test can check.
    """
    from harness.dsh_client import render_cordis_config
    from harness.profiles import ARCHITECT_PROFILE

    text = render_cordis_config(ARCHITECT_PROFILE.with_workdir(tmp_path)).read_text()
    # The Architect must have no model-facing file tool and no shell.
    assert "dsh-tool-fs" not in text, "the Architect must not be able to write files"
    assert "dsh-bash-local" not in text, "the Architect must not have a shell"


def test_the_developer_composition_renders_and_grants_nothing(tmp_path) -> None:
    """The Developer's least-privilege composition, asserted at the SOURCE.

    Previously this read `developer.cordis.yml` out of a completed run. That
    only worked while every node went through the harness, and it silently
    stopped asserting anything the moment `config.DEVELOPER_TRANSPORT` sent the
    Developer over http -- a config the node never renders is a config no test
    can check. Rendering the profile directly is strictly stronger: it holds
    whichever transport is configured, and it still fails offline rather than
    live if the composition is unrenderable.

    The Developer must have no HARNESS-NATIVE tools. It is the one node that
    edits the codebase, but it does so through DevToolbox -- declared as its own
    tool schema and executed by the Developer's own loop -- never through
    mounted plugins. Mounting them does not add a capability; it adds a SECOND
    toolbox whose names (bash/read/write/edit) the gates cannot observe, and an
    agent handed two toolboxes uses the one in its tool schema. When both were
    live it browsed until the timeout and produced no build at all.
    """
    from harness.dsh_client import render_cordis_config
    from harness.profiles import DEVELOPER_PROFILE

    dev_text = render_cordis_config(DEVELOPER_PROFILE.with_workdir(tmp_path)).read_text()
    assert "dsh-tool-fs" not in dev_text, "the Developer must edit via DevToolbox, not tool-fs"
    assert "dsh-bash-local" not in dev_text, "the Developer must not have a second, ungated shell"
    assert "toolBash: false" in dev_text, "the spine must not advertise bash either"


def test_the_developer_is_advertised_exactly_its_own_toolbox() -> None:
    """Least privilege must not depend on which transport is selected.

    This used to assert that `AsyncLLMClient.chat` had no `tools` parameter at
    all -- the guarantee held by ABSENCE. That was the wrong invariant, and it
    was expensive: a tool-calling model handed no schema emits its native
    tool-call syntax as prose, which is how runs_multi/run-280411c75b99 wrote
    zero bytes across four iterations.

    The schema exists now, so the property to pin is what is IN it. The
    Developer is advertised the ten DevToolbox tools and nothing else, and it
    still mounts no Cordis plugin -- so there is exactly one write surface, it
    is workspace-scoped, and the gates can see every call made through it.
    """
    from nodes.dev_tools import TOOL_NAMES, tool_schemas

    assert not DEVELOPER_PROFILE.capabilities, (
        "the Developer must mount no harness-native plugins: its tools are DevToolbox")

    advertised = {schema["function"]["name"] for schema in tool_schemas()}
    assert advertised == set(TOOL_NAMES), (
        "the advertised schema and the executable toolbox must be the same ten tools")
    # The names the harness-native surface would have used. A model that can see
    # them is a model working outside the gates.
    assert not advertised & {"bash", "read", "write", "edit", "shell", "str_replace_editor"}

    # Every advertised tool must really dispatch: `DevToolbox.call` looks up
    # `_t_<name>`, so an advertised name with no handler is an AttributeError
    # mid-episode rather than a bad reply.
    from nodes.dev_tools import DevToolbox

    for name in advertised:
        assert hasattr(DevToolbox, f"_t_{name}"), f"{name} is advertised but not implemented"


def test_developer_prompts_name_exactly_the_tools_that_exist() -> None:
    """The Developer's tool NARRATIVE must still be true, wherever it is written.

    The schema is what the model calls against, so drifted prose is no longer a
    fatal fault. It is still a misleading one: the prose is where the tools are
    given their job (which are advisory, which one is for new files and which
    for edits), and a name mentioned there that does not exist sends the model
    looking for it.

    CHECKED ACROSS BOTH LAYERS, because they are deliberately split: the persona
    carries what a Developer IS and the rules it works under, and the task
    carries what is true of THIS benchmark -- which tools are the gates here,
    and which fields are this evaluation's answer key. `sql_exec` is a gate for
    a SQL-backed target and would not exist for another one, so it belongs to
    the task. What must hold is that every tool is described SOMEWHERE.
    """
    from harness.profiles import DEVELOPER_PROFILE
    from nodes.dev_tools import TOOL_NAMES
    from nodes.developer import BENCHMARK_TOOL_NOTE

    persona = DEVELOPER_PROFILE.system_prompt
    prose = persona + BENCHMARK_TOOL_NOTE
    for tool in TOOL_NAMES:
        assert tool in prose, f"{tool} exists but neither the persona nor the task names it"
    # The names the harness-native surface would have used, which the model
    # reached for when both toolboxes were live.
    for ghost in ("read_file", "list_dir", "write_file"):
        assert ghost in TOOL_NAMES
    assert not DEVELOPER_PROFILE.capabilities, "the reasoning turn must mount no tools"


# ======================================================================
# Workspace lineage -- what makes the loop self-IMPROVING
# ======================================================================


async def test_iteration_one_starts_from_the_template_baseline(monkeypatch) -> None:
    """Without a baseline, iteration 1 measures a bootstrap attempt, not a design."""
    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)
    workspace = config.RUNS_DIR / "iter_1" / "workspace"
    assert (workspace / "memory_system" / "store.py").is_file()
    assert (workspace / "tests" / "test_rbac.py").is_file()


async def test_later_iterations_inherit_the_previous_workspace(monkeypatch) -> None:
    """Iteration N must edit iteration N-1's code, not a fresh empty directory.

    If this fails the loop is N independent attempts, and the Critic's advice
    from round N has nothing to attach to in round N+1.
    """
    await run_scenario("happy_path", monkeypatch, MAX_ITERATIONS=3)
    first = config.RUNS_DIR / "iter_1" / "workspace"
    second = config.RUNS_DIR / "iter_2" / "workspace"
    assert second.is_dir(), "iteration 2 never ran"

    # Iteration 1's migration artifact must still be present in iteration 2.
    assert (first / "migrations" / "iter_1.sql").is_file()
    assert (second / "migrations" / "iter_1.sql").is_file(), "lineage broken: iter_1 work was lost"
    assert (second / "migrations" / "iter_2.sql").is_file(), "iteration 2 recorded no migration"


async def test_the_schema_version_advances_with_the_migrations(monkeypatch) -> None:
    """A concrete, checkable delta between iterations."""
    import re

    await run_scenario("happy_path", monkeypatch, MAX_ITERATIONS=3)

    def version(iteration: int) -> int:
        text = (config.RUNS_DIR / f"iter_{iteration}" / "workspace" /
                "memory_system" / "store.py").read_text()
        return int(re.search(r"^SCHEMA_VERSION\s*=\s*(\d+)", text, re.MULTILINE).group(1))

    assert version(2) > version(1), "iteration 2 did not advance the schema generation"


def test_scratch_and_caches_are_never_copied_forward(tmp_path) -> None:
    """Carrying `__pycache__` forward would let iteration N run N-1's bytecode.

    Tested against the copy itself rather than against iteration 2's final
    contents: a node legitimately regenerates its own `.cordis` composition in
    place during its own iteration, so "present in iter_2" and "copied from
    iter_1" are different claims and only the second one is a bug.
    """
    from nodes.developer import _copy_tree

    source = tmp_path / "prev"
    (source / "memory_system").mkdir(parents=True)
    (source / "__pycache__").mkdir()
    (source / ".cordis").mkdir()
    (source / ".sessions").mkdir()
    (source / "memory_system" / "store.py").write_text("SCHEMA_VERSION = 3\n")
    (source / "memory_system" / "schema.sql").write_text("CREATE TABLE t(x);")
    (source / "__pycache__" / "store.cpython-310.pyc").write_bytes(b"\x00stale")
    (source / ".cordis" / "developer.cordis.yml").write_text("- id: x")
    (source / ".sessions" / "log.jsonl").write_text("{}")
    (source / "_smoke_runner.py").write_text("# harness scaffold")

    destination = tmp_path / "next"
    _copy_tree(source, destination)

    copied = {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()}
    assert copied == {"memory_system/store.py", "memory_system/schema.sql"}, copied


async def test_template_is_never_mutated_by_a_run(monkeypatch) -> None:
    """The baseline must survive a run intact, or run 2 starts from run 1's edits."""
    import re

    store = config.TEMPLATES_DIR / "memory_system" / "store.py"
    before = store.read_text()
    await run_scenario("happy_path", monkeypatch, MAX_ITERATIONS=3)
    assert store.read_text() == before, "the template baseline was modified in place"
    # The baseline declares SOME schema generation and it is the same one after
    # the run as before it. Pinning the literal number would make every
    # legitimate baseline migration look like a mutation, which is the opposite
    # of what this test is for.
    version = re.search(r"^SCHEMA_VERSION\s*=\s*(\d+)", before, re.MULTILINE)
    assert version, "the baseline must declare a SCHEMA_VERSION"
    assert re.search(rf"^SCHEMA_VERSION\s*=\s*{version.group(1)}$",
                     store.read_text(), re.MULTILINE)


# ======================================================================
# Transport equivalence
# ======================================================================


async def test_http_transport_produces_identical_results_to_dsh(monkeypatch) -> None:
    """Switching transport must not change a single number.

    `AGENT_TRANSPORT=http` exists as a fallback for the one unverified
    assumption in the real path (reaching OpenRouter through the harness's
    DEEPSEEK_BASE_URL). It is only a safe fallback if it is genuinely
    equivalent, so that is asserted rather than assumed.

    This also guards a bug that already bit once: the mock client used to infer
    the node's role by substring, and the Developer's prompt mentions "the
    Architect's work order" -- so every Developer call dispatched to the
    Architect responder and the ReAct loop spun until the recursion limit.
    """
    final_dsh = await run_scenario("happy_path", monkeypatch, MAX_ITERATIONS=3)

    import harness.dsh_client as dsh
    import nodes.medical_evaluator as evaluator

    dsh.reset_dsh_client()
    evaluator.reset_dataset()
    evaluator.reset_breakers()
    monkeypatch.setattr(config, "RUNS_DIR", config.RUNS_DIR.parent / "runs_http")

    final_http = await run_scenario(
        "happy_path", monkeypatch, MAX_ITERATIONS=3,
        AGENT_TRANSPORT="http", EVAL_TRANSPORT="http", JUDGE_TRANSPORT="http",
    )

    for key in ("mgs_score", "utility_score", "access_violation_rate",
                "forgetting_failure_rate", "iteration_count", "halt_reason"):
        assert final_dsh[key] == final_http[key], f"{key} differs between transports"


async def test_each_node_is_dispatched_under_its_own_role(monkeypatch) -> None:
    """Role must be passed explicitly, never inferred from prompt prose."""
    from llm.client import reset_llm_client
    from mocks.llm import MockLLMClient

    client = MockLLMClient()
    monkeypatch.setattr("llm.client._CLIENT", client, raising=False)
    monkeypatch.setattr(config, "AGENT_TRANSPORT", "http")
    monkeypatch.setattr(config, "EVAL_TRANSPORT", "http")
    monkeypatch.setattr(config, "JUDGE_TRANSPORT", "http")

    await run_scenario("immediate_success", monkeypatch, MAX_ITERATIONS=1)

    roles = {call["role"] for call in client.calls}
    assert {"architect", "developer", "critic", "evaluator", "judge"} <= roles, roles
    reset_llm_client()


# ======================================================================
# Action-shape tolerance -- what killed runs_fresh/iter_1
# ======================================================================


def test_flat_action_shape_is_normalized_not_dropped() -> None:
    """A model that writes `path` beside `tool` must still get its argument.

    runs_fresh/iter_1 halted with every gate false because three consecutive
    turns emitted {"tool": "read_file", "path": "memory_system/schema.sql"}.
    `dev_act` read `args`, found nothing, and called read_file with no path --
    reporting "no such file: None" for a 6007-byte file that was present the
    whole time. Each of those cost a retry, so MAX_DEV_RETRIES=3 was gone in
    80 seconds.
    """
    from nodes.developer import _normalize_action

    flat = {"tool": "read_file", "path": "memory_system/schema.sql"}
    assert _normalize_action(flat) == {
        "tool": "read_file", "args": {"path": "memory_system/schema.sql"},
    }

    # The documented shape must survive untouched.
    nested = {"tool": "write_file", "args": {"path": "a.py", "content": "x = 1"}}
    assert _normalize_action(nested) == nested

    # A no-argument tool has nothing to lift and must not grow an empty `args`
    # that would then differ from the nested form under repeat detection.
    assert _normalize_action({"tool": "compile_check"}) == {"tool": "compile_check"}

    # `_fallback` is set by developer.py itself and must not be lifted into args
    # -- `_consecutive_fallbacks` reads it off the top level.
    fb = _normalize_action({"tool": "compile_check", "args": {}, "_fallback": True})
    assert fb["_fallback"] is True and fb["args"] == {}


async def test_read_file_without_a_path_says_so() -> None:
    """The observation must name the real fault, not blame the file."""
    import tempfile

    from nodes.dev_tools import DevToolbox

    with tempfile.TemporaryDirectory() as tmp:
        box = DevToolbox(Path(tmp))
        result = await box.call("read_file", {})
        assert not result.ok
        assert "requires args.path" in result.stderr
        assert "no such file" not in result.stderr
