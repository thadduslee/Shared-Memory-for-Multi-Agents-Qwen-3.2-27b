"""`sql_exec` must expose a gate the Developer can actually clear.

WHAT BROKE (observed live). The Architect emitted a placeholder migration:

    CREATE TABLE principals (...);
    CREATE TABLE role_assignments (...);

`sql_exec` applies `memory_system/schema.sql` first and the migration second, so
this failed with a bare "table principals already exists". The migration is the
ARCHITECT's -- nothing in the workspace -- so the Developer could not fix it with
`write_file`, and the persona documented `sql_exec` as taking `args = {}` with no
override. `migration_ok` was therefore UNCLEARABLE: the Developer would burn all
of MAX_DEV_RETRIES and route back to the Architect having built nothing.

The tool already accepted an `args.migration_sql` override; it was simply
undocumented and its empty-string case was wrong. Both are fixed, and these
tests pin the three paths.
"""

from __future__ import annotations

import asyncio

import pytest

from harness.profiles import DEVELOPER_PROFILE
from nodes.dev_tools import DevToolbox

SCHEMA = "CREATE TABLE principals (id TEXT PRIMARY KEY);"

# The real placeholder migration, verbatim.
PLACEHOLDER_MIGRATION = (
    "-- Initial migration: schema creation\n"
    "PRAGMA journal_mode=WAL;\n"
    "CREATE TABLE principals (...);\n"
)


@pytest.fixture()
def box(tmp_path):
    (tmp_path / "memory_system").mkdir()
    (tmp_path / "memory_system" / "schema.sql").write_text(SCHEMA, encoding="utf-8")
    return DevToolbox(tmp_path, migration_sql=PLACEHOLDER_MIGRATION, iteration=1)


def call(box, args):
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        box.call("sql_exec", args))


def test_a_broken_architect_migration_still_fails(box) -> None:
    """The gate must not be weakened -- a bad migration is still a red gate."""
    assert not call(box, {}).ok


def test_the_failure_names_the_statement_and_its_source(box) -> None:
    """A bare sqlite message sends the Developer hunting in the wrong file."""
    result = call(box, {})
    assert "CREATE TABLE principals (...)" in result.stderr
    assert "Architect's migration_sql" in result.stderr
    assert "migration_sql" in result.stderr, "the override must be discoverable"


def test_an_explicit_empty_override_means_no_migration_needed(box) -> None:
    """KEY PRESENCE, not truthiness.

    Treating `""` as absent would silently fall back to the Architect's script --
    re-running the statements the Developer just declared unnecessary, so the
    gate could never be cleared this way.
    """
    result = call(box, {"migration_sql": ""})
    assert result.ok
    assert result.data["applied"] == 0


def test_a_corrected_override_applies_and_greens_the_gate(box) -> None:
    result = call(box, {"migration_sql": "CREATE INDEX i ON principals(id);"})
    assert result.ok
    assert result.data["applied"] == 1
    assert "i" in result.data["indexes"]


def test_the_architect_migration_is_used_when_no_override_is_given(tmp_path) -> None:
    (tmp_path / "memory_system").mkdir()
    (tmp_path / "memory_system" / "schema.sql").write_text(SCHEMA, encoding="utf-8")
    good = DevToolbox(tmp_path, migration_sql="CREATE INDEX i ON principals(id);", iteration=1)
    result = call(good, {})
    assert result.ok
    assert result.data["applied"] == 1


def test_a_missing_schema_is_reported_not_raised(tmp_path) -> None:
    result = call(DevToolbox(tmp_path, migration_sql="", iteration=1), {})
    assert not result.ok
    assert "schema.sql does not exist" in result.stderr


def test_the_schema_documents_the_override() -> None:
    """The capability is only useful if the model is told it exists.

    That telling moved: the Developer calls tools against a real schema now, so
    the place a per-argument affordance has to be described is the argument's
    own description -- which is the text the model reads while filling it in --
    rather than a paragraph of persona it may or may not connect to the call.
    """
    from nodes.dev_tools import TOOL_SCHEMAS

    sql_exec = next(
        schema["function"] for schema in TOOL_SCHEMAS
        if schema["function"]["name"] == "sql_exec"
    )
    assert "migration_sql" in sql_exec["parameters"]["properties"]
    assert "migration_sql" not in sql_exec["parameters"].get("required", []), (
        "the override must be optional; the Architect's migration is the default")
    described = sql_exec["description"] + str(sql_exec["parameters"])
    assert "no migration needed" in described


# ======================================================================
# The full recovery, through the real loop
# ======================================================================


async def test_the_developer_clears_every_gate_despite_a_broken_migration(
    monkeypatch, tmp_path
) -> None:
    """End to end: a placeholder migration must no longer be a dead end.

    Before, `migration_ok` could not be cleared by any action available to the
    Developer, so this episode ended in `developer exhausted` with a design that
    was never actually attempted. The reply script below mixes in the two
    degenerate shapes seen live, so the resample path is exercised too.
    """
    import config as cfg
    import nodes.developer as dev
    from harness.dsh_client import DSHResult

    workspace = tmp_path / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "schema.sql").write_text(SCHEMA, encoding="utf-8")
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "tests").mkdir()
    (workspace / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n",
                                                    encoding="utf-8")

    replies = [
        " ",                                    # degenerate: lone space
        'Thought: compile.\n```json\n{"tool": "compile_check", "args": {}}\n```',
        'Thought: migrate.\n```json\n{"tool": "sql_exec", "args": {}}\n```',
        # Having seen the failure, supply a corrected migration. It has to be
        # REAL SQL: an empty override is this tool's documented way to say "no
        # migration is needed", and an episode whose only output was that has
        # built nothing -- `finish` is refused for it (test_no_op_episode.py).
        ('Thought: the Architect\'s migration is a placeholder.\n```json\n'
         '{"tool": "sql_exec", "args": {"migration_sql": '
         '"CREATE INDEX idx_principals_id ON principals(id);"}}\n```'),
        'Thought: test.\n```json\n{"tool": "run_tests", "args": {"path": "tests"}}\n```',
        'Thought: done.\n```json\n{"tool": "finish", "args": {"summary": "green"}}\n```',
    ]
    seen = {"n": 0}

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        text = replies[min(seen["n"], len(replies) - 1)]
        seen["n"] += 1
        return DSHResult(ok=bool(text.strip()), text=text, profile=profile.name,
                         finish_reason="stop", usage={"total_tokens": 10},
                         error=None if text.strip() else "empty response")

    monkeypatch.setattr(dev, "agent_call", scripted)
    monkeypatch.setattr(cfg, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(cfg, "MOCK_MODE", False)

    state = {
        "instructions": "apply the migration", "sql_schema": SCHEMA,
        "migration_sql": PLACEHOLDER_MIGRATION, "workspace": str(workspace),
        "iteration": 1, "workspace_from": "template", "scratchpad": [],
        "retry_count": 0, "compile_ok": False, "tests_ok": False,
        "migration_ok": False, "lint_ok": False, "smoke_ok": False, "done": False,
        "pass_rate": 0.0, "last_stack_trace": None,
    }
    final = await dev.DeveloperSession(state).run()

    assert final["compile_ok"], "compile gate"
    assert final["migration_ok"], "migration gate -- this is the one that was unclearable"
    assert final["tests_ok"], "tests gate"
    assert final["done"], "finish must be accepted once all three are green"
    tools = [s["action"].get("tool") for s in final["scratchpad"]]
    assert "sql_exec" in tools
