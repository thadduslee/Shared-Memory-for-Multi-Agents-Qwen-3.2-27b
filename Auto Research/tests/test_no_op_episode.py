"""An episode that builds nothing must not report success.

WHAT BROKE (runs_4iter iteration 2, 2026-08-29). The Developer inherited
iteration 1's workspace, ran `compile_check`, `run_tests`, `sql_exec` and the
smoke test on it, saw four green gates, and stopped after seven turns having
written zero bytes. `_stop_reason` returned "gates_green", the orchestrator
recorded a clean episode, and the Evaluator and Judge went on to score code that
was byte-identical to iteration 1's:

    templates/memory_system/store.py   ec0a344aa4b568d3bc594d8ca6ce503d
    iter_1/.../store.py                11ed245c5c3c16c8ddf9786772bc75ab
    iter_2/.../store.py                11ed245c5c3c16c8ddf9786772bc75ab

That was the run's ONLY completed evaluation, so its one real measurement
measured nothing new -- and the summary reported `dev_set_pass_rate: 1.0` beside
`MGS: 0.0`, two true numbers that together read as a working loop.

Green gates on an inherited workspace prove the PREVIOUS iteration compiled.
Nothing else. A snapshot taken when the workspace is handed over is the only way
to tell "built it and it passes" from "ran the gates and left", so the toolbox
takes one, and both exits -- the implicit `gates_green` stop and an explicit
`finish` -- consult it.

Also pinned here: the Developer's own scratch files do not count as building
something, and they do not cross into the next iteration. runs_4iter carried
40-odd `_store_chunk_*.txt` dumps and two source-dumping pytest modules from
iteration 1 into iterations 2 and 3, which re-dumped them on every test run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import nodes.developer as dev
from nodes.dev_tools import DevToolbox, is_scratch


def _seeded(tmp_path: Path) -> Path:
    """A workspace as `prepare_workspace` hands it over: already working code."""
    workspace = tmp_path / "iter_2" / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "memory_system" / "store.py").write_text(
        "def retrieve():\n    return []\n", encoding="utf-8")
    (workspace / "memory_system" / "schema.sql").write_text(
        "CREATE TABLE records (id INTEGER PRIMARY KEY);", encoding="utf-8")
    return workspace


@pytest.fixture()
def box(tmp_path: Path) -> DevToolbox:
    return DevToolbox(workspace=_seeded(tmp_path))


# ----------------------------------------------------------------------
# the snapshot
# ----------------------------------------------------------------------


def test_an_untouched_workspace_reports_no_changes(box: DevToolbox) -> None:
    assert box.substantive_changes() == []
    assert box.built_anything() is False


def test_an_edited_file_is_reported(box: DevToolbox) -> None:
    asyncio.run(box._t_apply_patch({
        "path": "memory_system/store.py",
        "search": "return []", "replace": "return fetch_all()",
    }))

    assert box.substantive_changes() == ["memory_system/store.py"]
    assert box.built_anything() is True


def test_a_new_file_is_reported(box: DevToolbox) -> None:
    asyncio.run(box._t_write_file({"path": "memory_system/rbac.py", "content": "X = 1\n"}))

    assert "memory_system/rbac.py" in box.substantive_changes()


def test_the_baseline_survives_a_rebuilt_toolbox(box: DevToolbox) -> None:
    """`_toolbox` rebuilds the box if migration_sql changes mid-episode.

    A baseline re-taken then would record the work already done as the starting
    state, and the guard would go blind exactly when it was needed.
    """
    asyncio.run(box._t_write_file({"path": "memory_system/rbac.py", "content": "X = 1\n"}))

    rebuilt = DevToolbox(workspace=box.workspace, migration_sql="ALTER TABLE records ADD c TEXT;")

    assert rebuilt.substantive_changes() == ["memory_system/rbac.py"]


def test_the_baseline_is_not_stored_inside_the_workspace(box: DevToolbox) -> None:
    """It must not be inherited, linted, or collected by pytest."""
    assert not (box.workspace / "workspace_baseline.json").exists()
    assert (box.workspace.parent / "workspace_baseline.json").is_file()


# ----------------------------------------------------------------------
# scratch is not work
# ----------------------------------------------------------------------


@pytest.mark.parametrize("path", [
    "_store_chunk_7.txt", "_dump_store.py", "_retrieve.txt",
    "tests/_dump_retrieve_test.py", "_dump2_rbac_5.txt", "_peek_store.py",
])
def test_the_run_4iter_scratch_files_are_recognised(path: str) -> None:
    assert is_scratch(path)


@pytest.mark.parametrize("path", [
    "memory_system/__init__.py", "memory_system/store.py",
    "tests/test_rbac.py", "memory_system/schema.sql",
])
def test_real_source_is_not_mistaken_for_scratch(path: str) -> None:
    assert not is_scratch(path)


def test_dumping_source_to_scratch_is_not_building_something(box: DevToolbox) -> None:
    """The exact move iteration 1 made, 40-odd times."""
    for i in range(3):
        asyncio.run(box._t_write_file({"path": f"_store_chunk_{i}.txt", "content": "chunk"}))

    assert box.substantive_changes() == []
    assert box.built_anything() is False


def test_a_corrected_migration_does_count_as_building_something(box: DevToolbox) -> None:
    """It is the deliverable when the Architect's DDL was the only defect.

    It reaches the next node through state rather than through the workspace, so
    a guard that only watched files would reject a legitimate episode.
    """
    asyncio.run(box._t_sql_exec(
        {"migration_sql": "CREATE INDEX idx_records_patient ON records(id);"}
    ))

    assert box.substantive_changes() == []
    assert box.built_anything() is True


def test_an_empty_migration_override_is_not_a_deliverable(box: DevToolbox) -> None:
    """`{"migration_sql": ""}` says "no migration needed", not "I built one".

    WHAT BROKE (runs_smoke, 2026-08-29). This assertion used to read the other
    way, and it made the whole no-op guard bypassable by accident: the Developer
    read the workspace, ran the four gates, called `sql_exec` with an empty
    override because the work order needed no DDL, and exited `gates_green`
    having written zero bytes. `built_anything()` returned True, so neither the
    implicit exit in `_stop_reason` nor the `finish` guard fired -- and the
    iteration scored code it had not touched, which is the precise failure this
    module exists to prevent.
    """
    asyncio.run(box._t_sql_exec({"migration_sql": ""}))

    assert box.substantive_changes() == []
    assert box.built_anything() is False


# ----------------------------------------------------------------------
# both exits consult it
# ----------------------------------------------------------------------


def _green(workspace: Path, **over) -> dict:
    state = {
        "workspace": str(workspace), "compile_ok": True, "tests_ok": True,
        "migration_ok": True, "smoke_ok": True, "done": False, "retry_count": 0,
        "scratchpad": [],
    }
    state.update(over)
    return state


def test_gates_green_does_not_end_an_episode_that_built_nothing(
    tmp_path: Path, monkeypatch,
) -> None:
    workspace = _seeded(tmp_path)
    box = DevToolbox(workspace=workspace)
    monkeypatch.setitem(dev._TOOLBOXES, str(workspace), box)

    assert dev._stop_reason(_green(workspace)) == "", "the episode must keep working"


def test_gates_green_ends_an_episode_that_did(tmp_path: Path, monkeypatch) -> None:
    workspace = _seeded(tmp_path)
    box = DevToolbox(workspace=workspace)
    monkeypatch.setitem(dev._TOOLBOXES, str(workspace), box)
    asyncio.run(box._t_write_file({"path": "memory_system/rbac.py", "content": "X = 1\n"}))

    assert dev._stop_reason(_green(workspace)) == "gates_green"


async def test_finish_is_rejected_when_nothing_was_built(tmp_path: Path, monkeypatch) -> None:
    """The other door. Green gates plus an explicit `finish` must not pass either."""
    workspace = _seeded(tmp_path)
    monkeypatch.setitem(dev._TOOLBOXES, str(workspace), DevToolbox(workspace=workspace))
    state = _green(workspace, observation={"tool": "finish", "ok": True})

    out = await dev.dev_observe(state)

    assert out["done"] is False
    assert out["retry_count"] == 1
    rejection = out["scratchpad"][-1]["observation"]
    assert "has not changed any .py or .sql file" in rejection
    assert "scratch files and dumped text do not count" in rejection


async def test_finish_is_accepted_when_something_was_built(
    tmp_path: Path, monkeypatch,
) -> None:
    workspace = _seeded(tmp_path)
    box = DevToolbox(workspace=workspace)
    monkeypatch.setitem(dev._TOOLBOXES, str(workspace), box)
    await box._t_write_file({"path": "memory_system/rbac.py", "content": "X = 1\n"})
    state = _green(workspace, observation={"tool": "finish", "ok": True})

    out = await dev.dev_observe(state)

    assert out["done"] is True
    assert out.get("retry_count", 0) == 0


def test_the_guard_fails_open_when_there_is_no_toolbox_to_ask() -> None:
    """An unobservable must not be reported as a failure.

    Both callers run only after compile_check, run_tests and sql_exec have gone
    through DevToolbox, so a missing toolbox is a state the real graph cannot
    reach -- inventing a rejection out of it would be the same class of error
    the guard exists to prevent, pointed the other way.
    """
    assert dev._built_anything({"workspace": "/nowhere/at/all"}) is True


# ----------------------------------------------------------------------
# scratch does not cross the iteration boundary
# ----------------------------------------------------------------------


def test_scratch_files_are_not_inherited_by_the_next_iteration(tmp_path: Path) -> None:
    source = tmp_path / "iter_1" / "workspace"
    (source / "memory_system").mkdir(parents=True)
    (source / "memory_system" / "store.py").write_text("X = 1\n", encoding="utf-8")
    (source / "tests").mkdir()
    (source / "tests" / "test_rbac.py").write_text("def test_x(): pass\n", encoding="utf-8")
    (source / "tests" / "_dump_retrieve_test.py").write_text("assert False\n", encoding="utf-8")
    (source / "_store_chunk_0.txt").write_text("chunk", encoding="utf-8")
    (source / "_dump_store.py").write_text("dump\n", encoding="utf-8")

    destination = tmp_path / "iter_2" / "workspace"
    dev._copy_tree(source, destination)

    inherited = sorted(str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file())
    assert inherited == ["memory_system/store.py", "tests/test_rbac.py"]
