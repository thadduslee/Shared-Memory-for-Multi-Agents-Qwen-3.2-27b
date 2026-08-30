"""`tests_ok` means the suite passes -- not that some pytest run exited 0.

WHAT BROKE (runs_4iter iteration 1, 2026-08-29). Unable to see past a
2400-character slice of `store.py`, the Developer wrote pytest modules that
dumped 1500-character chunks of the source to disk and then asserted false, so
the runner would surface them:

    def test_dump_retrieve():
        src = pathlib.Path("memory_system/store.py").read_text()
        start = src.index("    def retrieve(")
        pathlib.Path("_retrieve_head.txt").write_text(src[start:start + 1500])
        assert False, "dump written"

It was using the test runner as a file reader. Each run cleared `tests_ok` and
cost a retry, so the workaround for one blindness destroyed the episode: 44
turns, five retries, `compile_ok=False tests_ok=False migration_ok=False`, and
no fix written. The workaround also outlived it -- `test_dump2.py` and
`test_dump_store.py` were inherited by iterations 2 and 3 and re-dumped their
chunks on every later test run.

Two holes, both pinned here. `run_tests` accepted any path, so a NARROW run
decided a gate that means "the suite passes" -- in either direction. And nothing
stopped a test file from failing on purpose. `read_file` now pages, so the
workaround has no reason to exist; this makes sure it cannot happen anyway.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nodes.dev_tools import DevToolbox


@pytest.fixture()
def toolbox(tmp_path: Path) -> DevToolbox:
    box = DevToolbox(workspace=tmp_path)
    (box.workspace / "tests").mkdir()
    (box.workspace / "tests" / "test_real.py").write_text(
        "def test_real():\n    assert True\n", encoding="utf-8")
    return box


def _write(toolbox: DevToolbox, path: str, content: str):
    return asyncio.run(toolbox._t_write_file({"path": path, "content": content}))


def _run_tests(toolbox: DevToolbox, **args):
    return asyncio.run(toolbox._t_run_tests(args))


# ----------------------------------------------------------------------
# 1. a test may not fail on purpose
# ----------------------------------------------------------------------


DUMP_TEST = '''\
import pathlib


def test_dump_retrieve():
    src = pathlib.Path("memory_system/store.py").read_text(encoding="utf-8")
    start = src.index("    def retrieve(")
    pathlib.Path("_retrieve_head.txt").write_text(src[start:start + 1500])
    assert False, "dump written"
'''


def test_the_run_4iter_dump_test_is_refused(toolbox: DevToolbox) -> None:
    result = _write(toolbox, "tests/_dump_retrieve_test.py", DUMP_TEST)

    assert not result.ok
    assert not (toolbox.workspace / "tests" / "_dump_retrieve_test.py").exists()


def test_the_refusal_points_at_the_tool_that_answers_the_question(
    toolbox: DevToolbox,
) -> None:
    """A refusal the model cannot act on just burns a retry."""
    result = _write(toolbox, "tests/_dump_retrieve_test.py", DUMP_TEST)

    assert "read_file" in result.stderr
    assert "offset" in result.stderr


def test_the_refusal_is_a_redirect_not_a_build_failure(toolbox: DevToolbox) -> None:
    """It says nothing about whether the design can be built, so it is free.

    Charging it would let a guard that exists to protect the episode end it --
    the same distinction the write_file rewrite guard draws.
    """
    result = _write(toolbox, "tests/_dump_retrieve_test.py", DUMP_TEST)

    assert result.redirect is True


@pytest.mark.parametrize("body", [
    "def test_x():\n    assert False\n",
    "def test_x():\n    assert False, 'dump written'\n",
    "def test_x():\n    assert 0\n",
])
def test_every_shape_of_deliberate_failure_is_caught(toolbox: DevToolbox, body: str) -> None:
    assert not _write(toolbox, "tests/test_x.py", body).ok


@pytest.mark.parametrize("body", [
    "def test_x():\n    assert value is False\n",
    "def test_x():\n    assert result.ok is False, 'the guard must refuse'\n",
    "def test_x():\n    assert not ok\n",
])
def test_a_legitimate_assertion_about_falsity_is_allowed(
    toolbox: DevToolbox, body: str,
) -> None:
    """`assert x is False` is a real test. Only a bare `assert False` is not."""
    assert _write(toolbox, "tests/test_x.py", body).ok


def test_non_test_code_may_still_say_assert_false(toolbox: DevToolbox) -> None:
    """The guard is about the suite, not about the token."""
    assert _write(toolbox, "memory_system/store.py",
                  "def unreachable():\n    assert False\n").ok


# ----------------------------------------------------------------------
# 2. only the suite sets the suite's gate
# ----------------------------------------------------------------------


def test_a_full_suite_run_is_marked_as_one(toolbox: DevToolbox) -> None:
    for target in ({}, {"path": "tests"}, {"path": "tests/"}):
        assert _run_tests(toolbox, **target).data["full_suite"] is True


def test_a_narrow_run_is_not(toolbox: DevToolbox) -> None:
    assert _run_tests(toolbox, path="tests/test_real.py").data["full_suite"] is False


def test_a_narrow_run_says_it_does_not_set_the_gate(toolbox: DevToolbox) -> None:
    result = _run_tests(toolbox, path="tests/test_real.py")

    assert "does NOT set tests_ok" in result.stdout
    assert '"path": "tests"' in result.stdout, "and names the run that does"


async def test_a_narrow_green_run_cannot_set_the_gate() -> None:
    """The hole: one trivially-passing file could satisfy `tests_ok`."""
    import nodes.developer as dev

    out = await dev.dev_observe({
        "scratchpad": [],
        "observation": {"tool": "run_tests", "ok": True,
                        "data": {"pass_rate": 1.0, "full_suite": False}},
    })

    assert "tests_ok" not in out


async def test_a_narrow_red_run_cannot_clear_it_either() -> None:
    """The other half, and the one that killed iteration 1."""
    import nodes.developer as dev

    out = await dev.dev_observe({
        "scratchpad": [], "tests_ok": True,
        "observation": {"tool": "run_tests", "ok": False,
                        "data": {"pass_rate": 0.0, "full_suite": False}},
    })

    assert "tests_ok" not in out, "a narrow failure is a lead, not a verdict on the suite"


async def test_a_full_suite_run_still_sets_the_gate() -> None:
    import nodes.developer as dev

    out = await dev.dev_observe({
        "scratchpad": [],
        "observation": {"tool": "run_tests", "ok": True,
                        "data": {"pass_rate": 1.0, "full_suite": True}},
    })

    assert out["tests_ok"] is True


async def test_an_observation_without_the_key_is_treated_as_a_suite_run() -> None:
    """Backward compatibility: absent means "not distinguished", not "narrow"."""
    import nodes.developer as dev

    out = await dev.dev_observe({
        "scratchpad": [],
        "observation": {"tool": "run_tests", "ok": True, "data": {"pass_rate": 1.0}},
    })

    assert out["tests_ok"] is True
