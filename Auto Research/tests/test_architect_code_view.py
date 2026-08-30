"""What the Architect is allowed to see before it proposes a design.

WHAT BROKE (smoke run, 2026-08-25). Iteration 1's Architect saw
"(no implementation yet -- this is iteration 1)" and designed a fresh API:
`documents` / `retrieve_documents(user_id)` / `forget_document(id)`. The
Developer was then handed `templates/` -- a green, 18/18-passing store whose
contract is `records` / `MemoryStore.retrieve` / `tombstone` / `is_deleted` /
`Decision` / `Evidence` -- and told to implement the Architect's design while
`run_tests` enforced the template's. It could not satisfy both, broke 3 of the
18 passing tests trying, burned all 5 of MAX_DEV_RETRIES and halted the
iteration before the Evaluator, Judge or Critic ever ran.

The cause is node ordering, not intent: `prepare_workspace` seeds the workspace
inside the DEVELOPER node, which runs after the Architect, so on iteration 1
there is genuinely nothing at `workspace` to look at. The fix is to show the
Architect the template it is really designing against.

TRUNCATION ORDER IS PART OF THE FIX, not a detail. The template is ~47k
characters. Under the old 24k budget and a plain alphabetical walk,
`memory_system/store.py` (22k) consumed nearly all of it and `tests/` fell off
the end -- so the naive version of this fix would have shown the Architect an
implementation with none of the assertions binding it. That is worse than
showing nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config
from nodes.architect import (
    CODE_VIEW_MAX_CHARS,
    _code_view_order,
    _code_view_source,
    _read_only_codebase_view,
)


@pytest.fixture()
def missing_workspace(tmp_path: Path) -> Path:
    """A workspace path that does not exist -- iteration 1, before the Developer."""
    return tmp_path / "iter_1" / "workspace"


# ----------------------------------------------------------------------
# the source: what the view is taken from
# ----------------------------------------------------------------------


def test_iteration_1_view_falls_back_to_the_template(missing_workspace: Path) -> None:
    source, label = _code_view_source(missing_workspace)
    assert source == config.TEMPLATES_DIR
    assert "templates/" in label


def test_a_populated_workspace_wins_over_the_template(tmp_path: Path) -> None:
    """Iteration N must see iteration N-1's code, not the pristine baseline."""
    workspace = tmp_path / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "store.py").write_text("# iteration N-1 code\n")
    source, label = _code_view_source(workspace)
    assert source == workspace
    assert "CURRENT IMPLEMENTATION" in label


def test_no_template_and_no_workspace_reports_honestly(
    missing_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SEED_FROM_TEMPLATE=false is a legitimate (harder) experiment."""
    monkeypatch.setattr(config, "SEED_FROM_TEMPLATE", False)
    source, _ = _code_view_source(missing_workspace)
    assert source is None
    assert "no implementation yet" in _read_only_codebase_view(missing_workspace)


# ----------------------------------------------------------------------
# the order: the contract must survive truncation
# ----------------------------------------------------------------------


def test_tests_are_ordered_before_implementation() -> None:
    paths = [
        Path("memory_system/agent.py"),
        Path("memory_system/store.py"),
        Path("tests/test_rbac.py"),
        Path("memory_system/__init__.py"),
        Path("tests/test_forgetting.py"),
    ]
    ordered = [p.as_posix() for p in _code_view_order(paths)]
    assert ordered[:2] == ["tests/test_forgetting.py", "tests/test_rbac.py"]
    assert all(part.startswith("memory_system/") for part in ordered[2:])


def test_the_contract_survives_a_budget_too_small_for_the_implementation(
    missing_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact regression: squeeze the budget, tests must still be there.

    Under alphabetical order this assertion fails -- store.py eats the budget.
    """
    monkeypatch.setattr("nodes.architect.CODE_VIEW_MAX_CHARS", 14_000)
    view = _read_only_codebase_view(missing_workspace)
    assert "def test_unassigned_clinician_is_denied" in view
    assert "[... code view truncated ...]" in view


# ----------------------------------------------------------------------
# the budget: the whole baseline has to fit
# ----------------------------------------------------------------------


def test_default_budget_fits_the_entire_template(missing_workspace: Path) -> None:
    """If templates/ grows past the budget, this fails before a run pays for it."""
    view = _read_only_codebase_view(missing_workspace)
    assert "[... code view truncated ...]" not in view, (
        "the Architect can no longer see the whole baseline; raise "
        "ARCHITECT_CODE_VIEW_MAX_CHARS or shrink templates/"
    )


def test_the_view_carries_the_contract_the_developer_is_gated_on(
    missing_workspace: Path,
) -> None:
    """Both halves: the assertions, and the API they bind."""
    view = _read_only_codebase_view(missing_workspace)
    # the gate itself
    assert "def test_unassigned_clinician_is_denied" in view
    assert "def test_deletion_request_creates_tombstone" in view
    # the API those tests import, which the design must not rename
    for symbol in ("Decision", "Evidence", "def retrieve", "def tombstone", "def is_deleted"):
        assert symbol in view, symbol


def test_the_view_tells_the_architect_the_tests_are_binding(
    missing_workspace: Path,
) -> None:
    """Seeing the tests is not the same as knowing they cannot be renamed."""
    view = _read_only_codebase_view(missing_workspace)
    assert "contract" in view.lower()
    assert "do not replace it" in view.lower()


def test_budget_is_large_enough_to_be_worth_having() -> None:
    """A regression fence: 24_000 was the value that truncated the baseline."""
    assert CODE_VIEW_MAX_CHARS >= 48_000
