"""Iteration N inherits the BEST code, not the last code.

WHAT THIS IS PINNING. `prepare_workspace` used to seed iteration N from
iteration N-1 unconditionally. That made the loop's lineage follow the most
recent measurement rather than the best one, so a change that lost MGS became
the permanent foundation of everything after it -- and in run-8cf58d33b311 that
compounded four times: 0.3172, then 0.2222, then 0.1830, then 0.1190, each
iteration inheriting the one just measured as worse.

The same unconditional rule also let a build that FAILED become a parent:
iteration 2 of that run hit its turn ceiling having written nothing, and
iteration 3 was seeded from its workspace anyway.

These tests drive the real `prepare_workspace` against a real temporary runs
directory, because the bug was in the interaction between a pure decision
(`scoreboard.champion_iteration`) and a filesystem operation, and only one of
those two halves can be tested without the disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import config
import scoreboard
from nodes.developer import prepare_workspace


@pytest.fixture
def runs(tmp_path, monkeypatch) -> Path:
    """A runs directory whose iteration workspaces this test writes by hand."""
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(config, "SEED_FROM_TEMPLATE", False)
    return tmp_path


def seed_iteration(runs: Path, iteration: int, marker: str) -> Path:
    """An iteration workspace whose contents identify which iteration it is."""
    workspace = runs / f"iter_{iteration}" / "workspace"
    (workspace / "memory_system").mkdir(parents=True, exist_ok=True)
    (workspace / "memory_system" / "store.py").write_text(
        f'MARKER = "{marker}"\n', encoding="utf-8"
    )
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    return workspace


def marker_of(workspace: Path) -> str:
    return (workspace / "memory_system" / "store.py").read_text(encoding="utf-8").strip()


# ======================================================================
# the ordinary case: nothing changes on a healthy run
# ======================================================================


def test_an_improving_run_still_inherits_the_previous_iteration(runs: Path) -> None:
    """THE ROLLBACK MUST BE INVISIBLE WHEN THE LOOP IS WORKING.

    If it were not, this change would be altering every run rather than only
    the ones that walked away from their own best answer.
    """
    seed_iteration(runs, 1, "iter1")
    seed_iteration(runs, 2, "iter2")
    history = [
        scoreboard.score_row(iteration=1, stage="dev", phase="p",
                             utility=0.3, access=0.1, forgetting=0.1, mgs=0.243),
        scoreboard.score_row(iteration=2, stage="dev", phase="p",
                             utility=0.6, access=0.1, forgetting=0.1, mgs=0.486),
    ]
    parent = scoreboard.champion_iteration({"score_history": history}, fallback=2)
    workspace, provenance = prepare_workspace(3, parent=parent)
    assert provenance == "iter_2"
    assert marker_of(workspace) == 'MARKER = "iter2"'


def test_the_default_parent_is_still_the_previous_iteration(runs: Path) -> None:
    """`parent=None` reproduces the old behaviour exactly, for every caller
    that has no score history to consult."""
    seed_iteration(runs, 1, "iter1")
    workspace, provenance = prepare_workspace(2)
    assert provenance == "iter_1"
    assert marker_of(workspace) == 'MARKER = "iter1"'


# ======================================================================
# the regression case: the whole point
# ======================================================================


def test_a_regressed_run_rolls_back_to_the_champions_code(runs: Path) -> None:
    """run-8cf58d33b311's iteration 6, as it would run now.

    The old rule would have handed it iteration 5's workspace -- MGS 0.1190,
    the worst code the run produced. It gets iteration 1's instead.
    """
    for iteration, marker in ((1, "iter1"), (3, "iter3"), (4, "iter4"), (5, "iter5")):
        seed_iteration(runs, iteration, marker)
    history = [
        scoreboard.score_row(iteration=1, stage="dev", phase="p",
                             utility=0.4444, access=0.1765, forgetting=0.1333, mgs=0.3172),
        scoreboard.score_row(iteration=3, stage="dev", phase="p",
                             utility=0.2222, access=0.0, forgetting=0.0, mgs=0.2222),
        scoreboard.score_row(iteration=4, stage="dev", phase="p",
                             utility=0.2222, access=0.1176, forgetting=0.0667, mgs=0.1830),
        scoreboard.score_row(iteration=5, stage="dev", phase="p",
                             utility=0.1667, access=0.1765, forgetting=0.1333, mgs=0.1190),
    ]
    parent = scoreboard.champion_iteration({"score_history": history}, fallback=5)
    assert parent == 1
    workspace, provenance = prepare_workspace(6, parent=parent)
    assert provenance == "iter_1"
    assert marker_of(workspace) == 'MARKER = "iter1"'


def test_a_failed_build_is_skipped_as_a_parent(runs: Path) -> None:
    """Iteration 2 built nothing and was never judged.

    Under the old rule iteration 3 was seeded from it regardless. With no score
    row it is not a champion candidate, so the lineage steps over it.
    """
    seed_iteration(runs, 1, "iter1")
    seed_iteration(runs, 2, "broken")   # the failed build's leftover workspace
    history = [scoreboard.score_row(iteration=1, stage="dev", phase="p",
                                    utility=0.4, access=0.1, forgetting=0.1, mgs=0.324)]
    parent = scoreboard.champion_iteration({"score_history": history}, fallback=2)
    workspace, provenance = prepare_workspace(3, parent=parent)
    assert provenance == "iter_1"
    assert marker_of(workspace) == 'MARKER = "iter1"'


def test_a_rollback_logs_a_warning_naming_both_iterations(runs, caplog) -> None:
    """A silent rollback is a rollback nobody can audit."""
    import logging

    seed_iteration(runs, 1, "iter1")
    seed_iteration(runs, 2, "iter2")
    with caplog.at_level(logging.WARNING, logger="orchestrator.developer"):
        prepare_workspace(3, parent=1)
    messages = [record.getMessage() for record in caplog.records]
    assert any("ROLLED BACK" in message for message in messages), caplog.text
    assert any("seeded from iteration 1" in message and "instead of iteration 2" in message
               for message in messages), caplog.text


# ======================================================================
# the edges
# ======================================================================


def test_an_already_populated_workspace_is_left_alone(runs: Path) -> None:
    """A resumed run, and an infrastructure retry of the same iteration.

    Re-seeding here would throw away work the episode had already landed, and
    re-taking the baseline would blind the did-anything-change guard.
    """
    seed_iteration(runs, 1, "iter1")
    existing = seed_iteration(runs, 2, "work-in-progress")
    workspace, provenance = prepare_workspace(2, parent=1)
    assert provenance == "existing"
    assert workspace == existing
    assert marker_of(workspace) == 'MARKER = "work-in-progress"'


def test_iteration_one_falls_through_to_the_template(runs, monkeypatch) -> None:
    monkeypatch.setattr(config, "SEED_FROM_TEMPLATE", True)
    workspace, provenance = prepare_workspace(1, parent=0)
    assert provenance == "template"
    assert (workspace / "memory_system" / "store.py").is_file()


def test_a_champion_whose_workspace_is_gone_falls_back_to_the_template(
    runs, monkeypatch
) -> None:
    """RUNS_DIR can be repointed or partly wiped between turns.

    Pointing at a directory that no longer exists must degrade to the template
    rather than raise on a routing path.
    """
    monkeypatch.setattr(config, "SEED_FROM_TEMPLATE", True)
    workspace, provenance = prepare_workspace(4, parent=2)  # iter_2 was never written
    assert provenance == "template"
    assert (workspace / "memory_system" / "store.py").is_file()


def test_the_baseline_marker_is_cleared_when_the_workspace_is_reseeded(runs: Path) -> None:
    """A stale baseline describes a workspace that no longer exists."""
    seed_iteration(runs, 1, "iter1")
    stale = runs / "iter_2" / "workspace_baseline.json"
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_text("{}", encoding="utf-8")
    prepare_workspace(2, parent=1)
    assert not stale.exists()


# ======================================================================
# the plateau case: ROLLBACK_TO_BEST=false, strict linear N<-N-1
# ======================================================================


def test_a_plateau_re_seeds_every_iteration_from_the_same_parent(runs: Path) -> None:
    """WHY THE LINEAR SETTING EXISTS, on run-b5d7565ddb4a's real numbers.

    A tie is never adopted, so a run that stops improving stops moving its
    champion, and every later iteration is seeded from the same workspace.
    Iterations 17 and 18 of that run are siblings rather than successors: both
    are iteration 4 plus one patch, and 18 contains none of 17's work.
    """
    for iteration, marker in ((4, "iter4"), (17, "iter17")):
        seed_iteration(runs, iteration, marker)
    history = [
        scoreboard.score_row(iteration=4, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
        scoreboard.score_row(iteration=17, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
    ]
    parent = scoreboard.champion_iteration({"score_history": history}, fallback=17)
    assert parent == 4, "a tie must not move the champion"
    workspace, provenance = prepare_workspace(18, parent=parent)
    assert provenance == "iter_4"
    assert marker_of(workspace) == 'MARKER = "iter4"'


def test_linear_lineage_advances_through_a_plateau(runs: Path, monkeypatch) -> None:
    """THE SETTING, on the same numbers: 18 inherits 17, not the champion."""
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    for iteration, marker in ((4, "iter4"), (17, "iter17")):
        seed_iteration(runs, iteration, marker)
    history = [
        scoreboard.score_row(iteration=4, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
        scoreboard.score_row(iteration=17, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
    ]
    parent = scoreboard.champion_iteration({"score_history": history}, fallback=17)
    assert parent == 17
    workspace, provenance = prepare_workspace(18, parent=parent)
    assert provenance == "iter_17"
    assert marker_of(workspace) == 'MARKER = "iter17"'


def test_linear_lineage_still_reports_the_best_iteration(runs: Path, monkeypatch) -> None:
    """The flag decides what is INHERITED, never what is REPORTED.

    `best_of` and the run summary are pure functions of the score history and
    must return iteration 4 either way -- that is the whole premise of running
    linear lineage while still tracking a high-water mark.
    """
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    history = [
        scoreboard.score_row(iteration=4, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
        scoreboard.score_row(iteration=17, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
        scoreboard.score_row(iteration=18, stage="dev", phase="p",
                             utility=0.5556, access=0.1176, forgetting=0.0, mgs=0.490196),
    ]
    assert scoreboard.best_of(history) == (0.539216, 4)
    summary = scoreboard.summary({"score_history": history})
    assert summary["best_mgs"] == 0.539216
    assert summary["best_iteration"] == 4
    assert summary["final_iteration"] == 18
    assert summary["regressed_from_best"] is True


def test_linear_lineage_does_not_log_a_rollback(runs: Path, monkeypatch, caplog) -> None:
    """`ROLLED BACK` in the log must mean a rollback actually happened."""
    import logging

    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    seed_iteration(runs, 17, "iter17")
    with caplog.at_level(logging.WARNING, logger="orchestrator.developer"):
        _, provenance = prepare_workspace(18, parent=17)
    assert provenance == "iter_17"
    assert not any("ROLLED BACK" in record.getMessage() for record in caplog.records), caplog.text


def test_linear_lineage_steps_over_a_failed_build(runs: Path, monkeypatch) -> None:
    """THE HOLE STRICT N-1 WOULD OPEN, and the default that closes it.

    Iteration 14 of run-b5d7565ddb4a ended `2 failed, 38 passed` and was never
    judged, but its workspace is on disk like any other. Unconditional N-1 would
    make that tree the foundation of iteration 15 and of everything after it.
    `LINEAGE_SKIP_FAILED_BUILDS` seeds from the most recent iteration that
    actually built instead -- 15 <- 13 -- so the lineage still compounds without
    compounding onto code whose tests do not pass.
    """
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    for iteration, marker in ((4, "iter4"), (13, "iter13"), (14, "broken")):
        seed_iteration(runs, iteration, marker)
    history = [
        scoreboard.score_row(iteration=4, stage="dev", phase="p",
                             utility=0.6111, access=0.1176, forgetting=0.0, mgs=0.539216),
        scoreboard.score_row(iteration=13, stage="dev", phase="p",
                             utility=0.5556, access=0.1176, forgetting=0.0, mgs=0.490196),
    ]
    parent, reason = scoreboard.lineage_parent({"score_history": history}, fallback=14)
    assert (parent, reason) == (13, scoreboard.LINEAGE_SKIPPED_FAILED)
    workspace, provenance = prepare_workspace(15, parent=parent, reason=reason)
    assert provenance == "iter_13", "13 built; 4 merely scored higher"
    assert marker_of(workspace) == 'MARKER = "iter13"'


def test_the_skip_is_not_logged_as_a_rollback(runs: Path, monkeypatch, caplog) -> None:
    """`ROLLED BACK` in the log must mean a SCORE rollback.

    Nothing scored worse when a build fails -- it produced no score at all --
    and a post-mortem that reads the two as the same event invents a regression
    that never happened.
    """
    import logging

    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    seed_iteration(runs, 13, "iter13")
    with caplog.at_level(logging.WARNING, logger="orchestrator.developer"):
        prepare_workspace(15, parent=13, reason=scoreboard.LINEAGE_SKIPPED_FAILED)
    messages = [record.getMessage() for record in caplog.records]
    assert any("SKIPPED A FAILED BUILD" in message for message in messages), caplog.text
    assert not any("ROLLED BACK" in message for message in messages), caplog.text
    assert any("its failure report still goes forward" in message for message in messages)


def test_unconditional_lineage_still_inherits_the_failed_build(
    runs: Path, monkeypatch
) -> None:
    """The old behaviour stays reachable, and stays a deliberate choice.

    `LINEAGE_SKIP_FAILED_BUILDS=false` is genuinely unconditional N-1, broken
    parents included. Pinned so that the default above is provably doing
    something rather than being a no-op.
    """
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    monkeypatch.setattr(config, "LINEAGE_SKIP_FAILED_BUILDS", False)
    for iteration, marker in ((13, "iter13"), (14, "broken")):
        seed_iteration(runs, iteration, marker)
    history = [scoreboard.score_row(iteration=13, stage="dev", phase="p",
                                    utility=0.5556, access=0.1176, forgetting=0.0, mgs=0.490196)]
    parent, reason = scoreboard.lineage_parent({"score_history": history}, fallback=14)
    assert (parent, reason) == (14, scoreboard.LINEAGE_DIRECT)
    workspace, provenance = prepare_workspace(15, parent=parent, reason=reason)
    assert provenance == "iter_14"
    assert marker_of(workspace) == 'MARKER = "broken"'


def test_a_degraded_iteration_is_still_a_valid_parent(runs: Path, monkeypatch) -> None:
    """A degraded row is a bad MEASUREMENT, not a bad BUILD.

    `best_row` excludes degraded rows because their scores cannot honestly be
    compared. The lineage question is different and must not borrow that answer:
    the answerer being unreachable says nothing about whether the code compiles,
    and that workspace built and was evaluated like any other.
    """
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    for iteration, marker in ((12, "iter12"), (13, "degraded")):
        seed_iteration(runs, iteration, marker)
    history = [
        scoreboard.score_row(iteration=12, stage="dev", phase="p",
                             utility=0.5, access=0.1, forgetting=0.0, mgs=0.45),
        scoreboard.score_row(iteration=13, stage="dev", phase="p",
                             utility=0.9, access=0.0, forgetting=0.0, mgs=0.9,
                             degraded=True),
    ]
    parent, reason = scoreboard.lineage_parent({"score_history": history}, fallback=14)
    assert (parent, reason) == (13, scoreboard.LINEAGE_SKIPPED_FAILED)
    _, provenance = prepare_workspace(15, parent=parent, reason=reason)
    assert provenance == "iter_13"
