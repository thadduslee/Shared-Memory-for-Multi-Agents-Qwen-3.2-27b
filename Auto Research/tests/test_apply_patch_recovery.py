"""A search block that does not match must be a step forward, not a dead end.

WHAT BROKE (runs_4iter iteration 3, 2026-08-29). The Developer tried to edit
`memory_system/store.py` with this search block:

    terms = _distinctive_terms(query)
    decision = Decision(query_terms=sorted(terms))

Those two lines are both in the file -- at 405 and 407, with

    term_hashes = _term_hashes(query)

between them. The Developer had never seen line 406, because `read_file` only
ever showed it a 2400-character slice of a 26KB file, so it was reconstructing
the block from memory. `apply_patch` answered "search block not found in file"
and nothing else, which told it exactly nothing it did not already know, so it
guessed again -- the same block at steps 67 and 72, variants at 65 and 69, and a
unified diff with no @@ headers at 71. Five failures, retry cap gone, `store.py`
never patched. The gates were all green the whole time, because it had changed
nothing, and the run halted there.

Three things are pinned here, in the order they would have stopped it:

  1. whitespace-tolerant matching, so a block that differs only in indentation
     applies instead of failing;
  2. a failure that PRINTS the real lines, so the next attempt can copy them;
  3. a hard stop on re-sending a call that has already failed twice, so an
     episode cannot spend its whole budget on one wrong guess.

Fuzz is deliberately conservative -- unique whole-line matches only. `patch(1)`
corrupts files by picking one of several candidates, and here that would be
inherited: `prepare_workspace` seeds iteration N from iteration N-1.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nodes.dev_tools import DevToolbox

# The real thing, near enough: the three lines and the shape around them.
STORE = '''\
"""SQL-backed, RBAC-filtered, tombstone-aware memory store."""


class MemoryStore:
    def retrieve(self, query, requester, role, episode):
        """Return the records this requester may see."""
        terms = _distinctive_terms(query)
        term_hashes = _term_hashes(query)
        decision = Decision(query_terms=sorted(terms))

        sql = (
            "SELECT r.record_id FROM records r "
        )
        return decision
'''


@pytest.fixture()
def toolbox(tmp_path: Path) -> DevToolbox:
    box = DevToolbox(workspace=tmp_path)
    (box.workspace / "store.py").write_text(STORE, encoding="utf-8")
    return box


def _patch(toolbox: DevToolbox, **args):
    return asyncio.run(toolbox._t_apply_patch({"path": "store.py", **args}))


# ----------------------------------------------------------------------
# 1. exact matching is unchanged
# ----------------------------------------------------------------------


def test_an_exact_block_still_applies(toolbox: DevToolbox) -> None:
    result = _patch(
        toolbox,
        search="        terms = _distinctive_terms(query)\n"
               "        term_hashes = _term_hashes(query)",
        replace="        terms = _distinctive_terms(query)\n"
                "        term_hashes = _term_hashes(query, fold_case=True)",
    )

    assert result.ok
    assert "fold_case=True" in (toolbox.workspace / "store.py").read_text()


# ----------------------------------------------------------------------
# 2. the block the run actually sent
# ----------------------------------------------------------------------


MISSING_MIDDLE_LINE = (
    "        terms = _distinctive_terms(query)\n"
    "        decision = Decision(query_terms=sorted(terms))"
)


def test_the_run_4iter_block_still_does_not_apply(toolbox: DevToolbox) -> None:
    """It must NOT fuzzy-match: a line it never saw sits inside the span.

    Applying it would silently delete `term_hashes = _term_hashes(query)`, and
    iteration 4 would inherit the deletion.
    """
    before = (toolbox.workspace / "store.py").read_text()

    result = _patch(toolbox, search=MISSING_MIDDLE_LINE, replace="        pass")

    assert not result.ok
    assert (toolbox.workspace / "store.py").read_text() == before


def test_the_failure_prints_the_lines_that_are_really_there(toolbox: DevToolbox) -> None:
    """The whole point: an error the model can act on."""
    result = _patch(toolbox, search=MISSING_MIDDLE_LINE, replace="        pass")

    assert "term_hashes = _term_hashes(query)" in result.stderr, (
        "the line it could not see must appear in the error that names the problem")
    assert "store.py lines" in result.stderr
    assert "Re-sending the same search block will fail the same way" in result.stderr


def test_the_failure_keeps_its_head_in_the_scratchpad(toolbox: DevToolbox) -> None:
    """The near-miss block is printed first, so tail truncation would eat it."""
    result = _patch(toolbox, search=MISSING_MIDDLE_LINE, replace="        pass")

    assert result.keep == "head"
    assert "term_hashes" in result.observation()


# ----------------------------------------------------------------------
# 3. whitespace fuzz, bounded by uniqueness
# ----------------------------------------------------------------------


def test_a_block_that_differs_only_in_indentation_applies(toolbox: DevToolbox) -> None:
    result = _patch(
        toolbox,
        search="terms = _distinctive_terms(query)\nterm_hashes = _term_hashes(query)",
        replace="        terms = _distinctive_terms(query, expand=True)\n"
                "        term_hashes = _term_hashes(query)",
    )

    assert result.ok, "indentation drift is the commonest near-miss and is recoverable"
    assert "expand=True" in (toolbox.workspace / "store.py").read_text()


def test_an_ambiguous_block_is_refused_rather_than_guessed(toolbox: DevToolbox) -> None:
    """Fuzz that picks one of two candidates is how patch(1) corrupts files.

    The indentation differs, so this can only reach the whitespace-tolerant
    path -- where two candidates must mean refusal, not a coin flip.
    """
    (toolbox.workspace / "dup.py").write_text(
        "def a():\n        value = 1\n        return value\n\n\n"
        "def b():\n            value = 1\n            return value\n",
        encoding="utf-8")
    before = (toolbox.workspace / "dup.py").read_text()

    # Two lines with a newline between them: no indentation of this block is a
    # substring of either copy, so the exact path cannot fire.
    result = asyncio.run(toolbox._t_apply_patch(
        {"path": "dup.py", "search": "value = 1\nreturn value", "replace": "    return 2"}))

    assert not result.ok, "two candidates means the tool cannot know which was meant"
    assert (toolbox.workspace / "dup.py").read_text() == before


def test_fuzz_replaces_only_the_matched_lines(toolbox: DevToolbox) -> None:
    """The span is the matched lines, not the region around them."""
    result = _patch(
        toolbox,
        search="term_hashes = _term_hashes(query)",
        replace="        term_hashes = _term_hashes(query, strict=True)",
    )
    after = (toolbox.workspace / "store.py").read_text()

    assert result.ok
    assert "strict=True" in after
    assert "        terms = _distinctive_terms(query)\n" in after, "the line above survives"
    assert "        decision = Decision(query_terms=sorted(terms))\n" in after, "and below"
    assert after.count("_term_hashes") == 1, "one call site, not two"


# ----------------------------------------------------------------------
# 4. the loop refuses to re-send a call that already failed twice
# ----------------------------------------------------------------------


async def test_a_thrice_repeated_failed_patch_is_short_circuited(tmp_path) -> None:
    import nodes.developer as dev

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "store.py").write_text(STORE, encoding="utf-8")
    args = {"path": "store.py", "search": MISSING_MIDDLE_LINE, "replace": "        pass"}
    failed = {"tool": "apply_patch", "args": args}
    state = {
        "workspace": str(workspace),
        "scratchpad": [
            {"action": failed, "observation": "FAILED apply_patch: ...", "ok": False},
            {"action": {"tool": "read_file", "args": {"path": "store.py"}},
             "observation": "OK read_file: ...", "ok": True},
            {"action": failed, "observation": "FAILED apply_patch: ...", "ok": False},
        ],
        "action": failed,
    }

    out = await dev.dev_act(state)

    assert out["observation"]["ok"] is False
    assert out["observation"]["data"]["repeated_failed_edit"] is True
    assert "already failed twice" in out["observation"]["text"]
    assert "read_file" in out["observation"]["text"], "it must be told what to do instead"


async def test_the_run_is_counted_across_the_whole_episode_not_just_consecutively(
    tmp_path,
) -> None:
    """runs_4iter interleaved the identical patch with other attempts.

    A strictly consecutive check would have seen a run of one every time and
    never fired.
    """
    import nodes.developer as dev

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    args = {"path": "store.py", "search": "a", "replace": "b"}
    failed = {"tool": "apply_patch", "args": args}
    other = {"tool": "apply_patch", "args": {"path": "store.py", "search": "c", "replace": "d"}}

    assert dev._identical_failed_edits(
        {"scratchpad": [
            {"action": failed, "ok": False},
            {"action": other, "ok": False},
            {"action": failed, "ok": False},
        ]}, "apply_patch", args) == 2


async def test_a_patch_that_succeeded_before_is_not_short_circuited(tmp_path) -> None:
    """Only FAILED repeats count; re-applying after a change is legitimate."""
    import nodes.developer as dev

    args = {"path": "store.py", "search": "a", "replace": "b"}
    green = {"action": {"tool": "apply_patch", "args": args}, "ok": True}

    assert dev._identical_failed_edits(
        {"scratchpad": [green, green, green]}, "apply_patch", args) == 0
