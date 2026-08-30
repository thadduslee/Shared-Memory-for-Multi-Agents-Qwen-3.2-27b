"""The Developer must not destroy a file by rewriting it from memory.

WHAT BROKE (runs_verify3, 2026-08-26). The Developer replaced the
22274-character `memory_system/store.py` with a 13319-character regeneration --
ONE `write_file` against five `apply_patch` calls in the same episode. It was
not trying to delete anything; it simply cannot reproduce 22k characters
faithfully, so the rewrite silently dropped `author_id` from the INSERT (every
test then died on `sqlite3.IntegrityError: NOT NULL constraint failed:
records.author_id`) and the `DEFAULT_ROLE_GRANTS` ancillary-care-team rows that
an earlier run had added to fix 9 of 18 utility failures. Four follow-up
`apply_patch` calls chased the symptom and restored neither. The Developer
exhausted MAX_DEV_RETRIES and the iteration ended with `tests_ok=False`, so the
Evaluator, Judge and Critic never ran.

WHY IT IS ENFORCED IN THE TOOL. The damage compounds: `prepare_workspace` seeds
iteration N from iteration N-1, so one destructive rewrite is inherited by every
later iteration -- the opposite of the lineage that makes the loop
self-improving. And per README section 4, this project's position is that a
capability boundary is structural, not instructional: telling the prompt not to
lose code is not a mechanism.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from nodes.dev_tools import DevToolbox


@pytest.fixture()
def toolbox(tmp_path: Path) -> DevToolbox:
    return DevToolbox(workspace=tmp_path)


def _write(toolbox: DevToolbox, path: str, content: str, **extra: object):
    return asyncio.run(toolbox._t_write_file({"path": path, "content": content, **extra}))


def _big(marker: str = "x") -> str:
    """A file comfortably over the guard's minimum size."""
    return f"# {marker}\n" + "".join(f"line_{i} = {i}\n" for i in range(600))


# ----------------------------------------------------------------------
# the refusal
# ----------------------------------------------------------------------


def test_destructive_rewrite_of_a_large_file_is_refused(toolbox: DevToolbox) -> None:
    target = toolbox.workspace / "store.py"
    target.write_text(_big())
    original = target.read_text()

    result = _write(toolbox, "store.py", "# regenerated from memory\n")

    assert not result.ok
    assert target.read_text() == original, "the file must be left untouched"


def test_the_refusal_names_apply_patch_and_the_force_escape(toolbox: DevToolbox) -> None:
    """A refusal the model cannot act on just burns a retry."""
    (toolbox.workspace / "store.py").write_text(_big())
    result = _write(toolbox, "store.py", "# tiny\n")
    assert "apply_patch" in result.stderr
    assert "force" in result.stderr


def test_the_refusal_reports_both_sizes(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "store.py").write_text(_big())
    result = _write(toolbox, "store.py", "# tiny\n")
    assert result.data["existing_chars"] > result.data["proposed_chars"]


def test_a_refused_write_is_not_recorded_as_written(toolbox: DevToolbox) -> None:
    """`files_written` feeds the codebase delta; a refusal is not a change."""
    (toolbox.workspace / "store.py").write_text(_big())
    _write(toolbox, "store.py", "# tiny\n")
    assert "store.py" not in toolbox.files_written


# ----------------------------------------------------------------------
# what must still be allowed
# ----------------------------------------------------------------------


def test_a_new_file_is_always_allowed(toolbox: DevToolbox) -> None:
    """The guard is about LOSS; there is nothing to lose in a new file."""
    result = _write(toolbox, "memory_system/new_module.py", "# brand new\n")
    assert result.ok
    assert (toolbox.workspace / "memory_system" / "new_module.py").is_file()


def test_a_small_file_may_be_rewritten_freely(toolbox: DevToolbox) -> None:
    """Below the size floor a rewrite is cheap to re-derive; the guard is noise."""
    target = toolbox.workspace / "__init__.py"
    target.write_text("from .store import MemoryStore\n")
    assert _write(toolbox, "__init__.py", "# replaced\n").ok


def test_growing_a_large_file_is_allowed(toolbox: DevToolbox) -> None:
    """Adding code is the normal case and must not trip the guard."""
    target = toolbox.workspace / "store.py"
    target.write_text(_big())
    assert _write(toolbox, "store.py", _big() + "\ndef added(): ...\n").ok


def test_a_modest_shrink_is_allowed(toolbox: DevToolbox) -> None:
    """Deleting a few lines is legitimate refactoring, not memory loss."""
    target = toolbox.workspace / "store.py"
    body = _big()
    target.write_text(body)
    assert _write(toolbox, "store.py", body[: int(len(body) * 0.9)]).ok


@pytest.mark.parametrize("escape", ["force", "replace"])
def test_an_explicit_force_still_permits_a_full_rewrite(
    toolbox: DevToolbox, escape: str
) -> None:
    """The guard makes intent visible; it does not remove the capability."""
    target = toolbox.workspace / "store.py"
    target.write_text(_big())
    result = _write(toolbox, "store.py", "# deliberate rewrite\n", **{escape: True})
    assert result.ok
    assert target.read_text() == "# deliberate rewrite\n"


# ----------------------------------------------------------------------
# the exact regression
# ----------------------------------------------------------------------


def test_the_observed_store_py_rewrite_would_be_refused(toolbox: DevToolbox) -> None:
    """22274 chars -> 13319 chars is 60%, comfortably inside the refusal band."""
    target = toolbox.workspace / "memory_system" / "store.py"
    target.parent.mkdir(parents=True)
    target.write_text("s" * 22_274)
    result = _write(toolbox, "memory_system/store.py", "s" * 13_319)
    assert not result.ok
    assert len(target.read_text()) == 22_274


# ----------------------------------------------------------------------
# retry accounting: the guard must not end the episode it protects
# ----------------------------------------------------------------------


def test_the_refusal_is_marked_as_a_redirect(toolbox: DevToolbox) -> None:
    (toolbox.workspace / "store.py").write_text(_big())
    assert _write(toolbox, "store.py", "# tiny\n").redirect is True


def test_an_ordinary_failure_is_not_a_redirect(toolbox: DevToolbox) -> None:
    result = asyncio.run(toolbox._t_write_file({"content": "no path given"}))
    assert not result.ok
    assert result.redirect is False


def test_a_redirect_does_not_consume_a_developer_retry() -> None:
    """Five insistent write_file calls must not exhaust MAX_DEV_RETRIES.

    Without this the guard would end the very episode it exists to protect.
    """
    from nodes.developer import dev_observe

    observation = {"tool": "write_file", "ok": False, "redirect": True,
                   "text": "refusing to overwrite store.py", "data": {}}
    update = asyncio.run(dev_observe({"observation": observation, "retry_count": 2}))
    assert "retry_count" not in update


def test_a_real_failure_still_consumes_a_retry() -> None:
    """The cap must keep bounding "this design cannot be built"."""
    from nodes.developer import dev_observe

    observation = {"tool": "run_tests", "ok": False, "redirect": False,
                   "text": "3 failed", "data": {}}
    update = asyncio.run(dev_observe({"observation": observation, "retry_count": 2}))
    assert update["retry_count"] == 3


def test_a_duplicate_skip_keeps_its_own_retry_semantics() -> None:
    """dev_act's repeat-steering path sets no `redirect` and must be unaffected."""
    from nodes.developer import dev_observe

    observation = {"tool": "compile_check", "ok": False,
                   "text": "skipped: already ran", "data": {"skipped_duplicate": True}}
    update = asyncio.run(dev_observe({"observation": observation, "retry_count": 1}))
    assert update["retry_count"] == 2
