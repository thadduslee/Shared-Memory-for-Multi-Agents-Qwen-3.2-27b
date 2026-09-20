"""The Architect's notebook: one iteration's memory of the ones before it.

WHAT WAS MISSING. `state["critique"]` holds exactly ONE critique -- the most
recent -- and `critique.md` is rewritten from scratch every iteration. So
iteration 4's Architect saw iteration 3's diagnosis and had no record at all of
what iterations 1 and 2 had already found and already tried, and nothing on disk
put the run's own history in one place. In runs_test2, iterations 1, 2 and 4 all
blamed the same `sanitize_and_decide` branch ordering.

The Architect now keeps that history itself, in `runs/critique_summary.md`. Its
turn runs in a fixed order: read iteration i's `critique.md`, read the notebook
of iterations 1..i-1, write the design, and only THEN append iteration i's
summary to the notebook.

WHAT THESE TESTS PIN
    * the cap is per entry, so the file grows linearly rather than compounding;
    * an iteration is summarised exactly once, even when two failed builds leave
      the same stale critique in state across three Architect turns;
    * an iteration that never reached the Critic still gets a row, because a gap
      in a numbered history reads as a lost record;
    * the notebook is APPENDED to after the design and read before it, so the
      history a turn designs from never already contains the critique it is
      answering;
    * the raw critique is what gets summarised -- never the notebook, which
      already contains the earlier summaries;
    * `critique.md` is the critique and nothing else, because that file is what
      the next Architect opens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import config
import nodes._transport as transport
from harness.dsh_client import DSHResult
from nodes._recap import (
    append_history,
    approx_tokens,
    cap_notes,
    cap_tokens,
    dev_failure_summary,
    fallback_critique_summary,
    load_notes,
    notes_from_notebook,
    render_notebook_prompt,
    write_notebook,
)
from nodes.architect import _recap_additions, architect_node
from nodes.critic import critic_node

CAP = 100


# ======================================================================
# the cap
# ======================================================================


def test_a_short_summary_is_left_exactly_as_written() -> None:
    text = "The Critic blamed top_k truncation in retrieve() and asked for an index."
    assert cap_tokens(text, CAP) == text


def test_a_whole_critique_is_cut_down_to_the_budget() -> None:
    """The input here is the size a real critique.md actually is (~11k chars)."""
    critique = ("The dominant failing term is U and the mechanism census "
                "isolates three distinct failure paths. ") * 120
    capped = cap_tokens(critique, CAP)
    assert approx_tokens(capped) <= CAP
    assert capped.endswith("...")
    assert len(capped) < len(critique) / 20


def test_the_truncation_marker_is_paid_for_out_of_the_budget() -> None:
    """An ellipsis is a token too; appending it after the cap would exceed it."""
    for length in range(60, 400, 37):
        capped = cap_tokens("word " * length, CAP)
        assert approx_tokens(capped) <= CAP, f"{length} words overflowed the cap"


def test_a_summary_is_flattened_to_one_paragraph() -> None:
    """It is rendered as one markdown list item, so a list inside it breaks the list."""
    capped = cap_tokens("Findings:\n- one thing\n- another thing\n", CAP)
    assert "\n" not in capped
    assert capped == "Findings: - one thing - another thing"


def test_the_estimate_never_undercounts_prose_or_identifiers() -> None:
    """Both rules of thumb are applied and the larger wins; see `approx_tokens`."""
    assert approx_tokens("") == 0
    assert approx_tokens("   \n  ") == 0
    # Long identifiers: few words, many characters -- the character rule binds.
    assert approx_tokens("med_episode_rewrite_en_011_first_seizure_ckpt_05") >= 11
    # Short words: many words, few characters -- the word rule binds.
    assert approx_tokens("a b c d e f g h i j") >= 13


# ======================================================================
# which iterations get a row, and how many
# ======================================================================


def _digest_state(**overrides):
    state = {
        "iteration_count": 3,
        "critique_iteration": 3,
        "critique_digest": [],
        "attribution": {"dominant_term": "U", "dominant_gain": 0.63,
                        "component": "store.retrieve", "observed_mechanisms": ["starved_by_rbac"],
                        "proposals": [{"change": "index it"}]},
        "dev_failure_report": {},
    }
    state.update(overrides)
    return state


def test_the_digest_row_is_derived_and_not_taken_from_the_model() -> None:
    """The digest is the record the run is AUDITED against.

    The Architect's own account of the round goes into the notebook it curates.
    This row is computed from the measured attribution instead, so a model that
    writes itself a flattering notebook is not also writing the history
    `check_learning.py` judges it by.
    """
    (entry,) = _recap_additions(
        _digest_state(), {"notebook_notes": ["whatever the model chose to keep"]},
        iteration=4, critique="the full critique",
    )
    assert entry["iteration"] == 3
    assert entry["kind"] == "critique"
    assert "whatever the model chose" not in entry["summary"]
    assert "U" in entry["summary"] and "store.retrieve" in entry["summary"]


def test_a_digest_row_is_still_bounded() -> None:
    """Not prompt text any more, so the budget is generous -- but not absent."""
    (entry,) = _recap_additions(_digest_state(), {}, iteration=4, critique="the full critique")
    assert approx_tokens(entry["summary"]) <= 400


def test_an_iteration_already_in_the_digest_is_not_recapped_again() -> None:
    """Two failed builds leave the SAME stale critique in state for three turns.

    Recapping it each time would put one finding in the history three times and
    make it read as three iterations independently reaching the same conclusion.
    """
    state = _digest_state(
        critique_iteration=3,
        critique_digest=[{"iteration": 3, "kind": "critique", "summary": "already recorded"}],
    )
    assert _recap_additions(state, {}, iteration=5,
                            critique="the same stale critique") == []


def test_an_unbuildable_iteration_still_gets_a_row_of_its_own() -> None:
    """A failed build never reaches the Critic, so it produces no critique at all."""
    state = _digest_state(
        critique_iteration=2,
        critique_digest=[{"iteration": 2, "kind": "critique", "summary": "iteration 2"}],
        dev_failure_report={
            "iteration": 3, "reason": "retries exhausted", "missing_gates": ["tests_ok"],
            "retries_used": 5, "retry_cap": 5, "pass_rate": 0.83, "signature": "sig-rbac",
        },
    )
    (entry,) = _recap_additions(state, {}, iteration=4, critique="iteration 2's critique")
    assert entry["iteration"] == 3
    assert entry["kind"] == "dev_failure"
    assert "tests_ok" in entry["summary"]
    assert approx_tokens(entry["summary"]) <= CAP


def test_iteration_1_has_nothing_to_recap() -> None:
    assert _recap_additions(_digest_state(critique_iteration=0), {},
                            iteration=1, critique="") == []


def test_rows_are_emitted_oldest_first() -> None:
    state = _digest_state(
        critique_iteration=2, critique_digest=[],
        dev_failure_report={"iteration": 3, "reason": "retries exhausted",
                            "missing_gates": [], "retries_used": 5, "retry_cap": 5,
                            "pass_rate": 0.0, "signature": "s"},
    )
    additions = _recap_additions(state, {}, iteration=4, critique="iteration 2's critique")
    assert [e["iteration"] for e in additions] == [2, 3]


# ======================================================================
# what the rendering says
# ======================================================================


NOTES = [
    "top_k=16 truncated the candidate scan; raising it to 40 moved U +0.05.",
    "Softening the tombstone gate raised U but cost A 0.09 -> 0.41; reverted.",
    "The action label is scored, so a correct answer labelled answer_redacted still fails.",
]


def test_the_notebook_holds_the_notes_it_was_given(tmp_path: Path) -> None:
    path = tmp_path / "critique_summary.md"
    write_notebook(path, NOTES)
    written = path.read_text(encoding="utf-8")
    assert written.startswith("# CRITIQUE SUMMARY")
    for note in NOTES:
        assert note in written
    assert notes_from_notebook(written) == NOTES, "the file must round-trip"


def test_the_notebook_is_replaced_so_a_stale_note_can_be_retired(tmp_path: Path) -> None:
    """The whole point of curation: a finding later rounds disproved must be
    removable, which an append-only log could never do."""
    path = tmp_path / "critique_summary.md"
    write_notebook(path, NOTES)
    write_notebook(path, [NOTES[2], "Raising top_k past 40 no longer helps; U is capped elsewhere."])
    written = path.read_text(encoding="utf-8")

    assert "top_k=16 truncated" not in written, "a retired note must actually go"
    assert "no longer helps" in written
    assert written.count("# CRITIQUE SUMMARY") == 1, "one header, not one per version"


def test_an_empty_note_list_cannot_erase_the_notebook(tmp_path: Path) -> None:
    """Under rewrite-in-place the run's memory is one bad reply away from gone:
    a timeout, a missing json block, a forgotten key."""
    path = tmp_path / "critique_summary.md"
    write_notebook(path, NOTES)
    before = path.read_text(encoding="utf-8")

    assert write_notebook(path, []) == ""
    assert path.read_text(encoding="utf-8") == before, "the previous notes must survive"


def test_nothing_to_write_leaves_no_notebook_at_all(tmp_path: Path) -> None:
    """An empty file under a heading reads as 'this was checked and there was
    nothing', which is a different and stronger claim than silence."""
    path = tmp_path / "critique_summary.md"
    assert write_notebook(path, []) == ""
    assert not path.exists()


def test_notes_are_dropped_whole_never_cut_in_half(tmp_path: Path) -> None:
    """The old scheme truncated every entry at 100 estimated tokens and cut 63%
    of them mid-sentence -- systematically losing the trailing clause, which is
    where a summary says what was DECIDED."""
    long_note = " ".join(f"word{i}" for i in range(200))
    capped = cap_notes([NOTES[0], long_note, NOTES[1]], max_notes=2, max_words=15)

    assert len(capped) == 2, "the note count is a hard stop"
    assert capped[0] == NOTES[0], "a note inside the budget is untouched"
    assert capped[1].endswith("..."), "an over-long note is marked, not silently cut"
    assert len(capped[1].split()) <= 16


def test_a_missing_notebook_falls_back_to_the_notes_in_state(tmp_path: Path) -> None:
    """A resumed run, or a repointed RUNS_DIR, must not cost the whole history."""
    path = tmp_path / "gone.md"
    assert load_notes(path, NOTES) == NOTES

    write_notebook(path, [NOTES[2]])
    assert load_notes(path, NOTES) == [NOTES[2]], (
        "the file on disk wins over state whenever it exists"
    )


def test_every_version_is_kept_in_the_history_log(tmp_path: Path) -> None:
    """Rewriting loses the audit trail the append-only file had for free. The
    log is what still tells a curation from a loss."""
    path = tmp_path / "notebook_history.md"
    append_history(path, 4, NOTES[:2])
    append_history(path, 5, [NOTES[2]])
    written = path.read_text(encoding="utf-8")

    assert written.count("# NOTEBOOK HISTORY") == 1
    assert "after iteration 4" in written and "after iteration 5" in written
    assert "top_k=16 truncated" in written, "a note dropped from the notebook stays here"


def test_the_prompt_asks_for_the_whole_list_and_states_both_budgets() -> None:
    block = render_notebook_prompt(NOTES, max_notes=40, max_words=45)
    assert "notebook_notes" in block
    assert "FULL list" in block
    assert "40 notes" in block and "45 words" in block, (
        "budgets must be stated in units a model can count, not BPE tokens"
    )
    for note in NOTES:
        assert note in block, "the model must see what it is revising"


def test_the_first_iteration_is_told_the_notebook_is_empty() -> None:
    """Not omitted: a model given no notebook section invents no notes, but a
    model told the notebook is empty knows it is the one starting it."""
    block = render_notebook_prompt([], max_notes=40, max_words=45)
    assert "Empty" in block
    assert "notebook_notes" in block

def test_the_deterministic_summaries_carry_the_numbers_that_matter() -> None:
    summary = fallback_critique_summary(
        {"dominant_term": "F", "dominant_gain": 0.21, "component": "the tombstone gate",
         "observed_mechanisms": ["deleted_content_returned"], "proposals": [1, 2]}, 3)
    assert "F" in summary and "+0.2100" in summary and "the tombstone gate" in summary
    assert "2 proposal(s)" in summary

    failure = dev_failure_summary(
        {"reason": "retries exhausted", "missing_gates": ["tests_ok", "migration_ok"],
         "retries_used": 5, "retry_cap": 5, "pass_rate": 0.83, "signature": "sig-rbac"})
    assert "tests_ok, migration_ok" in failure and "sig-rbac" in failure


# ======================================================================
# the nodes
# ======================================================================


def _reply(payload: dict, prose: str = "# Design\n\nthe design prose\n") -> DSHResult:
    return DSHResult(
        ok=True, text=prose + "\n```json\n" + json.dumps(payload) + "\n```",
        profile="architect", finish_reason="completed", usage={"total_tokens": 7},
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MOCK_MODE", True)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    import websearch
    websearch.reset_search_client()
    yield
    websearch.reset_search_client()


async def test_the_architect_is_asked_to_curate_the_notebook_every_iteration(
    monkeypatch, tmp_path: Path
) -> None:
    """Asked on iteration 1 too, unlike the old per-critique summary: the first
    round still learns what the baseline measured, and an empty reply cannot
    erase a file that does not exist yet."""
    seen: list[str] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        seen.append(task or "")
        return _reply({"schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"]})

    monkeypatch.setattr(transport, "agent_call", scripted)

    await architect_node({"iteration_count": 0, "memory_codebase": str(tmp_path / "ws")})
    assert "`notebook_notes`" in seen[0]
    assert "Empty" in seen[0], "the first iteration is told it is starting the notebook"


async def test_the_architect_rewrites_the_notebook_and_leaves_design_md_alone(
    monkeypatch, tmp_path: Path
) -> None:
    """The notes are the notebook's, not the design document's. `design.md` is
    the model's reply verbatim."""
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return _reply({
            "schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"],
            "notebook_notes": ["The deletion gate discards allowed records.",
                               "Raising the scan cap moved U +0.05."],
        })

    monkeypatch.setattr(transport, "agent_call", scripted)

    result = await architect_node({
        "iteration_count": 1, "memory_codebase": str(tmp_path / "ws"),
        "critique": "the previous critique", "critique_iteration": 1,
        "proposed_design": "the previous design",
    })

    notebook = config.critique_summary_path().read_text(encoding="utf-8")
    assert "The deletion gate discards allowed records." in notebook
    assert "Raising the scan cap moved U +0.05." in notebook
    assert result["notebook_notes"] == [
        "The deletion gate discards allowed records.",
        "Raising the scan cap moved U +0.05.",
    ], "the curated list is published to state as the disk-loss fallback"

    history = (config.RUNS_DIR / "notebook_history.md").read_text(encoding="utf-8")
    assert "after iteration 2" in history

    design_md = (config.RUNS_DIR / "iter_2" / "design.md").read_text(encoding="utf-8")
    assert design_md == result["proposed_design"], "design.md is the reply, unadorned"


async def test_a_reply_with_no_notes_leaves_the_notebook_standing(
    monkeypatch, tmp_path: Path
) -> None:
    """Under rewrite-in-place one bad reply could otherwise erase the run's
    memory, so the failure mode must be losing a lesson, not losing the file."""
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return _reply({"schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"]})

    monkeypatch.setattr(transport, "agent_call", scripted)
    write_notebook(config.critique_summary_path(), ["An earlier lesson worth keeping."])

    await architect_node({
        "iteration_count": 1, "memory_codebase": str(tmp_path / "ws"),
        "critique": "the previous critique", "critique_iteration": 1,
    })

    notebook = config.critique_summary_path().read_text(encoding="utf-8")
    assert "An earlier lesson worth keeping." in notebook


async def test_the_notebook_the_architect_reads_predates_the_critique_it_answers(
    monkeypatch, tmp_path: Path
) -> None:
    """THE ORDER IS THE POINT. Iteration i's critique is in the prompt in full;
    the notebook in the same prompt must still be what stood BEFORE it, or the
    same round arrives twice with nothing saying which one the design answers."""
    seen: list[str] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        seen.append(task or "")
        return _reply({
            "schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"],
            "notebook_notes": ["Iteration 1 blamed the scan cap.",
                               "Iteration 2 blamed the deletion gate."],
        })

    monkeypatch.setattr(transport, "agent_call", scripted)
    write_notebook(config.critique_summary_path(), ["Iteration 1 blamed the scan cap."])

    await architect_node({
        "iteration_count": 2, "memory_codebase": str(tmp_path / "ws"),
        "critique": "iteration 2's critique, in full", "critique_iteration": 2,
        "proposed_design": "the previous design",
    })

    assert "Iteration 1 blamed the scan cap." in seen[0], "the notebook is read into the prompt"
    assert "iteration 2's critique, in full" in seen[0]
    assert "Iteration 2 blamed the deletion gate." not in seen[0], (
        "the notebook must not yet contain this round's own lesson"
    )

    notebook = config.critique_summary_path().read_text(encoding="utf-8")
    assert "Iteration 2 blamed the deletion gate." in notebook, "written after the design"


async def test_the_critic_writes_the_critique_and_nothing_else(
    monkeypatch, tmp_path: Path
) -> None:
    """`critique.md` is what the next Architect opens, and it must be the same
    bytes as `state["critique"]`. The history is the Architect's notebook, not
    a section of this file -- if the earlier summaries were in here, every
    iteration would end up re-summarising its own summaries."""
    from tests.test_critic_feedback import _state

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return DSHResult(ok=True, text='the critique\n```json\n{"component": "agent"}\n```',
                         profile=profile.name, finish_reason="completed",
                         usage={"total_tokens": 5})

    monkeypatch.setattr(transport, "agent_call", scripted)

    state = _state(tmp_path)
    state["iteration_count"] = 4
    state["critique_digest"] = [
        {"iteration": 1, "kind": "critique", "summary": "Blamed the scan cap."},
    ]
    result = await critic_node(state)

    assert result["critique_iteration"] == 4

    written = (config.RUNS_DIR / "iter_4" / "critique.md").read_text(encoding="utf-8")
    assert written == result["critique"]
    for summary in ("Blamed top_k truncation.", "Blamed the RBAC scope check.",
                    "Could not build; tests_ok unmet."):
        assert summary not in written, "the recap is the notebook's, not this file's"
