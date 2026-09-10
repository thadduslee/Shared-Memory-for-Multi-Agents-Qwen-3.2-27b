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
    append_to_notebook,
    approx_tokens,
    cap_tokens,
    dev_failure_summary,
    fallback_critique_summary,
    load_notebook,
    render_notebook_prompt,
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


def test_the_model_summary_is_what_lands_in_the_digest() -> None:
    (entry,) = _recap_additions(
        _digest_state(), {"previous_critique_summary": "Blamed top_k truncation."},
        iteration=4, critique="the full critique",
    )
    assert entry == {"iteration": 3, "kind": "critique", "summary": "Blamed top_k truncation."}


def test_a_model_summary_over_the_cap_is_cut_to_it() -> None:
    (entry,) = _recap_additions(
        _digest_state(), {"previous_critique_summary": "over budget. " * 200},
        iteration=4, critique="the full critique",
    )
    assert approx_tokens(entry["summary"]) <= CAP


def test_a_missing_summary_falls_back_to_the_measured_attribution() -> None:
    """The attribution is computed in Python, so it survives an unreachable Critic."""
    (entry,) = _recap_additions(_digest_state(), {}, iteration=4, critique="the full critique")
    assert "U" in entry["summary"]
    assert "store.retrieve" in entry["summary"]
    assert approx_tokens(entry["summary"]) <= CAP


def test_an_iteration_already_in_the_digest_is_not_recapped_again() -> None:
    """Two failed builds leave the SAME stale critique in state for three turns.

    Recapping it each time would put one finding in the history three times and
    make it read as three iterations independently reaching the same conclusion.
    """
    state = _digest_state(
        critique_iteration=3,
        critique_digest=[{"iteration": 3, "kind": "critique", "summary": "already recorded"}],
    )
    assert _recap_additions(state, {"previous_critique_summary": "again"},
                            iteration=5, critique="the same stale critique") == []


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


DIGEST = [
    {"iteration": 2, "kind": "critique", "summary": "Blamed the RBAC scope check."},
    {"iteration": 1, "kind": "critique", "summary": "Blamed top_k truncation."},
    {"iteration": 3, "kind": "dev_failure", "summary": "Could not build; tests_ok unmet."},
]


def test_the_notebook_lists_every_earlier_iteration_in_order(tmp_path: Path) -> None:
    path = tmp_path / "critique_summary.md"
    append_to_notebook(path, DIGEST, CAP)
    written = path.read_text(encoding="utf-8")
    assert written.startswith("# CRITIQUE SUMMARY")
    assert written.index("iteration 1") < written.index("iteration 2") < written.index("iteration 3")
    assert "Blamed top_k truncation." in written
    assert "build failure" in written, "an unbuilt iteration must be labelled as one"


def test_the_notebook_is_appended_to_and_never_rewritten(tmp_path: Path) -> None:
    """It is an audit trail of what each turn was actually shown, so a later
    turn must not be able to correct or drop an earlier turn's row."""
    path = tmp_path / "critique_summary.md"
    append_to_notebook(path, [DIGEST[1]], CAP)
    first = path.read_text(encoding="utf-8")
    append_to_notebook(path, [DIGEST[0]], CAP)
    second = path.read_text(encoding="utf-8")

    assert second.startswith(first), "an append must not disturb what is already there"
    assert second.count("# CRITIQUE SUMMARY") == 1, "the header is written exactly once"
    assert "Blamed top_k truncation." in second and "Blamed the RBAC scope check." in second


def test_nothing_to_append_leaves_no_notebook_at_all(tmp_path: Path) -> None:
    """Iteration 1 has nothing before it; an empty file under a header would
    read as 'this was checked and there was nothing'."""
    path = tmp_path / "critique_summary.md"
    assert append_to_notebook(path, [], CAP) == ""
    assert not path.exists()


def test_a_missing_notebook_is_rebuilt_from_the_digest_in_state(tmp_path: Path) -> None:
    """A resumed run, or a repointed RUNS_DIR, must not cost the whole history."""
    path = tmp_path / "gone.md"
    rebuilt = load_notebook(path, DIGEST, CAP)
    assert "Blamed top_k truncation." in rebuilt
    assert rebuilt.index("iteration 1") < rebuilt.index("iteration 3")

    append_to_notebook(path, [DIGEST[1]], CAP)
    assert load_notebook(path, DIGEST, CAP) == path.read_text(encoding="utf-8"), (
        "the file on disk wins over the digest whenever it exists"
    )


def test_the_prompt_section_is_omitted_entirely_when_there_is_no_history() -> None:
    """An empty section under a heading reads as 'checked, nothing found'."""
    assert render_notebook_prompt("") == ""
    assert render_notebook_prompt("   \n ") == ""
    assert "iteration 1" in render_notebook_prompt(load_notebook(Path("/nonexistent"), DIGEST, CAP))


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


async def test_the_architect_is_asked_for_the_summary_only_when_there_is_one(
    monkeypatch, tmp_path: Path
) -> None:
    seen: list[str] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        seen.append(task or "")
        return _reply({"schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"]})

    monkeypatch.setattr(transport, "agent_call", scripted)

    await architect_node({"iteration_count": 0, "memory_codebase": str(tmp_path / "ws")})
    assert "`previous_critique_summary`" not in seen[0], "nothing to summarise on iteration 1"

    await architect_node({
        "iteration_count": 1, "memory_codebase": str(tmp_path / "ws"),
        "critique": "the previous critique", "critique_iteration": 1,
        "proposed_design": "the previous design",
    })
    assert "`previous_critique_summary`" in seen[1]


async def test_the_architect_appends_to_the_notebook_and_leaves_design_md_alone(
    monkeypatch, tmp_path: Path
) -> None:
    """The summary is the notebook's, not the design document's. `design.md` is
    the model's reply verbatim; `design.json` records what was appended."""
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        return _reply({
            "schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"],
            "previous_critique_summary": "Blamed the tombstone gate ordering.",
        })

    monkeypatch.setattr(transport, "agent_call", scripted)

    result = await architect_node({
        "iteration_count": 1, "memory_codebase": str(tmp_path / "ws"),
        "critique": "the previous critique", "critique_iteration": 1,
        "proposed_design": "the previous design",
    })

    notebook = (config.RUNS_DIR / "critique_summary.md").read_text(encoding="utf-8")
    assert "**iteration 1** (critique): Blamed the tombstone gate ordering." in notebook

    design_md = (config.RUNS_DIR / "iter_2" / "design.md").read_text(encoding="utf-8")
    assert "CHANGES IN THIS ITERATION" not in design_md
    assert design_md == result["proposed_design"], "design.md is the reply, unadorned"

    written = json.loads((config.RUNS_DIR / "iter_2" / "design.json").read_text(encoding="utf-8"))
    assert "change_summary" not in written, "the Architect no longer diffs its own designs"
    assert written["critique_recap_added"] == [
        {"iteration": 1, "kind": "critique", "summary": "Blamed the tombstone gate ordering."}
    ]
    assert result["critique_digest"] == written["critique_recap_added"]


async def test_the_notebook_the_architect_reads_predates_the_critique_it_answers(
    monkeypatch, tmp_path: Path
) -> None:
    """THE ORDER IS THE POINT. Iteration i's critique is in the prompt in full;
    the notebook in the same prompt must still be iterations 1..i-1, or the same
    critique arrives twice with nothing saying which one the design answers."""
    seen: list[str] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        seen.append(task or "")
        return _reply({
            "schema_ddl": "CREATE TABLE t (x);", "work_order": ["do a thing"],
            "previous_critique_summary": "Iteration 2 blamed the tombstone gate.",
        })

    monkeypatch.setattr(transport, "agent_call", scripted)

    append_to_notebook(
        config.critique_summary_path(),
        [{"iteration": 1, "kind": "critique", "summary": "Iteration 1 blamed top_k."}],
        CAP,
    )
    await architect_node({
        "iteration_count": 2, "memory_codebase": str(tmp_path / "ws"),
        "critique": "iteration 2's critique, in full", "critique_iteration": 2,
        "critique_digest": [{"iteration": 1, "kind": "critique",
                             "summary": "Iteration 1 blamed top_k."}],
        "proposed_design": "the previous design",
    })

    assert "Iteration 1 blamed top_k." in seen[0], "the notebook is read into the prompt"
    assert "iteration 2's critique, in full" in seen[0]
    assert "Iteration 2 blamed the tombstone gate." not in seen[0], (
        "the notebook must not yet contain a summary of the critique being answered"
    )

    notebook = config.critique_summary_path().read_text(encoding="utf-8")
    assert "Iteration 1 blamed top_k." in notebook
    assert "Iteration 2 blamed the tombstone gate." in notebook, "appended after the design"


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
    state["critique_digest"] = DIGEST
    result = await critic_node(state)

    assert result["critique_iteration"] == 4

    written = (config.RUNS_DIR / "iter_4" / "critique.md").read_text(encoding="utf-8")
    assert written == result["critique"]
    for summary in ("Blamed top_k truncation.", "Blamed the RBAC scope check.",
                    "Could not build; tests_ok unmet."):
        assert summary not in written, "the recap is the notebook's, not this file's"
