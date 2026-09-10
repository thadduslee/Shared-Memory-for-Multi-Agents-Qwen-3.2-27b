"""Utility-focused unit tests -- deterministic, no LLM.

Two properties the ranked gate-0 overlap scan is meant to buy:

1. VARIANT RECALL.  The stored term index is the distinctive terms UNION their
   stem variants, and query hashes are drawn from the SAME widened set, so a
   record that only ever said 'result' is retrieved by a query for 'results'.
   Without the union this test fails -- the two digests never meet.

2. RELEVANCE OUTRANKS RECENCY.  `retrieve()` ranks live candidates by
   (overlap desc, seq desc) and enforces `top_k` only on the `allowed` list, so
   a higher-overlap older piece is picked over a lower-overlap but newer one.
   Sorting by `seq` alone (recency) would pick the wrong record.
"""

from __future__ import annotations

import pytest

from memory_system.agent import GateMemAgent
from memory_system.store import MemoryStore

PATIENT = "patient_uzo"
CLINICIAN = "clinician_okafor"


@pytest.fixture()
def store() -> MemoryStore:
    """A store where the clinician is the patient's assigned clinician.

    Both test records classify as `restricted` (they mention a result), so the
    clinician needs the `assigned_clinician` relationship to clear gate 3.
    """
    st = MemoryStore(":memory:")
    st.upsert_principal(PATIENT, "patient")
    st.upsert_principal(CLINICIAN, "clinician")
    st.upsert_relationship("assigned_clinician", CLINICIAN, PATIENT)
    return st


def _clinician_turn(turn_id: str, text: str) -> dict:
    return {
        "turn_id": turn_id,
        "turn_kind": "note_update",
        "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
        "text": text,
    }


def test_variant_recall_singular_stored_plural_queried(store: MemoryStore) -> None:
    """Stored 'result' is recalled by a query for 'results' (no LLM)."""
    store.ingest_turn(
        "ep_util",
        _clinician_turn("t001", "The result came back today and is normal."),
        PATIENT,
    )

    decision = store.retrieve(
        requester_id=CLINICIAN, requester_role="clinician",
        patient_id=PATIENT, query="please tell me my results",
    )

    assert decision.allowed, "variant-matching record was not retrieved"
    bodies = " ".join(e.body for e in decision.allowed)
    assert "result came back" in bodies


def test_top_k_pick_prefers_higher_overlap_over_newer(store: MemoryStore) -> None:
    """Among live allowed evidence, relevance beats recency under the cap.

    The older record shares more of the query's terms (5 overlap) than the
    newer one (3 overlap).  With top_k=1 a recency-first scan would return the
    newer, lower-overlap record; the overlap-first scan must return the older,
    higher-overlap one.
    """
    old_id = store.ingest_turn(
        "ep_util",
        _clinician_turn("t010", "Patient reports test results and imaging results both normal today."),
        PATIENT,
    )
    new_id = store.ingest_turn(
        "ep_util",
        _clinician_turn("t020", "Patient reports test results are normal."),
        PATIENT,
    )
    assert new_id != old_id

    decision = store.retrieve(
        requester_id=CLINICIAN, requester_role="clinician",
        patient_id=PATIENT, query="test results imaging", top_k=1,
    )

    assert len(decision.allowed) == 1, "top_k cap must be honored on the allowed list"
    picked = decision.allowed[0].record_id
    assert picked == old_id, f"expected higher-overlap older record, got {picked}"
    assert picked != new_id, "newer lower-overlap record outranked a more relevant one"


# ----------------------------------------------------------------------
# iter-13 -- widened-cap probe (no LLM)
# ----------------------------------------------------------------------
# These tests construct GateMemAgent(":memory:", top_k=40) EXPLICITLY: 40 is a
# widened-cap PROBE, not the default.  The measured correct default is 16 as
# enforced by the iter-13 revert; raising the default to 40 evaluated records
# ranked 17-40 through the deny gates, flipped clean answers to
# answer_redacted (U 0.6667->0.6111), and fixed none of the seven
# cap-insensitive census checkpoints.  Do not raise the default.
# 1. A SPARSE logistics gold record must be admitted to `allowed` at the
#    widened 40 cap even when it ranks BELOW a swarm of chatty clinical rows
#    that reuse the query's vocabulary more heavily.
# 2. Widening the cap must not widen the A side: non-routine bodies that a
#    logistics-only family_member is not cleared for must still be denied, and
#    no restricted/confidential marker may reach the answer.


def _logistics_episode(episode_id: str) -> dict:
    """Clinician is the patient's assigned clinician (sees routine+restricted+
    confidential), so every row below is gate-3-clear for them."""
    return {
        "episode_id": episode_id,
        "entities": {
            "principals": [
                {"principal_id": PATIENT, "role": "patient", "display_name": "Uzo"},
                {"principal_id": CLINICIAN, "role": "clinician", "display_name": "Dr. Okafor"},
            ],
            "relationships": [
                {"type": "assigned_clinician", "clinician_id": CLINICIAN, "patient_id": PATIENT},
            ],
        },
    }


def _chatty_clinical_note(i: int) -> str:
    """A busy handoff row that reuses the query's words (time/EEG/Friday/April)
    but is plainly NOT the appointment.  Real-content overlap 4 -- above the
    sparse gold's 3 -- so it ranks in front of the gold under the overlap scan."""
    return (
        f"Handoff {i}: the EEG staffing team met on Friday and again in April to "
        f"review the roster, and they spent time on the calendar but made no "
        f"changes to the clinic schedule this week."
    )


def test_logistics_gold_admitted_below_chatty_rows_at_widened_top_k() -> None:
    """A sparse logistics gold ranked below ~24 chatty clinical rows is still
    admitted at the widened cap probe, so the content-missing mechanism cannot
    fire.  40 is an explicit probe here, NOT the default (16)."""
    agent = GateMemAgent(":memory:", top_k=40)
    agent.reset(_logistics_episode("ep_util"))

    # ~24 chatty clinical rows reusing the query vocabulary (all 'EEG'/'Friday').
    for i in range(1, 25):
        agent.ingest(_clinician_turn(f"t{i:03d}", _chatty_clinical_note(i)))
    # ONE sparse gold carrying the actual appointment, ranking last (real-content
    # overlap 3 < the chatty rows' 4).
    agent.ingest(_clinician_turn(
        "t025", "EEG scheduled Friday April 4 at 2:00 PM at lobby pickup"))

    out = agent.query({
        "checkpoint_id": "c_util", "episode_id": "ep_util",
        "as_of_turn_id": "t025",
        "asker": {"principal_id": CLINICIAN, "role": "clinician"},
        "query_text": "What time is the EEG on Friday April 4?",
    })

    assert out["action"] in {"answer", "answer_redacted"}, out["action"]
    assert "April 4" in out["answer"], f"gold date missing from answer: {out['answer']!r}"
    assert "2:00" in out["answer"], f"gold time missing from answer: {out['answer']!r}"


def test_cap_raise_does_not_leak_confidential_or_restricted_bodies() -> None:
    """A-side guard: with the raised default cap, non-routine bodies must stay
    out of `allowed` and their marker text out of the answer."""
    family = "family_uzo"
    agent = GateMemAgent(":memory:", top_k=40)
    agent.reset({
        "episode_id": "ep_fam",
        "entities": {
            "principals": [
                {"principal_id": PATIENT, "role": "patient", "display_name": "Uzo"},
                {"principal_id": CLINICIAN, "role": "clinician", "display_name": "Dr. Okafor"},
                {"principal_id": family, "role": "family_member", "display_name": "Auntie"},
            ],
            "relationships": [
                {"type": "assigned_clinician", "clinician_id": CLINICIAN, "patient_id": PATIENT},
                {"type": "family_contact", "family_id": family, "patient_id": PATIENT,
                 "scope": "logistics only"},
            ],
        },
    })
    # One routine logistics row the family may see.
    agent.ingest(_clinician_turn(
        "t001", "EEG scheduled Friday at 2:00 PM at the lobby pickup."))
    # Two non-routine rows responsive to the same query; the family must not
    # get them under ANY cap.
    agent.ingest(_clinician_turn(
        "t002", "EEG result Friday reviewed; the specialist will discuss next steps."))
    agent.ingest(_clinician_turn(
        "t003", "Confidential: EEG Friday discussion with the psychiatrist."))

    query = "EEG Friday"
    out = agent.query({
        "checkpoint_id": "c_fam", "episode_id": "ep_fam",
        "as_of_turn_id": "t003",
        "asker": {"principal_id": family, "role": "family_member"},
        "query_text": query,
    })

    decision = agent.store.retrieve(
        requester_id=family, requester_role="family_member",
        patient_id=PATIENT, query=query, top_k=agent.top_k,
    )
    assert decision.allowed, "the routine logistics row must be retrieved"
    # A side: only routine sensitivity may be admitted, however high the cap.
    assert all(e.sensitivity == "routine" for e in decision.allowed), (
        f"non-routine body admitted under cap {agent.top_k}: "
        f"{[e.sensitivity for e in decision.allowed]}"
    )
    # B side: no restricted/confidential marker text may reach the answer.
    markers = ("result", "specialist", "confidential", "psychiatr", "genetic", "hiv")
    low = out["answer"].lower()
    assert not any(m in low for m in markers), (
        f"restricted/confidential marker leaked into answer: {out['answer']!r}"
    )


# ----------------------------------------------------------------------
# iter-13 -- default-cap regression + census status attribution
# ----------------------------------------------------------------------


def test_default_top_k_cap_is_sixteen() -> None:
    """The measured correct default cap is 16: constructing GateMemAgent with
    no top_k must admit at most 16 records.  A future silent raise of the
    default (iter 12's 40) fails this build exactly as test_top_k_is_a_hard_cap
    does for the explicit-8 path."""
    agent = GateMemAgent(":memory:")
    agent.reset(_logistics_episode("ep_cap"))

    # 20 routine logistics rows, all responsive to the same query and all
    # cleared for the assigned clinician.
    for i in range(1, 21):
        agent.ingest(_clinician_turn(
            f"t{i:03d}", "Routine: patient pickup is at the lobby at 2:00 PM on Friday."))

    decision = agent.store.retrieve(
        requester_id=CLINICIAN, requester_role="clinician",
        patient_id=PATIENT,
        query="What time is the patient pickup on Friday?",
        top_k=agent.top_k,
    )
    assert decision.allowed, "routine rows must be retrieved"
    assert len(decision.allowed) <= 16, (
        f"default cap leaked: expected <= 16 allowed with default top_k, got "
        f"{len(decision.allowed)} (default top_k={agent.top_k})"
    )


def test_census_status_is_diagnostic_and_from_allowed_set() -> None:
    """Each candidate_census entry carries the gate outcome actually produced,
    drawn only from the fixed status set and non-empty per candidate.  The
    status field is diagnostics-only and capped with the existing census cap."""
    family = "family_uzo"
    agent = GateMemAgent(":memory:")
    agent.reset({
        "episode_id": "ep_census",
        "entities": {
            "principals": [
                {"principal_id": PATIENT, "role": "patient", "display_name": "Uzo"},
                {"principal_id": CLINICIAN, "role": "clinician", "display_name": "Dr. Okafor"},
                {"principal_id": family, "role": "family_member", "display_name": "Auntie"},
            ],
            "relationships": [
                {"type": "assigned_clinician", "clinician_id": CLINICIAN, "patient_id": PATIENT},
                {"type": "family_contact", "family_id": family, "patient_id": PATIENT,
                 "scope": "logistics only"},
            ],
        },
    })
    # One routine logistics row the family may see -> allowed.
    agent.ingest(_clinician_turn(
        "t001", "EEG scheduled Friday at 2:00 PM at the lobby pickup."))
    # Restricted body responsive to the query -> scope-denied for the family.
    agent.ingest(_clinician_turn(
        "t002", "EEG result Friday reviewed; the specialist will discuss next steps."))
    # Confidential body responsive to the query -> scope-denied for the family.
    agent.ingest(_clinician_turn(
        "t003", "Confidential: EEG Friday discussion with the psychiatrist."))
    # Unrelated row, scanned but not responsive -> gate-0-skipped.
    agent.ingest(_clinician_turn(
        "t004", "Patient had lunch and walked around the garden this morning."))

    decision = agent.store.retrieve(
        requester_id=family, requester_role="family_member",
        patient_id=PATIENT, query="EEG Friday pickup time", top_k=8,
    )
    allowed_statuses = {
        "allowed", "deny_rbac", "deny_scope", "deny_tombstone", "gate0_skip", "unscanned",
    }
    census = decision.candidate_census
    assert census, "candidate_census should not be empty"
    for entry in census:
        status = entry.get("status")
        assert status, f"census entry missing status: {entry}"
        assert status in allowed_statuses, (
            f"census status {status!r} not drawn from the allowed status set"
        )
    assert decision.allowed, "the routine logistics row must be admitted"
    # The unrelated scanned row must surface as a gate-0 skip, proving the
    # full gate-0 story is recorded in the census.
    assert any(e["status"] == "gate0_skip" for e in census), (
        "no gate0_skip entry in census for the unrelated scanned row"
    )


# ----------------------------------------------------------------------
# iter-16 -- rescue runs on the LLM path at the DEFAULT top_k (16)
# ----------------------------------------------------------------------


def test_rescue_reaches_sparse_gold_on_llm_path_at_default_top_k() -> None:
    """The logistics rescue must run on the LLM path too, at the DEFAULT top_k.

    Under the default cap of 16, a swarm of ~24 chatty clinical rows (overlap
    4) keeps the sparse gold logistics record (overlap 3) out of
    `decision.allowed`, so neither the model's summary nor the appended gated
    bodies can carry the gold date/time.  `_rescue_append` -- the same helper
    the no-LLM branch uses -- must recover that cleared body so the sparse gold
    (April 4 / 2:00 PM) is present verbatim in the final judged answer.  This
    proves the iter-16 rescue wiring now reaches the LLM branch, not just the
    no-LLM one."""
    calls: list[tuple[str, list]] = []

    def recorded_llm(query_text: str, evidence: list) -> dict:
        calls.append((query_text, evidence))
        return {"action": "answer", "answer": "The EEG appointment is scheduled in the morning."}

    agent = GateMemAgent(":memory:", llm=recorded_llm)
    agent.reset(_logistics_episode("ep_rescue_llm"))

    # ~24 chatty clinical rows reusing the query vocabulary, crowding out the
    # sparse gold at the default top_k=16.
    for i in range(1, 25):
        agent.ingest(_clinician_turn(f"t{i:03d}", _chatty_clinical_note(i)))
    # ONE sparse gold carrying the actual appointment (rank 25 under the cap).
    agent.ingest(_clinician_turn(
        "t025", "EEG scheduled Friday April 4 at 2:00 PM at lobby pickup"))

    out = agent.query({
        "checkpoint_id": "c_util", "episode_id": "ep_rescue_llm",
        "as_of_turn_id": "t025",
        "asker": {"principal_id": CLINICIAN, "role": "clinician"},
        "query_text": "What time is the EEG on Friday April 4?",
    })

    assert calls, "the recorded LLM must have been invoked on the LLM path"
    # The sparse gold ranked below top_k=16, so it must NOT have been among the
    # bodies handed to the model -- the rescue is what reaches it.
    assert out["action"] in {"answer", "answer_redacted"}, out["action"]
    assert "April 4" in out["answer"], f"gold date missing from answer: {out['answer']!r}"
    assert "2:00" in out["answer"], f"gold time missing from answer: {out['answer']!r}"
    # The gold arrived via the rescue 'Details:' append on the LLM path (the
    # model's own summary held only "in the morning").
    assert "lobby pickup" in out["answer"], (
        f"rescue body not appended on LLM path: {out['answer']!r}"
    )


# ----------------------------------------------------------------------
# iter-17 -- rescue reaches a cleared gold sharing ZERO indexed terms with
# the query (pure paraphrase divergence), and never a tombstoned row
# ----------------------------------------------------------------------


def test_rescue_reaches_zero_shared_term_cleared_gold(store: MemoryStore) -> None:
    """The logistics rescue must recover a cleared routine gold whose body
    shares NO indexed content term with the query.

    The gold body ('transport pickup Tuesday at 9:15 lobby') is a pure
    paraphrase of a 'when is the ride' question -- no shared token, so gate 0
    would never rank it, and a term_hash join on the candidate set (the bound
    this iteration removes/audits) would discard it before the logistics-token
    filter ever ran.  `rescue_missing_logistics` scans the live patient rows
    regardless of query overlap and returns the gold carrying the missing
    time/date; a sibling logistics row that HAS been tombstoned must not come
    back, and the gold itself must not be tombstoned."""
    gold_id = store.ingest_turn(
        "ep_rescue_z",
        _clinician_turn("t001", "Transport pickup is scheduled Tuesday at 9:15 in the main lobby."),
        PATIENT,
    )
    decoy_id = store.ingest_turn(
        "ep_rescue_z",
        _clinician_turn("t002", "Transport drop-off is set for Wednesday at 5:00 PM at the east entrance."),
        PATIENT,
    )
    store.ingest_turn(
        "ep_rescue_z",
        _clinician_turn("t003", "A wheelchair transport will be arranged for Thursday."),
        PATIENT,
    )
    # Tombstone a sibling logistics row: rescue must skip it yet still reach
    # the cleared, non-tombstoned gold.
    store.tombstone(decoy_id, requested_by=PATIENT, reason="not needed")
    assert store.is_deleted(decoy_id)

    rescued = store.rescue_missing_logistics(
        patient_id=PATIENT, requester_id=CLINICIAN,
        requester_role="clinician", as_of_seq=3,
        answer="the ride is confirmed on the schedule",
    )
    assert rescued, "rescue returned nothing despite a cleared gold logistics row"
    gold = next((e for e in rescued if e.record_id == gold_id), None)
    assert gold is not None, (
        f"gold body not rescued: {[e.record_id for e in rescued]}"
    )
    assert "9:15" in gold.body and "Tuesday" in gold.body, (
        f"rescued gold missing the time/date: {gold.body!r}"
    )
    assert not store.is_deleted(gold_id), "the rescued gold must not be tombstoned"
    assert all(e.record_id != decoy_id for e in rescued), (
        "a tombstoned logistics row leaked back through rescue"
    )
