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
# iter-12 -- cap-raise utility focus (no LLM)
# ----------------------------------------------------------------------
# 1. A SPARSE logistics gold record must be admitted to `allowed` at the
#    DEFAULT cap even when it ranks BELOW a swarm of chatty clinical rows that
#    reuse the query's vocabulary more heavily.  The old default of 16 cut the
#    gold (rank ~25); the iter-12 default of 40 keeps the ranked hard-stop
#    semantics on `allowed` while letting sparse gold through.
# 2. Raising the cap must not widen the A side: non-routine bodies that a
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


def test_logistics_gold_admitted_below_chatty_rows_at_default_top_k() -> None:
    """A sparse logistics gold ranked below ~24 chatty clinical rows is still
    admitted at the default cap, so the content-missing mechanism cannot fire."""
    agent = GateMemAgent(":memory:")
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
    agent = GateMemAgent(":memory:")
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
