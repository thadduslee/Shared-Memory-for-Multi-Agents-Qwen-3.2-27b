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
