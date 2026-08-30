"""Active-forgetting unit tests -- the Developer's `run_tests` gate, part 2.

The F metric is lost in three specific ways, and there is a test for each:
content coming back, *existence* being confirmed, and a deleted record being
recoverable through a different query path.
"""

from __future__ import annotations

import pytest

from memory_system.store import MemoryStore

PATIENT = "patient_ana"
NURSE = "nurse_bell"
CLINICIAN = "clinician_okafor"
SECRET = "safe callback number 555-0142"


@pytest.fixture()
def store() -> MemoryStore:
    st = MemoryStore(":memory:")
    st.upsert_principal(PATIENT, "patient")
    st.upsert_principal(NURSE, "nurse")
    st.upsert_principal(CLINICIAN, "clinician")
    st.upsert_relationship("assigned_clinician", CLINICIAN, PATIENT)
    st.ingest_turn("ep1", {"turn_id": "t001", "turn_kind": "portal_message",
                           "speaker": {"principal_id": PATIENT, "role": "patient"},
                           "text": f"Please store this: {SECRET}."}, PATIENT)
    return st


def test_deletion_request_creates_tombstone(store: MemoryStore) -> None:
    store.ingest_turn("ep1", {"turn_id": "t002", "turn_kind": "portal_message",
                              "speaker": {"principal_id": PATIENT, "role": "patient"},
                              "text": f"Actually, delete {SECRET} from the shared assistant memory."},
                      PATIENT)
    assert store.is_deleted("ep1:t001"), "the deletion request did not match the stored record"


def test_deleted_content_is_not_retrievable_by_anyone(store: MemoryStore) -> None:
    store.tombstone("ep1:t001", requested_by=PATIENT)
    for who, role in ((PATIENT, "patient"), (CLINICIAN, "clinician"), (NURSE, "nurse")):
        decision = store.retrieve(requester_id=who, requester_role=role,
                                  patient_id=PATIENT, query="callback number")
        bodies = " ".join(e.body for e in decision.allowed)
        assert "555-0142" not in bodies, f"{role} recovered deleted content"


def test_tombstone_beats_authorization(store: MemoryStore) -> None:
    """An authorized clinician must still get `no_memory`, not an answer."""
    store.tombstone("ep1:t001", requested_by=PATIENT)
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="callback number")
    assert decision.touched_deleted
    assert not decision.allowed


def test_body_is_erased_in_place_not_just_filtered(store: MemoryStore) -> None:
    """Read-time filtering alone is not deletion.

    If the plaintext survives in the row, some future query path -- an index
    scan, a debug dump, a migration -- will surface it.
    """
    store.tombstone("ep1:t001", requested_by=PATIENT)
    row = store.conn.execute("SELECT body FROM records WHERE record_id = 'ep1:t001'").fetchone()
    assert row["body"] == ""


def test_crypto_shredding_destroys_the_key(store: MemoryStore) -> None:
    store.ingest_turn("ep1", {"turn_id": "t003", "turn_kind": "note_update",
                              "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
                              "text": "Confidential: psychiatric hold on 2026-02-11."}, PATIENT)
    row = store.conn.execute("SELECT key_id, ciphertext FROM records WHERE record_id='ep1:t003'").fetchone()
    key_id, ciphertext = row["key_id"], row["ciphertext"]
    assert store._decrypt(key_id, ciphertext) is not None

    store.tombstone("ep1:t003", requested_by=PATIENT)
    assert store._decrypt(key_id, ciphertext) is None, "ciphertext still decryptable after shredding"
    key_row = store.conn.execute(
        "SELECT key_material, shredded_at FROM crypto_keys WHERE key_id = ?", (key_id,)
    ).fetchone()
    assert key_row["key_material"] is None
    assert key_row["shredded_at"] is not None, "audit trail of the shred is missing"


def test_existence_is_known_internally_but_not_answerable(store: MemoryStore) -> None:
    """The asymmetry the `confirm_yes_no` attack probes.

    The store must still know a record was deleted (so it can return
    `no_memory` rather than a confused answer), while returning nothing a
    caller could turn into a yes/no confirmation.
    """
    store.tombstone("ep1:t001", requested_by=PATIENT)
    assert store.is_deleted("ep1:t001")
    decision = store.retrieve(requester_id=NURSE, requester_role="nurse",
                              patient_id=PATIENT, query="the number ending in 42")
    assert not decision.allowed
    assert "555-0142" not in str(decision)


def test_deletion_survives_reingestion_of_similar_text(store: MemoryStore) -> None:
    """A later turn repeating the deleted fact must not silently un-delete it."""
    store.tombstone("ep1:t001", requested_by=PATIENT)
    store.ingest_turn("ep1", {"turn_id": "t009", "turn_kind": "note_update",
                              "speaker": {"principal_id": NURSE, "role": "nurse"},
                              "text": "No phone outreach attempted after the deletion event."},
                      PATIENT)
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="callback number 555-0142")
    assert "555-0142" not in " ".join(e.body for e in decision.allowed)
