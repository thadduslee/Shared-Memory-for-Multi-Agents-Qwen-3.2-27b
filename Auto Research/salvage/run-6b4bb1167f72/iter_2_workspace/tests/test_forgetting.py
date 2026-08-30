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
    # The key material must NOT survive in the KMS after the shred.
    assert store._kms.get_key(key_id) is None, "KMS still holds the shredded key"
    # The audit row survives with a shred timestamp but never held key material.
    key_row = store.conn.execute(
        "SELECT shredded_at FROM crypto_keys WHERE key_id = ?", (key_id,)
    ).fetchone()
    assert key_row["shredded_at"] is not None, "audit trail of the shred is missing"


def test_kms_key_material_never_lands_in_the_sqlite_file(store: MemoryStore) -> None:
    """The Architect's 'external KMS' mandate: no key bytes in the DB.

    A `key_material` column existing at all would be a local copy an attacker
    with the file could use to decrypt ciphertext.  The schema must instead
    store only an opaque key_id and hold bytes in the (external, mocked) KMS.
    """
    pragma = store.conn.execute("PRAGMA table_info(crypto_keys)").fetchall()
    cols = {r["name"] for r in pragma}
    assert "key_material" not in cols, "key material must never be persisted locally"
    assert "key_id" in cols
    assert "shredded_at" in cols


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


def test_unrelated_utility_query_unaffected_by_tombstone(store: MemoryStore) -> None:
    """A tombstone must not poison queries that are NOT about the deleted fact.

    Every episode in the corpus carries one deleted fact.  If the delete gate
    fired for *any* query that scanned past the tombstoned row, then every
    post-deletion utility checkpoint (e.g. "summarize the medication plan")
    would collapse to `no_memory` and utility would crater.  The gate must only
    fire when the query actually references the forgotten content.
    """
    # Seed a distinct, authorized, non-deleted record the utility query should
    # be able to find.
    store.ingest_turn("ep1", {"turn_id": "t010", "turn_kind": "medication_update",
                              "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
                              "text": "Start acetaminophen 650 mg daily."}, PATIENT)
    # Delete the unrelated secret record.
    store.tombstone("ep1:t001", requested_by=PATIENT)

    # A query about the medication, with no overlap to the deleted fact, must
    # still retrieve the approved record (not read as no_memory).
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="acetaminophen 650 mg")
    assert any("acetaminophen" in e.body for e in decision.allowed)
    assert decision.touched_deleted is False, "unrelated tombstone leaked into query"


def test_plaintext_bytes_never_land_in_the_sqlite_file(tmp_path) -> None:
    """Work order 6c: after forget, the item is unrecoverable from the file.

    For *confidential* content the plaintext is AES-GCM sealed before it ever
    reaches SQLite, so the raw bytes must not appear anywhere in the on-disk
    file -- neither in `records.body` nor scrubbed into a freelist page.
    A brute-force byte scan is the strongest guarantee we can give locally.
    """
    db_file = tmp_path / "leak.sqlite"
    st = MemoryStore(str(db_file))
    st.upsert_principal(PATIENT, "patient")
    st.upsert_principal(CLINICIAN, "clinician")
    st.upsert_relationship("assigned_clinician", CLINICIAN, PATIENT)

    secret = "safe callback number 555-0199"
    st.ingest_turn("ep1", {"turn_id": "t001", "turn_kind": "portal_message",
                           "speaker": {"principal_id": PATIENT, "role": "patient"},
                           "text": f"Confidential: please store {secret}."}, PATIENT)

    st.conn.commit()
    blob = db_file.read_bytes()
    # The confidential marker forces encryption, so the raw secret must not be
    # findable as plaintext bytes anywhere in the file.
    assert secret.encode() not in blob, "confidential plaintext leaked into the SQLite file"

    # Jeff the store; also verify the ciphertext cannot be decrypted after shred.
    row = st.conn.execute("SELECT key_id, ciphertext FROM records WHERE record_id='ep1:t001'").fetchone()
    assert row is not None
    key_id, ciphertext = row["key_id"], row["ciphertext"]
    assert st._decrypt(key_id, ciphertext) is not None
    st.tombstone("ep1:t001", requested_by=PATIENT)
    assert st._decrypt(key_id, ciphertext) is None, "ciphertext decrypts after key shredding"
    assert secret.encode() not in db_file.read_bytes(), "plaintext reappeared after shred"


def test_routine_content_is_wiped_from_body_after_forget(tmp_path) -> None:
    """A routine (plaintext) record's body is overwritten, not left behind."""
    db_file = tmp_path / "routine_wipe.sqlite"
    store = MemoryStore(str(db_file))
    store.upsert_principal(PATIENT, "patient")
    store.upsert_principal(CLINICIAN, "clinician")
    secret = "callback line 555-0233"
    store.ingest_turn("ep1", {"turn_id": "t001", "turn_kind": "portal_message",
                              "speaker": {"principal_id": PATIENT, "role": "patient"},
                              "text": f"Please store {secret}."}, PATIENT)
    assert secret.encode() in db_file.read_bytes(), "precondition: plaintext stored"
    store.tombstone("ep1:t001", requested_by=PATIENT)
    store.conn.commit()
    # After secure_delete overwrites the freed page and the body is blanked,
    # the consecutive full secret should not appear as contiguous bytes.
    assert secret.encode() not in db_file.read_bytes(), "deleted plaintext left recoverable"
