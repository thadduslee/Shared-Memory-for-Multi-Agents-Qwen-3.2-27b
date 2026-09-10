"""Regression tests for the two ways the deletion path has broken a run.

Both failures were invisible to the existing suite: a build that tombstoned the
whole episode on any delete request, and a `tombstone()` that raised
IntegrityError the second time it saw a record.  Each cost a full iteration.
"""

from __future__ import annotations

import pytest

from memory_system.store import MemoryStore

PATIENT = "patient_ana"
CLINICIAN = "clinician_okafor"
SECRET = "safe callback number 555-0142"
KEEP = "metformin 500 mg twice daily with food"


@pytest.fixture()
def store() -> MemoryStore:
    st = MemoryStore(":memory:")
    st.upsert_principal(PATIENT, "patient")
    st.upsert_principal(CLINICIAN, "clinician")
    st.upsert_relationship("assigned_clinician", CLINICIAN, PATIENT)
    st.ingest_turn("ep1", {"turn_id": "t001", "turn_kind": "portal_message",
                           "speaker": {"principal_id": PATIENT, "role": "patient"},
                           "text": f"Please store this: {SECRET}."}, PATIENT)
    st.ingest_turn("ep1", {"turn_id": "t002", "turn_kind": "portal_message",
                           "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
                           "text": f"Plan: {KEEP}."}, PATIENT)
    return st


def test_deletion_request_spares_unrelated_records(store: MemoryStore) -> None:
    """A delete request must tombstone what it names, not the whole episode."""
    store.ingest_turn("ep1", {"turn_id": "t003", "turn_kind": "portal_message",
                              "speaker": {"principal_id": PATIENT, "role": "patient"},
                              "text": f"Actually, delete {SECRET} from the shared assistant memory."},
                      PATIENT)
    assert store.is_deleted("ep1:t001"), "the deletion request did not match its target"
    assert not store.is_deleted("ep1:t002"), (
        "the deletion request tombstoned an unrelated record; a whole-episode "
        "wipe drives U to zero because nothing survives to retrieve"
    )
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="metformin dose")
    assert decision.allowed, "no record survived the deletion request"


def test_repeated_deletion_requests_do_not_raise(store: MemoryStore) -> None:
    """Two delete requests in one episode must not crash the shard."""
    for turn_id in ("t003", "t004"):
        store.ingest_turn("ep1", {"turn_id": turn_id, "turn_kind": "portal_message",
                                  "speaker": {"principal_id": PATIENT, "role": "patient"},
                                  "text": f"Please delete {SECRET} from memory."},
                          PATIENT)
    assert store.is_deleted("ep1:t001")


def test_tombstone_is_idempotent(store: MemoryStore) -> None:
    """`tombstone()` is called from several paths; the second must be a no-op."""
    store.tombstone("ep1:t001", requested_by=PATIENT, reason="first")
    store.tombstone("ep1:t001", requested_by=PATIENT, reason="second")
    assert store.is_deleted("ep1:t001")


# ---------------------------------------------------------------------------
# *_deleted_*_line regressions: a "delete the <facility> line" request that
# quotes only a facility's NAME must tombstone the phone-bearing logistics line
# and NOT sweep in a collocated appointment/scheduling record that merely names
# the same place (which would strip later retrievable scheduling content).
# ---------------------------------------------------------------------------

LINE = "River House front desk: call 415-555-0168 for pickup"
APPT = "River House: EEG scheduled Friday April 4 at 2:00 PM"
PHONE = "415-555-0168"


def _build_line_episode(store: MemoryStore, ep: str) -> None:
    store.ingest_turn(
        ep, {"turn_id": "t001", "turn_kind": "portal_message",
             "speaker": {"principal_id": PATIENT, "role": "patient"},
             "text": f"Please store the line: {LINE}."},
        PATIENT,
    )
    store.ingest_turn(
        ep, {"turn_id": "t002", "turn_kind": "scheduling",
             "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
             "text": f"Please store the appointment: {APPT}."},
        PATIENT,
    )


def _find_appointment(decision: object) -> str | None:
    for ev in decision.allowed:
        if "EEG" in ev.body:
            return ev.body
    return None


def test_facility_line_deletion_spares_collocated_scheduling(store: MemoryStore) -> None:
    """A 'delete the River House line' request tombstones the line record only.

    The appointment at the same facility shares River/House/contact words with
    the request but none of its phone digests, so it must survive -- a later
    'EEG Friday' retrieval still returns it with the 2:00 time intact.
    """
    _build_line_episode(store, "ep_line")
    store.ingest_turn(
        "ep_line", {"turn_id": "t003", "turn_kind": "portal_message",
                    "speaker": {"principal_id": PATIENT, "role": "patient"},
                    "text": "Please delete the River House line from my memory."},
        PATIENT,
    )
    assert store.is_deleted("ep_line:t001"), (
        "the line record the request names was not tombstoned"
    )
    assert not store.is_deleted("ep_line:t002"), (
        "the collocated appointment record was collateral-damaged by a line "
        "deletion that only quoted the facility name"
    )
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="EEG Friday")
    body = _find_appointment(decision)
    assert body and "2:00" in body, (
        "after a line-only deletion the EEG appointment must still be "
        "retrievable with its scheduled time"
    )


def test_phone_carrying_request_overrides_guard_and_deletes_target(
    store: MemoryStore,
) -> None:
    """A deletion request that itself names the phone still deletes the line.

    When the request carries the shared number (e.g. 'delete the River House
    415-555-0168 line'), the line record shares a phone digest with the request
    and must be tombstoned even though it also overlaps the facility's location
    words -- the phone override keeps true targets deleted (F stays at 0)
    instead of letting the location guard spare them.
    """
    _build_line_episode(store, "ep_phone")
    store.ingest_turn(
        "ep_phone", {"turn_id": "t003", "turn_kind": "portal_message",
                     "speaker": {"principal_id": PATIENT, "role": "patient"},
                     "text": f"Please delete the River House {PHONE} line from my memory."},
        PATIENT,
    )
    assert store.is_deleted("ep_phone:t001"), (
        "a phone-bearing deletion request must still tombstone the line record; "
        "the phone override should not let the logistics guard spare a true target"
    )
    assert not store.is_deleted("ep_phone:t002"), (
        "the collocated appointment remains intact"
    )
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="EEG Friday")
    body = _find_appointment(decision)
    assert body and "2:00" in body, (
        "the appointment should still answer an 'EEG Friday' retrieval"
    )
