"""RBAC unit tests -- the Developer's `run_tests` gate, part 1.

These are the *local* tests: fast, deterministic, no LLM, no benchmark data.
They exist so the Developer's ReAct loop can find an access-control regression
in a second rather than after a 579-checkpoint evaluation run.
"""

from __future__ import annotations

import pytest

from memory_system.store import MemoryStore
from memory_system.agent import GateMemAgent, sanitize_and_decide


PATIENT = "patient_ana"
ASSIGNED = "clinician_okafor"
COVERING = "clinician_lindqvist"
STRANGER = "clinician_stranger"
NURSE = "nurse_bell"
FAMILY = "family_mateo"


@pytest.fixture()
def store() -> MemoryStore:
    st = MemoryStore(":memory:")
    st.upsert_principal(PATIENT, "patient", "Ana")
    st.upsert_principal(ASSIGNED, "clinician", "Dr. Okafor")
    st.upsert_principal(COVERING, "clinician", "Dr. Lindqvist")
    st.upsert_principal(STRANGER, "clinician", "Dr. Nobody")
    st.upsert_principal(NURSE, "nurse", "Nurse Bell")
    st.upsert_principal(FAMILY, "family_member", "Mateo")
    st.upsert_relationship("assigned_clinician", ASSIGNED, PATIENT)
    st.upsert_relationship("covering_clinician", COVERING, PATIENT, "weekend")
    st.upsert_relationship("family_contact", FAMILY, PATIENT, "logistics only")

    st.ingest_turn("ep1", {"turn_id": "t001", "turn_kind": "medication_update",
                           "speaker": {"principal_id": ASSIGNED, "role": "clinician"},
                           "text": "Stop ibuprofen; start acetaminophen 650 mg."}, PATIENT)
    st.ingest_turn("ep1", {"turn_id": "t002", "turn_kind": "note_update",
                           "speaker": {"principal_id": ASSIGNED, "role": "clinician"},
                           "text": "Confidential: HIV screening result reactive."}, PATIENT)
    return st


def _retrieve(store: MemoryStore, who: str, role: str, query: str):
    return store.retrieve(requester_id=who, requester_role=role,
                          patient_id=PATIENT, query=query, checkpoint_id="test")


def test_confidential_is_encrypted_at_rest(store: MemoryStore) -> None:
    """A confidential body must not sit in plaintext in the `body` column."""
    row = store.conn.execute(
        "SELECT body, ciphertext, key_id, sensitivity FROM records WHERE turn_id = 't002'"
    ).fetchone()
    assert row["sensitivity"] == "confidential"
    assert row["body"] == ""
    assert row["ciphertext"] is not None
    assert row["key_id"] is not None


def test_assigned_clinician_sees_confidential(store: MemoryStore) -> None:
    decision = _retrieve(store, ASSIGNED, "clinician", "HIV screening result")
    bodies = " ".join(e.body for e in decision.allowed)
    assert "reactive" in bodies
    assert not decision.denied_scope


def test_covering_clinician_inherits_assigned_grant(store: MemoryStore) -> None:
    """Weekend coverage must not destroy utility."""
    decision = _retrieve(store, COVERING, "clinician", "HIV screening result")
    assert decision.allowed, "covering clinician should inherit the assigned grant"


def test_unassigned_clinician_is_denied(store: MemoryStore) -> None:
    decision = _retrieve(store, STRANGER, "clinician", "HIV screening result")
    assert not decision.allowed
    assert decision.touched_unauthorized


def test_nurse_has_no_confidential_grant(store: MemoryStore) -> None:
    """`role_grants` has no (nurse, confidential) row, so the deny is structural."""
    decision = _retrieve(store, NURSE, "nurse", "HIV screening result")
    assert not any(e.sensitivity == "confidential" for e in decision.allowed)
    assert decision.denied_rbac


def test_family_scoped_to_routine_only(store: MemoryStore) -> None:
    decision = _retrieve(store, FAMILY, "family_member", "HIV screening result")
    assert not decision.allowed
    assert decision.touched_unauthorized


def test_family_gets_no_confidential_even_on_routine_query(store: MemoryStore) -> None:
    decision = _retrieve(store, FAMILY, "family_member", "ibuprofen acetaminophen")
    assert all(e.sensitivity == "routine" for e in decision.allowed)


def test_decision_maps_to_refuse_not_answer() -> None:
    from memory_system.store import Decision

    decision = Decision(denied_rbac=["r1"])
    action, _ = sanitize_and_decide(decision, "family_member")
    assert action == "refuse"


def test_partial_authorization_is_redacted() -> None:
    from memory_system.store import Decision, Evidence

    decision = Decision(
        allowed=[Evidence("r1", "t1", "clinician", "dialogue", "routine", "take acetaminophen")],
        denied_rbac=["r2"],
    )
    action, _ = sanitize_and_decide(decision, "nurse")
    assert action == "answer_redacted"


def test_every_denial_is_audited(store: MemoryStore) -> None:
    _retrieve(store, FAMILY, "family_member", "HIV screening result")
    rows = store.conn.execute(
        "SELECT decision FROM access_log WHERE requester_id = ?", (FAMILY,)
    ).fetchall()
    assert rows, "denials must be written to access_log for Critic evidence"
    assert all(r["decision"] in {"allow", "deny_rbac", "deny_scope", "deny_tombstone"} for r in rows)


def test_agent_never_reads_hidden_annotations() -> None:
    """The field wall, tested at the agent boundary."""
    agent = GateMemAgent(":memory:")
    agent.reset({
        "episode_id": "ep1",
        "entities": {"principals": [{"principal_id": PATIENT, "role": "patient"}],
                     "relationships": []},
    })
    agent.ingest({"turn_id": "t001", "turn_kind": "dialogue",
                  "speaker": {"principal_id": PATIENT, "role": "patient"},
                  "text": "Stop ibuprofen; start acetaminophen 650 mg."})
    out = agent.query({
        "checkpoint_id": "c1", "episode_id": "ep1", "as_of_turn_id": "t001",
        "asker": {"principal_id": PATIENT, "role": "patient"},
        "query_text": "what should I take",
        # Poison: if any of these influence the output, the wall has a hole.
        "expected_action": "refuse",
        "judge_spec": {"include": ["SHOULD-NOT-APPEAR"]},
        "leak_targets": ["SHOULD-NOT-APPEAR"],
        "query_type": "utility",
    })
    assert "SHOULD-NOT-APPEAR" not in str(out)
    assert out["action"] in {"answer", "answer_redacted", "refuse", "no_memory"}
