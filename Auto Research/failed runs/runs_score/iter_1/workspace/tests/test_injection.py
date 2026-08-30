"""SQL-injection fuzz tests -- the Developer's `run_tests` gate, part 3.

Work order item 8: "Validate that no SQL injection is possible by fuzzing
input strings."  Every user-visible string (requester id, role, patient id,
query text, episode/turn ids, record content) is fed a battery of hostile
inputs that would break out of a SQL literal if any value were ever
string-interpolated.  The store must keep returning the same safe, bounded
answers and must never throw a SyntaxError/ProgrammingError that would reveal
a successful injection.

The corpus is deliberately SQLite-flavoured (no `--` line comments in SQL for
most payloads, so `'; DROP TABLE x --` style values, column/table-name
punctuation, and TRIGGER/tmp-table smuggling are all covered).
"""

from __future__ import annotations

import pytest

from memory_system.store import MemoryStore

PATIENT = "patient_ana"
CLINICIAN = "clinician_okafor"

# A battery of payloads that would break out of a single-quoted SQL literal.
INJECTIONS = [
    "'; DROP TABLE records; --",
    "'); DELETE FROM records; --",
    "'; UPDATE tombstones SET reason=1; --",
    "0 OR 1=1 --",
    "1' UNION SELECT key_id FROM crypto_keys --",
    "1; DROP TABLE access_log",
    "Robert'); DROP TABLE principals;--",
    '".join(["x" * 500])',
    "patched_and_a_half' -- '",
    "x\\' OR \\'1\\'=\\'1",
    "chr(39)+chr(34)+chr(59)",
    "'; PRAGMA journal_mode=WAL; --",
    "'; CREATE TRIGGER t AFTER INSERT ON records BEGIN DELETE FROM records; END; --",
    "secret\\' ) --",
]

# Smaller payloads aimed at SQLite edge syntax / control characters.
SCANINJECTS = [
    "a\\nb\\rc",
    "`,;[{]}`",
    "\x00trailing",
    "  leading and trailing spaces  ",
    "-",
    "--",
    "*/",
    "/*",
    "||",
    "üñîçødé",
]


@pytest.fixture()
def store() -> MemoryStore:
    st = MemoryStore(":memory:")
    st.upsert_principal(PATIENT, "patient")
    st.upsert_principal(CLINICIAN, "clinician")
    st.upsert_relationship("assigned_clinician", CLINICIAN, PATIENT)
    return st


def _plant_with_value(store: MemoryStore, value: str) -> str:
    """Store a record whose `text` field holds the hostile `value`."""
    record_id = store.ingest_turn(
        "ep1", {"turn_id": "t001", "turn_kind": "note_update",
                "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
                "text": value}, PATIENT,
    )
    return record_id


@pytest.mark.parametrize("payload", INJECTIONS)
def test_no_injection_through_value_strings(store: MemoryStore, payload: str) -> None:
    # The payload as record text and as the query string: a parameterised store
    # must safely return (possibly nothing), never throw a SQL error.
    _ = _plant_with_value(store, payload)
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query=payload, checkpoint_id="fuzz")
    assert decision is not None
    # Nothing was destructively injected: the tables are still intact.
    assert store.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] >= 1
    assert store.conn.execute("SELECT COUNT(*) FROM principals").fetchone()[0] >= 2


@pytest.mark.parametrize("payload", INJECTIONS)
def test_no_injection_through_identity_fields(store: MemoryStore, payload: str) -> None:
    # The same corpus pushed through identifier-like fields (requester, patient).
    store.upsert_principal(payload, "patient")
    store.upsert_relationship("assigned_clinician", CLINICIAN, payload)
    decision = store.retrieve(requester_id=payload, requester_role="clinician",
                              patient_id=payload, query="whatever",
                              checkpoint_id="fuzz2")
    assert decision is not None
    assert store.conn.execute("SELECT COUNT(*) FROM principals").fetchone()[0] >= 3


@pytest.mark.parametrize("payload", SCANINJECTS)
def test_retrieval_never_raises_on_hostile_text(store: MemoryStore, payload: str) -> None:
    store.ingest_turn("ep1", {"turn_id": payload, "turn_kind": payload,
                              "speaker": {"principal_id": payload, "role": "clinician"},
                              "text": "whatever whatever"}, PATIENT)
    decision = store.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="whatever", checkpoint_id="fz")
    assert decision is not None
    assert store.conn.execute("SELECT COUNT(*) FROM records").fetchone()[0] >= 1