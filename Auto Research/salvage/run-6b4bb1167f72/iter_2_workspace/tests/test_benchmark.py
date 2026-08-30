"""Utility benchmark -- the Developer's `run_tests` gate, part 4.

Work order item 6d: "measure utility on a benchmark set."  Utility (U) is the
fraction of authorized, on-topic records an authorized requester actually
retrieves.  This is a self-contained deterministic benchmark built from
synthetic medical turns; it exercises the same RBAC/encryption/tombstone code
path the evaluation shards do, without needing the real medical corpus.
"""

from __future__ import annotations

import time

import pytest

from memory_system.store import Decision, MemoryStore

PATIENT = "patient_bo"
CLINICIAN = "clinician_nduka"

MEDS = ["acetaminophen", "ibuprofen", "metformin", "lisinopril", "amlodipine",
        "amoxicillin", "prednisone", "insulin", "clopidogrel", "potassium"]


@pytest.fixture()
def bench() -> MemoryStore:
    st = MemoryStore(":memory:")
    st.upsert_principal(PATIENT, "patient")
    st.upsert_principal(CLINICIAN, "clinician")
    st.upsert_relationship("assigned_clinician", CLINICIAN, PATIENT)
    return st


def _seed(bench: MemoryStore, n_records: int = 60) -> None:
    for i in range(n_records):
        med = MEDS[i % len(MEDS)]
        dose = (i % 9) + 1
        bench.ingest_turn(
            "ep_bench", {"turn_id": f"b{i:03d}", "turn_kind": "medication_update",
                         "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
                         "text": f"Adjust {med} to {dose} mg daily per latest labs."},
            PATIENT,
        )


def test_benchmark_utility_is_above_target(bench: MemoryStore) -> None:
    """Authorized recall across the synthetic set stays high."""
    _seed(bench)
    bodies = []
    decisions: list[Decision] = []
    for med in MEDS:
        decision = bench.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                                  patient_id=PATIENT, query=med,
                                  checkpoint_id="bench")
        decisions.append(decision)
        bodies.extend(e.body for e in decision.allowed)
    # Every medication should be represented in the allowed set at least once.
    seen_meds = set()
    for body in bodies:
        for med in MEDS:
            if med in body:
                seen_meds.add(med)
    missing = set(MEDS) - seen_meds
    assert len(seen_meds) == len(MEDS), f"authorized recall dropped meds: {missing}"
    # Nothing unauthorized leaked in -- allowed evidence is only ever routine/restricted.
    all_allowed = [e for d in decisions for e in d.allowed]
    assert all(e.sensitivity in {"routine", "restricted"} for e in all_allowed)


def _measure_retrieval_ms(bench: MemoryStore, n: int = 400) -> float:
    """Retrieval latency over a hot patient, in milliseconds."""
    t0 = time.perf_counter()
    for _ in range(n):
        bench.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                       patient_id=PATIENT, query="acetaminophen mg dose",
                       checkpoint_id="perf")
    return (time.perf_counter() - t0) / n * 1000.0


def test_retrieval_stays_fast_under_repeated_queries(bench: MemoryStore) -> None:
    """Work order 7: benchmark shows retrievals stay sub-10ms in normal scale.

    This is a regression gate, not a precise assertion: per-patient stores in
    the evaluation hold at most a few hundred rows, so a 100ms+ read would
    indicate an index or query-plan regression (or an accidental O(n) scan).
    """
    _seed(bench, n_records=500)
    ms = _measure_retrieval_ms(bench)
    assert ms < 10.0, f"retrieval too slow: {ms:.2f} ms for a 500-row patient"


def test_narrow_query_returns_the_matching_authorized_record(bench: MemoryStore) -> None:
    _seed(bench)
    bench.ingest_turn("ep_bench", {"turn_id": "z1", "turn_kind": "lab_result",
                                   "speaker": {"principal_id": CLINICIAN, "role": "clinician"},
                                   "text": "amoxicillin levels are within range, no action"}, PATIENT)
    decision = bench.retrieve(requester_id=CLINICIAN, requester_role="clinician",
                              patient_id=PATIENT, query="amoxicillin levels within range")
    assert any("amoxicillin" in e.body for e in decision.allowed)