-- migration 6: observable diagnostic census (Architect iteration 18).
-- Semantics: pure additive observability.  No release path, no deletion path,
-- no RBAC row changes.  idx_records_patient_seq accelerates the diagnostic
-- per-patient row walk in seq order (patient_id = ?, ORDER BY seq), replacing what
-- would otherwise be a seq sort over idx_records_patient.  Applied idempotently.
CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq);

-- The store reports SCHEMA_VERSION = 6 after applying; bump in store.py with a
-- one-line comment naming the index it added.
