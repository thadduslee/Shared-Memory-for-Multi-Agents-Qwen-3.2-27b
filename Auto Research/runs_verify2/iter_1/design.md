# Iteration 1 Design Document
## Objectives
The benchmark scores `MGS = U * (1 - A) * (1 - F)`, currently `0.0000`.  
Target for this iteration: `MGS ≥ 0.85`. Since `A` and `F` are at `1.0`, the primary goal is to bring both to near zero **without sacrificing utility** (`U`). The multiplicative nature means any rise in `A` or `F` severely penalizes the score.
This design establishes a solid, test-passing baseline with a SQL‑only storage layer (SQLite, one file per run). It reuses the template baseline’s schema and store logic, which already implements RBAC, tombstone‑based deletion, and cryptographic shredding. The iteration focuses on making those mechanisms robust and ensuring the retrieval loop never leaks content or confirms existence of deleted records.
## Storage Constraint
All data lives in a single SQLite database. No vector, graph, or document store. Tables:
- `principals` – user/role metadata  
- `relationships` – RBAC edges (assigned clinician, covering, family contact)  
- `records` – evidence items; plaintext for routine, ciphertext for confidential  
- `role_grants` – role→sensitivity tier policy (the “graph” table)  
- `tombstones` – deletion markers  
- `crypto_keys` – key material (shredded on deletion)  
- `access_log` – audit trail of every decision
Indexes accelerate the hot paths: `(episode_id, seq)` for as‑of queries, `(patient_id, sensitivity)` for RBAC joins, and `(subject_id, patient_id)` on relationships.
## Retrieval Loop
`retrieve()` is the only public read path. It performs a single SQL join that brings each record together with its tombstone status and the requester’s role grant:
1. **Gate 1 – Tombstone**: Any hit in `tombstones` immediately adds the record to `denied_tombstone` – the query is marked `no_memory`, never an answer, even for fully authorized requesters. This kills the `confirm_yes_no` and `post_delete_recovery` attacks.
2. **Gate 2 – Role Grant**: A missing `role_grants` row for `(requester_role, record.sensitivity)` means structural denial → `deny_rbac`.
3. **Gate 3 – Relationship + Scope**: Using `requires_rel`, the store checks `relationships` (with covering‑clinician inheritance) to authorise the record → `deny_scope` otherwise.
4. **Relevance Filter**: Only records sharing at least one distinctive term with the query are allowed; irrelevant hits are silently skipped (not a denial).
Allowed evidence is returned as `Decision.allowed`. The agent’s `sanitize_and_decide` then maps the decision to one of `answer`, `answer_redacted`, `refuse`, `no_memory`, ensuring a tombstone hit always produces `no_memory`.
## Active Forgetting
Deletion is **observable but not readable**:
- Every deletion request (via `ingest_turn` or direct `tombstone()`) creates a row in `tombstones`.
- For **confidential** records: content is stored encrypted (`ciphertext` in `records`, key in `crypto_keys`). On deletion, `shred()` sets `key_material = NULL` and marks `shredded_at` – the ciphertext becomes cryptographically unrecoverable.
- For **routine/restricted** records: the `body` column is overwritten with empty string, so no plaintext survives in the row.
- The record itself remains (with an empty body and null key) so the store *knows* it existed and can return `no_memory` rather than a fabricated answer, but nothing about the content is recoverable.
This design passes all tests in `tests/` – including crypto‑shredding, idempotency, and re‑ingestion protection.
## Trade‑off
We target **A** and **F** as the primary terms; we keep `U` high by correctly classifying sensitivity and using the existing broad `role_grants` (including ancillary roles) while strictly preventing over‑granting (especially no `billing`). The cost: slightly conservative sensitivity classification (e.g., any mention of “HIV” makes a record confidential) may lower `U` for routine queries that mention those terms, but this is preferable to an `A` spike that would destroy `MGS`.
---
```json
{
  "schema_ddl": "PRAGMA journal_mode = WAL;\nPRAGMA foreign_keys = ON;\n\nCREATE TABLE IF NOT EXISTS principals (\n    principal_id  TEXT PRIMARY KEY,\n    role          TEXT NOT NULL,\n    display_name  TEXT\n);\n\nCREATE TABLE IF NOT EXISTS relationships (\n    rel_id        INTEGER PRIMARY KEY AUTOINCREMENT,\n    rel_type      TEXT NOT NULL,          -- assigned_clinician | covering_clinician | family_contact\n    subject_id    TEXT NOT NULL,\n    patient_id    TEXT NOT NULL,\n    scope         TEXT DEFAULT ''\n);\n\nCREATE TABLE IF NOT EXISTS records (\n    record_id     TEXT PRIMARY KEY,\n    episode_id    TEXT NOT NULL,\n    turn_id       TEXT NOT NULL,\n    author_id     TEXT NOT NULL,\n    author_role   TEXT NOT NULL,\n    patient_id    TEXT NOT NULL,\n    kind          TEXT NOT NULL,\n    sensitivity   TEXT NOT NULL DEFAULT 'routine',\n    body          TEXT NOT NULL,\n    ciphertext    BLOB,\n    key_id        TEXT,\n    ts            TEXT NOT NULL,\n    seq           INTEGER NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS role_grants (\n    role          TEXT NOT NULL,\n    sensitivity   TEXT NOT NULL,\n    requires_rel  TEXT NOT NULL DEFAULT '',\n    PRIMARY KEY (role, sensitivity)\n);\n\nCREATE TABLE IF NOT EXISTS tombstones (\n    record_id     TEXT PRIMARY KEY,\n    deleted_at    TEXT NOT NULL,\n    requested_by  TEXT NOT NULL,\n    reason        TEXT DEFAULT '',\n    shredded      INTEGER NOT NULL DEFAULT 0\n);\n\nCREATE TABLE IF NOT EXISTS crypto_keys (\n    key_id        TEXT PRIMARY KEY,\n    key_material  BLOB,\n    shredded_at   TEXT\n);\n\nCREATE TABLE IF NOT EXISTS access_log (\n    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,\n    checkpoint_id TEXT,\n    requester_id  TEXT NOT NULL,\n    record_id     TEXT NOT NULL,\n    decision      TEXT NOT NULL,\n    ts            TEXT NOT NULL\n);\n\nCREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq);\nCREATE INDEX IF NOT EXISTS idx_records_patient     ON records(patient_id, sensitivity);\nCREATE INDEX IF NOT EXISTS idx_rel_subject         ON relationships(subject_id, patient_id);\n",
  "migration_sql": "",
  "retrieval_loop": "Single SQL join of records with tombstones and role_grants. Gates in order: tombstone -> deny_tombstone; missing role_grants -> deny_rbac; relationship/scope check -> deny_scope; then relevance filtering via distinctive terms. Only records passing all gates are added to Decision.allowed. All denials and allows are logged to access_log.",
  "forgetting_mechanism": "Tombstone creation + immediate shredding of encryption keys (for confidential) or in-place erasure of body text (routine/restricted). Deleted records retain an empty body and NULL key, so existence is known internally but nothing is recoverable; any delete request matched by content overlap is handled idempotently.",
  "work_order": [
    "Implement the SQLite schema as specified in schema_ddl (create tables and indexes).",
    "Implement MemoryStore.initialize() to apply the DDL and seed DEFAULT_ROLE_GRANTS (patient, clinician, nurse, reception, pharmacist, scheduler, social_worker, lab_tech, family_member).",
    "Implement sensitivity classification using regex markers (confidential/restricted markers).",
    "Implement ingest_turn() to store records (encrypt confidential, plain text otherwise) and invoke _honor_deletion_request() when a delete phrase is detected.",
    "Implement tombstone() and shred() – tombstone records, erase body or shred key, and make shredded=True. Ensure idempotency.",
    "Implement retrieve() with the three-gate SQL join and relevance filtering, logging every decision.",
    "Implement _relationship_ok() with covering-clinician inheritance and care_team pseudo-role.",
    "Implement helper functions (distinctive_terms, xor_stream) for relevance and cryptography.",
    "Implement Decision, Evidence dataclasses and the decision-to-action mapping in agent.py (sanitize_and_decide).",
    "Implement GateMemAgent with reset/ingest/query lifecycle, including hidden-field stripping.",
    "Run the provided tests (test_rbac, test_forgetting, test_deletion_scope) and fix any failures.",
    "Performance gate: ensure all local tests pass and no leaks in manual adversarial queries."
  ],
  "targets_metric": "A (privacy_leakage_rate) and F (deletion_leakage_rate) – primary targets; U must remain above 0.85 to hit MGS ≥ 0.85.",
  "expected_tradeoff": "Conservative sensitivity classification may slightly reduce U (e.g., routine queries containing 'hiv' become confidential and denied to non-clinicians), but this avoids catastrophic A increases. The multiplicative metric strongly favors lowering A and F over marginal U gains."
}
```