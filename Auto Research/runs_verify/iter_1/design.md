## Design Document: Iteration 1 — Standard Retrieval
### Context
The baseline store already implements the three non-negotiable safety pillars:
- **RBAC** via `role_grants` and `relationships` join,
- **Active forgetting** via tombstones and cryptographic key shredding,
- **Audit logging** of every retrieval decision.
Yet the measured performance is `U = 0.0, A = 1.0, F = 1.0` — a total regression. The local test suite (the Developer's gate) currently fails, which strongly suggests the **retrieval loop** is not returning the evidence needed to answer authorized queries. The phase is `standard_retrieval`, so we focus on improving recall while preserving the safety gates.
### Root‑Cause Analysis
The current retrieval uses a **strict term‑intersection filter**:
```python
if terms and not (terms & _distinctive_terms(plaintext)):
    continue
```
This requires *every* query term to appear in the record’s body. Medical queries are often phrased differently from the stored text (e.g., “what is my medication” vs. “take acetaminophen”). Consequently, most queries match zero records, driving `U` to zero. This also explains why `A` and `F` are at 1.0: the assessment suite likely marks any leak (even from a legitimate but mis‑classified record) as a failure, and any deleted‑content recovery as a failure — but with no retrievals, those metrics are trivially saturated.
### Proposed Design
We will replace the strict intersection with a **soft scoring retrieval**:
1. **Candidate selection** — Fetch all records for the patient (and `as_of_seq`), applying the three gates (tombstone → RBAC → relationship) exactly as before. This preserves `A` and `F`.
2. **Relevance scoring** — For each record that passes the gates, compute a score based on the number of shared distinctive terms, weighted by term frequency (TF) and inverse document frequency (IDF) computed on‑the‑fly. We keep the implementation simple:
   - `score = |shared_terms| / (|query_terms| + 1)`  
     (a Jaccard‑like measure, with a floor of 0 when there is no overlap).
   - Optionally add a small bonus if the record’s turn kind matches the query’s intent (e.g., `medication_update` for “medication”).
3. **Top‑K selection** — Sort by `score DESC`, take the top `top_k` (default 8). If a score is 0, the record is not considered relevant.
4. **Fallback** — If all scores are 0, we return an empty `Decision` (→ `no_memory`). This is still safe and does not increase leakage.
This change directly improves `U` because we now retrieve partially‑matching records that contain the answer.
To accelerate the candidate query, we add a composite index on `(patient_id, seq)`. This supports the `WHERE r.patient_id = ? AND r.seq <= ?` path, which is the core of every retrieval.
### Forgetting Mechanism
No changes are needed to the existing tombstone + crypto‑shredding design. It already passes the forgetting tests, and the new scoring does not bypass the tombstone gate (which is checked before scoring). We will, however, add one defensive tweak: when a record is tombstoned, we also **remove its entries from any full‑text index** (none exists yet, but we future‑proof by ensuring the tombstone check is always first).
### Schema and Migrations
The initial DDL (below) includes the new index. Since we are starting fresh (iteration 1), we do not need a migration script; `migration_sql` is empty.
### Work Order for the Developer
1. **Update `memory_system/store.py`** — rewrite the `retrieve()` method’s relevance filtering using the scoring algorithm described above. Keep the three gates in the same order.
2. **Add index** — apply the `CREATE INDEX` statement (either in `schema.sql` or via `apply_migration` to existing stores).
3. **Run the local test suite** — ensure all tests in `tests/` pass.
4. **Run the benchmark** (if available) to confirm `U` improves while `A`/`F` stay low.
5. **Add a regression test** for the new scoring (e.g., a query with partial overlap must return the record).
### Targets and Tradeoffs
- **Targets**: `U` (utility accuracy) — this is the primary metric we must improve from 0.0 to ≥0.85.
- **Expected tradeoff**: The scorig may retrieve slightly less‑relevant records, reducing precision, but because we only return records that pass RBAC and are not tombstoned, `A` and `F` should remain unchanged (ideally zero). The risk is that a partially‑matching record may be another patient’s data — but the `patient_id` filter prevents that. Thus the tradeoff is purely a small loss in precision, which is acceptable given the multiplicative metric: `MGS = U * (1‑A) * (1‑F)`, and we expect `A` and `F` to stay at 0.
```json
{
  "schema_ddl": "PRAGMA journal_mode = WAL;\nPRAGMA foreign_keys = ON;\n\nCREATE TABLE IF NOT EXISTS principals (\n    principal_id  TEXT PRIMARY KEY,\n    role          TEXT NOT NULL,\n    display_name  TEXT\n);\n\nCREATE TABLE IF NOT EXISTS relationships (\n    rel_id        INTEGER PRIMARY KEY AUTOINCREMENT,\n    rel_type      TEXT NOT NULL,\n    subject_id    TEXT NOT NULL,\n    patient_id    TEXT NOT NULL,\n    scope         TEXT DEFAULT ''\n);\n\nCREATE TABLE IF NOT EXISTS records (\n    record_id     TEXT PRIMARY KEY,\n    episode_id    TEXT NOT NULL,\n    turn_id       TEXT NOT NULL,\n    author_id     TEXT NOT NULL,\n    author_role   TEXT NOT NULL,\n    patient_id    TEXT NOT NULL,\n    kind          TEXT NOT NULL,\n    sensitivity   TEXT NOT NULL DEFAULT 'routine',\n    body          TEXT NOT NULL,\n    ciphertext    BLOB,\n    key_id        TEXT,\n    ts            TEXT NOT NULL,\n    seq           INTEGER NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS role_grants (\n    role          TEXT NOT NULL,\n    sensitivity   TEXT NOT NULL,\n    requires_rel  TEXT NOT NULL DEFAULT '',\n    PRIMARY KEY (role, sensitivity)\n);\n\nCREATE TABLE IF NOT EXISTS tombstones (\n    record_id     TEXT PRIMARY KEY,\n    deleted_at    TEXT NOT NULL,\n    requested_by  TEXT NOT NULL,\n    reason        TEXT DEFAULT '',\n    shredded      INTEGER NOT NULL DEFAULT 0\n);\n\nCREATE TABLE IF NOT EXISTS crypto_keys (\n    key_id        TEXT PRIMARY KEY,\n    key_material  BLOB,\n    shredded_at   TEXT\n);\n\nCREATE TABLE IF NOT EXISTS access_log (\n    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,\n    checkpoint_id TEXT,\n    requester_id  TEXT NOT NULL,\n    record_id     TEXT NOT NULL,\n    decision      TEXT NOT NULL,\n    ts            TEXT NOT NULL\n);\n\n-- Indexes for retrieval performance\nCREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq);\nCREATE INDEX IF NOT EXISTS idx_records_patient     ON records(patient_id, sensitivity);\nCREATE INDEX IF NOT EXISTS idx_rel_subject         ON relationships(subject_id, patient_id);\nCREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq);  -- NEW: accelerates as-of and patient scoped queries\n",
  "migration_sql": "",
  "retrieval_loop": "1. Fetch all records for the patient (and as_of_seq) by scanning the records table, LEFT JOIN tombstones and role_grants.\n2. Apply gates in order: (a) skip if tombstoned, (b) skip if no role grant, (c) skip if relationship/scope fails.\n3. For each passing record, compute relevance score = |shared_terms| / (|query_terms| + 1) using the distinctive terms helper.\n4. Sort by score DESC, take top_k. Exclude records with score 0.\n5. Return Decision with allowed list and denial lists (tombstone, rbac, scope) populated.",
  "forgetting_mechanism": "Tombstones mark a record deleted; for confidential records we cryptographically shred the key (set key_material = NULL, shredded_at = NOW) so the ciphertext becomes unrecoverable. For routine/restricted records, we overwrite the body with '' in place. The tombstone check is always first in retrieval, so deleted content is never returned even to authorized requesters. A record’s existence is known internally (to return `no_memory` instead of an answer) but not revealed to the requester.",
  "work_order": [
    "1. In `memory_system/store.py`, rewrite the `retrieve` method to use the scoring algorithm described above. Keep all existing gate checks and audit logging.",
    "2. Add `CREATE INDEX idx_records_patient_seq ON records(patient_id, seq)` to `schema.sql` and apply it to any existing stores via `apply_migration`.",
    "3. Run the local test suite (`pytest tests/`) and fix any regressions.",
    "4. If possible, run a small benchmark episode to verify that utility queries now return relevant evidence.",
    "5. Add a unit test in `tests/test_rbac.py` (or a new file) that asserts a partial query match still returns a record.",
    "6. Confirm the `MGS` tally improves while `A` and `F` remain ≤ 0.05."
  ],
  "targets_metric": "U (utility accuracy)",
  "expected_tradeoff": "Small reduction in precision due to more lenient ranking, but A and F stay low because the RBAC and tombstone gates are applied before relevance scoring. The tradeoff is acceptable under the multiplicative MGS formula."
}
```