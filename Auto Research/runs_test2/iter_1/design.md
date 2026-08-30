## Design Document: Iteration 1 – FTS-Enhanced Retrieval with Strict Relationship Enforcement
### Problem Analysis
Current baseline fails the benchmark with **U=0.0, A=1.0, F=1.0**.  
- **U=0** → No authorized record is ever returned. Likely because naïve term-overlap retrieval is too weak for natural-language queries, and/or because the RBAC join denies every record (e.g., the `care_team` pseudo‑relationship allows all clinicians/nurses to see routine content, but they may not have an explicit relationship, causing the deny–scope path to reject all; or conversely, the grant is granted but the query terms never match).
- **A=1** → Every query leaks unauthorized content. This points directly at the `care_team` pseudo‑relationship: `_relationship_ok` returns `True` for *any* user with a role in `_CARE_TEAM_ROLES`, without verifying that the requester is actually linked to the patient. This lets any clinician/nurse/pharmacist read routine records of any patient.
- **F=1** → Deletion is not effective. The tombstone mechanism relies on term‑overlap matching and read‑time filtering, but a future query path (e.g., a full‑text search index) could still expose deleted content if we do not proactively remove it.
### Proposed Design
This iteration targets **U** (utility) while **keeping A and F unchanged or improving them** by fixing the two root causes of A and F.
1. **Add a SQLite FTS5 index on plaintext bodies** for `routine` and `restricted` records.  
   - This provides a fast, relevant candidate retrieval for natural‑language queries, drastically improving U.
   - Confidential records remain encrypted and are handled via the existing decryption‑based path (they are not indexed, preserving at‑rest security).
2. **Enforce relationship integrity for all grants.**  
   - Replace the `care_team` pseudo‑relationship with a real relationship check: any role requiring `care_team` must have *some* relationship row (`relationships` table) linking them to the patient. This eliminates the A‑leak.
3. **Actively remove tombstoned records from the FTS index.**  
   - On `tombstone()` we delete the row from `records_fts`, so no future query (including FTS) can resurrect deleted content. This secures F.
4. **Refine the deletion‑request matching** to use FTS when available (falling back to term‑overlap), improving F‑recall without over‑deletion.
### Schema Changes (DDL)
- Create `records_fts` (FTS5) with `record_id` and `body`.
- Add an index on `relationships(patient_id, subject_id)` to accelerate relationship lookups.
### Retrieval Loop (updated)
1. **Candidate acquisition**: For `routine`/`restricted` content, run an FTS `MATCH` query using the user’s query, getting candidate `record_id`s ordered by rank.
2. **RBAC gates**: For each candidate, apply the three‑gate check (tombstone, role grant, relationship+scope) exactly as in the baseline, but now starting from a smaller candidate set.
3. **Confidential fallback**: If FTS yields no candidates, or for `confidential` records (which are not indexed), fall back to the original term‑overlap scan over the whole episode (with RBAC applied).
4. **Return `Decision`** identical in shape to baseline for the agent.
### Forgetting Mechanism
- On tombstone: delete from `records_fts` and, for confidential records, shred the key (already implemented). Also set `body=''` for plaintext records.
- The `crypto_keys` audit row remains, but the key is NULL.
### Work Order (Delta)
1. **`memory_system/schema.sql`**: Add `CREATE VIRTUAL TABLE records_fts USING fts5(record_id UNINDEXED, body);` and an index on `relationships(patient_id, subject_id)`.
2. **`memory_system/store.py`**:
   - In `initialize()`, after schema execution, create the FTS table (via `CREATE VIRTUAL TABLE`).
   - In `ingest_turn()`, after inserting a record, if `sensitivity != 'confidential'`, insert `(record_id, body)` into `records_fts`.
   - In `tombstone()`, delete `record_id` from `records_fts` before (or after) the existing logic.
   - Modify `retrieve()` to:
     - Build an FTS query from `query` (e.g., `body:term*` for each term).
     - Run `SELECT record_id FROM records_fts WHERE records_fts MATCH ?` to get top candidates.
     - For each candidate, apply the three gates using the existing logic (but only on those candidates).
     - If FTS finds none, fall back to the current sequential scan.
   - Change `_relationship_ok` for `requires_rel == "care_team"` to check `SELECT 1 FROM relationships WHERE subject_id=? AND patient_id=?`.
3. **`memory_system/agent.py`**: no change needed (agent already consumes `Decision`).
All changes preserve the public API and the local test suite.
### Target & Tradeoff
- **Target**: `U` (utility) – primary gain via FTS.
- **Expected tradeoff**: Slight increase in query‑time complexity (FTS index maintenance) and a tiny memory overhead. A and F are both improved (not sacrificed) because we eliminate the `care_team` leak and remove tombstoned rows from FTS. The net effect on MGS is strongly positive.
---
```json
{
  "schema_ddl": "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(record_id UNINDEXED, body);\nCREATE INDEX IF NOT EXISTS idx_rel_patient_subject ON relationships(patient_id, subject_id);",
  "migration_sql": "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(record_id UNINDEXED, body);\nINSERT INTO records_fts(record_id, body) SELECT record_id, body FROM records WHERE sensitivity IN ('routine','restricted') AND body != '';\nCREATE INDEX IF NOT EXISTS idx_rel_patient_subject ON relationships(patient_id, subject_id);",
  "retrieval_loop": "1. For routine/restricted content, use FTS5 MATCH to get candidate record_ids.\n2. Apply tombstone gate (via LEFT JOIN tombstones) to exclude deleted records.\n3. Apply role grant (role_grants) and relationship+scope checks using _relationship_ok (now enforcing relationships for care_team).\n4. For confidential records or if FTS yields no matches, fall back to the original term-overlap scan with the same gates.\n5. Return a Decision with allowed, denied_rbac, denied_scope, denied_tombstone.",
  "forgetting_mechanism": "On tombstone(): (a) delete row from records_fts; (b) if encrypted, shred the key (NULL key_material) and set key_id=NULL; (c) else set body='' in records. The row remains in records and tombstones for audit, but content is unrecoverable.",
  "work_order": [
    "In memory_system/schema.sql: add the records_fts virtual table DDL and the idx_rel_patient_subject index.",
    "In memory_system/store.py initialize(): after executing schema.sql, run the migration SQL to create records_fts if not present and backfill from existing records.",
    "In memory_system/store.py ingest_turn(): after INSERT OR REPLACE into records, if sensitivity != 'confidential', INSERT OR REPLACE into records_fts(record_id, body) VALUES (?, ?).",
    "In memory_system/store.py tombstone(): before/after existing logic, run DELETE FROM records_fts WHERE record_id = ?.",
    "In memory_system/store.py retrieve(): replace the current loop with a two-step approach: (a) build and execute an FTS5 query for candidates, (b) for each candidate, apply the same gates; if no candidates, fall back to the original scan. Ensure that the gates and decision fields remain unchanged.",
    "In memory_system/store.py _relationship_ok(): modify the care_team case to check for any relationship row (SELECT 1 FROM relationships WHERE subject_id=? AND patient_id=?) instead of relying solely on role membership."
  ],
  "targets_metric": "U (utility_accuracy)",
  "expected_tradeoff": "Improves U via FTS retrieval; tradeoff is a small increase in storage (FTS index) and index maintenance overhead. A and F are both improved (no cost) because we fix the care_team relationship leak and remove tombstoned rows from FTS, so the net MGS gain is positive."
}
```