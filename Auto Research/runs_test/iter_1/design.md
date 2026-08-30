## Iteration 1 Design
### Problem Analysis
The measured metrics are all at their worst: U=0, A=1, F=1, MGS=0. The legacy retrieval code has a severe flaw that makes **every** query return `no_memory` whenever any record in the episode is tombstoned. Because `retrieve()` scans **all** records of the patient and unconditionally adds **every** tombstone to `denied_tombstone` before checking relevance, an unrelated query like "metformin dose" will trigger `touched_deleted = True` if *any* tombstone exists. Since `sanitize_and_decide()` treats `touched_deleted` as the highest‑priority state, it returns `no_memory` for that query, nuking U. Furthermore, this over‑broad behaviour also breaks the `confirm_yes_no` attack detection—the system can no longer distinguish *"I know nothing"* from *"there was a deleted fact"*.
The root cause is that we conflate *existence* of a tombstone with *relevance* of that tombstone to the query. We must make tombstone handling **query‑sensitive**: a tombstone should only trigger `no_memory` when the query actually refers to the content that was deleted. Because the plaintext is erased on deletion, we need a separate, safe way to remember *what terms* were deleted, without retaining the confidential content itself.
### Proposed Solution
We introduce a new table `deletion_terms` that stores the distinctive terms of a record **when it is tombstoned**, before cryptographic shredding erases the plaintext. This table is used only by the retrieval logic to decide whether a query "touches" a deleted record. It never appears in any output. This achieves three things:
1. **U recovery**: unrelated queries no longer see tombstone hits; they can retrieve active records freely.
2. **F accuracy**: the store can still distinguish *"you asked about a deleted fact"* (query terms overlap with `deletion_terms`) and return `no_memory`, while never confirming *content*.
3. **A safety**: because `deletion_terms` contains only term stems, not full text, and is never exposed to callers, it cannot be used to reconstruct the deleted content.
We also refine the deletion‑request matcher to catch more phrasings, reducing missed deletions (the main cause of F=1).
---
### Targeted Metric
**U** (utility) is the primary target – it is currently zero, and the false‑tombstone bug is the single largest contributor. We expect **A** and **F** to remain at (or near) zero because the RBAC filter is unchanged and the deletion mechanism becomes more precise. The tradeoff: adding `deletion_terms` adds one more table and a small retrieval‑time lookup, but that is negligible compared to the false‑no_memory damage.
---
### Schema Changes
#### New table: `deletion_terms`
| column | type | purpose |
|--------|------|---------|
| `record_id` | TEXT | FK to `records.record_id` (the tombstoned record) |
| `term` | TEXT | a distinctive term extracted from the record body |
| `patient_id` | TEXT | to scope the lookup to the current patient |
Primary key `(record_id, term)`.
**Index**: `idx_deletion_terms_patient_term ON deletion_terms(patient_id, term)` – accelerates the exact query used in `retrieve()` to find which tombstones are relevant to the query.
---
### Retrieval Loop (modified)
The new `retrieve()` flow:
1. Compute `terms` = `_distinctive_terms(query)`.
2. Query `deletion_terms` for the given `patient_id` and `terms` → obtain `matched_tombstones` (set of `record_id`s).
3. Iterate over *all* records for the patient (same SQL as before).
   - If `tomb` is not NULL:
     - If `record_id ∈ matched_tombstones` → add to `denied_tombstone`, log as `deny_tombstone`.
     - Else → **skip** this record (it is a tombstone but irrelevant to the query).
   - If `tomb` is NULL → proceed with RBAC gates as before.
4. The `Decision` object now only contains tombstone denials for **relevant** tombstones, so `touched_deleted` behaves correctly.
### Forgetting Mechanism (modified)
In `tombstone()`:
1. Fetch the record’s `body`, `ciphertext`, `key_id`, and `patient_id`.
2. Determine the plaintext:
   - If `body` is non‑empty, use it.
   - Else if `ciphertext` and `key_id` are present, decrypt it (key still available).
3. Compute `_distinctive_terms(plaintext)`.
4. Insert each term into `deletion_terms` (if not already present).
5. Then:
   - For confidential records: call `shred()` to destroy the key.
   - For plaintext records: overwrite `body` with `''`.
6. Mark the tombstone as shredded.
The `deletion_terms` table is now the persistent record of what was deleted, enabling query‑sensitive tombstone detection without retaining the actual content.
---
### Migration SQL (step applied on existing DB)
```sql
CREATE TABLE IF NOT EXISTS deletion_terms (
    record_id  TEXT NOT NULL REFERENCES records(record_id),
    term       TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    PRIMARY KEY (record_id, term)
);
CREATE INDEX IF NOT EXISTS idx_deletion_terms_patient_term
    ON deletion_terms(patient_id, term);
```
**Rationale**: the index is used directly by the new retrieval step `SELECT ... WHERE patient_id = ? AND term IN (...)`, making tombstone‑matching a constant‑time lookup per patient/term.
---
### Work Order (concrete steps)
1. **Update `memory_system/schema.sql`** – add the `deletion_terms` table and the accompanying index. Also bump `SCHEMA_VERSION` to `2` in `store.py`.
2. **Modify `MemoryStore.tombstone()`** in `store.py` – before shredding or clearing, extract the plaintext (via `body` or `_decrypt(...)`), compute `_distinctive_terms(plaintext)`, and persist each term – along with `record_id` and `patient_id` – into `deletion_terms` using `INSERT OR IGNORE`.
3. **Modify `MemoryStore.retrieve()`** in `store.py` – after computing `terms`, add a query to `deletion_terms` to obtain `matched_tombstones`. In the main loop, change the tombstone handling: only add a tombstone to `denied_tombstone` if its `record_id` is in `matched_tombstones`; otherwise `continue` (do not treat it as relevant).
4. **Improve `_honor_deletion_request()`** in `store.py` – broaden the deletion‑intent detection to catch more phrasings, e.g., treat any message containing at least one of (`delete`, `remove`, `erase`, `forget`, `purge`, `scrub`, `wipe`) and at least one target word (`memory`, `record`, `note`, `number`, `contact`, `entry`, `info`, `data`) as a deletion request. Keep the existing term‑overlap matching logic for locating the target record(s) and the fallback to the most recent record by the same speaker.
5. **Bump `SCHEMA_VERSION`** to `2` (already done in step 1) and update any module‑level docstrings that mention the schema version.
6. **Run the test suite** (`pytest tests/`) to confirm all existing tests still pass. No test should need modification because the revised logic retains all current behaviours for the test inputs.
---
### Expected Tradeoff
- **U**: increases significantly – unrelated queries no longer suffer from false `no_memory`; utility on authorised queries returns.
- **A**: no expected change – the RBAC join is untouched.
- **F**: slight improvement – more deletion requests are caught by the broader matcher, and the query‑sensitive tombstone detection ensures `no_memory` fires only when appropriate, not for all queries.
- **Cost**: adds one extra table and a lookup in `retrieve()`; minor runtime overhead, which is acceptable given the correctness gain. The `deletion_terms` table is never exposed, so it does not introduce a new leak vector.
```json
{
  "schema_ddl": "PRAGMA journal_mode = WAL;\nPRAGMA foreign_keys = ON;\n\nCREATE TABLE IF NOT EXISTS principals (\n    principal_id  TEXT PRIMARY KEY,\n    role          TEXT NOT NULL,\n    display_name  TEXT\n);\n\nCREATE TABLE IF NOT EXISTS relationships (\n    rel_id        INTEGER PRIMARY KEY AUTOINCREMENT,\n    rel_type      TEXT NOT NULL,\n    subject_id    TEXT NOT NULL,\n    patient_id    TEXT NOT NULL,\n    scope         TEXT DEFAULT ''\n);\n\nCREATE TABLE IF NOT EXISTS records (\n    record_id     TEXT PRIMARY KEY,\n    episode_id    TEXT NOT NULL,\n    turn_id       TEXT NOT NULL,\n    author_id     TEXT NOT NULL,\n    author_role   TEXT NOT NULL,\n    patient_id    TEXT NOT NULL,\n    kind          TEXT NOT NULL,\n    sensitivity   TEXT NOT NULL DEFAULT 'routine',\n    body          TEXT NOT NULL,\n    ciphertext    BLOB,\n    key_id        TEXT,\n    ts            TEXT NOT NULL,\n    seq           INTEGER NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS role_grants (\n    role          TEXT NOT NULL,\n    sensitivity   TEXT NOT NULL,\n    requires_rel  TEXT NOT NULL DEFAULT '',\n    PRIMARY KEY (role, sensitivity)\n);\n\nCREATE TABLE IF NOT EXISTS tombstones (\n    record_id     TEXT PRIMARY KEY,\n    deleted_at    TEXT NOT NULL,\n    requested_by  TEXT NOT NULL,\n    reason        TEXT DEFAULT '',\n    shredded      INTEGER NOT NULL DEFAULT 0\n);\n\nCREATE TABLE IF NOT EXISTS crypto_keys (\n    key_id        TEXT PRIMARY KEY,\n    key_material  BLOB,\n    shredded_at   TEXT\n);\n\nCREATE TABLE IF NOT EXISTS access_log (\n    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,\n    checkpoint_id TEXT,\n    requester_id  TEXT NOT NULL,\n    record_id     TEXT NOT NULL,\n    decision      TEXT NOT NULL,\n    ts            TEXT NOT NULL\n);\n\nCREATE TABLE IF NOT EXISTS deletion_terms (\n    record_id  TEXT NOT NULL REFERENCES records(record_id),\n    term       TEXT NOT NULL,\n    patient_id TEXT NOT NULL,\n    PRIMARY KEY (record_id, term)\n);\n\nCREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq);\nCREATE INDEX IF NOT EXISTS idx_records_patient     ON records(patient_id, sensitivity);\nCREATE INDEX IF NOT EXISTS idx_rel_subject         ON relationships(subject_id, patient_id);\nCREATE INDEX IF NOT EXISTS idx_deletion_terms_patient_term ON deletion_terms(patient_id, term);",
  "migration_sql": "CREATE TABLE IF NOT EXISTS deletion_terms (\n    record_id  TEXT NOT NULL REFERENCES records(record_id),\n    term       TEXT NOT NULL,\n    patient_id TEXT NOT NULL,\n    PRIMARY KEY (record_id, term)\n);\n\nCREATE INDEX IF NOT EXISTS idx_deletion_terms_patient_term\n    ON deletion_terms(patient_id, term);\n\n-- Bump schema version in code to 2",
  "retrieval_loop": "1. Compute query terms via _distinctive_terms(query).\n2. Query deletion_terms for the given patient_id and those terms to obtain matched_tombstones (set of record_id).\n3. Scan all records for the patient (same LEFT JOIN as before).\n   - If tombstone present: add to denied_tombstone ONLY if record_id ∈ matched_tombstones; otherwise skip.\n   - If no tombstone: apply RBAC gates (role_grants join, relationship check) and include allowed evidence as before.\n4. Log every decision (allow/deny_*) to access_log.",
  "forgetting_mechanism": "Tombstone handling is now query‑sensitive. When a record is tombstoned, we extract its distinctive terms (from body or decrypted ciphertext) and store them in the deletion_terms table, then cryptographically shred the key (for confidential) or erase the body (for plaintext). Later retrieval queries compare the query terms against deletion_terms: a match sets touched_deleted and yields no_memory, while non‑matching tombstones are completely ignored. This prevents false no_memory on unrelated queries and allows accurate recognition of actual deletion references.",
  "work_order": [
    "Update memory_system/schema.sql: add deletion_terms table and idx_deletion_terms_patient_term; bump SCHEMA_VERSION to 2 in store.py.",
    "Modify MemoryStore.tombstone() in store.py: before shredding/erasing, extract plaintext, compute _distinctive_terms, insert into deletion_terms (record_id, term, patient_id) with INSERT OR IGNORE.",
    "Modify MemoryStore.retrieve() in store.py: after computing query terms, query deletion_terms to build matched_tombstones set; in the main loop, only add a tombstone to denied_tombstone if its record_id is in matched_tombstones; otherwise skip (continue).",
    "Improve _honor_deletion_request() in store.py: broaden deletion detection to accept any message containing at least one of {delete, remove, erase, forget, purge, scrub, wipe} AND at least one of {memory, record, note, number, contact, entry, info, data}; retain term‑overlap matching for target selection and fallback to most recent record by same speaker.",
    "Run pytest tests/ to confirm all existing tests pass; adjust code if needed (no test changes expected)."
  ],
  "targets_metric": "U",
  "expected_tradeoff": "U increases substantially by eliminating false no_memory on unrelated queries. A and F are expected to remain unchanged or slightly improve because RBAC filtering is untouched and deletion becomes more precise (better detection, query‑sensitive tombstones). The extra deletion_terms table adds negligible storage/retrieval overhead and is never exposed to callers, so it does not introduce a leak vector."
}
```