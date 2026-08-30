## Design Document for Iteration 1
### Current State and Metric Analysis
- **U = 0.0**: No utility is generated. The current retrieval loop returns nothing for authorized queries. The primary cause is the relevance gate: a record is skipped unless its tokens intersect the query tokens. This overly strict filter eliminates all results when queries are paraphrased or sparse.
- **A = 1.0**: Total privacy leakage. This is likely a secondary effect of the same relevance filter: because no records are returned as `allowed`, the agent’s `answer_redacted` logic may misclassify, or the RBAC checks themselves are still exposing something. More likely, the baseline fails on benchmark attack patterns not covered by local tests.
- **F = 1.0**: Deletion leakage. Possible causes: deletion requests not matching the intended records, or the tombstone not being honored consistently across query paths.
### Proposed Changes
1. **Retrieval loop: replace binary relevance filter with scored ranking.**  
   Stop discarding records that have zero token overlap. Instead, after RBAC gates pass, compute a relevance score for every authorized record and return the `top_k` by score. This directly improves U by surfacing relevant records even when they don’t share exact tokens.
2. **Deletion matching: use similarity-based selection.**  
   Replace the fixed `overlap >= 2` threshold with Jaccard similarity between the deletion request and prior records. This makes deletion requests more precise and reduces F by ensuring the correct tombstone is created.
3. **Schema: add a composite index on `(patient_id, seq)`.**  
   Speeds up the common query path `WHERE patient_id = ? AND seq <= ? ORDER BY seq DESC`, which the new retrieval loop will use to fetch all candidates.
### Expected Trade‑off
- **U increases** because more authorized records are now considered and ranked.
- **A risk**: Relaxing the relevance filter could return records not exactly on‑topic. To mitigate, we keep RBAC gates strict and rank by relevance so only the most relevant records are chosen. This may slightly raise A if a non‑relevant record sneaks in, but the impact is expected to be minimal.
- **F decreases** with better deletion matching; no trade‑off expected.
### Schema Changes
#### Initial DDL (revision of baseline)
Include the new index:
```sql
-- (baseline tables unchanged)
CREATE TABLE IF NOT EXISTS principals (...);
...
CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq);
```
#### Migration SQL (if baseline already applied)
```sql
CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq);
-- Rationale: speeds up patient-scoped as-of retrieval.
```
### Retrieval Loop (pseudocode)
```
def retrieve(...):
    # 1. Fetch ALL records for patient (and as_of_seq) with tombstones and role_grants joined.
    sql = """SELECT r.*, t.record_id AS tomb, g.requires_rel
             FROM records r
             LEFT JOIN tombstones t ON r.record_id = t.record_id
             LEFT JOIN role_grants g ON g.role = ? AND g.sensitivity = r.sensitivity
             WHERE r.patient_id = ? [AND r.seq <= ?]
             ORDER BY r.seq DESC"""
    # 2. Iterate rows, applying gates as before (tombstone, RBAC, relationship/scope).
    # 3. For rows that pass all gates and have non‑empty plaintext,
    #    compute relevance score: score = len(query_terms ∩ body_terms).
    # 4. Collect (score, record) pairs; sort descending by score, then by seq.
    # 5. Take top_k as `decision.allowed`.
    # 6. Log each decision (allow/deny) and return Decision.
```
### Forgetting Mechanism
- Keep tombstone + crypto‑shredding baseline.
- Improve `_honor_deletion_request` to compute Jaccard similarity between the request’s distinctive terms and each prior record’s terms. If similarity ≥ 0.5, tombstone that record. Fallback to the most recent record by the same author if nothing meets the threshold.
### Developer Work Order
The Developer inherits the existing codebase. The following steps are deltas:
1. **`memory_system/store.py` — `retrieve()` rewrite**: Replace the loop that skips non‑overlapping records with a scored‑ranking approach over all authorized rows. Keep the existing RBAC gate order (tombstone → role grant → relationship/scope). After collecting candidates, sort by relevance score descending, tiebreak by `seq` descending, and select `top_k`. Do not change the method signature or return types.
2. **`memory_system/store.py` — `_honor_deletion_request()`**: Replace the `overlap >= 2` condition with a Jaccard similarity calculation. Compute `similarity = len(intersection) / len(union)` between the request’s distinctive terms and each prior record’s terms. Tombstone records with similarity ≥ 0.5. Keep the fallback to the most recent record from the same speaker.
3. **`memory_system/store.py` — SCHEMA_VERSION**: Bump from 1 to 2 and apply migration on initialization (use `apply_migration` or execute the CREATE INDEX directly in `initialize`).
4. **`memory_system/schema.sql`**: Add `CREATE INDEX idx_records_patient_seq` and increment version comment.
5. **Run tests**: Ensure all existing unit tests pass. Add a sanity test for the new retrieval scoring logic if needed.
### Target Metric
U (utility accuracy) — the primary goal is to increase the number of correct answers for authorized queries.
### Trade‑off
Minor risk of increased privacy leakage (A) if relevance scores accidentally rank non‑relevant records higher; however, strict RBAC gates remain unchanged, and ranking prioritizes the most relevant authorized content.
```json
{
  "schema_ddl": "PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON; CREATE TABLE IF NOT EXISTS principals (principal_id TEXT PRIMARY KEY, role TEXT NOT NULL, display_name TEXT); CREATE TABLE IF NOT EXISTS relationships (rel_id INTEGER PRIMARY KEY AUTOINCREMENT, rel_type TEXT NOT NULL, subject_id TEXT NOT NULL, patient_id TEXT NOT NULL, scope TEXT DEFAULT ''); CREATE TABLE IF NOT EXISTS records (record_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, turn_id TEXT NOT NULL, author_id TEXT NOT NULL, author_role TEXT NOT NULL, patient_id TEXT NOT NULL, kind TEXT NOT NULL, sensitivity TEXT NOT NULL DEFAULT 'routine', body TEXT NOT NULL, ciphertext BLOB, key_id TEXT, ts TEXT NOT NULL, seq INTEGER NOT NULL); CREATE TABLE IF NOT EXISTS role_grants (role TEXT NOT NULL, sensitivity TEXT NOT NULL, requires_rel TEXT NOT NULL DEFAULT '', PRIMARY KEY (role, sensitivity)); CREATE TABLE IF NOT EXISTS tombstones (record_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL, requested_by TEXT NOT NULL, reason TEXT DEFAULT '', shredded INTEGER NOT NULL DEFAULT 0); CREATE TABLE IF NOT EXISTS crypto_keys (key_id TEXT PRIMARY KEY, key_material BLOB, shredded_at TEXT); CREATE TABLE IF NOT EXISTS access_log (log_id INTEGER PRIMARY KEY AUTOINCREMENT, checkpoint_id TEXT, requester_id TEXT NOT NULL, record_id TEXT NOT NULL, decision TEXT NOT NULL, ts TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq); CREATE INDEX IF NOT EXISTS idx_records_patient_sens ON records(patient_id, sensitivity); CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq); CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id, patient_id);",
  "migration_sql": "CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq); -- rational: accelerated as-of retrieval for patient-scoped queries",
  "retrieval_loop": "Fetch all records for a patient (optionally bounded by as_of_seq) with LEFT JOINs to tombstones and role_grants. Apply gates: (1) tombstone, (2) role grant exists, (3) relationship/scope. For survivors, decrypt if needed, compute relevance score = len(set(query_terms) ∩ set(body_terms)). Collect (score, record) pairs, sort by score descending then seq descending, take top_k, and return as Decision.allowed. Log each decision (allow/deny) to access_log.",
  "forgetting_mechanism": "Tombstones and crypto-shredding remain. Deletion request matching now uses Jaccard similarity between the request's distinctive terms and prior record terms; if similarity >= 0.5, tombstone the record. If nothing meets the threshold, fallback to the most recent record from the same speaker. Idempotent tombstone and shred as before.",
  "work_order": [
    "Modify `MemoryStore.retrieve` in `memory_system/store.py` to use scored ranking: replace the 'skip if no token overlap' logic with a scoring phase that ranks all RBAC-passed rows by overlapping token count and returns the top_k.",
    "Modify `_honor_deletion_request` in the same file to use Jaccard similarity instead of a fixed overlap count; tombstone records whose similarity to the request exceeds 0.5.",
    "Apply schema migration: add `CREATE INDEX idx_records_patient_seq ON records(patient_id, seq)` and bump `SCHEMA_VERSION` to 2 in `memory_system/store.py`.",
    "Update `memory_system/schema.sql` to include the new index and updated version comment.",
    "Run the existing test suite; ensure all tests pass. If any new tests are required, add them in `tests/` without modifying existing ones."
  ],
  "targets_metric": "U (utility accuracy)",
  "expected_tradeoff": "Slight risk of A increase due to loosened relevance ranking, but mitigated by strict RBAC and score-based top_k selection; F decreases with better deletion matching."
}
```