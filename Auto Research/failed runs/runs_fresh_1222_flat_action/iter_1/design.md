 # Design Document for Iteration 3: Utility-Enhanced RBAC-Protected Memory Store

## 1. Overview

The baseline (iteration 2) already provides a solid foundation: RBAC filtering via `role_grants` × `relationships`, tombstone-based active forgetting, per-record AES-256-GCM encryption with external KMS, and an audit log. However, the current `retrieve()` uses a simplistic text match (likely `LIKE` on `body`), which underperforms on utility (`U`) for natural-language queries. The active-forgetting gate, while observant, can over-trigger on generic terms, causing false `no_memory` refusals.

This iteration focuses on **improving `U` without inflating `A` or `F`**. The approach:

- Add a **SQLite FTS5 full-text index** over record bodies and author roles to enable relevance-ranked retrieval (BM25 scoring). This boosts utility for legitimate queries.
- Refine the **tombstone trigger** to use *distinctive terms* extracted at deletion time, matched against the query via FTS, so only queries that specifically reference the deleted content are refused.
- Keep all RBAC and tombstone logic in SQL – no new storage engines, no external services.

## 2. Prior Art

We rely on established patterns:

- **RBAC with inheritance**: recursive role expansion via `role_hierarchy` (already present) mirrors common enterprise access control.
- **Tombstones**: soft-delete markers preserve the ability to distinguish "never existed" from "deleted", essential for preventing `post_delete_recovery` attacks.
- **Cryptographic shredding**: per-record keys stored outside the DB (KMS) allow irreversible content destruction.
- **FTS5 for SQLite**: native full-text search with BM25 ranking is a standard, high-performance approach for text retrieval in SQLite-only environments.

We explicitly avoid vector databases or document stores, honoring the storage constraint.

## 3. Proposed Changes

### 3.1 Schema Additions

- **`records_fts`**: an FTS5 virtual table indexing `record_id`, `body`, `author_role`, `patient_id`, and `terms`.
- **Triggers** to keep FTS synchronized with `records`:
  - `records_ai`: insert into FTS after new record.
  - `records_ad`: delete from FTS when a record is tombstoned or hard‑deleted.
  - `records_au`: update FTS when body, author_role, or patient_id changes (covers decryption upgrades).

- A **partial index** on `tombstones(terms)` to speed up the tombstone-term matching query.

### 3.2 Retrieval Enhancement

The `retrieve()` loop becomes:

1. **Candidate selection** via FTS `MATCH` on the query (using `record_id` to join back to `records`).
2. **RBAC filter** (unchanged): apply role grants, relationship existence, scope checks.
3. **Tombstone gate** (refined): check if any candidate’s term-set from `tombstones` appears in the query using the FTS index, but only if the record is not allowed by RBAC. If a deleted record is *allowed* but also *deleted*, we still refuse (as before).
4. **Ranking**: use `bm25()` score, take top_k.

The key improvement: FTS provides both relevance (for `U`) and a precise mechanism for the tombstone gate (for `F`).

### 3.3 Forgetting Mechanism

- On deletion, store a **term-set** in `tombstones.terms` – the top distinctive tokens from the body (e.g., rare drugs, lab codes, specific names).
- At retrieve time, run a lightweight FTS query against `tombstones` to detect if the user’s query contains any of those terms. Only if a tombstone term matches **and** the record is deducible from the query do we return `no_memory`. This avoids false refusals on unrelated queries.

- **Crypto shredding** remains unchanged: `destroy_key()` in KMS permanently orphans the ciphertext.

## 4. Migration SQL

The migration adds the FTS table, triggers, and the tombstone index. All statements are explicit `CREATE`/`CREATE INDEX` with rationale comments.

```sql
-- Migration from iteration 2 to iteration 3

-- FTS5 table for rapid text retrieval and relevance scoring
CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(
    record_id UNINDEXED,        -- row identifier, not searchable
    body,                       -- searchable content
    author_role,                -- searchable author information
    patient_id UNINDEXED,       -- not needed in search, but kept for joins
    terms UNINDEXED,            -- tombstone terms, not searched directly (used for gate)
    content=''                  -- external content table? We'll keep it internal for simplicity
);

-- Trigger to sync new records into FTS
CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
    INSERT INTO records_fts(record_id, body, author_role, patient_id, terms)
    VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, '');
END;

-- Trigger to remove records from FTS when tombstoned or hard-deleted
CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
    DELETE FROM records_fts WHERE record_id = OLD.record_id;
END;

-- Trigger to update FTS when record fields change (e.g., body update, decryption)
CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
    DELETE FROM records_fts WHERE record_id = OLD.record_id;
    INSERT INTO records_fts(record_id, body, author_role, patient_id, terms)
    VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, '');
END;

-- Partial index to accelerate tombstone term matching
-- Rationale: retrieval queries often filter by unshredded tombstones; this index speeds up that lookup.
CREATE INDEX IF NOT EXISTS idx_tombstones_terms_unshredded ON tombstones(terms) WHERE shredded = 0;
```

**Rationale for each statement** (as required):

- `CREATE VIRTUAL TABLE records_fts` — accelerates the main retrieval path (`WHERE records.record_id IN (SELECT record_id FROM records_fts WHERE records_fts MATCH ?)`), improving `U`.
- `CREATE TRIGGER records_ai/ad/au` — keeps FTS consistent with `records`, preventing stale search results that would hurt `U` and cause `A`/`F` errors.
- `CREATE INDEX idx_tombstones_terms_unshredded` — speeds up the tombstone-match subquery used in the forgetting gate, reducing the chance of false refusals (`F`).

## 5. Retrieval Loop (Pseudocode)

```
def retrieve(requester_id, requester_role, patient_id, query, as_of_seq, checkpoint_id, top_k):
    # Expand role via hierarchy (recursive CTE)
    roles = get_role_closure(requester_role)

    # Step 1: Find candidate record_ids using FTS
    fts_matches = SELECT record_id, bm25(records_fts) AS score
                  FROM records_fts
                  WHERE records_fts MATCH :query
                  ORDER BY score DESC LIMIT :top_k * 5  # over-fetch for filtering

    # Step 2: Join back to records, apply as-of and RBAC filters
    candidates = SELECT r.* FROM records r
                 JOIN fts_matches f ON r.record_id = f.record_id
                 WHERE r.patient_id = :patient_id
                   AND r.seq <= :as_of_seq
                   AND r.is_deleted = 0
                   AND (r.sensitivity, r.author_role) IN (SELECT ... FROM role_grants WHERE role IN :roles)
                   AND EXISTS (SELECT 1 FROM relationships rel
                               WHERE rel.subject_id = :requester_id
                                 AND rel.patient_id = r.patient_id
                                 AND rel.rel_type IN (allowed types))
                   AND (scope check, if applicable)
                 ORDER BY f.score DESC LIMIT :top_k

    # Step 3: Tombstone gate
    # For each candidate, check if any tombstone for that patient matches the query terms
    tombstone_hits = SELECT t.record_id FROM tombstones t
                      WHERE t.record_id IN (SELECT record_id FROM candidates)
                        AND t.shredded = 0
                        AND t.terms MATCH :query  -- FTS on tombstone terms
                        AND t.requested_by != :requester_id  # don't deny if requester initiated deletion?

    # Step 4: Build Decision
    allowed = [c for c in candidates if c.record_id not in tombstone_hits]
    denied_tombstone = [c for c in candidates if c.record_id in tombstone_hits]
    denied_rbac = any query that hit RBAC filter but not allowed
    ...

    return Decision(allowed, denied_rbac, denied_scope, denied_tombstone)
```

This loop explicitly satisfies the two MUST invariants:
- **No leakage without RBAC decision**: every returned record passes the `role_grants` × `relationships` join; there is no raw accessor.
- **Deletion observable but not readable**: tombstoned records are excluded from `allowed`, and if any tombstone matches the query, the whole answer is `no_memory` (preventing reconstruction).

## 6. Forgetting Mechanism

- On `delete_record(record_id, requester)`:  
  1. Extract `terms` from the body (using a simple tokenizer; stop-word removal, low-frequency terms).  
  2. Insert into `tombstones(record_id, deleted_at, requested_by, reason, terms)`.  
  3. Mark `records.is_deleted = 1` (or hard-delete the row? We keep the row for RBAC history but mark it deleted).  
  4. Call `kms.destroy_key(key_id)` — ciphertext becomes unreadable.

- On `query`: the tombstone gate uses FTS over `tombstones.terms` to see if the query contains any distinctive terms from deleted records. If a match occurs, the entire response is `no_memory` (unless the user has a special right to see deletions? Not per current spec). This prevents both confirming the existence *and* reconstructing content.

- **Crypto shredding** is the ultimate backstop: even if a malicious actor gains DB access, the key is gone.

## 7. Work Order

1. **Apply migration** (add FTS table, triggers, index).
2. **Update `store.py`**:
   - Implement `retrieve` using FTS for candidate selection and BM25 ranking.
   - Implement tombstone-term matching with FTS.
   - Adjust `delete_record` to store `terms` and destroy key.
3. **Update `kms.py`** if needed (no change planned).
4. **Write unit tests**:
   - Utility: queries that should return multiple records now rank correctly.
   - RBAC: authorized queries never see unauthorized records.
   - Forgetting: queries about deleted content return `no_memory`; unrelated queries are unaffected.
5. **Run full benchmark** to measure U, A, F.
6. **Iterate** if thresholds not met.

## 8. Expected Impact

- **U** should rise because FTS accurately retrieves relevant content beyond simple `LIKE`.
- **A** remains low: RBAC is unchanged; FTS only filters candidates, never bypasses grants.
- **F** stays low: tombstone matching is precise, reducing false refusals.

Potential tradeoff: FTS index increases storage and write overhead; sparse queries may over-match and return more candidates, but top_k truncation keeps response bounded.

The target metric for this iteration: **MGS = 0.60** (U=0.75, A=0.02, F=0.02). We expect to at least double the baseline MGS.

```json
{
  "schema_ddl": "CREATE TABLE IF NOT EXISTS principals (principal_id TEXT PRIMARY KEY, role TEXT NOT NULL, display_name TEXT); CREATE TABLE IF NOT EXISTS relationships (rel_id INTEGER PRIMARY KEY AUTOINCREMENT, rel_type TEXT NOT NULL, subject_id TEXT NOT NULL, patient_id TEXT NOT NULL, scope TEXT DEFAULT ''); CREATE TABLE IF NOT EXISTS records (record_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, turn_id TEXT NOT NULL, author_id TEXT NOT NULL, author_role TEXT NOT NULL, patient_id TEXT NOT NULL, kind TEXT NOT NULL, sensitivity TEXT NOT NULL DEFAULT 'routine', body TEXT NOT NULL, ciphertext BLOB, key_id TEXT, ts TEXT NOT NULL, seq INTEGER NOT NULL, is_deleted INTEGER NOT NULL DEFAULT 0); CREATE TABLE IF NOT EXISTS role_grants (role TEXT NOT NULL, sensitivity TEXT NOT NULL, requires_rel TEXT NOT NULL DEFAULT '', PRIMARY KEY (role, sensitivity)); CREATE TABLE IF NOT EXISTS role_hierarchy (child_role TEXT NOT NULL, parent_role TEXT NOT NULL, PRIMARY KEY (child_role, parent_role)); CREATE TABLE IF NOT EXISTS tombstones (record_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL, requested_by TEXT NOT NULL, reason TEXT DEFAULT '', shredded INTEGER NOT NULL DEFAULT 0, terms TEXT DEFAULT ''); CREATE TABLE IF NOT EXISTS crypto_keys (key_id TEXT PRIMARY KEY, shredded_at TEXT); CREATE TABLE IF NOT EXISTS access_log (log_id INTEGER PRIMARY KEY AUTOINCREMENT, checkpoint_id TEXT, requester_id TEXT NOT NULL, record_id TEXT NOT NULL, decision TEXT NOT NULL, ts TEXT NOT NULL); CREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq); CREATE INDEX IF NOT EXISTS idx_records_patient ON records(patient_id, sensitivity); CREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id, patient_id); CREATE INDEX IF NOT EXISTS idx_rel_type_subject ON relationships(rel_type, subject_id, patient_id); CREATE INDEX IF NOT EXISTS idx_role_grants_lookup ON role_grants(role, sensitivity); CREATE INDEX IF NOT EXISTS idx_records_episode_seq_sens ON records(episode_id, seq, sensitivity); CREATE INDEX IF NOT EXISTS idx_rel_patient_subject ON relationships(patient_id, subject_id); CREATE INDEX IF NOT EXISTS idx_tombstones_unshredded ON tombstones(record_id) WHERE shredded = 0; CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(record_id UNINDEXED, body, author_role, patient_id UNINDEXED, terms UNINDEXED); CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN INSERT INTO records_fts(record_id, body, author_role, patient_id, terms) VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, ''); END; CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN DELETE FROM records_fts WHERE record_id = OLD.record_id; END; CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN DELETE FROM records_fts WHERE record_id = OLD.record_id; INSERT INTO records_fts(record_id, body, author_role, patient_id, terms) VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, ''); END; CREATE INDEX IF NOT EXISTS idx_tombstones_terms_unshredded ON tombstones(terms) WHERE shredded = 0;",
  "migration_sql": "CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(record_id UNINDEXED, body, author_role, patient_id UNINDEXED, terms UNINDEXED);\nCREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN\n    INSERT INTO records_fts(record_id, body, author_role, patient_id, terms)\n    VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, '');\nEND;\nCREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN\n    DELETE FROM records_fts WHERE record_id = OLD.record_id;\nEND;\nCREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN\n    DELETE FROM records_fts WHERE record_id = OLD.record_id;\n    INSERT INTO records_fts(record_id, body, author_role, patient_id, terms)\n    VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, '');\nEND;\nCREATE INDEX IF NOT EXISTS idx_tombstones_terms_unshredded ON tombstones(terms) WHERE shredded = 0;\n-- Each statement has explicit rationale in the design document.",
  "retrieval_loop": "1. Expand requester role to closure via RECURSIVE CTE on role_hierarchy.\n2. Use FTS5 MATCH on records_fts with the query to fetch candidate record_ids ordered by bm25.\n3. Join back to records, filter by patient_id, as_of_seq, is_deleted=0.\n4. Apply RBAC: only include records whose sensitivity is allowed for the role closure AND requester has a relationship to the patient (role_grants × relationships).\n5. Check tombstones: for the candidate set, run an FTS query on tombstones.terms to detect if the query contains any distinctive terms from deleted records. If any deleted record matches, mark it as denied_tombstone.\n6. Build decision: allowed = candidates not in tombstone set; denied_rbac/scope = candidates that failed RBAC; denied_tombstone = candidates flagged.\n7. Return Decision(allowed, denied_rbac, denied_scope, denied_tombstone) which drives sanitize_and_decide().",
  "forgetting_mechanism": "On delete_record:\n  - Extract top distinctive terms from body (e.g., rare drug names, lab codes, specific identifiers).\n  - Insert into tombstones(record_id, deleted_at, requested_by, reason, terms).\n  - Mark records.is_deleted=1.\n  - Call kms.destroy_key(key_id) to permanently orphan ciphertext.\nOn retrieve:\n  - After RBAC filtering, run a lightweight FTS query against tombstones.terms (with unused term weighting) to see if any of the user's query tokens match a tombstone's stored terms.\n  - If a match exists, that candidate is flagged as deleted; if any candidate is flagged, the entire response becomes 'no_memory' via sanitize_and_decide (unless the requester is the one who requested deletion? not per current policy).\nThis preserves the invariant: deletion is observable (we refuse on queries that reference it) but the content is unrecoverable (crypto-shredded and RBAC-denied).",
  "work_order": [
    "Apply migration SQL to add records_fts table, sync triggers, and tombstone terms index.",
    "Update store.py retrieve() method to use FTS5 MATCH for candidate selection and BM25 ranking, then apply RBAC and tombstone gate as described.",
    "Update delete_record() to extract and store tombstone terms and destroy the KMS key.",
    "Add unit tests for: (a) utility - queries retrieve correct relevant records; (b) RBAC - unauthorized records never returned; (c) forgetting - deleted content triggers no_memory only when queried specifically, unrelated queries unaffected.",
    "Run full benchmark suite and collect U, A, F metrics.",
    "If MGS < 0.60, iterate on retrieval ranking weights or tombstone term extraction heuristics."
  ],
  "targets_metric": "MGS >= 0.60, with target U=0.75, A<=0.02, F<=0.02",
  "expected_tradeoff": "Adding FTS increases storage and write overhead, but improves retrieval relevance (U). The tombstone FTS matching reduces false refusals (lower F) at the cost of a slightly more complex query. No RBAC weakening occurs, so A stays low. Overall MGS improves due to higher U and lower F, with minor performance cost acceptable for a single-file SQLite store."
}
```