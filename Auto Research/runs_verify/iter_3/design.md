## Iteration 3 Design
### Root cause
The dominant failure is `U=0.1429`, not A or F. The previous iteration's critique correctly identifies the cause: in `MemoryStore.retrieve()` the `_is_responsive` filter is applied **before** `top_k`, so any gate-cleared record that does not share at least one lexical token with the query is silently dropped. The record therefore never reaches `decision.allowed`, and the answerer cannot surface its content. The fix is to stop using `_is_responsive` as a hard gate on the allow path, and instead treat relevance as a **rank**, not a keep/drop predicate.
### Proposed change
1. Remove the `responsive` list from `retrieve()`.
2. Sort the **entire** gate-cleared candidate list by `(_relevance_score, seq)` descending.
3. Take the first `top_k` candidates as `decision.allowed`.
4. Keep `_is_responsive` only for tallying `deny_rbac` and `deny_scope`, so irrelevant denials do not flood the refusal rationale.
5. Raise the effective `top_k` from 8 to 20 as a defensive margin. The observed checkpoint `allowed` counts were 2–8, but if unrelated cleared records exist, a larger room cuts the chance that a low-scoring but relevant record is truncated.
This change does not alter the three gates (tombstone → RBAC grant → relationship/scope), and it does not change the deletion or shredding mechanism. It only changes **which already-authorized records are released**, so A and F are unaffected; all released records have already passed the RBAC/relationship join, and tombstones still win over any allow decision.
### Schema
No schema change is required for this iteration. The existing SQLite schema already stores all records in one table with a `patient_id` index and a tombstone table. The retrieval scan is already a per-patient scan; the wastage was purely in the Python post-filter.
### Retrieval loop
After the change, the retrieval loop is:
1. Scan `records` filtered by `patient_id` and `as_of_seq`.
2. For each row:
   - If tombstoned → record in `denied_tombstone`, log `deny_tombstone`, skip.
   - If no `role_grants` row → if responsive, record in `denied_rbac`, log `deny_rbac`, skip.
   - If relationship/scope fails → if responsive, record in `denied_scope`, log `deny_scope`, skip.
   - Otherwise decrypt if needed and append `(score, seq, row, plaintext)` to `candidates`.
3. Sort `candidates` by `(relevance_score, seq)` descending.
4. Take the first `top_k` and append to `decision.allowed`.
5. Log each `allow`.
This guarantees that any record that cleared all three gates can be released, even if it shares no distinctive terms with the query. Relevance is used to order evidence, not to exclude it.
### Active forgetting
No changes. Deletion remains:
- Insert a tombstone row.
- If the record is confidential, null out its `crypto_keys.key_material` and set `shredded_at`; this makes the ciphertext unrecoverable.
- If the record is plaintext, overwrite `records.body` with an empty string.
- In `retrieve()`, tombstones are checked first, so deleted content always yields `no_memory`, even for an otherwise authorized requester.
### Tradeoff
**Target:** U.
**Cost:** Slightly noisier allowed evidence because low-relevance cleared records may now be returned. This may reduce answer precision when an LLM is wired in, but it does not increase A (all released records are already authorized) and does not increase F (deletion logic is untouched). For the observed failing checkpoints, the benefit is large: the exact required content is in the store, and this change surfaces it.
### Developer work order
```json
{
  "schema_ddl": "PRAGMA journal_mode = WAL;\nPRAGMA foreign_keys = ON;\nCREATE TABLE IF NOT EXISTS principals (principal_id TEXT PRIMARY KEY, role TEXT NOT NULL, display_name TEXT);\nCREATE TABLE IF NOT EXISTS relationships (rel_id INTEGER PRIMARY KEY AUTOINCREMENT, rel_type TEXT NOT NULL, subject_id TEXT NOT NULL, patient_id TEXT NOT NULL, scope TEXT DEFAULT '');\nCREATE TABLE IF NOT EXISTS records (record_id TEXT PRIMARY KEY, episode_id TEXT NOT NULL, turn_id TEXT NOT NULL, author_id TEXT NOT NULL, author_role TEXT NOT NULL, patient_id TEXT NOT NULL, kind TEXT NOT NULL, sensitivity TEXT NOT NULL DEFAULT 'routine', body TEXT NOT NULL, ciphertext BLOB, key_id TEXT, ts TEXT NOT NULL, seq INTEGER NOT NULL);\nCREATE TABLE IF NOT EXISTS role_grants (role TEXT NOT NULL, sensitivity TEXT NOT NULL, requires_rel TEXT NOT NULL DEFAULT '', PRIMARY KEY (role, sensitivity));\nCREATE TABLE IF NOT EXISTS tombstones (record_id TEXT PRIMARY KEY, deleted_at TEXT NOT NULL, requested_by TEXT NOT NULL, reason TEXT DEFAULT '', shredded INTEGER NOT NULL DEFAULT 0);\nCREATE TABLE IF NOT EXISTS crypto_keys (key_id TEXT PRIMARY KEY, key_material BLOB, shredded_at TEXT);\nCREATE TABLE IF NOT EXISTS access_log (log_id INTEGER PRIMARY KEY AUTOINCREMENT, checkpoint_id TEXT, requester_id TEXT NOT NULL, record_id TEXT NOT NULL, decision TEXT NOT NULL, ts TEXT NOT NULL);\nCREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq);\nCREATE INDEX IF NOT EXISTS idx_records_patient ON records(patient_id, sensitivity);\nCREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq);\nCREATE INDEX IF NOT EXISTS idx_rel_subject ON relationships(subject_id, patient_id);",
  "migration_sql": "",
  "retrieval_loop": "In MemoryStore.retrieve(): scan rows already filtered by patient_id/as_of_seq; for each row evaluate tombstones first, then RBAC grant, then relationship/scope, appending denied reasons only when _is_responsive() is true; for cleared rows decrypt if needed and append (score, seq, row, plaintext) to candidates. Do NOT apply a binary _is_responsive filter to the allowed path. Sort candidates by (score, seq) descending and take top_k. Log allow for each released row. This releases any gate-cleared record even if it shares no lexical tokens with the query, while preserving gate order and deny tallies.",
  "forgetting_mechanism": "Tombstone row inserted with record_id; for confidential rows, crypto_keys.key_material is set to NULL and shredded_at is set, and records.key_id is NULLed; for plaintext rows, records.body is overwritten with ''. Retrieve checks tombstones before RBAC and treats a wiped/encrypted body as deleted, so no path can reconstruct deleted content. No changes proposed.",
  "work_order": [
    "In memory_system/store.py, modify MemoryStore.retrieve(): delete the `responsive` list construction and the `responsive[:top_k]` loop; instead sort the full `candidates` list by (score, seq) descending and iterate over `candidates[:top_k]` to build `decision.allowed`. Keep `_is_responsive` for the deny_rbac/deny_scope tallies only.",
    "In the same file, update the retrieve() docstring to explain that relevance scores rank, not gate, the allow path.",
    "In memory_system/agent.py, change the default `top_k` from 8 to 20 (both in __init__ and, if set, in the retrieve() call). Do not change the public signature.",
    "Run the existing test suite: `pytest tests/ -q` and fix any regression; all current tests must pass unchanged.",
    "Run the Developer local smoke runner (`_smoke_runner.py`) to confirm the agent still produces valid actions.",
    "Run the eval runner on a representative sample of the previously failing checkpoints (or the full shard if feasible) and confirm allowed/evidence now contains the missing content (e.g. apixaban, estrogen patch, appointment times)."
  ],
  "targets_metric": "U (utility_accuracy)",
  "expected_tradeoff": "Increases U by releasing already-authorized records even when they share no lexical tokens with the query. May modestly reduce answer precision for LLM-wired runs due to noisier evidence, but A and F are unaffected because all released records still clear the RBAC/relationship gates and deletion/tombstone logic is unchanged."
}
```