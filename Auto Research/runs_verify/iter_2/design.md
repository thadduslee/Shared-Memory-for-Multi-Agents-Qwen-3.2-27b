## Design Document — Iteration 2: Fix U–0.0 by fixing the relevance drop and deny‑tally responsiveness
### Background
The previous iteration produced a fully instrumented store and retrieval path but scored `U = 0.0`, `A = 0.0`, `F = 0.0`. The Critic’s analysis is decisive: all seven failures are **design faults** in `MemoryStore.retrieve()` and the action‑mapping layer — not infrastructure, not prompt, not schema. Two specific defects were identified:
1. **Hard relevance drop** (`if terms and score <= 0.0: continue`) silently discards gate‑cleared records that share no distinctive token with the query. This causes `answered_but_content_missing` in six checkpoints even though the record passed tombstones, RBAC, and scope gates. The drop happens **before** `candidates[:top_k]`, so no top‑k truncation is involved.
2. **Deny‑list over‑counting** — `denied_rbac`/`denied_scope` are populated for *all* records that fail gates 2/3, regardless of whether the record is responsive to the query. That makes `sanitize_and_decide` label a fully authorized answer as `answer_redacted` when an irrelevant record is denied (checkpoint 006), and also mis‑labels other checkpoints.
### Design goals
- **Target metric:** `U` (utility accuracy).
- **Acceptable tradeoff:** No increase in `A` or `F`. The changes only widen the set of *allowed* records to include previously gate‑cleared but low‑relevance records, and they only shrink the *denied* tallies for records that could never have contributed to the answer. Neither change widens the set of records a requester is authorized to see.
### Proposed changes
All changes are in `memory_system/store.py`; no schema changes are required (see *Schema* below). The `sanitize_and_decide` logic remains unchanged except that it will now receive a more accurate picture of what is actually responsive.
#### 1. Remove the hard relevance drop in `retrieve()`
Replace the conditional `continue` with an unconditional append to `candidates`. The existing sort `(score DESC, seq DESC)` then ranks 0‑scorers last, and `candidates[:top_k]` fills the tail by recency — so a gate‑cleared record that shares no distinctive query term still reaches `decision.allowed` if there is room.
**Before**
```python
score = _relevance_score(terms, _distinctive_terms(plaintext))
if terms and score <= 0.0:
    continue  # simply not relevant; not a policy denial
candidates.append((score, int(row["seq"]), row, plaintext))
```
**After**
```python
score = _relevance_score(terms, _distinctive_terms(plaintext))
candidates.append((score, int(row["seq"]), row, plaintext))
```
This is the fix for the six `answered_but_content_missing` checkpoints.
#### 2. Make the denied‑tally lists responsive‑aware
In gates 2 and 3, before appending a record to `decision.denied_rbac` or `decision.denied_scope`, check whether the record shares at least one distinctive term with the query. If it does not, the record is irrelevant to the current query and its denial should not contribute to `touched_unauthorized`, which is what `sanitize_and_decide` uses to decide between `answer_redacted`, `refuse` and `answer`.
Because the plaintext may be encrypted (confidential records), we must decrypt it to compute its distinctive terms. This is done lazily — only when we need to decide whether to append to a denied list. (The allowed path already decrypts later when building `allowed` evidence.)
**Gate 2 (role grant)**
```python
if row["requires_rel"] is None:
    if _is_responsive(row, terms):
        decision.denied_rbac.append(record_id)
        self._log(checkpoint_id, requester_id, record_id, "deny_rbac")
    continue
```
**Gate 3 (relationship/scope)**
```python
if not self._relationship_ok(...):
    if _is_responsive(row, terms):
        decision.denied_scope.append(record_id)
        self._log(checkpoint_id, requester_id, record_id, "deny_scope")
    continue
```
Where `_is_responsive` is a small helper that decrypts the body (if needed) and computes `_distinctive_terms(plaintext) & terms` non‑empty. If the query has no distinctive terms (e.g., stopword‑only), the record is considered non‑responsive, and we skip the denial.
This fixes the `wrong_action_shape` failure in checkpoint 006 and may also normalize the labels on 002/012/013/016.
### Schema
No schema changes are necessary. The existing indexes — particularly `idx_records_patient_seq (patient_id, seq)` — already cover the ordered per‑patient scan used by `retrieve()`. The changes are purely in the retrieval logic and do not alter any query‑path acceleration needs. We therefore emit **no** DDL in `migration_sql`.
### Active forgetting
The existing active‑forgetting mechanisms remain untouched and pass all tests:
- Tombstone table for existence tracking.
- For plaintext (routine) records, the body is overwritten to `''` on tombstone.
- For confidential records, the encryption key is destroyed (`crypto_keys.key_material` set to `NULL`) and the `records.key_id` is cleared, making the ciphertext unrecoverable.
- Shredding is idempotent and audited via `tombstones.shredded`.
These changes do not affect forgetting; they only adjust how `retrieve()` computes relevance and deny tallies. The `F` metric is preserved.
### Expected impact
- **`U`**: Fixed for the six content‑missing checkpoints (allowing the correct record to surface) and the one action‑shape checkpoint (now `answer` instead of `answer_redacted`). We expect `U` to rise substantially.
- **`A`**: Unchanged — the RBAC gates are still evaluated identically; we only include records that already passed them, and we do not change which records are *authorized*, only whether their denial is tallied when irrelevant.
- **`F`**: Unchanged — no deletion or shredding logic is modified.
Thus `MGS = U * (1 - A) * (1 - F)` should increase toward the 0.85 target because the dominant term `U` improves without degrading the other factors.
### Work order for the Developer
The Developer should implement the two changes exactly as specified, then run the test suite and smoke test. The full shard evaluation will then confirm the metric improvement.
---
```json
{
  "schema_ddl": "",
  "migration_sql": "",
  "retrieval_loop": "In MemoryStore.retrieve(), remove the hard relevance drop and make deny-tally appends responsive-aware. Concretely:\n  - Replace `score = _relevance_score(...); if terms and score <= 0.0: continue` with an unconditional append to candidates.\n  - In gate 2 (requires_rel is None) and gate 3 (relationship not ok), before appending to denied_rbac/denied_scope, compute the record's plaintext (decrypt if necessary) and check whether it shares at least one distinctive term with the query. If it does not, skip the append and do not log a denial. The sort by (score, seq) and caps at top_k remain unchanged.",
  "forgetting_mechanism": "No changes. Existing tombstones, in-place body erasure for routine records, and key shredding for confidential records continue to guarantee active forgetting and pass the 'deleted content is not retrievable' tests. The modifications do not touch any deletion path.",
  "work_order": [
    "Edit memory_system/store.py: in retrieve(), remove the line `if terms and score <= 0.0: continue` and make the following line unconditional. (Alternative: keep the score calculation but always append to candidates.)",
    "Edit memory_system/store.py: add a helper `_is_responsive(row, terms)` that returns True if the record's plaintext shares at least one distinctive term with the query. For confidential records, decrypt using the stored key_id/ciphertext (key_id is available); for routine records use body. Handle empty plaintext (e.g., body already wiped) as non-responsive.",
    "In gate 2 (the `if row['requires_rel'] is None:` branch), replace `decision.denied_rbac.append(record_id)` and `self._log(... 'deny_rbac')` with a conditional call to `_is_responsive`; only if responsive, append and log.",
    "In gate 3 (the `if not self._relationship_ok(...)` branch), do the same conditional append and log for `denied_scope`.",
    "Run `pytest tests/` to confirm all existing tests still pass (test_rbac.py, test_forgetting.py, test_deletion_scope.py).",
    "Run `python _smoke_runner.py` (or equivalent smoke test) to ensure no crashes on sample checkpoints.",
    "Optionally run a partial shard evaluation on the previously failing checkpoint IDs to verify the fixes before full eval."
  ],
  "targets_metric": "U",
  "expected_tradeoff": "U increases without affecting A or F. The relevance change only adds gate-cleared records that were previously dropped; the deny-tally change only removes irrelevant denials from the counters that influence action labeling. Neither widens the authorized set, so A remains stable. Deletion logic is untouched, so F remains stable."
}
```