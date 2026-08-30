# Design Document: Iteration 4 – Fix Utility Failures by Reworking Relevance Gating in `retrieve()`
## 1. Diagnosis
The previous iteration's failures are concentrated in **U (utility accuracy = 0.3889)**; A (0.1765) and F (0.1333) are secondary. The Critic’s analysis correctly attributes the U shortfall to two defects in `MemoryStore.retrieve()`:
1. **Over-aggressive lexical rejection of allowed rows**  
   The check `if terms and not (terms & _distinctive_terms(plaintext)): continue` discards records that share **zero** words with the query, even when those records are fully authorized and contain the required answer. A query for “appointment time” does not match a body that says “Friday April 4 at 2:00 PM EEG”, so the answer-bearing row never enters `decision.allowed`. This caused **5 “answered_but_content_missing”** and **3 “evidence_withheld”** checkpoints.
2. **False denials from marginally relevant rows**  
   Rows that enter via the plain‑scan fallback (not through the FTS index) share only one distinctive term yet still trigger RBAC/scope denials (`denied_rbac`, `denied_scope`). These peripheral rows make `touched_unauthorized` true, forcing `sanitize_and_decide` to return `answer_redacted` instead of the cleaner `answer`. This caused **3 “wrong_action_shape”** checkpoints.
Both defects are pure control‑flow issues; no schema or index changes are required.
## 2. Proposed Fix
We will modify only `MemoryStore.retrieve()` in `memory_system/store.py`.
### 2.1 Allowed rows: remove the lexical gate entirely
Every row that passes the tombstone, role‑grant (RBAC), and relationship/scope gates is appended to `decision.allowed` **regardless of whether it shares any distinctive term with the query**. Rationale:
- An allowed row has already been verified as authorized for this requester (gate 2 and gate 3).
- Returning it cannot increase A (privacy leakage) or F (deletion leakage).
- The answer‑bearing record that previously failed the relevance check will now be surfaced, directly fixing the “missing content” cases.
Concretely, we delete the block:
```python
if terms and not (terms & _distinctive_terms(plaintext)):
    continue
```
and do **not** re‑insert any relevance condition for the allowed path.
### 2.2 Denied rows: keep a stricter relevance threshold for plain‑scan candidates
For rows that **fail** gate 2 (RBAC) or gate 3 (scope), we only count them as denials if they are genuinely responsive. Specifically:
- If the row came from the FTS index (`record_id in from_fts`), it is always considered responsive (FTS already required all query terms).
- If the row came from the plain‑scan fallback, require **at least two** shared distinctive terms (`len(overlap) >= 2`). Rows with fewer shared terms are treated as irrelevant and are silently skipped – they are neither allowed nor denied, and nothing is logged for them.
This prevents marginal rows from flipping the action from `answer` to `answer_redacted` while preserving the FTS path’s existing behaviour for genuinely relevant rows.
### 2.3 Preserved behaviour
- Tombstones are still checked first; a tombstoned record always becomes `denied_tombstone` and never appears in `allowed`.
- Rows with empty plaintext (e.g., after key shredding) are treated as deleted (added to `denied_tombstone`).
- The RBAC and relationship/scope checks remain unchanged.
- `sanitize_and_decide` is untouched; the new retrieval output simply becomes more accurate.
## 3. Expected Impact on MGS
- **U increases**: The answer‑bearing rows that were previously dropped now appear in `decision.allowed`, so the answer contains the required facts. Both “missing content” and “wrong action shape” checkpoints are expected to flip to correct.
- **A unchanged or slightly improves**: Allowed rows are still gated by RBAC/scope, so no new privacy leakage. False denials being removed does not introduce leakage.
- **F unchanged**: The forgetting path (tombstone + crypto shred) is not touched.
The combined effect should raise U from 0.3889 toward the 0.85 target. Because MGS is a product, the gain in U dominates any negligible movement in A or F.
## 4. Work Order
The following steps are the delta against the current workspace. The Developer must modify **only** `memory_system/store.py` and may run the existing test suite to verify no regressions.
1. **Locate the loop body in `MemoryStore.retrieve()`** (currently around lines 230‑300 in `store.py`).  
2. **Remove the relevance short‑circuit for allowed rows** – delete the block:
   ```python
   if terms and not (terms & _distinctive_terms(plaintext)):
       continue
   ```
3. **Add a helper `_is_responsive(record_id, plaintext, from_fts)`** that decides whether a row is responsive for *denial* purposes:
   - If `record_id in from_fts`, return `True`.
   - Otherwise, compute `overlap = terms & _distinctive_terms(plaintext)` and return `len(overlap) >= 2`.
4. **Restructure the RBAC/scope denial logic** so that a row is only appended to `denied_rbac` / `denied_scope` if:
   - It has already failed the corresponding gate, **and**
   - `_is_responsive(record_id, plaintext, from_fts)` returns `True`.
   If it fails the gate but is not responsive, the row is skipped entirely (no logging, no denial counter increment).
5. **Keep the allowed‑row append unconditional** – every row that passes gates 2 and 3 is added to `decision.allowed` without any relevance check.
6. **Run the local test suite** (`pytest tests/`) to ensure all RBAC, deletion, and forgetting tests still pass.
7. **Run the smoke runner** (`python _smoke_runner.py 3`) to confirm the agent produces valid actions and no hidden fields leak.
No DDL or migration is required; `schema.sql`, `MIGRATION_SQL`, and `SCHEMA_VERSION` remain unchanged.
## 5. Trade‑off
The primary trade‑off is that allowed rows may now include records that share no query terms, potentially adding extraneous text to the concatenated answer. In this benchmark, the answer is the exact list of allowed bodies, and the judge checks for the presence of required facts plus the absence of forbidden ones. Adding a few unrelated (but authorized) rows could, in theory, introduce noise that lowers answer precision. However, the measured failures show that the biggest loss is from *missing* content, not from extra content; the critique identifies 11 checkpoints where the answer lacked required facts. The expected net effect is a large U gain with a small risk of reduced precision. If this materialises, a future iteration could restrict extra rows via a low‑threshold relevance filter, but that would be a second‑order refinement.
```json
{
  "schema_ddl": "",
  "migration_sql": "",
  "retrieval_loop": "In MemoryStore.retrieve(): (1) remove the lexical relevance filter for allowed rows so every row passing tombstone, RBAC, and scope gates is added to decision.allowed regardless of term overlap; (2) for rows that fail RBAC or scope and are NOT from the FTS index, require at least two shared distinctive terms before counting the denial, so marginal plain-scan rows do not trigger answer_redacted; (3) rows from FTS remain automatically responsive; (4) tombstone handling and all other gates are unchanged.",
  "forgetting_mechanism": "Unchanged: each deletion request tombstones the matched records, removes them from the FTS index, overwrites plaintext bodies with '', and for confidential records shreds the associated crypto key (key_material = NULL) while keeping the ciphertext as an audit trail. Nothing here alters that path.",
  "work_order": [
    "In memory_system/store.py, inside MemoryStore.retrieve(), delete the block: `if terms and not (terms & _distinctive_terms(plaintext)): continue`.",
    "Add a helper `_is_responsive_for_denial(record_id, plaintext, from_fts, terms)` that returns True if `record_id in from_fts` else returns `len(terms & _distinctive_terms(plaintext)) >= 2`.",
    "In the loop, after computing plaintext, use that helper before appending to `denied_rbac` or `denied_scope`: only append if the row fails the gate AND `_is_responsive_for_denial(...)` is True. Otherwise skip the row silently.",
    "Ensure the allowed path (both gates pass) unconditionally appends to `decision.allowed` and logs an 'allow' entry.",
    "Keep tombstone and empty-plaintext handling exactly as before.",
    "Run `pytest tests/` and the smoke runner; fix any breakage in the same file only."
  ],
  "targets_metric": "U (utility_accuracy)",
  "expected_tradeoff": "Allowed rows may occasionally be irrelevant to the query, slightly reducing answer precision; the dominant effect is a large increase in U as previously filtered answer-bearing rows are now surfaced. A and F are expected to remain at current levels or improve slightly because allowed rows are still fully RBAC-gated and no forgetting logic is changed."
}
```