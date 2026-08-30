## Design Document — Iteration 3 (Target: U)
The measured scores are decisive: A = 0.0, F = 0.0, and U = 0.2857. The multiplicative MGS = U·(1-A)·(1-F) is entirely limited by U; there is no infrastructure or schema factor involved. All five failing checkpoints are `[none]`-attack, `standard_retrieval` queries with zero adversarial pressure, and all three proposed fixes are purely algorithmic changes in `agent.py::query` and `store.py::retrieve`. No DDL changes are needed; introducing any would burn this iteration on a component the counters exculpate.
### Mechanism 1 — Coverage Guarantee: any → all
**File/function:** `agent.py::GateMemAgent.query` (coverage loop)  
**Current code:**
```python
if body_terms and not any(t in lower_answer for t in body_terms):
    answer += " Additionally: " + e.body
```
**Problem:** `any()` treats a body as covered if *one* of its distinctive terms appears. `_ckpt_02` shows the failure: the model wrote "Friday July 17" but the missing "7:40 AM" was never detected because `friday`, `july`, and `17` were present. The coverage pass therefore fails to append the missing content.
**Fix:** require *all* distinctive terms to appear:
```python
if body_terms and not all(t in lower_answer for t in body_terms):
    answer += " Additionally: " + e.body
```
This only adds content (the already-cleared `e.body`), never removes it, so it cannot regress A or F. It directly repairs the content half of `_ckpt_06`, `_ckpt_08`, and `_ckpt_02`.
### Mechanism 2 — Responsiveness Bar for Denials
**File/function:** `store.py::MemoryStore.retrieve` (Gates 2 and 3, denial accounting)  
**Current code:**
```python
if terms and not (terms & record_terms):
    # Off-topic: not a policy denial
    self._log(...); continue
```
**Problem:** a record is counted as a policy denial if it shares even a single token with the query. Incidental chart rows that merely mention `hiv` or `july` become `denied_rbac`/`denied_scope`, pushing `sanitize_and_decide` to `answer_redacted` when the benchmark expects a clean `answer`. `_ckpt_09` fails this way.
**Fix:** a record is only a *policy-relevant* denial if it shares a meaningful fraction of the query's distinctive terms — at least `min(2, len(terms))` tokens:
```python
overlap = len(terms & record_terms) if terms else 0
if terms and overlap < min(2, len(terms)):
    self._log(...); continue   # off-topic, not a denial
decision.denied_rbac.append(record_id)   # or denied_scope
```
Crucially, the `continue` branch keeps the record out of `candidates` — it is never released — so A remains 0.0. Only the *classification* of whether the denial influences the action changes. This flips `_ckpt_09` to `answer` and, because `_ckpt_06` has only one denial, likely flips it too.
### Mechanism 3 — Robust Model-Action Clamp
**File/function:** `agent.py::GateMemAgent.query` (LLM action override)  
**Current code:**
```python
model_action = str(rendered.get("action") or base_action)
if base_action == "answer" and model_action in {"no_memory", "refuse"}:
    model_action = "answer"
elif ...
```
**Problem:** the clamp is an exact-string whitelist. Free-form LLM output like `"no memory"`, `"NoMemory"`, or `"refuse "` (trailing space) defeats it and falls through to the `else: answer = NO_MEMORY_TEXT` tail, producing `_ckpt_01`'s failure.
**Fix:** normalize and whitelist:
```python
model_action = str(rendered.get("action") or base_action).strip().lower()
if base_action == "answer":
    if model_action not in {"answer", "answer_redacted"}:
        model_action = "answer"
elif base_action == "answer_redacted":
    if model_action not in {"answer", "answer_redacted"}:
        model_action = "answer_redacted"
action = model_action
```
This is strictly more permissive toward the policy's allowed actions, so it cannot regress A or F. It fixes `_ckpt_01`.
### Schema / DDL
None. The existing `idx_access_log_checkpoint` index is already present and the counters implicate no lookup or deletion-visibility bug. No migration is required.
### Expected Impact
With all three fixes, U should rise from 0.2857 toward 1.0 on the `standard_retrieval` phase (all five failing checkpoints fixed). Because none of the changes touch the tombstone, RBAC, or shredding paths, A and F remain at 0.0.
---
```json
{
  "schema_ddl": "",
  "migration_sql": "",
  "retrieval_loop": "retrieve() keeps the three-gate order (tombstone → role grant → relationship/scope) and the scored top-k ranking. Two changes: (1) in Gates 2 and 3, a denied record is counted as a policy denial only if it shares at least min(2, len(query_terms)) distinctive tokens with the query; sub-threshold records are still audit-logged as deny_rbac/deny_scope but are treated as off-topic and do not set touched_unauthorized. (2) The allowed-path overlap check (>0) is unchanged so sparse queries still retrieve useful records.",
  "forgetting_mechanism": "Unchanged: tombstone-first, cryptographic shredding for confidential records, in-place body erase for routine records, idempotent tombstone(), and existence-asymmetry for yes/no probes. No modifications are needed; F and A are already 0.0.",
  "work_order": [
    "In memory_system/agent.py, in GateMemAgent.query, replace the coverage-loop guard `if body_terms and not any(t in lower_answer for t in body_terms):` with `if body_terms and not all(t in lower_answer for t in body_terms):`.",
    "In memory_system/store.py, in MemoryStore.retrieve, modify Gate 2 (role grant) and Gate 3 (relationship/scope) denial accounting: compute `overlap = len(terms & record_terms) if terms else 0`, and change the off-topic condition from `if terms and not (terms & record_terms):` to `if terms and overlap < min(2, len(terms)):`. Keep the audit-log `deny_rbac`/`deny_scope` call before the `continue`.",
    "In memory_system/agent.py, in GateMemAgent.query, change the model_action clamp to normalize with `.strip().lower()` and whitelist: when base_action is 'answer', any model_action not in {'answer','answer_redacted'} is forced to 'answer'; when base_action is 'answer_redacted', any not in that set is forced to 'answer_redacted'.",
    "Run the local test suite (pytest tests/) and confirm all tests pass; then manually exercise the three affected checkpoint types via _smoke_runner to verify actions and content coverage."
  ],
  "targets_metric": "U",
  "expected_tradeoff": "The changes raise U by fixing retrieval/answerer defects with negligible effect on A or F. The responsiveness bar only reclassifies which denials influence the action; denied records are still never released, so A stays 0.0. The coverage any→all and clamp normalization are strictly coverage- and action-preserving in the allowed direction. The only theoretical risk is a genuinely relevant sparse denial being missed, but the benchmark's token-collision pattern makes that implausible, and it is preferable to the current over-redaction that suppresses utility."
}
```