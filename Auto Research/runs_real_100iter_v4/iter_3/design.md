# Iteration 3 Design — Wire the answer-bearing denial count into the action label
## 1. Reading the table
The measured table shows a dead tie: iteration 2 moved **no term at all** (U, A, F all +0.0000). The critique attributes this not to a wrong hypothesis but to an unfinished one: iteration 2 added `Decision.query_answer_denials`, computed it in `store.py`, and never consumed it in `agent.py`. Two checkpoints (005/ckpt_04, 010/ckpt_04) still return `answer_redacted` because `sanitize_and_decide` keys branch 3 on the raw length of the denial lists instead of on the number of *answer-bearing* denials.
The last three iterations (dating back before this summary) have all targeted U and all failed to move it. But each failed for a *different, nameable* reason — and the previous critique, plus the full source above, shows the current dead code is the one unambiguously fixable cause. Reverting iteration 2's change wholesale would be wrong: the field and its computation are sound, and the widening it introduced is harmless. The correct edit is to **wire the field to the one place that decides the action label**.
## 2. Target and cost
- **Targets metric:** U (utility accuracy). A and F are untouched by this change.
- **Cost to A/F:** None. We are changing which denial *shapes the label*, not which records are released. `allowed` still contains only records that passed tombstone/RBAC/scope gates; denied lists stay populated and audited; deletion behaviour is untouched.
- **Measured expectation:** 2 of the 11 failing checkpoints should flip from `answer_redacted` to `answer`, holding everything else constant. If checkpoint 005/010 respond as the critique describes, U should rise from 0.3889 toward ~0.46. A and F should remain 0.0000.
## 3. The defect, precisely
In `memory_system/store.py` a retrieved `Decision` always carries an integer `query_answer_denials` (initialized to `0`; incremented only when an RBAC/scope denial shares a real content term with the query). A denial that shares *only* a phone-digit run or the synthetic contact tag leaves it at `0`.
In `memory_system/agent.py`, `sanitize_and_decide` never reads that field. It computes:
```python
withheld = len(decision.denied_rbac) + len(decision.denied_scope)
```
and treats any non-zero `withheld` as `answer_redacted`. So an incidental denial — a logistics/contact record the requester isn't cleared to see but that has nothing answer-bearing to do with the query — still downgrades a complete, correct `answer`. The benchmark scores the action *label* (`utility_correct = action_correct and include_ok`), so that downgrade is a hard zero for the whole answer.
### The fix (one edit in `agent.py`)
Use the answer-bearing denial count when it was computed, falling back to the legacy raw-length contract only for hand-constructed `Decision`s in tests (`None` means "not computed — treat every denial as answer-bearing").
```python
# memory_system/agent.py — inside sanitize_and_decide
qad = decision.query_answer_denials
withheld = qad if qad is not None else len(decision.denied_rbac) + len(decision.denied_scope)
```
This preserves every pinned unit test:
- `test_allowed_with_a_responsive_denial_is_answer_redacted`, `test_a_scope_denial_counts_the_same_as_an_rbac_denial`, and `test_partial_authorization_is_redacted` construct `Decision`s *without* `query_answer_denials`, so the fallback path fires and keeps their existing behaviour and rationale strings intact.
- A store-produced `Decision` with a genuinely answer-bearing denial still has `query_answer_denials >= 1`, so `answer_redacted` still fires when content was actually withheld.
- A store-produced `Decision` whose only denial is incidental has `query_answer_denials == 0`, so a complete answer is correctly labelled `answer`.
No schema, index, tombstone, rank, or gate change is involved. This is aimed at the single most verifiable regression cause the critique identified: **the field that was built to fix the label is dead code until this wiring exists.** Because the two nameable failing checkpoints (005/ckpt_04, 010/ckpt_04) are the exact behaviour the field was added for, this is also the smallest hypothesis that the measurements can falsify: either the action flips to `answer` on those two, or the critique's mechanism description is wrong.
## 4. Schema / migration
No storage layout change improves this metric. The retrieval cost of the query-path index is not the measured bottleneck, and any schema rewrite risks an out-of-turn build failure. Because every prior episode burned its full budget, the DDL and migration are empty and the entire iteration reduces to one Python edit plus the standard verify steps.
## 5. Retrieval loop and forgetting mechanism (unchanged — stated so the Developer doesn't "restate" work that already exists)
- **Retrieval loop** — `MemoryStore.retrieve()` already runs gate 0 responsiveness before the three gate pipeline, enforces `top_k` as a hard cap only on `allowed`, and logs every denial via `access_log`. No change.
- **Forgetting mechanism** — `tombstone()` purges `record_terms`, overwrites plaintext bodies in place, and cryptographically shreds keyed ciphertext in one transaction. No change.
## 6. Work order (kept deliberately tiny)
The previous episode exhausted 60/60 turns on an 8-step order and scored a failed build. This order has **one source edit + two verify steps**. Any order that names only existing behaviour as work, or that bundles the five-point content-missing diagnosis into this round, risks another ceiling hit.
## JSON
```json
{
  "schema_ddl": [],
  "migration_sql": [],
  "retrieval_loop": "Unchanged. MemoryStore.retrieve() already gates responsiveness before tombstone/RBAC/scope, ranks by (content_overlap, structural, seq), and hard-caps allowed at top_k. The fault under repair is not in retrieval; it is in how the resulting Decision is mapped to the action label.",
  "forgetting_mechanism": "Unchanged. tombstone() purges record_terms and overwrites plaintext in the same transaction; confidential rows are crypto-shredded by NULLing key_material and zeroing ciphertext.",
  "work_order": [
    "Edit memory_system/agent.py, function sanitize_and_decide, branch 3. Replace `withheld = len(decision.denied_rbac) + len(decision.denied_scope)` with `qad = decision.query_answer_denials; withheld = qad if qad is not None else len(decision.denied_rbac) + len(decision.denied_scope)`. Keep the existing branch structure and rationale strings exactly as they are; the fallback path guarantees the three pinned `answer_redacted` tests still pass because hand-built Decisions leave `query_answer_denials` as None.",
    "Run compile_check and confirm the package imports cleanly.",
    "Run the local test suite (`run_tests`) and confirm all deterministic tests in test_action_shape.py, test_rbac.py, test_forgetting.py, test_deletion_scope.py, and test_utility_focus.py still pass."
  ],
  "targets_metric": "U",
  "expected_tradeoff": "A and F are mathematically untouched: the change never alters which records are released, denied, logged, or shredded, so it cannot affect leakage rates. U is expected to rise by the two checkpoints (005/ckpt_04, 010/ckpt_04) whose answers already contain every required string but were scored zero because the label said answer_redacted. If the mechanism description is wrong, U will stay flat; this gives the loop a falsifiable read on whether the dead-field hypothesis is correct before any more expensive ranking change is attempted.",
  "previous_critique_summary": "The prior critique found U alone losing, with A and F both perfect, and blamed the tie not on a regression but on a dead fix: Decision.query_answer_denials is computed by store.py yet never read by sanitize_and_decide in agent.py, which still keys answer_redacted on the raw length of the RBAC/scope denial lists. It asked for that field to be consumed so incidental contact/digit-only denials stop demoting otherwise complete answers, while real content denials still force redaction."
}
```