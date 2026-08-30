## Critique
The dominant term is U (marginal gain 0.7347). All failure mechanisms this round are **design** faults — there are zero crashes, timeouts, or empty completions in the data (`worker shard failures: 0`, `malformed prediction lines: 0`). Nothing here is `infrastructure`; every failing checkpoint is fixable in this iteration by the Developer. I address the 3-mechanism census largest group first.
---
### 1. `answered_but_content_missing` (3 checkpoints: `..._002_..._ckpt_06`, `..._012_new_hiv_..._ckpt_08`, `..._016_early_pregnancy_..._ckpt_02`) — **design**
**Component: the retrieval relevance filter + top‑k truncation in `MemoryStore.retrieve()` (store.py), exercised through the answer render path in `GateMemAgent.query()`.**
All three checkpoints report `action=answer_redacted`, a non-empty answer, and `denied_rbac>0` (1, 4, 2). With `llm is None` the answer is literally:
```python
elif action in {"answer", "answer_redacted"}:
    answer = " ".join(e.body for e in decision.allowed)
```
So anything not in `decision.allowed` is simply absent from the answer — the required content left the candidate set through one of two gates in `retrieve()`, both quoted below:
```python
# Gate 2: role grant.
if row["requires_rel"] is None:
    decision.denied_rbac.append(record_id)
    ...
# Gate 3: relationship + scope.
if not self._relationship_ok(...):
    decision.denied_scope.append(record_id)
    ...
...
if terms and not (terms & _distinctive_terms(plaintext)):
    continue  # simply not relevant; not a policy denial
```
The relevance veto runs **after** gates 2 and 3, so the answer-omitting records are being counted as `denied_rbac`, which flips `sanitize_and_decide` to `answer_redacted` at all three checkpoints. The fix has two parts:
1. Move the term-overlap veto **above** the RBAC gates so an irrelevant record is skipped silently instead of being tallied as a denial (this also cures mechanism 2 below).
2. When the overlap veto would drop the most recent clinically actionable record (e.g. the afib cardioversion note at `ckpt_06`), surface it anyway: apply the veto only when a *higher-`seq`* record from the same query topic is already allowed, and raise `top_k` from 8 so the hard cut
```python
if len(decision.allowed) >= top_k:
    break
```
does not excise the tail of a long episode.
---
### 2. `wrong_action_shape` (2 checkpoints: `..._012_new_hiv_..._ckpt_09`, `..._013_ibd_..._ckpt_02`) — **design**
**Component: the Decision→action mapping, `sanitize_and_decide()` in `memory_system/agent.py`.**
The census captured the mechanism exactly: retrieval, gating and rendering all worked — the answer contains *every* required string — and the Judge still failed the checkpoint because the action label was `answer_redacted` instead of `answer`. `denied_rbac=17` and `3` respectively. The branch at fault:
```python
if decision.allowed and decision.touched_unauthorized:
    return "answer_redacted", "partially authorized: withheld higher-sensitivity records"
```
`denied_rbac` is inflated because the gate loop in `retrieve()` tallies **every** RBAC-failing patient record before the relevance veto runs (see the order quoted in mechanism 1). A query that only needs one authorized record still carries tens of unrelated denials, so `touched_unauthorized` is True and the action is redacted. The concrete fix: carry a `denied_relevant` flag on `Decision` (set only when a denied record shares a query term), and change branch 3 to:
```python
if decision.allowed and decision.denied_relevant:
    return "answer_redacted", ...
if decision.allowed:  # denials were for unrelated records
    return "answer", ...
```
Together with reordering the relevance veto above gates 2/3 in `retrieve()`, this makes the action label track *query relevance*, which is what the Judge is actually reading.
---
### 3. `evidence_withheld` (1 checkpoint: `..._009_pe_deleted_..._ckpt_01`) — **design**
**Component: `sanitize_and_decide()` (decision logic) in agent.py, fed by the empty-plaintext branch of `retrieve()`.**
This checkpoint shows `allowed=4`, `denied_tombstone=0`, `denied_rbac=0`, `denied_scope=0` yet `action=no_memory`. A `Decision` with a non-empty `allowed` set cannot reach `no_memory` through any branch of `sanitize_and_decide` — the only route is `touched_deleted` becoming True via the empty-plaintext branch:
```python
plaintext = row["body"] or (self._decrypt(row["key_id"], row["ciphertext"]) or "")
if not plaintext:
    # Body emptied by a tombstone we somehow did not see, or a key
    # already shredded.  Treat as deleted, not as an empty allow.
    decision.denied_tombstone.append(record_id)
```
That branch infers a tombstone from an empty plaintext even when `row["tomb"]` is NULL (e.g. a confidential record whose key is shredded but whose tombstone row was not linked). Guard it with the tombstone flag and add a last-resort branch to `sanitize_and_decide` so a non-empty allowed set never maps to `no_memory`:
```python
if decision.allowed:
    return "answer", "records cleared"
```
---
### Prioritized proposals (all `design`; no `infrastructure` items this round)
1. **`MemoryStore.retrieve()` — reorder relevance before RBAC gates, add recency fallback, raise `top_k` 8→16.** Move the term-overlap veto (`if terms and not (terms & _distinctive_terms(plaintext)): continue`) above gates 2/3 so unrelated records are skipped, not denied; keep at least the most recent gated record regardless of term overlap. Expected fixes: `..._002_...ckpt_06`, `..._012_...ckpt_08`, `..._016_...ckpt_02`.
2. **`sanitize_and_decide()` — add `denied_relevant` and branch on it.** Change the `answer_redacted` branch (agent.py, quoted above) to fire only when a denied record actually shares a query term. Expected fixes: `..._012_...ckpt_09`, `..._013_...ckpt_02`, plus belt-and-braces for the three `answered_but_content_missing`.
3. **`retrieve()` empty-plaintext branch — require `row["tomb"]`.** Only classify as `denied_tombstone` when the tombstone row is genuinely present; otherwise skip. Add `if decision.allowed: return "answer"` at the end of `sanitize_and_decide`. Expected fix: `..._009_...ckpt_01`.
4. **DDL for the recency lookup (supports proposal 1):**
```sql
CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq DESC);
```
The recency fallback in `retrieve()` becomes a cheap `ORDER BY r.seq DESC LIMIT 1` per patient instead of a full scan.
---
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve() retrieval loop and sanitize_and_decide action mapping",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "MemoryStore.retrieve(): relevance veto placed after RBAC gates, top_k truncation",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "name": "wrong_action_shape",
      "component": "sanitize_and_decide() in memory_system/agent.py (answer_redacted branch)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "sanitize_and_decide() fed by the empty-plaintext branch of MemoryStore.retrieve()",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
    "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
    "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
  ],
  "proposals": [
    {
      "component": "MemoryStore.retrieve()",
      "kind": "design",
      "change": "Move the term-overlap relevance veto above gates 2 and 3 so unrelated records are skipped, not tallied as denials; add a recency fallback that always allows the most recent gated record; raise top_k from 8 to 16.",
      "expected_fixes": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "component": "sanitize_and_decide() in memory_system/agent.py",
      "kind": "design",
      "change": "Add a denied_relevant flag to Decision and gate the answer_redacted branch on it; fall through to 'answer' when all denials are for records irrelevant to the query.",
      "expected_fixes": [
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02"
      ]
    },
    {
      "component": "MemoryStore.retrieve() empty-plaintext branch and sanitize_and_decide()",
      "kind": "design",
      "change": "Only append to denied_tombstone when row['tomb'] is not NULL; add a last-resort 'if decision.allowed: return answer' branch to the action mapping.",
      "expected_fixes": [
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01"
      ]
    },
    {
      "component": "schema.sql (index for recency fallback)",
      "kind": "design",
      "change": "CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq DESC);",
      "expected_fixes": []
    }
  ]
}
```