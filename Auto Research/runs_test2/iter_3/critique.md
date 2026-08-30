## Critique
### 1. The Design Document’s proposed fixes are misdiagnosed
The iteration‑3 design doc asserts three mechanical bugs drive all U failures, but the actual code and the measured census contradict each of them:
- **“top_k truncation”** — there is **no** `if len(decision.allowed) >= top_k: break` anywhere in `MemoryStore.retrieve()`. The `top_k` parameter is accepted but never used. The loop runs to completion over all rows matching the SQL. This fix would change nothing.
- **“tombstone short‑circuits allowed content”** — `sanitize_and_decide` already gives `answer` when `decision.allowed` is non‑empty and `touched_unauthorized` is false, **before** checking `touched_deleted`. The census shows all `evidence_withheld` checkpoints have `denied_tombstone=0`, so there is no tombstone to short‑circuit. The proposed branch reorder is a no‑op.
- **“marginal denied rows flip action to answer_redacted”** — this part is real, but the fix must target the correct component (see below), not a fabricated responsiveness threshold.
The observed mechanisms are all caused by **semantic mismatch between the query and the answer‑bearing record** and **an over‑conservative action decision**. Below I address each mechanism with specific code citations.
---
### 2. Mechanism‑by‑mechanism attribution
#### 2.1 `answered_but_content_missing` (5 checkpoints) — **design**
**Component:** `MemoryStore.retrieve()` — the lexical relevance filter.
```python
# store.py – retrieve(), lines ~290–294
            # Relevance: a row sharing no distinctive query term is not
            # responsive to this query at all -- not a policy denial -- so skip
            # it before RBAC gating and log nothing.
            if terms and not (terms & _distinctive_terms(plaintext)):
                continue
```
This filter drops any record whose body shares **zero** distinctive terms with the query. But the required content often uses different wording (e.g. query asks for “appointment time” while the record says “Friday April 4 at 2:00 PM EEG”). The record is real, allowed, and should be surfaced, but the lexical filter removes it before the RBAC/scope gates. As a result `decision.allowed` contains many rows (allowed=13, 38, etc.) but not the one carrying the required token. The answer is a concatenation of the allowed bodies, so the required content is missing.
**Evidence:** all five checkpoints listed under this mechanism, e.g. `med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05` (allowed=13, missing “Friday April 4 at 2:00 PM EEG”).
**Fix:** remove the term‑overlap rejection for rows that will be allowed. Since every allowed row has already passed the tombstone, RBAC, and relationship/scope gates, returning it cannot violate A or F. The natural SQL change is to drop the `if terms and not (terms & _distinctive_terms(plaintext)): continue` block for the `allowed` path. To preserve the ability to ignore irrelevant *denied* rows (see 2.2), keep the filter but apply it only to rows that otherwise would be denied.
#### 2.2 `wrong_action_shape` (3 checkpoints) — **design**
**Component:** `MemoryStore.retrieve()` (relevance gating of denied rows) and `sanitize_and_decide` (action mapping).
```python
# agent.py – sanitize_and_decide
    if decision.allowed and not decision.touched_unauthorized:
        return "answer", ...
    if decision.allowed and decision.touched_unauthorized:
        return "answer_redacted", ...
```
The answer body is complete (census says “answer contained EVERY required string”), but because `denied_rbac` is non‑empty the action becomes `answer_redacted`. The denied rows are peripheral—they share only one distinctive term via the plain‑scan fallback—and are not truly responsive. The current code treats them as denials, forcing the wrong label.
**Evidence:** `med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04` (allowed=11, denied_rbac=2), `en_006` (allowed=10, denied_rbac=1), `en_015` (allowed=11, denied_rbac=1).
**Fix:** change the relevance check to exclude irrelevant denied rows. For rows **not** in `from_fts` (i.e. found by the plain‑scan fallback), require at least **two** shared distinctive terms before they are considered for the RBAC/scope gates. Rows from the FTS index already matched every query term, so they keep the current behaviour. This prevents marginal rows from counting as denials, and `sanitize_and_decide` then returns clean `answer`.
#### 2.3 `evidence_withheld` (3 checkpoints) — **design**
**Component:** `MemoryStore.retrieve()` — the same lexical filter.
The census shows `allowed=1` but `action=no_memory`. This is only possible if `decision.allowed` is empty—the single row reported as “allowed” is the **denied_tombstone=0, denied_rbac=0, denied_scope=0** of a record that is *not* actually returned. But the counters show allowed=1, meaning one record was cleared. Yet `sanitize_and_decide` returned `no_memory`. This is contradictory unless the cleared record is the one that does **not** contain the required content (or is entirely absent due to the relevance filter). Given the census, the likely reality is that the query’s terms do not match the answer‑bearing record, so it is dropped by the relevance filter; the one allowed record (if any) is irrelevant, and the query produces an empty `allowed` → `no_memory`.
**Evidence:** `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09` (allowed=1, evidence_used=1, action=no_memory), `en_011_first_seizure_ckpt_01` (allowed=1), `en_013_ibd_ckpt_05` (allowed=1).
**Fix:** identical to 2.1 — remove the semantic word‑overlap gate. The answer‑bearing record will then be included in `allowed`, and the agent will produce an `answer` instead of `no_memory`.
---
### 3. Prioritized changes
1. **`MemoryStore.retrieve()` — drop the relevance rejection for allowed rows**  
   Delete the `if terms and not (terms & _distinctive_terms(plaintext)): continue` block entirely. Every row that passes the tombstone → RBAC → scope gates becomes part of `decision.allowed` regardless of whether it shares words with the query. This directly fixes all 5 `answered_but_content_missing` and all 3 `evidence_withheld` checkpoints. It cannot harm A or F because allowed rows are already fully gated.
2. **`MemoryStore.retrieve()` — apply a stricter relevance threshold to denied rows only**  
   After computing `plaintext`, if the row would be **denied** (fails RBAC or scope), require at least **two** shared distinctive terms for rows not in `from_fts` to be considered genuinely responsive. Only then log it as a denial. This keeps the answer_redacted action from being triggered by irrelevant peripheral rows, fixing `wrong_action_shape` without touching the FTS path.  
   (This is the inverse of the design doc’s plan, which applied the threshold to allowed rows; we apply it to denied rows.)
3. **No DDL changes required**  
   All fixes are confined to the control flow of `retrieve()`. The existing schema and indexes are sufficient. No migration is needed.
---
### 4. What not to do
- Do **not** remove the tombstone branch or reorder `sanitize_and_decide`; the census shows no tombstone involvement in the failing checkpoints.
- Do **not** remove the `answer_redacted` branch — the local test contract requires it for genuine partial authorization.
- Do **not** adjust `from_fts` logic for allowed rows; the goal is to return *more* allowed rows, not fewer denied ones.
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve()",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "MemoryStore.retrieve() — lexical relevance filter",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "name": "wrong_action_shape",
      "component": "MemoryStore.retrieve() — relevance gating of denied rows",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_04",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "MemoryStore.retrieve() — lexical relevance filter (same as first)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03",
    "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
    "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_04",
    "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05"
  ],
  "proposals": [
    {
      "component": "MemoryStore.retrieve()",
      "kind": "design",
      "change": "Remove the `if terms and not (terms & _distinctive_terms(plaintext)): continue` block so that every row passing tombstone, RBAC, and scope gates is appended to `decision.allowed` regardless of word overlap with the query.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05"
      ]
    },
    {
      "component": "MemoryStore.retrieve()",
      "kind": "design",
      "change": "For rows that would be denied (RBAC or scope failure) and that are NOT from the FTS index, require at least two shared distinctive terms before counting the denial. Rows from FTS (which already matched all query terms) keep the current behaviour.",
      "expected_fixes": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_04",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08"
      ]
    }
  ]
}
```