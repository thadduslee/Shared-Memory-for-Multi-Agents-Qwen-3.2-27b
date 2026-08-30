## Critique
### Global notes
The measured census is authoritative. Two distinct mechanisms killed U this round: `answered_but_content_missing` (6 checkpoints) and `evidence_withheld` (3 checkpoints). Both are design fixable; there are no infrastructure failures. The precomputed attribution correctly isolates U as the dominant term (marginal gain 0.3569). I address the larger mechanism first.
### Mechanism 1: `answered_but_content_missing` (6 checkpoints) — design
**Component:** `MemoryStore.retrieve()` — the lexical relevance gate for allowed rows.
The retrieval loop in `store.py` contains:
```python
# Relevance: a row sharing no distinctive query term is not
# responsive to this query at all – not a policy denial – so skip
# it before RBAC gating and log nothing.
if terms and not (terms & _distinctive_terms(plaintext)):
    continue
```
This line *removes* a row from consideration entirely – even if it passes tombstone, RBAC, and scope gates – simply because it shares zero literal tokens with the query. The judge requires semantically present facts (e.g., a query about “pre‑op instructions” must return the body “no deodorant on the biopsy morning” — two token sets with zero overlap). As a result the answer‑bearing row never reaches `decision.allowed`, and the concatenated answer omits the required content. The retrieved counters confirm this: every checkpoint in this mechanism has `allowed>0` and `evidence_used>0`, yet specific required fragments are missing. The design proposal to delete this gate is correct and is the highest‑priority fix.
**Evidence:** `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04`, `_011_first_seizure_deleted_harbor_steps_line_ckpt_05`, `_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08`, `_019_ascites_deleted_juniper_house_line_ckpt_16`, `_020_anemia_deleted_river_house_line_ckpt_10`, `_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03`.
**Proposed fix:** Delete the relevance short‑circuit for allowed rows, as the Architect’s design specifies. Every row that passes tombstone, RBAC, and scope gates must be appended to `decision.allowed` regardless of term overlap. This directly addresses all 6 checkpoints.
### Mechanism 2: `evidence_withheld` (3 checkpoints) — design
**Component:** `sanitize_and_decide()` in `agent.py` (the action‑mapping logic).
The census states that for these checkpoints `allowed=1` and all denial counters are zero, yet the final action is `no_memory`. That is impossible under the current logic:
```python
if decision.allowed and not decision.touched_unauthorized:
    return "answer", ...
if decision.allowed and decision.touched_unauthorized:
    return "answer_redacted", ...
if decision.touched_deleted:
    return "no_memory", ...
if decision.touched_unauthorized:
    return "refuse", ...
return "no_memory", "no responsive records"
```
If `allowed` is non‑empty and no denial is present, the first branch fires and returns `"answer"`. Even if `touched_deleted` were true, the allowed check would still take precedence. The only way to reach `no_memory` is if `decision.allowed` is empty. Yet the recorded counters say `allowed=1`. This discrepancy suggests either (a) the retrieval loop appended to a *different* `Decision` object than the one returned, or (b) the agent’s `query()` method is not using the returned decision’s `allowed` list — but the code clearly does. Since we must treat the census as fact, something in the mapping is defective in practice even if the current source appears correct. The design proposal does not touch this path, so it would not fix these three checkpoints.
**Evidence:** `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09`, `_011_first_seizure_deleted_harbor_steps_line_ckpt_01`, `_013_ibd_deleted_pine_harbor_line_ckpt_05`.
**Proposed fix:** Make the decision logic unconditional on the presence of `allowed`. In `sanitize_and_decide()`, add an explicit guard immediately after the allowed checks: if `decision.allowed` is truthy, never return `no_memory`. Also add a defensive assertion in `GateMemAgent.query()` that if `decision.allowed` is non‑empty, the action must be `"answer"` or `"answer_redacted"`; if it is `"no_memory"`, force it to `"answer"` (or `"answer_redacted"` if `touched_unauthorized`) and log a warning. This will guarantee that allowed content is never suppressed. Even though the current logic should already satisfy this, the recorded behavior shows otherwise, and the guard makes the invariant explicit and unbreakable.
### Prioritized changes
1. **Remove the lexical relevance gate for allowed rows** in `store.py` (highest impact, fixes 6 checkpoints).
2. **Harden `sanitize_and_decide`** so that a non‑empty `allowed` list always yields an answer action (fixes 3 checkpoints).
3. **Keep the denial‑responsiveness filter** exactly as proposed in the design doc; it does not conflict with the above and preserves the anti‑noise property for RBAC/scope denials.
No DDL changes are needed; the schema and indexes are fine. The `top_k` parameter is unused and does not contribute.
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve() relevance gate and sanitize_and_decide() decision logic",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "MemoryStore.retrieve() lexical relevance gate",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "sanitize_and_decide() action mapping",
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
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
    "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "MemoryStore.retrieve()",
      "kind": "design",
      "change": "Delete the block `if terms and not (terms & _distinctive_terms(plaintext)): continue` so that every row passing tombstone, RBAC, and scope gates is appended to decision.allowed regardless of term overlap.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "sanitize_and_decide()",
      "kind": "design",
      "change": "Add a final guard: if decision.allowed is truthy, never return 'no_memory'. In GateMemAgent.query(), assert that action is 'answer' or 'answer_redacted' whenever decision.allowed is non-empty; if it is not, force it and log a warning.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05"
      ]
    }
  ]
}
```