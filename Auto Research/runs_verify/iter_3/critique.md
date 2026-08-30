## Critique
The dominant term is `U` (current 0.1429, would recover +0.8571). Every cited checkpoint fails on the same mechanism: the retrieval loop discards a record that shares no distinctive term with the query, even though that record is fully authorized. The missing content (e.g. `apixaban`, `stop the estrogen patch`, appointment times) never reaches `decision.allowed`, so no answer prompt can surface it. This is a **design** flaw in the retrieval filter, not an infrastructure failure.
### Mechanism: `answered_but_content_missing` (6 checkpoints)
**Component:** `MemoryStore.retrieve()` — the zero‑relevance floor on the allow path.
**Evidence:** All six failing checkpoints show `allowed > 0` and `denied_tombstone = 0`, yet the expected answer content is absent. The only way that happens is that the required record was excluded from `candidates` before `top_k` was applied. The code explicitly does this:
```python
score = _relevance_score(terms, _distinctive_terms(plaintext))
if score <= 0.0:
    # Shares no distinctive term with the query...
    continue
candidates.append((score, int(row["seq"]), row, plaintext))
```
That `if score <= 0.0: continue` is a binary gate: any cleared record sharing **zero** distinctive query terms is dropped. For example, the query “what medication should I take?” shares no tokens with a record stating “start apixaban” – the record is stripped from the candidate pool even though it is authorized and exactly answers the query. The observed `allowed` counts (2–19) are well below `top_k=20`, so the floor, not truncation, is what discards the needed record.
The design intent (docstring) says “relevance is a RANK, not a GATE”, but the floor violates that: it is precisely the same binary keep/drop that the previous iteration tried to remove. The fix is to eliminate the floor entirely and let `top_k` bound the output.
**Proposal:**
- **Remove** the `if score <= 0.0: continue` branch in `retrieve()`.
- Append **every** gate-cleared record to `candidates`, regardless of relevance score.
- Sort by `(score, seq)` descending as before, then take the first `top_k`.
- Update the docstring to state that all authorized records are eligible; relevance only orders them.
This does not change the three gates (tombstone → RBAC → relationship/scope) and does not affect denied tallies. It increases `U` because the required content now reaches the answerer.
**Expected fixes:** All six checkpoints (`..._ckpt_06`, `..._ckpt_01`, `..._ckpt_08`, `..._ckpt_09`, `..._ckpt_02`, `..._ckpt_02`).
**Secondary note (not a separate mechanism):** Checkpoints `..._ckpt_08` and `..._ckpt_09` also show `denied_rbac > 0`; the action becomes `answer_redacted`. This is a separate policy question about whether those denied records are truly unauthorized, but the census attributes the missing content to the same retrieval omission. The fix above is necessary regardless; whether to also relax `sanitize_and_decide` can be assessed separately after the floor is removed.
---
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve()",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieval filter (zero-relevance floor)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
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
      "change": "Remove the `if score <= 0.0: continue` branch. Append every gate-cleared record to `candidates` regardless of relevance score. Keep the ranking sort and `top_k` truncation. Update the docstring to state that relevance only orders, never excludes, authorized records.",
      "expected_fixes": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    }
  ]
}
```