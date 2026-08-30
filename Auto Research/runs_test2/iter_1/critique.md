## Critique
The dominant failing term is **U** (0.1813 current MGS vs. 0.8157 if U perfect). The measured mechanism census shows **three distinct design faults** that together account for all 12 utility failures. Retrieval (FTS index, RBAC gates, tombstone handling) is working correctly — every failing checkpoint has `allowed ≥ 1` and the answer strings carried the required content when the action label was wrong. The problems live in two files: the denial bookkeeping in `store.py` and the action/answer orchestration in `agent.py`. None are infrastructure failures; all are fixable within the current iteration.
---
### 1. `wrong_action_shape` (5 checkpoints) — `design`
**Mechanism:** The retrieval returns the correct allowed records and the answer text contains every required string, but the action is `answer_redacted` instead of `answer`. The cause is that *irrelevant* denied records are counted as `touched_unauthorized`.
In `memory_system/store.py::retrieve`, after the RBAC and relationship gates, denied records are appended to `denied_rbac` / `denied_scope` **without** a relevance check:
```python
# Step 2 ... after gates
if row["requires_rel"] is None:
    decision.denied_rbac.append(record_id)
    self._log(checkpoint_id, requester_id, record_id, "deny_rbac")
    continue
# Gate 3...
if not self._relationship_ok(...):
    decision.denied_scope.append(record_id)
    self._log(checkpoint_id, requester_id, record_id, "deny_scope")
    continue
```
Only the *allowed* path applies the relevance filter (`if terms and not (terms & _distinctive_terms(plaintext)): continue`). Therefore any candidate row that matches the FTS query but is denied will set `touched_unauthorized = True`, even if its content is irrelevant to the question. `sanitize_and_decide` then returns `answer_redacted` instead of `answer`. The checkpoint `med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04` shows `allowed=8, denied_rbac=4` — the four denied rows are likely siblings that share terms but are not needed for the answer.
**Fix:** Apply the same relevance filter to denied records before appending to their denied buckets. This makes the decision's `touched_unauthorized` only reflect denials that actually matter to the query. This is a code change in `retrieve()` — no schema change required.
---
### 2. `evidence_withheld` (4 checkpoints) — `design`
**Mechanism:** The retrieval allowed at least one record (`allowed=1`) with zero denials, yet the final action is `no_memory`. Per the census, this happens on deleted‑content queries (e.g., `..._deleted_maple_house_line_ckpt_09`). The decision object passed to `sanitize_and_decide` should yield `answer` (allowed>0, touched_unauthorized False), so the override must come from the LLM action branch.
In `memory_system/agent.py::query`:
```python
if action in {"answer", "answer_redacted"} and self.llm is not None:
    rendered = self.llm(...)
    answer = str(rendered.get("answer") or "")
    action = str(rendered.get("action") or action)   # <-- LLM can override action
```
The LLM is free to return `no_memory` even when evidence was provided, and that overrides the deterministic `sanitize_and_decide` result. For these checkpoints, the safe action is to **answer** (the store has cleared content), so the LLM is incorrectly refusing.
**Fix:** Decouple action from answer generation. Use the action from `sanitize_and_decide` unconditionally; the LLM should only produce the answer text for `answer`/`answer_redacted`, never influence whether an answer is given. This keeps the deterministic policy in control and prevents an adversarial query from coaxing a refusal that U punishes. No schema change.
---
### 3. `answered_but_content_missing` (3 checkpoints) — `design`
**Mechanism:** Retrieval succeeds (`allowed=8`), a non‑empty answer is produced, but the answer omits required content. All allowed rows are in `decision.allowed`, so the omission is either from `top_k` truncation or from the LLM summarization dropping details. For these checkpoints `allowed=8` equals `top_k`, so truncation is not the culprit — the LLM is summarising and losing content.
**Fix:** Either increase `top_k` (though the evidence already spans the top 8) or, more robustly, bypass the LLM for answer generation when the benchmark expects verbatim content. The pure concatenation path (`" ".join(e.body for e in decision.allowed)`) guarantees every allowed record appears verbatim, preventing content loss. Keeping the LLM for action is fine (see fix above), but using concatenation for the final answer eliminates the omission category. This is a change in `agent.py`; no schema impact.
---
## Proposals and Evidence
All three are `design` fixes, each targeting a distinct mechanism. Checkpoint ids cited below.
1. **Filter denied records by relevance** (component: `memory_system/store.py::retrieve`)  
   - Change: before appending to `denied_rbac`/`denied_scope`, check that the row actually matches the query terms (`terms & _distinctive_terms(plaintext)`). If not relevant, skip.
   - Expected fixes: `..._005_behavioral_bridge..._ckpt_04`, `..._006_hyperkalemia..._ckpt_04`, `..._010_breast_biopsy..._ckpt_04`, `..._013_ibd..._ckpt_09`, `..._015_ms_relapse..._ckpt_08` (all `wrong_action_shape`).
2. **Do not let the LLM override action** (component: `memory_system/agent.py::query`)  
   - Change: keep `action` from `sanitize_and_decide`; use `rendered["answer"]` only for answer text, never for the action.
   - Expected fixes: `..._010_breast_biopsy..._ckpt_09`, `..._011_first_seizure..._ckpt_01`, `..._013_ibd..._ckpt_05`, `..._017_melanoma..._ckpt_13` (all `evidence_withheld`).
3. **Use concatenation for answer generation** (component: `memory_system/agent.py::query`)  
   - Change: when `action` is `answer`/`answer_redacted`, always build the answer by joining `decision.allowed` bodies; drop the LLM call for answer text (or keep LLM only for phrasing when the benchmark tolerates it, but concatenation is safest for verbatim content).
   - Expected fixes: `..._011_first_seizure..._ckpt_05`, `..._012_new_hiv..._ckpt_02`, `..._019_ascites..._ckpt_16` (all `answered_but_content_missing`).
These three changes target all 12 utility failures. No infrastructure issues observed; the harness and worker health are clean.
```json
{
  "dominant_term": "U",
  "component": "memory_system/agent.py (sanitize_and_decide + query action/answer handling)",
  "mechanisms": [
    {
      "name": "wrong_action_shape",
      "component": "memory_system/store.py::retrieve (denial bookkeeping)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_04",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "memory_system/agent.py::query (LLM action override)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    },
    {
      "name": "answered_but_content_missing",
      "component": "memory_system/agent.py::query (answer generation)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_02",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
    "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_04",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_02",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
  ],
  "proposals": [
    {
      "component": "memory_system/store.py::retrieve",
      "kind": "design",
      "change": "Apply the same relevance filter (terms ∩ distinctive terms of plaintext) to denied records before appending to denied_rbac/denied_scope; skip irrelevant denials.",
      "expected_fixes": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_04",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08"
      ]
    },
    {
      "component": "memory_system/agent.py::query",
      "kind": "design",
      "change": "Do not let the LLM override the action; keep action from sanitize_and_decide and use LLM output only for answer text (the evidence list).",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_01",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    },
    {
      "component": "memory_system/agent.py::query",
      "kind": "design",
      "change": "For answer/answer_redacted, build the answer by concatenating all decision.allowed bodies (or raise top_k to cover all candidates) to guarantee verbatim content.",
      "expected_fixes": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_02",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    }
  ]
}
```