## Critique
### Regression verdict: REVERT the cap raise
The only behavioral edit this iteration was `top_k` 16 → 64 in `GateMemAgent.__init__`. The regression section reports U fell by 0.1667 (= 3 of 18 checkpoints) against iteration 5, and the observed census shows exactly three checkpoints newly failing as `no_memory` (evidence_withheld) while the remaining seven were already failing in iteration 5 (they match the 7 failures of the previous best run, since U was 0.6111 = 11/18 passes).
Those three `no_memory` failures — `med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13`, `..._ckpt_22`, and `med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16` — all show `allowed` > 16 (33, 41, 39) with `denied_tombstone` counts of 5, 4, and 1 respectively. If the cap had stayed at 16, the retrieval loop would have stopped after the 16th allowed record and never evaluated those tombstones ranked beyond that threshold. Raising the cap to 64 caused the loop to process more candidates, surfacing additional tombstoned records that are not covered by the allowed content, and `sanitize_and_decide` branch 1 (any responsive tombstone ⇒ `no_memory`) then fired on benign queries that used to be answered. The design even predicted this risk: *"a checkpoint at the cap could newly surface a tombstoned candidate ranked 17–64 whose content no allowed record restates, flipping `answer`→`no_memory`."* That is precisely what happened, and it costs exactly the 3 new failures (3/18 = 0.1667) that account for the entire U drop.
The cap raise produced no measurable fix: the seven `answered_but_content_missing` checkpoints remain, and allowed counts there are all ≥ 12, so the gold records were already within the previous 16‑candidate window or would have been anyway (some exceed 16 now, but they still fail). We have no evidence the cap helped any checkpoint. **Revert `top_k` to 16** and keep the rest of the design unchanged (the ranking, gate order, marginal `query_answer_denials`, and tombstone coverage‑skip are all sound and measured non‑movers across iterations 5–8).
---
### Mechanism 1: `evidence_withheld` (3 checkpoints) — caused by the cap raise
- **Component**: retrieval hard cap `top_k` in `GateMemAgent.__init__` / `MemoryStore.retrieve`.
- **Kind**: `design` (wrongly widened cutoff, not an infrastructure fault).
- **Checkpoints**: `..._ckpt_13`, `..._ckpt_22`, `..._ckpt_16`.
- **Evidence**: `action=no_memory`, `allowed` > 16, `denied_tombstone` > 0. Those tombstones were not evaluated under `top_k=16` because the loop breaks when `len(decision.allowed) >= top_k`. Under 64 they were reached, deferred, and admitted because their preserved content is not covered by the union of allowed records — forcing `no_memory`. Reverting the cap restores the prior behaviour (these tombstones were never gated and therefore never silenced the answer).
**Fix**: revert the default `top_k` from 64 to 16 and remove the associated comment block in `GateMemAgent.__init__`. This is a pure revert; no other change is needed for these three checkpoints.
---
### Mechanism 2: `answered_but_content_missing` (7 checkpoints) — pre‑existing, not caused by this iteration
- **Component(s)**: 
  - For checkpoints with no denials (`..._ckpt_07`, `..._ckpt_11...ckpt_05`, `..._ibd...ckpt_05`, `..._ibd...ckpt_09`, `..._anemia...ckpt_10`, `..._gender_clinic...ckpt_03`): the gold content is absent from `decision.allowed` and from every denied bucket, so the record never passed gate 0. That is a term‑coverage/`_is_responsive` problem in the candidate scan.
  - For `..._breast_biopsy...ckpt_04` (`denied_rbac=3`, `action=answer_redacted`): the gold content may live in an RBAC‑denied record; if so, this is a grant coverage issue, not a retrieval-breadth one.
- **Kind**: `design`.
- **Evidence**: all seven have `allowed` ≥ 12, many > 16, and some have denials, yet the expected strings ("Rose Lodge desk 415-555-0198", "Friday April 4 at 2:00 PM EEG", etc.) do not appear in the joined answer text. Since the answer is a verbatim join of every allowed body when `llm=None`, the gold string is not in any allowed record. I have not opened the full `_match_terms` / `_index_record` code (the store source is truncated), so I cannot name a line‑level fault here. What is certain: the gold‑carrying record is not surfacing as a candidate at all for most of these, and for `..._ckpt_04` it may be denied by an overly strict role grant.
**Prioritized actions** (after the revert):
1. **First, revert `top_k` to 16** — this recovers the 3 new `no_memory` failures and restores the iteration‑5 baseline.
2. **Then, as a standalone next iteration, instrument the seven content‑missing checkpoints** to log the query’s term hashes and the `_is_responsive` score of every record that contains any gold phrase (derived from the expected answer). This will reveal whether the gold record is missing from `candidates` because (a) its `record_terms` digests lack a query‑matching term, or (b) it was ranked below the cap and excluded. Since `allowed` > 16 on many of them, (b) is unlikely for those with `allowed` ≥ 16, so the dominant hypothesis for most is (a) — a gate‑0 term emission gap.
3. **For `..._breast_biopsy...ckpt_04`**, check whether the requester’s role should be granted access to the denied records that contain the missing content (e.g., a `restricted` or `confidential` grant may be warranted under the policy). This would be a `DEFAULT_ROLE_GRANTS` or relationship-grant change, not a schema/index change.
I deliberately do not propose a DDL or schema change for these seven: the failure is in query‑path term matching and policy grants, both of which are code‑level, not SQL index‑level. Forcing a migration would churn storage for no measured benefit.
---
### JSON
```json
{
  "dominant_term": "U",
  "component": "retrieval hard cap (top_k) in GateMemAgent.__init__",
  "mechanisms": [
    {
      "name": "evidence_withheld",
      "component": "retrieval top_k widening → extra tombstone evaluation",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    },
    {
      "name": "answered_but_content_missing",
      "component": "gate-0 term coverage / RBAC grant for gold record",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_007_thyroid_biopsy_deleted_rose_lodge_contact_ckpt_07",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
  ],
  "proposals": [
    {
      "component": "GateMemAgent.__init__ (top_k default)",
      "kind": "revert",
      "change": "Set top_k back to 16 (from 64) and remove the comment block explaining the raise. Keep everything else identical.",
      "expected_fixes": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    },
    {
      "component": "store.py _match_terms / _is_responsive (gate 0)",
      "kind": "design",
      "change": "Add a debug trace to record which term hashes the gold‑content records carry vs. which query‑term hashes are searched. Determine why the gold record fails gate‑0 (missing term emission). Then widen the emitted term set for the specific missing terms (e.g., include common abbreviations, synonyms, or morphological variants). Do not blindly relax the whole matcher.",
      "expected_fixes": [
        "med_episode_rewrite_en_007_thyroid_biopsy_deleted_rose_lodge_contact_ckpt_07",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "store.py DEFAULT_ROLE_GRANTS / relationships (RBAC)",
      "kind": "design",
      "change": "For med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04, inspect the three denied records to see whether they contain the missing gold content. If so, determine whether the requester’s role legitimately should have access under the policy and add the corresponding grant (or relationship) rather than capping the answer.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    }
  ],
  "regression_verdict": "revert",
  "regression_cause": "Raising the allowed‑set cap from 16 to 64 caused the retrieval loop to evaluate additional tombstoned records ranked beyond 16; these tombstones were not covered by allowed content and forced three benign checkpoints to `no_memory`, dropping U by 0.1667."
}
```