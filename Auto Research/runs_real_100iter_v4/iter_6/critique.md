```json
{
  "dominant_term": "U",
  "component": "answer formatting (agent.py query())",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "answer boundary / post-answer verification (agent.py _logistics_suffix and answer assembly)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "tombstone gate threshold (store.py retrieve() gate 1)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "agent.py query() — answer assembly",
      "kind": "design",
      "change": "Always use the deterministic join of `decision.allowed` bodies as the answer text, regardless of whether `self.llm` is set. In `query()`, replace the branch that calls LLM rendering with a direct join (or, if LLM is required for natural wording, append the full text of every allowed body as an appendix). This guarantees every cleared record’s content appears verbatim, eliminating all six `answered_but_content_missing` failures where retrieval succeeded.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "store.py retrieve() — gate‑1 tombstone admission",
      "kind": "design",
      "change": "Refine the tombstone admission condition so that a tombstone only forces `no_memory` when the live allowed records do NOT already cover the tombstone’s query‑matching content. Concretely: after collecting `decision.allowed`, compute the union of real‑content term hashes of allowed records. For each candidate tombstone row that passes the current overlap threshold, check whether the tombstone’s full content‑term set (from `tombstone_terms` where `is_structural=0`) is a subset of that allowed union. If it is a subset, skip (do not admit to `denied_tombstone`) — the live answer already carries the information and the deletion does not represent a unique fact. Only admit when the tombstone adds content not present in the live allowed evidence.",
      "expected_fixes": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "The A‑term increase (0.0588 → 0.1176) cannot be traced to either the logistics‑suffix post‑processor or the gate‑1 threshold change; the census shows no RBAC leak failures (role_mismatch, cross_patient) among the listed checkpoints, and both edits target the U and F terms exclusively, leaving the RBAC allow/deny boundary untouched."
}
```