## Critique
### Regression verdict
This iteration's only change was the two‑phase gate‑1 coverage skip in `MemoryStore.retrieve()` (deferring tombstones and admitting them only when their restored digests are not fully covered by the final allowed set). The measured terms are **unchanged** relative to iteration 5: U, A, and F each moved `+0.0000`. No checkpoint improved, none regressed. The change therefore caused **no regression** – it simply failed to move the metric, because the dominant failure mechanism (`answered_but_content_missing`, 6 of the 7 U failures) is **not** caused by tombstones at all. Those checkpoints already returned `answer` or `answer_redacted` with non‑empty content; the problem is answer composition, not a silenced shard. The coverage skip only targets the single `evidence_withheld` case (ckpt …_017), and that one remained a failure because the allowed live records apparently do not carry the tombstone’s content under the exact term‑digest matching. Since the change neither helped nor hurt, I classify it as **not_the_cause** rather than a regression‑causing edit. The next iteration should attack the true bottleneck: answer assembly, not tombstone admission.
---
### Mechanism census (each bucket addressed)
**1. `answered_but_content_missing` (6 checkpoints) – component: answer prompt / assembly**  
These all show `allowed` ≥ 12 and a non‑empty answer, so retrieval delivered the relevant records. The failure is that the final answer did not contain the *exact* required phrases (dates, times, phone numbers, medication instructions). The current `agent.py` assembly calls the LLM (when present) to summarise the allowed bodies, then appends a `_logistics_suffix` that only re‑emits *tokens* matching the narrow regex list. That suffix will not reconstruct multi‑token requirements such as `"Friday April 4 at 2:00 PM EEG"` or `"Monday June 15 at 1:00 PM pharmacist call"`. The LLM paraphrase drops them and the suffix can’t restore them because it only extracts pieces.  
**Kind: design** — this is fixable by changing how the answer text is built.
**2. `evidence_withheld` (1 checkpoint) – component: tombstone coverage decision**  
ckpt `med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13` has `action=no_memory` despite `allowed=16`. The deferred‑tombstone coverage check apparently decided the allowed records do not cover the tombstone’s term digests, so the tombstone fired. Even with 16 allowed records, if the deleted content is paraphrased differently, the digest‑based overlap is zero. This is a genuine case where a fact is duplicated in live, authorised records, but the gate‑1 coverage logic refuses to admit it.  
**Kind: design** — the coverage condition is too strict (exact‑term subset). A looser “non‑empty intersection” or content‑aware proximity could fix it, but would need careful F‑guarding.
---
### Prioritised proposals
1. **Fix answer assembly to guarantee inclusion of all allowed content**  
   `agent.py`, `query()` — in the `action in {"answer", "answer_redacted"}` branches, replace the LLM‑only summarisation with a verbatim concatenation of `decision.allowed` bodies (or append a `"Details: "` section containing them). This ensures every required string that exists in the cleared evidence appears in the answer. The LLM is optional; if kept, prepend a one‑sentence lead but always append the full, raw allowed bodies.  
   *Expected fixes*: the 6 `answered_but_content_missing` checkpoints, because their required content is present in `allowed` (evidence_used ≥ 2).  
   *Alternative* (less aggressive): widen `_logistics_suffix` to capture full date–time–event phrases and to include the entire sentences around any missing token. However, that still risks missing arbitrary phrasings; the concatenation approach is deterministic and safe since every body in `allowed` already passed all policy gates.
2. **Loosen the tombstone‑coverage skip in gate 1**  
   `store.py`, `MemoryStore.retrieve()` – change the admission condition from “tombstone digests **⊆** covered_union” to “tombstone digests **∩** covered_union ≠ ∅ **and** covered_union non‑empty”. This would let a tombstone be skipped when any allowed live record shares at least one real‑content digest, addressing ckpt …017 where the deleted fact is restated in live content even if not every term matches.  
   *Expected fix*: ckpt `med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13`.  
   *F‑risk*: must verify that no F checkpoint (where the deleted fact is *only* in the tombstone) gets an allowed record that coincidentally shares a common word. Run the full F suite after this change.
3. **Instruction hardening in the LLM prompt** (if LLM is retained)  
   Add a directive to the answer prompt in `agent.py` instructing the model to preserve *every* date, time, phone number, dosage, and instruction **verbatim** from the provided evidence, and to quote them exactly. This is a lower‑priority complement to proposal 1; the concatenation approach alone should already fix the mechanism without relying on model obedience.
---
### Regression verdict
`regression_verdict`: `not_the_cause`  
`regression_cause`: The two‑phase gate‑1 coverage skip in `store.py` did not alter any measured term (U/A/F all +0.0000), so it neither caused a regression nor fixed the target; the true bottleneck is answer composition, not tombstone admission.
---
```json
{
  "dominant_term": "U",
  "component": "answer_prompt_assembly (agent.py)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "agent.py answer assembly",
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
      "component": "store.py gate-1 tombstone coverage",
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
      "component": "agent.py query()",
      "kind": "design",
      "change": "For action in {answer, answer_redacted}, set the answer to the verbatim concatenation of all decision.allowed bodies (optionally with a one-line lead-in). This ensures every required detail that exists in cleared evidence is included in the final answer.",
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
      "component": "store.py retrieve() gate-1 coverage",
      "kind": "design",
      "change": "Change the tombstone admission condition from 'digests ⊆ covered_union' to 'covered_union non-empty and digests ∩ covered_union non-empty'. Run full F suite to guard against leaks.",
      "expected_fixes": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "The two-phase gate-1 coverage skip produced no change to U, A, or F (each +0.0000) and therefore did not cause the tie; the dominant answered_but_content_missing failures are unrelated to tombstones and were not addressed."
}
```