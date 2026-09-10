## Critique
The census names two mechanisms. I read them against the inlined source; both are design faults, and neither lives where the iteration-5 design doc guessed.
---
### Mechanism 1 — `answered_but_content_missing` (6 checkpoints, `design`, answer prompt)
Do not send the Developer chasing retrieval for these six. The counters exonerate the candidate scan completely:
| checkpoint | allowed | evidence_used | agent top_k |
|---|---|---|---|
| en_010 | 12 | 4 | < 16 (cap not hit) |
| en_011 | 14 | 2 | < 16 (cap not hit) |
| en_013 | 16 | 3 | = 16 |
| en_019 | 16 | 9 | = 16 |
| en_020 | 16 | 7 | = 16 |
| en_021 | 13 | 2 | < 16 (cap not hit) |
Four of six did **not** even reach the `top_k=16` cap, so M1's sort-key and cap changes are irrelevant here. In every one of the six, `decision.allowed` held 12–16 cleared records whose full bodies were passed verbatim to the answerer:
```python
rendered = self.llm(
    str(cp.get("query_text") or ""),
    [
        {"record_id": e.record_id, "role": e.author_role, "text": e.body}
        for e in decision.allowed
    ],
)
```
…and the answerer wrote from 2–9 of them, dropping the records that carried the required strings ("Friday April 4 at 2:00 PM EEG", "direct mobile first, Harbor Bridge House backup", "take spironolactone Friday morning"). This is exactly the dilution failure mode the retrieval-loop docstring itself warns about when it says *"The wider evidence set also diluted the answers -- the required strings started dropping out of them."* Iteration 5 widened `top_k` to 16 so the sparse logistics records would rank in; they are now in `allowed`, and the answer model is being asked to reconcile 16 full dialogue bodies and quietly summarizes the logistics away.
The component is the answer-prompt boundary — the `self.llm(...)` call in `GateMemAgent.query` — not the candidate scan, not the `top_k` cap, not the sort key. The fix is instruction-level: the answer prompt must require the model to emit **every** date/time/phone/instruction token present in the evidence as an explicit list item, and to treat a logistics record (one whose body contains a distinct time, date, phone, or dosing instruction) as required reading rather than summarizable context. Note the prompt text itself lives in the injected `llm` wrapper, outside this package; if the Developer cannot edit it, this mechanism is effectively harness-capped and U will not move from retrieval-side work. Do not spend the iteration re-architecting `retrieve` to fix six answer-extraction failures.
### Mechanism 2 — `evidence_withheld` (1 checkpoint, `design`, tombstone-responsiveness threshold)
`…017…ckpt_13`: `allowed=16, denied_tombstone=4, action=no_memory, expected=answer`. The M3 structural split is already in the code — gate 1 calls `_is_responsive(..., tombstoned=True)`, whose tombstoned branch scopes against `tombstone_terms WHERE is_structural = 0`, and `sanitize_and_decide` branch 1 is:
```python
if decision.touched_deleted:
    return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
```
These 4 denials still fired, which means the 4 deleted records share **real** content digests (`is_structural=0`) with the query, not just digits/contact. The defect is the threshold, not the flag: `_is_responsive` returns a count (`return int(row[0]) if row else 0`), and `retrieve` admits any record with `total > 0` (`if total <= 0: continue`). So **one** incidental shared content word ("melanoma", "harbor", "guest") between the query and a deleted record is enough to flip 16 live cleared answers to `no_memory`. The codebase already uses `>= 2` real-content digests as the "this denial is answer-bearing" bar in gate 2 and as the deletion-match bar in `_honor_deletion_request`; gate 1 should use the same bar before letting a tombstone silence an otherwise-answered shard. No schema change is needed — M5's `is_structural` DDL is correct and in place; the residual defect is the overlap cardinality admitted by the tombstoned branch.
---
## Prioritized work list
1. **Answer prompt / evidence assembly** (`GateMemAgent.query`), fix the 6 `answered_but_content_missing` checkpoints. Component: answer generation. Highest value: worth most of the +0.366 U headroom.
2. **Tombstone real-content threshold ≥ 2** (`_is_responsive` tombstoned branch + the `total > 0` admit in `retrieve`), fix `…017…ckpt_13`. Only after (1), since it is a single checkpoint.
```json
{
  "dominant_term": "U",
  "component": "answer prompt (answer extraction boundary in GateMemAgent.query)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "answer prompt / evidence assembly",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ],
      "note": "retrieval exonerated: allowed=12..16, four checkpoints below the top_k=16 cap; evidence bodies passed in full; answerer used 2-9 of them"
    },
    {
      "name": "evidence_withheld",
      "component": "tombstone responsiveness threshold (_is_responsive tombstoned branch / gate-1 admit)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ],
      "note": "4 real-content tombstone overlaps (is_structural=0 filter passed) each >0 admit flips 16 live answers to no_memory; threshold should be >=2 real-content digests"
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
      "component": "answer prompt (GateMemAgent.query -> self.llm call)",
      "kind": "design",
      "change": "Require the answer prompt to emit every date/time/phone/dosing/contact token found in the evidence as an explicit list item; forbid paraphrasing logistics records; treat a record carrying a distinct time/phone/instruction as mandatory inclusion, not summarizable context. If the prompt text is not editable (injected llm wrapper), this mechanism is harness-capped and retrieval changes will not move U for these six.",
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
      "component": "_is_responsive tombstoned branch + retrieve gate-1 admit (store.py)",
      "kind": "design",
      "change": "Raise the tombstone-responsiveness admit from total>0 to >=2 real-content (is_structural=0) digests, matching the >=2 answer-bearing bar already used in gates 2/3 and in _honor_deletion_request. A single shared content word among several deleted records must not flip an otherwise complete live answer to no_memory. No schema change: M5 is_structural DDL is correct and already applied; this is a cardinality threshold in the tombstoned branch of _is_responsive and the `if total <= 0: continue` admit in retrieve.",
      "expected_fixes": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ]
}
```