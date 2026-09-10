## Critique
### Reading the census first
Every failing checkpoint this round is the **same mechanism**: `answered_but_content_missing`. Retrieval demonstrably returned rows (`allowed`=12–16 on all six), `denied_tombstone`=0 on all six, and a non-empty answer was produced with `evidence_used` between 2 and 9. No crash, no empty model response, no malformed line, no shard loss — `infrastructure` has nothing to attribute here. This is a `design` fault in **answer coverage**, and it is *not* a tombstone fault: the iteration-11 plan (P4, query-centred tombstone admission) targets exactly the `ckpt_17` no_memory-flip shape, and there is not one such checkpoint in this census. P4 will land with zero measured U movement because the six failing checkpoints all have `denied_tombstone=0` already.
### Why the content is missing when the answer is appended to
Look at the answer assembly the code already performs in `agent.py`:
```python
if action in {"answer", "answer_redacted"} and self.llm is not None:
    ...
    if decision.allowed:
        answer += "\nDetails: " + " ".join(e.body for e in decision.allowed)
```
and the no-LLM branch: `answer = " ".join(e.body for e in decision.allowed)`. Either way, **every allowed body is emitted verbatim**. So if the gold phrase ("Friday April 4 at 2:00 PM EEG", "415-555-0168 ask for Mina", "take spironolactone Friday morning") existed in `decision.allowed`, it would be verbatim in the judged string and the include-check would pass. It did not pass. That forces one conclusion: the gold-bearing record is **not in `decision.allowed`** — it ranked out under the cap.
The culprit is in `store.py` PASS 2:
```python
            decision.allowed.append(
                Evidence(record_id=record_id, ...)
            )
            allowed_content_sets.append(carried)
            self._log(checkpoint_id, requester_id, record_id, "allow")
            if len(decision.allowed) >= top_k:
                break
```
with `GateMemAgent.__init__` pinned at `top_k: int = 16`. Three of the six (ckpt_09, ckpt_10, ckpt_03) have `denied_rbac=0, denied_scope=0, denied_tombstone=0` — nothing was withheld at all, the action was a clean `answer`, and the gold still did not appear. The only way that happens is the sparse logistics/contact record carrying the gold content ranked at position 17+ behind a mud of chatty clinical rows that scored higher on real-content overlap, so the `len(decision.allowed) >= 16` break fired before it was reached. This is precisely the mechanism the previous iteration's ranking change (`candidates.sort(key=lambda item: (item[0]-item[1], item[0], item[1], item[2]), reverse=True)`) cannot rescue: a lone record whose whole answer is one date/time/phone/dose phrase has structurally fewer content digests than a wordy note that reuses the query's vocabulary.
### Why the iteration-9 regression no longer applies
Raising `top_k` was reverted because re-evaluating rankings 16–64 flipped clean answers to `no_memory`. That regression had two legs, and **both are now structurally defused in the current code**:
1. Extra *denied* candidates pushing `query_answer_denials` above 0 — defused by the marginal accounting `sum(1 for s in denied_content_sets if not (s <= allowed_union))`, which counts only denials adding content the allowed union lacks.
2. Extra *tombstoned* candidates being admitted — defused by the deferred-tombstone coverage pass already in `retrieve()`: a tombstone is admitted only when `query_shared and not (query_shared <= covered_union)`. More allowed rows → larger `covered_union` → *fewer* admissions, so a cap raise cannot re-open the no_memory flip. The `F` probes that matter leave `covered_union` empty (nothing allowed), where the conservative admission default is untouched.
So the safe, targeted move is to **raise the allowed cap back up while keeping it a ranked hard stop** — the distinct semantics the code comments themselves insist on.
### Prioritized proposals
1. `GateMemAgent.__init__`: `top_k = 16` → `top_k = 40`. Sole lever that can move all six content-missing checkpoints in one edit, with the regression legs defused as above. Expected fixes: ckpt_05, ckpt_09, ckpt_16, ckpt_10, ckpt_03.
2. ckpt_04 (`expected=answer` got `answer_redacted`, `denied_rbac=3`): label fault from marginal accounting counting a denial whose digest is genuinely outside `allowed_union`. Not fixable blind — the candidate_census serialized via P1 is the *first* place the denied record's rank/digest geometry becomes observable. Do not "fix" this with a ranking tweak this round; it is one more reason P1's output must be read next round rather than another speculative gate edit.
No DDL. These are not lookup or deletion-visibility gaps; the candidate scan returned rows on every failing checkpoint. Schema churn would buy zero measured MGS and is correctly absent from the current design.
```json
{
  "dominant_term": "U",
  "component": "retrieval allowed-list ranked cap (top_k=16) in MemoryStore.retrieve / GateMemAgent.__init__",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "allowed-list top_k ranked truncation before answer assembly",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "GateMemAgent.__init__ (agent.py) top_k default",
      "kind": "design",
      "change": "Raise top_k from 16 to 40, keeping the ranked hard-stop semantics on allowed (do not remove the stop). Regression legs are defused by the existing marginal query_answer_denials accounting and the deferred-tombstone query-centred coverage skip in MemoryStore.retrieve.",
      "expected_fixes": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "candidate_census observability (retrieve/query debug)",
      "kind": "design",
      "change": "Read the serialized candidate_census for ckpt_04 next round to observe the denied record's rank and digests before designing any marginal-accounting change; do not guess at it this round.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    }
  ]
}
```