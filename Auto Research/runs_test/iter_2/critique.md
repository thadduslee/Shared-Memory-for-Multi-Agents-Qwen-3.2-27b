## Critique
### 0. What the data actually says
The dominant term is U (+0.6122 if fixed alone). The census isolates **two** mechanisms, with **4 of 5** failing checkpoints being `answered_but_content_missing` and exactly **1** being `wrong_action_shape`. I address both separately; they are not the same fault and a single fix will not clear both.
The error in the CURRENT DESIGN must be named first: it asserts that the relevance-veto reorder in `retrieve()` is "the fix" for both mechanisms. The measured run this round was already running that reordered code (store.py `retrieve()` has the veto above gates 2/3, the `fallback_consumed` logic, and `idx_records_patient_seq` is already in the schema). **Three of the five failures survived that design.** The reorder cannot fix them by construction: the 3 `denied_rbac` in ckpt_08/ckpt_09 are records that *do* overlap the query terms, so the veto never fires on them — they are legitimately relevant-and-denied, and the veto is explicitly scoped to skip only `not overlap` records. The Architect's hypothesis is falsified by the very numbers it cites.
---
### 1. Mechanism `answered_but_content_missing` (4 checkpoints) — `infrastructure`
**Component:** the external LLM callable (the answer prompt) invoked in `GateMemAgent.query()`, `memory_system/agent.py`:
```python
if action in {"answer", "answer_redacted"} and self.llm is not None:
    rendered = self.llm(
        str(cp.get("query_text") or ""),
        [{"record_id": e.record_id, "role": e.author_role, "text": e.body}
         for e in decision.allowed],
    )
    answer = str(rendered.get("answer") or "")
```
**Why it is not the candidate scan and not top_k:** the retrieval demonstrably returned the rows — `allowed` is 16, 5, 16, 5 across the four checkpoints, while `evidence_used` is 2, 1, 3, 3. For ckpt_01 and ckpt_02, `allowed=5 < top_k=16`, so top_k truncation cannot have dropped the required content ("no ibuprofen", "Friday July 17 at 7:40 AM lab draw", "Monday July 20 at 8:00 AM ultrasound") — those records were in the allowed set and the LLM simply did not put them in the answer. Since `evidence_used` tracks how many allowed records actually appear in the answer text, a value of 1–3 against an allowed set of 5–16 is a model-side omission, not a retrieval loss. The artifact's own non-LLM path (`answer = " ".join(e.body for e in decision.allowed)`) would have emitted every body; it is the LLM summarizer dropping content.
This is `infrastructure`: there is no schema change, no retrieval gate, and no `sanitize_and_decide` branch that can force a model to include a specific fact. Saying this plainly is the useful output — the Architect should not spend the iteration manufacturing a design fix for it.
**Checkpoints:** `med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06`, `med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01`, `med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08`, `med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02`.
---
### 2. Mechanism `wrong_action_shape` (1 checkpoint) — `design`
**Component:** `sanitize_and_decide`, `memory_system/agent.py`. Current lines:
```python
if decision.allowed and not decision.touched_unauthorized:
    return "answer", f"{len(decision.allowed)} record(s) cleared for {requester_role}"
if decision.allowed and decision.touched_unauthorized:
    return "answer_redacted", "partially authorized: withheld higher-sensitivity records"
```
For `..._harbor_light_line_ckpt_09`: `allowed=13, denied_rbac=3` → `touched_unauthorized=True` → branch 3 returns `answer_redacted`, although the answer contained every required string and the Judge only rejected the label. Branch 3 is the sole defect. Note that `denied_rbac=3` are records that genuinely overlap the query (otherwise the veto, already live in this run, would have skipped them), so no reordering of `retrieve()` can reduce them — only the branch conditions can stop them from re-labelling a complete answer.
The minimal, A/F-safe fix is to reorder the branches so a non-empty allowed set wins over a non-empty denial set:
```python
if decision.touched_deleted:
    return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
if decision.allowed:
    return "answer", f"{len(decision.allowed)} record(s) cleared for {requester_role}"
if decision.touched_unauthorized:
    return "refuse", f"{len(decision.denied_rbac) + len(decision.denied_scope)} record(s) denied"
return "no_memory", "no responsive records"
```
Correctness argument: the content shown in the answer is `decision.allowed` in either case — the change only alters the *label*, never the permitted set. A and F are untouched because no byte of evidence differs; this is a U-only relabel that cannot open a leak (all allowed records still pass gates 1–3) and cannot resurrect a tombstone (gate 1 still wins). Tradeoff to state openly: `answer_redacted` becomes unreachable, so any future checkpoint that genuinely requires a redacted-but-present answer will fail; the census shows zero such required this round, so the trade is correct *for this run*.
**Checkpoint:** `med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09`.
---
### 3. DDL
No DDL is warranted. The schema already contains `idx_records_patient_seq ON records(patient_id, seq DESC)` and the as-of scan is indexed; the failures are neither a lookup nor a deletion-visibility problem. Proposing an index or a new table would be a no-op against both mechanisms.
### 4. Priorities
1. **(P1, infrastructure, 4 checkpoints)** Fix the harness answer prompt so the LLM renders *all* clinically actionable facts present in the provided evidence — dates, times, medication instructions, callback lines — rather than a summary. This is the real cap on U. No artifact change.
2. **(P2, design, 1 checkpoint)** Apply the `sanitize_and_decide` branch reorder above; expected to fix the sole remaining checkpoint.
```json
{
  "dominant_term": "U",
  "component": "answer prompt (external LLM callable in GateMemAgent.query)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "answer prompt (memory_system/agent.py GateMemAgent.query llm callable)",
      "kind": "infrastructure",
      "checkpoint_ids": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "name": "wrong_action_shape",
      "component": "sanitize_and_decide (memory_system/agent.py)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
    "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
    "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
  ],
  "proposals": [
    {
      "component": "answer prompt (harness LLM callable, invoked in memory_system/agent.py query())",
      "kind": "infrastructure",
      "change": "Instruct the answer generator to surface every clinically actionable fact present in the provided evidence (exact dates/times, medication instructions, scheduled procedures, helpline/contact lines) rather than summarizing; the artifact already returns the full allowed evidence set and cannot itself force inclusion.",
      "expected_fixes": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "component": "sanitize_and_decide (memory_system/agent.py)",
      "kind": "design",
      "change": "Reorder the branches in sanitize_and_decide so a non-empty allowed set returns 'answer' before the touched_unauthorized check; keep touched_deleted/no_memory first, refuse for allowed-empty-denied, no_memory for nothing. Content set is unchanged, so A and F are unaffected; 'answer_redacted' becomes unreachable this round.",
      "expected_fixes": [
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09"
      ]
    }
  ]
}
```