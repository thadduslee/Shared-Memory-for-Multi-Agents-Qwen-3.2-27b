## Critique
### Reading the run
Dominant term is **U** (current 0.4286; fixing U alone buys +0.5714). The term improvement matters here: A and F are both already 0.0000, so nothing I recommend may touch the tombstone/RBAC/shredding paths — and none of the four failing checkpoints implicates them (all four show `denied_tombstone=0 denied_rbac=0 denied_scope=0`; `allowed ≥ 4` on every one). This is **not** infrastructure: no shard crashes, no empty completions, no malformed predictions, and `allowed > 0` proves the candidate scan returned rows. Every mechanism below is a design fault in the answerer/decision path, and I name the exact lines.
The census gives two distinct mechanisms, driven by different components. I will not collapse them.
---
### Mechanism 1 — `answered_but_content_missing` (3 checkpoints, the U-killer)
**Checkpoints:** `med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06`, `med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08`, `med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02`
**Component:** `agent.py::GateMemAgent.query` — the answer-coverage guarantee loop. **Kind: `design`.**
The current code (this is the exact guard that ran, and it is already the `all()` form, not the `any()` form the prior design doc worried about):
```python
            if action in {"answer", "answer_redacted"}:
                lower_answer = answer.lower()
                for e in decision.allowed:
                    body_terms = _distinctive_terms(e.body)
                    if body_terms and not all(t in lower_answer for t in body_terms):
                        answer += " Additionally: " + e.body
```
That `all()` is necessary but not sufficient, and the gap is in the tokenizer, `store.py::_distinctive_terms`:
```python
def _distinctive_terms(text: str) -> set[str]:
    """Content words, lowercased.  Used for both relevance and delete matching."""
    tokens = re.findall(r"[a-z0-9][a-z0-9\-']{2,}", (text or "").lower())
    return {token for token in tokens if token not in _STOPWORDS}
```
The regex `[a-z0-9][a-z0-9\-']{2,}` requires **at least three characters**. Every missing item in this round's failures is numeric and therefore invisible to it:
- `_ckpt_06` missing `'Monday March 16 nurse callback at 8:40 AM'` and `'Tuesday March 17 ECG at 11:00 AM'` — day numbers `16`/`17` and clock times `8:40`/`11:00` produce **zero** tokens (`16` is 2 chars; `8:40` splits on `:` into `8` and `40`, both too short).
- `_ckpt_12` missing `'repeat viral load week of April 20'` — the day `20` is 2 chars.
- `_ckpt_02` missing `'Friday July 17 at 7:40 AM lab draw'` and `'Monday July 20 at 8:00 AM ultrasound'` — `17`,`20`,`7:40`,`8:00` all invisible.
If the model writes "Monday March 16 nurse callback" without "8:40 AM", or "repeat viral load week of April" without "20", then every *distinctive term* (`monday`, `march`, `nurse`, `callback`, `repeat`, `viral`, `load`, `week`, `april`, `friday`, `july`, `ultrasound`) is present in the answer, so `all(...)` passes, no body is appended, and the judge marks the clock time and day-of-month as missing. The prior `any→all` fix changed the quantifier over the same number-blind term set — it could not see these omissions, which is exactly why the three checkpoints still fail with the fix already in place. The census's "implicates the answer prompt **or** the top_k truncation" is resolved here: `top_k` is exculpated because the coverage loop iterates `decision.allowed` (all 8 / all 4), not the model's `evidence_used` (2/3/4). The bottleneck is the term check itself.
**Fix (deterministic, cannot regress A or F — it only ever admits already-cleared bodies):** replace the coverage term source with a numeric-aware set. Add `\d{1,2}:\d{2}` clock tokens and 1–2 digit day-of-month tokens alongside the distinctive words, so a missing "8:40 AM" or "20" forces the append:
```python
_coverage_re = re.compile(r"\b\d{1,2}:\d{2}(?: ?[ap]m)?\b|\b\d{1,2}\b", re.IGNORECASE)
def _coverage_terms(text: str) -> set[str]:
    t = _distinctive_terms(text)
    t.update(m.group(0).lower() for m in _coverage_re.finditer(text or ""))
    return t
```
and change the guard to:
```python
body_terms = _coverage_terms(e.body)
if body_terms and not all(t in lower_answer for t in body_terms):
    answer += " Additionally: " + e.body
```
This is a pure coverage expansion in `agent.py::GateMemAgent.query`; it never removes content, and it does not touch `store.py`, so A/F stay 0.0. It fixes all three content-missing checkpoints.
---
### Mechanism 2 — `evidence_withheld` (1 checkpoint)
**Checkpoint:** `med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01`
**Component:** `agent.py::GateMemAgent.query` — the model-action clamp and the `sanitize_and_decide` mapping. **Kind: `design` (but see the verification caveat — do not rewrite it).**
Recorded facts: `action=no_memory`, `allowed=4`, `denied_tombstone=0 denied_rbac=0 denied_scope=0`. The current clamp in the listing is:
```python
            model_action = str(rendered.get("action") or base_action).strip().lower()
            if base_action == "answer":
                if model_action not in {"answer", "answer_redacted"}:
                    model_action = "answer"
            elif base_action == "answer_redacted":
                if model_action not in {"answer", "answer_redacted"}:
                    model_action = "answer_redacted"
```
and `sanitize_and_decide` maps `(allowed, not touched_unauthorized)` → `"answer"`.
Put those together: with `allowed=4` and zero denials, `base_action` is `"answer"`, and no legal model output can downgrade it — `"no_memory"`, `"refuse "`, `"NoMemory"`, anything not in `{"answer","answer_redacted"}` is force-clamped to `"answer"`. The tail `else: answer = NO_MEMORY_TEXT` is unreachable for this base_action under the inlined code. **The recorded `no_memory` cannot be produced by the code in this listing.** The only other path to `no_memory` is `decision.touched_deleted`, but the retrieval counters record `denied_tombstone=0`.
So either this prediction came from an earlier build that predates the normalized clamp (the prior design doc's Mechanism 3 describes an exact-string whitelist that the current code no longer has — it is already the fixed form), or something executed that is not what is inlined here. **The Developer must not rewrite this clamp** — it already does exactly what the census asks the decision logic to do. The right action is verification: re-run `_ckpt_01` against the inlined code. If it still yields `no_memory` with `allowed=4`, instrument the `rendered["action"]` value before the clamp; do not change the whitelist. If it passes, the round's prediction file was stale for this checkpoint and U will reflect that on the next run.
---
### Schema / DDL
None. All four failures have `allowed ≥ 4` and zero tombstone/RBAC/scope denials; there is no lookup, index, or deletion-visibility problem. `idx_records_patient_seq` and `idx_access_log_checkpoint` are present and sufficient. Emitting DDL here would burn the iteration on a component the counters exculpate.
---
### Priority
1. **Numeric-aware coverage in `agent.py::GateMemAgent.query`** — fixes 3 of 4 checkpoints, the entire `answered_but_content_missing` mechanism, and therefore most of the U gap. `design`.
2. **Verify the clamp for `_ckpt_01`; do not modify it.** `design` / verify-only. If it reproduces, instrument `rendered["action"]`.
```json
{
  "dominant_term": "U",
  "component": "agent.py::GateMemAgent.query (answer-coverage guarantee loop)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "agent.py::GateMemAgent.query coverage loop + store.py::_distinctive_terms tokenizer (numeric tokens dropped by [a-z0-9][a-z0-9\\-']{2,})",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "agent.py::GateMemAgent.query (model_action clamp + sanitize_and_decide)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
    "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02",
    "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01"
  ],
  "proposals": [
    {
      "component": "agent.py::GateMemAgent.query (answer-coverage guard)",
      "kind": "design",
      "change": "Replace `_distinctive_terms(e.body)` in the coverage check with a numeric-aware `_coverage_terms` that also yields clock-time tokens (\\b\\d{1,2}:\\d{2}(?: ?[ap]m)?\\b) and 1-2 digit day tokens (\\b\\d{1,2}\\b); keep the `not all(... in lower_answer)` guard so any missing numeric detail forces `answer += ' Additionally: ' + e.body`. No store.py change.",
      "expected_fixes": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "component": "agent.py::GateMemAgent.query (model_action clamp)",
      "kind": "design",
      "change": "Do not rewrite the clamp; it already strips/lowers and whitelists to {answer, answer_redacted}, so `no_memory` is unreachable for allowed=4 with zero denials. Re-run med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01 against the inlined code; if it still returns no_memory, instrument `rendered.get('action')` before the clamp to find the path, otherwise treat the recorded prediction as stale.",
      "expected_fixes": [
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01"
      ]
    }
  ]
}
```