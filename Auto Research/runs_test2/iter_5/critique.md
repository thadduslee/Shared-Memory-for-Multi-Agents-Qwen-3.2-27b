## Critique
### What the census says, and what the code says
Two distinct mechanisms were observed this round, and they implicate two different functions. Neither is infrastructure: the scoring-health block is clean (0 malformed, 0 shard failures, 0 missing predictions), and every failing checkpoint produced a `non-empty` answer from a live retrieval. Both are `design` faults in code I can name and quote.
Before the per-mechanism analysis, one finding dominates everything else: **the Iteration-5 design's Mechanism-1 fix was never applied.** The design promised to delete the relevance-gate `continue` block. It is still on the floor of `MemoryStore.retrieve()`:
```python
            # Relevance gate (allow side).  A row that is not responsive to the
            # query must never be surfaced ...
            if terms and record_id not in from_fts and not (
                terms & _distinctive_terms(plaintext)
            ):
                continue
```
That is the allow-side filter, and it is the only allow-side filter in the current code: `top_k` is never applied (`retrieve()`'s SQL has no `LIMIT` and `rows` is never trimmed after the scan), and the answer is not model-generated — `query()` hard-codes `answer = " ".join(e.body for e in decision.allowed)`. So the two hypotheses the census floated for mechanism 1 ("the answer prompt", "the top_k truncation") cannot hold against this implementation: there is no answer prompt, and no `top_k` truncation. The census's own caveat — "Where they disagree with the OBSERVED PIPELINE BEHAVIOR, the observation wins" — resolves in favor of the relevance gate. The observed `allowed=N` means the candidate scan returned rows; the fact that the *specific* required fragment is absent from the concatenated answer means the record carrying that fragment was `continue`d out of `allowed` by the gate above. The fix is therefore the deletion the design already spelled out and the developer did not land.
### Mechanism 1 — `answered_but_content_missing` (6 checkpoints): the relevance gate in `retrieve()`
Evidence: `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04`, `ckpt_05`, `ckpt_08`, `ckpt_16`, `ckpt_10`, `ckpt_21`. All six show `answer=non-empty` and `allowed >= 1`, with zero tombstone/RBAC/scope denials on most. The required fragment (e.g. `'no deodorant on the biopsy morning'`, `'Friday April 4 at 2:00 PM EEG'`, `'River House front desk 415-555-0168'`) is missing from an answer that is the verbatim concatenation of `decision.allowed` bodies. The only way a required fragment can be absent is that its record never reached `allowed` — and the only allow-side filter is the relevance `continue`. A generic scheduling/instruction query (e.g. "what appointments do I have next week?", "what should I avoid before the procedure?") shares zero content terms with the appointment/instruction record → the record is silently skipped and never surfaces, even though the requester holds a grant. `'no'` is not in `_STOPWORDS`, so `no deodorant on the biopsy morning` contributes terms like `deodorant`, `biopsy`, `morning`; a query phrased as "what should I avoid?" yields zero overlap and the `continue` fires. That is the complete mechanism, directly quotable from `store.py`.
The safety guard the design required — the `retired` flag with `idx_records_active_patient` — **is** already implemented (`ingest_turn` sets `retired = 1` on delete-request turns and drops them from `records_fts`; `retrieve` filters `r.retired = 0`). So the gate can be deleted now without reopening the delete-request-echo leak the design warned about. `from_fts` is still computed and still feeds `_is_responsive_for_denial`, so deny-side responsiveness behavior is preserved. No schema change is needed for this mechanism: `idx_records_active_patient (patient_id, seq) WHERE retired = 0` already supports the always-on full scan. Do not touch the schema for this.
### Mechanism 2 — `wrong_action_shape` (1 checkpoint): `sanitize_and_decide` branch conditions
Evidence: `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09`. Observed: `action=answer_redacted`, `allowed=16`, `denied_rbac=1`, `denied_scope=0`, `denied_tombstone=6`, expected `answer`. The census states the answer contained every required string — rendering succeeded — and only the action label was wrong. The component is `sanitize_and_decide` in `memory_system/agent.py`, specifically the second branch:
```python
    if decision.allowed and decision.touched_unauthorized:
        return "answer_redacted", "partially authorized: withheld higher-sensitivity records"
```
Here `touched_unauthorized` is True because one record (post-as-of or unrelated to the query) tripped `denied_rbac`, so the *entire action* downgrades from `answer` to `answer_redacted` even though the allowed set (16 rows) already contains everything the query asked for. `query()`'s existing forced-overwrite guard only rescues actions outside `{"answer", "answer_redacted"}` — it cannot promote `answer_redacted` back to `answer`. The fix is in the branch conditions: only return `answer_redacted` when the denied set would have withheld information the query actually asked about. `Decision` already carries `query_terms`, so `sanitize_and_decide` can test whether every query term appears in at least one allowed body; if the allowed set fully covers the query terms, the denied rows were superfluous and the answer is plain `answer`. This is a pure action-mapping change — no retrieval, schema, or index change can move it, exactly as the census said.
### Prioritized work
1. **Delete the relevance-gate `continue` in `MemoryStore.retrieve()`** (`memory_system/store.py`). This is mechanism 1, the dominant group (6 checkpoints), and the change was already designed but not applied. Keep the FTS query for `from_fts` only. Kind: `design`. Expected to fix all 6 `answered_but_content_missing` checkpoints. This is the single largest lever on U.
2. **Add query-term coverage to the `answer_redacted` branch in `sanitize_and_decide`** (`memory_system/agent.py`). Return `answer` when the allowed set already covers every query term even if `touched_unauthorized` is True. Kind: `design`. Expected to fix `ckpt_09`.
3. **No infrastructure work.** 0 shard failures, 0 malformed lines — nothing here is a crash, timeout, or empty completion. The metric is not capped by the harness.
One A-risk note: removing the relevance gate widens the `allowed` set to every scoped row, which is how U recovers, but it also makes authorized-but-irrelevant rows answerable. That is the intended tradeoff from the Iteration-5 design (A risk ≈ +0.01, dominated by the U gain of +0.32 on the margin). Do not hedge it back by re-adding a lexical bar — that would re-create the exact six failures.
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve() relevance gate (memory_system/store.py)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "MemoryStore.retrieve() relevance gate (the `if terms and record_id not in from_fts and not (terms & _distinctive_terms(plaintext)): continue` block)",
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
      "name": "wrong_action_shape",
      "component": "sanitize_and_decide() (memory_system/agent.py) — the `if decision.allowed and decision.touched_unauthorized: return \"answer_redacted\"` branch",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "MemoryStore.retrieve() (memory_system/store.py)",
      "kind": "design",
      "change": "Delete the allow-side relevance gate block (`if terms and record_id not in from_fts and not (terms & _distinctive_terms(plaintext)): continue`). Keep the FTS query and `from_fts` computation — it feeds `_is_responsive_for_denial` on the deny side and must stay. Every patient-scoped, non-retired, seq<=as_of row is then evaluated through tombstone → plaintext → RBAC → relationship/scope, and passing rows carry their real body regardless of token overlap. The `retired` flag protecting delete-request turns already exists (ingest_turn sets it; retrieve filters it), so no schema change is needed; do not add DDL.",
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
      "component": "sanitize_and_decide() (memory_system/agent.py)",
      "kind": "design",
      "change": "Change the `if decision.allowed and decision.touched_unauthorized:` branch so it returns `answer_redacted` only when the query's terms are NOT fully covered by the allowed bodies; if every term in `decision.query_terms` appears in the concatenated allowed text, return `answer` instead (the denied rows were superfluous to what was asked). Uses the already-present `decision.query_terms`; no signature change.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09"
      ]
    }
  ]
}
```