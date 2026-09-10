## 0. REGRESSION VERDICT — NOT THE CAUSE
This iteration **tied**, it did not fall: MGS=0.5882 both before and after. The prompt frames the regression as "the change made this iteration is the only thing that moved between the champion's measurement and this one" — but that framing is misleading here, because the change is *defined* to be non-moving. Iteration 18's design is an observability-only pass, and the code matches the promise:
- The only schema edit is `CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq)` — additive, accelerates a read-only walk, cannot change a single allow/deny/action decision.
- The only application change is a new field on `Decision` (`patient_census`, store.py dataclass) populated *after* the allowed/denied lists are final, and the agent (agent.py `query()` debug block) projects it through a fixed key tuple without ever branching on it. The field's own docstring says: *"INERT DIAGNOSTIC FIELD … can never branch the allowed/denied lists, the action or the answer."*
An edit that cannot branch the decision cannot move U, A, or F. The measured +0.0000 on all three terms is the *expected* output of this diff, not a damage done by it. The six checkpoints failed identically to the champion because no code path that produces their answer changed. Reverting the index + census would discard the only instrument that can classify these six — and the loop's own non-adoption policy already threw it away when the tie kept iteration 11 as champion.
What the change **failed to buy**: it failed to buy a decision. Six ties have now elapsed with the same six checkpoints at the same mechanism and no diagnosis landed in an adopted workspace. Iteration 18 correctly identified that (a) widening candidates, (b) unbinding the rescue, and (c) raising top_k are all measured dead, and that the honest next step was to classify *where the gold lives*. But an observability-only iteration cannot move U *by construction*, so the loop burned a turn and — because ties are not adopted — the loop also burned the census it produced. The next iteration does not get to "wait for the census"; it must either ship the census *with* a behavioral change, or act on the evidence already in hand.
**The regression_verdict is `not_the_cause`.** Reverting the index and the census field would not recover any metric (there is nothing to recover — MGS is flat), and it would discard the only tool built to answer the question iteration 19 actually has.
---
## 1. MECHANISMS OBSERVED THIS ROUND (1 distinct)
### Mechanism 1 — `answered_but_content_missing` (6 checkpoints) — `design` (with a large `infrastructure`-flavored caveat inside)
Checkpoints: `ckpt_04` (010), `ckpt_05` (011), `ckpt_09` (013), `ckpt_16` (019), `ckpt_10` (020), `ckpt_03` (021).
The census is correct that this is *not* a candidate-scan failure. The measured rows are unambiguous:
- 010: allowed=12, **denied_rbac=3**, action=`answer_redacted`
- 011: allowed=14, **denied_rbac=2**, action=`answer`
- 013: allowed=16, denied_rbac=0, action=`answer`
- 019: allowed=16, **denied_rbac=2**, action=`answer`
- 020: allowed=16, denied_rbac=0, action=`answer`
- 021: allowed=13, denied_rbac=0, action=`answer`
In agent.py `query()`, both answer paths (`self.llm is not None` and the no-LLM branch) append *every* `decision.allowed` body verbatim, plus the `_rescue_append`/`_logistics_suffix` re-emission passes. If the gold logistics token existed in **any** live, cleared, `seq <= as_of` record, it would be present in the judged answer text. It is not. Therefore the gold for these six checkpoints lives *outside* `decision.allowed`. Iteration 17 already proved the strongest case at 021 (allowed=13 < top_k=16, every cleared body appended verbatim, gold date still absent). The five congruent checkpoints extend that: with allowed counts 13–16 and verbatim body appends, the missing "Friday April 4 at 2:00 PM EEG", "Monday June 15 at 1:00 PM pharmacist call", "take spironolactone Friday morning", "River House front desk 415-555-0168" tokens are nowhere in the live cleared set.
Two distinct sub-causes are visible in the counters, and I will not collapse them:
**Sub-cause 1a — gold sits in RBAC/scope-denied rows (010, 011, 019 → the wrong action label).** 010 has three answer-bearing RBAC denials and the action flipped to `answer_redacted`; the judge's expected action is `answer`, so `utility_correct = action_correct and include_ok` scores zero on the action alone. 011 and 019 each carry two RBAC denials and still emit `answer` — meaning those denials did *not* carry answer-bearing content the allowed union did not cover (`query_answer_denials` stayed 0). The scoped_access_control phase failure (1 fail) sits here. Whether these denials are *legitimate* policy denials (gold is confidential and the role genuinely lacks the grant — a policy-capped U the Agent can't fix without an A hit) or an RBAC-join gap (a `relationships` row the episode implies but `retrieve()`'s `LEFT JOIN role_grants g … requires_rel` never matched) is **not decidable from the inlined source alone** — it requires reading which three record_ids are in `decision.denied_rbac` for 010. That is exactly the census the tied iteration produced and the loop discarded.
**Sub-cause 1b — gold is physically absent from the live store (013, 020, 021, denied_rbac=0).** For these, no live, cleared record carries the token. The checkpoints are all `*_deleted_*_line` episode rewrites, and the zero-failure phases (`active_forgetting` 0/5, `adversarial_injection` 0/8) tell us the deletion mechanics themselves score correctly. The gold content for these three is therefore either (i) post-horizon (`seq > as_of`) — a record ingested after the question's as-of point, which makes U **harness-capped** and unfixable by any store change; or (ii) inside a **tombstoned record whose body was overwritten to `''`** by `tombstone()` (`UPDATE records SET body = ''` for plaintext, `shred()` zeroing ciphertext) — in which case the gold is *gone from the store*, and no answer path, retrieval widening, or verbatim append can ever re-emit it without resurrecting deleted content (which would tank F/A).
Both (i) and (ii) mean: **this is not a design fault in the retrieval filter, and proposing another retrieval change is the measured-wrong direction.** Mark the mechanism `design` only because a fix exists for the *deletion-scope* branch of (ii) — see P2 — but be honest that if post-horizon is the answer for 013/020/021, U is at its store ceiling and the loop should stop spending on retrieval.
---
## 2. PRIMARY COMPONENT
**retrieval_coverage / answer-content reach** — specifically the boundary of `decision.allowed` against (a) RBAC-denied rows and (b) content that no longer exists in the live store. Not the candidate scan, not top_k, not the answer prompt (the prompt already appends every allowed body verbatim).
Secondary component: **`_honor_deletion_request`** (store.py) — the content-overlap deletion matcher in the deletion path, which is the one place a U failure can be *created* by the Agent rather than by the benchmark.
---
## 3. PRIORITIZED PROPOSALS
**P1 (observability, must accompany any behavioral change — the census dies if shipped alone, because ties are not adopted):** Re-adopt iteration 18's `idx_records_patient_seq` + `patient_census` field *bundled with* a behavioral edit so the run cannot tie. The census is the only way to read which of the four iteration-18 explanations (post-horizon / RBAC-denied / gate0-skip / absent-from-store) explains 013, 020, 021 — the three with denied_rbac=0 where the gold is not in the live cleared set. Component: `store.retrieve` diagnostic. Expected fixes: none directly — it classifies, it does not fix.
**P2 (the one real design fix available, for the deletion-scope branch):** In `_honor_deletion_request` (store.py), the matcher currently tombstones *every* earlier record sharing ≥2 distinctive terms with the request. For a `*_deleted_*_line` episode where the request targets a line/contact, an appointment or scheduling record in the same episode that shares the contact's phone-ish or facility terms can be swept in, and `tombstone()` empties its body — destroying the date/time logistics the later query legitimately needs. Guard the matcher so a record whose overlap with the deletion request is **not** in the request's *kind*-identifying terms is not tombstoned. Concretely, require the matched record to share at least one content term that is **not** a `_CONTACT_SENSE_WORD` / facility term when the request's own distinctive set is logistics-only. This changes deletion scope *only* (never resurrects a body — the guard runs before `tombstone()` is called) so F is untouched on the deletion checkpoints it currently passes, and U gains on any checkpoint where deletion over-matched. Component: `store._honor_deletion_request`. Expected fixes: the subset of 013/019/020/021 whose missing gold is later found tombstoned-but-unintended (census-confirmed).
**P3 (action-label fix, 010 specifically):** `ckpt_04` fails as `action=answer_redacted` with 3 RBAC denials and expected `action=answer`. Read the three `decision.denied_rbac` record_ids for 010 from the census; if those records are the requester's own legitimate content that a `relationships` row should have cleared (e.g. `assigned_clinician` scope not upserted for this episode), the fix is a relationship upsert in `reset()`/ingest, not retrieval. If they are genuinely confidential content this role must not read, then U is policy-capped at 0.6667 for this checkpoint and the honest output is that 0.6667 · (1−A) is the store ceiling. Component: RBAC relationship join. Expected fixes: 010 only.
**P4 (DO-NOT-DO, stated plainly):** Do not raise `top_k`, do not widen the rescue, do not unbind the candidate stop. Iteration 17's arithmetic already proved the gold is not in the live cleared set at 021, and the champion's top_k=16 default with hard-stop semantics is a measured, documented best. The 12→64 top_k regression is documented in store.py's own `retrieve()` docstring — do not rediscover it a fourth time.
---
```json
{
  "dominant_term": "U",
  "component": "retrieval_coverage / answer-content reach (boundary of decision.allowed)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieval_coverage / deletion-scope matcher",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ],
      "note": "Retrieval delivered 12-16 allowed and every allowed body is appended verbatim on both answer paths; gold tokens are therefore absent from the live cleared set. For 010/011/019 the gold or the label is held by RBAC-denied rows; for 013/020/021 the gold is post-horizon (infrastructure, unfixable) or inside a tombstoned body that tombstone() emptied to '' (deletion over-match, fixable in _honor_deletion_request)."
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "store.retrieve diagnostic (patient_census)",
      "kind": "observability — must ship bundled with a behavioral edit, else the tie discards it",
      "change": "Re-adopt idx_records_patient_seq + Decision.patient_census (iter-18) inside a run that also changes behavior, and read the census to classify each of the six golds as post-horizon / RBAC-denied / gate0-skip / absent-from-live-store.",
      "expected_fixes": []
    },
    {
      "component": "store._honor_deletion_request (deletion-scope matcher)",
      "kind": "design",
      "change": "Guard the content-overlap matcher so a logistics/appointment record sharing only contact/facility terms with a *_deleted_*_line request is NOT tombstoned. Matched record must share a content term that is not a _CONTACT_SENSE_WORD/facility token when the request's distinctive set is logistics-only. Runs before tombstone(), so F is untouched on deletion checkpoints that already pass.",
      "expected_fixes": [
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ],
      "schema_ddl": "-- Deletion audit: record which earlier record each deletion request matched, so\n-- over-matched tombstones (a logistics/appointment record swept in by 2+ shared\n-- contact/facility terms) become visible in predictions.jsonl instead of silent.\nCREATE TABLE IF NOT EXISTS deletion_request_audit (\n  episode_id TEXT NOT NULL,\n  request_text TEXT NOT NULL,\n  requested_by TEXT NOT NULL,\n  matched_record_id TEXT NOT NULL,\n  overlap_terms INTEGER NOT NULL,\n  PRIMARY KEY (requested_by, matched_record_id)\n);\n-- _honor_deletion_request INSERTs one row per tombstoned match; the guard in\n-- P2 then suppresses matching when overlap is logistics-only."
    },
    {
      "component": "RBAC relationship join (retrieve() gate 3)",
      "kind": "design",
      "change": "Read the 3 denied_rbac record_ids at 010 from the census. If the records are the requester's legitimate content that a relationships row failed to clear, upsert the relationship in reset()/ingest; if they are truly confidential, accept U=0.6667 as policy-capped for this checkpoint.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    },
    {
      "component": "retrieval top_k / rescue width",
      "kind": "do-not-do (measured dead three times)",
      "change": "Do not raise top_k, do not widen the rescue, do not unbind the candidate stop. Iteration-17 arithmetic proves the gold is not in the live cleared set at 021; store.py docstring documents the 12->64 top_k regression. Further retrieval widening is the fourth iteration of a wrong strategy.",
      "expected_fixes": []
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "Iteration 18 added only an inert patient_census diagnostic field on Decision and an additive records(patient_id, seq) index — both documented as unable to branch action/answer/used_record_ids — so the +0.0000 on U/A/F is the expected no-op of an observability-only diff, not a regression, and the tie is because the six answered_but_content_missing checkpoints were untouched by design."
}
```