## 0. Regression verdict: REVERT the cap raise
**The only behavioral diff between the champion (iteration 11, U≈0.667) and this run (iteration 12, U=0.6111) is `top_k: int = 40` in `GateMemAgent.__init__`** (memory_system/agent.py), raised from 16, plus an inert census slice (`[:24]`→`[:64]` in `query()` that the agent reads nowhere). The design's stated target was the six `answered_but_content_missing` checkpoints, blamed on gold records ranked 17+ behind chatty rows. The measurement refutes that mechanism, and the cap raise self-refutes: **all seven targeted checkpoints still fail**, and three of them prove the cap never bound.
**Evidence the cap is not the mechanism, and did not fix the target:**
1. **Three failing checkpoints have `allowed` counts below the OLD cap of 16** — so the retrieval loop processed fewer than 16 allowed records and the `if len(decision.allowed) >= top_k: break` guard in `retrieve()` never fired in either iteration. Their allowed/denied/answer are byte-identical under `top_k=16`:
   - `med_episode_rewrite_en_010...ckpt_04`: allowed=12 (cap-irrelevant)
   - `med_episode_rewrite_en_011...ckpt_05`: allowed=14 (cap-irrelevant)
   - `med_episode_rewrite_en_021...ckpt_03`: allowed=13 (cap-irrelevant)
   A gold record ranked 17+ is definitionally impossible on these; truncation is not their failure.
2. **The content-missing mechanism cannot be a breadth problem at all.** The agent's answer for `action=answer` is `summary + "\nDetails: " + " ".join(e.body for e in decision.allowed)` — every allowed body is appended verbatim — plus `_logistics_suffix(...)` over the same gated bodies. So *any* gold content present in *any* `decision.allowed` body appears in the judged answer text verbatim. The measured fact that content is missing therefore forces the conclusion: **the gold content is not in `decision.allowed`** — it was dropped at gate 0 (relevance), denied at gates 2/3 (RBAC/scope), or its tombstone was coverage-skipped. None of those is changed by widening a ranked stop on `allowed`; a wider cap cannot put into `allowed` a record gate 0 rejected or gates 2/3 denied.
3. Four of the seven have `denied_rbac=0, denied_scope=0, denied_tombstone=0` with `allowed` 13–36 and the answer already containing every allowed body verbatim — the gold record is absent from the entire decision, allowed *and* denied. Cap manipulation is structurally inert there.
Since the seven census checkpoints are cap-insensitive (three provably, the rest because verbatim append makes content-presence a property of `allowed`, not breadth), the measured −0.0490 U regression must live on **cap-sensitive shards not listed in the census**: shards where the raise newly fed records ranked 17–40 into gates 2/3 (at cap 16 those records were skipped before any gate), some were denied RBAC/scope, and their content was *uncovered* by the allowed union so marginal `query_answer_denials` counted them and flipped a clean `answer` to `answer_redacted` — the exact iteration-9 mechanism the design claimed was defused, defused only for denials whose content the allowed union already carries.
Revert is the disciplined call. The cap raise achieved zero of its seven targeted fixes and is the only change available to explain the regression; a wrong revert here costs one iteration, but keeping a change that moved U in the wrong direction and fixed nothing costs the same and leaves a false theory in the code comment.
**Edit to undo:** `memory_system/agent.py`, `GateMemAgent.__init__` signature — restore `top_k: int = 16` (currently `top_k: int = 40`). Revert the explanatory comment with it. Leave the census slice; it is inert and needed for diagnosis.
---
## 1 & 2. Mechanisms and components
**Mechanism census this round: exactly one bucket, `answered_but_content_missing` (7 checkpoints).** Its own census note blames "the answer prompt or the top_k truncation — not the candidate scan, which demonstrably returned rows." Both imputations are refuted by the verbatim append: the answer *is* the allowed bodies, so missing content means content never entered `allowed`. Component is therefore **admission into `decision.allowed`**, not the answerer and not `top_k`.
- **ckpt_04 / ckpt_05 / ckpt_19** (`denied_rbac` = 3 / 2 / 4): gold content is plausibly inside an RBAC-denied record. `query_answer_denials` marginal accounting correctly counts a denial whose real-content digests the allowed union lacks — ckpt_04's `expected=answer got=answer_redacted` is the label side of that same geometry: the 12 allowed records do not carry what the judge requires and 3 responsive records were denied. This is a grants/relationship-scope geometry fault (`role_grants` x `relationships` join, `_relationship_ok`), not retrieval breadth. **design.**
- **ckpt_07 / ckpt_13 / ckpt_20 / ckpt_21** (all denials = 0; names `..._deleted_..._line/contact`): gold content is in neither allowed nor denied — the record either failed gate-0 responsiveness (`_is_responsive`/`_match_terms` overlap) or was tombstoned and its admission coverage-skipped by the deferred pass. Because `denied_tombstone=0`, this is *not* a deletion-lookup gap at present: no tombstone fired, so **no DDL is warranted** for these. It is an admission/logic question. **design**, but with the counter-evidence noted: a deleted record's judge-required literal (phone/date/dose token) is a structural digest, which the coverage pass excludes on both sides (`is_structural = 0` filters), so a tombstone whose real-content terms an allowed record restates can be skipped while its required literal is gone — but skipping is what keeps `action=answer`; admitting it would flip these to `no_memory` and fail them at the label. I cannot resolve which is intended without the episode's deletion semantics, and I will not patch it blind.
**Infrastructure:** none. Scoring health is clean (0 malformed, 0 missing, 0 shard failures). Every failure is a design-side admission question.
---
## 3. Prioritized changes
- **P0 — REVERT the cap** (component: `retrieve()` loop / `GateMemAgent.__init__`). Change: `top_k` 40→16. Expected fixes: the unseen label-flip regressions that account for the −0.0490 U drop (no ids available — they are not in the captured census, which only lists the seven that *were already failing*). Explicitly expect it to fix **none** of the seven listed checkpoints, because those are cap-insensitive by the byte-identical argument above.
- **P1 — Read the census before theorizing again.** The census is now 64-deep precisely so ckpt_04/05/19's denied records' rank/digest geometry is visible. Before any further gate edit, dump `decision.candidate_census` for all seven ckpts and locate where the gold record ranked and which gate dropped it. Change: no production code; a test that asserts, for each of the seven, the gold record's census rank against the gate that excluded it. Component: observability. These fixes make the *next* proposal concrete instead of speculative; expected to fix none of the seven on its own.
- **P2 — do NOT touch gate 0, the answer prompt, or the deferred-tombstone skip this round.** The answer prompt is exonerated by the verbatim append; gate 0 is untested-by-evidence; the tombstone skip is load-bearing for F and flipping it turns the four deleted-line ckpts from content-missing into label-failing `no_memory`. Component: none (deliberate restraint).
Schema: no DDL this round. Six of seven show `denied_tombstone=0` — there is no deletion-visibility lookup gap, and the one checkpoint that is a redaction failure (ckpt_04) is an RBAC/grants geometry question that DDL to `records`/`tombstones` does not touch.
```json
{
  "dominant_term": "U",
  "component": "retrieve() admission into decision.allowed (gates 0/2/3 + deferred-tombstone coverage pass)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "admission into decision.allowed — answer is verbatim concat of allowed bodies, so missing content means gold never reached allowed (RBAC denial on ckpt_04/05/19; gate-0 miss or coverage-skipped tombstone on ckpt_07/13/20/21)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_007_thyroid_biopsy_deleted_rose_lodge_contact_ckpt_07",
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
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04 (allowed=12 < old cap 16)",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05 (allowed=14 < old cap 16)",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03 (allowed=13 < old cap 16)",
    "med_episode_rewrite_en_007_thyroid_biopsy_deleted_rose_lodge_contact_ckpt_07 (allowed=36, all denials=0)",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10 (allowed=19, all denials=0)"
  ],
  "proposals": [
    {
      "component": "memory_system/agent.py GateMemAgent.__init__ / retrieve() loop",
      "kind": "revert",
      "change": "Restore 'top_k: int = 16' (currently 40); revert the explanatory comment. Do NOT touch the census slice [:64] — it is inert and needed for diagnosis.",
      "expected_fixes": [
        "the unseen answer->answer_redacted label flips on cap-sensitive shards that account for the -0.0490 U regression (ids not in the captured census)"
      ]
    },
    {
      "component": "observability / retrieve() candidate_census",
      "kind": "design",
      "change": "Add a regression test per cited checkpoint asserting which gate excluded the gold record (gate-0 vs deny_rbac vs deny_scope vs deferred-tombstone coverage skip), using the 64-deep census entry the iteration already emits.",
      "expected_fixes": []
    }
  ],
  "regression_verdict": "revert",
  "regression_cause": "Raising top_k from 16 to 40 in GateMemAgent.__init__ fixed none of the seven targeted content-missing checkpoints (three provably byte-identical below the old cap, the rest insensitive because the answer already appends every allowed body verbatim) and is the only change available to explain the -0.0490 U drop via label flips on shards where it newly evaluated records ranked 17-40."
}
```