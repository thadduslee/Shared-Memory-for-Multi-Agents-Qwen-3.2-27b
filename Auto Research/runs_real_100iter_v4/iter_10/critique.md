## 0. Regression verdict
**not_the_cause.** Three of this iteration's own attribution numbers say so: U:+0.0000, A:+0.0000, F:+0.0000 relative to the champion. The iteration's diff was (a) reverting the `top_k` default 64→16 in `GateMemAgent.__init__` and (b) planting an "inert" `candidate_census` on the Decision. Reverting the cap is what *recovered* the champion state from iteration 9's 0.4444 — the raw terms match the champion exactly (U=0.6111 A=0.0588 F=0.0000). The change did not cause a fall; it caused a **tie**, which is the correct reading of the "REGRESSION" header (it is a plateau report, the delta column is all zeros).
What it *failed to buy*: the seven content-missing failures are untouched by the cap revert (they existed at iteration 5 under the same cap), and the diagnostic this iteration planted is **unreadable**. `query()` builds its `debug` dict from `n_allowed / n_denied_rbac / n_denied_scope / n_denied_tombstone / rationale` — `decision.candidate_census` is attached to the Decision and then never serialized anywhere the next iteration, `_eval_runner`, or the predictions file can see it. The design even instructs the agent not to read it into `debug`. An evidence-gathering iteration that emits no evidence burns the round: iteration 11 wakes up exactly as blind as iteration 9. That, not the revert, is the wasted edit.
## 1. Mechanism census (2 observed)
### M1 — answered_but_content_missing (6 checkpoints) — `design`
ckpt_04, ckpt_05, ckpt_09, ckpt_16, ckpt_10, ckpt_03.
The decisive fact is in the answer path, not the prompt. In `memory_system/agent.py` `query()`, for any `answer`/`answer_redacted` action the code appends **every** `decision.allowed` body verbatim (`answer += "\nDetails: " + " ".join(e.body for e in decision.allowed)`), and `_logistics_suffix` re-emits any logistics token present in an allowed body but missing from the text. Therefore a gold string is absent from the judged answer **iff the record carrying it never reached `decision.allowed`**. This is not an answer-prompt weakness; the prompt already dumps all gated bodies verbatim. The six subdivide by *why* the gold record missed the allowed window, and the retrieval counters support that split:
- **Ranked/truncated out at `top_k=16`, zero denials** — ckpt_09 (`allowed=16`, rbac=0), ckpt_10 (`allowed=16`, rbac=0): the cap is binding at 16 and no denial explains the gap. The gold-carrying record scored below the 16 that made the window. Both gold payloads are logistics-sparse ("Monday June 15 at 1:00 PM pharmacist call"; "portal okay, River House front desk 415-555-0168 ask for Mina") — a record that reuses the query's content vocabulary plus structural digit/contact digests will outrank a sparse appointment/contact row on real-content overlap. Fixing these is a retrieval-filter ranking/scoring issue, and the `_is_structural` demotion of phone-digit runs and the `__contact__` tag is the prime suspect: when the query itself is *about* a phone/contact/date, structural-only overlap is being discounted as non-answer-bearing.
- **Gold record denied at RBAC, labelled content-missing** — ckpt_05 (`denied_rbac=2`), ckpt_16 (`denied_rbac=2`), ckpt_04 (`denied_rbac=3`, and the Judge message here is literally `expected=answer got=answer_redacted` — a *label* error, not a missing-phrase error). When a gold phrase is in a denied record and the marginal accounting (`query_answer_denials = sum(1 for s in denied_content_sets if not (s <= allowed_union))`) uses **digest-equality**, a paraphrase in allowed content (allowed 12–16 records exist!) does not suppress the denial, branch answer_redacted fires, and the label alone kills U even though the gist is present. ckpt_04's Judge message names exactly this: content was there, the wrong label went out.
- **ckpt_03** (`allowed=13`, rbac=0, denials all zero): the loop ended without filling the cap, so the gold "direct mobile first, Harbor Bridge House backup" fact is in a record scored non-responsive at gate 0 or a tombstone the counters missed; a contact-route fact scored against the `__contact__` structural handling is the mechanism consistent with the others.
I am not forcing these under one story: two sub-mechanisms (ranked-out vs. RBAC-denied/labelled) coexist. The census's own suggestion ("answer prompt or top_k truncation") overweights the prompt — the verbatim Details append rules the prompt out, so the component is the retrieval filter (ranking + top_k) for the 0-denial cases, and the RBAC/classification *labelling* for the denied cases.
### M2 — evidence_withheld (1 checkpoint) — `design`
ckpt_17 (`allowed=16`, `denied_tombstone=4`, action=no_memory).
Records were allowed (16 of them) yet the action is no_memory, so `sanitize_and_decide`'s branch 1 — *tombstone hit wins over everything*, returning `no_memory` even when `decision.allowed` is non-empty — fired. Upstream, the deferred-tombstone coverage skip in `retrieve()` admitted all 4 tombstones to `denied_tombstone` because the skip only passes when the tombstone's preserved digests are an **exact subset** of `covered_union` (the union of allowed records' live `record_terms` digests). A live record that re-states the deleted fact in different words (different day-form, different phrasing of the Harbor Guest back line) has different digests, the subset test fails, the tombstone is admitted, and a legitimately answerable question is silenced. This is the decision logic mapping a Decision to an action — `sanitize_and_decide` branch 1 combined with the digest-exactness of the coverage skip. Design fault, fixable in code.
## 2. Infrastructure
None. Worker shards: 0, missing predictions: 0, local pass rate 1.000. U is capped by design faults in retrieval ranking and in the deny-label/tombstone admission logic — not by any crash or empty completion.
## 3. Prioritized proposals
**P1 — make the planted census actually observable (component: agent.py `query()` debug dict).**
In `memory_system/agent.py`, `query()`'s returned `debug` dict currently omits `candidate_census`. Emit a compact serialization of it (e.g. `debug["candidate_census"] = decision.candidate_census[:8]` with record_id + content_overlap) so the next iteration can see, per failing checkpoint, which gold-carrying record was ranked out vs. denied. One line, provably non-branching to U/A/F. *Expected fix: none by itself — it unblocks P2/P3 with evidence instead of a fourth blind guess. This is the thing iteration 10 should have done and didn't.*
**P2 — fix the 0-denial, cap-bound content-missing (component: store.py responsiveness scoring).**
For ckpt_09 and ckpt_10 (`allowed=16`, zero denials), the gold record lost the ranking. In `retrieve()` PASS-1, ranking leads on real-content overlap and discounts structural digit-run/`__contact__` overlap. When `match_terms` contains **no** non-structural terms (the query is purely a phone/date/contact ask), stop demoting structural overlap — a logistics query whose whole payload is a phone number or appointment must be answerable on those digests. Change is in `_match_terms`/`_is_structural` handling, code-level. *Expected fixes: ckpt_10 (River House 415-555-0168), and plausibly ckpt_03 (direct-mobile/Harbor Bridge backup route) — confirm against the emitted P1 census before finalizing.*
**P3 — stop `digest-equality` from mislabelling answers as answer_redacted (component: store.py marginal accounting; agent.py sanitize).**
ckpt_04 is `expected=answer got=answer_redacted` with 12 allowed records; the 3 `denied_rbac` records were counted as answer-bearing because `not (s <= allowed_union)` under exact-digest comparison, even where allowed records re-state the fact. Change `query_answer_denials` to count a denial as marginal only when its carried digests are absent from *both* `allowed_union` **and** the verbatim bodies the answerer will emit (the bodies are available at that point). Fewer phantom answer_redacted labels ⇒ label returns to `answer`. *Expected fixes: ckpt_04; verify ckpt_05/ckpt_16 once the denied record's content is compared against the actual allowed bodies.*
**P4 — soften tombstone admission when the allowed answer already covers the queried fact (component: store.py deferred-tombstone coverage skip).**
For ckpt_17: admit a deferred tombstone to `denied_tombstone` (forcing no_memory) only when the query's own real-content digests are *not* already present in `covered_union`. Currently the skip tests whether the deleted record's digests ⊆ covered_union; it should test whether the **asked-about** content is answerable from live records — if every query-relevant digest the tombstone carries is already in some allowed body, the deleted fact survives in cleared evidence and no_memory is the wrong action. *Expected fix: ckpt_17.*
No DDL is proposed: none of the seven failures is a lookup/deletion-visibility problem (6 of 7 have `denied_tombstone=0`; the one tombstone case is a decision-logic/admission problem, not an index gap). The `role_grants` values that feed P3's denials are code-seeded in `DEFAULT_ROLE_GRANTS`, so any grant widening would be a code change with an A-risk, and it is deliberately not proposed ahead of the P1 census telling us *which* role/sensitivity tier is gating the gold.
```json
{
  "dominant_term": "U",
  "component": "retrieval filter (ranking/top_k) plus deny-labelling (sanitize_and_decide / query_answer_denials)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieval filter ranking and top_k for the 0-denial cases (ckpt_09/ckpt_10); query_answer_denials marginal-digest labelling for the denied cases (ckpt_04/ckpt_05/ckpt_16)",
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
      "component": "deferred-tombstone coverage skip in store.retrieve() combined with sanitize_and_decide branch 1 (tombstone wins over allowed)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "agent.py query() debug dict",
      "kind": "design",
      "change": "Serialize decision.candidate_census into debug output so the ranked-out-vs-denied split is observable to the next iteration.",
      "expected_fixes": []
    },
    {
      "component": "store.py responsiveness/ranking (retrieve PASS-1, _is_structural handling)",
      "kind": "design",
      "change": "When the query carries no non-structural terms, do not demote structural phone/date/contact overlap in ranking so sparse logistics gold records can reach the allowed window.",
      "expected_fixes": [
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "store.py query_answer_denials marginal accounting",
      "kind": "design",
      "change": "Count a denial as answer-bearing only when its carried digests are absent from the verbatim allowed bodies the answer will emit, not merely from the digest-union.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    },
    {
      "component": "store.py deferred-tombstone coverage skip",
      "kind": "design",
      "change": "Admit a deferred tombstone (forcing no_memory) only when the query's real-content digests are not already covered by allowed records' content.",
      "expected_fixes": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "Iteration 10's top_k 64->16 revert plus inert candidate_census tied the champion exactly (all attribution deltas +0.0000), so the edit caused no regression; it merely failed to advance U because the seven pre-existing content-missing failures sit in ranking and deny-labelling that the cap revert never touches, and the planted diagnostic is never serialized into any readable output."
}
```