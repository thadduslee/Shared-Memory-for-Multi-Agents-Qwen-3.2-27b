## 0. The "regression": there is none — this change was expected to do nothing, and it did exactly that
Hard measurables first: U, A, F all moved **+0.0000** against iteration 5. Nothing got worse. Nothing got better. MGS tied the champion at 0.5752, and per the harness rule the workspace is being rolled back to iteration 5's. So the question is not "which edit cost the metric" — no edit did. The question is "why did the edit fail to buy anything," and that is fully answerable from the code inlined above.
**Verdict: `not_the_cause`.** The iteration's only edit was appending `decision.allowed` bodies verbatim inside the `self.llm is not None` branch of `GateMemAgent.query()`. That branch **does not run at eval time**. The iteration's own design doc asserts it: `_eval_runner.py constructs GateMemAgent(":memory:") with llm=None`, and the constructor default is `llm: ... | None = None`. With `llm=None`, control flows past the LLM branch into:
```python
        elif action in {"answer", "answer_redacted"}:
            answer = " ".join(e.body for e in decision.allowed)
```
That `else` branch has **always** emitted every allowed body verbatim. It is the eval path. The body-append the iteration added to the `self.llm is not None` branch is dead code in the harness's configuration, which is precisely why U did not move by a single checkpoint. The change cannot be blamed for a loss (it caused none), and it cannot be credited for a win (it is unreachable under `llm=None`).
The deeper, sharper conclusion: **the answer-assembly theory is now falsified by measurement.** Iterations 5, 6 and 7's shared premise — that a summarising LLM drops multi-token gold phrases and single-token logistics regexes can't recover them — cannot explain the 6 `answered_but_content_missing` failures, because on the path that actually runs, no summarising LLM exists and the answer *is* the verbatim concatenation of every allowed body. If a gold string were inside any allowed record, it would already be in the judged answer text. It is not. Therefore the gold-carrying record is **not in `decision.allowed`** — the failure sits upstream of answer assembly, in retrieval.
This is the third consecutive iteration on the wrong side of the bottleneck. Stop the answer-side work.
---
## 1. Mechanism census — every mechanism, addressed
Two mechanisms were observed. **No infrastructure failures at all** (`worker shard failures: 0`, `malformed prediction lines: 0`, `missing predictions: 0`). Both mechanisms are `design`.
### Mechanism 1 — `answered_but_content_missing` (6 checkpoints) — `design`
Checkpoints: `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04`, `med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05`, `med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09`, `med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16`, `med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10`, `med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03`.
**Component: the retrieval allowed-set production — specifically the `top_k` enforcement at the tail of PASS 2 in `store.py::retrieve()`, plus candidate production at gate 0.** Not the answer prompt.
The proof is the verbatim-join guarantee above. Since the no-LLM branch (the eval path) concatenates **every** `decision.allowed` body, and these answers are non-empty with 12–16 allowed records each, a missing gold string (`'Friday April 4 at 2:00 PM EEG'`, `'take spironolactone Friday morning'`, `'Harbor Bridge House backup'`, …) means the record that carries it was never admitted to `decision.allowed`. It was either (a) cut by the cap, (b) denied by a gate, or (c) never a candidate at gate 0.
Look at the allowed counts against the cap:
```
ckpt_09   allowed=16  (= top_k, cap hit)   denied_rbac=0  denied_scope=0  denied_tombstone=0
ckpt_10   allowed=16  (= top_k, cap hit)   denied_rbac=0  denied_scope=0  denied_tombstone=0
ckpt_16   allowed=16  (= top_k, cap hit)   denied_rbac=2  ...
ckpt_04   allowed=12  denied_rbac=3        → action=answer_redacted (label failure)
ckpt_05   allowed=14  denied_rbac=2
ckpt_03   allowed=13  denied_rbac=0        → candidates exhausted, no denials at all
```
Four of the seven failures sit at exactly `allowed=16 = top_k`. The block at the end of PASS 2 is the suspect for them:
```python
            decision.allowed.append(...)
            allowed_content_sets.append(carried)
            self._log(checkpoint_id, requester_id, record_id, "allow")
            if len(decision.allowed) >= top_k:
                break
```
Any gold-carrying record ranked 17th or lower never gets evaluated and never lands in `allowed` **or** in a denied list — it silently vanishes from the answer even though it cleared every gate.
Critically, **raising `top_k` is label-safe by this module's own design**. The comment directly above `top_k: int = 16` in `__init__` states the cap raise from 8→11 was made safe because `query_answer_denials` counts **marginal** content — a denial only increments when it adds a digest the allowed union lacks. Allowing *more* records can only grow `allowed`; it cannot add a denial, so it cannot flip `answer` → `answer_redacted`. The iteration-3 disaster (U halving from a wide evidence set) was caused by denials the marginal accounting now neutralizes; the same comment says so. So this lever is cheap and bounded.
Note what it will **not** fix: `ckpt_03` shows `allowed=13`, zero denials, zero tombstones — the loop ended because the candidate list was exhausted, well under the cap. Its gold record never passed gate 0's responsiveness scan at all. That is a term-emission/`_match_terms` coverage problem, and the cap is not the cause there. And `ckpt_04`/`ckpt_05` carry RBAC denials — if the gold content lives in one of the denied records (possible given `denied_rbac=3`/`2` and `action=answer_redacted` on ckpt_04 where the gold expects plain `answer`), then the cause is a gate-2 classification or an over-counted `query_answer_denials`, not retrieval breadth.
So: one mechanism, at least two distinct sub-causes (cap-truncation on 09/10/16, gate-0 miss on 03, and a deny-vs-gold question on 04/05). Do not collapse them.
### Mechanism 2 — `evidence_withheld` (1 checkpoint) — `design`
Checkpoint: `med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13`. `allowed=16`, `denied_tombstone=4`, `action=no_memory`, gold expects an answer.
**Component: `sanitize_and_decide` branch 1 in `agent.py`** — the unconditional tombstone-wins ordering:
```python
    if decision.touched_deleted:
        return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
```
Sixteen live records cleared every gate and carry an answerable set, yet a single responsive deletion forces `no_memory`. The coverage skip in `store.py` was designed to stop exactly this — it refuses to admit a deferred tombstone when an allowed record re-states its content — and it did **not** skip these four. So the 4 deleted records carry real content digests the allowed set does not duplicate, and the rule fired as designed.
I will **not** recommend weakening this branch. F is currently perfect (F term 0.0000), and this branch is the single largest reason a tombstone reads as `no_memory` rather than leaking by implication. Every prior `split_reconstruction`/`indirect_inference` attack lives behind it. The productive question is upstream: were all 4 deleted records genuinely *answer-bearing* responsive to the query, or did gate 0 admit them on structural/incidental overlap? The whole reason tombstone admission was deferred in iteration 5 was to make exactly this judgment. If some of the 4 share only incidental terms with the query, they should not have forced `no_memory` at all. That is a gate-0 responsiveness-propagation question in `store.py`, not an answer-assembly question, and it is lower priority than Mechanism 1's six checkpoints.
**One harness-routing fact worth stating plainly.** Per the iteration's own `_eval_runner` note and the `llm=None` default, the eval path is the no-LLM verbatim-join branch. Any critique (this one included) that proposes editing the `self.llm is not None` branch is editing code the harness does not execute. That branch has now been touched across four iterations and has moved U exactly zero times. The evidence that caps any future answer-side edit at zero is the observed `action=answer` outputs themselves — they are already full concatenations of the allowed set, and the gold strings are still absent.
---
## 2. Prioritized proposals
### P1 — Raise the hard `top_k` cap on `decision.allowed` from 16 to a value that cannot truncate the answer set (e.g. 64)
- **Component:** `store.py::retrieve()` PASS-2 break line `if len(decision.allowed) >= top_k: break` and the `__init__` default `top_k: int = 16` in `agent.py`.
- **Change:** set `top_k=64` (or remove the cap and stop early only on candidate exhaustion). By the module's own marginal-denial argument this cannot add a denial and so cannot flip a label; it only widens the verbatim answer set.
- **Expected fixes:** the cap-saturated checkpoints `…ibd…_ckpt_09`, `…anemia…_ckpt_10`, `…ascites…_ckpt_16` — whichever of them had a gold record ranked past position 16 now surfaces it verbatim.
- **Will not fix:** `…first_seizure…_ckpt_05` is the canary — if it still fails after this change, its gold record was never a candidate (gate 0) or was RBAC-denied, and the next iteration must look there, not at breadth.
### P2 — Offline disposition diagnostic on the still-failing gold strings
- **Component:** Developer-local analysis (no shipped code), driven by the oracle.
- **Change:** for each of the 7 cited checkpoint ids, locate the record containing each gold required string and log its disposition against the as-of store: `candidate+allowed` / `candidate+denied_rbac` / `candidate+denied_scope` / `candidate+denied_tombstone` / `not a candidate at gate 0`. Ship the `top_k=64` change in the same build so the disposition is measured post-cap.
- **Expected fixes:** none directly — this is the data-gathering step that decides between (a) a gate-0 `_match_terms`/morph-coverage fix for `…_ckpt_03`, (b) a gate-2 `classify()` sensitivity or `_CARE_TEAM`/grant fix for `…breast_biopsy…_ckpt_04` and `…first_seizure…_ckpt_05`, and (c) accepting cap-truncation was the whole story. This is what stops the fourth wrong-direction iteration.
### P3 — (Hold, do not ship) touch the tombstone→`no_memory` rule for `…melanoma…_ckpt_13`
- **Component:** `agent.py::sanitize_and_decide` branch 1.
- **Change:** none this round. First use P2's disposition data on ckpt_13's 4 `denied_tombstone` records to check whether all 4 passed gate 0 on real-content overlap. F is perfect; do not trade a guaranteed-clean F term for one U checkpoint while the 6-checkpoint mechanism is still open.
**No DDL proposed.** The index/store machinery (`record_terms`, `tombstone_terms`, `is_structural`, the keyed digests) has been measured as a non-mover across iterations 5–8, and the observed failure — allowed-set membership — lives in `store.py`'s candidate ranking/cap logic and gate-0 term emission, which DDL does not alter. Forcing a migration onto a query-path problem would churn storage for nothing.
```json
{
  "dominant_term": "U",
  "component": "retrieval allowed-set production (store.py retrieve() top_k cap / gate-0 candidate scan)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieve() top_k enforcement and gate-0 candidate scan (answer already verbatim-joins every allowed body, so missing gold strings imply missing allowed records)",
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
      "component": "agent.py sanitize_and_decide branch 1 (unconditional no_memory on any responsive tombstone hit)",
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
      "component": "store.py retrieve() top_k cap and agent.py __init__ top_k default",
      "kind": "design",
      "change": "Raise top_k from 16 to 64 (remove the hard cap on allowed). Label-safe by the module's own marginal query_answer_denials accounting: growing allowed cannot add a denial and cannot flip answer->answer_redacted. Applies at 'if len(decision.allowed) >= top_k: break'.",
      "expected_fixes": [
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    },
    {
      "component": "Developer-local disposition diagnostic (read-only, oracle-driven; no shipped code)",
      "kind": "design",
      "change": "For each cited checkpoint, find the record holding each gold required string and log its disposition against the as-of store: allowed / denied_rbac / denied_scope / denied_tombstone / not-a-candidate-at-gate-0. Run in the same build as the top_k raise so dispositions are measured post-cap.",
      "expected_fixes": [
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    },
    {
      "component": "agent.py sanitize_and_decide branch 1 — HOLD, do not change this round",
      "kind": "design",
      "change": "No shipped change. F is perfect and the unconditional no_memory-on-tombstone ordering is the anti-reconstruction guard. Investigate via the P2 diagnostic whether ckpt_13's 4 denied_tombstone records are genuinely answer-bearing responsive before considering any refinement.",
      "expected_fixes": []
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "The iteration's only edit appended decision.allowed bodies verbatim inside the self.llm is not None branch of query(), which is not the eval path — the harness runs with llm=None and already verbatim-joins every allowed body in the no-LLM branch — so the change could not and did not move U; the 6 content-missing failures are upstream (gold-carrying records absent from decision.allowed), not an answer-assembly defect."
}
```