## Critique — Iteration 17 tied at MGS=0.5882
**0. Regression verdict first.** This round did not fall; it tied. Iteration 17's only functional edit — un-bounding the candidate supply in `store.py::rescue_missing_logistics` — moved **none** of U/A/F (all `+0.0000`), and the same six checkpoints that failed iterations 13/15/16 failed again. A tie is not evidence the edit harmed anything. So the verdict is `not_the_cause` — with this evidence:
Take `med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03`: `allowed=13`, `denied_rbac=0`, `denied_tombstone=0`. The cap is 16, so the top-k stop is **not binding** at 13 — every responsive, cleared, live record `seq <= as_of` is inside `decision.allowed`, and the champion code appends **every** allowed body verbatim on both answer paths:
- LLM branch: `answer += "\nDetails: " + " ".join(e.body for e in decision.allowed)`
- no-LLM branch: `answer = " ".join(e.body for e in decision.allowed)`
The required gold for 021 — *"Harbor Bridge House (?:backup )?(?:second|only if direct mobile fails) | direct mobile first, Harbor Bridge House backup"* — is absent from the assembled answer even though every live cleared record was joined verbatim. A rescue scan that ignores `top_k` would add nothing on 021 because nothing was `top_k`-starved there. Item 011 (`allowed=14`, also below cap, `denied_rbac=2`) proves the same for its missing `Friday April 4 at 2:00 PM EEG` / `Tuesday April 8 at 7:15 PM MRI` content. For the three at the cap (`allowed=16`: 013/019/020), the rescue exists precisely to reach records the cap starves — and the tie says it reached nothing that changed the answer. The edit was net-zero because there was nothing in net for it to catch.
Five consecutive ties at the same six checkpoints, with this round's change specifically enlarging the rescue's reach, is the measured falsification of the candidate-reach hypothesis: **the missing gold is not sitting in a live, cleared, `seq <= as_of` record that any scan widening can reach.** The change is not the cause; the pre-existing plateau is.
**1. Mechanism census.** Exactly one mechanism was observed: `answered_but_content_missing` (6 checkpoints — 010/011/013/019/020/021). All six share: `allowed > 0`, `denied_tombstone = 0`, non-empty answer already emitted, full verbatim join of every allowed body present, plus the unbounded rescue that just tied. For a checkpoints-only question (kinematics/rate problems, response-metric questions), where the expected numeric value isn't in *any* allowed/denied body under any askable metric or strategy description, no candidate-widening can surface it because it does not exist in the corpus.
Because every code path that could surface content from a live cleared record is already enabled and still returns the six empties, this mechanism does **not** implicate the answer verifier (which operates only over `decision.allowed`, a set already demonstrated to be exhausted); it implicates the **evidence horizon** — the content is either in rows the requester is *denied* (RBAC/scope) and legitimately cannot receive, or it is outside the visible `seq <= as_of` window, or it exists in no record at all (a benchmark-synthesized gold). Each of those is a harness/curriculum property, not a retrievable-design property. I mark the residual cause `infrastructure` below with the evidence that this round's own experiment established it.
Note: 010/011/019 each carry real RBAC denials (`denied_rbac` = 3/2/2) at the same time as the census counts them `answered_but_content_missing`; if the candidate census shows those denials carry the gold, those three are **unfixable without a leak** — any edit that surfaces them to U costs A. That makes the three-DBG trio a distinct, policy-walled subcase of the same recorded mechanism. I keep them in one bucket because that is how the census bucketed them, but the RBAC denials are the reason the trio must not be "fixed" by widening.
**2. Component + kind per mechanism:**
| Mechanism | Checkpoints | Component | Kind |
|---|---|---|---|
| `answered_but_content_missing` | 010, 011, 013, 019, 020, 021 | Evidence horizon (gold not in live cleared `seq<=as_of` set; 010/011/019 possibly behind RBAC denials) | `infrastructure` for the deny-free four; `design`-**blocked** for the RBAC trio — widening to reach them is an A leak and must not be attempted |
The six are the entire U deficit. If they are horizon-capped, U = 0.6667 is the measured ceiling of this store-and-agent under this benchmark's gold construction, and every iteration spent on retrieval after this one is iteration 6 of a strategy the algebra has now disproved five times.
**3. Actionable proposals.**
**P1 — Observability first. Ship nothing else until the census answers where the gold lives.**
Component: `store.py::retrieve` (inert `Decision.candidate_census`). The debug keys `has_logistics` / `token_in_answer` already exist in the projection tuple. Next iteration must extend the census rows to record, per patient record `seq <= as_of` (live and tombstoned) that carries any content/logistics token, its disposition drawn from the fixed set (`allowed`/`deny_rbac`/`deny_scope`/`deny_tombstone`/`gate0_skip`) **plus** whether it is write-horizon (`seq > as_of`), and then read the six highlighted rows out of `predictions.jsonl`. This is a read-only decoration change, zero branch impact. Expected outcome: for each of 010/011/013/019/020/021 we learn which of "RBAC-truthfully-denied", "post-horizon", "gate-0 skipped non-logistics phrase", or "absent from every record" explains the gold. Three possible verdicts: (a) gold in `deny_rbac` rows → U is capped by A, stop retrieving; (b) gold post-horizon → U is capped by the harness, stop; (c) gold `gate0_skip` live-and-cleared with a non-logistics phrase → that is the one design-fixable residual and the only case warranting further retrieval work.
**P2 — Halt all candidate-widening until P1 lands.** The champion's `rescue_missing_logistics` already scans "the patient's live (non-tombstoned) records as-of `as_of_seq`" and "returns Evidence this requester is already cleared to read (same role x sensitivity x scope policy `retrieve()` applies) whose body carries a logistics token absent from `answer`". Iteration 17 made that scan's candidate supply un-bounded by query overlap. The tie proves enlarging the parameter of this scan does not move U; no further widening of the rescue, the gates, or `top_k` should be proposed. Expected fix: none of the six — this proposal is a cap on wasted iterations.
**P3 — Do not "fix" the RBAC trio.** 010 (3 denials), 011 (2), 019 (2) all fired `answer_redacted` while the benchmark expects plain `answer` with the missing content included. If P1 shows the expected content in the `deny_rbac` set, those three checkpoints are unsatisfiable without an A regression; increase no grant row, do not un-gate. Expected fix: deliberately none for U; the honest next move is to measure A rather than trade it.
No DDL is proposed. All six `denied_tombstone=0` with `allowed>0` — no index or deletion-visibility gap exists; proposing one would be manufacture.
```json
{
  "dominant_term": "U",
  "component": "evidence_horizon",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "evidence_horizon (gold not in any live cleared seq<=as_of record; RBAC trio 010/011/019 potentially behind legitimate denials)",
      "kind": "infrastructure",
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
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
  ],
  "proposals": [
    {
      "component": "store.py::retrieve (inert Decision.candidate_census)",
      "kind": "design",
      "change": "Extend the existing census rows to record, for every patient record seq <= as_of carrying content or logistics tokens, a disposition from {allowed, deny_rbac, deny_scope, deny_tombstone, gate0_skip, post_horizon} plus has_logistics/token_in_answer; keep it read-only and non-branching. Read the six failing rows from predictions.jsonl to classify each gold as RBAC-withheld vs post-horizon vs gate0-skipped-non-logistics vs absent.",
      "expected_fixes": []
    },
    {
      "component": "store.py::rescue_missing_logistics / agent.py answer assembly",
      "kind": "design",
      "change": "Halt all further candidate-widening (rescue scope, gate relaxation, top_k raises). Iteration 17's un-bounded rescue tied on all six checkpoints, proving the gold is not reachable by enlarging the live-cleared scan. Iteration 18 must not spend on retrieval.",
      "expected_fixes": []
    },
    {
      "component": "role_grants / sanitize_and_decide (RBAC policy surface)",
      "kind": "design",
      "change": "Do not increase any grant to surface the denied_rbac content on 010/011/019; if the census places their gold in the denial set, treat those three as U-capped-by-A and leave them unfixed.",
      "expected_fixes": []
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "Iteration 17's un-bounding of rescue_missing_logistics candidate supply is not the cause of the tie -- all three terms are unchanged at +0.0000 and the same six checkpoints failed identically; the evidence (ckpt_021 at allowed=13 < top_k with full verbatim body join, ckpt_011 at allowed=14 similarly) shows the gold was never in a starved or skipped live cleared record for the edit to reach."
}
```