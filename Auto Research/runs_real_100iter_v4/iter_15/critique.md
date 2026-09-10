## Critique
### 0. Regression Verdict
**NOT THE CAUSE** — and that is precisely the problem. This iteration added a `rescue_missing_logistics` mechanism, but it is placed **only in the `self.llm is None` branch** of `GateMemAgent.query` (agent.py lines ~570–580). The GateMem evaluation harness supplies an LLM callable, so the actual run takes the `self.llm is not None` path, which **never calls the rescue**. The six content‑missing failures persisted unchanged because the new code was never executed. The tie is not a regression from the edit—it is a confirmation that the edit is **inert**. Reverting it would remove no working behavior, but keeping it as‑is wastes an iteration; the correct fix is to move the rescue into the LLM branch (or run it after answer assembly regardless of LLM presence).
### 1. Mechanism Census
All six failing checkpoints share one measured mechanism: `answered_but_content_missing` — the retrieval returned allowed rows, the agent produced a non‑empty answer, but the gold logistics content is absent. No tombstone fired (`denied_tombstone=0`), and `allowed>0` for every one. This is a **design** failure of the retrieval/answer pipeline, not an infrastructure crash.
Crucially, three checkpoints (`013`, `020`, `021`) have `denied_rbac=0` and `denied_scope=0` — the gold content must live in a record that **never entered `decision.allowed`**, either because it ranked below the `top_k=16` cap or was skipped by the gate‑0 relevance test. The rescue mechanism was meant to find those by a content scan, but it did not run. The remaining three (`010`, `011`, `019`) have `denied_rbac>0`; for `010` this produced a label flip (`answer_redacted` instead of `answer`), while `011`/`019` still returned `answer` (so their denials were incidental, not answer‑bearing). For these, the gold is also likely missing from `allowed`, but the cause may additionally involve spurious RBAC denials—a policy‑grant issue that the rescue cannot fix.
### 2. Component Attribution
- **Primary component: answer assembly / retrieval recall** — the LLM path joins `decision.allowed` verbatim but never scans the full shard for missed logistics records. The rescue was written for the no‑LLM branch and is therefore dead code under the eval harness.
- Secondary: **retrieval top_k cap/ranking** for the three clean `denied_rbac=0` cases where the gold record is simply not in the allowed top‑16.
- Tertiary (for `010`): **RBAC grants** causing a spurious `answer_redacted` label.
### 3. Evidence
Checkpoint IDs 010, 011, 013, 019, 020, 021 all show `allowed>0` and `denied_tombstone=0`, with non‑empty answers lacking required tokens. The failure is not caused by a lookup or schema gap—no mechanism points to DDL.
### 4. Prioritized Proposals
1. **Move `rescue_missing_logistics` into the LLM path** (design change to `agent.query`). After the LLM answer and the existing verbatim `Details:` append, call the rescue and append any additionally cleared bodies. This directly addresses all six checkpoints if the gold record exists anywhere live and passes gates. It is safe for `A`/`F` because the rescue applies the identical role×sensitivity×scope checks as `retrieve()`. Expected to fix the three `denied_rbac=0` checkpoints (`013`, `020`, `021`) and likely help `011`/`019` as well.
2. **For checkpoints with `denied_rbac>0`** (010, 011, 019), add a diagnostic log of `denied_rbac` record snippets to the next run, so we can verify whether those denials are answer‑bearing. If the gold is genuinely being withheld by an overly‑strict grant, adjust `DEFAULT_ROLE_GRANTS` or the sensitivity classifier. This is a separate mechanism from content recovery and should not be conflated.
3. Do not raise `top_k` again — prior iterations measured that as a regression (iter 12). The rescue, once moved, will sidestep the cap without relaxing the ranked hard stop.
```json
{
  "dominant_term": "U",
  "component": "answer assembly / retrieval recall (rescue not invoked on LLM path)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "answer assembly / retrieval recall",
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
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "agent.query – LLM answer path",
      "kind": "design",
      "change": "Move the rescue_missing_logistics call so it runs after the LLM answer and the Details: join, regardless of whether self.llm is set. Currently it only executes in the self.llm is None branch, which is not the path used by the evaluation harness.",
      "expected_fixes": [
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03",
        "possibly 011 and 019"
      ]
    },
    {
      "component": "DEFAULT_ROLE_GRANTS / sensitivity classifier",
      "kind": "design",
      "change": "Add a run diagnostic that logs deny_rbac snippets for the checkpoints with denied_rbac>0; if those denials are answer-bearing, expand the relevant role grants (do not touch confidential classification if that would leak). This addresses the label flip on checkpoint 010.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "The rescue_missing_logistics mechanism was added only to the no-LLM branch of agent.query, so it never executed under the LLM-based evaluation harness; the six content-missing failures persisted unchanged, producing a tie rather than a gain."
}
```