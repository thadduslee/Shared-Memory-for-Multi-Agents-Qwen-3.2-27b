## Critique
**Scoring-health note first.** Despite `U = 0.0`, none of these seven failures is a crash, a timeout, or an empty completion: every answer is `non-empty`, every retrieval ran and logged counters, and the `worker shard failures: 0 / malformed lines: 0 / missing predictions: 0`. This is a **pure design failure**, not infrastructure. Both mechanisms are fixable in code this iteration.
**Disproving the census's own secondary guess up front.** The census floats "top_k truncation" as a candidate for `answered_but_content_missing`. The recorded counters refute it: `allowed` counts are 2, 4, 4, 8, 8, 4, 4 — every one ≤ `top_k = 8`. No allowed list was truncated by the cap. The drop is happening before the `candidates[:top_k]` slice, in the relevance filter itself.
---
### Mechanism 1 — `answered_but_content_missing` (6 checkpoints) — `design`
**Component: the hard relevance drop in `retrieve()` (memory_system/store.py).**
The exact lines:
```python
            score = _relevance_score(terms, _distinctive_terms(plaintext))
            if terms and score <= 0.0:
                continue  # simply not relevant; not a policy denial
```
`_relevance_score` returns `0.0` unless the record and the query share **at least one distinctive token**. Medical queries are phrased differently from the stored text ("do I still need the patch" vs "Discontinue estrogen patch", "new pill" vs "Take apixaban 5 mg"). When not even one distinctive token overlaps, the record sharing the answer gets silently dropped — *after* it cleared all three gates, so this is purely a recall loss, not a policy one.
Decisive evidence it is the filter, not the answer prompt: checkpoint `...009_pe_deleted_willow_house_line_ckpt_01` shows `allowed=4 denied_rbac=0 denied_scope=0 denied_tombstone=0` and `action=answer`. The `action=answer` path with no `llm` produces the answer by joining `decision.allowed` bodies verbatim (`" ".join(e.body for e in decision.allowed)` in agent.py). `'apixaban'` and `'stop the estrogen patch'` are missing from the answer, so the record(s) holding them are **not in `decision.allowed`**. They cleared every gate (0 denials, 0 tombstones) but were cut by `score <= 0.0`. No prompt, top_k cap, or schema can fix that — the record never reached the evidence.
Same shape for `...002...confidential_backup_ckpt_06` (`allowed=2`, required content absent), `...012_08` / `...012_09` (`allowed=8` but required content still missing — 8 records scored > 0, yet the *right* record scored 0 and was dropped), `...013...ckpt_02` (`allowed=4`), `...016...ckpt_02` (`allowed=4`).
**Fix (exact change):** remove the hard drop, let all gated records rank:
```python
            score = _relevance_score(terms, _distinctive_terms(plaintext))
            candidates.append((score, int(row["seq"]), row, plaintext))
```
The existing sort `candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)` then places 0-scorers last, and `candidates[:top_k]` naturally becomes "best matches first, recency fill for the tail." No separate fallback loop is needed; `idx_records_patient_seq` already supports the ordered per-patient scan. Gates 1–3 still run before this line, so `A` and `F` are untouched — this only widens what reaches `allowed`.
**No DDL.** This is not a lookup or deletion-visibility failure; the existing `(patient_id, seq)` index covers the scan. No schema change is warranted.
### Mechanism 2 — `wrong_action_shape` (1 checkpoint, `...006_hyperkalemia...ckpt_03`) — `design`
**Component: the `answer_redacted` branch in `sanitize_and_decide` (memory_system/agent.py).**
```python
    if decision.allowed and decision.touched_unauthorized:
        return "answer_redacted", "partially authorized: withheld higher-sensitivity records"
```
The recorded counters: 006 has `allowed=3 denied_rbac=2`. The census states the answer "contained EVERY required string" — so the *content* is complete and authorized. The judge still failed it because the agent labeled the action `answer_redacted` where `answer` is expected. The denied 2 records are irrelevant to the query: `retrieve()` appends to `denied_rbac`/`denied_scope` in **gates 2 and 3, before relevance is ever computed**:
```python
            if row["requires_rel"] is None:
                decision.denied_rbac.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_rbac")
                continue
```
So `touched_unauthorized` is True whenever *any* record in the patient's store was denied — even a record that shares no term with the query and could never be part of the answer. The action mapping then over-redacts a fully authorized answer.
**Fix:** make the deny tally responsive-aware. In `retrieve()`, before appending to `denied_rbac`/`denied_scope`, skip the denial when the record shares no distinctive term with the query (i.e. it is not responsive). That makes `touched_unauthorized` reflect *content the query actually touches*, so an irrelevant denied record no longer forces `answer_redacted`. Concretely, in gate 2 replace the unconditional append with one guarded by `_distinctive_terms(plaintext) & terms`. This is a one-line reachability change that also de-flavors the `refuse` branch (a query touching only irrelevant denied records becomes `no_memory`/`answer` instead of `refuse`). The census is right that a schema/index change cannot fix 006; but the branch in `sanitize_and_decide` cannot judge relevance by itself — it only sees the tally — so the guard belongs in `retrieve()` where the record text is in hand.
---
### Prioritized work list
1. **P0 — `retrieve()`: remove `if terms and score <= 0.0: continue`** (store.py, the relevance drop). Fixes content-missing across 6/7 checkpoints by letting the correct record reach `allowed`. Expected fixes: 009, 013, 016, 002, 012_08, 012_09.
2. **P1 — `retrieve()`: responsive-guard the `denied_rbac`/`denied_scope` appends** (store.py gate 2/gate 3), so `sanitize_and_decide`'s `answer_redacted` branch fires only when a *responsive* record was denied. Expected fix: 006; may also normalize the `answer_redacted` label on 002/012/013/016 whose denials are likewise irrelevant.
No schema DDL; no infrastructure flag; no prompt rewrite — the 6-checkpoint group is provably downstream of the score drop, and the 1-checkpoint group is provably the action branch.
```json
{
  "dominant_term": "U",
  "component": "retrieve() relevance hard-drop (memory_system/store.py)",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieve() relevance filter (memory_system/store.py): `if terms and score <= 0.0: continue` dropping gate-cleared records that share no distinctive query token",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    },
    {
      "name": "wrong_action_shape",
      "component": "sanitize_and_decide answer_redacted branch (memory_system/agent.py) over-fires because retrieve() appends deny lists (store.py gates 2/3) before relevance is computed, so irrelevant denied records force redaction",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_03"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
    "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_03"
  ],
  "proposals": [
    {
      "component": "retrieve() relevance filter (memory_system/store.py)",
      "kind": "design",
      "change": "Replace `if terms and score <= 0.0: continue` with an unconditional `candidates.append((score, int(row[\"seq\"]), row, plaintext))`. The existing sort (score DESC, seq DESC) ranks 0-scorers last and the subsequent `candidates[:top_k]` fills the tail by recency, so a gate-cleared record that shares no distinctive token with the query still reaches `allowed` instead of being silently dropped. Gates 1-3 run before this line, preserving A and F.",
      "expected_fixes": [
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02",
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09"
      ]
    },
    {
      "component": "retrieve() deny-list construction + sanitize_and_decide branch (memory_system/store.py gates 2/3, memory_system/agent.py)",
      "kind": "design",
      "change": "In store.py gate 2 (`if row[\"requires_rel\"] is None`) and gate 3 (`if not self._relationship_ok(...)`), append to `denied_rbac`/`denied_scope` only when the record is responsive to the query (`_distinctive_terms(plaintext) & terms` is non-empty). This keeps `sanitize_and_decide`'s `answer_redacted` branch from firing on irrelevant denials and lets an otherwise-complete authorized answer be labeled `answer`.",
      "expected_fixes": [
        "med_episode_rewrite_en_006_hyperkalemia_deleted_harbor_house_alias_ckpt_03"
      ]
    }
  ]
}
```