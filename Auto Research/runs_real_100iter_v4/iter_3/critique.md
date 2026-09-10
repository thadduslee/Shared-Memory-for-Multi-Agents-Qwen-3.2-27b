## Critique — round 4 (U dominates: U=0.4444, A=0, F=0; all 10 failures on `standard_retrieval`, attack `none`)
**A critical fact about the current code before anything else.** The iteration-3 design proposes to *wire* `Decision.query_answer_denials` into `sanitize_and_decide`, describing that field as dead code. The implementation inlined above shows that wiring is **already present**:
```python
qad = decision.query_answer_denials
withheld = qad if qad is not None else len(decision.denied_rbac) + len(decision.denied_scope)
```
That is exactly the proposed edit, verbatim. So if the census was measured against this code, then the "wire the field" hypothesis is *falsified* — the field is wired and ckpt `..._010_..._ckpt_04` **still** returns `answer_redacted` on a complete answer. Any plan that tells the Developer to add this wiring again would be rewriting code that already works — forbidden. And it is falsified for a nameable reason: the label does not depend on `qad` being *read*, it depends on `retrieve()` deciding that the 2 RBAC denials **were** answer-bearing. That decision happens in `store.py`:
```python
if row["requires_rel"] is None:
    decision.denied_rbac.append(record_id)
    if overlap - structural > 0:
        decision.query_answer_denials += 1
```
For ckpt_04 both RBAC denials incremented the count (`overlap - structural > 0`), so `withheld >= 1` and `sanitize_and_decide` correctly — by its own contract — returns `answer_redacted`. The store judged "any real-content overlap" as answer-bearing; the Judge judged otherwise. That threshold is the fault site, not the agent branch. I address this below under Mechanism 1.
---
### Mechanism 1 — `wrong_action_shape` — 1 checkpoint (ckpt_04)
*Component: `store.py::retrieve` answer-bearing classification condition (`overlap - structural > 0`) under gate 2. Design.*
This is the census's smallest bucket but the one the iteration-3 design specifically claimed it would fix, and reading the source shows the claim is already spent. The action label is produced by `sanitize_and_decide`, which is now correct (field wired). The reason label is still `answer_redacted` is that `retrieve()` told it the 2 RBAC denials were answer-bearing. For a checkpoint whose every required string is present in `allowed`, incrementing `query_answer_denials` on *any* positive content overlap is too coarse: a single shared content token (e.g., `eeg`) between the query and an unauthorized record that the gold answer does **not** require is enough to redact the whole otherwise-complete answer.
Evidence: `med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04` — `allowed=8`, `denied_rbac=2`, `action=answer_redacted`, answer non-empty and complete per the Judge.
Do **not** touch `sanitize_and_decide` again. The change belongs in the gate-2 (and gate-3) increment conditions in `retrieve()` — raise the bar so a denied record only counts toward `query_answer_denials` when it shares *more than one* substantive content term, or when its overlap is at least as large as the record that actually satisfied the query.
---
### Mechanism 2 — `answered_but_content_missing` — 5 checkpoints (ckpt_05, ckpt_09, ckpt_08, ckpt_10, ckpt_03)
*Component: `MemoryStore.retrieve()` PASS-2 candidate cap and the join-answer path it feeds. Design.*
Every one of these shows `allowed=8`, zero denials, `action=answer`, non-empty answer with required content missing. Where the agent runs with the default `llm=None` (`agent.py`), the entire `allowed` set is joined verbatim:
```python
elif action in {"answer", "answer_redacted"}:
    answer = " ".join(e.body for e in decision.allowed)
```
In that path a missing required string is **proof the record carrying it never reached `allowed`** — no prompt can invent it. The required content is logistics/appointment/contact-shaped: `"Friday April 4 at 2:00 PM EEG"`, `"portal okay, River House front desk 415-555-0168 ask for Mina"`, `"Harbor Bridge House … direct mobile first"`. These records share few *content* terms with a natural-language query, so under the PASS-2 sort `(overlap desc, structural desc, seq desc)` they rank below chatty dialogue rows that share more vocabulary, and the hard cap at `top_k=8` (`if len(decision.allowed) >= top_k: break`) drops them before `allowed` fills.
The census hedges "answer prompt or the top_k truncation — not the candidate scan, which demonstrably returned rows." I can resolve the hedge from the source: in the no-LLM branch the answer prompt *is* the `allowed` join, so the only way content is missing is truncation/ranking. The candidate scan returned rows, but not *the required* rows. Where the harness supplies an LLM (`rendered = self.llm(...)` in `query()`), omission of an allowed record *would* be an answerer fidelity issue with no prompt code in this repo — that half is potentially `infrastructure`. The measured `evidence_used` values (1–4 of 8 allowed) are consistent with an LLM writing from a subset, so I cannot rule the answerer half out; but the deterministic default path already proves the retrieval half is real.
Evidence: `..._011_...ckpt_05`, `..._013_...ckpt_09`, `..._015_...ckpt_08`, `..._020_...ckpt_10`, `..._021_...ckpt_03`.
---
### Mechanism 3 — `evidence_withheld` — 4 checkpoints (ckpt_09, ckpt_13, ckpt_22, ckpt_16)
*Component: gate-0 tombstone responsiveness default in `store.py::_is_responsive`, consumed by branch 1 of `sanitize_and_decide`. Design; deletion-visibility problem.*
Look at the counters: ckpt_13/22 show `denied_tombstone=15` with `allowed=8` and action `no_memory` on checkpoints whose gold answer is **not** a deletion. Where does a tombstone count of 15 come from? `_honor_deletion_request` only tombstones records matching the deletion request with `overlap >= 2`. But once tombstoned, every one of those records answers `_is_responsive` with the fail-open default:
```python
if tombstoned:
    return _DEFAULT_OVERLAP
```
`_DEFAULT_OVERLAP = 1000`, so **every** tombstoned row in the patient shard clears gate 0, is denied as tombstone, and makes `decision.touched_deleted` true. Branch 1 of `sanitize_and_decide` then wins over 8 allowed live records:
```python
if decision.touched_deleted:
    return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
```
That is a deletion-visibility failure: content that still lives in `allowed` is suppressed because an *unrelated* deletion in the same shard is assumed responsive to everything. The reason the terms were purged is legitimate privacy (the docstring's "index goes with the body"), but the cost is that the store can no longer tell "this query is about what was deleted" from "this query concerns live records and a different deletion happened nearby." The fix must preserve shredding while recording, at tombstone time, enough term evidence to answer future responsiveness without resurrecting the body.
Evidence: `..._010_...ckpt_09` (allowed=8, denied_tombstone=6, no_memory), `..._017_...ckpt_13` & `ckpt_22` (denied_tombstone=15, no_memory), `..._019_...ckpt_16` (denied_tombstone=9, no_memory).
---
## Prioritized work list
1. **Fix the answer-bearing classification in `retrieve()` (Mechanism 1, 1 checkpoint).** In `store.py`, gate 2 and gate 3: replace `if overlap - structural > 0:` with a requirement of more than one substantive content term (e.g., `if overlap - structural >= 2:`). This is the actual lever that flips `..._010_...ckpt_04` from `answer_redacted` to `answer`. Expected: +1 U. No A/F impact — denied lists and audit rows are unchanged.
2. **Stop unrelated tombstone denials from shadowing complete live answers (Mechanism 3, 4 checkpoints).** This is a deletion-visibility problem; propose explicit DDL:
```sql
-- remember, at tombstone time, WHICH content the deletion actually covered,
-- so a later query can tell responsive-deletion from unrelated-deletion.
CREATE TABLE IF NOT EXISTS tombstone_terms (
    record_id TEXT NOT NULL REFERENCES tombstones(record_id) ON DELETE CASCADE,
    term_hash TEXT NOT NULL,
    PRIMARY KEY (record_id, term_hash)
);
```
Populate it inside `tombstone()` — before `DELETE FROM record_terms` — by copying that record's term digests into `tombstone_terms` (Hmac-digested under the same per-store key; the plaintext never reappears). Then change `_is_responsive`'s tombstoned branch from `return _DEFAULT_OVERLAP` to counting overlap against `tombstone_terms` so a tombstone only forces `denied_tombstone` (and thus branch-1 `no_memory`) when the query genuinely shares terms with the content that was deleted. A query about live logistics rows that do not mention the deleted content stops tripping `touched_deleted` and returns `answer`. Expected: −4 no_memory failures on `..._013_10? (ckpt_09)`, `..._017 ckpt_13/22`, `..._019 ckpt_16`. This is the largest single lever.
3. **Ensure required logistics/contact records survive the `top_k` cap (Mechanism 2, 5 checkpoints).** In `retrieve()` PASS 2, the cap `if len(decision.allowed) >= top_k: break` truncates by overlap-rank and consistently drops the *sparse-term* logistics records that carry the gold strings. Rather than raising the global cap (which run-8cf58d33b311 showed inflating denial lists and halving U — and with `query_answer_denials` now wired, the redaction hazard is reduced but not gone), reserve a small allowance: keep collecting beyond `top_k` only while an un-collected live candidate carries `_CONTACT_TAG` or a digit-run-heavy term set and the answer would otherwise contain no contact row. Concretely: break the loop only when `allowed >= top_k` **and** `any(e.body for logistics-tagged)` … but that uses body text — simpler and safer is to run PASS 2 once more with the cap lifted by a fixed margin *only for records that share a substantive (non-structural) content term with the query*, since such records cannot be incidental and cannot inflate denials (they already cleared all gates). Expected to restore the specific gold strings at `..._011 ckpt_05`, `..._013 ckpt_09`, `..._015 ckpt_08`, `..._020 ckpt_10`, `..._021 ckpt_03`, binding to whatever record carries each string. If the harness is injecting an LLM for these checkpoints, confirm which path fires; the deterministic no-LLM path is the verifiable one.
Recheck priorities against the algebra: with U dominant and A=F=0, fixing Mechanism 2 (5 checkpoints) and Mechanism 3 (4) and Mechanism 1 (1) together is the whole failing set. No index/schema change helps Mechanism 2 (records exist but are capped out) and none helps Mechanism 1 (all gating is correct; only the answer-bearing threshold is wrong). The only mechanism with a clean DDL story is 3.
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve PASS-2 top_k cap + tombstone gate-0 fail-open default",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "MemoryStore.retrieve() PASS-2 candidate collection (top_k=8 hard cap, overlap-rank ordering)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "store.py _is_responsive tombstoned default (_DEFAULT_OVERLAP=1000) + sanitize_and_decide branch 1",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    },
    {
      "name": "wrong_action_shape",
      "component": "store.py retrieve() gate-2/gate-3 answer-bearing classification (overlap - structural > 0)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "store.py retrieve(), gates 2 & 3",
      "kind": "design",
      "change": "Raise the answer-bearing threshold from `overlap - structural > 0` to `overlap - structural >= 2` when incrementing query_answer_denials, so a single incidental shared content token cannot redact a complete answer.",
      "expected_fixes": ["med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"]
    },
    {
      "component": "tombstone responsiveness (deletion visibility)",
      "kind": "design",
      "change": "Add table tombstone_terms(record_id, term_hash); copy a record's term digests into it inside tombstone() before purging record_terms; change _is_responsive's tombstoned branch to count overlap against tombstone_terms instead of returning _DEFAULT_OVERLAP, so only genuinely related deletions force no_memory.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    },
    {
      "component": "MemoryStore.retrieve() PASS-2 top_k cap",
      "kind": "design",
      "change": "Lift the break `if len(decision.allowed) >= top_k` for live candidate records that share a substantive (non-structural) content term with the query, so sparse-term logistics/contact records that carry the gold content survive the cap without widening the denied lists (which stay bounded by gate outcomes, not by allowed count).",
      "expected_fixes": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    }
  ]
}
```