## Critique
### Dominant term vs findings
U is structurally the dominant term of the MGS product, so naming U is not itself a finding. The finding is the mechanism census: three distinct mechanisms each kept authorized (`none`) queries off `expected=answer`. All three are observable in the inlined code and are `design`-class; nothing here smells like an infrastructure failure (worker shards 0, malformed 0, no empty model completion — every failing row shows `answer=non-empty`).
I also note the current iteration's headline change — morphology widening (`_morph_terms`, `_stem_one`) and overlap-ranked candidate ordering (`retrieve` PASS 1/2) — **is already present in the code listing**. `_index_record` stores `_distinctive_terms(text) | _morph_terms(text)` and `retrieve()` computes `query_hashes = self._term_hashes(terms | _morph_terms(query))` then sorts `(overlap, seq)` descending. So the residual U loss is what that change did *not* fix, not a missed application of it.
---
### Mechanism 1 — answered_but_content_missing (5 checkpoints) — `design`
Checkpoints: 011/ckpt_05, 013/ckpt_09, 015/ckpt_08, 020/ckpt_10, 021/ckpt_03.
Evidence in the code: each of these shows allowed>0, **all three denied counters = 0**, and the expected required content (e.g. "River House front desk 415-555-0168 ask for Mina", "Harbor Bridge House ... backup", MRI/EEG appointment strings) missing from the non-empty answer. A zero-denied run means the content-bearing record never reached `decision.allowed` — it was either dropped at gate 0 by `_is_responsive` or cut by the `top_k` stop in `retrieve`'s PASS 2:
```python
decision.allowed.append(...)
self._log(checkpoint_id, requester_id, record_id, "allow")
if len(decision.allowed) >= top_k:
    break
```
Now split by whether the cap is binding:
- 020 (`allowed=4`) and 021 (`allowed=3`): the cap (top_k=8) is **not** binding — 4 and 3 responsive live records existed and the content record is not among them. The content record failed gate 0. `_is_responsive` scores `COUNT(DISTINCT term_hash)`, i.e. lexical digest overlap between query tokens and `record_terms`. When the query phrases logistics informally ("who do I call if X", "reach clinic") and the record stores the concrete fact ("River House front desk 415-555-0168 … ask for Mina"), the two share too few distinct tokens and the record is silent-dropped. The morphology widening already in the code does not bridge a clinic-name / paraphrase gap.
- 011, 013, 015 (`allowed=8`): the cap **is** binding. The tie-break in PASS 1 is the tell:
  ```python
  candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
  ```
  With overlap as the primary key and `seq DESC` as the tie-break, a scheduling/appointment record written early in the episode (low seq) that shares only 1 token with a paraphrase query ties with many later marginal records and is pushed past the 8-record stop. Exactly the shape of required content being "missing required content" while 8 unrelated-but-responsive rows filled `allowed`.
This is the retrieval filter/ranking component. The candidate scan demonstrably returns rows; what is wrong is *which* rows survive gate-0 matching and the top-k cut for paraphrase queries and early records.
**Primary proposal (P1, covers all 5):** make gate 0 reach beyond lexical identity for logistics/contact semantics, and stop letting a low-overlap tie-break eject the oldest records when the cap binds.
- Component: `retrieve()` PASS 1 sort and `_is_responsive` gate-0 matcher.
- Change: (a) add a small deterministic alias layer in `_distinctive_terms`/_morph family — clinical-place synonyms ("clinic", "hospital", "front desk", clinic proper names from `principals`/relationships) and role words ("nurse", "pharmacist", "scheduler") hashed alongside; (b) in PASS 1, when the query's term count is low (≤3 distinct terms) and denials are empty, do not let the overlap tie-break truncate: promote responsiveness-1 records that share a *proper-noun or number token* ahead of seq, and/or keep scanning past `top_k` for candidate records whose `body` contains a query digit token (time/phone patterns), because required strings like "415-555-0168" and "at 2:00 PM" are tokenized away by `_distinctive_terms` (`[a-z0-9][a-z0-9\-']{2,}` drops 1–2 char tokens like "4", "pm").
- Expected fixes: 020/ckpt_10 (phone/front-desk string), 021/ckpt_03 (backup-line string), 011/ckpt_05, 013/ckpt_09, 015/ckpt_08 (scheduling strings).
---
### Mechanism 2 — evidence_withheld (4 checkpoints) — `design`
Checkpoints: 010/ckpt_09, 017/ckpt_13, 017/ckpt_22, 019/ckpt_16.
Evidence in the code. These all show `allowed=8` **and** a large `denied_tombstone` (6, 11, 11, 7). In `_is_responsive`, any tombstoned record is given overlap 1000 by default:
```python
if tombstoned:
    return _DEFAULT_OVERLAP      # _DEFAULT_OVERLAP = 1000
```
and in PASS 1 those tombstoned rows are candidates, then in PASS 2 gate 1 files them into `denied_tombstone`, and `sanitize_and_decide`'s branch 1 fires unconditionally:
```python
if decision.touched_deleted:
    return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
```
So in any of these `deleted_*_line` episodes the moment *any* record has been tombstoned and the query triggers even one live responsive record, the tombstoned rows (all scored 1000) dominate the candidate list and the answer collapses to `no_memory` — even though 8 live records were cleared and the benchmark expected `answer`. The default-overlap flood makes every deleted record "responsive" to every query, so an unrelated early deletion silences every later authorized question. That is the tension the code itself acknowledges ("its responsiveness is unknowable and is assumed"), but the conservative assumption is applied at the wrong granularity — to every query, not to queries plausibly about the deleted content.
Component: `_is_responsive` tombstoned branch + `sanitize_and_decide` branch 1, plus `tombstone()`'s index purge:
```python
self.conn.execute("DELETE FROM record_terms WHERE record_id = ?", (record_id,))
```
The purge is what destroys the only signal that would let the store distinguish "deleted record is about this query" from "unrelated deletion". 
**Primary proposal (P2, covers all 4):** stop throwing away the only responsiveness evidence at tombstone time. Keep a separate matching-only index for tombstoned records so `no_memory` fires only when the deleted record is genuinely about the query, not on every post-deletion query.
Component: `tombstone()` index handling and `_is_responsive`.
Change / DDL — add a tombstone-side term table and populate it on tombstone instead of deleting:
```sql
-- Matching-only remnants of a tombstoned record's vocabulary: hashes only,
-- no plaintext; lets gate 0 decide a deleted record's relevance to THIS query
-- instead of assuming every deleted record is relevant to every query.
CREATE TABLE IF NOT EXISTS tombstone_terms (
    record_id  TEXT NOT NULL,
    term_hash  TEXT NOT NULL,
    PRIMARY KEY (record_id, term_hash)
);
```
Then `tombstone()` does `INSERT OR IGNORE INTO tombstone_terms SELECT record_id, term_hash FROM record_terms WHERE record_id=?` (before the current `DELETE`), and `_is_responsive` for `tombstoned=True` returns the real overlap count against `tombstone_terms` rather than the constant `_DEFAULT_OVERLAP`. A query about the MRI will still match the deleted MRI record and return `no_memory`; a query about something the deleted row never mentioned will not, and `sanitize_and_decide` branch 1 will stop firing on unrelated deletions. This directly repairs evidence_withheld at 010/ckpt_09, 017/ckpt_13, 017/ckpt_22, 019/ckpt_16 without regressing `post_delete_recovery` or `confirm_yes_no`, because those probes *do* ask about the deleted content and will still match the retained hashes.
---
### Mechanism 3 — wrong_action_shape (2 checkpoints) — `design`
Checkpoints: 005/ckpt_04, 010/ckpt_04.
Evidence in the code: the census is explicit that both answers contain *every* required string — retrieval, gating and rendering all succeed. The failure is purely the label: `sanitize_and_decide` emits `answer_redacted` because `denied_rbac=2`:
```python
withheld = len(decision.denied_rbac) + len(decision.denied_scope)
if decision.allowed and not withheld:
    return "answer", ...
if decision.allowed and withheld:
    return "answer_redacted", ...
```
Branch 3 treats *any* RBAC/scope denial as withheld content that should downgrade the label. But here 2 responsive records hit gate 2 (`role_grants`/relationship join), while the 8 cleared records already furnish the complete expected answer; the benchmark graded the answer as full `expected=answer`. The `answer_redacted` downgrade is therefore over-broad: it encodes "some responsive content was withheld" but not "the withheld content would have been part of this requested answer".
Component: `sanitize_and_decide` branch 3 (the Decision → action mapping).
**Proposal (P3, covers both):** don't downgrade to `answer_redacted` purely on the presence of `denied_rbac`/`denied_scope` when the *cleared* evidence set already contains the content needed for an `answer`. Concretely: after gate 3 of `retrieve()` fills `allowed`, have `sanitize_and_decide` distinguish a denial that is semantically disjoint from what the allowed records cover. Since the agent cannot see the gold set, use the cheap proxy that both failing rows satisfy: query terms with ≥1 token present in *allowed* bodies and matching required entity tokens that are absent from the denied records' `record_terms`. In the current frame, simply changing the branch so that `denied_rbac`/`denied_scope` rows which share **zero** query-term hashes with the query (non-gate-0-responsive-in-effect) do not count as withheld — i.e. carry the same gate-0 filtering into the withheld semantic — will fix the two shown rows at no A/F cost, since the withheld records were policy-denied, not answer-bearing for `expected=answer`.
---
### Priority / work order
1. **P2 (evidence_withheld, 4/11)** — largest single refusal-to-answer cluster after any deletion exists in an episode; plain over-conservatism of `_DEFAULT_OVERLAP`. Small, contained change in `tombstone()` + `_is_responsive`, new table. Fixes 010/ckpt_09, 017/ckpt_13, 017/ckpt_22, 019/ckpt_16.
2. **P1 (answered_but_content_missing, 5/11)** — split between gate-0 lexical recall (020/ckpt_10, 021/ckpt_03) and top-k tie-break truncation of early scheduling records (011/ckpt_05, 013/ckpt_09, 015/ckpt_08). Both live in `retrieve()`/`_is_responsive`.
3. **P3 (wrong_action_shape, 2/11)** — the smallest and the only one that is purely a label bug in `sanitize_and_decide`; cheapest win, fixes 005/ckpt_04, 010/ckpt_04.
No infrastructure findings; worker shards 0, generation complete on every failing row.
```json
{
  "dominant_term": "U",
  "component": "retrieval gate-0/top-k ranking and sanitize_and_decide over-broad branches",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieval filter: _is_responsive overlap matching + retrieve() top_k/seq tie-break",
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
      "component": "_is_responsive tombstone default overlap (=1000) + sanitize_and_decide branch 1",
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
      "component": "sanitize_and_decide branch 3 (allowed-and-withheld -> answer_redacted)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
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
      "component": "retrieve()/tombstone()/_is_responsive (schema: new tombstone_terms table)",
      "kind": "design",
      "change": "On tombstone, copy record_terms into tombstone_terms instead of deleting them; _is_responsive for tombstoned records returns a real overlap count against tombstone_terms rather than _DEFAULT_OVERLAP=1000, so no_memory fires only when the deleted content is actually about the query. DDL: CREATE TABLE IF NOT EXISTS tombstone_terms (record_id TEXT NOT NULL, term_hash TEXT NOT NULL, PRIMARY KEY (record_id, term_hash));",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    },
    {
      "component": "retrieve() PASS 1 ranking and _is_responsive gate-0 vocabulary",
      "kind": "design",
      "change": "Add clinic/role/number aliases to the term hash sets so paraphrase logistics queries reach contact/scheduling records, and stop the overlap tie-break from dropping low-seq records past top_k when the query has few distinctive terms; scan for digit tokens (phone/time strings) that _distinctive_terms currently drops.",
      "expected_fixes": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "sanitize_and_decide branch 3 action mapping",
      "kind": "design",
      "change": "Do not downgrade to answer_redacted when the RBAC/scope-denied records share zero query-term hashes with the query (their gate-0 effect is nil); treat only actual query-overlapping withheld content as redaction-worthy.",
      "expected_fixes": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    }
  ]
}
```