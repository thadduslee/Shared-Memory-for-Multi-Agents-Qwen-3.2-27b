## Critique — round on iteration 4 (U=0.5, MGS=0.4706)
### What the measurement table does and does not say
The algebra names U the dominant term with a marginal gain of +0.4706, and the inlined design doc is itself an iteration-4 attack on U. That is not a finding. The finding is the mechanism census below: three distinct mechanisms, all still present after iteration 4's changes, all in the `none` attack class, none of them infrastructure (scoring health is clean: 0 malformed, 0 missing, 0 shard failures, local pass rate 1.000). The census is the round's real content and I address it in full, largest group first.
A precondition-check on every claim below: the run is the **deterministic no-LLM path**. In `agent.py::query`, `answer = " ".join(e.body for e in decision.allowed)` when `self.llm is None`. For that reason the "answer prompt" arm of the content-missing mechanism description does not apply to this run: there is no model phrasing the evidence. Omitted required content is, on this path, *purely a selection phenomenon* — the required string lives in a record that did not reach `decision.allowed`. Everything follows from that.
---
### Mechanism 1 — `answered_but_content_missing` (7 checkpoints) — **design**
Checkpoint ids:
- med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04
- med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05
- med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05
- med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09
- med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16
- med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10
- med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03
Evidence that this is **top-`k` truncation in `MemoryStore.retrieve`, not the candidate scan**: every one of these checkpoints reports `allowed=11` — exactly the agent default `top_k` in `agent.py::GateMemAgent.__init__` (`top_k: int = 11`) — and the content-missing members other than ckpt_04 report `denied_rbac=0`, `denied_scope=0`, `denied_tombstone=0`. So retrieval demonstrably found 11 records, cleared all three gates, and stopped at the cap. The record carrying the required string ("Friday April 4 at 2:00 PM EEG", "Monday June 15 at 1:00 PM pharmacist call", "no lodge backup", "take spironolactone Friday morning") was responsive and authorized but ranked 12th or lower and was never examined. The break is explicit in `store.py::retrieve`, PASS 2:
```python
decision.allowed.append(
    Evidence(...)
)
self._log(checkpoint_id, requester_id, record_id, "allow")
if len(decision.allowed) >= top_k:
    break
```
The iteration-4 hypothesis was that raising the *agent default* from 8 to 11 would capture "required sparse logistics records ranked out of the default `top_k=8`". It did not. All seven still saturate the new cap exactly, which is direct evidence the gold records were not merely the 9th, 10th, or 11th most relevant — they are being out-ranked by chatty narrative rows that reuse the same vocabulary with higher overlap, under the rank `candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)` where `item[0]` is total overlap including structural digit-run and contact-tag digests.
The pinned contract is `test_top_k_is_a_hard_cap_on_cleared_evidence` (20 relevant records, `assert len(allowed) == 8`), and the docstring in `retrieve` is explicit that removing the cap halved U in run-8cf58d33b311. I am not proposing to remove the cap. Two bounded levers remain:
1. Raise the agent default again (11 → ~16). This is test-safe: no local test pins the *agent* default, and the signature default `top_k: int = 8` in `retrieve` is untouched. The label-flip risk that made run-8cf58d33b311's 8→21 fail is now bounded — see Mechanism 2's fix, which prevents incidental denials from flipping labels.
2. Change the tie-break for the sparse-logistics case: a logistics record shares the `__contact__` tag and digit-runs with the query, but those are the *structural* terms `store.py` already separates at gates 2/3 (`_is_structural_term`, `overlap - structural >= 2`). The ranking, however, uses `item[0]` = total overlap, so a record whose only real content overlap is 1 term and whose remaining overlap is digit-runs/contact still ranks below three chatty rows sharing 2 real words each. Ranking live candidates by *real content overlap* first (`total - structural`) — the same quantity gate 2 already computes — would promote exactly the records this mechanism keeps losing, without touching the cap semantics.
---
### Mechanism 2 — `wrong_action_shape` (1 checkpoint) — **design**
Checkpoint: med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09
Observed: `action=answer_redacted` (expected `answer`), answer non-empty and — per the census — containing **every required string**; retrieval `allowed=11`, `denied_rbac=10`, `denied_scope=0`, `denied_tombstone=0`.
The census locates this in `sanitize_and_decide` in `memory_system/agent.py`. With respect to the census, the branch logic there is *not* the root: look at what feeds it. `Decision.query_answer_denials` is incremented at gates 2 and 3 in `store.py::retrieve`:
```python
if row["requires_rel"] is None:
    decision.denied_rbac.append(record_id)
    # one incidental shared content token must not count as answer-bearing
    if overlap - structural >= 2:
        decision.query_answer_denials += 1
```
The iteration-4 design raised the threshold to `>= 2` and added structural-subtraction precisely so "additional evaluated-and-denied records no longer force `answer_redacted` unless they carry ≥2 content words in common". That did not rescue ckpt_09: with `denied_rbac=10`, each of the 10 denied records individually shares ≥2 real content terms with the bi-opsy query (they are the same appointment/vocabulary as the allowed 11), so `qad >= 1`, `sanitize_and_decide` correctly reads "allowed and withheld" and emits `answer_redacted`:
```python
if decision.allowed and withheld:
    return (
        "answer_redacted",
        f"partially authorized: {withheld} responsive record(s) withheld",
    )
```
The branch is behaving as designed; the *input* `qad` is wrong. Ten records were denied, all responsive, all answer-bearing *individually* — yet the Judge confirms the allowed union was complete. That means none of the 10 denied records withholds anything the question actually asks for that the allowed set does not already cover. A per-record content-overlap accumulator cannot express "withholds new content"; it can only express "overlaps the query", and 10 records overlapping the query is not ten withheld answers.
The fix belongs in the accumulator in `store.py`, not in `sanitize_and_decide`: a denial should only count toward `query_answer_denials` when its content adds at least one *content* digest not already present in the union of the allowed records' content digests. Compute the allowed content-term union as PASS 2 fills `allowed` (from `record_terms` where `is_structural = 0`), then score each gate-2/gate-3 denial against that union rather than against the raw query. A denial whose content vocabulary is a subset of the allowed vocabulary withholds nothing answerable → `qad` stays 0 → branch 3 does not fire → label is `answer`, and the Judge's content check passes. This is exactly the surgery the census wants, performed one layer up where the evidence lives.
This fix also makes any further `top_k` raise in Mechanism 1 safe: the extra evaluated-and-denied records this round's design worried about will, under marginal-content accounting, stop flipping `answer` to `answer_redacted`.
---
### Mechanism 3 — `evidence_withheld` / tombstone over-responsiveness (1 checkpoint) — **design**
Checkpoint: med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13
Observed: `action=no_memory`, `allowed=11`, `denied_tombstone=4`, `denied_rbac=0`, `denied_scope=0`. `sanitize_and_decide` correctly fires the tombstone branch first:
```python
if decision.touched_deleted:
    return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
```
The fault is upstream, in `store.py::_is_responsive`'s tombstoned branch, which counts overlap against `tombstone_terms` *without distinguishing structural digests*:
```python
if tombstoned:
    captured = self.conn.execute(
        "SELECT COUNT(*) AS n FROM tombstone_terms WHERE record_id = ?", (record_id,)
    ).fetchone()
    if not captured or int(captured["n"]) == 0:
        return _DEFAULT_OVERLAP
    row = self.conn.execute(
        "SELECT COUNT(DISTINCT term_hash) FROM tombstone_terms "
        f"WHERE record_id = ? AND term_hash IN ({placeholders})",
        [record_id, *query_hashes],
    ).fetchone()
    return int(row[0]) if row else 0
```
The problem is the index itself. `tombstone()` now preserves a tombstoned record's digests in `tombstone_terms` — but `_index_record` hashed the **full** `_match_terms` set, which `_add_contact_tag` and `_distinctive_terms` widen with digit-runs *and* the synthetic `__contact__` tag. Every phone-bearing or contact-word-bearing turn in the episode carries the same `__contact__` digest. So in this "deleted harbor guest line" episode, a query that is itself contact-shaped (an authorized question whose terms include a touching logistics/phone vocabulary) shares the `__contact__` digest with **every** deleted logistics record in the shard, and 4 of them clear the `term_hash IN (...)` probe as "related deletions". Gate 1 then wins over 11 fully-authorized live records and forces `no_memory` on a question the live records could answer.
This is the exact structural/content split the store already implements for gates 2 and 3 (`_is_structural_term`, `_CONTACT_TAG`, `term.isdigit()`) but never applies to gate 1. The fix: a tombstoned record is a *related* deletion only when it shares a **real content** digest with the query. A deletion whose only overlap is digits or the contact tag cannot be reconstructed from the question and must not silence an otherwise-answerable query. The conservative fallback (`_DEFAULT_OVERLAP` when a tombstone has no captured rows) stays, because "we cannot prove what was deleted" still must read as `no_memory`.
This is a deletion-visibility problem, so the schema change is explicit DDL — `record_terms` and `tombstone_terms` must each carry an `is_structural` flag so the tombstone's preserved digests can be filtered at query time:
```sql
-- Migration 5: distinguish structural (digit-run/contact-tag) digests from real
-- content digests in BOTH the live and the preserved term indexes, so gate 1
-- (tombstone responsiveness) can apply the same structural/content split that
-- gates 2 and 3 already apply to query_answer_denials.
ALTER TABLE record_terms
    ADD COLUMN is_structural INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tombstone_terms
    ADD COLUMN is_structural INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_tombstone_terms_structural
    ON tombstone_terms(record_id, is_structural, term_hash);
```
(`_index_record` writes `is_structural = 1` for a digest whose term satisfies `_is_structural_term`; `tombstone()` copies the flag across verbatim; `_is_responsive`'s tombstoned branch counts overlap only over `term_hash IN (...)` rows with `is_structural = 0`, and only falls back to `_DEFAULT_OVERLAP` when the tombstone has *no real-content rows at all*. Since each episode opens a fresh single-file store, the migration runs on an empty index and the backfill burden is nil.)
F-probes (`test_deleted_content_is_not_retrievable_by_anyone`, `test_tombstone_beats_authorization`, `test_deletion_survives_reingestion_of_similar_text`) all query the deleted fact by its own content words, so they retain real-content overlap and still trip `no_memory`. A query that reconstructs a deletion needs its content words, not digits and a contact tag.
---
### Prioritized work list for the Developer
1. **Ranking/top-k truncation in `retrieve` (7 checkpoints, largest U lever).** Mechanism 1's seven checkpoints all saturate `allowed == top_k == 11` with zero denials. Rank live candidates by real-content overlap (`total - structural`) before structural/recency, and raise the agent-default `top_k` from 11 to ~16 (signature default and pinned `top_k=8`/`top_k=1` tests untouched). Expected fixes: the seven content-missing ids.
2. **Marginal-content accounting for `query_answer_denials` (1 checkpoint).** Count a gate-2/gate-3 denial as answer-bearing only if it adds a content digest absent from the allowed union. Expected fix: ckpt_09 010.
3. **Structural-excluded tombstone responsiveness (1 checkpoint).** DDL above + `_index_record`/`tombstone()`/`_is_responsive` to carry and honor `is_structural`. Expected fix: ckpt_13 017.
No infrastructure failures observed; I am not inventing one.
```json
{
  "dominant_term": "U",
  "component": "MemoryStore.retrieve candidate ranking / top_k truncation",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "MemoryStore.retrieve (PASS 2 top_k break + candidate sort) in memory_system/store.py",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "name": "wrong_action_shape",
      "component": "Decision.query_answer_denials accumulator in MemoryStore.retrieve gates 2/3 (memory_system/store.py)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09"
      ]
    },
    {
      "name": "evidence_withheld",
      "component": "MemoryStore._is_responsive tombstoned branch + tombstone_terms schema (memory_system/store.py)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
    "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
    "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
    "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
    "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
    "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
  ],
  "proposals": [
    {
      "component": "MemoryStore.retrieve candidate ranking and agent top_k default",
      "kind": "design",
      "change": "Rank live candidates by real-content overlap (total - structural) before structural-then-recency, so sparse logistics/date records out-rank chatty rows that merely reuse vocabulary; raise the agent default top_k from 11 to ~16 in GateMemAgent.__init__ (signature default in retrieve and pinned top_k=8/1 test semantics untouched).",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "Decision.query_answer_denials accumulator in MemoryStore.retrieve gates 2/3",
      "kind": "design",
      "change": "Count a gate-2/gate-3 denial as answer-bearing only when its real-content digests add at least one digest absent from the union of allowed records' content digests (computed as PASS 2 fills allowed). A denial whose content vocabulary is a subset of the allowed union withholds nothing answerable, so query_answer_denials stays 0 and sanitize_and_decide emits 'answer'.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09"
      ]
    },
    {
      "component": "MemoryStore._index_record / tombstone() / _is_responsive + record_terms & tombstone_terms schema",
      "kind": "design",
      "change": "Add is_structural to record_terms and tombstone_terms (DDL above), tag digests at index time via _is_structural_term, propagate the flag in tombstone(), and in the _is_responsive tombstoned branch count overlap only over is_structural=0 digests so a structurally-only-related deletion (contact tag / digit runs) no longer forces no_memory. Preserve the _DEFAULT_OVERLAP fallback when a tombstone has no real-content rows.",
      "expected_fixes": [
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13"
      ]
    }
  ]
}
```