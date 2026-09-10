## 0. The "regression": a tie that bought nothing — NOT a regression, but a dead fix
This iteration scored **exactly** what the champion scored (MGS 0.3889 → 0.3889, U 0.3889 → 0.3889). The diff did not cost one checkpoint of U, A, or F — the failure census is identical to the champion's (same 11 ids, same three mechanisms). So this is **not_the_cause** in the sense that nothing was lost on purpose; but the change also fixed nothing, and the reason it fixed nothing is a concrete, nameable defect in the diff itself, not bad luck:
**Mechanism 3 was left as dead code.** The whole point of `Decision.query_answer_denials` was to stop incidentals from downgrading `answer` → `answer_redacted` on 005/ckpt_04 and 010/ckpt_04. The diff added the field (`store.py`), and `retrieve()` diligently increments it:
```python
# memory_system/store.py, PASS 2, Gate 2 / Gate 3
if row["requires_rel"] is None:
    decision.denied_rbac.append(record_id)
    if overlap - structural > 0:
        decision.query_answer_denials += 1
```
…but the only consumer that was supposed to read it — `sanitize_and_decide` in `memory_system/agent.py` — never looks at it. That function still keys its branch 3 on the **length** of the denial lists:
```python
withheld = len(decision.denied_rbac) + len(decision.denied_scope)
if decision.allowed and not withheld:
    return "answer", ...
if decision.allowed and withheld:
    return ("answer_redacted", ...)
```
`query_answer_denials` is computed and then discarded. Both `wrong_action_shape` checkpoints therefore survived unchanged (`answer_redacted` was returned again with `denied_rbac=1` and `denied_rbac=2` respectively). This is the single clearest reason the intended U gain did not land, and it is verifiable entirely from the inlined source: `agent.py` never reads the field `store.py` spends code producing.
## 1. Mechanism-by-mechanism
### Mechanism A — `wrong_action_shape` (2 checkpoints) — `design`
**Component:** `sanitize_and_decide`, `memory_system/agent.py`, branch 3.
**Checkpoints:** 005/ckpt_04 (`action=answer_redacted`, `denied_rbac=1`), 010/ckpt_04 (`action=answer_redacted`, `denied_rbac=2`).
For both, the observed run says `answer=non-empty` and the census confirms "the answer contained EVERY required string … and the Judge still failed … because the ACTION LABEL was wrong." Retrieval, gating, and rendering all worked. No schema, index, tombstone, or rank change can touch these — only the decision→action mapping. The fix the iteration intended is half-built: `query_answer_denials` exists but is never consulted. This is the cheapest, highest-confidence repair available this round, and it is what the diff was *for*.
**Concrete change** — re-key branch 3 on the answer-bearing denial count, with the back-compat fallback the `Decision` docstring already blesses ("`None` means 'not computed — treat every denial as answer-bearing'"):
```python
# memory_system/agent.py — sanitize_and_decide, replacing the two lines above
qad = decision.query_answer_denials
withheld = qad if qad is not None else len(decision.denied_rbac) + len(decision.denied_scope)
if decision.allowed and not withheld:
    return "answer", f"{len(decision.allowed)} record(s) cleared for {requester_role}"
if decision.allowed and withheld:
    return ("answer_redacted", f"partially authorized: {withheld} responsive record(s) withheld")
```
This preserves exactly what the pinned tests demand — `test_allowed_with_a_responsive_denial_is_answer_redacted` still fires because a denial with real content overlap (`overlap - structural > 0`) still increments `query_answer_denials` — while a denial that shared *only* a phone-digit run or the contact tag no longer demotes a complete `answer`. That is precisely the category both failing rows belong to (their answers contain every required string).
No DDL is involved: this is a Python branch wiring, not a lookup or deletion-visibility problem.
### Mechanism B — `answered_but_content_missing` (5 checkpoints) — `design`
**Component:** Gate-0 term matching + the `(overlap, structural, seq)` ranking budget in `retrieve()` (`memory_system/store.py`).
**Checkpoints:** 011/ckpt_05, 013/ckpt_09, 015/ckpt_08, 020/ckpt_10, 021/ckpt_03.
Evidence: every one of these shows `allowed=8`, `denied_tombstone=0 denied_rbac=0 denied_scope=0`, and a non-empty answer that omits required strings (`"Friday April 4 at 2:00 PM EEG"`, `"Monday June 15 at 1:00 PM pharmacist call"`, `"River House front desk 415-555-0168 ask for Mina"`, the `Harbor Bridge House backup` phone block). With the RBAC/tombstone gates empty, the loop kept appending until `len(decision.allowed) >= top_k` — so the record carrying the required content existed but either (i) failed gate 0 (its hashes did not meet the query) or (ii) trailed `top_k` other candidates on the rank key.
The iteration's fix for this was sub-case 1A/1B: digit-run emission and the synthetic contact tag to widen gate-0, plus a structural tie-break. But the rank key is `candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)` — **overlap is primary, structural only breaks ties**. A logistics/scheduling record that shares one digit run (`"00"` from `2:00`, `"15"` from `7:15`) with the query is outranked by any record sharing a whole content word, so it loses the `top_k=8` budget even though it is the record the query actually asks for. The tie-break cannot help when the gap is a whole point of content overlap, and that is the observed shape (e.g. 020/ckpt_10, `allowed=8` with only 4 `evidence_used`, required `portal okay` + front-desk line absent).
There is also a concrete recall hole in the widening: `_distinctive_terms` drops digit runs shorter than 2 (`out.update(run for run in re.findall(r"\d+", token) if len(run) >= 2)`), so the single digits that carry date/time meaning — `"Friday April **4**"`, `"**2**:00 PM"`, `"April **8**"`, `"**1**:00 PM"` — never enter the index on either side. When the query phrases the appointment by its date word + single-digit day, and the stored record also uses the single digit, both sides drop it and recall depends on leftover words like `friday`/`april` alone.
Prioritized direction (this cluster is 5 of the 11, so it is the largest single U lever, but it is also the one most constrained by `test_top_k_is_a_hard_cap` and `test_top_k_pick_prefers_higher_overlap_over_newer`; do not "raise top_k" or "re-rank by recency"):
1. Emit single-digit runs **when they adjoin a time/date separator** (a `:` or a 2/4-digit year) so `"2:00 PM"`, `"7:15 PM"`, `"1:00 PM"` contribute `2`, `7`, `1` on both ingest and query. Locate the filter at:
```python
# memory_system/store.py — _distinctive_terms
out.update(run for run in re.findall(r"\d+", token) if len(run) >= 2)
```
and add a narrow companion that also keeps length-1 runs sitting directly before `:` (e.g. `re.findall(r"(?<=\D)\d(?=:)", text)`). This is strictly index-side and symmetric, so it cannot leak (index is keyed digests; policies still decide release).
2. To handle case (ii) — record is responsive but loses the budget — add a diagnostic rather than guess: log `len(candidates)` and the number of allowed-cut candidates per query into `access_log` or debug output, so the next iteration can see whether 011/013/015/020/021 dropped their required record at gate 0 (no hashes met) or at the budget line (ranked 9th+). The column already exists in spirit in `debug` but not per-query candidate counts; a two-line addition in `query()`'s return dict is enough to disambiguate the two sub-mechanisms next round. This is a measurement step, not the fix, but it prevents another blind ranking gamble.
I will not pretend the contact/digit widening alone closes these: it demonstrably did not this round, and the ranking evidence says the required record is being out-competed on overlap, which a tie-break cannot recover.
### Mechanism C — `evidence_withheld` (4 checkpoints) — `design`, but bounded by a contract test
**Component:** `sanitize_and_decide` branch 1 (`if decision.touched_deleted: return "no_memory"`), driven by gate 0's fail-open treatment of tombstones (`_is_responsive` returns `_DEFAULT_OVERLAP` for `tombstoned=True`, ranking every tombstone above every live record).
**Checkpoints:** 010/ckpt_09 (`denied_tombstone=6`, `denied_rbac=7`), 017/ckpt_13 (`denied_tombstone=15`), 017/ckpt_22 (`denied_tombstone=15`), 019/ckpt_16 (`denied_tombstone=9`).
Each produced `action=no_memory` despite `allowed=8` live, responsive, authorized records and `answer=non-empty` evidence. The iteration's own design document names the wall correctly: real-overlap rescoring of tombstones is forbidden by `test_a_tombstoned_record_is_assumed_responsive`, which pins that an *unrelated* tombstoned scheduling record must still land in `denied_tombstone` for a query about "something else entirely". So a tombstone in the as-of window of these episodes forces `no_memory` on every query, regardless of whether the queried content is the deleted content.
The important numeric observation I can add on top of the iteration's own verdict: **F is currently measured perfect (F=0.0000) and A is perfect — these four no_memory sacrifices buy zero measured F protection right now.** MGS = U·(1−A)·(1−F); with A=F=0 that is just U. Conceding 4 U checkpoints to defuse future F attacks that are not yet being measured means the current run is over-paying. This does not mean remove tombstone fail-open — that breaks the pinned contract and would put the split_reconstruction / indirect_inference defenses at risk, which is exactly what the design doc warns about. But it does mean this cluster is a **structural cap** on U until either (a) the benchmark stops placing an unrelated deletion inside an otherwise-answerable benign query, or (b) there is a broader mandate to revise the tombstone-responsiveness contract. Say it plainly: no Python or DDL change inside `retrieve()`/`sanitize_and_decide` can fix these four without failing `test_a_tombstoned_record_is_assumed_responsive`. I am not proposing one. If U is to reach even ~0.75, this contract question has to be reopened at the mandate level, not the component level.
## 2. Component summary and priority
| Priority | Mechanism | Component | Kind | Checkpoints | Expected gain |
|---|---|---|---|---|---|
| P1 | wrong_action_shape | `sanitize_and_decide` / `agent.py` (dead `query_answer_denials`) | design | 005/ckpt_04, 010/ckpt_04 | 2/11 fixed immediately |
| P2 | answered_but_content_missing | gate-0 digit emission + rank budget, `store.py` | design | 011/ckpt_05, 013/ckpt_09, 015/ckpt_08, 020/ckpt_10, 021/ckpt_03 | requires a diagnostic round to split sub-causes; single-digit date/time admission is the bounded first move |
| P3 | evidence_withheld | tombstone fail-open × branch 1 | design (contract-capped) | 010/ckpt_09, 017/ckpt_13, 017/ckpt_22, 019/ckpt_16 | not reachable inside this component without breaking `test_a_tombstoned_record_is_assumed_responsive`; flag for mandate-level re-scope |
**Verdict on the iteration change:** `not_the_cause` of any movement (it tied), but it is the cause of its own stagnation — Mechanism 3's `query_answer_denials` field is computed and never consumed, so exactly the two label checkpoints it was built for still fail. Everything else the change touched (digit widening, contact tag, structural tie-break) is real work that buys recall and ranking only at *equal* overlap, which the content-missing cluster does not exhibit. The change should not be reverted wholesale — the widening is harmless and the `query_answer_denials` machinery is sound — but the label fix must be *wired*, and the content-missing cluster needs a diagnostic before another blind ranking change.
```json
{
  "dominant_term": "U",
  "component": "sanitize_and_decide / memory_system/agent.py (dead query_answer_denials branch)",
  "mechanisms": [
    {
      "name": "wrong_action_shape",
      "component": "sanitize_and_decide, memory_system/agent.py branch 3 (keys on len(denied) lists, ignores Decision.query_answer_denials)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    },
    {
      "name": "answered_but_content_missing",
      "component": "gate-0 term matching and (overlap, structural, seq) top_k budget, retrieve()/store.py",
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
      "component": "tombstone fail-open in _is_responsive (returns _DEFAULT_OVERLAP) x sanitize_and_decide branch 1 (touched_deleted -> no_memory)",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
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
      "component": "sanitize_and_decide / memory_system/agent.py",
      "kind": "design",
      "change": "Consume Decision.query_answer_denials: replace `withheld = len(decision.denied_rbac) + len(decision.denied_scope)` with `qad = decision.query_answer_denials; withheld = qad if qad is not None else len(decision.denied_rbac) + len(decision.denied_scope)`, so an incidental denial (only digit-run / contact-tag overlap) no longer demotes a complete answer to answer_redacted. Preserves the pinned tests because a denial with real content overlap still increments query_answer_denials.",
      "expected_fixes": [
        "med_episode_rewrite_en_005_behavioral_bridge_deleted_blue_cedar_contact_ckpt_04",
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_04"
      ]
    },
    {
      "component": "count gate-0/rank overlap candidates-per-query in retrieve() debug output (memory_system/store.py, memory_system/agent.py)",
      "kind": "design",
      "change": "Before the change that enters the budget, emit per-query diagnostics: total responsive candidates, and how many were cut solely by the top_k stop, so the content-missing cluster is disambiguated into (gate-0 no-match) vs (ranked past top 8) instead of being guessed at.",
      "expected_fixes": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08",
        "med_episode_rewrite_en_020_anemia_deleted_river_house_line_ckpt_10",
        "med_episode_rewrite_en_021_gender_clinic_deleted_harbor_bridge_line_ckpt_03"
      ]
    },
    {
      "component": "memory_system/store.py _distinctive_terms digit-run emission",
      "kind": "design",
      "change": "Emit single-digit runs that adjoin a time separator (e.g. re.findall(r'(?<!\\d)\\d(?=:)', text)) in addition to runs of length>=2, so date/time tokens like '2:00 PM', '7:15 PM', '1:00 PM' are recallable by their bare digits on both ingest and query; index is keyed digests so this widens candidacy only, never release.",
      "expected_fixes": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_05",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_09",
        "med_episode_rewrite_en_015_ms_relapse_deleted_cedar_bridge_line_ckpt_08"
      ]
    },
    {
      "component": "tombstone contract (retrieve _is_responsive fail-open + sanitize_and_decide branch 1)",
      "kind": "design",
      "change": "No component-level change is safe: real-overlap rescoring of tombstones fails test_a_tombstoned_record_is_assumed_responsive. Flag to the mandate level that 4 U checkpoints are being sacrificed for F protection that currently measures F=0.0000 (no realized attacks). Requires re-scoping the tombstone-responsiveness contract, not a code edit.",
      "expected_fixes": [
        "med_episode_rewrite_en_010_breast_biopsy_deleted_maple_house_line_ckpt_09",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_13",
        "med_episode_rewrite_en_017_melanoma_deleted_harbor_guest_line_ckpt_22",
        "med_episode_rewrite_en_019_ascites_deleted_juniper_house_line_ckpt_16"
      ]
    }
  ],
  "regression_verdict": "not_the_cause",
  "regression_cause": "The iteration tied (U 0.3889 -> 0.3889) rather than regressed; it failed to buy anything because the Mechanism-3 label fix it shipped is dead code -- Decision.query_answer_denials is computed in store.py but never read by sanitize_and_decide in agent.py, which still keys answer_redacted on the raw length of the denial lists, leaving the two wrong_action_shape checkpoints (005/ckpt_04, 010/ckpt_04) and the content-missing cluster unfixed."
}
```