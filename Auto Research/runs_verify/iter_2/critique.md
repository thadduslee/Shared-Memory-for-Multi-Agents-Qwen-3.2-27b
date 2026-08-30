## Critique
### Mechanism census and attribution
The single observed mechanism is **`answered_but_content_missing`** (6 checkpoints). In every case retrieval **succeeded** (`allowed > 0`), the answer was non‑empty, and no tombstone was hit. The missing content (e.g. `apixaban`, `stop the estrogen patch`, appointment times) simply never appears in the allowed evidence. That points at the **retrieval filter** that decides *which* gate‑cleared records become part of `decision.allowed`, not at the answer prompt and not at the candidate scan (which did return rows).
Reading the actual code, the cause is the hard **responsiveness filter** applied immediately before the `top_k` selection:
```python
# memory_system/store.py, inside MemoryStore.retrieve()
responsive = [c for c in candidates if self._is_responsive(c[2], terms)]
responsive.sort(key=lambda c: (c[0], c[1]), reverse=True)
for score, seq, row, plaintext in responsive[:top_k]:
    decision.allowed.append( ... )
```
Any gate‑cleared record that does **not** share at least one distinctive term with the query is silently dropped before `top_k`. So a record containing the requested fact (e.g. the note with the medication change) is not released when the query phrasing does not contain those exact lexical tokens. That is exactly the residual hard relevance drop the previous iteration’s design claimed to remove — but the drop still exists in the form of this `responsive` filter.
This is a **design** fault in the retrieval filter. It is not infrastructure: no crashes, no empty completions.
### Why the other hypotheses are not supported
- **Answer prompt**: The missing content never reaches the answerer; `allowed` does not contain the record. The prompt cannot conjure text absent from its evidence.
- **Top‑k truncation**: The `responsive` filter discards the record *before* `top_k` is applied. The allowed counts in the checkpoints (2–8) are well below `top_k=8`, so even if the record ranked at the tail, it would still fit within the budget once unfiltered.
- **Schema/index**: The scan demonstrably found the right rows (gates passed for `allowed`); it is only the post‑gate relevance filter that removes them.
### The fix
Remove the `responsive` filter from the allowed‑evidence selection. Sort *all* gate‑cleared `candidates` by relevance score (including zero‑score rows), then take `top_k`. A zero‑score record will rank last, but it will still be released if there is room — and for these checkpoints there is.
Concretely, change in `memory_system/store.py`:
```python
# before
responsive = [c for c in candidates if self._is_responsive(c[2], terms)]
responsive.sort(key=lambda c: (c[0], c[1]), reverse=True)
for score, seq, row, plaintext in responsive[:top_k]:
    ...
# after
candidates.sort(key=lambda c: (c[0], c[1]), reverse=True)
for score, seq, row, plaintext in candidates[:top_k]:
    ...
```
The `_is_responsive` helper can still be used for the deny tallies (to avoid over‑counting irrelevant denials), but it must **not** gate the allowed‑release path.
Secondary consideration: if the store holds more than `top_k` gate‑cleared records and the relevant one happens to rank below the cutoff (possible if many unrelated records share some terms), increasing `top_k` to e.g. 20 would be a safe fallback. Given the observed allowed counts (2–8), the filter removal alone should be sufficient; raising `top_k` is a defensive extra.
---
```json
{
  "dominant_term": "U",
  "component": "retrieval_filter",
  "mechanisms": [
    {
      "name": "answered_but_content_missing",
      "component": "retrieval_filter",
      "kind": "design",
      "checkpoint_ids": [
        "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
        "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
        "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
        "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
        "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
      ]
    }
  ],
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_002_cardiology_afib_cardioversion_confidential_backup_ckpt_06",
    "med_episode_rewrite_en_009_pe_deleted_willow_house_line_ckpt_01",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_08",
    "med_episode_rewrite_en_012_new_hiv_deleted_harbor_light_line_ckpt_09",
    "med_episode_rewrite_en_013_ibd_deleted_pine_harbor_line_ckpt_02",
    "med_episode_rewrite_en_016_early_pregnancy_deleted_haven_house_line_ckpt_02"
  ],
  "proposals": [
    {
      "component": "retrieval_filter",
      "kind": "code_change",
      "change": "In MemoryStore.retrieve(), remove the `responsive` filtering step. Sort the entire `candidates` list (which already contains all gate-cleared records) by (relevance_score, seq) descending, and take the first `top_k` as `decision.allowed`. The `_is_responsive` helper remains only for deny tally counting, not for allowed selection.",
      "expected_fixes": ["all 6 failing checkpoints"]
    },
    {
      "component": "retrieval_filter",
      "kind": "fallback",
      "change": "Consider raising `top_k` from 8 to at least 20 to ensure the unfiltered candidate set covers all relevant records, in case many unrelated gate-cleared records exist.",
      "expected_fixes": ["backstop for any remaining surfaced-content misses"]
    }
  ]
}
```