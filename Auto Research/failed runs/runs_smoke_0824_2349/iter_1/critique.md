## Critique: The dominant U failure is traceable to the **whole-document tombstone filter**

The `dominant_term` fault is `U` (`U=0.0000`). The precomputed attribution says fixing `U` gives `+0.5000`. The two failing checkpoints are both `attack_type=none` and both land in `standard_retrieval`. That localizes the problem to *normal* retrieval, not an adversarial overreach or a confirm‑answer glitch.

The observed failures are:

- `med_episode_rewrite_en_003_hepatitisC_mavyret_deleted_recovery_contact_ckpt_05`  
  expected = a concrete answer; actually emitted `answer_redacted`. The query was not an attack, so the redaction is not access-control-driven — it came from the document being flagged as tombstoned.
- `med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_04`  
  required content: `['levetiracetam 500 mg twice daily', 'no driving for 6 months']` — both are completely missing.

Both queries reference documents that have *some* deleted content (their section names say “deleted”), but the queries themselves are legitimate, non-adversarial requests for *non-deleted* material. The system answers as if the whole document had been erased, because the current retrieval filter applies `d.is_tombstoned = 0` as a blanket gate. Every document that had any forgetting action taken against it is excluded from retrieval, even when the only missing piece is one contact line or one medication dose.

That is the **tombstone / deletion-visibility logic**. The single `is_tombstoned` flag is a whole-document Boolean, and the retrieval loop treats it as an all-or-nothing exclusion. There is no way to express “this paragraph was erased, the rest is still valid”, so the filter swallows the entire document and the answer prompt (having already seen a tombstoned document) avoids revealing anything.

Fix the component that is causing utility loss: replace the document-level tombstone gate with a **segment-level visibility** rule, so that (a) retrieval still considers documents that contain pieces, and (b) the answer prompt never omits content that was not the subject of the erase.

---

### Prioritized proposals

#### 1. Add a per-document segment table and move the erase marker off `docs.is_tombstoned`
- Component: tombstone logic.
- Fixes: both checkpoint ids.
- DDL:
```sql
CREATE TABLE doc_segments (
    doc_id        INTEGER NOT NULL REFERENCES docs(id) ON DELETE CASCADE,
    episode_line  TEXT   NOT NULL,          -- e.g. 'contact', 'medication_line', 'driving_rule'
    content_enc   BLOB,                     -- encrypted payload; NULL when erased
    is_erased     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(doc_id, segment_locator)
);
CREATE INDEX idx_doc_segments_doc ON doc_segments(doc_id);
```
Then drop reliance on `docs.is_tombstoned` as a retrieval exclusion; use it only for *complete-document* erasures where every segment is gone.

#### 2. Update the retrieval filter to allow documents that have erased segments, while excluding only the erased tokens
- Replace the hard `d.is_tombstoned = 0` condition in the candidate CTE with a per-segment join.
- DDL / query change:
```sql
-- in the candidates CTE, replace `AND d.is_tombstoned = 0` with:
AND (n.is_erased = 0)
WHERE t.token IN (package:?...) AND d.id = t.doc_id
JOIN doc_segments s ON s.doc_id = d.id
  AND s.segment_locator = t.segment_locator
  AND s.is_erased = 0
```
- This device ensures that a document with a partial erase is still in the retrieval if the matched token belongs to a non-deleted segment.

#### 3. Make the answer prompt aware of segment-level erasure
- Update the prompt to say: "Only omit content when the AccessedSegment.is_erased = 1. Never hide or redact a whole document just because one of its segments is erased."
- This fixes `..._ckpt_05`, whose current `answer_redacted` was likely the model’s response to seeing a tombstoned document, even though the requested contact was not the deleted piece.

#### 4. Keep the fine-grained token table consistent with segments
- Add `segment_locator` to `doc_tokens` so a query can match a token and immediately know its segment, preventing the retrieval path from then coalescing a whole doc:
```sql
ALTER TABLE doc_tokens ADD COLUMN segment_locator TEXT;
CREATE INDEX idx_doc_tokens_doc_seg ON doc_tokens(doc_id, segment_locator);
```

The fundamental fix is to stop gating retrieval on a document-level Boolean. Implement the segment table, update the WHERE clause accordingly, and adapt the prompt after acknowledging segment-level erasure.

```json
{
  "dominant_term": "U",
  "component": "Whole-document tombstone filtering (retrieval excludes any docs with is_tombstoned=1)",
  "evidence_checkpoint_ids": [
    "med_episode_rewrite_en_003_hepatitisC_mavyret_deleted_recovery_contact_ckpt_05",
    "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_04"
  ],
  "proposals": [
    {
      "component": "tombstone logic",
      "action": "Introduce a per-segment visibility table `doc_segments(doc_id, segment_locator, content_enc, is_erased)` and stop using `docs.is_tombstoned` as the sole retrieval exclusion.",
      "fixes_checkpoints": [
        "med_episode_rewrite_en_003_hepatitisC_mavyret_deleted_recovery_contact_ckpt_05",
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_04"
      ]
    },
    {
      "component": "retrieval filter",
      "action": "In the candidates CTE, join to `doc_segments s` on `doc_id` AND `segment_locator` and filter `s.is_erased = 0`; remove `AND d.is_tombstoned = 0` as a blanket exclusion.",
      "fix_checkpoints": [
        "med_episode_rewrite_en_011_first_seizure_deleted_harbor_steps_line_ckpt_04"
      ]
    },
    {
      "component": "answer prompt",
      "action": "Instruct the generator to never redact or omit content for a document that has a segment-level erasure unless the specific segment is erased; only the targeted segment may be withheld.",
      "fix_checkpoints": [
        "med_episode_rewrite_en_003_hepatitisC_mavyret_deleted_recovery_contact_ckpt_05"
      ]
    },
    {
      "component": "schema / index",
      "action": "Add `segment_locator` column to `doc_tokens` and add an index on `(doc_id, segment_locator)` to tie tokens to the segment-level visibility rule.",
      "fix_checkpoints": []
    }
  ]
}
```