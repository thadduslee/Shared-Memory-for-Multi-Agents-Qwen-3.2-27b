ALTER TABLE record_terms ADD COLUMN is_structural INTEGER NOT NULL DEFAULT 0;
ALTER TABLE tombstone_terms ADD COLUMN is_structural INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_tombstone_terms_structural ON tombstone_terms(record_id, is_structural, term_hash);
-- rationale: gate-1 tombstone responsiveness filters overlap by is_structural=0 (structural/content split), so the index must be keyed by (record_id, is_structural, term_hash).