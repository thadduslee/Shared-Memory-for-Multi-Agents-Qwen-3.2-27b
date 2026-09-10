-- Iteration 1 adds no new tables or columns. The only storage change is the composite index
-- supporting the gate-0 overlap pass.
-- Accelerates gate-0 overlap ranking by letting the planner serve the record_id->term_hash
-- probe (and conversely) from one index instead of two single-column seeks.
CREATE INDEX IF NOT EXISTS idx_record_terms_lookup ON record_terms(record_id, term_hash);