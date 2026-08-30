-- Iteration 3 migration (SCHEMA_VERSION -> 2)
-- One new index: the Critic aggregates access_log per (checkpoint_id, decision) to
-- attribute evidence; currently that scan is log_id-ordered. Composite index makes
-- the per-checkpoint evidence census an index range scan.
CREATE INDEX IF NOT EXISTS idx_access_log_checkpoint
    ON access_log(checkpoint_id, decision);
-- No data migration is needed; the behaviour fixes below are in code, and
-- tightening the responsiveness filter is forward-only on the rows already stored.