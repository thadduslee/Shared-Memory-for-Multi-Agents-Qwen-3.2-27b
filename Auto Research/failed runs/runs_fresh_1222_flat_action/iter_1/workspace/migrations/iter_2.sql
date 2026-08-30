-- Critic attributed the loss to U (utility): authorized queries were
-- missing evidence because the as-of ordering scan was unindexed.
CREATE INDEX IF NOT EXISTS idx_records_asof ON records(episode_id, seq DESC, sensitivity);