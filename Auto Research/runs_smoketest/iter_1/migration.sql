-- Two additive indexes; accelerations for the new deletion-fallback scan and the audit path.
CREATE INDEX IF NOT EXISTS idx_records_episode_author_seq ON records(episode_id, author_id, seq);  -- accelerates _honor_deletion_request fallback 'WHERE episode_id=? AND author_id=? ORDER BY seq DESC'
CREATE INDEX IF NOT EXISTS idx_accesslog_checkpoint_requester ON access_log(checkpoint_id, requester_id);  -- accelerates the auditor / Critic evidence query on a specific checkpoint/requester
