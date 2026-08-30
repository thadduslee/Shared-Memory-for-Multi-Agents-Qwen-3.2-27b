CREATE TABLE IF NOT EXISTS forgotten_facts (
    fact_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id    TEXT NOT NULL,
    deleted_at    TEXT NOT NULL,
    requested_by  TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    terms_json    TEXT NOT NULL
);
-- Accelerates _honor_deletion_request's per-episode scan for forgettable content.
CREATE INDEX IF NOT EXISTS idx_forgotten_episode ON forgotten_facts(episode_id);
-- Partial index: retrieve()/delete-scan only ever touch non-empty bodies; skip wiped rows.
CREATE INDEX IF NOT EXISTS idx_records_active_body ON records(episode_id, seq) WHERE body != '';