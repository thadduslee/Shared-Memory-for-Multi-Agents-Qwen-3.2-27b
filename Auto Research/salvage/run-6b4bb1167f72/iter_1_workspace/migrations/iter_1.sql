-- iteration 1: baseline schema, no migration needed beyond creation.
CREATE INDEX IF NOT EXISTS idx_tombstones_deleted_at ON tombstones(deleted_at);