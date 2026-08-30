PRAGMA table_info(records); -- Developer guards: only run the ALTER when 'retired' is absent (fresh schema.sql already has the column, so this is for legacy DBs)
ALTER TABLE records ADD COLUMN retired INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_records_active_patient ON records(patient_id, seq) WHERE retired = 0;
CREATE INDEX IF NOT EXISTS idx_access_log_checkpoint ON access_log(checkpoint_id, decision);