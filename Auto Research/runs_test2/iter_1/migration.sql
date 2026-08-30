CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(record_id UNINDEXED, body);
INSERT INTO records_fts(record_id, body) SELECT record_id, body FROM records WHERE sensitivity IN ('routine','restricted') AND body != '';
CREATE INDEX IF NOT EXISTS idx_rel_patient_subject ON relationships(patient_id, subject_id);