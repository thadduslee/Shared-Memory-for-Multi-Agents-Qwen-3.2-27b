-- Iteration-2 migration: two lookup indexes only; no table rewrite.
CREATE INDEX IF NOT EXISTS idx_records_patient_seq ON records(patient_id, seq);
-- rationale: retrieve() filters r.patient_id = ? AND r.seq <= ? then ORDER BY r.seq DESC; this composite covers filter + sort on one path.
CREATE INDEX IF NOT EXISTS idx_access_log_requester ON access_log(requester_id, decision);
-- rationale: access_log is queried by requester_id (audit gate + Critic evidence); adds the decision filter to the same seek.
