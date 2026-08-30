CREATE TABLE IF NOT EXISTS deletion_terms (
    record_id  TEXT NOT NULL REFERENCES records(record_id),
    term       TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    PRIMARY KEY (record_id, term)
);

CREATE INDEX IF NOT EXISTS idx_deletion_terms_patient_term
    ON deletion_terms(patient_id, term);

-- Bump schema version in code to 2