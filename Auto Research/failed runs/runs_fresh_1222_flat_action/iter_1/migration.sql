CREATE VIRTUAL TABLE IF NOT EXISTS records_fts USING fts5(record_id UNINDEXED, body, author_role, patient_id UNINDEXED, terms UNINDEXED);
CREATE TRIGGER IF NOT EXISTS records_ai AFTER INSERT ON records BEGIN
    INSERT INTO records_fts(record_id, body, author_role, patient_id, terms)
    VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, '');
END;
CREATE TRIGGER IF NOT EXISTS records_ad AFTER DELETE ON records BEGIN
    DELETE FROM records_fts WHERE record_id = OLD.record_id;
END;
CREATE TRIGGER IF NOT EXISTS records_au AFTER UPDATE ON records BEGIN
    DELETE FROM records_fts WHERE record_id = OLD.record_id;
    INSERT INTO records_fts(record_id, body, author_role, patient_id, terms)
    VALUES (NEW.record_id, NEW.body, NEW.author_role, NEW.patient_id, '');
END;
CREATE INDEX IF NOT EXISTS idx_tombstones_terms_unshredded ON tombstones(terms) WHERE shredded = 0;
-- Each statement has explicit rationale in the design document.