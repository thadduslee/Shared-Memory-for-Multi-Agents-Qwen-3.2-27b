-- Baseline DDL for the multi-principal medical memory store.
-- SQLite only, one file per run (brief section 6.1 storage constraint).
--
-- Design notes (these are the Architect's iteration-0 choices; the Architect
-- may replace them via migrations on later iterations):
--   * `records` is the single evidence table.  Sensitivity is a column rather
--     than a separate table so the RBAC join stays a two-table join.
--   * Deletion is a TOMBSTONE, never a DELETE.  A hard DELETE would make
--     "was there ever such a record?" unanswerable, and the benchmark's
--     `post_delete_recovery` and `confirm_yes_no` attacks specifically probe
--     whether the system can still tell the difference.  We must know a record
--     was deleted in order to refuse *without confirming its content*.
--   * Ciphertext is stored separately from the key so that shredding the key
--     renders the payload unrecoverable without rewriting the row.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS principals (
    principal_id  TEXT PRIMARY KEY,
    role          TEXT NOT NULL,
    display_name  TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    rel_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_type      TEXT NOT NULL,          -- assigned_clinician | covering_clinician | family_contact
    subject_id    TEXT NOT NULL,          -- the principal the relationship grants rights to
    patient_id    TEXT NOT NULL,
    scope         TEXT DEFAULT ''         -- free-text scope limit, e.g. 'logistics only'
);

CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT PRIMARY KEY,
    episode_id    TEXT NOT NULL,
    turn_id       TEXT NOT NULL,
    author_id     TEXT NOT NULL,
    author_role   TEXT NOT NULL,
    patient_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,          -- dialogue | note_update | lab_result | ...
    sensitivity   TEXT NOT NULL DEFAULT 'routine',   -- routine | restricted | confidential
    body          TEXT NOT NULL,          -- plaintext for routine content
    ciphertext    BLOB,                   -- set for shreddable content
    key_id        TEXT,                   -- FK into crypto_keys; NULL once shredded
    ts            TEXT NOT NULL,
    seq           INTEGER NOT NULL        -- monotonic ordering for as-of queries
);

-- Which roles may read which sensitivity tier.  The RBAC "graph" is this table
-- joined against `relationships`: a grant is only live if the requester also
-- holds a relationship to the patient.
CREATE TABLE IF NOT EXISTS role_grants (
    role          TEXT NOT NULL,
    sensitivity   TEXT NOT NULL,
    requires_rel  TEXT NOT NULL DEFAULT '',   -- '' = no relationship needed
    PRIMARY KEY (role, sensitivity)
);

CREATE TABLE IF NOT EXISTS tombstones (
    record_id     TEXT PRIMARY KEY,
    deleted_at    TEXT NOT NULL,
    requested_by  TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    shredded      INTEGER NOT NULL DEFAULT 0
);

-- Terms of every tombstoned record, harvested at deletion time.  A later
-- query is only treated as "touching deleted content" when it shares one of
-- these terms with the deleted record -- so a request about an unrelated topic
-- cannot be turned into a yes/no existence probe (`confirm_yes_no`), while a
-- genuine attempt to recall the deleted fact still reads as `no_memory`.
CREATE TABLE IF NOT EXISTS deletion_terms (
    record_id  TEXT NOT NULL REFERENCES records(record_id),
    term       TEXT NOT NULL,
    patient_id TEXT NOT NULL,
    PRIMARY KEY (record_id, term)
);

CREATE TABLE IF NOT EXISTS crypto_keys (
    key_id        TEXT PRIMARY KEY,
    key_material  BLOB,                   -- NULLed on shred; the row survives as an audit trail
    shredded_at   TEXT
);

-- Every retrieval decision is logged.  Auditability is a requirement of the
-- artifact, and it is also how the Critic cites evidence for a specific
-- checkpoint id.
CREATE TABLE IF NOT EXISTS access_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT,
    requester_id  TEXT NOT NULL,
    record_id     TEXT NOT NULL,
    decision      TEXT NOT NULL,          -- allow | deny_rbac | deny_tombstone | deny_scope
    ts            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq);
CREATE INDEX IF NOT EXISTS idx_records_patient     ON records(patient_id, sensitivity);
CREATE INDEX IF NOT EXISTS idx_rel_subject         ON relationships(subject_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_deletion_terms_patient_term
    ON deletion_terms(patient_id, term);
