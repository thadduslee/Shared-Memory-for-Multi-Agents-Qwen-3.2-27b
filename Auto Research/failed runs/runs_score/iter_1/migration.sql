-- Baseline migration (iteration 1): no prior schema exists, so this is the full
-- idempotent DDL. Later iterations append additive migrations here.
PRAGMA journal_mode = DELETE;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS principals (
    principal_id  TEXT PRIMARY KEY,
    role          TEXT NOT NULL,
    display_name  TEXT
);

CREATE TABLE IF NOT EXISTS relationships (
    rel_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    rel_type      TEXT NOT NULL,
    subject_id    TEXT NOT NULL,
    patient_id    TEXT NOT NULL,
    scope         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS records (
    record_id     TEXT PRIMARY KEY,
    episode_id    TEXT NOT NULL,
    turn_id       TEXT NOT NULL,
    author_id     TEXT NOT NULL,
    author_role   TEXT NOT NULL,
    patient_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    sensitivity   TEXT NOT NULL DEFAULT 'routine',
    body          TEXT NOT NULL,
    ciphertext    BLOB,
    key_id        TEXT,
    ts            TEXT NOT NULL,
    seq           INTEGER NOT NULL,
    is_deleted    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS role_grants (
    role          TEXT NOT NULL,
    sensitivity   TEXT NOT NULL,
    requires_rel  TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (role, sensitivity)
);

CREATE TABLE IF NOT EXISTS role_hierarchy (
    child_role    TEXT NOT NULL,
    parent_role   TEXT NOT NULL,
    PRIMARY KEY (child_role, parent_role)
);

CREATE TABLE IF NOT EXISTS tombstones (
    record_id     TEXT PRIMARY KEY,
    deleted_at    TEXT NOT NULL,
    requested_by  TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    shredded      INTEGER NOT NULL DEFAULT 0,
    terms         TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS crypto_keys (
    key_id        TEXT PRIMARY KEY,
    shredded_at   TEXT
);

CREATE TABLE IF NOT EXISTS access_log (
    log_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT,
    requester_id  TEXT NOT NULL,
    record_id     TEXT NOT NULL,
    decision      TEXT NOT NULL,
    ts            TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_episode_seq       ON records(episode_id, seq);
CREATE INDEX IF NOT EXISTS idx_records_episode_seq_sens  ON records(episode_id, seq, sensitivity);
CREATE INDEX IF NOT EXISTS idx_records_patient           ON records(patient_id, sensitivity);
CREATE INDEX IF NOT EXISTS idx_rel_subject               ON relationships(subject_id, patient_id);
CREATE INDEX IF NOT EXISTS idx_rel_patient_subject       ON relationships(patient_id, subject_id);
CREATE INDEX IF NOT EXISTS idx_role_grants_lookup        ON role_grants(role, sensitivity);
CREATE INDEX IF NOT EXISTS idx_tombstones_unshredded     ON tombstones(record_id) WHERE shredded = 0;