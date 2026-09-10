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

-- THE RESPONSIVENESS INDEX.  One row per (record, distinctive term), where the
-- term is stored as a keyed hash rather than as text.
--
-- WHY IT EXISTS.  `retrieve()` has to answer "was this record about the query?"
-- for records it is going to DENY, not only for records it is going to return.
-- Without that, a record the requester may not read but which has nothing to do
-- with the question still lands in `denied_rbac`, `Decision.touched_unauthorized`
-- goes true, and `sanitize_and_decide` downgrades a complete authorized answer
-- to `answer_redacted`.  That single confusion accounted for 12 of the 18
-- utility checkpoints in the seeded dev slice: every one of them was a correct,
-- complete answer scored zero because the ACTION LABEL was wrong.
--
-- WHY HASHED, AND WHAT THAT IS AND IS NOT WORTH.  Confidential bodies are
-- stored as ciphertext with the plaintext discarded, so a side table holding
-- their content words in the clear would undo that at rest.  The hash is
-- HMAC-SHA256 under a per-store key kept in `store_meta`, which means the index
-- is not a plaintext copy of confidential content.  It is NOT a defence against
-- an attacker holding the database file -- the key is in it -- and it is not
-- meant to be; the threat it addresses is the one the encryption addresses.
--
-- WHY IT IS PURGED ON TOMBSTONE.  Cryptographic shredding destroys the body by
-- destroying the key; an index that outlived it would leave the record's content
-- words recoverable and make shredding a half-measure.  `tombstone()` deletes
-- these rows in the same transaction, which is why a tombstoned record's
-- responsiveness is conservatively assumed when nothing real was preserved --
-- see `MemoryStore._is_responsive`.
--
-- `is_structural` (iteration 5): digit-run and synthetic-contact digests are
-- flagged 1, real content 0, at index time.  Gate 1 (tombstone responsiveness)
-- scopes overlap over `is_structural = 0` rows only, so the structural/content
-- split gates 2/3 already use is applied to tombstoned records too.
CREATE TABLE IF NOT EXISTS record_terms (
    record_id     TEXT NOT NULL,
    term_hash     TEXT NOT NULL,
    is_structural INTEGER NOT NULL DEFAULT 0,   -- 1 = digit-run/contact digest, 0 = real content
    PRIMARY KEY (record_id, term_hash)
);

-- TOMBSTONE TERM DIGESTS (iteration 4).  When a record is tombstoned its
-- `record_terms` rows are deleted (the index goes with the body, so shredding
-- is not a half-measure), but those digests are FIRST copied here and kept.
-- That lets a later query tell a RELATED deletion -- one whose preserved terms
-- overlap the question -- apart from an UNRELATED one, so scoped responsiveness
-- survives the tombstone instead of every tombstoned record being assumed
-- relevant to everything.  Kept separate from `record_terms` because these rows
-- must never be purged by `tombstone()`.  `is_structural` is copied verbatim
-- from the purged `record_terms` rows so gate 1 can filter by real content.
CREATE TABLE IF NOT EXISTS tombstone_terms (
    record_id     TEXT NOT NULL REFERENCES tombstones(record_id) ON DELETE CASCADE,
    term_hash     TEXT NOT NULL,
    is_structural INTEGER NOT NULL DEFAULT 0,   -- 1 = digit-run/contact digest, 0 = real content
    PRIMARY KEY (record_id, term_hash)
);

-- Per-store scalars.  Currently just the responsiveness index's HMAC key, which
-- has to survive reopening the file or a resumed run would hash the same term
-- to a different digest and match nothing.
CREATE TABLE IF NOT EXISTS store_meta (
    key           TEXT PRIMARY KEY,
    value         BLOB
);

CREATE INDEX IF NOT EXISTS idx_record_terms_hash  ON record_terms(term_hash);
CREATE INDEX IF NOT EXISTS idx_tombstone_terms_hash ON tombstone_terms(term_hash);
-- Gate-1 tombstone responsiveness filters overlap by is_structural=0, so the
-- scoped index is keyed by (record_id, is_structural, term_hash).
CREATE INDEX IF NOT EXISTS idx_tombstone_terms_structural ON tombstone_terms(record_id, is_structural, term_hash);
CREATE INDEX IF NOT EXISTS idx_records_episode_seq ON records(episode_id, seq);
CREATE INDEX IF NOT EXISTS idx_records_patient     ON records(patient_id, sensitivity);
CREATE INDEX IF NOT EXISTS idx_rel_subject         ON relationships(subject_id, patient_id);
