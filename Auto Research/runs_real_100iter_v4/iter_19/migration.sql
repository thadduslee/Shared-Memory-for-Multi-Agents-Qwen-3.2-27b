CREATE TABLE IF NOT EXISTS deletion_request_audit (
  audit_id             INTEGER PRIMARY KEY AUTOINCREMENT,
  episode_id           TEXT NOT NULL,
  request_turn_id      TEXT NOT NULL,
  requested_by         TEXT NOT NULL,
  matched_record_id    TEXT NOT NULL,
  overlap_content      INTEGER NOT NULL DEFAULT 0,
  overlap_loc_only     INTEGER NOT NULL DEFAULT 0,
  shares_phone_digest  INTEGER NOT NULL DEFAULT 0,
  decision             TEXT NOT NULL,
  ts                   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_deletion_audit_matched ON deletion_request_audit(matched_record_id);
-- Rationale: serves the per-record post-mortem over which earlier record a
-- *_deleted_*_line request actually tombstoned; the loop currently has zero
-- visibility into deletion decisions. Bump SCHEMA_VERSION in store.py to 7.
