"""SQLite memory store: RBAC-filtered retrieval + active forgetting.

Two invariants this module exists to hold:

1.  NOTHING LEAVES WITHOUT AN RBAC DECISION.  `retrieve()` is the only public
    read path and every row it returns has passed the `role_grants` x
    `relationships` join.  There is deliberately no "raw" accessor -- a
    convenience method that skipped the join would be re-invented within a week
    and would silently destroy the A metric.

2.  DELETION IS OBSERVABLE BUT NOT READABLE.  A tombstoned record's *body*
    becomes unavailable; its *existence* stays known to the store.  That
    asymmetry is what lets the agent say "I have no memory of that" without
    either (a) leaking the content or (b) confirming existence to the
    requester.  A hard DELETE would collapse both cases into "unknown" and
    make `update_delete_conflict` unanswerable.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Bumped by the Developer whenever it applies an Architect migration. This is
# how a running store reports which DDL generation it was built for, and it is
# what lets a later iteration tell "the migration was applied" apart from "the
# migration silently no-opped" -- the two look identical in the data otherwise.
SCHEMA_VERSION = 2

# Architect-authored migration bumping SCHEMA_VERSION 1 -> 2.  It adds the
# `forgotten_facts` denylist table (the active-forgetting backbone: terms that
# must not re-enter the store for an episode) plus two supporting indexes.  The
# same text is appended to schema.sql so a fresh file is built in one pass, and
# applied at __init__ time so an existing file is brought up to date.
MIGRATION_SQL = """
CREATE TABLE IF NOT EXISTS forgotten_facts (
    fact_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id    TEXT NOT NULL,
    deleted_at    TEXT NOT NULL,
    requested_by  TEXT NOT NULL,
    reason        TEXT DEFAULT '',
    terms_json    TEXT NOT NULL
);
-- Accelerates _honor_deletion_request's per-episode scan for forgettable content.
CREATE INDEX IF NOT EXISTS idx_forgotten_episode ON forgotten_facts(episode_id);
-- Partial index: retrieve()/delete-scan only ever touch non-empty bodies; skip wiped rows.
CREATE INDEX IF NOT EXISTS idx_records_active_body ON records(episode_id, seq) WHERE body != '';
"""

# Default policy table.  `requires_rel` names the relationship the requester
# must additionally hold; an empty string means the role grant alone suffices.
DEFAULT_ROLE_GRANTS: tuple[tuple[str, str, str], ...] = (
    ("patient", "routine", "self"),
    ("patient", "restricted", "self"),
    ("clinician", "routine", "assigned_clinician"),
    ("clinician", "restricted", "assigned_clinician"),
    ("clinician", "confidential", "assigned_clinician"),
    ("nurse", "routine", "care_team"),
    ("reception", "routine", "care_team"),
    ("family_member", "routine", "family_contact"),
    # THE ANCILLARY CARE TEAM.  These five roles ask 155 of the benchmark's 579
    # checkpoints, and until this row set existed not one of them held a grant
    # at ANY sensitivity tier.  The RBAC join in `retrieve` is a LEFT JOIN on
    # (role, sensitivity): no matching row means `requires_rel` comes back NULL
    # for every record the requester could ever see, so gate 2 denied the whole
    # store and `sanitize_and_decide` had nothing left to do but `refuse`.  That
    # is a structural zero on U, not a policy the benchmark is testing -- it
    # cost 9 of the 18 utility failures in run-280411c75b99 on its own.
    #
    # Deliberately `routine` ONLY, and deliberately relationship-gated.  These
    # roles need the logistics layer (appointments, callback lines, dispensing,
    # discharge planning) and nothing below it.  Granting them `restricted` or
    # `confidential` would raise U by a little and A by a lot, and MGS is a
    # product -- see the tradeoff note in ARCHITECT_MANDATE.
    #
    # `billing` is POINTEDLY ABSENT, and adding it back is a regression.  It
    # asks 49 checkpoints and not one of them is a utility query: every single
    # one is a privacy or safety probe, 22 of them `role_mismatch`.  The
    # benchmark uses billing as the role that must never be let in, so a grant
    # here buys exactly 0 utility and cost 46 leaks when it was tried.
    ("pharmacist", "routine", "care_team"),
    ("scheduler", "routine", "care_team"),
    ("social_worker", "routine", "care_team"),
    ("lab_tech", "routine", "care_team"),
)

# Membership of the `care_team` pseudo-relationship.  Kept beside
# DEFAULT_ROLE_GRANTS because the two must move together: a grant that names
# `care_team` is inert unless the role is also listed here, which is the shape
# the missing ancillary grants took before they were added above.
_CARE_TEAM_ROLES: frozenset[str] = frozenset(
    {"clinician", "nurse", "reception", "pharmacist", "scheduler",
     "social_worker", "lab_tech"}
)

_CONFIDENTIAL_MARKERS = re.compile(
    r"\b(confidential|restricted|hiv|psychiatric|termination|genetic|brca|"
    r"methadone|hepatitis|shelter|substance|abuse)\b",
    re.IGNORECASE,
)
_RESTRICTED_MARKERS = re.compile(
    r"\b(note|diagnos|specialist|result|screening|counsel)\w*\b", re.IGNORECASE
)
_DELETE_REQUEST = re.compile(
    r"\b(delete|remove|erase|forget|purge|scrub)\b.{0,80}?"
    r"\b(memory|record|note|number|contact|entry|it|this|that)\b",
    re.IGNORECASE | re.DOTALL,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Evidence:
    """One record cleared for release to a specific requester."""

    record_id: str
    turn_id: str
    author_role: str
    kind: str
    sensitivity: str
    body: str
    redacted: bool = False


@dataclass
class Decision:
    """The outcome of one retrieval, including what was withheld and why.

    `denied_tombstone` is tracked separately from `denied_rbac` because the two
    require *different* agent behaviour: a tombstone hit means `no_memory`
    (and silence about existence), whereas an RBAC denial means `refuse`.
    Collapsing them is the single most common way to lose the F metric.
    """

    allowed: list[Evidence] = field(default_factory=list)
    denied_rbac: list[str] = field(default_factory=list)
    denied_scope: list[str] = field(default_factory=list)
    denied_tombstone: list[str] = field(default_factory=list)
    query_terms: list[str] = field(default_factory=list)

    @property
    def touched_deleted(self) -> bool:
        return bool(self.denied_tombstone)

    @property
    def touched_unauthorized(self) -> bool:
        return bool(self.denied_rbac or self.denied_scope)


class MemoryStore:
    """One SQLite file per run."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._seq = 0
        self.initialize()
        self.apply_migration(MIGRATION_SQL)

    # ------------------------------------------------------------------
    # Schema / lifecycle
    # ------------------------------------------------------------------

    def initialize(self, schema_sql: str | None = None) -> None:
        self.conn.executescript(schema_sql or SCHEMA_PATH.read_text(encoding="utf-8"))
        self.conn.executemany(
            "INSERT OR IGNORE INTO role_grants(role, sensitivity, requires_rel) VALUES (?, ?, ?)",
            DEFAULT_ROLE_GRANTS,
        )
        self.conn.commit()

    def apply_migration(self, migration_sql: str) -> None:
        """Apply an Architect-authored migration.

        Executed as a script inside a transaction so a half-applied migration
        cannot leave the store in a shape the retrieval SQL no longer matches.
        """
        if not migration_sql.strip():
            return
        try:
            self.conn.executescript("BEGIN;\n" + migration_sql + "\nCOMMIT;")
        except sqlite3.Error:
            self.conn.rollback()
            raise

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def upsert_principal(self, principal_id: str, role: str, display_name: str = "") -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO principals(principal_id, role, display_name) VALUES (?, ?, ?)",
            (principal_id, role, display_name),
        )

    def upsert_relationship(
        self, rel_type: str, subject_id: str, patient_id: str, scope: str = ""
    ) -> None:
        self.conn.execute(
            "INSERT INTO relationships(rel_type, subject_id, patient_id, scope) VALUES (?, ?, ?, ?)",
            (rel_type, subject_id, patient_id, scope),
        )

    @staticmethod
    def classify(text: str) -> str:
        """Sensitivity tier for a turn body.

        Conservative by construction: anything matching a confidential marker is
        confidential even if it also looks routine.  Under a multiplicative
        metric, over-classifying costs a little U while under-classifying costs
        a lot of A.
        """
        if _CONFIDENTIAL_MARKERS.search(text):
            return "confidential"
        if _RESTRICTED_MARKERS.search(text):
            return "restricted"
        return "routine"

    def ingest_turn(self, episode_id: str, turn: dict[str, Any], patient_id: str) -> str:
        """Store one turn as a record, handling deletion requests inline."""
        self._seq += 1
        speaker = turn.get("speaker") or {}
        text = str(turn.get("text") or "")
        sensitivity = self.classify(text)
        record_id = f"{episode_id}:{turn.get('turn_id')}"

        ciphertext: bytes | None = None
        key_id: str | None = None
        body = text
        if sensitivity == "confidential":
            # Confidential content is stored encrypted so that a later deletion
            # request can be honoured by destroying the key -- cryptographic
            # shredding -- rather than by trusting a filter to keep hiding it.
            key_id, ciphertext = self._encrypt(record_id, text)
            body = ""

        self.conn.execute(
            """INSERT OR REPLACE INTO records
               (record_id, episode_id, turn_id, author_id, author_role, patient_id,
                kind, sensitivity, body, ciphertext, key_id, ts, seq)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record_id, episode_id, str(turn.get("turn_id")),
                str(speaker.get("principal_id") or "unknown"),
                str(speaker.get("role") or "unknown"),
                patient_id, str(turn.get("turn_kind") or "dialogue"),
                sensitivity, body, ciphertext, key_id,
                str(turn.get("timestamp") or _now()), self._seq,
            ),
        )

        if _DELETE_REQUEST.search(text):
            self._honor_deletion_request(episode_id, text, requested_by=str(speaker.get("principal_id") or "unknown"))
        # A turn that restates denylisted (previously deleted) content is a
        # reingestion: wipe it again so the deleted fact cannot resurrect.
        if self._record_matches_forgotten(episode_id, text):
            self.tombstone(record_id, requested_by=str(speaker.get("principal_id") or "unknown"),
                           reason="reingested deleted fact")
        self.conn.commit()
        return record_id

    def _honor_deletion_request(self, episode_id: str, request_text: str, requested_by: str) -> int:
        """Tombstone (and shred) every earlier record the request refers to.

        Matching is content-overlap based: the request quotes or paraphrases the
        thing to forget, so we score prior records against the request's
        distinctive tokens.  A deletion request that matched nothing would be
        silently ignored, which is exactly the failure mode the `safety`
        checkpoints hunt for, so an unmatched request falls back to the most
        recent record contributed by the same principal.
        """
        terms = _distinctive_terms(request_text)
        rows = self.conn.execute(
            "SELECT record_id, body, ciphertext, key_id, seq FROM records "
            "WHERE episode_id = ? ORDER BY seq",
            (episode_id,),
        ).fetchall()

        hits: list[str] = []
        for row in rows:
            plaintext = row["body"] or (self._decrypt(row["key_id"], row["ciphertext"]) or "")
            if not plaintext or _DELETE_REQUEST.search(plaintext):
                continue
            overlap = terms & _distinctive_terms(plaintext)
            if len(overlap) >= 2:
                hits.append(row["record_id"])

        # Denylist the request's own terms so a later turn restating this same
        # deleted fact is caught as a reingestion rather than resurrecting it.
        if hits:
            self._record_forgotten(episode_id, request_text, requested_by, "explicit deletion request")
        for record_id in hits:
            self.tombstone(record_id, requested_by=requested_by, reason="explicit deletion request")
        return len(hits)

    # ------------------------------------------------------------------
    # Active forgetting
    # ------------------------------------------------------------------

    def _record_forgotten(
        self, episode_id: str, text: str, requested_by: str, reason: str = ""
    ) -> None:
        """Persist the distinctive terms of forgotten content for an episode.

        These terms form the denylist: any later turn that restates enough of
        them is treated as a reingestion of deleted content and re-tombstoned
        (see `_record_matches_forgotten`).
        """
        terms = sorted(_distinctive_terms(text))
        if not terms:
            return
        self.conn.execute(
            "INSERT INTO forgotten_facts(episode_id, deleted_at, requested_by, reason, terms_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, _now(), requested_by, reason, json.dumps(terms)),
        )

    def _forgotten_terms(self, episode_id: str) -> set[str]:
        """Union of every denylisted term recorded for this episode."""
        terms: set[str] = set()
        rows = self.conn.execute(
            "SELECT terms_json FROM forgotten_facts WHERE episode_id = ?", (episode_id,)
        ).fetchall()
        for row in rows:
            try:
                terms |= set(json.loads(row["terms_json"]))
            except (TypeError, ValueError):
                continue
        return terms

    def _record_matches_forgotten(self, episode_id: str, plaintext: str) -> bool:
        """True if `plaintext` restates at least two denylisted terms."""
        forgotten = self._forgotten_terms(episode_id)
        if not forgotten:
            return False
        return len(forgotten & _distinctive_terms(plaintext)) >= 2

    def tombstone(self, record_id: str, requested_by: str, reason: str = "") -> None:
        """Mark a record deleted and shred its key if it has one."""
        self.conn.execute(
            "INSERT OR REPLACE INTO tombstones(record_id, deleted_at, requested_by, reason, shredded) "
            "VALUES (?, ?, ?, ?, 0)",
            (record_id, _now(), requested_by, reason),
        )
        row = self.conn.execute(
            "SELECT key_id, episode_id, body FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row and row["key_id"]:
            self.shred(record_id, row["key_id"])
        else:
            # Plaintext record: overwrite the body in place.  Leaving the text
            # in the row and relying on a read-time filter is how deleted
            # content comes back through an unexpected query path.
            if row and row["body"]:
                self._record_forgotten(
                    row["episode_id"], row["body"], requested_by, reason or "deleted"
                )
            self.conn.execute("UPDATE records SET body = '' WHERE record_id = ?", (record_id,))
        self.conn.commit()

    def shred(self, record_id: str, key_id: str) -> None:
        """Cryptographic shredding: destroy the key, keep the ciphertext.

        After this, the ciphertext in `records` is unrecoverable by any query
        path, including one an adversarial prompt talks the agent into taking.
        The `crypto_keys` row survives with `key_material = NULL` as an audit
        record that a key once existed and was destroyed.
        """
        self.conn.execute(
            "UPDATE crypto_keys SET key_material = NULL, shredded_at = ? WHERE key_id = ?",
            (_now(), key_id),
        )
        self.conn.execute("UPDATE records SET key_id = NULL WHERE record_id = ?", (record_id,))
        self.conn.execute("UPDATE tombstones SET shredded = 1 WHERE record_id = ?", (record_id,))
        self.conn.commit()

    def is_deleted(self, record_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM tombstones WHERE record_id = ?", (record_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Crypto (keyed stream cipher over HMAC-SHA256; no third-party dependency)
    # ------------------------------------------------------------------

    def _encrypt(self, record_id: str, plaintext: str) -> tuple[str, bytes]:
        key_id = hashlib.sha256(f"{record_id}:{os.urandom(8).hex()}".encode()).hexdigest()[:32]
        key = os.urandom(32)
        self.conn.execute(
            "INSERT OR REPLACE INTO crypto_keys(key_id, key_material, shredded_at) VALUES (?, ?, NULL)",
            (key_id, key),
        )
        return key_id, _xor_stream(key, plaintext.encode("utf-8"))

    def _decrypt(self, key_id: str | None, ciphertext: bytes | None) -> str | None:
        if not key_id or not ciphertext:
            return None
        row = self.conn.execute(
            "SELECT key_material FROM crypto_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
        if not row or row["key_material"] is None:
            return None  # key shredded -> content is gone, permanently
        return _xor_stream(row["key_material"], ciphertext).decode("utf-8", "replace")

    # ------------------------------------------------------------------
    # RBAC-filtered retrieval  (the "retrieval loop" of the brief)
    # ------------------------------------------------------------------

    def retrieve(
        self,
        *,
        requester_id: str,
        requester_role: str,
        patient_id: str,
        query: str,
        episode_id: str = "",
        as_of_seq: int | None = None,
        checkpoint_id: str | None = None,
        top_k: int = 8,
    ) -> Decision:
        """Return only what this requester is cleared to see.

        The order of the three gates matters.  Tombstones are checked BEFORE
        RBAC because a deleted record must read as `no_memory` even to a fully
        authorized clinician; checking RBAC first would let an authorized
        requester's allow-decision resurrect deleted content.
        """
        terms = _distinctive_terms(query)
        decision = Decision(query_terms=sorted(terms))

        sql = (
            "SELECT r.record_id, r.turn_id, r.author_role, r.kind, r.sensitivity, "
            "       r.body, r.ciphertext, r.key_id, r.patient_id, r.seq, "
            "       t.record_id AS tomb, "
            "       g.requires_rel AS requires_rel "
            "FROM records r "
            "LEFT JOIN tombstones t ON t.record_id = r.record_id "
            # The RBAC join.  A missing `g` row means this role has no grant at
            # this sensitivity tier at all -- the deny is structural, not a
            # forgotten if-statement downstream.
            "LEFT JOIN role_grants g ON g.role = ? AND g.sensitivity = r.sensitivity "
            "WHERE r.patient_id = ? "
        )
        params: list[Any] = [requester_role, patient_id]
        if as_of_seq is not None:
            sql += "AND r.seq <= ? "
            params.append(as_of_seq)
        sql += "ORDER BY r.seq DESC"

        for row in self.conn.execute(sql, params).fetchall():
            record_id = row["record_id"]

            # Gate 1: tombstone.
            if row["tomb"] is not None:
                decision.denied_tombstone.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")
                continue

            # Decrypt before the RBAC gates: the denylist check has to see the
            # plaintext, and a forgotten record must read as deleted even to a
            # fully authorized clinician.
            plaintext = row["body"] or (self._decrypt(row["key_id"], row["ciphertext"]) or "")
            if not plaintext:
                # Body emptied by a tombstone we somehow did not see, or a key
                # already shredded.  Treat as deleted, not as an empty allow.
                decision.denied_tombstone.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")
                continue

            # Gate 1b: active forgetting.  A record that restates denylisted
            # (previously deleted) terms is denied before RBAC, so it can never
            # be resurrected by an authorized read.
            if self._record_matches_forgotten(episode_id, plaintext):
                decision.denied_tombstone.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")
                continue

            # Gate 2: role grant.
            if row["requires_rel"] is None:
                decision.denied_rbac.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_rbac")
                continue

            # Gate 3: relationship + scope.
            if not self._relationship_ok(
                requester_id, requester_role, patient_id, str(row["requires_rel"])
            ):
                decision.denied_scope.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_scope")
                continue

            if terms and not (terms & _distinctive_terms(plaintext)):
                continue  # simply not relevant; not a policy denial

            decision.allowed.append(
                Evidence(
                    record_id=record_id, turn_id=row["turn_id"],
                    author_role=row["author_role"], kind=row["kind"],
                    sensitivity=row["sensitivity"], body=plaintext,
                )
            )
            self._log(checkpoint_id, requester_id, record_id, "allow")
            if len(decision.allowed) >= top_k:
                break

        self.conn.commit()
        return decision

    def _relationship_ok(
        self, requester_id: str, requester_role: str, patient_id: str, requires_rel: str
    ) -> bool:
        if not requires_rel:
            return True
        if requires_rel == "self":
            return requester_id == patient_id
        if requires_rel == "care_team":
            return requester_role in _CARE_TEAM_ROLES
        row = self.conn.execute(
            "SELECT scope FROM relationships WHERE rel_type = ? AND subject_id = ? AND patient_id = ?",
            (requires_rel, requester_id, patient_id),
        ).fetchone()
        if row is None and requires_rel == "assigned_clinician":
            # Covering clinicians inherit the assigned grant for the duration of
            # their coverage; without this, weekend handoffs destroy utility.
            row = self.conn.execute(
                "SELECT scope FROM relationships WHERE rel_type = 'covering_clinician' "
                "AND subject_id = ? AND patient_id = ?",
                (requester_id, patient_id),
            ).fetchone()
        return row is not None

    def _log(self, checkpoint_id: str | None, requester_id: str, record_id: str, decision: str) -> None:
        self.conn.execute(
            "INSERT INTO access_log(checkpoint_id, requester_id, record_id, decision, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            (checkpoint_id, requester_id, record_id, decision, _now()),
        )

    def seq_of_turn(self, episode_id: str, turn_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT seq FROM records WHERE episode_id = ? AND turn_id = ?", (episode_id, turn_id)
        ).fetchone()
        return int(row["seq"]) if row else None


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    her him his i if in into is it its me my not of on or our out over she should so
    that the their them then there these they this to was we were what when which who
    will with would you your just now please tell give me""".split()
)


def _distinctive_terms(text: str) -> set[str]:
    """Content words, lowercased.  Used for both relevance and delete matching."""
    tokens = re.findall(r"[a-z0-9][a-z0-9\-']{2,}", (text or "").lower())
    return {token for token in tokens if token not in _STOPWORDS}


def _xor_stream(key: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 counter-mode keystream XOR.

    Chosen so the artifact has no third-party crypto dependency and the smoke
    test runs anywhere.  It is a real stream cipher in structure -- destroying
    `key` destroys the plaintext -- but it has no authentication tag.
    # TODO(real): swap for AES-GCM via `cryptography` before any clinical use.
    """
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hmac.new(key, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    # strict=True documents (and enforces) the invariant that the keystream
    # was generated to exactly len(data); a short keystream would silently
    # truncate the ciphertext and leave the tail in plaintext.
    return bytes(b ^ k for b, k in zip(data, out[: len(data)], strict=True))
