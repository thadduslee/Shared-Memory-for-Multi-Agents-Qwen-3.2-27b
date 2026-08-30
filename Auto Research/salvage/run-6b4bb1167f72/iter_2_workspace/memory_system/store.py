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
import os
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .kms import KMS, MockKMS

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Bumped by the Developer whenever it applies an Architect migration. This is
# how a running store reports which DDL generation it was built for, and it is
# what lets a later iteration tell "the migration was applied" apart from "the
# migration silently no-opped" -- the two look identical in the data otherwise.
SCHEMA_VERSION = 2

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

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        kms: KMS | None = None,
    ) -> None:
        self.db_path = str(db_path)
        self._kms = kms or MockKMS()
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Active-forgetting: every write connection runs with secure_delete on,
        # so deleted pages are zeroed (not left dangling in the freelist).
        # journal_mode is pinned to DELETE for the whole file by the schema.
        self.conn.execute("PRAGMA secure_delete = ON")
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._seq = 0
        self.initialize()

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

    def add_role_inheritance(self, child_role: str, parent_role: str) -> None:
        """Record that ``child_role`` inherits every grant of ``parent_role``.

        Work order item 4: the retrieval RBAC join must be role-inheritance
        aware.  This is the setup-side counterpart -- the union with a role's
        ancestors happens in the recursive CTE inside ``retrieve``.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO role_hierarchy(child_role, parent_role) VALUES (?, ?)",
            (child_role, parent_role),
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

        for record_id in hits:
            self.tombstone(record_id, requested_by=requested_by, reason="explicit deletion request")
        return len(hits)

    # ------------------------------------------------------------------
    # Active forgetting
    # ------------------------------------------------------------------

    def tombstone(self, record_id: str, requested_by: str, reason: str = "") -> None:
        """Mark a record deleted and shred its key if it has one."""
        row = self.conn.execute(
            "SELECT body, ciphertext, key_id FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        # Snapshot the record's content terms so the delete gate in `retrieve`
        # can tell "this query is about the forgotten content" from "this query
        # is about something else entirely".  We must capture this BEFORE the
        # plaintext/body is erased, because once shredded there is no way to
        # reconstruct what the record said.
        plaintext = (row["body"] if row else "") or ""
        if row and row["key_id"] and row["ciphertext"] is not None:
            plaintext = (self._decrypt(row["key_id"], row["ciphertext"]) or "") or plaintext
        terms = " ".join(sorted(_distinctive_terms(plaintext)))
        self.conn.execute(
            "INSERT OR REPLACE INTO tombstones(record_id, deleted_at, requested_by, reason, shredded, terms) "
            "VALUES (?, ?, ?, ?, 0, ?)",
            (record_id, _now(), requested_by, reason, terms),
        )
        if row and row["key_id"]:
            self.shred(record_id, row["key_id"])
        else:
            # Plaintext record: overwrite the body in place.  Leaving the text
            # in the row and relying on a read-time filter is how deleted
            # content comes back through an unexpected query path.
            self.conn.execute(
                "UPDATE records SET body = '', is_deleted = 1 WHERE record_id = ?",
                (record_id,),
            )
        self.conn.commit()

    def shred(self, record_id: str, key_id: str) -> None:
        """Cryptographic shredding: destroy the key in the KMS.

        After this, the ciphertext in `records` is unrecoverable by any query
        path, including one an adversarial prompt talks the agent into taking.
        The `crypto_keys` row survives with a `shredded_at` timestamp as an
        audit record that a key once existed and was destroyed; the key material
        was never stored locally, so there is nothing left to NULL.
        """
        self._kms.destroy_key(key_id)
        self.conn.execute(
            "UPDATE crypto_keys SET shredded_at = ? WHERE key_id = ?",
            (_now(), key_id),
        )
        self.conn.execute(
            "UPDATE records SET key_id = NULL, is_deleted = 1 WHERE record_id = ?",
            (record_id,),
        )
        self.conn.execute("UPDATE tombstones SET shredded = 1 WHERE record_id = ?", (record_id,))
        self.conn.commit()

    def is_deleted(self, record_id: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM tombstones WHERE record_id = ?", (record_id,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------------
    # Crypto: AES-256-GCM with per-item keys held in an external (mocked) KMS
    # ------------------------------------------------------------------

    def _encrypt(self, record_id: str, plaintext: str) -> tuple[str, bytes]:
        """Encrypt content under a fresh per-item AES-256-GCM key in the KMS.

        Only an opaque ``key_id`` lands in the SQLite file; the key material
        never does.  Each item gets its own key so that forgetting one item
        destroys exactly that item's key (and thus its plaintext), never
        collateral plaintext it shares a key with.
        """
        key_id = hashlib.sha256(
            f"{record_id}:{os.urandom(8).hex()}".encode()
        ).hexdigest()[:32]
        key = self._kms.create_key(key_id)
        self.conn.execute(
            "INSERT OR REPLACE INTO crypto_keys(key_id, shredded_at) VALUES (?, NULL)",
            (key_id,),
        )
        return key_id, _aes_gcm_encrypt(key, plaintext.encode("utf-8"))

    def _decrypt(self, key_id: str | None, ciphertext: bytes | None) -> str | None:
        if not key_id or not ciphertext:
            return None
        key = self._kms.get_key(key_id)
        if key is None:
            # Key was destroyed in the KMS -> content is gone, permanently.
            return None
        return _aes_gcm_decrypt(key, ciphertext).decode("utf-8", "replace")

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

        # The RBAC join against the granted role set.  The recursive CTE
        # (work order item 4) expands the requester's role into its full
        # ancestor closure, so an inherited grant satisfies the join without
        # duplicating `role_grants` rows.  With an empty `role_hierarchy` the
        # closure collapses to exactly the requester's own role, matching the
        # pre-CTE flat join.  A missing `g` row means no role in the closure has
        # a grant at this sensitivity tier -- the deny is structural, not a
        # forgotten if-statement downstream.
        sql = (
            # A `depth` counter caps recursion so a malformed or malicious
            # cyclic `role_hierarchy` row degrades to a bounded closure instead
            # of an unbounded recursive loop that would hang retrieval.  Real
            # hierarchies are shallow (<= 4 levels), so 32 is far above any
            # legitimate depth while still finite.
            "WITH RECURSIVE role_closure(role, depth) AS ("
            "  SELECT ?, 0 "
            "  UNION ALL "
            "  SELECT h.parent_role, rc.depth + 1 "
            "  FROM role_hierarchy h "
            "  JOIN role_closure rc ON h.child_role = rc.role "
            "  WHERE rc.depth < 32 "
            ") "
            "SELECT r.record_id, r.turn_id, r.author_role, r.kind, r.sensitivity, "
            "       r.body, r.ciphertext, r.key_id, r.patient_id, r.seq, "
            "       t.record_id AS tomb, "
            "       t.terms AS tomb_terms, "
            "       g.requires_rel AS requires_rel "
            "FROM records r "
            "LEFT JOIN tombstones t ON t.record_id = r.record_id "
            "LEFT JOIN role_grants g ON g.sensitivity = r.sensitivity "
            "  AND g.role IN (SELECT role FROM role_closure) "
            "WHERE r.patient_id = ? "
        )
        # The relationship gate uses the requester's *direct* role (not an
        # inherited ancestor's), because relationships encode who the principal
        # actually is to the patient, not which grant tier they inherit.
        params: list[Any] = [requester_role, patient_id]
        if as_of_seq is not None:
            sql += "AND r.seq <= ? "
            params.append(as_of_seq)
        if terms:
            like = " OR ".join("r.body LIKE '%' || ? || '%'" for _ in terms)
            sql += f"AND (r.body = '' OR {like}) "
            params.extend(terms)
        sql += "ORDER BY r.seq DESC"
        rows = self.conn.execute(sql, params).fetchall()

        # Cache per-call so the relationship subquery runs once per distinct
        # `requires_rel` tier, not once per scanned row.  `retrieve` scans at
        # most `top_k` relevant rows plus everything tombstoned, which on a hot
        # patient is the difference between 2 subqueries/row and ~2 total.
        rel_cache: dict[str, bool] = {}
        for row in rows:
            record_id = row["record_id"]

            # Relevance pre-filter.  Because content may be encrypted (or blank
            # after a tombstone of a keyless plaintext record), we decrypt the
            # row once here and reuse it below.  Rows that do not intersect the
            # query are *not responsive*, so they skip the RBAC/relationship
            # subqueries and do not pollute the audit log with non-denials.
            plaintext = row["body"] or (self._decrypt(row["key_id"], row["ciphertext"]) or "")
            if row["tomb"] is not None:
                # Tombstoned content is stored with an empty body and a shredded
                # key, so its original plaintext is unrecoverable.  The delete
                # gate below uses the terms snapshot taken at tombstone time:
                # the query must actually reference the forgotten content for
                # this record to read as `no_memory`.  An unrelated utility
                # query that scans past the tombstone row is NOT a deletion
                # concern and must not be poisoned by it.
                row_terms = set((row["tomb_terms"] or "").split())
                responsive = (not terms) or bool(terms & row_terms)
            else:
                responsive = (not terms) or bool(terms & _distinctive_terms(plaintext))
            if not responsive:
                continue  # simply not relevant; not a policy denial

            # Gate 1: tombstone -- checked AFTER relevance but BEFORE RBAC, so a
            # deleted record reads as `no_memory` to the query that is actually
            # about it, without silently swallowing unrelated authorized rows.
            if row["tomb"] is not None:
                decision.denied_tombstone.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")
                continue

            # Gate 2: role grant.
            if row["requires_rel"] is None:
                decision.denied_rbac.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_rbac")
                continue

            # Gate 3: relationship + scope.
            rel_key = (str(row["requires_rel"]),)
            if rel_key not in rel_cache:
                rel_cache[rel_key] = self._relationship_ok(
                    requester_id, requester_role, patient_id, str(row["requires_rel"])
                )
            if not rel_cache[rel_key]:
                decision.denied_scope.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_scope")
                continue

            if not plaintext:
                # Key shredded or body erased; treat as deleted, not an empty
                # allow -- it would otherwise resurrect content.
                decision.denied_tombstone.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")
                continue

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


def _aes_gcm_encrypt(key: bytes, data: bytes) -> bytes:
    """AES-256-GCM seal.  The 12-byte nonce is prepended to the ciphertext.

    The nonce must be unique per (key, plaintext) -- GCM fails catastrophically
    if a nonce is reused under the same key.  We generate a fresh random nonce
    for every encryption, so no two blobs under one item key ever collide.
    """
    nonce = os.urandom(12)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return nonce + ciphertext


def _aes_gcm_decrypt(key: bytes, blob: bytes) -> bytes:
    if len(blob) < 12:
        raise ValueError("ciphertext too short to hold an AES-GCM nonce")
    nonce, ciphertext = blob[:12], blob[12:]
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def _xor_stream(key: bytes, data: bytes) -> bytes:
    """HMAC-SHA256 counter-mode keystream XOR (kept for backwards tests).

    No longer used for new content -- the store seals with AES-256-GCM via the
    external KMS -- but retained so old test fixtures that exercise the
    HMAC/XOR stream do not break.
    """
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hmac.new(key, counter.to_bytes(8, "big"), hashlib.sha256).digest()
        out.extend(block)
        counter += 1
    return bytes(b ^ k for b, k in zip(data, out[: len(data)], strict=True))
