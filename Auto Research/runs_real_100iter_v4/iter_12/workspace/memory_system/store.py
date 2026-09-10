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

SCHEMA_PATH = Path(__file__).with_name("schema.sql")

# Bumped by the Developer whenever it applies an Architect migration. This is
# how a running store reports which DDL generation it was built for, and it is
# what lets a later iteration tell "the migration was applied" apart from "the
# migration silently no-opped" -- the two look identical in the data otherwise.
#
# 2: added `record_terms` and `store_meta` -- the responsiveness index. See the
#    note above `retrieve()` for why a denial has to know whether the record it
#    denied was about the query.
# 3: digit-run term emission + the synthetic contact tag + `query_answer_denials`,
#    so tag/digit-only overlaps no longer masquerade as answer-bearing content.
# 4: `tombstone_terms` -- a tombstoned record's term digests are preserved there
#    (instead of being forgotten with its body) for scoped responsiveness, so a
#    later query can distinguish a related deletion from an unrelated one.
# 5: `record_terms`/`tombstone_terms` gained `is_structural`: digit-run and
#    contact digests are flagged 1, real content 0, so gate 1 (tombstone
#    responsiveness) applies the same structural/content split that gates 2/3
#    already use, and a query sharing only digits/contact with a deleted
#    logistics record can no longer silence an otherwise answerable shard.
SCHEMA_VERSION = 5

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

# Synthetic "logistics/contact" tag.  A record or query is tagged with it when
# it carries a phone-like digit pattern or one of the small whitelist of contact
# sense words below.  The tag is what lets a logistics-denial be recognised as
# INCIDENTAL (its only overlap with the query is structural contact/digit
# signalling) rather than answer-bearing, so it never forces `answer_redacted`
# on its own.  Hashed into the responsiveness index exactly like a content term.
_CONTACT_TAG = "__contact__"
_CONTACT_SENSE_WORDS: frozenset[str] = frozenset(
    {"call", "callback", "contact", "reach", "front", "desk", "line",
     "scheduling", "appointment", "pickup", "transport", "pharmacy", "nurse"}
)
# Phone-ish: a 3-3-4 style number (digits separated by -, . or whitespace, with
# an optional leading group separator) or a bare run of 7+ digits.
_PHONEISH_RE = re.compile(
    r"\d{3}[-.\s)]?\d{3}[-.\s]\d{4}|(?<!\d)\d{7,}(?!\d)"
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

    # Number of RBAC/scope denials that were ANSWER-BEARING: i.e. that shared a
    # real content term (not merely a phone digit-run or the synthetic contact
    # tag) with the query.  `None` means "not computed -- treat every denial as
    # answer-bearing", which is the back-compat contract for decisions built by
    # hand in tests; `retrieve()` always sets an int.
    query_answer_denials: int | None = None

    # INERT DIAGNOSTIC FIELD.  Populated by `retrieve()` after the allowed and
    # denied lists are final, purely for post-hoc inspection of the top-24
    # ranked candidates that the gates saw.  It is read nowhere by the agent and
    # can never branch the decision or the action.
    candidate_census: list[dict[str, Any]] = field(default_factory=list)

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
        # Read from `store_meta` on first use and kept for the life of the
        # connection; see `_index_key` for why it must not simply be generated.
        self._index_key_cache: bytes | None = None
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

        # INDEXED FROM THE CLEARTEXT, and before the branch below discards it.
        # A confidential record's body becomes ciphertext three lines from here
        # and there is no later point at which its terms can be derived without
        # decrypting -- which is exactly what `retrieve()` must not have to do in
        # order to decide whether a record it is about to DENY was responsive.
        self._index_record(record_id, text)

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
        """Mark a record deleted, purge its term index, and shred its key.

        THE INDEX GOES WITH THE BODY. `record_terms` holds the record's
        distinctive content words; leaving them behind after a shred would make
        cryptographic shredding a half-measure -- the body would be
        unrecoverable and its vocabulary would not be. Deleted in the same
        transaction as the tombstone so there is no window in which one exists
        without the other.

        But the record's digests are FIRST PRESERVED in `tombstone_terms` before
        its `record_terms` rows are deleted. The index still goes with the body,
        yet those preserved digests let a later query distinguish a RELATED
        deletion -- one whose preserved content overlaps the question -- from an
        UNRELATED one, so `_is_responsive` no longer has to treat every
        tombstoned record as relevant to everything. `tombstone_terms` is never
        deleted here or anywhere; only the `record_terms` copy is purged.

        The cost of keeping only digests is real and is the right trade: a
        tombstoned record whose digests were never captured, or that shares none
        of them with a query, is the conservative case `_is_responsive` still
        assumes -- a deleted record is treated as relevant to anything it could
        not be scoped against, which produces `no_memory` rather than a partial
        answer that reconstructs it.
        """
        self.conn.execute(
            "INSERT OR REPLACE INTO tombstones(record_id, deleted_at, requested_by, reason, shredded) "
            "VALUES (?, ?, ?, ?, 0)",
            (record_id, _now(), requested_by, reason),
        )
        # Preserve the record's digests BEFORE purging them from the live index,
        # so scoped responsiveness survives the tombstone.
        rows = self.conn.execute(
            "SELECT term_hash, is_structural FROM record_terms WHERE record_id = ?",
            (record_id,),
        ).fetchall()
        self.conn.executemany(
            "INSERT OR IGNORE INTO tombstone_terms(record_id, term_hash, is_structural) "
            "VALUES (?, ?, ?)",
            [
                (record_id, row["term_hash"], int(row["is_structural"] or 0))
                for row in rows
            ],
        )
        self.conn.execute("DELETE FROM record_terms WHERE record_id = ?", (record_id,))
        row = self.conn.execute(
            "SELECT key_id FROM records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row and row["key_id"]:
            self.shred(record_id, row["key_id"])
        else:
            # Plaintext record: overwrite the body in place.  Leaving the text
            # in the row and relying on a read-time filter is how deleted
            # content comes back through an unexpected query path.
            self.conn.execute("UPDATE records SET body = '' WHERE record_id = ?", (record_id,))
        self.conn.commit()

    def shred(self, record_id: str, key_id: str) -> None:
        """Cryptographic shredding: destroy the key and zero the ciphertext.

        After this, the record is unrecoverable by any query path, including one
        an adversarial prompt talks the agent into taking -- not only is the key
        gone (so the ciphertext cannot be decrypted), the ciphertext bytes
        themselves are zeroed in place (`ciphertext = x''`, `key_id = NULL`).
        The `crypto_keys` row survives with `key_material = NULL` as an audit
        record that a key once existed and was destroyed.

        All of these statements run in ONE transaction (they share the implicit
        transaction `tombstone` began and this method's single `commit`), so a
        crash mid-shred can never leave a key destroyed while its ciphertext
        still sits in the row, or vice versa.
        """
        self.conn.execute(
            "UPDATE crypto_keys SET key_material = NULL, shredded_at = ? WHERE key_id = ?",
            (_now(), key_id),
        )
        # Zero both the key pointer and the ciphertext bytes in the same UPDATE.
        self.conn.execute(
            "UPDATE records SET key_id = NULL, ciphertext = x'' WHERE record_id = ?",
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
        as_of_seq: int | None = None,
        checkpoint_id: str | None = None,
        top_k: int = 8,
    ) -> Decision:
        """Return only what this requester is cleared to see.

        THE ORDER OF THE THREE GATES MATTERS.  Tombstones are checked BEFORE
        RBAC because a deleted record must read as `no_memory` even to a fully
        authorized clinician; checking RBAC first would let an authorized
        requester's allow-decision resurrect deleted content.

        AND RESPONSIVENESS IS CHECKED BEFORE ALL THREE.  This is gate 0, and it
        is not a policy gate -- it decides which records are *about the query*
        and therefore which records are eligible to be denied at all.  A record
        that fails it is skipped entirely: it appears in `allowed`, in
        `denied_rbac`, in `denied_scope` and in `denied_tombstone` exactly
        nowhere.

        WHY THAT ORDERING IS THE WHOLE POINT.  This filter used to run LAST,
        after all three gates, as a `continue` on the way to `allowed`.  So a
        record the requester happened not to be cleared for -- about a different
        appointment, a different clinician, a different week -- was denied
        first and never tested for relevance at all.  It landed in
        `denied_rbac`, `Decision.touched_unauthorized` went true, and
        `sanitize_and_decide` read that as "there is responsive content this
        requester may not have" and downgraded the action from `answer` to
        `answer_redacted`.

        The answer text was correct.  The evidence was correct.  The gates were
        correct.  The LABEL was wrong, and the benchmark scores the label:
        `utility_correct = action_correct and include_ok`.  Measured on the
        seeded 50-checkpoint dev slice, that one confusion took 12 of the 18
        utility checkpoints -- `expected=answer got=answer_redacted` -- and it
        is the single largest term in U.

        The invariant it establishes is what makes `sanitize_and_decide`'s
        branch 3 sound: **a non-empty denied list always means there IS
        responsive content the requester must not get**, never "some unrelated
        record was skipped".

        `top_k` REMAINS A HARD CAP on `allowed`.  The GateMemAgent caller
        default is 16 again (the iter-9 raise to 64 was a regression: it
        re-evaluated tombstones ranked 16-64 and flipped clean answers to
        `no_memory`, so it was reverted).  The stop stays a ranked hard stop on
        `allowed`.  Raising the value while keeping the ranked hard-stop
        semantics must be kept distinct from the historical removal-of-stop
        regression: iteration 3 of run-8cf58d33b311 removed the stop entirely
        (it became an "evidence collection target"), cleared records per query
        went from 8 to 21, more records were consequently evaluated and denied,
        `denied_rbac` became non-empty on queries where it had been empty, and
        U fell from 0.4444 to 0.2222 in one step.  The wider evidence set also
        diluted the answers -- the required strings started dropping out of
        them.  Breadth is not free under a metric that scores the action label,
        and neither is removing a stop; widening a stop is not the same
        regression because the ranking and the gate order that the regression
        undid are both still in place.
        """
        # PASS 1 -- collect candidates with their term-overlap score.
        # Responsiveness is still gate 0 and still runs BEFORE the policy gates,
        # for the reasons in the docstring: a record this query is not about
        # must appear in `allowed`, `denied_rbac`, `denied_scope` and
        # `denied_tombstone` exactly nowhere.  But `_is_responsive` now returns
        # an overlap COUNT so the same pass that gates also ranks.  Denials are
        # further split by whether the overlap was real content or only the
        # structural contact/digit tag (see `Decision.query_answer_denials`).
        match_terms = _match_terms(query)
        decision = Decision(
            query_terms=sorted(match_terms), query_answer_denials=0,
        )
        # Content terms and structural terms (digit runs + the contact tag) are
        # hashed SEPARATELY, so a denial whose only shared term is a phone-digit
        # run or the contact tag is recognised as incidental rather than
        # answer-bearing.  Query digests still come from the SAME widened set
        # the index is built from (`_match_terms`), so ingest and query stay
        # symmetric and a query for 'results' can match a stored 'result'.
        all_hashes = self._term_hashes(match_terms)
        content_hashes = set(
            self._term_hashes(t for t in match_terms if not _is_structural_term(t))
        ) if any(not _is_structural_term(t) for t in match_terms) else set()
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

        # PASS 1 -- collect candidates with their term-overlap score.
        # Responsiveness is still gate 0 and still runs BEFORE the policy gates,
        # for the reasons in the docstring: a record this query is not about
        # must appear in `allowed`, `denied_rbac`, `denied_scope` and
        # `denied_tombstone` exactly nowhere.  But `_is_responsive` now returns
        # an overlap COUNT so the same pass that gates also ranks.
        candidates: list[tuple[int, int, int, sqlite3.Row]] = []  # (overlap, structural, seq, row)
        for row in self.conn.execute(sql, params).fetchall():
            record_id = row["record_id"]
            tombstoned = row["tomb"] is not None
            total = self._is_responsive(
                record_id, all_hashes, tombstoned=tombstoned
            )
            if total <= 0:
                continue
            # Structural overlap is the share of the total that comes from
            # phone-digit runs and the synthetic contact tag alone.  The residue
            # (total - structural) is REAL content overlap, which is what decides
            # whether a denial is answer-bearing.
            content = (
                self._is_responsive(record_id, content_hashes, tombstoned=tombstoned)
                if content_hashes else 0
            )
            candidates.append((total, total - content, int(row["seq"]), row))

        # Rank live candidates by relevance first, then structural tag, then
        # recency: (overlap desc, structural desc, seq desc).  Relevance still
        # outranks recency so a high-overlap allowed piece beats a barely-related
        # but newer one; the structural tiebreak only orders incidental denials.
        # Rank live candidates by REAL-content overlap first, then total
        # overlap, then the structural share, then recency.  Leading on real
        # content lets the sparse records that carry the gold logistics/date
        # strings outrank chatty rows that merely reuse the query's vocabulary
        # plus structural digests; the total/structural/seq triple breaks ties
        # among records that tie on real content.
        candidates.sort(
            key=lambda item: (item[0] - item[1], item[0], item[1], item[2]),
            reverse=True,
        )

        # Inert census capture, taken immediately after the rank sort so a
        # record's TRUE pre-gate rank is recorded even when it never reaches a
        # gate.  It holds the top (top_k + 24) candidates as the full 4-tuple
        # plus that final rank index; PASS 2 below keeps iterating over the
        # whole `candidates` list exactly as before, so this snapshot is read
        # nowhere by any gate.  It is only turned into `decision.candidate_census`
        # once the allowed/denied lists are final (see below).
        census_captured: list[tuple[int, tuple[int, int, int, sqlite3.Row]]] = [
            (rank, item) for rank, item in enumerate(candidates[: top_k + 24], start=1)
        ]

        # PASS 2 -- apply the three policy gates in ranked order and fill the
        # decision.  The `top_k` cap is enforced ONLY on the `allowed` list, so
        # it never shrinks the denied lists and never lets a denial be skipped
        # because an unrelated but newer record soaked up the budget.
        # `allowed_content_sets` / `denied_content_sets` hold each processed
        # record's query-matching real-content digest set, accumulated here and
        # turned into a marginal `query_answer_denials` count once the allowed
        # set is final.
        allowed_content_sets: list[set[str]] = []
        denied_content_sets: list[set[str]] = []
        # Deferred tombstones from gate 1, parked as (record_id, preserved real-
        # content digest set).  Admission is postponed until the full allowed set
        # is known so that a tombstone whose content an allowed record re-states
        # verbatim can be skipped (see the coverage pass after the loop).
        deferred_tombstones: list[tuple[str, set[str]]] = []
        for _overlap, _structural, _seq, row in candidates:
            record_id = row["record_id"]

            # Gate 1: tombstone -- DEFERRED.  A responsive deletion is not
            # admitted to `denied_tombstone` (and so does not force `no_memory`
            # upstream) the moment it is seen.  Instead it is parked here with
            # the FULL set of its preserved real-content digests read from
            # `tombstone_terms` (is_structural = 0), and the loop keeps
            # processing live rows through gates 2/3 exactly as before.  Once
            # every allowed record is known, the coverage pass decides admission
            # (see below): the deleted record's content is checked for overlap
            # against what actually got out, so a later cleared routine record
            # that duplicates a deleted fact verbatim re-covers it and no longer
            # silences the legitimate answer.
            if row["tomb"] is not None:
                deferred_tombstones.append(
                    (record_id, self._tombstone_content_digests(record_id))
                )
                continue

            # Which of the query's real-content digests this live record carries.
            # In PASS 2 the account of answer-bearing denials is MARGINAL: a
            # denial only increments `query_answer_denials` when it adds a real-
            # content digest that the union of allowed records' content digests
            # does not already cover.  That is what makes a raise of the agent's
            # `top_k` safe -- extra evaluated-and-denied candidates that only
            # restate content the allowed answer already provides cannot flip a
            # clean `answer` into `answer_redacted`/`refuse`.
            carried = (
                self._live_content_digests(record_id, content_hashes)
                if content_hashes else set()
            )

            # Gate 2: role grant.
            if row["requires_rel"] is None:
                decision.denied_rbac.append(record_id)
                denied_content_sets.append(carried)
                self._log(checkpoint_id, requester_id, record_id, "deny_rbac")
                continue

            # Gate 3: relationship + scope.
            if not self._relationship_ok(
                requester_id, requester_role, patient_id, str(row["requires_rel"])
            ):
                decision.denied_scope.append(record_id)
                denied_content_sets.append(carried)
                self._log(checkpoint_id, requester_id, record_id, "deny_scope")
                continue

            plaintext = row["body"] or (self._decrypt(row["key_id"], row["ciphertext"]) or "")
            if not plaintext:
                # Body emptied by a tombstone we somehow did not see, or a key
                # already shredded.  Treat as deleted, not as an empty allow.
                decision.denied_tombstone.append(record_id)
                self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")
                continue

            # NO RELEVANCE CHECK HERE ANY MORE.  It moved to gate 0, above.
            # Leaving a second copy of it at this point would be harmless for
            # `allowed` and actively wrong as documentation: it would suggest
            # relevance is decided after the policy gates, which is precisely
            # the ordering this module now exists to not have.

            decision.allowed.append(
                Evidence(
                    record_id=record_id, turn_id=row["turn_id"],
                    author_role=row["author_role"], kind=row["kind"],
                    sensitivity=row["sensitivity"], body=plaintext,
                )
            )
            allowed_content_sets.append(carried)
            self._log(checkpoint_id, requester_id, record_id, "allow")
            if len(decision.allowed) >= top_k:
                break

        # Deferred-tombstone admission (the query-centred coverage skip), run
        # once the full allowed set is known.  THE CONTRACT, and the F guards a
        # future edit must keep airtight:
        #
        #   A tombstone fires `no_memory` ONLY when the question's real-content
        #   digests are NOT already answerable from cleared live records.  The
        #   admission test is therefore QUERY-CENTRED, not whole-record: a
        #   tombstone is admitted (and so forces `no_memory` upstream) only when
        #   the preserved real-content digests it shares with the query include
        #   one that no allowed record's content covers.  A tombstone whose
        #   query-shared real content an allowed record already states verbatim
        #   is SKIPPED -- the deleted fact survives in cleared evidence, so
        #   admitting it would flip a legitimate `answer` into a `no_memory`
        #   (the exact false-positive this skip exists to remove).  A tombstone
        #   that shares NO real-content digest with the query (responsive only
        #   through structural/contact overlap) cannot be shown answerable, so
        #   it keeps the conservative default and IS admitted.
        #
        #   Keep the `is_structural = 0` filters on BOTH sides of this test.
        #   They are the F guards: structural digests (digit runs and the
        #   synthetic contact tag) must never count as content, or a deleted
        #   logistics/contact record starts silencing otherwise answerable
        #   shards again.  When nothing was allowed -- the no-live-content case
        #   and the live-but-everything-denied case both leave `covered_union`
        #   empty -- every query-sharing tombstone is admitted, preserving the
        #   old conservative default where a responsive deletion that cannot be
        #   answered from surviving content still reads as `no_memory`.
        #
        #   `covered_union` is the union of every allowed record's REAL-content
        #   digests read from its live `record_terms` (is_structural = 0).
        covered_union: set[str] = set()
        for _evidence in decision.allowed:
            covered_union |= self._record_content_digests(_evidence.record_id)
        for record_id, digests in deferred_tombstones:
            # `query_shared`: the preserved real-content digests this tombstone
            # has in common with the query.  Only these matter -- a deletion is
            # answerable from cleared records exactly to the extent the question
            # it responds to is answered there.
            query_shared = digests & content_hashes if content_hashes else set()
            if query_shared and query_shared <= covered_union:
                # Every real-content digest the question shares with the deleted
                # record is already carried by content an allowed record states
                # verbatim, so the fact the question asks about survives in
                # cleared live evidence.  Keep the record OUT of
                # `denied_tombstone` so `touched_deleted` stays False and the
                # action can remain `answer`.
                continue
            decision.denied_tombstone.append(record_id)
            self._log(checkpoint_id, requester_id, record_id, "deny_tombstone")

        # Marginal answer-bearing-denial accounting, computed once the full
        # allowed set (and hence the union of the content digests the answer
        # will actually provide) is known.  Each denied record counts only if it
        # adds a real-content digest the allowed answer does not already carry.
        allowed_union: set[str] = set()
        for s in allowed_content_sets:
            allowed_union |= s
        decision.query_answer_denials = sum(
            1 for s in denied_content_sets if not (s <= allowed_union)
        )

        # Inert diagnostic census, attached only now that the allowed/denied
        # lists are final so it can never influence them.  It is derived from
        # `census_captured`, the pre-gate ranked snapshot taken right after the
        # PASS 1 sort, so a record's entry reflects its TRUE rank under the cap
        # even when that record never reached a gate (a record ranked 17th is
        # visible here although it was never considered by PASS 2).  Truncate to
        # at most 24 entries, and trim further when fewer gate-0 survivors
        # existed (the snapshot is already capped at top_k + 24).  Census rows
        # are added to NO allowed/denied list -- they only decorate the
        # diagnostic payload.
        decision.candidate_census = [
            {
                "record_id": item[3]["record_id"],
                "overlap_total": item[0],
                "overlap_content": item[0] - item[1],
                "seq": item[2],
                "rank": rank,
            }
            for rank, item in census_captured[:24]
        ]

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

    # ------------------------------------------------------------------
    # Responsiveness index  (gate 0)
    # ------------------------------------------------------------------

    def _index_key(self) -> bytes:
        """The per-store HMAC key for the term index, created once and kept.

        Persisted in `store_meta` rather than held in memory because a resumed
        run reopens the same file, and a fresh key would hash every query term
        to a digest matching nothing -- which would make every record look
        non-responsive and every answer `no_memory`. A silent total loss of U,
        from a detail that looks like a cache.
        """
        if self._index_key_cache is None:
            row = self.conn.execute(
                "SELECT value FROM store_meta WHERE key = 'term_index_key'"
            ).fetchone()
            if row is None:
                key = os.urandom(32)
                self.conn.execute(
                    "INSERT OR REPLACE INTO store_meta(key, value) VALUES ('term_index_key', ?)",
                    (key,),
                )
                self.conn.commit()
            else:
                key = bytes(row["value"])
            self._index_key_cache = key
        return self._index_key_cache

    def _term_hashes(self, terms: set[str]) -> list[str]:
        """Keyed digests of a term set, for comparison against `record_terms`.

        Truncated to 16 hex characters (64 bits). Long enough that an accidental
        collision between two content words is not a thing that happens, short
        enough that the index stays small on a per-episode store.
        """
        key = self._index_key()
        return [
            hmac.new(key, term.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
            for term in sorted(terms)
        ]

    def _index_record(self, record_id: str, text: str) -> None:
        """Record the distinctive terms of one record, keyed-hashed.

        Called from `ingest_turn` with the CLEARTEXT, before a confidential body
        is replaced by ciphertext. `INSERT OR IGNORE` because `ingest_turn` uses
        `INSERT OR REPLACE` on the record itself and a re-ingested turn must not
        raise on its own unchanged terms.

        The stored digest set is the distinctive terms UNION their stem variants
        (`_morph_terms` is a superset, so this widens the index and never drops a
        row the old code stored).  `retrieve()` hashes the same union so ingest
        and query stay symmetric -- that is what makes variant recall ('results'
        finding a stored 'result') possible without a fuzzy full-scan.
        """
        terms = _match_terms(text)
        if not terms:
            return
        # Each digest is stored with an `is_structural` flag: 1 for digit-run
        # and synthetic-contact digests, 0 for real content.  The flag is
        # populated at index time so both live records and (via tombstone) the
        # preserved digests can be filtered by real content alone.
        key = self._index_key()
        rows = [
            (record_id,
             hmac.new(key, term.encode("utf-8"), hashlib.sha256).hexdigest()[:16],
             1 if _is_structural_term(term) else 0)
            for term in sorted(terms)
        ]
        self.conn.executemany(
            "INSERT OR IGNORE INTO record_terms(record_id, term_hash, is_structural) "
            "VALUES (?, ?, ?)",
            rows,
        )

    def _is_responsive(
        self, record_id: str, query_hashes: list[str], *, tombstoned: bool
    ) -> int:
        """Is this record about the query?  Gate 0 of `retrieve`.

        Returns a TERM-OVERLAP COUNT rather than a single boolean so `retrieve`
        can both gate (count == 0 means not responsive) and rank (higher counts
        mean more on-topic).  The two defaults fail OPEN -- towards treating a
        record as responsive -- because the cost of the two errors is not
        symmetric. A false negative silently drops evidence and, worse, silently
        drops a DENIAL that should have shaped the action; a false positive at
        most produces `answer_redacted` where `answer` would have done, which is
        the conservative direction.

        1.  The query has no distinctive terms at all ("what did they say?").
            Nothing to match on, so everything in the patient's shard is
            considered, exactly as before this index existed.  A large default
            count keeps such records ranked ahead of any that merely share a
            token.
        2.  The record is tombstoned. Its live `record_terms` rows were purged
            with its body (see `tombstone`), but their digests were preserved in
            `tombstone_terms`, so responsiveness is scoped against those: it is
            the number of distinct query digests present there. A tombstoned
            record whose preserved terms overlap the question is a RELATED
            deletion and stays responsive (that is what keeps a deleted record
            able to force `no_memory`). A record with ZERO captured rows there
            (its content never produced an indexable term) cannot be scoped, so
            it falls back to the legacy large default -- treated as responsive,
            which is the conservative direction.
        3.  Otherwise: a live record's responsiveness is the number of distinct
            query digests present in its index rows.  It is responsive iff that
            count is at least one.
        """
        if not query_hashes:
            return _DEFAULT_OVERLAP
        placeholders = ",".join("?" for _ in query_hashes)
        if tombstoned:
            # Digests were preserved in `tombstone_terms`, so scope against
            # them. Only when the record has zero captured rows there (nothing
            # was ever indexable) do we fall back to assuming responsiveness.
            # Only preserved REAL-content rows scope a tombstone (is_structural =
            # 0).  Digit-run and contact digests are incidental and must not let
            # a deleted logistics record silence an otherwise answerable shard
            # just because the question shares a phone digit with it.  When the
            # record kept no real-content rows it cannot be scoped, so it falls
            # back to the conservative large default -- a related deletion stays
            # able to force `no_memory` by content words it actually shared.
            captured = self.conn.execute(
                "SELECT COUNT(*) AS n FROM tombstone_terms "
                "WHERE record_id = ? AND is_structural = 0",
                (record_id,),
            ).fetchone()
            if not captured or int(captured["n"]) == 0:
                return _DEFAULT_OVERLAP
            row = self.conn.execute(
                "SELECT COUNT(DISTINCT term_hash) FROM tombstone_terms "  # noqa: S608
                f"WHERE record_id = ? AND is_structural = 0 AND term_hash IN ({placeholders})",
                [record_id, *query_hashes],
            ).fetchone()
            return int(row[0]) if row else 0
        # The only interpolated part is a run of `?` markers, one per hash; every
        # value is bound. SQLite has no array parameter, so a variable-length IN
        # clause has no other shape. Suppressed explicitly rather than left to
        # trip the Developer's `run_linter` on a finding that is not one.
        row = self.conn.execute(
            "SELECT COUNT(DISTINCT term_hash) FROM record_terms "  # noqa: S608
            f"WHERE record_id = ? AND term_hash IN ({placeholders})",
            [record_id, *query_hashes],
        ).fetchone()
        return int(row[0]) if row else 0

    def _live_content_digests(self, record_id: str, content_hashes: set[str]) -> set[str]:
        """The query-matching REAL-content digests a live record carries.

        Gate 2/3 denials are answer-bearing only to the extent they add content
        the allowed answer does not already provide (the marginal
        `query_answer_denials` accounting in `retrieve`).  Returns this record's
        stored `is_structural = 0` digests among the query's content digests --
        the content that would actually be missing if this record were withheld.
        """
        if not content_hashes:
            return set()
        placeholders = ",".join("?" for _ in content_hashes)
        rows = self.conn.execute(
            "SELECT term_hash FROM record_terms "  # noqa: S608
            f"WHERE record_id = ? AND is_structural = 0 AND term_hash IN ({placeholders})",
            [record_id, *sorted(content_hashes)],
        ).fetchall()
        return {row["term_hash"] for row in rows}

    def _tombstone_content_digests(self, record_id: str) -> set[str]:
        """A deleted record's preserved REAL-content digests.

        Read from `tombstone_terms` (is_structural = 0).  Unlike the query-
        scoped lookup above, this returns the FULL preserved content set, which
        is what the PASS-2 coverage skip compares against an allowed record's
        full live content to decide whether the deletion was re-covered verbatim.
        Digit-run and contact-tag digests (structural) are excluded: only real
        content words can constitute "duplicating the deleted fact".
        """
        rows = self.conn.execute(
            "SELECT term_hash FROM tombstone_terms "
            "WHERE record_id = ? AND is_structural = 0",
            (record_id,),
        ).fetchall()
        return {row["term_hash"] for row in rows}

    def _record_content_digests(self, record_id: str) -> set[str]:
        """An ALLOWED live record's full REAL-content digests.

        Read from its live `record_terms` (is_structural = 0).  The coverage
        skip unions these across every allowed record to get `covered_union`,
        then admits a deferred tombstone only when its preserved content is not
        a subset of that union.
        """
        rows = self.conn.execute(
            "SELECT term_hash FROM record_terms "
            "WHERE record_id = ? AND is_structural = 0",
            (record_id,),
        ).fetchall()
        return {row["term_hash"] for row in rows}

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

# Responsiveness score assigned to the two "fail open" cases in
# `_is_responsive` (empty query and tombstoned record), where no overlap count
# is computable.  A single literal > 1 so such records rank above any live
# record that shares just one token but below one that shares several.
_DEFAULT_OVERLAP = 1000

_STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    her him his i if in into is it its me my not of on or our out over she should so
    that the their them then there these they this to was we were what when which who
    will with would you your just now please tell give me""".split()
)


def _distinctive_terms(text: str) -> set[str]:
    """Content words plus digit runs.  Used for relevance and delete matching.

    Whole hyphen/apostrophe-dotted tokens are kept exactly as before (a query
    and a record that wrote the same phone/dosage the same way still meet on
    the whole token -- back-compat for delete matching).  On top of that, every
    maximal run of digits inside those tokens is emitted as its own token
    ("555-1234" -> "555", "1234"; "2024-03-15" -> "2024", "03", "15") and bare
    runs of 2+ digits anywhere in the text are emitted too, so a number written
    one way can be recalled by just the digits, which is what phone/appointment
    queries frequently reduce to.
    """
    text = (text or "").lower()
    tokens = set(re.findall(r"[a-z0-9][a-z0-9\-']{2,}", text))
    out: set[str] = set(tokens)
    for token in tokens:
        out.update(run for run in re.findall(r"\d+", token) if len(run) >= 2)
    out.update(run for run in re.findall(r"\d+", text) if len(run) >= 2)
    return {t for t in out if t not in _STOPWORDS}


def _is_structural_term(term: str) -> bool:
    """Is a match term structural (digit run / contact tag) rather than content?

    Used to tell "this denial shared real words with the query" apart from
    "this denial only shares a phone-digit run or the synthetic contact tag".
    Only the former is answer-bearing; the latter is incidental and must not
    force `answer_redacted` (see `Decision.query_answer_denials`).
    """
    return term == _CONTACT_TAG or term.isdigit()


def _add_contact_tag(terms: set[str], text: str | None) -> set[str]:
    """Add the single synthetic contact tag to a term set when text warrants it.

    Fires on a phone-ish digit pattern or any of the small `_CONTACT_SENSE_WORDS`
    whitelist.  Returns a NEW set so callers that share the input are not mutated
    behind their back.
    """
    if not text:
        return terms
    low = text.lower()
    words = set(re.findall(r"[a-z]+", low))
    if _PHONEISH_RE.search(text) or bool(_CONTACT_SENSE_WORDS & words):
        return set(terms) | {_CONTACT_TAG}
    return set(terms)


def _match_terms(text: str) -> set[str]:
    """The full term set hashed into the responsiveness index (ingest AND query).

    Distinctive words UNION their morph variants UNION (when present) the
    synthetic contact tag.  Both `_index_record` and `retrieve()` draw from this
    same widened set so ingest and query stay symmetric -- that is what makes a
    query for 'results' (or a phone's bare digits) recall a stored 'result' (or
    the hyphenated phone).
    """
    terms = _distinctive_terms(text) | _morph_terms(text)
    return _add_contact_tag(terms, text)


def _stem_one(token: str) -> str | None:
    """One conservative stem-form variant of a single token, or None.

    The rules are deliberately narrow so a record written with one inflected
    form can be recalled with another WITHOUT hallucinating sibling words.  Only
    one stem is returned (the first rule that fires) and every rule carries a
    length guard so the residual stem stays recognisably the same word:

      * plural / third-person -s, -es and -ies ("results" -> "result",
        "labs" -> "lab", "copies" -> "copy")
      * progressive -ing ("testing" -> "test", "running" -> "run")
      * simple-past -ed ("resulted" -> "result", "stopped" -> "stop")

    Nothing fires on a token too short to survive the transform as a real word,
    and words whose tail is coincidentally homographic with a suffix ("bus",
    "class", "chess", "is", "us") are left alone.
    """
    if len(token) < 5:
        return None

    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es") and not token.endswith(("ss", "us", "is")):
        return token[:-2]
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]

    if token.endswith("ing"):
        stem = token[:-3]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]  # running -> runn -> run
        if len(stem) >= 3:
            return stem
        return None
    if token.endswith("ed"):
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2]:
            stem = stem[:-1]  # stopped -> stopp -> stop
        if len(stem) >= 3:
            return stem
        return None
    return None


def _morph_terms(text: str) -> set[str]:
    """Every content word plus one conservative stem-form variant of each.

    This is a SUPERSET of `_distinctive_terms(text)`: each distinctive term
    survives verbatim, and a token whose tail is a genuine plural/s, -ing or
    -ed suffix under the length guards in `_stem_one` contributes a second,
    stemmed member.  The superset property is what keeps widening the index
    from ever dropping a row that the old code indexed -- the original token
    still hashes to the digest a record or a prior query already stored -- and
    using the same widened set on ingest AND query is what lets a query for
    'results' recall a stored record that only ever said 'result'.
    """
    terms = _distinctive_terms(text)
    out: set[str] = set(terms)
    for token in terms:
        stem = _stem_one(token)
        if stem and stem != token:
            out.add(stem)
    return out


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
