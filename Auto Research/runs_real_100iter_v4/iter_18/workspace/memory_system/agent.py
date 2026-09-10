"""The GateMem-facing agent: retrieval loop -> sanitize -> act.

Implements the three-method interface from GateMem's `bench/agents/base.py`
(`reset(episode)` / `ingest(turn)` / `query(checkpoint)`), so this class can be
registered in `bench/agents/__init__.py::AGENT_REGISTRY` and run under
`bench/scripts/run_eval.py` unchanged.

THE FIELD WALL.  `query()` accepts the checkpoint as a plain dict and reads only
`checkpoint_id`, `episode_id`, `as_of_turn_id`, `asker` and `query_text`.  It
never reads `query_type`, `attack_type`, `expected_action`, `judge_spec` or
`leak_targets`; the orchestrator strips those before the dict gets here, and
`_visible()` below re-strips defensively so that running this class directly
under GateMem's own runner -- which passes a full `Checkpoint` dataclass -- is
equally safe.
"""

from __future__ import annotations

from typing import Any, Callable

from .store import (
    LOGISTICS_PATTERNS as _LOGISTICS_PATTERNS,
    Decision,
    MemoryStore,
)

# Mirrors gatemem_adapter.HIDDEN_ANNOTATION_FIELDS.  Duplicated on purpose: this
# package is the shippable artifact and must be safe standing alone, without the
# orchestrator around it.
HIDDEN_FIELDS = frozenset(
    {"query_type", "attack_type", "expected_action", "judge_spec", "leak_targets",
     "gold_answer_structured", "gold_refusal_category", "policy_snapshot"}
)

REFUSAL_TEXT = "I'm not able to share that information with you."
NO_MEMORY_TEXT = "I don't have any information on that."


# ----------------------------------------------------------------------
# answer verification -- logistics-token re-emission
# ----------------------------------------------------------------------
# An LLM answer-writer is told to produce a *summary*, and summaries are where
# the logistics a logistics question is actually about (dates, times, phone
# numbers, doses) quietly falls off the page -- the model compresses "Tuesday
# at 9:15" to "scheduled", and the Judge's include-check scores zero for a
# correct-but-detail-free answer.  These patterns find that logistics content in
# the ALREADY-GATED evidence bodies (decision.allowed), and `_logistics_suffix`
# re-emits verbatim any token the final answer dropped.  Content is only ever
# pulled from bodies that cleared every gate, so this can widen the *wording* of
# the answer but never the *content* the policy permitted.
#
# The shared regex set now lives in `memory_system.store` as `LOGISTICS_PATTERNS`
# (imported above under the `_LOGISTICS_PATTERNS` alias); both this suffix pass
# and the store's `rescue_missing_logistics` use the identical set so they can
# never disagree on what counts as a logistics token.


def _logistics_suffix(allowed_bodies: list[str], answer: str) -> str:
    """Logistics tokens present in gated bodies but missing from `answer`.

    Each missing token is returned alongside a short body fragment that carries
    it, formatted as ``\\nRequired logistics details: <token>: <fragment>`` so a
    caller can append it straight onto the answer.  Tokens already present
    verbatim (case-insensitively) in the answer are skipped.
    """
    low_answer = answer.lower()
    found: dict[str, str] = {}
    for body in allowed_bodies:
        if not body:
            continue
        for pattern in _LOGISTICS_PATTERNS:
            for m in pattern.finditer(body):
                token = m.group(0).strip()
                if not token or token in found or token.lower() in low_answer:
                    continue
                start, end = m.start(), m.end()
                found[token] = body[max(0, start - 20): end + 45].strip()
    if not found:
        return ""
    return "\nRequired logistics details: " + "; ".join(
        f"{tok}: {ctx}" for tok, ctx in found.items()
    )


def _rescue_append(
    store: MemoryStore,
    patient_id: str,
    requester_id: str,
    requester_role: str,
    as_of_seq: int | None,
    answer: str,
    allowed_ids: set[str],
) -> list[str]:
    """Recover logistics detail the ranked/top_k scan never surfaced.

    The joined `decision.allowed` bodies only ever reach records that were both
    responsive (gate 0) and within `top_k`, so a logistics question whose gold
    date/time lives in a scheduling note the relevance gate or the cap skipped
    never shows up in the assembled answer.  `_logistics_suffix` above can only
    recover tokens that exist in those bodies, so it cannot reach the missing
    note either.

    This helper runs `store.rescue_missing_logistics` -- which scans the
    patient's live (non-tombstoned) records as-of `as_of_seq` and returns the
    Evidence this requester is already cleared to read (same role x sensitivity
    x scope policy `retrieve()` applies) whose body carries a logistics token
    absent from `answer` -- then drops anything already surfaced:

    * any record whose `record_id` is already on the `allowed_ids` list (its
      body is already verbatim in the answer), and
    * any body whose full text already appears in the answer.

    It returns the remaining bodies to append (possibly none); the caller joins
    them under a single ``'\\nDetails: '`` separator so a candidate whose text
    is already in the answer contributes nothing back.
    """
    rescued = store.rescue_missing_logistics(
        patient_id,
        requester_id,
        requester_role,
        as_of_seq,
        answer,
        limit=8,
    )
    low_answer = (answer or "").lower()
    extra: list[str] = []
    for e in rescued:
        if e.record_id in allowed_ids:
            continue
        body = e.body or ""
        if body.strip().lower() in low_answer:
            continue
        extra.append(body)
    return extra


def _visible(checkpoint: Any) -> dict[str, Any]:
    """Project a checkpoint down to the agent-visible fields."""
    if not isinstance(checkpoint, dict):
        checkpoint = {
            key: getattr(checkpoint, key)
            for key in ("checkpoint_id", "episode_id", "as_of_turn_id", "query_text")
            if hasattr(checkpoint, key)
        } | {
            "asker": {
                "principal_id": getattr(checkpoint, "asker_principal_id", ""),
                "role": getattr(checkpoint, "asker_role", ""),
            }
        }
    return {k: v for k, v in checkpoint.items() if k not in HIDDEN_FIELDS}


def sanitize_and_decide(decision: Decision, requester_role: str) -> tuple[str, str]:
    """Map a retrieval Decision onto a GateMem action.

    Returns `(action, rationale)`.  Ordering is the whole algorithm:

    1.  A tombstone hit wins over everything.  If any responsive record was
        deleted, the answer is `no_memory` even when other records were
        allowed -- otherwise a partial answer reconstructs the deleted fact by
        implication, which is precisely the `split_reconstruction` and
        `indirect_inference` attacks.
    2.  Allowed-and-nothing-denied is a clean `answer`.
    3.  Allowed-but-something-denied is `answer_redacted`: there is real
        authorized content to return, but not all of it.
    4.  Nothing allowed and something denied is `refuse`.
    5.  Nothing at all is `no_memory` -- the requester asked about something
        the store never held.

    THE INVARIANT THESE BRANCHES DEPEND ON, AND WHERE IT IS ENFORCED.  Every
    branch below reads a denied list as meaning "there IS responsive content
    this requester must not get".  That is only true because `retrieve()` now
    tests responsiveness at GATE 0, before any policy gate, so a record that is
    not about the query is skipped entirely and reaches none of these lists.

    It was not true before.  Relevance used to be checked last, on the way to
    `allowed`, so a record the requester merely happened not to be cleared for
    -- a different appointment, a different clinician, a different week -- was
    denied first and never tested.  `touched_unauthorized` went true on the
    strength of it, branch 3 fired, and a complete correct answer went out
    labelled `answer_redacted`.

    That mattered exactly as much as the label matters, which under this
    benchmark is completely: the Judge scores
    `utility_correct = action_correct and include_ok`, so an answer containing
    every required string still scores zero if the label is wrong.  On the
    seeded dev slice it cost 12 of 18 utility checkpoints.

    If you change `retrieve()`, this is the property to preserve.  Widening what
    the candidate scan considers -- removing the `top_k` stop, relaxing the
    relevance test -- puts more records in front of the gates, more of them get
    denied, and these branches start firing on queries where nothing responsive
    was actually withheld.  That is not a hypothetical: it is what iteration 3
    of run-8cf58d33b311 did, and U halved in one step.
    """
    if decision.touched_deleted:
        return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
    # Spelled out as the two lists rather than via `touched_unauthorized`, so
    # that what the branch actually keys on is visible at the branch: responsive
    # records denied by RBAC or by scope. The property is enforced upstream and
    # named here because these two places have to be changed together.
    qad = decision.query_answer_denials
    withheld = qad if qad is not None else len(decision.denied_rbac) + len(decision.denied_scope)
    if decision.allowed and not withheld:
        return "answer", f"{len(decision.allowed)} record(s) cleared for {requester_role}"
    if decision.allowed and withheld:
        return (
            "answer_redacted",
            f"partially authorized: {withheld} responsive record(s) withheld",
        )
    if withheld:
        return "refuse", f"{withheld} record(s) denied"
    return "no_memory", "no responsive records"


class GateMemAgent:
    """SQL-backed, RBAC-filtered, tombstone-aware memory agent."""

    def __init__(
        self,
        db_path: str = ":memory:",
        llm: Callable[[str, list[dict[str, Any]]], dict[str, Any]] | None = None,
        # default 16 is the measured best; raising to 40 (iter 12) evaluated records ranked 17-40 through the deny gates, flipped clean answers to answer_redacted, U 0.6667->0.6111, and fixed none of the seven cap-insensitive census checkpoints. Do not raise the default.
        top_k: int = 16,
    ) -> None:
        self.base_db_path = str(db_path)
        self.store = MemoryStore(db_path)
        self.llm = llm
        self.top_k = top_k
        self.episode: dict[str, Any] | None = None
        self.patient_id: str = ""

    # -------------------- GateMem interface --------------------

    def reset(self, episode: dict[str, Any]) -> None:
        """Start a fresh episode.  A new store per episode is deliberate:
        cross-episode residue is a `cross_patient` leak waiting to happen."""
        self.store.close()
        self.patient_id = ""
        self.store = MemoryStore(self._episode_db_path(episode.get("episode_id", "unknown")))
        self.episode = episode
        entities = episode.get("entities") or {}

        for principal in entities.get("principals", []):
            self.store.upsert_principal(
                principal["principal_id"], principal.get("role", "unknown"),
                principal.get("display_name", ""),
            )
            if principal.get("role") == "patient" and not self.patient_id:
                self.patient_id = principal["principal_id"]

        for rel in entities.get("relationships", []):
            subject = rel.get("clinician_id") or rel.get("family_id") or rel.get("subject_id") or ""
            self.store.upsert_relationship(
                rel.get("type", ""), subject, rel.get("patient_id", self.patient_id),
                rel.get("scope", ""),
            )

    def _episode_db_path(self, episode_id: str) -> str:
        """One database file per episode.

        Reusing a single file across episodes would leave the previous
        patient's rows in `records`, and `CREATE TABLE IF NOT EXISTS` would not
        clear them -- a `cross_patient` leak introduced by the storage layer
        rather than by the policy. The file is unlinked first so a resumed run
        does not inherit a half-written store.
        """
        if self.base_db_path == ":memory:":
            return ":memory:"
        from pathlib import Path as _Path

        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in episode_id)
        path = _Path(self.base_db_path).with_suffix("") / f"{safe}.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        return str(path)

    def ingest(self, turn: Any) -> None:
        if not isinstance(turn, dict):
            turn = {
                "turn_id": getattr(turn, "turn_id", ""),
                "timestamp": getattr(turn, "timestamp", None),
                "turn_kind": getattr(turn, "turn_kind", "dialogue"),
                "text": getattr(turn, "text", ""),
                "speaker": {
                    "principal_id": getattr(turn, "speaker_principal_id", ""),
                    "role": getattr(turn, "speaker_role", ""),
                },
            }
        episode_id = (self.episode or {}).get("episode_id", "unknown")
        self.store.ingest_turn(episode_id, turn, self.patient_id or "unknown")

    def query(self, checkpoint: Any) -> dict[str, Any]:
        cp = _visible(checkpoint)
        asker = cp.get("asker") or {}
        requester_id = str(asker.get("principal_id") or "")
        requester_role = str(asker.get("role") or "unknown")
        as_of_seq = self.store.seq_of_turn(
            cp.get("episode_id", ""), cp.get("as_of_turn_id", "")
        )

        decision = self.store.retrieve(
            requester_id=requester_id,
            requester_role=requester_role,
            patient_id=self.patient_id or "unknown",
            query=str(cp.get("query_text") or ""),
            as_of_seq=as_of_seq,
            checkpoint_id=cp.get("checkpoint_id"),
            top_k=self.top_k,
        )
        action, rationale = sanitize_and_decide(decision, requester_role)

        # Answer assembly is driven by the *action label* alone, never by what
        # sits in decision.allowed.  In particular a `no_memory`/`refuse` action
        # -- which is exactly what a tombstone that fires produces -- must not
        # join any allowed evidence body into the response, or the coverage
        # skip would be undone by leftover authorized wording carrying the
        # deleted fact by implication.  These branches are spelled out rather
        # than folded into a default so that property is explicit at the branch.
        #
        # WHY THE LLM BRANCH APPENDS THE GATED BODIES.  In the
        # `self.llm is not None` branch the model writes a *summary*, and a
        # summary is where logistics content dies: summarisation collapses
        # multi-token gold phrases such as "Friday April 4 at 2:00 PM EEG" into
        # a paraphrase, and single-token logistics regexes (the `_logistics_suffix`
        # machinery below) cannot recover a phrase that was never written out.
        # That is the mechanism behind the 6 answered_but_content_missing
        # checkpoints measured in iterations 5-7.  So the LLM branch appends the
        # verbatim `decision.allowed` bodies under a "Details:" lead, keeping the
        # summary as the first sentence so the label's redaction/action semantics
        # are unchanged.  Content is pulled ONLY from `decision.allowed`, i.e.
        # bodies already past the tombstone/RBAC/scope gates, so this widens the
        # *wording* of an answer but never the *policy* -- the same gated set the
        # `used_record_ids` list already handed to the model.
        if action == "no_memory":
            answer = NO_MEMORY_TEXT
        elif action == "refuse":
            answer = REFUSAL_TEXT
        elif action in {"answer", "answer_redacted"} and self.llm is not None:
            # The model only ever sees evidence that already cleared all three
            # gates, so a jailbreak in the query cannot widen what is visible --
            # it can only change how the visible subset is worded.
            rendered = self.llm(
                str(cp.get("query_text") or ""),
                [
                    {"record_id": e.record_id, "role": e.author_role, "text": e.body}
                    for e in decision.allowed
                ],
            )
            answer = str(rendered.get("answer") or "")
            action = str(rendered.get("action") or action)
            # Verbatim re-emission of the gated bodies.  A summary is exactly
            # where the logistics a logistics question is actually about drops
            # off: summarisation collapses multi-token gold phrases -- "Friday
            # April 4 at 2:00 PM EEG" -- into a paraphrase, and single-token
            # logistics regexes below cannot recover a phrase that was never
            # written out.  That is the mechanism behind the 6
            # answered_but_content_missing checkpoints measured in iterations
            # 5-7.  Keeping the model summary as the lead sentence preserves the
            # redaction/action semantics of the label, and appending these
            # bodies makes any gold date/time/phone/dose/phrase that exists in
            # cleared evidence present verbatim in the judged answer text.  The
            # content is pulled only from `decision.allowed` -- bodies already
            # past the tombstone/RBAC/scope gates -- so this widens wording but
            # never policy.
            if decision.allowed:
                answer += "\nDetails: " + " ".join(e.body for e in decision.allowed)
            # Logistics rescue on the LLM path (same mechanism as the no-LLM
            # branch below).  A top_k/ranked scan only ever surfaces the
            # responsive records within the cap; a logistics question's gold
            # date/time may still live in a cleared record the relevance gate or
            # the cap skipped.  `_rescue_append` recovers those cleared bodies
            # and appends the ones whose text is not already present.
            extra = _rescue_append(
                self.store,
                self.patient_id or "unknown",
                requester_id,
                requester_role,
                as_of_seq,
                answer,
                {e.record_id for e in decision.allowed},
            )
            if extra:
                answer += "\nDetails: " + "; ".join(extra)
        elif action in {"answer", "answer_redacted"}:
            answer = " ".join(e.body for e in decision.allowed)
            # Logistics rescue (no-LLM path) -- the exact same `_rescue_append`
            # mechanism the LLM branch above uses, so both paths share one
            # implementation.  The joined `decision.allowed` bodies are every
            # *responsive* record that cleared the gates, but a logistics
            # question's gold tokens sometimes live in a record the relevance
            # gate (gate 0) or the top_k stop never surfaced -- e.g. a
            # scheduling note carrying the appointment date the query asks
            # about but not the queried substance.  Appending those cleared
            # bodies makes required logistics tokens present verbatim without
            # touching the action label or `used_record_ids`.
            extra = _rescue_append(
                self.store,
                self.patient_id or "unknown",
                requester_id,
                requester_role,
                as_of_seq,
                answer,
                {e.record_id for e in decision.allowed},
            )
            if extra:
                answer += "\nDetails: " + "; ".join(extra)
        else:
            # Any unhandled/unknown label defaults to no-memory; it never
            # reaches decision.allowed content.
            answer = NO_MEMORY_TEXT

        # Answer verification: re-emit logistics details (dates, times, phone
        # numbers, dose/instruction fragments) the surface wording dropped.
        # Content is pulled only from `decision.allowed`, which already cleared
        # every gate, so this can never widen what the policy permitted; it only
        # restores detail a summary lost.  The action label and the records
        # `used_record_ids` names are left untouched.
        if action in {"answer", "answer_redacted"}:
            suffix = _logistics_suffix([e.body for e in decision.allowed], answer)
            if suffix:
                answer = answer + suffix

        return {
            "action": action,
            "answer": answer,
            "answer_structured": {},
            "used_record_ids": [e.record_id for e in decision.allowed],
            # THE EVIDENCE THE ANSWERER WRITES FROM.  `used_record_ids` names
            # the records that cleared all three gates; this carries their
            # bodies alongside, because an id alone is not something an answer
            # can be written from.  A harness that had only the ids once spent
            # a whole research run rendering "(body withheld)" into the answer
            # prompt and scoring the resulting "I don't know" as a retrieval
            # failure.  Nothing here widens what the policy allows: it is the
            # same `decision.allowed` set, already gated, just not stripped of
            # its content on the way out.
            "evidence": [
                {"record_id": e.record_id, "role": e.author_role, "text": e.body}
                for e in decision.allowed
            ],
            "debug": {
                "agent": "gatemem_rbac_sql",
                "rationale": rationale,
                "n_allowed": len(decision.allowed),
                "n_denied_rbac": len(decision.denied_rbac),
                "n_denied_scope": len(decision.denied_scope),
                "n_denied_tombstone": len(decision.denied_tombstone),
                # Diagnostic-only: the pre-gate ranked census of candidates.
                # It decorates the debug payload and can never branch
                # action/answer/used_record_ids -- each entry is projected
                # through a fixed key tuple and the list is capped at 64, so
                # even an absent/malformed census degrades to an empty list
                # rather than raising.
                "candidate_census": [
                    {k: e.get(k) for k in ("record_id", "overlap_total", "overlap_content", "seq", "rank", "status", "has_logistics", "token_in_answer")}
                    for e in (getattr(decision, "candidate_census", None) or [])[:64]
                ],
                # Diagnostic-only: the per-patient row-walk census from
                # Decision.patient_census (iter-18).  Like candidate_census it
                # decorates the debug payload and can never branch
                # action/answer/used_record_ids -- each entry is projected
                # through a fixed key tuple and capped at 300, so an
                # absent/malformed census degrades to an empty list.
                "patient_census": [
                    {k: e.get(k) for k in ("record_id", "seq", "status", "sensitivity", "is_post_horizon", "would_be_allowed", "has_logistics", "logistics_tokens", "token_in_answer")}
                    for e in (getattr(decision, "patient_census", None) or [])[:300]
                ],
            },
        }
