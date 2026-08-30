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

from .store import Decision, MemoryStore

# Mirrors gatemem_adapter.HIDDEN_ANNOTATION_FIELDS.  Duplicated on purpose: this
# package is the shippable artifact and must be safe standing alone, without the
# orchestrator around it.
HIDDEN_FIELDS = frozenset(
    {"query_type", "attack_type", "expected_action", "judge_spec", "leak_targets",
     "gold_answer_structured", "gold_refusal_category", "policy_snapshot"}
)

REFUSAL_TEXT = "I'm not able to share that information with you."
NO_MEMORY_TEXT = "I don't have any information on that."


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
    """
    if decision.touched_deleted:
        return "no_memory", f"{len(decision.denied_tombstone)} responsive record(s) tombstoned"
    if decision.allowed and not decision.touched_unauthorized:
        return "answer", f"{len(decision.allowed)} record(s) cleared for {requester_role}"
    if decision.allowed and decision.touched_unauthorized:
        return "answer_redacted", "partially authorized: withheld higher-sensitivity records"
    if decision.touched_unauthorized:
        return "refuse", f"{len(decision.denied_rbac) + len(decision.denied_scope)} record(s) denied"
    return "no_memory", "no responsive records"


class GateMemAgent:
    """SQL-backed, RBAC-filtered, tombstone-aware memory agent."""

    def __init__(
        self,
        db_path: str = ":memory:",
        llm: Callable[[str, list[dict[str, Any]]], dict[str, Any]] | None = None,
        top_k: int = 20,
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

        if action in {"answer", "answer_redacted"} and self.llm is not None:
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
        elif action in {"answer", "answer_redacted"}:
            answer = " ".join(e.body for e in decision.allowed)
        elif action == "refuse":
            answer = REFUSAL_TEXT
        else:
            answer = NO_MEMORY_TEXT

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
            },
        }
