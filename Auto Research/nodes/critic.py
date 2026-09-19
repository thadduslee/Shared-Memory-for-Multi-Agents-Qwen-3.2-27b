"""Critic node -- quantitative failure attribution, not vibes.

Input: the Judge's report (U, A, F, MGS, per-checkpoint failures) AND the
current design document and SQL schema.
Output: `critique` (Markdown, written to `runs/iter_n/critique.md`) and
`attribution` (the ranked marginal contributions).

THE ARTIFACT IS EXACTLY THE STATE FIELD.  `runs/iter_n/critique.md` is this
critique and nothing else, so the Architect that opens that file at the start of
the next iteration reads the same bytes that `state["critique"]` carries.

THE HISTORY OF EARLIER ITERATIONS IS NOT THIS NODE'S TO KEEP.  It lives in
`runs/critique_summary.md`, which the Architect writes -- it is the only node
that reads a critique and writes the design answering it in the same turn.  A
recap appended here would also mean the next Architect summarising a file that
already contains the earlier summaries, and a summary of a summary is where a
feedback loop quietly stops carrying information.  See nodes/_recap.py.

THE ATTRIBUTION IS COMPUTED IN PYTHON, NOT ASKED OF THE MODEL.
`MGS = U * (1 - A) * (1 - F)` is arithmetic, and a model asked to rank three
terms by their effect on a product will sometimes rank them by which one *looks*
worst.  So the ranking is computed here and handed to the model as a fact it is
told not to re-derive.  The model's job starts after the ranking: tracing the
dominant term to a component, citing checkpoint ids, and proposing DDL.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import config
import scoreboard
from gatemem_adapter import memory_governance_score
from harness.dsh_client import looks_like_tool_call_markup, strip_tool_call_markup
from harness.profiles import CRITIC_PROFILE
from nodes._common import node_span, usage_delta, write_artifact
from nodes._transport import agent_call_json
from state import OrchestratorState

log = logging.getLogger("orchestrator.critic")


def marginal_contributions(utility: float, access: float, forgetting: float) -> dict[str, Any]:
    """What would MGS have been if each term alone were perfect?

    For a product, the marginal value of fixing one term is not proportional to
    how bad that term looks in isolation -- it depends on the other two.  With
    U=0.9, A=0.30, F=0.05, MGS is 0.599; perfecting A yields 0.855 (+0.256)
    while perfecting F yields only 0.630 (+0.031), even though both are
    "failures".  A ranked list of these deltas is the only honest way to say
    which metric is dragging MGS down.
    """
    current = memory_governance_score(utility, access, forgetting)
    perfect = {
        # U -> 1.0 (perfect utility)
        "U": memory_governance_score(1.0, access, forgetting),
        # A -> 0.0 (no access-control violations)
        "A": memory_governance_score(utility, 0.0, forgetting),
        # F -> 0.0 (no active-forgetting failures)
        "F": memory_governance_score(utility, access, 0.0),
    }
    gains = {term: round(value - current, 6) for term, value in perfect.items()}
    ranked = sorted(gains.items(), key=lambda kv: -kv[1])
    return {
        "current_mgs": round(current, 6),
        "mgs_if_perfect": {k: round(v, 6) for k, v in perfect.items()},
        "marginal": gains,
        "ranked": [term for term, _ in ranked],
        "dominant_term": ranked[0][0] if ranked else "U",
        "dominant_gain": ranked[0][1] if ranked else 0.0,
        "raw": {"U": utility, "A": access, "F": forgetting},
    }


# Which component of the design each term's failure most plausibly lives in.
# Handed to the model as a starting hypothesis it must confirm or replace with
# evidence -- not as an answer.
_COMPONENT_HYPOTHESES = {
    "U": [
        "the retrieval filter (relevance cut / top_k truncation before the as-of scan)",
        "the schema/index supporting the as-of ordering scan",
        "the answer prompt (over-cautious phrasing on authorized queries)",
    ],
    "A": [
        "the RBAC check (role_grants join and the relationship/scope predicate)",
        "the sensitivity classifier that assigns records to a tier at ingest",
        "the schema (missing composite index causing scope checks after truncation)",
    ],
    "F": [
        "the tombstone logic (gate ordering, or deletion-request matching at ingest)",
        "the tombstone representation in the schema (per-row lookup vs denormalized flag)",
        "the answer prompt (confirming existence of deleted content)",
    ],
}


# ==========================================================================
# Observed behavior -- read back off predictions.jsonl, classified in Python
# ==========================================================================
#
# WHY THIS EXISTS: the Critic used to be handed metrics, checkpoint ids and the
# Judge's prose reason, and nothing about what the pipeline actually DID.  With
# no observation to anchor it, a model asked to "name the ONE component" will
# name one -- and on a run where every utility checkpoint failed for a different
# reason it invented an implementation (an `bm25()` subquery in a scoring
# function that scores by term overlap and contains no SQL at all) and sent the
# Architect after a component that was working correctly.
#
# The fix is the same one the module docstring already argues for the ranking:
# compute the fact in Python and hand it over as a fact.  Each failing
# checkpoint carries retrieval counters and an answer string, and those two
# together identify the failing stage unambiguously -- evidence retrieved but no
# answer text is the answerer, zero candidates surviving tombstones is the
# deletion path, and neither is a "relevance filter" problem.

_MECHANISM_MEANING: dict[str, str] = {
    "crashed_or_missing": (
        "the shard raised before writing a row. The Judge scores a missing "
        "prediction as BOTH a privacy leak and a deletion leak, so these inflate "
        "A and F on their own. Implicates exception safety in the implementation "
        "-- NOT the retrieval or policy design."
    ),
    "empty_answer_despite_evidence": (
        "retrieval succeeded and evidence records were selected, but the answer "
        "string came back empty. The retrieval path is CORRECT for these. This is "
        "the answer-rendering call (evaluator model timeout, token cap, or empty "
        "content). Proposing a retrieval or schema change for these is a wrong fix."
    ),
    "answered_but_content_missing": (
        "retrieval succeeded and a non-empty answer was produced, but it omitted "
        "required content. Implicates the answer prompt or the top_k truncation -- "
        "not the candidate scan, which demonstrably returned rows."
    ),
    "wrong_action_shape": (
        "the answer contained EVERY required string -- retrieval, gating and "
        "rendering all worked -- and the Judge still failed the checkpoint "
        "because the ACTION LABEL was wrong (e.g. `answer_redacted` where the "
        "benchmark expects `answer`). Implicates one function: the Decision -> "
        "action mapping, `sanitize_and_decide` in memory_system/agent.py. A "
        "retrieval, schema or index change cannot fix these and will not move U; "
        "the branch conditions are the fix."
    ),
    "starved_by_tombstones": (
        "nothing survived to the answerer because tombstones removed the "
        "candidates. Implicates the deletion-request matching at ingest (too "
        "broad a match tombstones records the request never named)."
    ),
    "starved_by_rbac": (
        "nothing survived to the answerer because the RBAC/scope predicate denied "
        "the candidates. Implicates role_grants or the relationship check."
    ),
    "no_candidates_scanned": (
        "the candidate scan returned nothing at all -- no allows and no denials. "
        "Implicates the as-of scan predicate or ingest never storing the record."
    ),
    "evidence_withheld": (
        "records were allowed by retrieval, yet the action was refuse/no_memory. "
        "Implicates the decision logic that maps a Decision to an action."
    ),
    "other": "no counters available; classify from the Judge reason alone.",
}

# How far denials must outnumber survivors before a checkpoint counts as starved
# rather than mis-decided. 5:1 keeps a normal RBAC round (a few denials beside a
# healthy allow set) out of the starvation buckets.
_STARVATION_RATIO = 5


def _unwrap(row: dict[str, Any]) -> dict[str, Any]:
    """Prediction rows are `{checkpoint_id, output:{...}}`; tolerate flat rows."""
    inner = row.get("output")
    return inner if isinstance(inner, dict) else row


def _classify(fact: dict[str, Any]) -> str:
    """Bucket one checkpoint into the stage that actually failed."""
    if fact.get("missing"):
        return "crashed_or_missing"
    counters = fact.get("retrieval") or {}
    allowed = int(counters.get("n_allowed") or 0)
    tomb = int(counters.get("n_denied_tombstone") or 0)
    rbac = int(counters.get("n_denied_rbac") or 0) + int(counters.get("n_denied_scope") or 0)
    action = str(fact.get("action") or "")
    answered = action in {"answer", "answer_redacted"}

    if answered and not fact.get("answer_text"):
        return "empty_answer_despite_evidence"
    if answered:
        # Content present, label rejected. Distinguishing these two is the
        # difference between "the answerer omitted the appointment times" and
        # "the answerer printed every appointment time and called it
        # `answer_redacted`". They live in different functions, and bucketing
        # the second as the first sends the Developer to widen a retrieval path
        # that already returned everything the Judge asked for.
        if fact.get("action_correct") is False and not fact.get("missed_patterns"):
            return "wrong_action_shape"
        return "answered_but_content_missing"
    if allowed > 0:
        # A checkpoint that kept 1 record out of 93 did not fail because of the
        # action mapping -- it failed because the other 92 were removed. Judging
        # starvation on `allowed == 0` files that case under the decision logic
        # and sends the Developer to the wrong function, so judge it on the ratio.
        if tomb >= _STARVATION_RATIO * allowed and tomb >= rbac:
            return "starved_by_tombstones"
        if rbac >= _STARVATION_RATIO * allowed:
            return "starved_by_rbac"
        return "evidence_withheld"
    if tomb > 0 and tomb >= rbac:
        return "starved_by_tombstones"
    if rbac > 0:
        return "starved_by_rbac"
    if not counters:
        return "other"
    return "no_candidates_scanned"


# Files the mechanisms above actually live in, in the order a diagnosis needs
# them: the action mapping first, then the retrieval loop, then the schema.
_SOURCE_VIEW_FILES = (
    "memory_system/agent.py",
    "memory_system/store.py",
    "memory_system/schema.sql",
)
_SOURCE_VIEW_MAX_CHARS = 60000


def _source_view(workspace: str) -> str:
    """Inline the implementation the Critic is asked to diagnose.

    WHY INLINE AND NOT A FILE TOOL. The Critic's profile grants `fs_read`, which
    is only a real capability on the `dsh` transport; on `AGENT_TRANSPORT=http`
    -- the setting this project actually runs -- the call advertises no tools at
    all. The task used to open with "READ THE SOURCE FIRST ... Open the files",
    and a model told to open a file it has no tool for emits the call as prose
    and stops. Every critique in runs_5iter was that: tool-call markup, no
    critique, no proposals, three iterations with no feedback loop.

    Passing the content instead of the capability is the same construction the
    Architect already uses, and it makes the instruction true on both
    transports.
    """
    root = Path(workspace or "")
    if not root.is_dir():
        return "(workspace unavailable -- diagnose from the counters and the design)"
    chunks: list[str] = []
    budget = _SOURCE_VIEW_MAX_CHARS
    for name in _SOURCE_VIEW_FILES:
        path = root / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # a workspace that moved mid-run is not fatal here
            log.warning("critic: could not read %s: %s", name, exc)
            continue
        header = f"\n----- {name} ({len(text)} chars) -----\n"
        if budget - len(header) <= 0:
            chunks.append("\n[... source view truncated ...]")
            break
        body = text[: max(0, budget - len(header))]
        chunks.append(header + body)
        budget -= len(header) + len(body)
    return "".join(chunks) or "(workspace is empty)"


def _merge_verdicts(
    facts: dict[str, dict[str, Any]], report: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Fold the Judge's per-checkpoint verdict into the observed facts.

    The prediction row says what the pipeline DID; the verdict says which part
    of it the Judge rejected. `_classify` needs both to tell a content failure
    from an action-label failure.
    """
    for cid, verdict in (report.get("verdicts") or {}).items():
        fact = facts.get(str(cid))
        if isinstance(fact, dict) and isinstance(verdict, dict):
            fact["action_correct"] = bool(verdict.get("action_correct"))
            fact["missed_patterns"] = list(verdict.get("missed_patterns") or [])
    return facts


def _prediction_facts(predictions_path: str) -> dict[str, dict[str, Any]]:
    """Per-checkpoint observed behavior, plus shard errors for missing rows.

    Never raises: a Critic that dies because an artifact moved would stall the
    whole feedback loop, and a degraded critique still beats no critique.
    """
    facts: dict[str, dict[str, Any]] = {}
    path = Path(predictions_path or "")
    if not path.is_file():
        return facts
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            cid = row.get("checkpoint_id")
            if not cid:
                continue
            out = _unwrap(row)
            debug = out.get("debug") or {}
            facts[str(cid)] = {
                "action": out.get("action"),
                "answer_text": bool(str(out.get("answer") or "").strip()
                                    or out.get("answer_structured")),
                "n_used": len(out.get("used_record_ids") or []),
                "retrieval": debug.get("retrieval") or {},
            }
    except (OSError, ValueError) as exc:
        log.warning("critic: could not read predictions for observed behavior: %s", exc)
        return facts

    # Crashed checkpoints never reach predictions.jsonl; their tracebacks are the
    # single most actionable thing in the round, so pull them off the sibling
    # shard report rather than reporting a bare count.
    report_path = path.with_name("shard_report.json")
    if report_path.is_file():
        try:
            shard_report = json.loads(report_path.read_text(encoding="utf-8"))
            for shard in shard_report.get("shards") or []:
                for err in shard.get("errors") or []:
                    cid = str(err.get("checkpoint_id") or "")
                    if cid:
                        facts.setdefault(cid, {"missing": True,
                                               "error": str(err.get("error") or "")})
        except (OSError, ValueError) as exc:
            log.warning("critic: could not read shard report: %s", exc)
    return facts


def _observed_block(cited: list[dict[str, Any]], facts: dict[str, dict[str, Any]]) -> str:
    """One line per cited checkpoint: what the pipeline actually did."""
    if not facts:
        return "(predictions unavailable -- classify from the Judge reason alone)"
    lines = []
    for offender in cited:
        cid = offender["checkpoint_id"]
        fact = facts.get(cid)
        if fact is None:
            lines.append(f"- {cid}: no prediction row and no recorded shard error")
            continue
        if fact.get("missing"):
            lines.append(f"- {cid}: CRASHED before producing a row -- {fact.get('error')}")
            continue
        counters = fact.get("retrieval") or {}
        lines.append(
            f"- {cid}: action={fact.get('action')} "
            f"answer={'non-empty' if fact.get('answer_text') else 'EMPTY'} "
            f"evidence_used={fact.get('n_used')} | retrieval: "
            f"allowed={counters.get('n_allowed', '?')} "
            f"denied_tombstone={counters.get('n_denied_tombstone', '?')} "
            f"denied_rbac={counters.get('n_denied_rbac', '?')} "
            f"denied_scope={counters.get('n_denied_scope', '?')} "
            f"[mechanism: {_classify(fact)}]"
        )
    return "\n".join(lines)


def _mechanism_census(
    cited: list[dict[str, Any]], facts: dict[str, dict[str, Any]]
) -> tuple[str, list[str]]:
    """Group the cited failures by mechanism, largest group first."""
    groups: dict[str, list[str]] = defaultdict(list)
    for offender in cited:
        fact = facts.get(offender["checkpoint_id"])
        if fact is not None:
            groups[_classify(fact)].append(offender["checkpoint_id"])
    if not groups:
        return "(no observed behavior available)", []
    ordered = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    lines = []
    for name, ids in ordered:
        lines.append(f"- {name} ({len(ids)} checkpoint(s)): {_MECHANISM_MEANING[name]}")
        lines.extend(f"    - {cid}" for cid in ids)
    return "\n".join(lines), [name for name, _ in ordered]



def _regression_block(regression: dict[str, Any]) -> str:
    """The headline section, when the last iteration failed to beat the best.

    Placed ABOVE the marginal ranking rather than below it, because it is the
    only part of this prompt that is about what CHANGED. Everything else in the
    Critic's task describes the current state -- which checkpoints fail, which
    attack types concentrate failure, what the retrieval counters said -- and
    none of it can distinguish "this has always been the weak term" from "we
    broke this last iteration". Only the movement can.
    """
    if not regression:
        return ""
    terms = regression.get("terms") or {}
    moved = terms.get("moved") or {}
    cost = terms.get("delta_mgs_from") or {}
    worst = str(terms.get("worst_term") or "")
    streak = int(regression.get("streak") or 0)
    verdict = str(regression.get("verdict") or "")

    headline = (
        f"Iteration {regression.get('iteration')} scored MGS="
        f"{float(regression.get('current_mgs') or 0.0):.4f}. The best iteration so far is "
        f"{regression.get('champion_iteration')} at MGS="
        f"{float(regression.get('champion_mgs') or 0.0):.4f} "
        f"({float(regression.get('delta') or 0.0):+.4f})."
    )
    movement_header = (
        "WHICH TERM MOVED, and what that movement cost MGS "
        f"(against iteration {terms.get('baseline_iteration')}):"
    )
    lines = [
        "## !! REGRESSION -- THIS IS THE FINDING",
        headline,
        "",
        movement_header,
        f"  U: {moved.get('U', 0.0):+.4f}   -> MGS {cost.get('U', 0.0):+.4f}",
        f"  A: {moved.get('A', 0.0):+.4f}   -> MGS {cost.get('A', 0.0):+.4f}",
        f"  F: {moved.get('F', 0.0):+.4f}   -> MGS {cost.get('F', 0.0):+.4f}",
    ]
    if worst:
        lines.append(f"  The costliest movement was {worst}.")
    if verdict == scoreboard.VERDICT_TIED:
        lines.append(
            "\nThis iteration TIED rather than fell. Its code is not being adopted: "
            "the next Developer inherits the champion's workspace instead. Say what "
            "the change failed to buy, and what would have to be different."
        )
    if streak >= 2:
        lines.append(
            f"\n!! {streak} CONSECUTIVE ITERATIONS have now failed to beat iteration "
            f"{regression.get('champion_iteration')}. The direction being pushed is "
            "not working. A critique that recommends more of it is the fourth "
            "iteration of a strategy already measured as wrong."
        )
    lines.append(
        "\nTHE DIFF BELOW IS THE PRIME SUSPECT. The change made this iteration is "
        "the only thing that moved between the champion's measurement and this one, "
        "so start there: name the specific edit you believe cost the metric, and say "
        "which cited checkpoints it explains. If the evidence does not support the "
        "change being at fault, say THAT plainly and give the evidence -- a wrong "
        "revert costs an iteration exactly like a wrong redesign does.\n"
    )
    return "\n".join(lines) + "\n"


def _regression_task_step(regression: dict[str, Any]) -> str:
    """The numbered instruction that makes a revert a first-class recommendation.

    Without it, the Critic's task lists only "name the component" and "give
    prioritized changes", and a model asked for changes gives changes. `revert`
    has to be spelled out as an allowed answer or it is not one.
    """
    if not regression:
        return ""
    return (
        "0. START WITH THE REGRESSION. Say which edit from this iteration's diff you\n"
        "   believe caused it, and choose ONE of:\n"
        "     (a) REVERT -- name the exact edit to undo. This is a complete and\n"
        "         legitimate recommendation. A loop that cannot undo its own changes\n"
        "         can only drift, and 'revert X' is a better critique than a new\n"
        "         mechanism invented to compensate for X.\n"
        "     (b) KEEP AND FIX -- only if you can name what the edit got RIGHT that\n"
        "         is worth the loss, and the specific follow-up that recovers it.\n"
        "     (c) NOT THE CAUSE -- with the evidence that exonerates it.\n"
        "   Put your choice in `regression_verdict` in the JSON block.\n"
    )


def _regression_json_keys(regression: dict[str, Any]) -> str:
    if not regression:
        return ""
    return (
        ", plus `regression_verdict` (one of `revert`, `keep_and_fix`, `not_the_cause`) "
        "and `regression_cause` (the specific edit, in one sentence)"
    )


def _failure_bucket_table(report: dict[str, Any]) -> str:
    """The judge's utility-failure buckets, rendered for the Critic's prompt.

    The Critic already received `by_curriculum_phase` and never passed a word of
    it on: across two 20-iteration runs it named the tombstone branch in 14 of 16
    critiques while the Architect, which sees only this critique, never changed
    it. Handing the Critic the counts and telling it that it is the only node
    holding them is what turns a diagnosis into something the next node can act
    on. See nodes/judge.failure_buckets.
    """
    buckets = (report.get("utility_failure_buckets") or {})
    counts = dict(buckets.get("counts") or {})
    if not counts:
        return "(not available for this stage)"
    total = sum(counts.values())
    lines = [
        f"{total} utility checkpoints failed, grouped by the mechanism that lost them:"
    ]
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = 100.0 * count / total if total else 0.0
        lines.append(f"- {name}: {count} of {total} ({share:.1f}%)")
    discarded = int(buckets.get("authorized_records_discarded_by_tombstone") or 0)
    if discarded:
        lines.append(
            f"- in the suppressed_by_tombstone group, {discarded} records that were "
            "retrieved, RBAC-cleared and authorized for the requester were discarded "
            "before the answerer ever saw them"
        )
    return "\n".join(lines)


async def critic_node(state: OrchestratorState) -> dict[str, Any]:
    iteration = int(state.get("iteration_count", 1))
    phase = str(state.get("current_curriculum_phase") or "")
    report = state.get("judge_report") or {}

    async with node_span("critic", iteration, phase) as span:
        utility = float(state.get("utility_score", 0.0))
        access = float(state.get("access_violation_rate", 1.0))
        forgetting = float(state.get("forgetting_failure_rate", 1.0))
        attribution = marginal_contributions(utility, access, forgetting)
        dominant = attribution["dominant_term"]

        # THE OTHER HALF OF THE ATTRIBUTION, and until now the missing half.
        #
        # `marginal_contributions` answers "which term is worth the most if
        # perfected". Under MGS = U*(1-A)*(1-F) with A and F small, the answer to
        # that question is structurally U almost regardless of the data -- and it
        # was U in all four critiques of run-8cf58d33b311, whose real story was
        # that iteration 3's change had knocked U from 0.4444 to 0.2222 and every
        # subsequent iteration pushed the same lever harder. There was no verdict
        # in the Critic's vocabulary for "the last change made this worse; undo
        # it", so it never said so.
        #
        # `regression_report` answers the complementary question -- which term
        # actually MOVED, and what that movement cost -- against the best
        # iteration so far rather than against a hypothetical perfect one. When
        # it is non-empty it OUTRANKS the marginal ranking in the prompt below,
        # because a term that just fell is a fact about this run and a term that
        # would be valuable if perfect is a fact about the metric's algebra.
        regression = scoreboard.regression_report(state)
        attribution["regression"] = regression
        attribution["verdict_vs_best"] = str(report.get("verdict_vs_best") or "")

        # Every offender is normalized to carry a `checkpoint_id` before it is
        # formatted below. The Judge always sets one, but this report can also
        # arrive from a resumed checkpointer snapshot written by an older run,
        # and a missing key here would raise inside an f-string -- failing the
        # Critic for a cosmetic reason and stalling the whole feedback loop.
        offenders = [
            {**o, "checkpoint_id": str(o.get("checkpoint_id") or "(unknown)")}
            for o in (report.get("worst_offenders") or [])
            if isinstance(o, dict)
        ]
        # Bias the cited evidence toward the dominant term so the Architect gets
        # checkpoint ids that are actually about the thing being fixed.
        term_to_type = {"U": "utility", "A": "privacy", "F": "safety"}
        relevant = [o for o in offenders if o.get("query_type") == term_to_type.get(dominant)]
        cited = (relevant or offenders)[:12]

        by_attack = report.get("by_attack_type") or {}
        worst_attacks = sorted(
            ((k, v) for k, v in by_attack.items() if v.get("n")),
            key=lambda kv: -(kv[1]["fail"] / kv[1]["n"]),
        )[:5]

        # What the pipeline actually did, per cited checkpoint. Computed here so
        # the model is anchored to an observation instead of guessing at an
        # implementation it has not read.
        facts = _merge_verdicts(
            _prediction_facts(str(state.get("predictions_path") or "")), report
        )
        observed = _observed_block(cited, facts)
        census, mechanisms = _mechanism_census(cited, facts)
        n_mechanisms = len(mechanisms) or 1
        source_view = _source_view(str(state.get("memory_codebase") or ""))

        task = f"""{_regression_block(regression)}## PRECOMPUTED ATTRIBUTION (do not re-derive; do not argue with it)
current MGS         = {attribution['current_mgs']:.4f}
dominant_term       = {dominant}
dominant failing metric: {dominant}
marginal gains      = {attribution['marginal']}
MGS if each were perfect = {attribution['mgs_if_perfect']}
raw terms           = U={utility:.4f} A={access:.4f} F={forgetting:.4f}

Fixing {dominant} alone would move MGS by {attribution['dominant_gain']:+.4f}, more than
either other term. NOTE WHAT THIS RANKING IS AND IS NOT: it is a fact about the
algebra of MGS = U*(1-A)*(1-F), not about this run. With A and F small, "perfect
U" is worth the most almost regardless of the measurements, so this ranking will
name U on nearly every round and naming it again is not a finding. If there is a
REGRESSION section above, it is the finding, and it outranks this.

## COMPONENT HYPOTHESES FOR {dominant} (priors only -- the census below outranks these)
These are generic starting guesses written before this run existed. Where they
disagree with the OBSERVED PIPELINE BEHAVIOR section, the observation wins.
""" + "\n".join(f"- {h}" for h in _COMPONENT_HYPOTHESES.get(dominant, [])) + """

## FAILURE CONCENTRATION BY ATTACK TYPE (fail rate)
""" + "\n".join(
            f"- {name}: {stats['fail']}/{stats['n']} = {stats['fail'] / stats['n']:.2f}"
            for name, stats in worst_attacks
        ) + f"""

## UTILITY FAILURES BY MECHANISM -- THE MOST IMPORTANT TABLE HERE
{_failure_bucket_table(report)}
YOU ARE THE ONLY NODE THAT SEES THIS. The Architect reads your critique, not the
verdicts, so a mechanism you do not name is a mechanism it cannot act on. Each
group names a DIFFERENT repair, and the largest group is worth the most MGS:

- suppressed_by_tombstone: retrieval CLEARED the records and the decision layer
  then discarded all of them. The answer was already in hand -- this is a
  decision-layer repair, and no retrieval change will touch it.
- wrong_action_label: the content was correct and the action label was not, which
  scores zero because utility_correct = action_correct AND include_ok.
- withheld_other: refused or answered no_memory for some other reason.
- content_missing: answered, content genuinely absent -- the only group a
  retrieval change would help.

Your `dominant_term` and your `proposals` must speak to the LARGEST group unless
you can name a measured reason not to. Say the counts out loud in your prose:
"N of M utility failures are <group>" is the sentence the Architect needs, and it
is the sentence that has been missing.

## FAILURE CONCENTRATION BY CURRICULUM PHASE
{report.get('by_curriculum_phase', {})}

## FAILING CHECKPOINTS (cite these ids)
""" + "\n".join(
            f"- {o['checkpoint_id']} [{o.get('attack_type', 'none')}] expected="
            f"{o.get('expected_action')}: {o.get('reason', '')}"
            for o in cited
        ) + f"""

## OBSERVED PIPELINE BEHAVIOR (measured from predictions.jsonl -- do not contradict)
Retrieval counters and the answer string for every cited checkpoint. These are
recorded facts about this run. If a checkpoint shows `allowed=8 answer=EMPTY`,
retrieval worked and the answerer failed -- do not blame retrieval for it.

{observed}

## MECHANISM CENSUS (each cited checkpoint bucketed exactly once)
{census}

## SCORING HEALTH
malformed prediction lines: {report.get('n_malformed_lines', 0)}
missing predictions:        {report.get('n_missing_predictions', 0)}
worker shard failures:      {state.get('n_worker_failures', 0)}
developer local pass rate:  {state.get('dev_set_pass_rate', 0.0):.3f}

## CURRENT SQL SCHEMA
```sql
{str(state.get('sql_schema') or '(none)')[:6000]}
```

## CURRENT DESIGN
{strip_tool_call_markup(str(state.get('proposed_design') or '(none)'))[:8000]}

## CURRENT IMPLEMENTATION (the files the mechanisms above live in)
{source_view}

## YOUR TASK
YOU HAVE NO TOOLS ON THIS CALL. There is no file system to open and nothing will
execute a tool call you write out; emitting one only ends your turn with an
empty critique. The source you need is inlined immediately above -- all of it.

Quote the function and the lines you are changing, out of that listing. Never
describe an implementation that is not in it: a proposal that misdescribes the
current code sends the Developer to rewrite something that already works and
costs a whole iteration.

Write the critique:

{_regression_task_step(regression)}1. Address EVERY mechanism in the census, largest group first. The census is
   derived from recorded counters, so a mechanism listed there happened. Do not
   collapse distinct mechanisms into a single cause because a single cause reads
   better -- {n_mechanisms} distinct mechanism(s) were observed this round.
2. Name the component per mechanism, and mark each as `design` (the Architect and
   Developer can fix it) or `infrastructure` (a crash, a timeout, an empty model
   response -- the harness must fix it). Saying so plainly is more useful than
   inventing a design fault; if the dominant mechanism is infrastructure, say that
   is what is capping {dominant} and do not manufacture a design change to fill
   the gap.
3. Cite the checkpoint ids above as evidence and give prioritized changes. Where
   the failure is a lookup or deletion-visibility problem, propose the schema
   change as explicit DDL.

End with a ```json block containing dominant_term, component (the primary one),
mechanisms[] (each with name, component, kind: design|infrastructure,
checkpoint_ids[]), evidence_checkpoint_ids and proposals[] (each with component,
kind, change, expected_fixes[]){_regression_json_keys(regression)}."""

        result, block = await agent_call_json(
            CRITIC_PROFILE, task, Path(state.get("memory_codebase") or config.PROJECT_ROOT)
        )
        span["tokens"] = result.usage.get("total_tokens", 0)

        # A CRITIQUE THAT IS NOT A CRITIQUE IS A FAILED CRITIQUE, and the three
        # ways to get one are worth distinguishing only in the log line: the
        # call failed, the reply was tool-call markup, or the reply carried no
        # JSON block and therefore no proposals. All three end with the
        # Architect reading something it cannot act on -- and, in runs_5iter,
        # imitating the markup it was shown. The deterministic attribution-only
        # critique is worse than a real one and far better than any of these.
        markup = looks_like_tool_call_markup(result.text)
        if not result.ok or markup or not block:
            log.error(
                "critic produced no usable critique (%s); emitting attribution-only critique",
                result.error or ("tool-call markup" if markup else "no parseable json block"),
            )
            critique = _fallback_critique(
                attribution, cited, worst_attacks, census, regression
            )
            block = {}
        else:
            # Stripped even on the happy path: a reply can carry a usable JSON
            # block AND a stray tag, and this text is pasted verbatim into the
            # next Architect's prompt.
            critique = strip_tool_call_markup(result.text)

        attribution["component"] = block.get("component")
        attribution["proposals"] = block.get("proposals") or []
        # The model's answer to the revert question, kept whether or not it
        # answered: an absent verdict on a round that HAD a regression is itself
        # worth seeing in attribution.json, and `check_learning.py` reads it.
        if regression:
            attribution["regression_verdict"] = str(block.get("regression_verdict") or "")
            attribution["regression_cause"] = str(block.get("regression_cause") or "")
        # The measured decomposition, kept even when the model ignores it: it is
        # the only record of what the pipeline did, and `check_learning.py` needs
        # it to tell a design regression from a harness outage.
        attribution["observed_mechanisms"] = mechanisms
        attribution["mechanisms"] = block.get("mechanisms") or []
        attribution["evidence_checkpoint_ids"] = (
            block.get("evidence_checkpoint_ids") or [o["checkpoint_id"] for o in cited]
        )

        iter_dir = config.iteration_dir(iteration)
        # THE FILE IS THE CRITIQUE, VERBATIM. Nothing is appended: the running
        # history of earlier iterations is `runs/critique_summary.md`, which the
        # Architect keeps, and the next Architect turn opens this file for
        # iteration i and that notebook for iterations 1..i-1 as two separate
        # reads. See the module docstring.
        write_artifact(iter_dir / "critique.md", critique)
        write_artifact(iter_dir / "attribution.json", attribution)

        log.info(
            "critic iter=%d: dominant=%s gain=%+.4f component=%s "
            "mechanisms=%s (%d proposals)",
            iteration, dominant, attribution["dominant_gain"],
            attribution.get("component"), ",".join(mechanisms) or "(none)",
            len(attribution["proposals"]),
        )
        if regression:
            # At WARNING, and naming the verdict the model gave: a critique that
            # was handed a regression and answered with neither a revert nor an
            # exoneration is the case where the loop is about to build on top of
            # a measured loss again, and it should be visible in the log at the
            # moment it happens rather than in the post-mortem.
            log.warning(
                "critic iter=%d: REGRESSION vs iteration %d (%+.4f, streak %d); "
                "worst-moving term=%s; critic verdict=%s (%s)",
                iteration, regression.get("champion_iteration"),
                float(regression.get("delta") or 0.0), int(regression.get("streak") or 0),
                (regression.get("terms") or {}).get("worst_term") or "?",
                attribution.get("regression_verdict") or "(none given)",
                attribution.get("regression_cause") or "no cause named",
            )

        return {
            "critique": critique,
            # Stamped so the next Architect knows WHICH iteration this critique
            # is from and summarises it into its notebook exactly once. A failed
            # build leaves the Critic unrun and this value unchanged, which is
            # precisely the case that would otherwise be recorded twice.
            "critique_iteration": iteration,
            "attribution": attribution,
            "token_usage": usage_delta(result.usage),
            "node_timings": [span],
        }


def _fallback_critique(
    attribution: dict[str, Any], cited: list[dict[str, Any]], worst_attacks: list[Any],
    census: str = "",
    regression: dict[str, Any] | None = None,
) -> str:
    """Deterministic critique used when the Critic model is unreachable.

    THE REGRESSION SURVIVES A DEAD MODEL. Everything else here is a restatement
    of the marginal ranking, which -- see the note in `critic_node` -- names U on
    nearly every round whatever happened. If the run just lost MGS, that is the
    one thing the Architect must be told, and it is computed in Python from the
    score history, so an unreachable Critic is no reason to drop it.
    """
    dominant = attribution["dominant_term"]
    lines: list[str] = [
        "# Critique (attribution-only fallback -- the Critic model was unreachable)",
        "",
    ]
    if regression:
        terms = regression.get("terms") or {}
        moved = terms.get("moved") or {}
        cost = terms.get("delta_mgs_from") or {}
        headline = (
            f"Iteration {regression.get('iteration')} scored "
            f"{float(regression.get('current_mgs') or 0.0):.4f} against iteration "
            f"{regression.get('champion_iteration')}'s "
            f"{float(regression.get('champion_mgs') or 0.0):.4f} "
            f"({float(regression.get('delta') or 0.0):+.4f}), a streak of "
            f"{int(regression.get('streak') or 0)}."
        )
        movement = (
            f"Term movement: U {moved.get('U', 0.0):+.4f} (MGS {cost.get('U', 0.0):+.4f}), "
            f"A {moved.get('A', 0.0):+.4f} (MGS {cost.get('A', 0.0):+.4f}), "
            f"F {moved.get('F', 0.0):+.4f} (MGS {cost.get('F', 0.0):+.4f}). "
            f"Costliest movement: {terms.get('worst_term') or 'unknown'}."
        )
        action = (
            "Recommended action: REVERT this iteration's change unless there is "
            "specific evidence it was not the cause."
        )
        lines += ["## !! REGRESSION", headline, movement, action, ""]
    dominant_line = (
        f"Dominant failing term: **{dominant}** "
        f"(perfecting it would move MGS by {attribution['dominant_gain']:+.4f})."
    )
    lines += [
        "## Attribution",
        dominant_line,
        f"Marginal gains: {attribution['marginal']}",
        "",
        "## Hypothesised components",
    ]
    lines += [f"- {h}" for h in _COMPONENT_HYPOTHESES.get(dominant, [])]
    lines += ["", "## Failure concentration"]
    lines += [f"- {name}: {s['fail']}/{s['n']}" for name, s in worst_attacks]
    lines += ["", "## Observed mechanisms (measured, not inferred)", census or "(unavailable)"]
    lines += ["", "## Evidence"]
    lines += [f"- `{o['checkpoint_id']}`: {o.get('reason', '')}" for o in cited]
    return "\n".join(lines)
