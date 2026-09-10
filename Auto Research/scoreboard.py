"""The loop's memory of its own SCORES -- history, champion, regression.

WHY THIS MODULE EXISTS
======================
Until this existed, `OrchestratorState` held exactly one number about quality:
`mgs_score`, a scalar the Judge overwrote every iteration. Nothing anywhere kept
the previous value. Four consequences followed, and run-8cf58d33b311 hit all
four at once -- MGS 0.3172 -> 0.2222 -> 0.1830 -> 0.1190 across five iterations,
with every single iteration's design predicting a rise:

1.  THE LINEAGE COULD ONLY DRIFT. `prepare_workspace` seeded iteration N from
    iteration N-1 unconditionally, so the code the loop carried forward was the
    LAST code, never the BEST code. A change that lost 0.095 MGS became the
    permanent foundation of everything after it.
2.  THE ARCHITECT COULD NOT SEE THE FALL. Its prompt showed the latest U/A/F/MGS
    and nothing else, so "my last change made this worse" was not a fact
    available to it. It re-targeted U for four consecutive iterations.
3.  THE CRITIC HAD NO WORD FOR "REVERT". Its whole vocabulary was
    `marginal_contributions`, which ranks terms by what perfecting each would be
    worth. Under MGS = U*(1-A)*(1-F) with A and F small, that ranking names U
    every time regardless of what actually changed.
4.  THE RUN SUMMARY LIED. `halt_reason_for` formatted the string "best
    MGS=%.4f" using the CURRENT score. The run reported `best MGS=0.1190` when
    its best was 0.3172.

Everything here is a pure function of a JSON-safe history list. Nothing does
I/O, nothing calls a model, and `state` is only ever read -- so the whole
self-correction mechanism is testable by constructing dicts, which is what
`tests/test_scoreboard.py` does.

THE COMPARABLE SERIES
---------------------
A run can score two stages: `dev` (the seeded 50-checkpoint slice, run every
iteration) and `full` (all 579, run only when the dev gate opens). Only `dev`
is produced by every iteration and only `dev` is the same population every time,
so **`dev` is the series the champion is chosen from**. Full-stage rows are
recorded for the record and reported, but they never decide lineage: comparing a
50-checkpoint score against a 579-checkpoint score and calling the larger one
"better" is a measurement error, not a decision.

WHY THE CHAMPION UPDATES ON STRICTLY-GREATER
--------------------------------------------
A tie leaves the OLDER iteration as champion, so an iteration that scores
exactly what its parent scored is not adopted. That is deliberate. The failure
this module exists to stop is unmonitored drift, and a tie is drift with a
better cover story: the code changed, the measurement did not, and adopting it
buys nothing while carrying whatever it broke that the 50-checkpoint slice
cannot see. The Architect is TOLD when this happens (see `trend_table`), so a
tie is visible rather than silent, and `ROLLBACK_TOLERANCE` exists for anyone
who wants to require a margin instead of a hair.
"""

from __future__ import annotations

from typing import Any

import config

#: Stage whose scores are comparable across iterations. See the module docstring.
CHAMPION_STAGE = "dev"

#: Verdicts `verdict_for` can return, worst-first in the order a reader cares.
VERDICT_REGRESSION = "regression"
VERDICT_TIED = "tied"
VERDICT_IMPROVED = "improved"
VERDICT_FIRST = "first"


# ======================================================================
# Rows
# ======================================================================


def score_row(
    *,
    iteration: int,
    stage: str,
    phase: str,
    utility: float,
    access: float,
    forgetting: float,
    mgs: float,
    n_checkpoints: int = 0,
    workspace: str = "",
    degraded: bool = False,
) -> dict[str, Any]:
    """One judged measurement, JSON-safe, for `state["score_history"]`.

    `workspace` is recorded because it is what makes a rollback executable: the
    champion row names the directory whose code produced the champion score, so
    `prepare_workspace` does not have to reconstruct a path from an iteration
    number and hope the run layout has not changed under it.
    """
    return {
        "iteration": int(iteration),
        "stage": str(stage or CHAMPION_STAGE),
        "phase": str(phase or ""),
        "U": round(float(utility), 6),
        "A": round(float(access), 6),
        "F": round(float(forgetting), 6),
        "MGS": round(float(mgs), 6),
        "n_checkpoints": int(n_checkpoints),
        "workspace": str(workspace or ""),
        # Whether the answerer was actually answering when this was measured.
        # A degraded row is a real measurement of a DIFFERENT system -- the
        # gating layer with raw record bodies pasted in as the answer -- so it
        # is recorded and reported but never compared against a healthy one.
        # See `rows_for_stage` and the note on `best_row`.
        "degraded": bool(degraded),
    }


def rows_for_stage(
    history: Any, stage: str = CHAMPION_STAGE, *, comparable_only: bool = False
) -> list[dict[str, Any]]:
    """Every row for one stage, in iteration order.

    `comparable_only` drops rows measured while the answerer was degraded. Those
    are real numbers about a real system, but not about the SAME system: with
    the renderer unreachable the pipeline scores the gating layer with raw
    evidence pasted in as the answer, which is a different -- and generally
    higher-scoring -- thing. Ranking one against the other is the same class of
    measurement error as ranking a 50-checkpoint dev score against a
    579-checkpoint full score, which this module already refuses to do.

    Defensive about shape because `history` arrives from graph state, which a
    resumed run rehydrates from a checkpointer: a malformed entry must not take
    the run down on the next routing decision.
    """
    rows = [
        row for row in (history or [])
        if isinstance(row, dict) and str(row.get("stage") or CHAMPION_STAGE) == stage
        and not (comparable_only and row.get("degraded"))
    ]
    return sorted(rows, key=lambda row: int(row.get("iteration") or 0))


def latest_row(history: Any, stage: str = CHAMPION_STAGE) -> dict[str, Any] | None:
    """The most recently judged row for a stage."""
    rows = rows_for_stage(history, stage)
    return rows[-1] if rows else None


def best_row(history: Any, stage: str = CHAMPION_STAGE) -> dict[str, Any] | None:
    """The champion row: highest MGS, earliest iteration on a tie.

    `max` over `(MGS, -iteration)` rather than over MGS alone, so that the tie
    rule in the module docstring is a property of this function instead of an
    accident of list ordering.
    """
    rows = rows_for_stage(history, stage, comparable_only=True)
    if not rows:
        return None
    return max(rows, key=lambda row: (float(row.get("MGS") or 0.0), -int(row.get("iteration") or 0)))


# ======================================================================
# The numbers every other module asks for
# ======================================================================


def best_of(state: Any, stage: str = CHAMPION_STAGE) -> tuple[float, int]:
    """`(best_mgs, best_iteration)` -- 0.0 and 0 when nothing has been judged.

    THIS IS WHAT `halt_reason_for` MUST CALL. The bug it replaces printed
    `state["mgs_score"]` under the label "best MGS", which is the LAST score;
    on a run that regressed, the two differ by the whole size of the regression.
    """
    row = best_row(_history(state), stage)
    if row is None:
        return 0.0, 0
    return float(row.get("MGS") or 0.0), int(row.get("iteration") or 0)


def _history(state: Any) -> list[dict[str, Any]]:
    """`score_history` out of a state dict, or a bare list if given one."""
    if isinstance(state, list):
        return state
    return list((state or {}).get("score_history") or [])


#: Why iteration N's parent is not iteration N-1. Empty means it IS N-1 and
#: nothing needs explaining; the other two are DIFFERENT facts about the run and
#: are worded differently everywhere they surface, because telling an Architect
#: its workspace was "rolled back to the champion" when the truth is "the
#: previous build failed and was discarded" points the next redesign at a
#: regression that never happened.
LINEAGE_DIRECT = ""
LINEAGE_CHAMPION = "champion"
LINEAGE_SKIPPED_FAILED = "failed_build"


def lineage_parent(state: Any, *, fallback: int) -> tuple[int, str]:
    """`(parent iteration, why it is not `fallback`)` -- the whole lineage rule.

    THREE MODES, and which one is in force is a property of the run rather than
    of the code. See `docs/self_correction.md`.

      ROLLBACK_TO_BEST                      N inherits the CHAMPION, the
                                            highest-scoring iteration judged so
                                            far. Protects a high-water mark; on
                                            a plateau it also freezes the
                                            lineage, because a tie never moves
                                            the champion.

      linear + LINEAGE_SKIP_FAILED_BUILDS   N inherits the most recent iteration
                                            that actually BUILT -- 15 <- 13 when
                                            14's build failed. The lineage
                                            compounds, and never compounds onto
                                            a tree whose tests do not pass.

      linear, nothing skipped               N inherits N-1 unconditionally,
                                            failed builds included. The original
                                            behaviour, kept only to reproduce an
                                            old run.

    `fallback` is what to use when nothing eligible has been judged -- iteration
    1, or a run whose every build so far has failed. The caller passes
    `iteration - 1`.

    THE SKIP IS ABOUT THE WORKSPACE, NOT ABOUT THE KNOWLEDGE. A skipped
    iteration's failure still reaches the next Architect in full: the workspace
    and the failure report are separate channels, and `dev_failure_report` /
    `dev_failure_history` are untouched by anything here. The returned reason is
    what makes that legible -- `trend_table` renders it so the Architect is told
    that its workspace is iteration 13's *because* iteration 14 failed, rather
    than being left to infer it from a gap in the numbering.
    """
    if config.ROLLBACK_TO_BEST:
        best = best_row(_history(state))
        if best is None:
            return fallback, LINEAGE_DIRECT
        parent = int(best.get("iteration") or fallback)
        return parent, (LINEAGE_CHAMPION if parent != fallback else LINEAGE_DIRECT)

    if not config.LINEAGE_SKIP_FAILED_BUILDS:
        return fallback, LINEAGE_DIRECT

    # Linear, skipping failed builds: the most recent iteration at or before
    # `fallback` that left a score row. Degraded rows COUNT -- the answerer
    # being down says nothing about whether the code compiles, and that
    # workspace built and was evaluated like any other. This is the one place
    # `comparable_only` would be the wrong question to ask.
    built = [
        int(row.get("iteration") or 0)
        for row in rows_for_stage(_history(state))
        if int(row.get("iteration") or 0) <= int(fallback)
    ]
    if not built:
        return fallback, LINEAGE_DIRECT
    parent = max(built)
    return parent, (LINEAGE_SKIPPED_FAILED if parent != fallback else LINEAGE_DIRECT)


def champion_iteration(state: Any, *, fallback: int) -> int:
    """The iteration whose workspace the next Developer episode should inherit.

    The parent half of `lineage_parent`, kept as its own name because that is
    what every caller and every test asks for. See `lineage_parent` for the rule.
    """
    return lineage_parent(state, fallback=fallback)[0]


def champion_workspace(state: Any) -> str:
    """The champion row's recorded workspace path, or "" if there is none."""
    best = best_row(_history(state))
    return str((best or {}).get("workspace") or "")


#: Verdict for a row measured while the answerer was down. Not a comparison at
#: all: there is nothing honest to compare it with.
VERDICT_DEGRADED = "degraded"


def verdict_for(history: Any, iteration: int, stage: str = CHAMPION_STAGE) -> str:
    """How one iteration's score compares with everything judged BEFORE it.

    Deliberately not "compared with the previous iteration": on a run that has
    already rolled back, iteration N's parent is the champion rather than N-1,
    so the champion is the honest comparison and the one a revert decision is
    actually made against.
    """
    rows = rows_for_stage(history, stage)
    mine = next((row for row in rows if int(row.get("iteration") or 0) == int(iteration)), None)
    if mine is None:
        return ""
    if mine.get("degraded"):
        return VERDICT_DEGRADED
    earlier = [
        row for row in rows
        if int(row.get("iteration") or 0) < int(iteration) and not row.get("degraded")
    ]
    if not earlier:
        return VERDICT_FIRST
    best_before = max(float(row.get("MGS") or 0.0) for row in earlier)
    mgs = float(mine.get("MGS") or 0.0)
    if mgs > best_before + config.ROLLBACK_TOLERANCE:
        return VERDICT_IMPROVED
    if mgs < best_before - config.ROLLBACK_TOLERANCE:
        return VERDICT_REGRESSION
    return VERDICT_TIED


def regression_streak(history: Any, stage: str = CHAMPION_STAGE) -> int:
    """How many of the most recent judged iterations failed to beat the best.

    Counts back from the end over rows whose verdict is `regression` or `tied`.
    A streak of 1 is an ordinary bad iteration; a streak of 3 is the loop
    walking away from its own best answer and is worth shouting about, which is
    what `trend_table` does with it.
    """
    rows = rows_for_stage(history, stage)
    streak = 0
    for row in reversed(rows):
        if verdict_for(rows, int(row.get("iteration") or 0), stage) in {
            VERDICT_REGRESSION, VERDICT_TIED
        }:
            streak += 1
        else:
            break
    return streak


# ======================================================================
# Term-level decomposition -- what the Critic needs and did not have
# ======================================================================


def term_deltas(
    baseline: dict[str, Any] | None, current: dict[str, Any] | None
) -> dict[str, Any]:
    """Which of U, A and F actually MOVED between two rows, and by how much.

    THE POINT OF THIS FUNCTION. `critic.marginal_contributions` answers "which
    term is worth the most if perfected", and under a product metric with two
    small terms the answer is structurally always U. That is a fine question for
    a system that has never been measured twice. It is the wrong question
    entirely once there IS a previous measurement, because it cannot distinguish
    "U was always the weak term" from "U just fell 0.22 because of the change we
    made last iteration" -- and only the second calls for a revert.

    Signs are normalised so that `delta_mgs_from` is always "how much MGS this
    term's movement cost or bought", which is the sentence a critique wants to
    write. A term that did not move contributes exactly 0.0.
    """
    if not baseline or not current:
        return {}

    def value(row: dict[str, Any], key: str) -> float:
        return float(row.get(key) or 0.0)

    base = {key: value(baseline, key) for key in ("U", "A", "F")}
    now = {key: value(current, key) for key in ("U", "A", "F")}

    def mgs(u: float, a: float, f: float) -> float:
        return u * (1.0 - a) * (1.0 - f)

    baseline_mgs = mgs(base["U"], base["A"], base["F"])
    # Counterfactual per term: move ONLY that term to its new value and see what
    # MGS would have been. The three do not sum exactly to the total change
    # (a product has cross terms) and that is stated rather than hidden -- the
    # residual is reported as `interaction`.
    contributions = {
        "U": mgs(now["U"], base["A"], base["F"]) - baseline_mgs,
        "A": mgs(base["U"], now["A"], base["F"]) - baseline_mgs,
        "F": mgs(base["U"], base["A"], now["F"]) - baseline_mgs,
    }
    total = mgs(now["U"], now["A"], now["F"]) - baseline_mgs
    moved = {key: round(now[key] - base[key], 6) for key in ("U", "A", "F")}
    ranked = sorted(contributions.items(), key=lambda kv: kv[1])
    return {
        "baseline_iteration": int(baseline.get("iteration") or 0),
        "current_iteration": int(current.get("iteration") or 0),
        "baseline_mgs": round(baseline_mgs, 6),
        "current_mgs": round(mgs(now["U"], now["A"], now["F"]), 6),
        "delta_mgs": round(total, 6),
        "moved": moved,
        "delta_mgs_from": {key: round(value, 6) for key, value in contributions.items()},
        "interaction": round(total - sum(contributions.values()), 6),
        # Worst first: the term that COST the most MGS is the one a revert
        # should target, and it is the first thing the Critic should say.
        "ranked_worst_first": [key for key, _ in ranked],
        "worst_term": ranked[0][0] if ranked else "",
        "worst_delta": round(ranked[0][1], 6) if ranked else 0.0,
    }


def regression_report(state: Any, stage: str = CHAMPION_STAGE) -> dict[str, Any]:
    """The whole regression story for the iteration just judged, or `{}`.

    Returned empty when there is nothing to say -- the first judged iteration,
    or an iteration that improved. Callers can therefore treat truthiness as
    "there is a regression to explain", which is how both the Critic and the
    Architect use it.
    """
    history = _history(state)
    current = latest_row(history, stage)
    if current is None:
        return {}
    iteration = int(current.get("iteration") or 0)
    verdict = verdict_for(history, iteration, stage)
    if verdict in {"", VERDICT_FIRST, VERDICT_IMPROVED}:
        return {}
    earlier = [
        row for row in rows_for_stage(history, stage)
        if int(row.get("iteration") or 0) < iteration
    ]
    baseline = max(earlier, key=lambda row: float(row.get("MGS") or 0.0)) if earlier else None
    if baseline is None:
        return {}
    return {
        "verdict": verdict,
        "iteration": iteration,
        "champion_iteration": int(baseline.get("iteration") or 0),
        "champion_mgs": round(float(baseline.get("MGS") or 0.0), 6),
        "current_mgs": round(float(current.get("MGS") or 0.0), 6),
        "delta": round(float(current.get("MGS") or 0.0) - float(baseline.get("MGS") or 0.0), 6),
        "streak": regression_streak(history, stage),
        "terms": term_deltas(baseline, current),
    }


# ======================================================================
# Rendering -- the block the Architect actually reads
# ======================================================================


def _verdict_label(verdict: str) -> str:
    return {
        VERDICT_FIRST: "first measurement",
        VERDICT_IMPROVED: "new best",
        VERDICT_TIED: "TIED -- not adopted",
        VERDICT_REGRESSION: "REGRESSION",
        VERDICT_DEGRADED: "NOT COMPARABLE -- answerer was down",
    }.get(verdict, verdict)


def trend_table(
    state: Any,
    *,
    stage: str = CHAMPION_STAGE,
    failed_iterations: Any = None,
    rolled_back_to: int = 0,
    lineage_reason: str = LINEAGE_CHAMPION,
) -> str:
    """The Architect's view of the whole run so far, as plain text.

    THIS BLOCK REPLACES FOUR SCALARS. What the Architect used to be shown was
    the latest U, A, F and MGS under the heading "MEASURED PERFORMANCE SO FAR",
    which is true of one iteration and says nothing about the run. It could not
    tell an improving loop from a collapsing one, and in run-8cf58d33b311 it
    designed four consecutive iterations without ever being told that each of
    its previous three had scored worse than the one before.

    Iterations with no row are printed too, with the reason, because a gap in a
    numbered history reads as a lost record rather than as a build that failed.
    """
    rows = rows_for_stage(_history(state), stage)
    failed = {
        int(entry.get("iteration") or 0): str(entry.get("missing_gates") or "")
        for entry in (failed_iterations or [])
        if isinstance(entry, dict)
    }
    if not rows and not failed:
        return "(nothing has been judged yet -- this is the first iteration)\n"

    lines = [
        "  iter        U        A        F      MGS     delta   verdict",
    ]
    known = sorted({int(row.get("iteration") or 0) for row in rows} | set(failed))
    by_iteration = {int(row.get("iteration") or 0): row for row in rows}
    previous: float | None = None
    for iteration in known:
        row = by_iteration.get(iteration)
        if row is None:
            lines.append(
                f"  {iteration:<4d}     -- build failed, never evaluated "
                f"(unmet gates: {failed.get(iteration) or 'unknown'}) --"
            )
            continue
        mgs = float(row.get("MGS") or 0.0)
        delta = "      --" if previous is None else f"{mgs - previous:+8.4f}"
        lines.append(
            f"  {iteration:<4d}  {float(row.get('U') or 0.0):7.4f}  "
            f"{float(row.get('A') or 0.0):7.4f}  {float(row.get('F') or 0.0):7.4f}  "
            f"{mgs:7.4f}  {delta}   {_verdict_label(verdict_for(rows, iteration, stage))}"
        )
        previous = mgs

    best = best_row(rows, stage)
    current = latest_row(rows, stage)
    if best and current:
        lines.append("")
        lines.append(
            f"BEST SO FAR: iteration {int(best.get('iteration') or 0)} "
            f"(MGS={float(best.get('MGS') or 0.0):.4f}).  "
            f"MOST RECENT: iteration {int(current.get('iteration') or 0)} "
            f"(MGS={float(current.get('MGS') or 0.0):.4f}).  "
            f"TARGET: {config.MGS_TARGET:.2f}."
        )

    report = regression_report(rows, stage)
    if report:
        lines.append("")
        lines.extend(_regression_prose(report, rolled_back_to, lineage_reason))
    elif rolled_back_to:
        lines.append("")
        lines.append(_parent_note(rolled_back_to, lineage_reason))
    return "\n".join(lines) + "\n"


def _parent_note(parent: int, reason: str) -> str:
    """One sentence saying whose code the Architect is looking at, and why.

    The `why` is load-bearing and the two reasons are not interchangeable. A
    champion rollback means "the last iteration scored worse and its changes
    were discarded"; a skipped failed build means "the last iteration never
    produced a score at all, because it could not build". An Architect told the
    first when the truth is the second will write a work order undoing a
    regression that did not happen, and will not address the build failure --
    which is the one thing it has actually been asked to fix.
    """
    if reason == LINEAGE_SKIPPED_FAILED:
        return (
            f"Your workspace is iteration {parent}'s code -- the most recent build that "
            f"PASSED ITS GATES. The iteration(s) since then failed to build and their "
            f"edits were discarded, so nothing they wrote is in the code you are looking "
            f"at. Read the build-failure section below before you design: your job is to "
            f"get past that failure, not to re-propose the work order that hit it."
        )
    return f"Your workspace is iteration {parent}'s code (the current champion)."


def _regression_prose(
    report: dict[str, Any], rolled_back_to: int, lineage_reason: str = LINEAGE_CHAMPION
) -> list[str]:
    """The two or three sentences that tell the Architect what to DO about it."""
    terms = report.get("terms") or {}
    moved = terms.get("moved") or {}
    cost = terms.get("delta_mgs_from") or {}
    worst = str(terms.get("worst_term") or "")
    streak = int(report.get("streak") or 0)

    headline = (
        f"!! THE LAST ITERATION DID NOT BEAT THE BEST. Iteration "
        f"{report.get('iteration')} scored {float(report.get('current_mgs') or 0.0):.4f} "
        f"against iteration {report.get('champion_iteration')}'s "
        f"{float(report.get('champion_mgs') or 0.0):.4f} "
        f"({float(report.get('delta') or 0.0):+.4f})."
    )
    lines = [headline]
    if streak >= 2:
        lines.append(
            f"!! THIS HAS NOW HAPPENED {streak} ITERATIONS IN A ROW. The direction "
            "you have been pushing is not working. Do not push it further -- change "
            "the mechanism or narrow the change until you can see which part helps."
        )
    if worst:
        lines.append(
            f"   Which term moved: U {moved.get('U', 0.0):+.4f} "
            f"(MGS {cost.get('U', 0.0):+.4f}), A {moved.get('A', 0.0):+.4f} "
            f"(MGS {cost.get('A', 0.0):+.4f}), F {moved.get('F', 0.0):+.4f} "
            f"(MGS {cost.get('F', 0.0):+.4f}). The costliest movement was {worst}."
        )
    if rolled_back_to and lineage_reason == LINEAGE_SKIPPED_FAILED:
        lines.append("   " + _parent_note(rolled_back_to, lineage_reason))
    elif rolled_back_to:
        lines.append(
            f"   THE LOSING CHANGES ARE ALREADY GONE. Your workspace below has been "
            f"rolled back to iteration {rolled_back_to}'s code -- the champion. Do "
            f"NOT write a work order that reverts iteration {report.get('iteration')}'s "
            f"changes; they are not in the code you are looking at. Design a "
            f"DIFFERENT approach to the same problem, and say in "
            f"`expected_tradeoff` why this one will not fail the way that one did."
        )
    else:
        lines.append(
            "   Reverting the change that caused this is a legitimate work order "
            "and is usually the right one. Name the specific edit to undo."
        )
    return lines


def summary(state: Any, stage: str = CHAMPION_STAGE) -> dict[str, Any]:
    """The scoreboard block for `summary_run-*.json` and for the final log line."""
    history = _history(state)
    best_mgs, best_iteration = best_of(history, stage)
    current = latest_row(history, stage)
    return {
        "best_mgs": round(best_mgs, 6),
        "best_iteration": best_iteration,
        "final_mgs": round(float((current or {}).get("MGS") or 0.0), 6),
        "final_iteration": int((current or {}).get("iteration") or 0),
        "regressed_from_best": bool(
            current is not None and float(current.get("MGS") or 0.0) < best_mgs
        ),
        "regression_streak": regression_streak(history, stage),
        "history": rows_for_stage(history, stage),
    }
