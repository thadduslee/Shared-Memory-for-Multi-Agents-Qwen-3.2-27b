"""Every conditional edge in the graph, as a named function.

Each router answers exactly one question and says WHY that question is worth a
branch.  They are pure functions of state -- no I/O, no model calls -- so
`tests/test_routing.py` can drive each path by constructing a state dict.

Router precedence, highest first, is the same everywhere:

    1. budget guard      -- never spend past the cap, whatever else is true
    2. hard halt         -- circuit breaker / developer exhaustion / node failure
    3. curriculum        -- a failed phase halts the remaining phases
    4. the router's own question
"""

from __future__ import annotations

import logging
from typing import Any

import config
from nodes._common import budget_snapshot

log = logging.getLogger("orchestrator.router")


# ======================================================================
# Budget guard
# ======================================================================


def budget_exhausted(state: dict[str, Any]) -> str | None:
    """Return a halt reason if the run has spent its wall clock or tokens.

    Checked at the top of every router rather than in one place, because the
    graph has several exits and a guard that only fires at one of them is a
    guard that does not fire.
    """
    snapshot = budget_snapshot(state)
    if snapshot["over_wallclock"]:
        return (
            f"wall-clock budget exhausted: {snapshot['elapsed_s']:.0f}s > "
            f"{config.MAX_WALLCLOCK_S:.0f}s"
        )
    if snapshot["over_tokens"]:
        return (
            f"token budget exhausted: {snapshot['total_tokens']:,} > "
            f"{config.MAX_TOTAL_TOKENS:,}"
        )
    return None


# ======================================================================
# 1. Developer -> ?
# ======================================================================


def route_after_developer(state: dict[str, Any]) -> str:
    """Did the Developer produce a build worth evaluating?

    WHY THIS EDGE EXISTS: evaluating a build that does not compile burns the
    entire fan-out to learn something `compile_check` already told us for free.
    A Developer that exhausted its retries is evidence about the *design*, so
    the feedback goes back to the Architect rather than forward to the
    Evaluator.
    """
    if budget_exhausted(state):
        return "halt"
    if state.get("halt_reason"):
        log.warning("developer -> architect: %s", state["halt_reason"])
        # Even a failed build must respect MAX_ITERATIONS, or a design the
        # Developer can never build spins the loop forever.
        if int(state.get("iteration_count", 0)) >= config.MAX_ITERATIONS:
            return "halt"
        return "architect"
    return "evaluate"


# ======================================================================
# 2. Fan-in -> ?
# ======================================================================


def route_after_collect(state: dict[str, Any]) -> str:
    """Is this batch worth scoring?

    WHY THIS EDGE EXISTS: two distinct fail-fast conditions land here.

    (a) The circuit breaker tripped -- `FAILFAST_SIGNATURE_K` shards died with
        the same normalized signature.  Scoring the survivors would produce a
        number that describes a broken build's *lucky* shards, and the Critic
        would then attribute a real metric loss to a design that was never
        actually tested.

    (b) Nothing came back at all.  An empty predictions file scores 0.0 on
        every term, which is indistinguishable from a system that answered
        everything wrong -- a diagnosis the Critic must not be handed.
    """
    if budget_exhausted(state):
        return "halt"

    if state.get("failure_signature") and state.get("halt_reason"):
        log.error("fail-fast: %s", state["halt_reason"])
        if int(state.get("iteration_count", 0)) >= config.MAX_ITERATIONS:
            return "halt"
        return "architect"

    if int(state.get("n_checkpoints_evaluated", 0)) == 0:
        log.error("no predictions produced; skipping the judge")
        if int(state.get("iteration_count", 0)) >= config.MAX_ITERATIONS:
            return "halt"
        return "architect"

    return "judge"


# ======================================================================
# 3. Judge -> ?  (the scale-up gate and the curriculum)
# ======================================================================


def route_after_judge(state: dict[str, Any]) -> str:
    """Scale up to the full run, advance the curriculum, or go diagnose.

    WHY THIS EDGE EXISTS: the 579-checkpoint run is ~11x the cost of the
    50-checkpoint dev slice.  Paying it for a build that cannot clear 0.80 on
    the cheap slice buys a more precise measurement of a number we already know
    is too low.  `DEV_GATE_MGS` is the price of admission.

    CURRICULUM: a phase that fails halts the REMAINING phases immediately
    (brief 7.1).  Running `adversarial_injection` against a system that cannot
    yet pass `scoped_access_control` produces failures whose cause is already
    known, and the Critic would then attribute the loss to the hardest phase
    rather than to the first one that actually broke.
    """
    if budget_exhausted(state):
        return "halt"

    stage = str(state.get("eval_stage") or "dev")
    report = state.get("judge_report") or {}
    phase_score = float(report.get("phase_score", 1.0))

    if phase_score < config.CURRICULUM_PASS_THRESHOLD:
        log.warning(
            "curriculum phase %s FAILED (%.3f < %.2f): halting remaining phases",
            state.get("current_curriculum_phase"), phase_score, config.CURRICULUM_PASS_THRESHOLD,
        )
        return "critic"

    if stage == "dev":
        if config.SKIP_FULL_STAGE:
            # An operator-level cost guard, checked before the gate so that it
            # cannot be defeated by a good dev score. The gate decision is still
            # computed and still recorded in judge_report.json -- you can see
            # what WOULD have happened without paying for it.
            log.info("SKIP_FULL_STAGE is set: not scaling up (dev MGS=%.4f, gate would have %s)",
                     float(state.get("mgs_score", 0.0)),
                     "opened" if state.get("proceed_to_full") else "stayed shut")
            return "critic"
        if state.get("proceed_to_full"):
            log.info("dev gate PASSED (MGS=%.4f >= %.2f): scaling up to the full run",
                     float(state.get("mgs_score", 0.0)), config.DEV_GATE_MGS)
            return "scale_up"
        log.info("dev gate FAILED (MGS=%.4f < %.2f): skipping the full run",
                 float(state.get("mgs_score", 0.0)), config.DEV_GATE_MGS)
        return "critic"

    return "critic"


# ======================================================================
# 4. Critic -> ?  (the macro loop)
# ======================================================================


def route_after_critic(state: dict[str, Any]) -> str:
    """Iterate, or stop.

    WHY THIS EDGE EXISTS: it is the research loop's only termination condition.
    `MGS_TARGET` (0.85) is the STOP condition and is deliberately distinct from
    `DEV_GATE_MGS` (0.80), the SCALE-UP gate -- a system can be worth the full
    579-checkpoint run well before it is actually finished.  Terminating on the
    gate instead of the target would stop the loop the moment it earned the
    right to measure itself properly.
    """
    reason = budget_exhausted(state)
    if reason:
        log.warning("halting: %s", reason)
        return "halt"

    mgs = float(state.get("mgs_score", 0.0))
    iteration = int(state.get("iteration_count", 0))

    if mgs >= config.MGS_TARGET:
        log.info("TARGET REACHED: MGS=%.4f >= %.2f after %d iteration(s)",
                 mgs, config.MGS_TARGET, iteration)
        return "halt"
    if iteration >= config.MAX_ITERATIONS:
        log.info("iteration budget reached (%d); best MGS=%.4f", config.MAX_ITERATIONS, mgs)
        return "halt"

    log.info("MGS=%.4f < %.2f: iterating (%d/%d)", mgs, config.MGS_TARGET, iteration, config.MAX_ITERATIONS)
    return "architect"


# ======================================================================
# 5. Architect -> ?
# ======================================================================


def route_after_architect(state: dict[str, Any]) -> str:
    """The Architect only fails fatally; there is nothing to build without it."""
    if state.get("halt_reason") or budget_exhausted(state):
        return "halt"
    return "developer"


# ======================================================================
# Curriculum bookkeeping (called by the terminal node, not an edge)
# ======================================================================


def advance_curriculum(state: dict[str, Any]) -> tuple[str, bool]:
    """Next phase, and whether the curriculum is complete.

    Advances only on pass, which is what makes the ordering a curriculum rather
    than a fixed schedule.
    """
    current = str(state.get("current_curriculum_phase") or config.CURRICULUM_PHASES[0])
    report = state.get("judge_report") or {}
    passed = float(report.get("phase_score", 0.0)) >= config.CURRICULUM_PASS_THRESHOLD

    try:
        index = config.CURRICULUM_PHASES.index(current)
    except ValueError:
        return config.CURRICULUM_PHASES[0], False

    if not passed:
        return current, False  # repeat the failed phase next iteration
    if index + 1 >= len(config.CURRICULUM_PHASES):
        return current, True
    return config.CURRICULUM_PHASES[index + 1], False


def halt_reason_for(state: dict[str, Any]) -> str:
    """The human-readable reason a run stopped, for `halt_reason` at END."""
    if state.get("halt_reason"):
        return str(state["halt_reason"])
    budget = budget_exhausted(state)
    if budget:
        return budget
    mgs = float(state.get("mgs_score", 0.0))
    if mgs >= config.MGS_TARGET:
        return f"target reached: MGS={mgs:.4f} >= {config.MGS_TARGET}"
    if int(state.get("iteration_count", 0)) >= config.MAX_ITERATIONS:
        return f"iteration budget exhausted after {config.MAX_ITERATIONS}; best MGS={mgs:.4f}"
    return "completed"
