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
import scoreboard
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

    ...UNLESS THE BUILD DIED OF INFRASTRUCTURE, which is not evidence about the
    design at all. run-8cf58d33b311 iteration 2 spent its whole episode on
    OpenRouter timeouts -- eight minutes of `http transport exceeded 150s`,
    empty replies, the 60-turn ceiling -- and never once called `run_tests` or
    `sql_exec`. The report it produced said "unmet mandatory gates: tests_ok,
    migration_ok", the Architect read that as a design critique, and the
    redesign it wrote in response ("remove the top_k stop") is the change that
    cost the run 0.095 MGS in the next iteration. Gates that were never RUN are
    not gates that FAILED, and the fix for a flapping endpoint is to run the
    same iteration again, not to change the schema.
    """
    if budget_exhausted(state):
        return "halt"
    if state.get("halt_reason"):
        # Even a failed build must respect MAX_ITERATIONS, or a design the
        # Developer can never build spins the loop forever.
        if int(state.get("iteration_count", 0)) >= config.MAX_ITERATIONS:
            log.warning("developer -> halt: %s", state["halt_reason"])
            return "halt"
        if _retryable_infrastructure_failure(state):
            report = state.get("dev_failure_report") or {}
            log.warning(
                "developer -> developer: episode %d/%d died of INFRASTRUCTURE "
                "(%s), not of the design -- re-running the SAME iteration rather "
                "than redesigning. %s",
                int(state.get("infra_retry_count", 0)) + 1, config.MAX_INFRA_RETRIES,
                report.get("classification_reason") or "unclassified",
                state["halt_reason"],
            )
            return "retry_developer"
        log.warning("developer -> architect: %s", state["halt_reason"])
        return "architect"
    return "evaluate"


def _retryable_infrastructure_failure(state: dict[str, Any]) -> bool:
    """Is this failed build worth re-running unchanged?

    Only when the Developer itself classified the episode as `infrastructure`
    (see `nodes/developer.py::classify_failure`) and the per-iteration retry
    budget is not spent. `inconclusive` deliberately does NOT qualify: an
    episode that read files for sixty turns and wrote nothing has a work-order
    problem, and re-running the same work order would reproduce it.
    """
    if config.MAX_INFRA_RETRIES <= 0:
        return False
    report = state.get("dev_failure_report") or {}
    if not isinstance(report, dict) or not report:
        return False
    if str(report.get("classification") or "") != "infrastructure":
        return False
    return int(state.get("infra_retry_count", 0)) < config.MAX_INFRA_RETRIES


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

    # (c) The answerer was not answering. Predictions exist and will score, but
    #     they were written by the retrieval layer rather than by a model, so
    #     the numbers describe a different system.
    #
    #     THIS IS A HALT, NOT A RETRY. A dead endpoint does not fix itself, and
    #     run-c993a6e93050 proved what the alternative costs: 23 iterations, 3
    #     hours and 12.5 million tokens of designing, rolling back and
    #     critiquing against numbers produced with the vLLM evaluator down the
    #     whole time. The loop cannot tell -- every mechanism downstream of here
    #     works perfectly on the fiction it is handed -- so the check belongs
    #     here, before the Judge turns it into a score.
    if state.get("render_degraded") and config.HALT_ON_DEGRADED_EVAL:
        log.error(
            "EVALUATOR DEGRADED: %d of the answering checkpoints (%.0f%%) fell back "
            "to raw evidence because the answerer did not respond. Halting rather "
            "than scoring them: these numbers would measure the gating layer, not "
            "the system. Check the evaluator endpoint and re-run.",
            int(state.get("n_render_degraded", 0)),
            float(state.get("render_degraded_rate", 0.0)) * 100,
        )
        return "halt"

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
        # The BEST, out of the score history -- not `mgs`, which is the LAST.
        # The two are the same only on a run that never regressed, and this line
        # claimed "best MGS=0.1190" on a run whose best was 0.3172.
        best_mgs, best_iteration = scoreboard.best_of(state)
        log.info("iteration budget reached (%d); best MGS=%.4f (iteration %d), final MGS=%.4f",
                 config.MAX_ITERATIONS, best_mgs, best_iteration, mgs)
        return "halt"

    log.info("MGS=%.4f < %.2f: iterating (%d/%d)", mgs, config.MGS_TARGET, iteration, config.MAX_ITERATIONS)
    return "architect"


# ======================================================================
# 5. Architect -> ?
# ======================================================================


#: Substrings that mark an Architect failure as TRANSPORT rather than as a
#: design the model could not produce. Matched against `halt_reason`, which is
#: where `architect_node` puts the provider's error verbatim.
_TRANSIENT_ARCHITECT_MARKERS = (
    "timed out", "timeout", "connecterror", "connection", "empty response",
    "in_flight", "in-flight", "http 402", "http 408", "http 429",
    "http 500", "http 502", "http 503", "http 504",
)


def _architect_failure_is_transient(state: dict[str, Any]) -> bool:
    """Did the Architect fail to answer, or fail to be reachable?

    Only the second is worth retrying, and only a bounded number of times.
    """
    reason = str(state.get("halt_reason") or "").lower()
    if "architect invocation failed" not in reason and "architect failed" not in reason:
        return False
    return any(marker in reason for marker in _TRANSIENT_ARCHITECT_MARKERS)


def route_after_architect(state: dict[str, Any]) -> str:
    """Is there a design to build, and if not, is that the model's fault?

    THE ARCHITECT FAILING IS FATAL TO THE ITERATION -- there is nothing for the
    Developer to build without a design, and letting it invent its own spec is
    worse than stopping. But "fatal to the iteration" was implemented as "fatal
    to the RUN", for every cause including a two-minute rate limit.

    run-c993a6e93050 ended on its 23rd iteration, after 3 hours and 12.5 million
    tokens, on a single HTTP 402 whose own body read:

        "This request would exceed your available credits given your current
         in-flight requests. Retry after in-flight requests settle, or add
         credits."   ... "Retry-After": "120"

    The provider said when to come back. Nothing asked. `llm/client.py` now
    treats that 402 as retryable and honours the stated wait, and this edge is
    the second line of defence for everything that still gets through: a
    transport failure sends the SAME iteration back to the Architect, bounded by
    `MAX_INFRA_RETRIES`, instead of ending the run.
    """
    if budget_exhausted(state):
        return "halt"
    if state.get("halt_reason"):
        if (
            _architect_failure_is_transient(state)
            and int(state.get("architect_retry_count", 0)) < config.MAX_INFRA_RETRIES
        ):
            log.warning(
                "architect -> architect: attempt %d/%d, the failure was TRANSPORT "
                "not design -- retrying the same iteration rather than ending the "
                "run. %s",
                int(state.get("architect_retry_count", 0)) + 1, config.MAX_INFRA_RETRIES,
                str(state["halt_reason"])[:200],
            )
            return "retry_architect"
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
    """The human-readable reason a run stopped, for `halt_reason` at END.

    "best MGS" MEANS THE BEST MGS. It used to be formatted from
    `state["mgs_score"]`, which is the LAST score, so run-8cf58d33b311 -- whose
    scores were 0.3172, 0.2222, 0.1830, 0.1190 -- signed off with
    `iteration budget exhausted after 5; best MGS=0.1190`. That is not a
    rounding difference or a cosmetic slip: it is the run's headline number
    reporting its worst result as its best, on exactly the runs where the
    distinction matters most.
    """
    if state.get("halt_reason"):
        return str(state["halt_reason"])
    budget = budget_exhausted(state)
    if budget:
        return budget
    if state.get("render_degraded") and config.HALT_ON_DEGRADED_EVAL:
        return (
            f"evaluator degraded: {int(state.get('n_render_degraded', 0))} answering "
            f"checkpoint(s) ({float(state.get('render_degraded_rate', 0.0)) * 100:.0f}%) "
            "fell back to raw evidence because the answerer did not respond; the "
            "scores would not describe the system under test"
        )
    mgs = float(state.get("mgs_score", 0.0))
    if mgs >= config.MGS_TARGET:
        return f"target reached: MGS={mgs:.4f} >= {config.MGS_TARGET}"
    if int(state.get("iteration_count", 0)) >= config.MAX_ITERATIONS:
        best_mgs, best_iteration = scoreboard.best_of(state)
        tail = (
            f"; best MGS={best_mgs:.4f} (iteration {best_iteration})"
            if best_iteration else f"; best MGS={best_mgs:.4f}"
        )
        if best_iteration and abs(best_mgs - mgs) > 1e-9:
            tail += f", final MGS={mgs:.4f}"
        return f"iteration budget exhausted after {config.MAX_ITERATIONS}{tail}"
    return "completed"
