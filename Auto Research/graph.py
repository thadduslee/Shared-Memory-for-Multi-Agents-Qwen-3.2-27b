"""Macro-graph assembly.

TOPOLOGY
========

                                  START
                                    |
                                    v
                         +---------------------+
                         |      architect      |  reads iter i's critique.md,
                         |  (no write, no sh)  |  then critique_summary.md;
                         |                     |  proposes design + migration
                         |                     |  + Developer work order; then
                         |                     |  appends iter i's summary to
                         |                     |  critique_summary.md
                         +----------+----------+
                                    | route_after_architect
                                    v
                         +---------------------+
                         |      developer      |   <-- ONE self-driving agent
             +---------->|  calls its own      |       (nodes/developer.py)
             |           |  tools, reads its   |       real tool schema, one
             |           |  own output, loops  |       conversation, its own loop
             |           +----------+----------+
             |                      | route_after_developer
             | infrastructure       |
             | +-----------+        |
             +-+infra_retry|        |
               +-----------+        |
                    exhausted       |          ok
              +---------------------+----------+
              |                                v
              |                     +---------------------+
              |                     |    eval_dispatch    | strip hidden fields,
              |                     |                     | shard, write manifest
              |                     +----------+----------+
              |                                | fan_out_evaluator  (Send x N)
              |                    +-----------+-----------+
              |                    v           v           v
              |              +---------+ +---------+ +---------+
              |              |eval_    | |eval_    | |eval_    |   concurrent,
              |              |worker 0 | |worker 1 | |worker N |   semaphore-bounded
              |              +----+----+ +----+----+ +----+----+
              |                   +-----------+-----------+
              |                               v
              |                     +---------------------+
              |                     |    eval_collect     | fan-in: one
              |                     |                     | predictions.jsonl
              |                     +----------+----------+
              |                                | route_after_collect
              |        fail-fast / empty       |          ok
              +<-------------------------------+----------+
              |                                           v
              |                                +---------------------+
              |                                |        judge        | U, A, F, MGS
              |                                |  (reads annotations)|
              |                                +----------+----------+
              |                                           | route_after_judge
              |                          scale_up (dev gate passed)  |
              |                        +--------------------+        | critic
              |                        v                             v
              |             back to eval_dispatch            +---------------------+
              |               with stage="full"              |       critic        |
              |                                              | marginal attribution|
              |                                              +----------+----------+
              |                                                         | route_after_critic
              |         architect (MGS < target, iters left)             |
              +<--------------------------------------------------------+
                                                                        | halt
                                                                        v
                                                                   +---------+
                                                                   | finalize|
                                                                   +----+----+
                                                                        v
                                                                       END

WHY THE TOPOLOGY LOOKS LIKE THIS
--------------------------------
* `developer` is ONE node, not a sub-graph. It used to be a ReAct micro-graph
  whose `think`/`act`/`observe` were separate LangGraph nodes, which meant the
  macro-graph drove the Developer's loop and the model itself had no tools --
  it emitted a fenced action and something else ran it. The ten DevToolbox
  tools are a real tool schema now and the loop lives inside the node, so the
  Developer builds, runs and reads its own work. The macro-graph's job at this
  edge is unchanged: it asks for a build and finds out whether it got one.
* `eval_dispatch` and `eval_collect` exist as separate nodes because `Send`
  fan-out requires a node to dispatch FROM and a node to converge INTO.  The
  collector runs exactly once after the whole superstep, which is what makes
  the fan-in deterministic.
* The `judge -> eval_dispatch` back-edge is the 50 -> 579 scale-up.  It is a
  loop rather than a second pair of nodes so that both stages provably run the
  same code; a duplicated "full_eval" subgraph could drift from the dev one and
  the gate would stop meaning anything.
* Almost every failure edge points at `architect`, never at `developer`.  A
  build that will not build and a batch that dies identically on every shard are
  both evidence about the DESIGN, and only the Architect can change that.  The
  ONE exception is `infra_retry`: an episode the Developer classified as an
  INFRASTRUCTURE failure -- transport timeouts, empty replies, a turn ceiling
  reached without a single gate tool ever running -- is not evidence about the
  design at all, and sending it to the Architect asks for a redesign in answer
  to a network outage.  run-8cf58d33b311 did exactly that: iteration 2 never
  called `run_tests` or `sql_exec`, its report said "unmet mandatory gates:
  tests_ok, migration_ok" anyway, and the redesign that answered it cost the run
  0.095 MGS.  See `nodes/developer.py::classify_failure` for how the three
  classifications are told apart, and note that only `infrastructure` retries:
  an episode that read files for sixty turns and wrote nothing has a work-order
  problem, and re-running the same work order would reproduce it.  Note that
  this is the edge for a build that failed AFTER the Developer had already
  looped on it: retrying the implementation is the Developer's own job and it
  has spent `MAX_DEV_RETRIES` doing exactly that before this edge is taken.  The
  evidence travels with the verdict: an exhausted Developer writes
  `dev_failure_report`, which `nodes/architect.py` renders into the next design
  turn as a critique.  This path is the one with no Critic feedback at all --
  a failed build never reaches the Judge -- so that report is the whole of what
  the Architect learns from the iteration.
* `finalize` exists so that `halt_reason` is populated on exactly one path into
  END, instead of each router having to remember to set it.
* The loop's memory of its own history is a FILE the Architect keeps, not an
  edge: `runs/critique_summary.md`.  The Critic writes one iteration's
  `critique.md` and stops there; the Architect is the only node that reads a
  critique and writes the design answering it in the same turn, so it is the one
  that can summarise a critique without a second model call and without a second
  reader to disagree about what the critique said.  It reads the notebook BEFORE
  it designs and appends to it AFTER, so the history a turn designs from is
  strictly the iterations before the critique in front of it.

`MOCK_MODE` is not consulted anywhere in this file. Flipping it swaps clients
behind the interfaces; the topology is byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

import config
import routers
import scoreboard
from nodes._common import node_span
from nodes.architect import architect_node
from nodes.critic import critic_node
from nodes.developer import developer_node
from nodes.judge import judge_node
from nodes.medical_evaluator import (
    eval_collect_node,
    eval_dispatch_node,
    eval_worker_node,
    fan_out_evaluator,
)
from state import OrchestratorState

log = logging.getLogger("orchestrator.graph")


async def scale_up_node(state: OrchestratorState) -> dict[str, Any]:
    """Flip the evaluation stage from the dev slice to the full run.

    A node rather than a router side effect: routers must stay pure so the
    routing tests can call them, and `eval_stage` has exactly one writer.
    """
    log.info("scaling up: dev slice passed the gate, moving to the full run")
    return {
        "eval_stage": "full",
        # `eval_results` accumulates across the dev shards; the full run needs a
        # clean accumulator or the collector would concatenate dev predictions
        # into the full file. The collector also filters by (iteration, stage),
        # so this is belt and braces -- deliberately.
        "predictions_path": "",
    }


async def infra_retry_node(state: OrchestratorState) -> dict[str, Any]:
    """Charge one infrastructure retry and send the SAME iteration back to build.

    A node rather than a bare `developer -> developer` self-edge for the reason
    `scale_up_node` is a node: routers stay pure so the routing tests can call
    them, and `infra_retry_count` gets exactly one writer.

    `dev_failure_report` is cleared on the way through. It is the Architect's
    only feedback channel for a failed build, and leaving a report behind that
    describes an OpenRouter outage would make the NEXT Architect turn -- after a
    retry that succeeded -- redesign against a failure that has already been
    recovered from.
    """
    iteration = int(state.get("iteration_count", 0))
    attempt = int(state.get("infra_retry_count", 0)) + 1
    async with node_span(
        "infra_retry", iteration, str(state.get("current_curriculum_phase") or "")
    ) as span:
        report = state.get("dev_failure_report") or {}
        span["attempt"] = attempt
        span["classification_reason"] = report.get("classification_reason") or ""
        log.warning(
            "infrastructure retry %d/%d for iteration %d: %s",
            attempt, config.MAX_INFRA_RETRIES, iteration,
            report.get("classification_reason") or "unclassified",
        )
        return {
            "infra_retry_count": attempt,
            "dev_failure_report": {},
            "halt_reason": None,
            "node_timings": [span],
        }


async def architect_retry_node(state: OrchestratorState) -> dict[str, Any]:
    """Charge one Architect retry and send the SAME iteration back to design.

    A node rather than a bare self-edge for the reason `infra_retry_node` is
    one: routers stay pure so the routing tests can call them, and
    `architect_retry_count` gets exactly one writer.

    `halt_reason` is cleared on the way through -- it is the flag the router
    keyed on, and leaving it set would make the retried Architect's own
    `route_after_architect` see a failure that has already been recovered from.

    NOTE the counter is NOT reset by `architect_node` the way `infra_retry_count`
    is: the Architect resets that one at the top of its own turn, so an Architect
    retry sharing it would clear its own budget on every attempt and loop until
    the wall clock. This counter is reset by the Developer instead, once an
    iteration has actually got as far as being built.
    """
    iteration = int(state.get("iteration_count", 0)) + 1
    attempt = int(state.get("architect_retry_count", 0)) + 1
    async with node_span(
        "architect_retry", iteration, str(state.get("current_curriculum_phase") or "")
    ) as span:
        span["attempt"] = attempt
        span["failure"] = str(state.get("halt_reason") or "")[:200]
        log.warning(
            "architect retry %d/%d for iteration %d: %s",
            attempt, config.MAX_INFRA_RETRIES, iteration,
            str(state.get("halt_reason") or "")[:200],
        )
        return {
            "architect_retry_count": attempt,
            "halt_reason": None,
            "node_timings": [span],
        }


async def finalize_node(state: OrchestratorState) -> dict[str, Any]:
    """The single exit. Populates `halt_reason` and advances the curriculum."""
    iteration = int(state.get("iteration_count", 0))
    async with node_span("finalize", iteration, str(state.get("current_curriculum_phase") or "")) as span:
        reason = routers.halt_reason_for(state)
        next_phase, complete = routers.advance_curriculum(state)
        board = scoreboard.summary(state)
        span["halt_reason"] = reason
        span["best_mgs"] = board["best_mgs"]
        span["best_iteration"] = board["best_iteration"]
        log.info(
            "RUN COMPLETE | %s | final MGS=%.4f U=%.4f A=%.4f F=%.4f | iterations=%d | "
            "curriculum: %s%s",
            reason, float(state.get("mgs_score", 0.0)), float(state.get("utility_score", 0.0)),
            float(state.get("access_violation_rate", 1.0)),
            float(state.get("forgetting_failure_rate", 1.0)), iteration,
            next_phase, " (complete)" if complete else "",
        )
        # The trend, printed once at the end, at WARNING when the run finished
        # below its own best. A run that walked away from its high water mark
        # should not be able to end on an INFO line that looks like every other
        # INFO line.
        if board["regressed_from_best"]:
            log.warning(
                "RUN REGRESSED: best MGS=%.4f at iteration %d, but the run ended at "
                "%.4f (iteration %d). The final workspace is NOT the best one -- "
                "see the `best_iteration` field in the run summary.",
                board["best_mgs"], board["best_iteration"],
                board["final_mgs"], board["final_iteration"],
            )
        elif board["best_iteration"]:
            log.info("RUN BEST: MGS=%.4f at iteration %d",
                     board["best_mgs"], board["best_iteration"])
        return {
            "halt_reason": reason,
            "current_curriculum_phase": next_phase,
            "node_timings": [span],
        }


async def curriculum_node(state: OrchestratorState) -> dict[str, Any]:
    """Advance (or repeat) the curriculum phase before the next iteration.

    Sits between `critic` and `architect` so that the Architect always sees the
    phase it is actually designing for.  On a phase failure this returns the
    SAME phase, which is what "halt the remaining phases and feed back" means:
    the loop does not move on until the current phase passes.
    """
    next_phase, complete = routers.advance_curriculum(state)
    current = str(state.get("current_curriculum_phase") or "")
    if next_phase != current:
        log.info("curriculum: %s PASSED -> advancing to %s", current, next_phase)
    else:
        log.info("curriculum: staying on %s (not yet passed)", current)
    return {"current_curriculum_phase": next_phase,
            "curriculum_history": [{"advanced_to": next_phase, "complete": complete}]}


def build_graph(checkpointer: Any | None = None):
    """Assemble and compile the macro-graph."""
    builder = StateGraph(OrchestratorState)

    builder.add_node("architect", architect_node)
    builder.add_node("developer", developer_node)
    builder.add_node("eval_dispatch", eval_dispatch_node)
    builder.add_node("eval_worker", eval_worker_node)
    builder.add_node("eval_collect", eval_collect_node)
    builder.add_node("judge", judge_node)
    builder.add_node("scale_up", scale_up_node)
    builder.add_node("critic", critic_node)
    builder.add_node("curriculum", curriculum_node)
    builder.add_node("infra_retry", infra_retry_node)
    builder.add_node("architect_retry", architect_retry_node)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "architect")

    builder.add_conditional_edges(
        "architect", routers.route_after_architect,
        {
            "developer": "developer",
            "retry_architect": "architect_retry",
            "halt": "finalize",
        },
    )
    builder.add_edge("architect_retry", "architect")

    # A Developer that cannot build the design sends the design back, not
    # forward -- unless the episode died of INFRASTRUCTURE, in which case it
    # goes back to the Developer through `infra_retry`, which is a node rather
    # than a bare self-edge so that `infra_retry_count` has exactly one writer
    # and the retry is visible in `node_timings`.
    builder.add_conditional_edges(
        "developer", routers.route_after_developer,
        {
            "evaluate": "eval_dispatch",
            "retry_developer": "infra_retry",
            "architect": "curriculum",
            "halt": "finalize",
        },
    )
    builder.add_edge("infra_retry", "developer")

    # THE FAN-OUT. A conditional edge is the only place `Send` is meaningful.
    builder.add_conditional_edges(
        "eval_dispatch", fan_out_evaluator, ["eval_worker", "eval_collect"],
    )
    # Every worker converges here; LangGraph runs the collector once, after the
    # whole superstep completes.
    builder.add_edge("eval_worker", "eval_collect")

    builder.add_conditional_edges(
        "eval_collect", routers.route_after_collect,
        {"judge": "judge", "architect": "curriculum", "halt": "finalize"},
    )

    # The scale-up back-edge: dev slice -> full run, same nodes, same code.
    builder.add_conditional_edges(
        "judge", routers.route_after_judge,
        {"scale_up": "scale_up", "critic": "critic", "halt": "finalize"},
    )
    builder.add_edge("scale_up", "eval_dispatch")

    builder.add_conditional_edges(
        "critic", routers.route_after_critic,
        {"architect": "curriculum", "halt": "finalize"},
    )
    builder.add_edge("curriculum", "architect")
    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=checkpointer)


def default_checkpointer():
    """In-memory checkpointer so a run is resumable within the process.

    # TODO(real): swap for `langgraph.checkpoint.sqlite.SqliteSaver` (or the
    #   Postgres saver) to survive a process restart. The graph takes any
    #   `BaseCheckpointSaver`, so this is a one-line change at the call site --
    #   deliberately not baked into `build_graph`.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver()
