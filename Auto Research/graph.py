"""Macro-graph assembly.

TOPOLOGY
========

                                  START
                                    |
                                    v
                         +---------------------+
                         |      architect      |  proposes design + migration
                         |  (no write, no sh)  |  + Developer work order
                         +----------+----------+
                                    | route_after_architect
                                    v
                         +---------------------+
                         |      developer      |   <-- ONE self-driving agent
                         |  calls its own      |       (nodes/developer.py)
                         |  tools, reads its   |       real tool schema, one
                         |  own output, loops  |       conversation, its own loop
                         +----------+----------+
                                    | route_after_developer
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
* Every failure edge points at `architect`, never at `developer`.  A build that
  will not build and a batch that dies identically on every shard are both
  evidence about the DESIGN, and only the Architect can change that.  Note that
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

`MOCK_MODE` is not consulted anywhere in this file. Flipping it swaps clients
behind the interfaces; the topology is byte-identical.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

import routers
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
from nodes._common import node_span
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


async def finalize_node(state: OrchestratorState) -> dict[str, Any]:
    """The single exit. Populates `halt_reason` and advances the curriculum."""
    iteration = int(state.get("iteration_count", 0))
    async with node_span("finalize", iteration, str(state.get("current_curriculum_phase") or "")) as span:
        reason = routers.halt_reason_for(state)
        next_phase, complete = routers.advance_curriculum(state)
        span["halt_reason"] = reason
        log.info(
            "RUN COMPLETE | %s | MGS=%.4f U=%.4f A=%.4f F=%.4f | iterations=%d | "
            "curriculum: %s%s",
            reason, float(state.get("mgs_score", 0.0)), float(state.get("utility_score", 0.0)),
            float(state.get("access_violation_rate", 1.0)),
            float(state.get("forgetting_failure_rate", 1.0)), iteration,
            next_phase, " (complete)" if complete else "",
        )
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
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "architect")

    builder.add_conditional_edges(
        "architect", routers.route_after_architect,
        {"developer": "developer", "halt": "finalize"},
    )

    # A Developer that cannot build the design sends the design back, not forward.
    builder.add_conditional_edges(
        "developer", routers.route_after_developer,
        {"evaluate": "eval_dispatch", "architect": "curriculum", "halt": "finalize"},
    )

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
