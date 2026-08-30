"""Global (macro-graph) state and its reducers.

REDUCER RULE (brief section 5)
------------------------------
A field needs an explicit reducer if and only if more than one node can write
it *within a single superstep*.  In this graph that means exactly the fan-out
workers dispatched by `Send`.  Every other field is written by exactly one node
per superstep and therefore uses LangGraph's default last-write-wins.

Each field below is annotated with its writer.  If you add a node that writes a
field already owned by another node, you must either give the field a reducer
or reconsider the topology -- concurrent writes to a reducer-less field raise
`InvalidUpdateError` at runtime, which is the behaviour we want.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

# `dev` = the seeded 50-checkpoint slice; `full` = all 579.
EvalStage = Literal["dev", "full"]


def merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    """Shallow-merge reducer for dicts written by more than one writer.

    Right wins on key collision.  Used for `token_usage`, which the fan-out
    workers all contribute to; numeric keys are summed rather than overwritten
    because a token count is an accumulator, not a status.
    """
    out = dict(left or {})
    for key, value in (right or {}).items():
        if isinstance(value, (int, float)) and isinstance(out.get(key), (int, float)):
            out[key] = out[key] + value
        else:
            out[key] = value
    return out


# Sentinel a node returns to CLEAR a reduced field.
#
# This exists because of a real LangGraph property that is easy to get wrong:
# a reducer is applied to every update, so a node returning `[]` for an
# `operator.add` field produces `left + []`, which is `left` -- the accumulator
# is NOT cleared.  A research loop that re-runs the evaluator once per iteration
# must be able to clear it, so the reset is explicit and greppable rather than
# an accident waiting in a `[]` literal.
RESET = "__reset__"


def accumulate_or_reset(
    left: list[Any] | None, right: list[Any] | str | None
) -> list[Any]:
    """`operator.add` semantics plus an explicit `RESET` sentinel."""
    if right == RESET:
        return []
    if right is None:
        return list(left or [])
    return list(left or []) + list(right)


def keep_first_signature(left: str | None, right: str | None) -> str | None:
    """Reducer for `failure_signature`.

    The *first* signature recorded is the diagnostically interesting one -- it
    is the failure that tripped the circuit breaker.  Later workers failing the
    same way must not overwrite it, and a later worker failing a *different*
    way must not mask the dominant mode either.  `None` from a healthy worker
    never clears an existing signature; only the explicit `RESET` sentinel does,
    which the Architect sends at the top of each iteration.
    """
    if right == RESET:
        return None
    return left if left else right


class OrchestratorState(TypedDict, total=False):
    """State threaded through the macro research loop."""

    # ---------------- Architect writes ----------------
    proposed_design: str            # Markdown design doc + machine-parseable JSON block
    sql_schema: str                 # current SQL DDL for the memory store
    migration_sql: str              # ALTER TABLE / CREATE INDEX statements for this iteration
    dev_instructions: str           # Architect -> Developer ordered work order

    # ---------------- Developer writes ----------------
    memory_codebase: str            # path to the workspace holding the current implementation
    codebase_delta: str             # unified diff of what the Developer changed this iteration
    dev_set_pass_rate: float        # local unit-test pass rate from the Developer's own suite
    dev_retries_used: int
    dev_files_changed: list[str]    # source files this episode actually changed (not inherited)
    # WHY THE DEVELOPER REPORTS ITS FAILURE IN DETAIL, not just a signature.
    #
    # `failure_signature` is a hash: it is enough to trip a circuit breaker and
    # enough to notice the SAME failure twice, and it is useless as feedback --
    # the Architect cannot read a hash and infer that its work order asked for a
    # column the tests forbid. A build that never compiled is a critique of the
    # design (see the routing note in routers.route_after_developer), so the
    # evidence has to travel with the verdict.
    #
    # Written by the Developer alone, on EVERY run: `{}` on a green build, so a
    # stale report from iteration N-1 can never be read as this iteration's news.
    dev_failure_report: dict[str, Any]
    # One compact entry per FAILED build, accumulated across iterations. A
    # signature that recurs is the diagnostically important case: it means the
    # last redesign did not address the thing that actually broke, and the
    # Architect needs to be told that rather than left to rediscover it.
    dev_failure_history: Annotated[list[dict[str, Any]], operator.add]

    # ---------------- Medical Evaluator writes ----------------
    # `eval_results` is the fan-in accumulator: one entry per Send shard.
    # A reducer is mandatory here -- every worker in the superstep writes it.
    # `accumulate_or_reset` rather than bare `operator.add` so the Architect can
    # clear it between iterations (see the RESET note above).
    eval_results: Annotated[list[dict[str, Any]], accumulate_or_reset]
    eval_stage: EvalStage           # written by the dispatcher, read by workers
    # The fan-out plan. Written by eval_dispatch (alone), read by the
    # `fan_out_evaluator` conditional edge. Carries ids and paths only -- never
    # checkpoint bodies -- so the checkpointer stays small.
    shard_plan: list[dict[str, Any]]
    predictions_path: str           # written by the collector after concatenating shards
    n_checkpoints_evaluated: int
    n_worker_failures: int

    # ---------------- Judge writes ----------------
    judge_report: dict[str, Any]    # full per-metric / per-category breakdown
    mgs_score: float                # MGS = U * (1 - A) * (1 - F)
    utility_score: float            # U  (legacy key: utility_accuracy)
    access_violation_rate: float    # A  (legacy key: privacy_leakage_rate)
    forgetting_failure_rate: float  # F  (legacy key: deletion_leakage_rate)
    proceed_to_full: bool           # the 50 -> 579 scale-up gate decision

    # ---------------- Critic writes ----------------
    critique: str
    attribution: dict[str, Any]     # per-term marginal contribution ranking

    # ---------------- Router / control-plane writes ----------------
    iteration_count: int
    current_curriculum_phase: str
    curriculum_history: Annotated[list[dict[str, Any]], operator.add]

    # `failure_signature` is written by fan-out workers AND by the Developer, so
    # it carries a reducer even though the Developer writes it alone.
    failure_signature: Annotated[str | None, keep_first_signature]
    halt_reason: str | None

    # ---------------- Budget accounting ----------------
    started_at: float               # monotonic clock at graph entry
    token_usage: Annotated[dict[str, Any], merge_dicts]
    node_timings: Annotated[list[dict[str, Any]], operator.add]


class WorkerState(TypedDict, total=False):
    """Payload carried by one `Send` to the evaluator fan-out.

    This is deliberately *not* the full OrchestratorState: a worker gets only
    the shard it must run plus the context needed to run it.  Keeping the
    payload small is what makes partial failure cheap -- a dead worker loses one
    shard, not the run.

    `checkpoints` here have ALREADY been through
    gatemem_adapter.strip_hidden_fields(); see medical_evaluator.py.
    """

    shard_index: int
    shard_total: int
    stage: EvalStage
    iteration: int
    checkpoints: list[dict[str, Any]]
    episodes_index: dict[str, Any]
    workspace: str
    curriculum_phase: str


class DeveloperState(TypedDict, total=False):
    """Local state of one Developer episode.

    Lives inside `DeveloperSession` only; `developer_node` returns a small
    projection of it back into OrchestratorState (see nodes/developer.py).
    That projection is what keeps the Developer's own chatter -- its scratchpad,
    its retry counter, its conversation -- out of the research loop's state and
    out of every checkpointer snapshot.

    The reducers annotated below are applied by `DeveloperSession._apply` rather
    than by LangGraph: the episode is a plain loop now, not a sub-graph, so
    nothing else is going to fold these updates. They are still declared here
    because they are still the contract -- `scratchpad` and `failures`
    accumulate, `tokens_used` sums -- and a phase returning `{"scratchpad":
    [entry]}` means "append" wherever it is read.
    """

    instructions: str
    sql_schema: str
    migration_sql: str
    workspace: str
    iteration: int
    # "template" | "iter_N" | "empty" | "existing" -- where this iteration's
    # starting code came from. Surfaced in the status block so the model knows
    # whether it is editing an existing baseline or bootstrapping.
    workspace_from: str

    # The loop proper.
    scratchpad: Annotated[list[dict[str, Any]], operator.add]
    thought: str
    action: dict[str, Any]          # {"tool": str, "args": {...}}
    observation: dict[str, Any]     # {"ok": bool, "stdout": str, ...}
    retry_count: int
    last_stack_trace: str | None
    # Every retry-charged failure, in order.  The scratchpad holds the whole
    # episode including its green steps; this holds only the steps that spent
    # the retry budget, which is the part the Architect needs to see.  Redirects
    # are deliberately absent -- they are not charged, and a toolbox refusing an
    # action says nothing about whether the design can be built.
    failures: Annotated[list[dict[str, Any]], operator.add]

    # Exit-condition evidence.  All three must be True to leave the loop.
    compile_ok: bool
    tests_ok: bool
    migration_ok: bool
    lint_ok: bool
    smoke_ok: bool
    pass_rate: float
    # A SUMMING accumulator, not last-write-wins. Every `dev_think` turn writes
    # this, and when it was overwritten instead a 20-turn episode reported the
    # cost of one turn. See the note in nodes/developer.py on why the
    # Developer's spend has to be counted.
    tokens_used: Annotated[dict[str, Any], merge_dicts]

    done: bool
    failure_signature: str | None
    halt_reason: str | None
    codebase_delta: str


def initial_state(*, workspace: str, started_at: float) -> OrchestratorState:
    """Seed state for a fresh run.

    Fields with reducers must be initialised to an empty container of the right
    type, otherwise the first reduction has nothing to add to.
    """
    from config import CURRICULUM_PHASES

    return OrchestratorState(
        proposed_design="",
        sql_schema="",
        migration_sql="",
        dev_instructions="",
        memory_codebase=workspace,
        codebase_delta="",
        dev_set_pass_rate=0.0,
        dev_retries_used=0,
        dev_files_changed=[],
        dev_failure_report={},
        dev_failure_history=[],
        eval_results=[],  # seeded empty; cleared later via the RESET sentinel
        eval_stage="dev",
        predictions_path="",
        n_checkpoints_evaluated=0,
        n_worker_failures=0,
        judge_report={},
        mgs_score=0.0,
        utility_score=0.0,
        access_violation_rate=1.0,
        forgetting_failure_rate=1.0,
        proceed_to_full=False,
        critique="",
        attribution={},
        iteration_count=0,
        current_curriculum_phase=CURRICULUM_PHASES[0],
        curriculum_history=[],
        failure_signature=None,
        halt_reason=None,
        started_at=started_at,
        token_usage={"total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        node_timings=[],
    )
