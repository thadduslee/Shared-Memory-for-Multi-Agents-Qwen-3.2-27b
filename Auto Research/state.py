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
    # THE LOOP'S MEMORY OF ITS OWN EARLIER ITERATIONS.
    #
    # One capped entry per earlier iteration -- `{iteration, kind, summary}` --
    # written by the Architect that read that iteration's critique. `critique`
    # below only ever holds the LAST one, so before this existed iteration 4 had
    # no record at all of what iterations 1 and 2 had already diagnosed and
    # tried.
    #
    # THE ARTIFACT IS `runs/critique_summary.md` -- the Architect's notebook,
    # which it appends these same rows to at the END of its turn, after the
    # design is written. This list is the structured mirror of that file: it is
    # what the once-per-iteration dedupe reads, and it is what the notebook is
    # rebuilt from if RUNS_DIR was wiped or repointed between turns.
    #
    # `operator.add` because it is append-only across the run and, like
    # `dev_failure_history`, is never reset between iterations -- resetting it
    # would throw away the only thing it is for. Entries are capped at
    # config.ARCHITECT_CRITIQUE_RECAP_MAX_TOKENS each; see nodes/_recap.py.
    critique_digest: Annotated[list[dict[str, Any]], operator.add]

    # ---------------- Developer writes ----------------
    memory_codebase: str            # path to the workspace holding the current implementation
    codebase_delta: str             # unified diff of what the Developer changed this iteration
    dev_set_pass_rate: float        # local unit-test pass rate from the Developer's own suite
    dev_retries_used: int
    # Turns the last Developer episode spent, and the ceiling it spent them
    # against. Surfaced to the Architect because work-order SIZE is the lever it
    # controls and turn exhaustion is the failure that lever causes: an episode
    # that runs out of turns before calling its gate tools is scored as a failed
    # build and every edit it made is discarded.
    dev_turns_used: int
    dev_turn_ceiling: int
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
    # How many times the CURRENT iteration has been re-run because its Developer
    # episode died of infrastructure rather than of the design. Reset to 0 by
    # the Architect at the top of every iteration; bounded by
    # config.MAX_INFRA_RETRIES. See routers.route_after_developer.
    infra_retry_count: int
    # How many times the CURRENT iteration's ARCHITECT call has been retried
    # after a transport failure. Separate from `infra_retry_count` because the
    # Architect resets that one at the top of its own turn -- sharing a counter
    # would let an Architect retry clear its own budget and loop forever.
    architect_retry_count: int

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
    # WHETHER THE ANSWERER WAS ACTUALLY ANSWERING.
    #
    # A render call that fails falls back to the gated record bodies, which is
    # right for one checkpoint and, unflagged, catastrophic for a whole run:
    # every downstream number then describes the gating layer with raw evidence
    # pasted in rather than the system under test. run-c993a6e93050 spent 23
    # iterations and 12.5M tokens in exactly that state -- 0 of 357 answering
    # predictions written by a model -- and reported `best MGS=0.8366` with no
    # caveat. Written by the collector; read by the Judge, the scoreboard and
    # `routers.route_after_collect`.
    render_degraded: bool
    render_degraded_rate: float
    n_render_degraded: int

    # ---------------- Judge writes ----------------
    judge_report: dict[str, Any]    # full per-metric / per-category breakdown
    mgs_score: float                # MGS = U * (1 - A) * (1 - F)
    utility_score: float            # U  (legacy key: utility_accuracy)
    access_violation_rate: float    # A  (legacy key: privacy_leakage_rate)
    forgetting_failure_rate: float  # F  (legacy key: deletion_leakage_rate)
    proceed_to_full: bool           # the 50 -> 579 scale-up gate decision
    # THE LOOP'S MEMORY OF ITS OWN SCORES -- one row per (iteration, stage) the
    # Judge scored, appended and never reset.
    #
    # The four scalars above are the LATEST measurement and nothing else; every
    # iteration overwrites them. That is what made a self-improving loop unable
    # to tell improvement from collapse: `prepare_workspace` inherited the last
    # code rather than the best code, the Architect was shown one iteration's
    # numbers with no trend, the Critic had no way to say "revert", and
    # `halt_reason` printed the final score under the label "best". All four
    # read this list now. See scoreboard.py for the row shape and every
    # derived quantity -- best-so-far, the champion, the regression verdict.
    #
    # `operator.add` and never reset, for the same reason as `critique_digest`:
    # discarding it is discarding the only thing it is for.
    score_history: Annotated[list[dict[str, Any]], operator.add]

    # ---------------- Critic writes ----------------
    critique: str
    # WHICH iteration `critique` is from -- not always `iteration_count - 1`.
    #
    # A build that never compiled never reaches the Judge, so the Critic does not
    # run and `critique` stays as whatever the last iteration that DID build
    # produced. The Architect summarises the critique it is handed exactly once,
    # and this is what it dedupes on: without it, a run with two failed builds
    # would write the same critique into the recap three times and read that
    # repetition as three separate iterations reaching the same conclusion.
    critique_iteration: int
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
    # WHETHER EACH GATE WAS EVER ACTUALLY ATTEMPTED, as opposed to merely being
    # False. The three booleans above start False and are only ever set by their
    # tool running, so `tests_ok=False` is ambiguous between "run_tests failed"
    # and "run_tests was never called" -- and the failure report reported both
    # as "unmet mandatory gate: tests_ok". run-8cf58d33b311 iteration 2 spent
    # its episode on transport timeouts and never called run_tests OR sql_exec;
    # the Architect was told its design had failed the tests, and redesigned.
    #
    # Shape: {gate_name: {"ran": int, "ok": bool}} -- `ran` counts completed
    # calls of the gate's tool, `ok` is the latest outcome. Written by
    # dev_observe; read by `classify_failure` and by the failure report.
    gate_attempts: dict[str, Any]
    # Model turns lost to the TRANSPORT rather than to the code: a timed-out
    # completion, an empty reply, a dead endpoint. Counted separately from
    # `retry_count` because they are evidence about the network, not about
    # whether the design can be built. See classify_failure.
    transport_failures: int
    transport_errors: Annotated[list[str], operator.add]
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
        dev_turns_used=0,
        dev_turn_ceiling=0,
        dev_files_changed=[],
        dev_failure_report={},
        dev_failure_history=[],
        infra_retry_count=0,
        architect_retry_count=0,
        eval_results=[],  # seeded empty; cleared later via the RESET sentinel
        eval_stage="dev",
        predictions_path="",
        n_checkpoints_evaluated=0,
        n_worker_failures=0,
        render_degraded=False,
        render_degraded_rate=0.0,
        n_render_degraded=0,
        judge_report={},
        mgs_score=0.0,
        utility_score=0.0,
        access_violation_rate=1.0,
        forgetting_failure_rate=1.0,
        proceed_to_full=False,
        score_history=[],
        critique="",
        critique_iteration=0,
        critique_digest=[],
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
