"""Developer node -- ONE self-driving agent that builds, runs and reads its own work.

    +-------------------------------------------------------------+
    |  developer_node                                              |
    |                                                              |
    |    think ---> act ---> observe ---+--> failed? --> think      |
    |      ^                            |                          |
    |      +----------------------------+                          |
    |                                   +--> gates green --> done  |
    +-------------------------------------------------------------+

WHAT CHANGED, AND WHY
=====================
This used to be a LangGraph MICRO-GRAPH: `think`, `act` and `observe` were three
separate nodes, and the graph -- not the Developer -- drove the loop. That split
cost two things, both measured:

1.  THE MODEL HAD NO TOOLS. Its persona described ten tools in prose, it emitted
    one fenced JSON action, and the orchestrator executed it and re-prompted.
    But DEVELOPER_MODEL is a tool-calling model, and a tool-calling model handed
    no tool schema does not fall back to prose -- it emits its NATIVE tool-call
    syntax as plain text. runs_multi/run-280411c75b99 produced 27 of those,
    parsed zero, wrote zero bytes, and evaluated byte-identical code across four
    iterations. `parse_xml_action` below was the salvage.

2.  EVERY TURN STARTED FROM NOTHING. Each `think` rebuilt the whole task string
    from scratch and sent it as a single user message, so the model never saw
    its own transcript -- only a six-entry rendering of it. It could not tell
    "I ran the tests and they failed" from "someone told me the tests failed".

So the loop lives HERE now, and the Developer drives it: the ten DevToolbox
tools are declared as a real tool schema (`nodes/dev_tools.TOOL_SCHEMAS`), the
model calls them, their real output comes back as `tool` messages in ITS OWN
conversation, and it keeps going until the work is done or the retries are gone.
`think`, `act` and `observe` are still here as functions -- they are the phases
of that loop, and each one is a policy worth testing on its own -- but nothing
between them is a graph edge any more.

EXIT CONDITION (brief 6.2): compile_check AND run_tests AND sql_exec must all be
green.  `run_linter` and `run_sandbox_smoke_test` are run but are advisory --
a style finding must not be able to block a correct build, and the smoke test
overlaps with the evaluation that follows.

WHAT STILL BOUNDS THE EPISODE. The Developer looping on its own failures is the
point, but it is not a licence to loop forever, and three limits are unchanged
in meaning:
  * `MAX_DEV_RETRIES` -- failed observations. This is the "this design cannot be
    built" budget, which is why a redirect and a decoder stutter are exempt.
  * `DSH_DEVELOPER_THINK_TIMEOUT_S` -- one model turn.
  * `DSH_DEVELOPER_TIMEOUT_S` -- the whole episode's wall clock. This used to be
    documented as the outer stop and read by nothing at all; the loop enforces
    it now, which is what replaces the sub-graph's `recursion_limit`.

ON EXHAUSTION: set `halt_reason`, `failure_signature` and `dev_failure_report`,
then route back to the Architect.  A Developer that cannot build the design is
evidence about the *design*, so the feedback goes to the node that owns it --
and it goes with the WHERE, not just the fact.  `dev_failure_report` carries the
unmet gates, the error line that ended each retry-charged step and the final
trace; nodes/architect.py renders it as a critique of the design in the very
next iteration.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config
import scoreboard
from harness.dsh_client import DSHProfile, DSHResult, extract_json_block
from harness.profiles import DEVELOPER_PROFILE
from nodes._common import node_span, write_artifact
from nodes._transport import agent_call
from nodes.dev_tools import (
    TOOL_NAMES,
    DevToolbox,
    ToolResult,
    is_scratch,
    signature_from_trace,
    tool_schemas,
)
from state import DeveloperState, OrchestratorState, merge_dicts

log = logging.getLogger("orchestrator.developer")

# One toolbox per running episode, keyed by workspace.  The episode's state must
# stay JSON-serializable -- it is written to disk as `developer_scratchpad.json`
# and projected into the checkpointed macro-graph state -- so the toolbox (which
# holds an open filesystem scope) lives beside it rather than inside it.
_TOOLBOXES: dict[str, DevToolbox] = {}


def _toolbox(state: DeveloperState) -> DevToolbox:
    workspace = str(state.get("workspace") or "")
    box = _TOOLBOXES.get(workspace)
    if box is None or box.migration_sql != state.get("migration_sql", ""):
        box = DevToolbox(
            Path(workspace),
            migration_sql=str(state.get("migration_sql") or ""),
            iteration=int(state.get("iteration", 1)),
        )
        _TOOLBOXES[workspace] = box
    return box


# Which tool call sets each exit-condition flag.  A flag is set ONLY as a side
# effect of that tool actually running in this episode -- see dev_observe.  The
# mapping exists so the rejection can name the missing *tool*, not just the
# missing flag: in runs/iter_1 the model spent all five retries re-asserting in
# prose that the migration had been applied, having never called `sql_exec`.
GATE_TOOLS: dict[str, str] = {
    "compile_ok": "compile_check",
    "tests_ok": "run_tests",
    "migration_ok": "sql_exec",
}

# Tools whose result cannot change unless something is written in between, so a
# consecutive identical call is by definition uninformative.  Write tools and
# `finish` are deliberately absent: re-writing a file is a legitimate step, and
# a repeated `finish` is already handled by the gate below.
_IDEMPOTENT_TOOLS = frozenset(
    {"read_file", "list_dir", "compile_check", "run_tests", "run_linter",
     "run_sandbox_smoke_test"}
)

# Tools that change the workspace. Re-running one after a SUCCESS is legitimate;
# re-sending the byte-identical call that already FAILED is not -- see
# `_identical_failed_edits`.
_EDIT_TOOLS = frozenset({"apply_patch", "write_file"})

_SCHEMA_VERSION_RE = re.compile(r"^SCHEMA_VERSION\s*=\s*(\d+)", re.MULTILINE)


def _schema_version(workspace: Path) -> int | None:
    """The store's declared schema generation, or None if it declares none.

    Reported in the status block so the loop can distinguish "this iteration's
    migration has been applied" from "it has not" without re-running sql_exec.
    """
    store = workspace / "memory_system" / "store.py"
    if not store.is_file():
        return None
    match = _SCHEMA_VERSION_RE.search(store.read_text(encoding="utf-8", errors="replace"))
    return int(match.group(1)) if match else None


def _last_error_line(trace: str | None) -> str:
    """The exception line of a trace, or "" -- never an IndexError.

    `trace.strip().splitlines()[-1]` raises on a WHITESPACE-ONLY trace: the
    string is truthy, so the guard passes, but stripping it leaves "" and
    `"".splitlines()` is the empty list. That is reachable in a real run --
    `run_sandbox_smoke_test` stores the child's stderr as the trace, and a child
    that writes a single newline before failing produces exactly this. The crash
    would land in `_status_block`, i.e. inside `dev_think`, taking down the
    Developer node and the whole graph on a cosmetic detail of a log line.
    """
    lines = (trace or "").strip().splitlines()
    return lines[-1].strip() if lines else ""


# ======================================================================
# failure evidence -- what the Architect is shown when the build dies
# ======================================================================
#
# WHY THIS EXISTS. A Developer that exhausts its retries routes back to the
# Architect (see routers.route_after_developer) because an unbuildable design is
# a fact about the DESIGN. Until now the only thing that travelled with it was
# `failure_signature` -- a hash -- and a `halt_reason` naming which booleans were
# false. The Architect could therefore see THAT the build failed and never WHERE,
# so iteration N+1 restated the same work order and failed the same way: the
# feedback edge existed in the topology but carried no information across it.
#
# What follows distils the episode into the part that is actually evidence: the
# steps that spent the retry budget, with the error line that ended each one.

# Per-failure observation excerpt. Small on purpose: the Architect needs the
# assertion that fired, not the full pytest run, and there can be MAX_DEV_RETRIES
# of these plus the trace below.
_FAILURE_OBSERVATION_MAX_CHARS = 1200
# The final trace is the one the model was still looking at when it ran out of
# retries, so it gets more room than the per-step excerpts.
_FAILURE_TRACE_MAX_CHARS = 3000
# An argument value longer than this is summarized by length. `write_file`
# content is the whole point: a 20k-character file body in the report would
# crowd out every other failure and tells the Architect nothing it cannot read
# in the code view.
_FAILURE_ARG_MAX_CHARS = 120


# A line that starts with an exception type -- "AssertionError: ...".  Matched
# against the same shape `signature_from_trace` keys on, so the line the
# Architect is shown and the signature the run is deduplicated by describe the
# same event.
_EXCEPTION_LINE_RE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Warning)\b")


def _error_line(text: str) -> str:
    """The exception line of a trace, preferring it over the trailing line.

    `_last_error_line` takes the FINAL line, which is right for a bare traceback
    and wrong for the output the test gate actually produces: a pytest run ends
    with "1 failed, 17 passed in 0.09s" -- a scoreboard that names no cause.
    That is fine in the status block, where the Developer can just re-read the
    output, and useless here, because this line is the one-line summary the
    Architect is being asked to change a design in response to.
    """
    for line in reversed((text or "").strip().splitlines()):
        stripped = line.strip()
        if _EXCEPTION_LINE_RE.match(stripped):
            return stripped
    return _last_error_line(text)


def _tail(text: str, max_chars: int) -> str:
    """Truncate from the FRONT, keeping the end.

    Same reasoning as `ToolResult.observation`: the exception line is last, and
    it is the only part of a traceback the reader actually needs.
    """
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return "[... truncated ...]\n" + text[-max_chars:]


def _compact_args(action: dict[str, Any]) -> dict[str, Any]:
    """The action's arguments, with long values replaced by their length."""
    args = action.get("args") if isinstance(action, dict) else None
    if not isinstance(args, dict):
        return {}
    compact: dict[str, Any] = {}
    for key, value in args.items():
        if value is None or isinstance(value, (bool, int, float)):
            compact[key] = value
            continue
        text = str(value)
        compact[key] = text if len(text) <= _FAILURE_ARG_MAX_CHARS else f"<{len(text)} chars>"
    return compact


def _failure_entry(state: DeveloperState, observation: dict[str, Any]) -> dict[str, Any]:
    """One retry-charged failure, distilled.

    `kind` separates the two things that can spend a retry, because they call
    for opposite responses from the Architect. A "build" failure is the design
    meeting the code: a test that will not pass, DDL that will not apply. A
    "loop" failure is the episode failing to progress -- a repeated idempotent
    call, or a turn that produced no parseable action -- and redesigning the
    schema in response to one would be treating a decoder stutter as a
    requirement. An episode made ENTIRELY of loop failures is still worth
    saying, though: it usually means the work order was too vague to act on.
    """
    data = observation.get("data") or {}
    action = state.get("action") or {}
    text = str(observation.get("text") or "")
    trace = str(observation.get("stack_trace") or "")
    return {
        "step": len(state.get("scratchpad") or []) + 1,
        "tool": str(observation.get("tool") or ""),
        "args": _compact_args(action),
        "kind": "loop" if data.get("skipped_duplicate") or action.get("_fallback") else "build",
        "signature": observation.get("signature"),
        "error": _error_line(trace) or _error_line(text),
        "observation": _tail(text, _FAILURE_OBSERVATION_MAX_CHARS),
    }


#: What `classify_failure` can conclude, and what each conclusion means for the
#: graph. `routers.route_after_developer` retries `infrastructure` and only
#: `infrastructure`; the other two go back to the Architect as they always did.
CLASSIFY_DESIGN = "design"
CLASSIFY_INFRASTRUCTURE = "infrastructure"
CLASSIFY_INCONCLUSIVE = "inconclusive"


def gate_status(final: dict[str, Any]) -> dict[str, str]:
    """Per-gate: `passed`, `failed`, or `never_ran`.

    THE DISTINCTION THIS DRAWS IS THE WHOLE POINT. The three gate booleans start
    False and are only ever set by their tool completing, so `tests_ok=False`
    means EITHER "run_tests ran and the suite failed" OR "run_tests was never
    called". Those are opposite facts about a design and the report used to
    render both as `unmet mandatory gates: tests_ok`.

    In run-8cf58d33b311 iteration 2 the truth was the second one -- the episode
    died of transport timeouts having never called `run_tests` or `sql_exec` --
    and the Architect, told its tests had failed, rewrote the retrieval loop.
    """
    attempts = final.get("gate_attempts") or {}
    status: dict[str, str] = {}
    for gate in GATE_TOOLS:
        entry = attempts.get(gate) if isinstance(attempts.get(gate), dict) else None
        ran = bool(entry and int(entry.get("ran", 0)))
        # A TRUE FLAG IS ITSELF PROOF THE TOOL RAN. `dev_observe` is the only
        # writer of these three booleans and it only ever sets one as a side
        # effect of that gate's tool completing, so `compile_ok is True` cannot
        # mean anything except that `compile_check` ran and passed. Inferring it
        # here rather than requiring the attempt record keeps this function
        # correct for episode states that predate `gate_attempts` -- a resumed
        # run rehydrated from an older checkpointer snapshot, or a test fixture.
        if bool(final.get(gate)):
            status[gate] = "passed"
        elif ran:
            status[gate] = "failed"
        else:
            status[gate] = "never_ran"
    return status


def classify_failure(final: dict[str, Any]) -> tuple[str, str]:
    """Why this episode really failed: `(classification, human reason)`.

    Three answers, and the graph treats them differently:

    * `infrastructure` -- the episode never got a fair attempt. Either the
      transport ate at least half the charged retries, or no mandatory gate was
      ever run AND the transport failed at least once. Nothing here is evidence
      about the design, so `routers.route_after_developer` sends the SAME
      iteration back to the Developer rather than asking for a redesign.
    * `inconclusive` -- the episode ran cleanly and still never exercised a
      mandatory gate: it read, listed and thought its way to the turn ceiling
      without building anything. That is a work-order problem, not a schema
      problem, and re-running the identical work order would reproduce it -- so
      it goes to the Architect, but labelled, so the Architect does not read a
      phantom test failure into it.
    * `design` -- a mandatory gate actually ran and actually failed. This is the
      only case the original report was ever right about, and it is the only one
      that should provoke a redesign.
    """
    transport = int(final.get("transport_failures", 0) or 0)
    retries = int(final.get("retry_count", 0) or 0)
    status = gate_status(final)
    ran_any = any(value != "never_ran" for value in status.values())
    failed_gates = [gate for gate, value in status.items() if value == "failed"]
    never_ran = [gate for gate, value in status.items() if value == "never_ran"]

    if transport and (not ran_any or transport * 2 >= max(retries, 1)):
        return CLASSIFY_INFRASTRUCTURE, (
            f"{transport} of {retries} charged retry/ies were transport failures"
            + ("" if ran_any else "; no mandatory gate tool ever completed")
        )
    if failed_gates:
        return CLASSIFY_DESIGN, (
            "gate(s) ran and failed: " + ", ".join(sorted(failed_gates))
        )
    if never_ran:
        return CLASSIFY_INCONCLUSIVE, (
            "no mandatory gate failed; "
            + ", ".join(f"`{GATE_TOOLS[gate]}` never ran" for gate in sorted(never_ran))
        )
    return CLASSIFY_DESIGN, "all mandatory gates passed but the episode did not finish"


def _failure_report(
    final: dict[str, Any],
    *,
    iteration: int,
    provenance: str,
    signature: str,
    reason: str,
    files_written: list[str],
) -> dict[str, Any]:
    """The whole build failure, as the Architect will read it.

    Pure so it can be tested without running an episode, and JSON-safe
    because it goes into macro-graph state and therefore into every
    checkpointer snapshot.
    """
    gates = {
        name: bool(final.get(name))
        for name in ("compile_ok", "tests_ok", "migration_ok", "lint_ok", "smoke_ok")
    }
    trace = str(final.get("last_stack_trace") or "")
    failures = [entry for entry in (final.get("failures") or []) if isinstance(entry, dict)]
    status = gate_status(final)
    classification, classification_reason = classify_failure(final)
    transport_errors = [str(e) for e in (final.get("transport_errors") or []) if e]
    return {
        "iteration": iteration,
        "reason": reason,
        "signature": signature,
        "gates": gates,
        # Only the three MANDATORY gates. Reporting `lint_ok` as "unmet" would
        # invite the Architect to redesign around a style finding that never
        # blocked anything (see the advisory note at the top of this module).
        "missing_gates": [name for name in GATE_TOOLS if not gates[name]],
        # `missing_gates` conflates two opposite facts; this separates them.
        # A gate that never ran is not a gate that failed, and the Architect
        # renders these rather than the bare list. See `gate_status`.
        "gate_status": status,
        "gates_failed": sorted(g for g, v in status.items() if v == "failed"),
        "gates_never_ran": sorted(g for g, v in status.items() if v == "never_ran"),
        # Why the episode really died, and whether the graph should redesign in
        # response or just run the same iteration again. See `classify_failure`.
        "classification": classification,
        "classification_reason": classification_reason,
        "n_transport_failures": int(final.get("transport_failures", 0) or 0),
        # Deduped, newest last: a flapping endpoint produces the same string
        # every time and five copies of it says nothing six does not.
        "transport_errors": list(dict.fromkeys(transport_errors))[:5],
        "gate_tools": dict(GATE_TOOLS),
        "pass_rate": float(final.get("pass_rate", 0.0)),
        "retries_used": int(final.get("retry_count", 0)),
        "retry_cap": int(config.MAX_DEV_RETRIES),
        "exhausted": int(final.get("retry_count", 0)) >= config.MAX_DEV_RETRIES,
        "workspace_from": provenance,
        "files_written": files_written,
        "last_error": _error_line(trace),
        "last_stack_trace": _tail(trace, _FAILURE_TRACE_MAX_CHARS),
        "failures": failures,
        "n_build_failures": sum(1 for e in failures if e.get("kind") == "build"),
        "n_loop_failures": sum(1 for e in failures if e.get("kind") == "loop"),
    }


def _failure_history_entry(report: dict[str, Any]) -> dict[str, Any]:
    """The one-line version, kept across iterations.

    Compact because it accumulates for the life of the run and lives in state:
    enough to recognise the same failure recurring, not enough to re-diagnose it.
    """
    build_errors = [
        str(entry.get("error") or "")
        for entry in report.get("failures") or []
        if entry.get("kind") == "build" and entry.get("error")
    ]
    return {
        "iteration": report.get("iteration"),
        "signature": report.get("signature"),
        "missing_gates": report.get("missing_gates"),
        # Carried into the history because `_repeat_warning` and the run summary
        # both read it: "the same SIGNATURE recurred" means something different
        # when both occurrences were transport failures.
        "classification": report.get("classification"),
        "gates_failed": report.get("gates_failed"),
        "gates_never_ran": report.get("gates_never_ran"),
        "n_transport_failures": report.get("n_transport_failures", 0),
        "pass_rate": report.get("pass_rate"),
        "retries_used": report.get("retries_used"),
        "error": build_errors[0] if build_errors else report.get("last_error") or "",
    }


def _landing_call(state: DeveloperState) -> str:
    """The gate tool to run NEXT when the turn budget is nearly spent, or "".

    Ordered `compile_check` -> `run_tests` -> `sql_exec` because that is the
    order in which each is cheapest to satisfy: there is no point running the
    suite against code that does not parse, and no point applying DDL against a
    suite that is red.
    """
    for gate, tool in GATE_TOOLS.items():
        if not state.get(gate):
            return tool
    return ""


def _turn_budget_block(state: DeveloperState, turns_used: int) -> str:
    """How much of the episode is left, and -- near the end -- what to do with it.

    WHY THIS EXISTS. The turn ceiling used to be a cliff the model could not
    see. The status block reported `retry=0/5` and said nothing at all about
    turns, so an episode could arrive at turn 59 with every file written, its
    tests green, and no idea it was one turn from being cut off with nothing
    scored.

    run-b3275eb7e373 lost two of five iterations exactly there. Iteration 5
    spent 60 of its 69 steps on `read_file`, had `run_tests` green at
    pass_rate 1.0 and two source files written, and never called
    `compile_check` or `sql_exec` -- it was two tool calls from a green build
    when the ceiling took it. Iteration 3 was the same shape at 53 reads of 70
    steps with all three files written and no gate ever run. Both were
    classified `inconclusive`, which is accurate and was cold comfort.

    An episode that ends without running its gates scores NOTHING -- the build
    is not evaluated, the Judge never sees it, and every byte the Developer
    wrote is discarded. So the cheapest useful intervention is to say how many
    turns remain and, once that number is small, to say plainly that reading is
    now the wrong move.

    Deliberately NOT a bigger ceiling. Those two episodes were 76% and 87%
    `read_file`: they were over-reading, not under-working, and more budget is
    more room to over-read.
    """
    ceiling = _max_turns()
    remaining = max(0, ceiling - turns_used)
    line = f"turn={turns_used}/{ceiling} ({remaining} left)"
    if remaining > int(config.DEVELOPER_LANDING_TURNS):
        return line
    tool = _landing_call(state)
    if not tool:
        return line + "\n!! LAND IT: all three mandatory gates are green -- call `finish` NOW."
    ungated = [f"`{GATE_TOOLS[name]}`" for name in GATE_TOOLS if not state.get(name)]
    return (
        line
        + "\n!! TURN BUDGET NEARLY SPENT. STOP READING AND STOP EDITING."
        + f"\n!! Call {tool} NOW, then " + ", then ".join(t for t in ungated if t != f"`{tool}`")
        + (", then `finish`." if len(ungated) > 1 else " `finish`.")
        + "\n!! An episode that ends without running its gates is scored as a FAILED BUILD:"
        + "\n!! it is never evaluated, and every edit you have already made is discarded."
        + "\n!! Whatever is half-finished, land what works. A green build of less is worth"
        + "\n!! more than a perfect design that never ran."
    )


# The benchmark-specific half of the Developer's instructions: which tools are
# gates HERE, and which fields are THIS evaluation's answer key. The persona
# states the rules ("every gate the task names", "never read the evaluation's
# label fields"); the names belong to the round, so pointing this loop at
# another target means changing the task and not the agent.
#
# tests/test_graph_paths.py checks the persona and this note TOGETHER against
# the real tool list, so a tool may move between the two but cannot go unnamed.
BENCHMARK_TOOL_NOTE = """## GATES FOR THIS BUILD
`compile_check`, `run_tests` and `sql_exec` are the gates -- all three must have
run and passed in this episode before you call `finish`. `run_linter` and
`run_sandbox_smoke_test` are advisory: run them, but a finding in either blocks
nothing.

## LABEL FIELDS YOU MUST NOT READ
`query_type`, `attack_type`, `expected_action`, `judge_spec`, `leak_targets`.
These are the scoring labels for the checkpoints this system is evaluated on.
"""


def _status_block(state: DeveloperState, box: DevToolbox, turns_used: int = 0) -> str:
    """The machine-readable status the model reacts to.

    Everything the next action depends on is in here, which is what makes the
    loop replayable: given the same status block, the same action follows.
    """
    # The suffix set must cover EVERY file the work order can ask for. A file
    # type missing here never appears in `files_present`, so the model is told
    # it still needs writing and rewrites it forever -- the loop spins until the
    # recursion limit instead of progressing.
    present = sorted(
        str(p.relative_to(box.workspace))
        for p in box.workspace.rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".sql", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml"}
        and "__pycache__" not in p.parts
        and p.name not in {"_smoke_runner.py", "_eval_runner.py"}
        and ".cordis" not in p.parts
        and ".sessions" not in p.parts
    )
    return (
        "<STATUS>\n"
        f"iteration={state.get('iteration', 1)}\n"
        f"workspace_from={state.get('workspace_from', 'empty')}\n"
        f"schema_version={_schema_version(box.workspace)}\n"
        f"retry={state.get('retry_count', 0)}/{config.MAX_DEV_RETRIES}\n"
        f"{_turn_budget_block(state, turns_used)}\n"
        f"files_present={','.join(present)}\n"
        f"compile_ok={str(bool(state.get('compile_ok'))).lower()}\n"
        f"migration_ok={str(bool(state.get('migration_ok'))).lower()}\n"
        f"tests_ok={str(bool(state.get('tests_ok'))).lower()}\n"
        f"lint_ok={str(bool(state.get('lint_ok'))).lower()}\n"
        f"smoke_ok={str(bool(state.get('smoke_ok'))).lower()}\n"
        f"last_error={_last_error_line(state.get('last_stack_trace'))}\n"
        "</STATUS>"
    )


# ======================================================================
# loop phases
# ======================================================================


def _consecutive_fallbacks(state: DeveloperState) -> int:
    """How many turns in a row ended in the no-parseable-action fallback."""
    count = 0
    for entry in reversed(state.get("scratchpad") or []):
        if (entry.get("action") or {}).get("_fallback"):
            count += 1
        else:
            break
    return count


def _consecutive_repeats(state: DeveloperState, tool: str, args: dict[str, Any]) -> int:
    """How many green, identical calls to `tool` immediately precede this one.

    Only successful prior calls count: if the last attempt failed, re-running it
    is a legitimate way to check a fix.
    """
    count = 0
    for entry in reversed(state.get("scratchpad") or []):
        action = entry.get("action") or {}
        same = str(action.get("tool") or "") == tool and (action.get("args") or {}) == args
        if same and entry.get("ok"):
            count += 1
        else:
            break
    return count


def _identical_failed_edits(state: DeveloperState, tool: str, args: dict[str, Any]) -> int:
    """How many times this exact edit has already been tried and failed.

    WHY (runs_4iter, iteration 3). `apply_patch` is not idempotent, so it is
    absent from `_IDEMPOTENT_TOOLS`, and `_consecutive_repeats` only counts
    SUCCESSFUL prior calls -- both correct on their own, and together they left
    a repeated failing edit completely uncounted. The Developer sent one search
    block that did not match the file four times, in between two variants that
    also did not match, and exhausted the retry cap having written nothing.

    Unlike the consecutive-repeat check this scans the whole episode: the run
    interleaved the identical patch with other attempts, so a strictly
    consecutive test would have seen a run of one every time.
    """
    if tool not in _EDIT_TOOLS:
        return 0
    return sum(
        1 for entry in (state.get("scratchpad") or [])
        if not entry.get("ok")
        and str((entry.get("action") or {}).get("tool") or "") == tool
        and ((entry.get("action") or {}).get("args") or {}) == args
    )


def _ungated(state: DeveloperState) -> list[str]:
    """The exit-condition flags still unset, in gate order."""
    return [name for name in ("compile_ok", "tests_ok", "migration_ok") if not state.get(name)]


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    """Accept the flat action shape as well as the documented nested one.

    The persona documents {"tool": t, "args": {...}}, but models drift to the
    flat {"tool": t, "path": ...} -- it is shorter and reads just as well. The
    cost of not accepting it is brutal and silent: `dev_act` reads `args`, finds
    nothing, and calls the tool with NO arguments, so `read_file` reports "no
    such file: None" for a file that is sitting right there. Every such turn
    also costs a retry, so three drifted actions exhaust MAX_DEV_RETRIES in
    under two minutes and the run halts with every gate false -- which is
    exactly how runs_fresh/iter_1 died.

    Reading the intended arguments off the top level is unambiguous (the tool
    schemas share no key names with the envelope), so there is no reason to
    spend a retry teaching the model a syntax we can simply accept.
    """
    if not isinstance(action, dict):
        return {}
    args = action.get("args")
    if isinstance(args, dict):
        # The mirror-image drift, seen on the tool-calling path rather than the
        # prose one: a model that has been told the envelope is {"tool", "args"}
        # sometimes repeats the envelope INSIDE the schema's arguments object,
        # arriving as args={"args": {"path": ...}}. Unwrapping is unambiguous --
        # no tool has a parameter called `args` -- and the cost of not doing it
        # is the same silent one, a tool called with nothing.
        inner = args.get("args")
        if isinstance(inner, dict) and len(args) == 1:
            log.info("normalized double-wrapped args for %r", action.get("tool"))
            return {**action, "args": inner}
        return action
    # `_fallback` is set by this module, not the model; `thought` is prose.
    stray = {
        key: value
        for key, value in action.items()
        if key not in {"tool", "args", "thought"} and not key.startswith("_")
    }
    if not stray:
        return action
    normalized = {k: v for k, v in action.items() if k not in stray}
    normalized["args"] = stray
    log.info("normalized flat action for %r: lifted %s into args",
             action.get("tool"), sorted(stray))
    return normalized


# --------------------------------------------------------------------------
# XML tool-call recovery
# --------------------------------------------------------------------------
#
# WHY THIS EXISTS (regression, runs_multi/run-280411c75b99): DEVELOPER_PROFILE
# mounts no harness-native tools, on purpose -- see the long note in
# harness/profiles.py. But DEVELOPER_MODEL is a tool-calling model, and a
# tool-calling model handed no tool schema does not fall back to the prose the
# persona asks for. It falls back to emitting its NATIVE tool-call syntax as
# plain text. Across four iterations that run produced 27 of these, parsed zero
# of them, and wrote zero bytes to the workspace: every iteration evaluated
# byte-identical code, and MGS never moved off 0.0.
#
# The fix is the same judgement call `_normalize_action` already makes below:
# the intent is unambiguous, so accept the syntax rather than spend a retry
# teaching the model one it is not going to learn mid-run. Five dialects showed
# up in that run's scratchpads, so all five are handled:
#
#   A  <tool_calls><invoke name="read_file">
#          <parameter name="path">store.py</parameter></invoke></tool_calls>
#   B  <tool_calls><invoke name="run_tests"><args>{"path": "tests"}</args></invoke>
#   C  <read_file><path>store.py</path></read_file>
#   D  <tool>read_file</tool><path>store.py</path>
#   E  the same shapes with DeepSeek's <|DSML|> sentinel in place of the tag
#      name, which is what a corrupted function-call decode looks like on
#      the wire.
#
# Tag names are matched against TOOL_NAMES rather than accepted on sight, so
# the <thought>, <thinking> and <retry> wrappers the model also emits are never
# mistaken for a tool call. Only the FIRST tool is taken: this loop executes one
# action per turn, and the <STATUS> block re-anchors the model on the next one.
_XML_INVOKE = re.compile(
    r"""<invoke\s+name=["']([^"']+)["']\s*>(.*?)(?:</invoke>|\Z)""",
    re.DOTALL | re.IGNORECASE,
)
_XML_PARAM = re.compile(
    r"""<parameter\s+name=["']([^"']+)["']\s*>(.*?)(?:</parameter>|\Z)""",
    re.DOTALL | re.IGNORECASE,
)
# DeepSeek sentinel form: <|DSML| name="read_file">...</|DSML|>. The sentinel
# replaces BOTH the invoke and the parameter tag, so nesting is ambiguous; the
# name= attribute is the only reliable signal and is read positionally.
_XML_DSML = re.compile(
    r"""<[|\uff5c]DSML[|\uff5c]\s+name=["']([^"']+)["']\s*>(.*?)(?:</[|\uff5c]DSML[|\uff5c]>|\Z)""",
    re.DOTALL | re.IGNORECASE,
)
_XML_FLAT_TOOL = re.compile(r"<tool>\s*([A-Za-z_][A-Za-z0-9_]*)\s*</tool>", re.IGNORECASE)
# Bare-element form, tolerating the unclosed opening tag the model also emits
# (`<list_dir\n<path>.</path>`) -- hence `[>\n]` rather than a required `>`.
_XML_BARE = re.compile(
    r"<(" + "|".join(sorted(TOOL_NAMES)) + r")[>\n]", re.IGNORECASE
)
_XML_ARG = re.compile(
    r"<([A-Za-z_][A-Za-z0-9_]*)\s*>(.*?)</\1\s*>", re.DOTALL
)


def _coerce_arg(raw: str) -> Any:
    """Unwrap an argument body, parsing it as JSON when it plainly is JSON.

    `write_file` content is arbitrary text and must survive verbatim, so a
    failed parse returns the stripped string rather than raising.
    """
    # A response cut off mid-tag leaves a partial close ("...**/*.py</parame")
    # glued to the value. Drop it: a truncated tag is never part of the content.
    text = re.sub(r"</[A-Za-z_|\uff5c][^>]*$", "", raw).strip()
    if text[:1] in "{[" or text in {"true", "false", "null"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def _args_from_body(body: str) -> dict[str, Any]:
    """Collect `<parameter name=k>v</parameter>`, `<args>{...}</args>` and `<k>v</k>`."""
    args: dict[str, Any] = {}
    for key, value in _XML_PARAM.findall(body):
        args.setdefault(key, _coerce_arg(value))
    if args:
        return args
    # A sentinel-corrupted decode nests its parameters inside the invoke body
    # rather than beside it, so shape A can still be carrying shape E's args.
    known = {name.lower() for name in TOOL_NAMES}
    for key, value in _XML_DSML.findall(body):
        # First wins: an unclosed <invoke> runs its body to end-of-text and so
        # sweeps up the arguments of every LATER call too. This turn executes
        # one action, and it is the first one the model asked for.
        if key.strip().lower() not in known:
            args.setdefault(key.strip(), _coerce_arg(value))
    if args:
        return args
    # Shape B: a single <args> element holding a JSON object.
    for key, value in _XML_ARG.findall(body):
        if key.lower() == "args":
            parsed = _coerce_arg(value)
            return parsed if isinstance(parsed, dict) else {}
        args[key] = _coerce_arg(value)
    return args


def parse_xml_action(text: str) -> dict[str, Any]:
    """Recover one `{"tool": ..., "args": {...}}` action from XML tool-call prose.

    Returns `{}` when the text holds no recognisable call, which keeps the
    caller's existing prose-fallback path intact.
    """
    if not text or "<" not in text:
        return {}
    known = {name.lower(): name for name in TOOL_NAMES}

    # Shapes A/B: <invoke name="tool">.
    for raw_name, body in _XML_INVOKE.findall(text):
        tool = known.get(raw_name.strip().lower())
        if tool:
            return {"tool": tool, "args": _args_from_body(body)}

    # Shape E: DeepSeek sentinel. The first match naming a known tool is the
    # call; the rest of the matches are its parameters.
    dsml = _XML_DSML.findall(text)
    for index, (raw_name, _body) in enumerate(dsml):
        tool = known.get(raw_name.strip().lower())
        if not tool:
            continue
        # The lazy body of the tool element swallows its own nested parameters,
        # so look inside it before falling back to the following siblings.
        args = {
            key.strip(): _coerce_arg(value)
            for key, value in _XML_DSML.findall(_body)
            if key.strip().lower() not in known
        } or {
            key.strip(): _coerce_arg(value)
            for key, value in dsml[index + 1 :]
            if key.strip().lower() not in known
        }
        return {"tool": tool, "args": args}

    # Shape D: <tool>name</tool> with sibling argument elements.
    flat = _XML_FLAT_TOOL.search(text)
    if flat:
        tool = known.get(flat.group(1).strip().lower())
        if tool:
            args = {
                key: _coerce_arg(value)
                for key, value in _XML_ARG.findall(text[flat.end():])
                if key.lower() not in known
            }
            return {"tool": tool, "args": args}

    # Shape C: a bare <tool_name> element, possibly never closed.
    bare = _XML_BARE.search(text)
    if bare:
        tool = known[bare.group(1).lower()]
        args = {
            key: _coerce_arg(value)
            for key, value in _XML_ARG.findall(text[bare.end():])
            if key.lower() not in known
        }
        return {"tool": tool, "args": args}
    return {}


def _parse_action(text: str) -> dict[str, Any]:
    """One reply -> one action, trying every syntax the model actually uses."""
    action = _normalize_action(extract_json_block(text) or {})
    if "tool" in action:
        return action
    # The model emitted its native tool-call syntax instead of the fenced
    # JSON the persona asks for. The intent is unambiguous, so honour it.
    recovered = parse_xml_action(text)
    if recovered:
        log.info("recovered XML-shaped action for %r: args=%s",
                 recovered["tool"], sorted(recovered.get("args") or {}))
        return _normalize_action(recovered)
    return {}


# The shortest slice worth spending on a sample. Below this a request is more
# likely to be killed mid-answer than to return one.
_MIN_SAMPLE_S = 45


def _add_usage(left: dict[str, int], right: dict[str, Any] | None) -> dict[str, int]:
    """Sum two usage dicts, ignoring anything non-numeric."""
    total = dict(left)
    for key, value in (right or {}).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[key] = int(total.get(key, 0)) + int(value)
    return total


def _sample_timeout(attempt: int, resamples: int) -> int:
    """Split the think budget across the samples instead of multiplying it.

    Resampling must not turn a 300s think turn into a 1200s one. The whole turn
    still costs at most DSH_DEVELOPER_THINK_TIMEOUT_S, which is the contract the
    rest of the loop was built around -- MAX_DEV_RETRIES only bounds the episode
    if a single turn is bounded first.

    The first sample gets HALF the budget because it is the one most likely to
    be a legitimate long answer: a `write_file` action embeds an entire source
    file in its arguments, and the template's `store.py` alone is 22KB. The
    resamples split the remaining half, which is ample for an ordinary action
    (the real ones observed are 85-280 characters) and is deliberately too short
    to let a repetition loop run on -- cutting it off is the entire point. This
    slice is now the ONLY thing that ends such a loop: DEVELOPER_MAX_TOKENS is
    uncapped by default, so there is no token ceiling behind it to catch one.

    A cut-off sample is discarded, never used: a truncated reply cannot yield a
    parseable action, because unterminated JSON fails both `json.loads` and the
    repair pass. So the worst case of a too-short slice is one more resample,
    never a half-written file.
    """
    budget = float(config.DSH_DEVELOPER_THINK_TIMEOUT_S)
    if resamples <= 0:
        return int(budget)
    share = budget / 2.0 if attempt == 0 else budget / (2.0 * resamples)
    # A floor keeps a small budget from starving every sample into a guaranteed
    # timeout, which would be strictly worse than not resampling at all.
    return max(_MIN_SAMPLE_S, int(share))


def _actions_from(result: DSHResult) -> list[dict[str, Any]]:
    """Every action this turn asked for, native calls first.

    NATIVE CALLS ARE AUTHORITATIVE. When the model used the tool schema, its
    calls come back structurally with the `id` each answering `tool` message has
    to quote, and there is nothing to parse. `_parse_action` is what remains for
    the turns that still arrive as prose: a provider that dropped the schema, a
    model that narrated a call instead of making one, or the mock's fenced
    blocks. Both paths produce the same `{"tool", "args"}` shape, so everything
    downstream is blind to which one ran.

    A turn may legitimately ask for SEVERAL tools; the caller executes them in
    order and answers every id.
    """
    if result.tool_calls:
        return [
            _normalize_action({
                "tool": call["name"],
                "args": call["arguments"],
                "_id": call["id"],
            })
            for call in result.tool_calls
        ]
    action = _parse_action(result.text)
    return [action] if "tool" in action else []


@dataclass
class Turn:
    """One model turn: what it said, what it wants to run, what it cost.

    A dataclass rather than a bare tuple because the loop needs all three parts
    and they are not interchangeable: `result` is replayed into the
    conversation, `actions` are executed, `update` is folded into the state.
    """

    result: DSHResult
    actions: list[dict[str, Any]] = field(default_factory=list)
    update: dict[str, Any] = field(default_factory=dict)


async def _sample_turn(
    messages: list[dict[str, Any]], box: DevToolbox
) -> tuple[DSHResult, list[dict[str, Any]]]:
    """Ask for the next turn, RESAMPLING when the reply contains no action.

    WHY RESAMPLE RATHER THAN SPEND A RETRY (measured on DEVELOPER_MODEL).
    Roughly half of this model's replies to the Developer prompt degenerate into
    a repetition loop -- `<thought ... response` emitted until the token cap --
    and never reach an action. It is a sampling malfunction, not a statement
    about the work: an immediate re-ask usually succeeds, and the three observed
    shapes (a repetition loop, an empty reply, a lone space) all clear on a
    fresh draw.

    Charging that to `MAX_DEV_RETRIES` conflates two different things. That
    budget means "this design cannot be built" and routes back to the Architect
    when it runs out; retiring an episode because the decoder stuttered sends
    the Architect a redesign request for a design that was never attempted.

    Each resample nudges the temperature up. A near-greedy decode is precisely
    the regime these loops live in, so redrawing at the SAME temperature is the
    draw least likely to differ. The first attempt keeps the profile's
    temperature so ordinary turns stay as reproducible as they were.

    Handing the model a tool schema is expected to make this rarer -- a native
    tool call is a decode path that cannot end in `<thought ... response` -- but
    "rarer" is not "gone", and the mitigation costs nothing when nothing is
    wrong, so it stays.
    """
    resamples = max(0, int(config.DEV_THINK_RESAMPLES))
    # A WALL-CLOCK DEADLINE, not just a per-sample slice. The per-sample timeout
    # below is not a real bound on its own: on the http transport a timed-out
    # request is retried inside `AsyncLLMClient.chat` up to HTTP_MAX_RETRIES
    # times, so one "sample" can quietly cost several times its slice. This
    # deadline is what actually holds the turn to its budget.
    deadline = time.monotonic() + float(config.DSH_DEVELOPER_THINK_TIMEOUT_S)
    tools = tool_schemas()
    # `required` is available because of the failure this function exists for:
    # the model's prose-only turns are a decode malfunction, and forcing a tool
    # call removes the decode path that produces them. It is not the default
    # because a model that is genuinely stuck should be allowed to say so rather
    # than being made to call something -- and the resample below already
    # recovers the common case. See config.DEV_TOOL_CHOICE.
    choice = str(config.DEV_TOOL_CHOICE or "auto")
    result: DSHResult | None = None
    # Usage from EVERY sample, not just the winning one. A degenerate sample
    # runs to the 16384-token cap and is billed in full; dropping its cost
    # because its text was unusable would under-report the single most
    # expensive thing this loop does.
    spent: dict[str, int] = {}
    for attempt in range(resamples + 1):
        profile = DEVELOPER_PROFILE if attempt == 0 else _warmer(DEVELOPER_PROFILE, attempt)
        remaining = deadline - time.monotonic()
        slice_s = _sample_timeout(attempt, resamples)
        result = await agent_call(
            profile, "", box.workspace,
            timeout_s=int(min(slice_s, remaining)) if attempt else slice_s,
            messages=messages, tools=tools, tool_choice=choice,
        )
        spent = _add_usage(spent, result.usage)
        actions = _actions_from(result) if result.ok else []
        if actions:
            if attempt:
                log.info("developer: resample %d produced %s", attempt,
                         [a["tool"] for a in actions])
            result.usage = spent
            return result, actions
        if attempt >= resamples:
            break
        remaining = deadline - time.monotonic()
        if remaining < _MIN_SAMPLE_S:
            log.warning(
                "developer: no action after %d attempt(s) and the think budget is "
                "spent (%.0fs left); not resampling", attempt + 1, remaining,
            )
            break
        log.warning(
            "developer: no action in reply (attempt %d/%d, ok=%s, %d chars, "
            "%.0fs left); resampling at temperature %.2f",
            attempt + 1, resamples + 1, result.ok, len(result.text or ""),
            remaining, _warmer(DEVELOPER_PROFILE, attempt + 1).temperature,
        )
    assert result is not None  # the loop always runs at least once
    result.usage = spent
    return result, []


def _warmer(profile: DSHProfile, attempt: int) -> DSHProfile:
    """The same profile with a higher temperature, capped at 0.8.

    Frozen dataclass, so this returns a copy -- the module-level profile stays
    immutable and shared, exactly as `with_workdir` does. Note this only bites
    on the `http` transport: the dsh harness config has no temperature field
    (see DSHProfile's docstring), so on that path a resample is simply a fresh
    draw.
    """
    temperature = min(0.8, profile.temperature + 0.25 * attempt)
    return DSHProfile(**{**profile.__dict__, "temperature": temperature})


async def dev_think(
    state: DeveloperState, messages: list[dict[str, Any]]
) -> Turn:
    """Ask the model what to do next, given everything it has already done.

    `messages` is the Developer's OWN conversation, not a task string rebuilt
    from scratch: its previous tool calls and their real output are in there, so
    "the tests fail on author_id" is something it watched happen rather than
    something it is being told. The status block appended to the latest tool
    result is still the ground truth for the durable facts, because the loop
    prunes old turns (see `DeveloperSession._prune`) and `files_present` must
    survive that pruning.
    """
    box = _toolbox(state)
    result, actions = await _sample_turn(messages, box)

    if not result.ok and not actions:
        # A transport failure is not a code failure; count it as a retry so a
        # flapping endpoint cannot spin the loop forever.
        #
        # AND COUNT IT SEPARATELY. `retry_count` alone cannot distinguish "this
        # design cannot be built" from "the endpoint was down", and the failure
        # report is read by the Architect as a critique of the design. In
        # run-8cf58d33b311 iteration 2 every one of the episode's charged
        # retries came from here -- eight minutes of `http transport exceeded`
        # against OpenRouter -- and the report that reached the Architect
        # mentioned no transport error at all: `last_error` was empty,
        # `n_build_failures` was 0, and the headline said the mandatory gates
        # were unmet. `classify_failure` reads this counter to stop that.
        log.warning("developer: transport failure charged to the episode: %s", result.error)
        return Turn(
            result=result,
            actions=[{"tool": "compile_check", "args": {}}],
            update={
                "thought": f"(transport error: {result.error})",
                "retry_count": int(state.get("retry_count", 0)) + 1,
                "transport_failures": int(state.get("transport_failures", 0)) + 1,
                "transport_errors": [str(result.error or "unknown transport failure")],
                "last_stack_trace": result.error or "",
            },
        )

    thought = _thought_from(result)

    if not actions:
        # EVERY sample in this turn -- the first plus DEV_THINK_RESAMPLES more --
        # came back without an action. Rather than guessing, re-anchor on the
        # cheapest informative tool; the status block will push it forward.
        #
        # But repeats are NOT free. A turn that gets here has already spent
        # several completions, and runs/iter_1 spent its first eleven
        # consecutive turns in this branch. So the second such turn in a row
        # costs a retry, which bounds the episode at MAX_DEV_RETRIES instead of
        # letting it spin to the recursion limit.
        prior = _consecutive_fallbacks(state)
        # Log the REPLY, not just a counter (regression, runs_smoke/iter_1).
        # The bare count cannot distinguish the two failures that produce it --
        # "the model said nothing actionable" and "the model chose a tool and
        # the transport dropped it" -- and telling them apart from the outside
        # cost a full session-log archaeology dig. The raw text and the block
        # types present in the response answer it in one line.
        block_types = sorted({
            str(block.get("type"))
            for event in (result.events or [])
            if isinstance(event, dict) and event.get("type") == "assistant/message"
            for block in (((event.get("data") or {}).get("message") or {}).get("content") or [])
            if isinstance(block, dict)
        })
        log.warning(
            "developer produced no parseable action (consecutive=%d); "
            "defaulting to compile_check | finish=%s blocks=%s reply=%r",
            prior + 1, result.finish_reason, block_types or "(none)", result.text[:1500],
        )
        update: dict[str, Any] = {"thought": thought, "tokens_used": result.usage}
        if prior:
            update["retry_count"] = int(state.get("retry_count", 0)) + 1
            update["last_stack_trace"] = (
                "the last two replies contained no tool call. Call one of the ten "
                "tools you were given -- do not describe the call in prose."
            )
        return Turn(
            result=result,
            actions=[{"tool": "compile_check", "args": {}, "_fallback": True}],
            update=update,
        )
    return Turn(
        result=result,
        actions=actions,
        update={"thought": thought, "tokens_used": result.usage},
    )


def _thought_from(result: DSHResult) -> str:
    """The reasoning the model showed, for the scratchpad and the failure report.

    A turn that is pure tool calls has no prose at all, which is normal and not
    a problem -- the action IS the statement. Naming the tools keeps the
    scratchpad readable rather than leaving a run of blank thoughts.
    """
    text = result.content or result.text or ""
    match = re.search(r"Thought:\s*(.+?)(?:\n\n|```)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    if text.strip():
        return text.strip()[:400]
    if result.tool_calls:
        return "(no prose) calling " + ", ".join(c["name"] for c in result.tool_calls)
    return ""


async def dev_act(
    state: DeveloperState, action: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Execute one tool the Developer called, for real.

    `action` defaults to `state["action"]` -- a turn may ask for several tools,
    and the loop walks them one at a time rather than the state holding exactly
    one. Everything the tools do is unchanged: DevToolbox is still the only
    write surface in the whole graph, and it is still workspace-scoped.
    """
    box = _toolbox(state)
    action = action if action is not None else (state.get("action") or {})
    tool = str(action.get("tool") or "compile_check")
    raw_args = action.get("args")
    args = raw_args if isinstance(raw_args, dict) else {}

    # An idempotent tool re-run with identical arguments and nothing written in
    # between cannot report anything new, but the harness call that chose it was
    # not free. Replay the steering fact the model is actually missing -- which
    # gates remain unset and which tool sets each -- instead of the same green
    # observation it has already seen.
    repeats = _consecutive_repeats(state, tool, args) if tool in _IDEMPOTENT_TOOLS else 0
    if repeats:
        ungated = _ungated(state)
        if ungated:
            steer = "Still unset: " + ", ".join(
                f"{name} (set by `{GATE_TOOLS[name]}`)" for name in ungated
            ) + ". Call one of those tools next."
        else:
            steer = "All three gates are set -- call `finish` next."
        return {"observation": {
            "tool": tool,
            # The first repeat is free steering; a second in a row is the loop
            # failing to progress, and costs a retry like any other dead end.
            "ok": repeats < 2,
            "text": (
                f"skipped: `{tool}` already ran with these arguments and nothing has "
                f"been written since, so the result is unchanged. {steer}"
            ),
            "data": {"skipped_duplicate": True, "repeats": repeats},
            "stack_trace": None,
            "signature": None,
        }}

    # An edit that has already been rejected verbatim will be rejected again --
    # the file has not changed, and neither have the arguments. Spend the turn
    # telling the model that, rather than replaying the same refusal a third and
    # fourth time while the retry budget drains.
    if _identical_failed_edits(state, tool, args) >= 2:
        return {"observation": {
            "tool": tool,
            "ok": False,
            "text": (
                f"FAILED {tool}: this exact call has already failed twice in this episode "
                f"and the file has not changed since, so it cannot succeed now. Stop "
                f"re-sending it.\n"
                f'Read the region first -- {{"tool": "read_file", "args": '
                f'{{"path": "{args.get("path", "memory_system/store.py")}", "offset": <line>}}}} '
                f"-- and copy the search block out of what you actually see. The previous "
                f"failure printed the closest matching lines; use those."
            ),
            "data": {"repeated_failed_edit": True},
            "stack_trace": None,
            "signature": None,
        }}

    result: ToolResult = await box.call(tool, args)
    return {"observation": {
        "tool": result.tool, "ok": result.ok, "text": result.observation(),
        "data": result.data, "stack_trace": result.stack_trace,
        "signature": result.signature(), "redirect": result.redirect,
    }}


async def dev_observe(state: DeveloperState) -> dict[str, Any]:
    """Fold the observation into the exit-condition evidence."""
    observation = state.get("observation") or {}
    tool = str(observation.get("tool") or "")
    ok = bool(observation.get("ok"))
    update: dict[str, Any] = {
        "scratchpad": [{
            "thought": state.get("thought", ""),
            "action": state.get("action", {}),
            "observation": observation.get("text", ""),
            "ok": ok,
        }],
    }

    # GATE PROVENANCE, recorded before the gate flags themselves. This is what
    # lets the failure report say "run_tests never ran" instead of "tests_ok is
    # false", which are the same three characters of state and completely
    # different news for the Architect. Only a gate tool that actually completed
    # is recorded, so a turn the transport ate leaves no trace here -- which is
    # exactly the signal `classify_failure` keys on.
    gate_for_tool = {tool_name: gate for gate, tool_name in GATE_TOOLS.items()}
    if tool in gate_for_tool:
        attempts = {name: dict(entry) for name, entry
                    in (state.get("gate_attempts") or {}).items()}
        entry = attempts.setdefault(gate_for_tool[tool], {"ran": 0, "ok": False})
        entry["ran"] = int(entry.get("ran", 0)) + 1
        entry["ok"] = ok
        update["gate_attempts"] = attempts

    if tool == "compile_check":
        update["compile_ok"] = ok
    elif tool == "run_tests":
        data = observation.get("data") or {}
        # A narrow run reports on one file, not the suite, so it must not decide
        # the gate either way -- see the note in DevToolbox._t_run_tests. Absent
        # key means a caller that predates the distinction: treat it as a suite
        # run rather than silently refusing to gate.
        if data.get("full_suite", True):
            update["tests_ok"] = ok
        update["pass_rate"] = float(data.get("pass_rate", 0.0))
    elif tool == "sql_exec":
        update["migration_ok"] = ok
    elif tool == "run_linter":
        # Advisory: mark it attempted either way so the loop moves on.
        update["lint_ok"] = True
    elif tool == "run_sandbox_smoke_test":
        update["smoke_ok"] = ok
    elif tool == "finish":
        update["done"] = True

    if not ok and tool == "run_linter":
        # Advisory, and said so in three places already: this module's header,
        # the Developer's system prompt ("a finding in either blocks nothing")
        # and `lint_ok` above, which is set True whatever the linter returns.
        # Charging a style finding against MAX_DEV_RETRIES contradicts all
        # three and retires an episode for a reason that has nothing to do with
        # whether the design can be built. The finding still reaches the model
        # as the observation, which is the whole reason for running it.
        #
        # This was latent until now: `_ruff_command` (nodes/dev_tools.py) found
        # no ruff on PATH under `.venv/bin/python main.py`, so the gate always
        # took the fallback path and always came back green.
        log.info("developer: lint findings are advisory (not charged against retries)")
        update["last_stack_trace"] = None
    elif not ok and observation.get("redirect"):
        # A tool that refused an action and named the right one is not evidence
        # about whether the design can be built, so it must not consume the
        # budget that bounds exactly that question -- the same distinction
        # DEV_THINK_RESAMPLES draws for a decoder stutter. Without this, the
        # write_file rewrite guard could end an episode it exists to protect:
        # five insistent write_file calls would exhaust MAX_DEV_RETRIES having
        # changed nothing. The steer still reaches the model as the observation.
        log.info("developer: %s redirected (not charged against retries)", tool)
        update["last_stack_trace"] = None
    elif not ok:
        # Any other failed observation costs a retry. Writing code is cheap; the
        # cap exists to stop the loop grinding on a design that cannot be built.
        update["retry_count"] = int(state.get("retry_count", 0)) + 1
        update["last_stack_trace"] = observation.get("stack_trace") or observation.get("text", "")
        update["failure_signature"] = observation.get("signature")
        # Recorded HERE rather than reconstructed from the scratchpad at the
        # end, because this is the only point where the structured half of the
        # observation still exists: `scratchpad` keeps the rendered text, not
        # `signature`, `stack_trace` or `data`.
        update["failures"] = [_failure_entry(state, observation)]
    else:
        # A green step clears the stale trace so the next Thought is not
        # reasoning about a problem that has already been fixed.
        update["last_stack_trace"] = None

    # A `finish` claimed before the gates are green is refused: the model does
    # not get to declare victory, the tools do.
    compile_ok = update.get("compile_ok", state.get("compile_ok", False))
    tests_ok = update.get("tests_ok", state.get("tests_ok", False))
    migration_ok = update.get("migration_ok", state.get("migration_ok", False))
    if update.get("done") and not (compile_ok and tests_ok and migration_ok):
        update["done"] = False
        update["retry_count"] = int(state.get("retry_count", 0)) + 1
        missing = [
            name for name, value in (
                ("compile_ok", compile_ok),
                ("tests_ok", tests_ok),
                ("migration_ok", migration_ok),
            ) if not value
        ]
        why = "\n".join(
            f"  - {name} is false because `{GATE_TOOLS[name]}` has not completed "
            f"successfully in this episode."
            for name in missing
        )
        update["scratchpad"] = update["scratchpad"] + [{
            "thought": "(gate) finish rejected",
            "action": {"tool": "gate"},
            "observation": (
                f"finish rejected: compile_ok={compile_ok} tests_ok={tests_ok} "
                f"migration_ok={migration_ok}; all three must be true.\n"
                f"{why}\n"
                "A gate is set only by running its tool here, in this episode. "
                "Describing the result in prose, or having run it in a previous "
                "iteration, does not set it. Call "
                + ", ".join(f"`{GATE_TOOLS[name]}`" for name in missing)
                + " -- one tool per step -- then finish."
            ),
            "ok": False,
        }]
        # A rejected `finish` costs a retry like any other dead end, so it is
        # part of the evidence too -- and it is the one failure the Architect
        # must NOT read as a broken design. It means the Developer declared
        # victory without running the tools, which is a fact about the episode.
        update["failures"] = list(update.get("failures") or []) + [{
            "step": len(state.get("scratchpad") or []) + len(update["scratchpad"]),
            "tool": "finish",
            "args": {},
            "kind": "loop",
            "signature": None,
            "error": f"finish rejected: {', '.join(missing)} not set",
            "observation": update["scratchpad"][-1]["observation"],
        }]

    # Gates green, but nothing built. The same refusal as the `gates_green` exit
    # in `_stop_reason`, at the other door: an episode that inherited a working
    # workspace can satisfy every gate without touching a file, and the loop is
    # only self-improving if each iteration changes the code it inherited.
    elif update.get("done") and not _built_anything(state):
        update["done"] = False
        update["retry_count"] = int(state.get("retry_count", 0)) + 1
        update["scratchpad"] = update["scratchpad"] + [{
            "thought": "(gate) finish rejected",
            "action": {"tool": "gate"},
            "observation": (
                "finish rejected: the gates are green but this episode has not changed "
                "any .py or .sql file. The workspace was seeded from the previous "
                "iteration, so green gates here only confirm that the PREVIOUS "
                "iteration built -- they are not evidence about the work order you "
                "were given.\n"
                "Implement the work order, then re-run the gate tools and finish. "
                "If you believe the change is already present, say which file and "
                "which lines implement it and change them; scratch files and dumped "
                "text do not count.\n"
                "A work order that describes code already in the workspace is a fact "
                "about the DESIGN, and it is not permission to finish: an episode "
                "that ends here produces an iteration byte-identical to the last "
                "one. Pick the step in the work order that is NOT literally in the "
                "code and implement that."
            ),
            "ok": False,
        }]
        update["failures"] = list(update.get("failures") or []) + [{
            "step": len(state.get("scratchpad") or []) + len(update["scratchpad"]),
            "tool": "finish",
            "args": {},
            "kind": "loop",
            "signature": None,
            "error": "finish rejected: no source file changed in this episode",
            "observation": update["scratchpad"][-1]["observation"],
        }]
    return update


def _stop_reason(state: DeveloperState) -> str:
    """Why the episode should end, or "" to keep going.

    The successor to the sub-graph's router. It is a plain predicate on the
    state, so the loop reads as a loop and the policy is still testable without
    running a model.
    """
    if state.get("done"):
        return "finish"
    if int(state.get("retry_count", 0)) >= config.MAX_DEV_RETRIES:
        return "exhausted"
    if (
        state.get("compile_ok")
        and state.get("tests_ok")
        and state.get("migration_ok")
        and state.get("smoke_ok")
    ):
        # All gates green even without an explicit `finish`: stop rather than
        # spend turns waiting for the model to notice.
        #
        # ...unless nothing was built. Green gates on an inherited workspace say
        # the PREVIOUS iteration compiled, which is not this episode's news. See
        # the note on DevToolbox.changed_files: runs_4iter iteration 2 exited
        # here after seven turns and zero bytes written, and the Judge went on to
        # score code identical to iteration 1's.
        if not _built_anything(state):
            return ""
        return "gates_green"
    return ""


def _built_anything(state: DeveloperState) -> bool:
    """Has this episode produced work since it was handed the workspace?

    Fails OPEN when there is no toolbox to ask. Both callers are reached only
    after `compile_check`, `run_tests` and `sql_exec` have run, and all three go
    through DevToolbox -- so in the real graph a missing toolbox is not "an
    episode that built nothing", it is a state that cannot occur. Blocking on it
    would mean inventing a failure out of an unobservable, which is the one
    thing a guard against false success must not do.
    """
    box = _TOOLBOXES.get(str(state.get("workspace") or ""))
    return True if box is None else box.built_anything()


# ======================================================================
# the loop
# ======================================================================


# How many recent turns stay in the conversation verbatim. Old turns are dropped
# WHOLE -- an assistant message and every `tool` message answering it go together
# or the transcript is malformed -- and what they carried that still matters is
# in the status block, which is re-stated after every batch. That was already
# the policy when the loop re-rendered a six-entry scratchpad each turn; the
# difference is that these are the model's real messages rather than a summary
# of them.
#
# It is a real bound, not hygiene: a `write_file` call embeds a whole source
# file in its arguments and the template's store.py alone is 22KB, so a
# twenty-turn episode that kept everything would spend most of its context
# re-reading files it has already written.
_KEEP_TURNS = 8

# A per-episode ceiling on model turns, replacing the sub-graph's
# `recursion_limit`. MAX_DEV_RETRIES only counts FAILED observations, so a loop
# making slow green progress -- read, read, list, read -- is bounded by nothing
# else except the wall clock.
#
# READ AT CALL TIME, not bound here, so `config.DEVELOPER_MAX_TURNS` can be
# overridden per run and by tests -- the same rule every other threshold in this
# module follows. The module-level name is kept as a thin accessor because the
# failure report quotes the ceiling in its prose.
def _max_turns() -> int:
    return int(config.DEVELOPER_MAX_TURNS)


class DeveloperSession:
    """One Developer episode: its conversation, its toolbox, its state.

    The whole loop is `run()`. Everything else here is the bookkeeping that lets
    the model see its own work: `_record` puts each turn and each real tool
    result into the conversation, and `_prune` keeps that conversation from
    growing past the point where it is useful.
    """

    def __init__(self, state: DeveloperState) -> None:
        self.state: DeveloperState = dict(state)
        self.box = _toolbox(self.state)
        self.turns = 0
        self.deadline = time.monotonic() + float(config.DSH_DEVELOPER_TIMEOUT_S)
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": DEVELOPER_PROFILE.system_prompt},
            {"role": "user", "content": self._work_order()},
        ]

    # ------------------------------------------------------------------
    # the loop
    # ------------------------------------------------------------------

    async def run(self) -> DeveloperState:
        while True:
            stop = self._should_stop()
            if stop:
                log.info("developer: episode ends (%s) after %d turn(s), %d retry/ies",
                         stop, self.turns, int(self.state.get("retry_count", 0)))
                self.state["halt_reason"] = None if stop in {"finish", "gates_green"} else stop
                return self.state

            self.turns += 1
            turn = await dev_think(self.state, self.messages)
            self._apply(turn.update)
            executed = await self._execute(turn.actions)
            self._record(turn, executed)
            self._prune()

    async def _execute(self, actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run this turn's tools in order, folding each result as it lands.

        Each result is observed BEFORE the next tool runs, because that is what
        the gates and the retry budget are counted from: a turn that calls
        `run_tests` twice must see the first result charged before the second is
        allowed to be a duplicate. If the episode ends part-way through a batch,
        the remaining calls are still answered -- every `tool_call_id` needs a
        reply for the transcript to be well formed, and an unexplained gap in
        `developer_scratchpad.json` is worse than a one-line note.
        """
        executed: list[dict[str, Any]] = []
        for action in actions:
            if executed and self._should_stop():
                executed.append({
                    "action": action,
                    "observation": {
                        "tool": str(action.get("tool") or ""),
                        "ok": False,
                        "text": "not executed: the episode ended earlier in this turn.",
                        "data": {}, "stack_trace": None, "signature": None,
                    },
                    "skipped": True,
                })
                continue
            self.state["action"] = action
            self._apply(await dev_act(self.state, action))
            self._apply(await dev_observe(self.state))
            executed.append({"action": action, "observation": self.state.get("observation") or {}})
        return executed

    def _should_stop(self) -> str:
        reason = _stop_reason(self.state)
        if reason:
            return reason
        # The two outer bounds. Both are reported as exhaustion, because that is
        # what they are from the Architect's point of view: the Developer did not
        # produce a build, and the report says which limit ended it.
        if time.monotonic() >= self.deadline:
            log.warning("developer: episode wall clock (%.0fs) expired",
                        float(config.DSH_DEVELOPER_TIMEOUT_S))
            return "timeout"
        if self.turns >= _max_turns():
            log.warning("developer: turn ceiling (%d) reached", _max_turns())
            return "turn_cap"
        return ""

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def _apply(self, update: dict[str, Any]) -> None:
        """Fold one phase's update into the episode state.

        The reducers the sub-graph's `DeveloperState` declared, applied here
        instead: `scratchpad` and `failures` accumulate, `tokens_used` sums, and
        everything else is last-write-wins. They are spelled out rather than
        inherited because the update dicts these phases return are unchanged --
        `dev_observe` still returns `{"scratchpad": [entry]}` meaning "append",
        and reading that as "replace" would silently throw the episode away.
        """
        for key, value in (update or {}).items():
            if key in {"scratchpad", "failures", "transport_errors"}:
                self.state[key] = list(self.state.get(key) or []) + list(value or [])
            elif key == "tokens_used":
                self.state[key] = merge_dicts(self.state.get(key) or {}, value or {})
            else:
                self.state[key] = value

    # ------------------------------------------------------------------
    # the conversation
    # ------------------------------------------------------------------

    def _work_order(self) -> str:
        # The benchmark-specific half of the Developer's instructions: which
        # tools are gates here, and which fields are this evaluation's answer
        # key. The persona states the RULE ("every gate the task names", "never
        # read the evaluation's label fields"); the names belong to the round.
        return f"""{BENCHMARK_TOOL_NOTE}
## WORK ORDER FROM THE ARCHITECT
{self.state.get('instructions', '')}

## MIGRATION TO APPLY
```sql
{self.state.get('migration_sql', '') or '(none)'}
```

{_status_block(self.state, self.box, self.turns)}

Start working. Call the tools you need; you will see each result before you
choose the next one."""

    def _record(self, turn: Turn, executed: list[dict[str, Any]]) -> None:
        """Put the turn and its real results back into the model's own context.

        This is the whole point of the change. The assistant message is replayed
        with its tool calls intact -- `result.content`, not `result.text`, so a
        turn is never handed a prose transcript of a call it also really made --
        and each tool's genuine output comes back as a `tool` message quoting
        the id it answers.

        The prose path is the fallback. When a reply carried no native call the
        action was recovered from text, there is no id to answer, and the
        observation goes back as a `user` message instead. Mixing the two shapes
        in one transcript is fine; quoting an id that was never issued is not.
        """
        native = [item for item in executed if (item["action"].get("_id"))]
        if native:
            self.messages.append({
                "role": "assistant",
                "content": turn.result.content or "",
                "tool_calls": [
                    {
                        "id": str(item["action"]["_id"]),
                        "type": "function",
                        "function": {
                            "name": str(item["action"].get("tool") or ""),
                            "arguments": json.dumps(item["action"].get("args") or {}),
                        },
                    }
                    for item in native
                ],
            })
        else:
            self.messages.append({
                "role": "assistant",
                "content": turn.result.content or turn.result.text or "(no reply)",
            })

        status = _status_block(self.state, self.box, self.turns)
        for index, item in enumerate(executed):
            body = str((item["observation"] or {}).get("text") or "")
            # The status block rides on the LAST result of the batch only: it is
            # a snapshot of the state after everything in this turn ran, and
            # repeating a stale copy of it beside each earlier result would give
            # the model several disagreeing answers to the same question.
            if index == len(executed) - 1:
                body = f"{body}\n\n{status}"
            call_id = item["action"].get("_id")
            if call_id:
                self.messages.append({
                    "role": "tool", "tool_call_id": str(call_id), "content": body,
                })
            else:
                self.messages.append({"role": "user", "content": body})

        if not executed:
            # A turn that asked for nothing still has to be answered, or the
            # next request replays an assistant message with no reply after it
            # and the model simply continues its own sentence.
            self.messages.append({"role": "user", "content": status})

    def _prune(self) -> None:
        """Drop the oldest whole turns, keeping the system prompt and work order.

        A turn is an assistant message plus every message answering it, and the
        boundary is the next `assistant` role. Cutting anywhere else strands a
        `tool` message whose call is gone, which is not a degraded transcript --
        it is a 400 from the provider on the very next request.
        """
        boundaries = [
            index for index, message in enumerate(self.messages)
            if index >= 2 and message.get("role") == "assistant"
        ]
        if len(boundaries) <= _KEEP_TURNS:
            return
        cut = boundaries[-_KEEP_TURNS]
        dropped = cut - 2
        self.messages = self.messages[:2] + self.messages[cut:]
        log.debug("developer: pruned %d message(s) from the conversation", dropped)


# ======================================================================
# macro-graph node wrapper
# ======================================================================


# Files and directories that must never be carried between iterations: they are
# per-run scratch belonging to the previous iteration's tooling, not source.
_NON_SOURCE = {"__pycache__", ".sessions", ".cordis", ".pytest_cache", ".ruff_cache"}
_SCAFFOLDS = {"_smoke_runner.py", "_eval_runner.py"}


def prepare_workspace(
    iteration: int, *, parent: int | None = None, reason: str = scoreboard.LINEAGE_CHAMPION
) -> tuple[Path, str]:
    """Materialize the workspace the Developer will edit, and say where it came from.

    THE LINEAGE IS THE POINT, AND IT IS THE BEST CODE, NOT THE LAST CODE.
    Iteration N must start from a workspace that has already been measured, or
    the loop is not self-improving -- it is N independent attempts, and the
    Critic's advice from round 1 has nothing to attach to in round 2.

      iteration 1  <- templates/ (a known-good baseline), or empty if
                      SEED_FROM_TEMPLATE is false
      iteration N  <- a copy of the CHAMPION's workspace: the highest-scoring
                      iteration judged so far, which is iteration N-1 whenever
                      the loop is improving and is something earlier whenever
                      it is not

    `parent` is that champion, computed by the caller from `score_history` (see
    `scoreboard.champion_iteration`); passing `None` falls back to N-1, which is
    the behaviour this function had unconditionally.

    WHY THE UNCONDITIONAL N-1 WAS A BUG. It made the lineage follow the most
    recent code regardless of what the measurement said, so a change that lost
    MGS became the permanent foundation of everything after it. run-8cf58d33b311
    scored 0.3172, then 0.2222, then 0.1830, then 0.1190 -- each iteration
    inheriting the one that had just been measured as worse, none of them able
    to get back to the code that was working. Two iterations of that is bad
    luck; four is a structural inability to keep a good answer.

    It also meant a build that FAILED could still be a parent: iteration 2 of
    that run hit the turn ceiling having written nothing, and iteration 3 was
    seeded from it anyway. A workspace with no judged score is not a champion
    candidate at all now, so that cannot happen.

    A copy rather than a shared directory so each iteration's code stays on disk
    exactly as it was evaluated. When the Critic cites a checkpoint from
    iteration 3, you can still read the code that produced it.
    """
    workspace = config.iteration_dir(iteration) / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    # Already populated (a resumed run, or an infrastructure retry of this same
    # iteration): leave it alone, and leave its baseline alone too -- re-taking
    # it here would record the resumed work as the starting state and blind the
    # did-anything-change guard.
    if any(p.suffix == ".py" for p in workspace.rglob("*")):
        return workspace, "existing"

    # Freshly seeded below, so any baseline from an earlier run of this
    # directory describes a workspace that no longer exists.
    (workspace.parent / "workspace_baseline.json").unlink(missing_ok=True)

    fallback = iteration - 1
    source_iteration = int(parent if parent is not None else fallback)
    previous = (
        config.iteration_dir(source_iteration) / "workspace"
        if source_iteration >= 1 else None
    )
    if previous and previous.is_dir():
        _copy_tree(previous, workspace)
        if source_iteration != fallback and reason == scoreboard.LINEAGE_SKIPPED_FAILED:
            # NOT a rollback, and the log must not call it one. Nothing scored
            # worse here; iteration `fallback` produced no score at all because
            # it could not build, so its workspace is not a foundation to stand
            # on. Reading `ROLLED BACK` against a build failure is how a run's
            # post-mortem invents a regression that never happened.
            log.warning(
                "workspace iter=%d SKIPPED A FAILED BUILD: seeded from iteration %d "
                "(the most recent iteration whose gates passed) instead of iteration "
                "%d, which never built; its failure report still goes forward",
                iteration, source_iteration, fallback,
            )
        elif source_iteration != fallback:
            log.warning(
                "workspace iter=%d ROLLED BACK: seeded from iteration %d (the "
                "current champion) instead of iteration %d, whose score was lower",
                iteration, source_iteration, fallback,
            )
        else:
            log.info("workspace iter=%d seeded from iteration %d", iteration, source_iteration)
        return workspace, f"iter_{source_iteration}"

    if config.SEED_FROM_TEMPLATE and config.TEMPLATES_DIR.is_dir():
        _copy_tree(config.TEMPLATES_DIR, workspace)
        log.info("workspace iter=%d seeded from %s", iteration, config.TEMPLATES_DIR)
        return workspace, "template"

    log.info("workspace iter=%d starting empty (SEED_FROM_TEMPLATE is off)", iteration)
    return workspace, "empty"


def _copy_tree(source: Path, destination: Path) -> None:
    """Copy source files only -- no caches, no scaffolds, no session logs.

    The Developer's own working notes do not cross the iteration boundary
    either. In runs_4iter, iteration 1's source-dumping scratch files -- 40-odd
    `_store_chunk_*.txt` and `_dump2_*.txt`, plus `_dump_store.py` and
    `tests/_dump_retrieve_test.py` -- were inherited by iterations 2 and 3 and
    re-dumped on every later pytest run. Lineage is for the deliverable.
    """
    for item in source.rglob("*"):
        if not item.is_file():
            continue
        if any(part in _NON_SOURCE for part in item.parts) or item.name in _SCAFFOLDS:
            continue
        if is_scratch(str(item.relative_to(source))):
            continue
        target = destination / item.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(item.read_bytes())


async def developer_node(state: OrchestratorState) -> dict[str, Any]:
    """Run one Developer episode and project its result into the macro state."""
    iteration = int(state.get("iteration_count", 1))
    phase = str(state.get("current_curriculum_phase") or config.CURRICULUM_PHASES[0])
    # THE CHAMPION, not the predecessor. `champion_iteration` returns the
    # fallback unchanged when nothing has been judged yet or when the best
    # iteration IS the previous one, so on an improving run this is a no-op.
    parent, lineage_reason = scoreboard.lineage_parent(state, fallback=iteration - 1)
    workspace, provenance = prepare_workspace(iteration, parent=parent, reason=lineage_reason)
    rolled_back = parent != iteration - 1 and provenance == f"iter_{parent}"
    _TOOLBOXES.pop(str(workspace.resolve()), None)  # fresh toolbox per iteration

    async with node_span("developer", iteration, phase) as span:
        initial: DeveloperState = {
            "instructions": str(state.get("dev_instructions") or ""),
            "sql_schema": str(state.get("sql_schema") or ""),
            "migration_sql": str(state.get("migration_sql") or ""),
            "workspace": str(workspace.resolve()),
            "iteration": iteration,
            "workspace_from": provenance,
            "scratchpad": [],
            "failures": [],
            "retry_count": 0,
            "compile_ok": False, "tests_ok": False, "migration_ok": False,
            "lint_ok": False, "smoke_ok": False, "done": False,
            "pass_rate": 0.0, "last_stack_trace": None,
            "gate_attempts": {}, "transport_failures": 0, "transport_errors": [],
        }
        # The episode runs to one of its own stop conditions -- gates green, a
        # `finish` accepted, the retry budget gone, the wall clock, or the turn
        # ceiling. There is no `recursion_limit` to set because there is no
        # sub-graph to bound: `DeveloperSession._should_stop` owns all of it, in
        # one place, in terms the failure report can name.
        session = DeveloperSession(initial)
        final = await session.run()
        span["turns"] = session.turns

        exhausted = int(final.get("retry_count", 0)) >= config.MAX_DEV_RETRIES and not final.get("done")
        gates_green = bool(final.get("compile_ok") and final.get("tests_ok") and final.get("migration_ok"))
        # Why the loop actually stopped, when it was not the retry budget: the
        # wall clock and the turn ceiling both end an episode with retries to
        # spare, and "exhausted 5 retries" would be a false report of both.
        stopped_by = str(final.get("halt_reason") or "")

        iter_dir = config.iteration_dir(iteration)
        write_artifact(iter_dir / "developer_scratchpad.json", final.get("scratchpad") or [])
        delta = _summarize_delta(workspace)
        write_artifact(iter_dir / "codebase_delta.txt", delta)

        # The Developer's spend, NOT zero (regression). This span reported 0 and
        # the node returned no `token_usage` at all, so the loop's single biggest
        # consumer -- ~20 turns per episode, each bounded by the think timeout rather
        # than by a token ceiling (DEVELOPER_MAX_TOKENS is uncapped by default),
        # plus every degenerate resample billed in full -- was invisible to
        # MAX_TOTAL_TOKENS. A cost cap that cannot see the biggest spender is
        # not a cap, and it matters MORE with the ceiling gone. Same class of bug
        # as the camelCase miss in `_usage_from_events`; the other half of it.
        tokens = _add_usage({}, final.get("tokens_used"))
        span["tokens"] = tokens.get("total_tokens", 0)
        span["workspace_from"] = provenance
        span["rolled_back"] = rolled_back
        # WHY the parent is not N-1, alongside the bare fact that it isn't. A
        # post-mortem reading `rolled_back: true` alone cannot tell a score
        # rollback from a skipped failed build, and they call for opposite
        # conclusions about the run.
        span["lineage_reason"] = lineage_reason
        span["retries"] = final.get("retry_count", 0)
        span["transport_failures"] = int(final.get("transport_failures", 0) or 0)
        span["gates_green"] = gates_green
        span["gate_status"] = gate_status(final)

        # What this episode actually built, as opposed to what it inherited.
        # Reported next to the gates because the two are easy to confuse and
        # runs_4iter confused them: "gates green" and "code unchanged" were both
        # true of iteration 2, and only the first was in the summary.
        episode_box = _TOOLBOXES.get(str(workspace.resolve()))
        changed = episode_box.substantive_changes() if episode_box else []
        span["files_changed"] = len(changed)

        update: dict[str, Any] = {
            # The iteration got as far as a build, so whatever transport trouble
            # its Architect turn had is over. Reset here rather than in the
            # Architect, which would clear the budget on the very retry it is
            # meant to bound. See graph.architect_retry_node.
            "architect_retry_count": 0,
            "memory_codebase": str(workspace.resolve()),
            "codebase_delta": delta,
            "dev_set_pass_rate": float(final.get("pass_rate", 0.0)),
            "dev_retries_used": int(final.get("retry_count", 0)),
            # How much of the episode's turn budget the work order actually
            # cost. The Architect calibrates the NEXT work order against it:
            # in run-b3275eb7e373 a 5-step order landed in 17-19 turns while
            # 8- and 10-step orders hit the 60-turn ceiling with the gates
            # never run, and nothing told the Architect that was the pattern.
            "dev_turns_used": int(session.turns),
            "dev_turn_ceiling": _max_turns(),
            "dev_files_changed": changed,
            "token_usage": tokens,
            "node_timings": [span],
        }

        if exhausted or not gates_green:
            signature = final.get("failure_signature") or signature_from_trace(
                str(final.get("last_stack_trace") or "developer exhausted with no trace"),
                category="developer_exhaustion",
            )
            if stopped_by in {"timeout", "turn_cap"}:
                limit = (
                    f"the {config.DSH_DEVELOPER_TIMEOUT_S:.0f}s episode wall clock"
                    if stopped_by == "timeout" else f"the {_max_turns()}-turn ceiling"
                )
                ended = f"developer hit {limit} after {int(final.get('retry_count', 0))} retry/ies"
            else:
                ended = f"developer exhausted {config.MAX_DEV_RETRIES} retries"
            reason = (
                f"{ended} "
                f"(compile_ok={final.get('compile_ok')} tests_ok={final.get('tests_ok')} "
                f"migration_ok={final.get('migration_ok')})"
            )
            box = _TOOLBOXES.get(str(workspace.resolve()))
            report = _failure_report(
                final,
                iteration=iteration,
                provenance=provenance,
                signature=signature,
                reason=reason,
                files_written=sorted(box.files_written) if box else [],
            )
            update["failure_signature"] = signature
            update["halt_reason"] = reason
            # The evidence travels with the verdict: this is what the next
            # Architect reads as a critique of the design it just proposed.
            update["dev_failure_report"] = report
            update["dev_failure_history"] = [_failure_history_entry(report)]
            write_artifact(iter_dir / "developer_failure.json", report)
            # The log line says which gates FAILED and which never RAN, rather
            # than one undifferentiated `missing=` list. Told apart at the point
            # of failure they are one word each; told apart afterwards they cost
            # a session-log archaeology dig, which is how run-8cf58d33b311's
            # iteration 2 got read as a test failure for the rest of the run.
            log.warning(
                "developer exhausted at iteration %d: classification=%s (%s) "
                "signature=%s failed=%s never_ran=%s transport_failures=%d last_error=%s",
                iteration, report["classification"], report["classification_reason"],
                signature, ",".join(report["gates_failed"]) or "(none)",
                ",".join(report["gates_never_ran"]) or "(none)",
                report["n_transport_failures"], report["last_error"] or "(none)",
            )
        else:
            # A green build must CLEAR the report. `dev_failure_report` has no
            # reducer, so leaving it unwritten leaves iteration N-1's failure in
            # state -- and the next Architect would redesign around a build
            # problem that has already been fixed.
            update["dev_failure_report"] = {}
        return update


def _summarize_delta(workspace: Path) -> str:
    """A compact manifest of what the Developer produced.

    A real `git diff` would be richer, but the workspace is per-iteration and
    not a repository; the manifest is enough for the Architect's read-only view
    and does not pretend to be a diff it is not.
    """
    lines: list[str] = [f"workspace: {workspace}"]
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or path.name == "_smoke_runner.py":
            continue
        if path.suffix not in {".py", ".sql"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines.append(f"  {path.relative_to(workspace)}  {len(text)} chars, {text.count(chr(10)) + 1} lines")
    return "\n".join(lines)
