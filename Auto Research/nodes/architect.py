"""Architect node -- the Proposer.

Reads: the Critic's latest critique, the previous iteration's DEVELOPER BUILD
FAILURE if there was one, the current SQL schema, a read-only view of the
codebase, and a web survey of prior art.
Writes: `proposed_design`, `sql_schema`, `migration_sql`, `dev_instructions`.

Privilege: `ARCHITECT_PROFILE` mounts no file-write plugin and no shell, so the
Architect physically cannot edit the codebase.  Its read-only view of the code
is inlined into the task text below rather than granted as a tool, because the
harness's file-tool plugin is a single read+write surface -- passing content is
safe where passing the capability is not.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import config
from harness.profiles import ARCHITECT_PROFILE
from nodes._common import node_span, usage_delta, write_artifact
from harness.dsh_client import looks_like_tool_call_markup, strip_tool_call_markup
from nodes._transport import agent_call_json
from state import RESET, OrchestratorState
from websearch import get_search_client

log = logging.getLogger("orchestrator.architect")

# Queries the Architect runs before proposing.  Fixed rather than model-chosen
# so that iteration N and N+1 see the same prior art and the design delta is
# attributable to the critique rather than to search drift.
PRIOR_ART_QUERIES = (
    "multi-principal shared memory governance LLM agents access control",
    "RBAC filtered retrieval augmented generation row level security",
    "active forgetting tombstoning cryptographic shredding right to erasure",
)

CODE_VIEW_MAX_CHARS = config.ARCHITECT_CODE_VIEW_MAX_CHARS


def _code_view_source(workspace: Path) -> tuple[Path | None, str]:
    """Where the Architect's read-only view comes from, and how to label it.

    ITERATION 1 IS THE SUBTLE CASE, and getting it wrong is expensive. The
    workspace does not exist yet when the Architect runs: `prepare_workspace`
    seeds it inside the *Developer* node, which is the next node in the graph.
    A view of `workspace` therefore shows nothing on iteration 1 and the
    Architect designs an API in a vacuum.

    That is not a neutral omission. With SEED_FROM_TEMPLATE on, the Developer is
    about to be handed `templates/` -- a green implementation whose tests are the
    gate it has to pass. An Architect that has not seen them proposes a different
    contract entirely (OBSERVED: `documents` / `retrieve_documents(user_id)` /
    `forget_document(id)` against the template's `records` / `MemoryStore.retrieve`
    / `tombstone` / `Decision` / `Evidence`), and the Developer then burns every
    one of MAX_DEV_RETRIES trying to satisfy two incompatible contracts at once.
    It broke 3 of the template's 18 green tests doing so, and the iteration died
    without the Evaluator, Judge or Critic ever running.

    So the Architect is shown what the Developer will actually start from.
    """
    if workspace.is_dir() and any(workspace.rglob("*.py")):
        return workspace, "CURRENT IMPLEMENTATION"
    if config.SEED_FROM_TEMPLATE and config.TEMPLATES_DIR.is_dir():
        return config.TEMPLATES_DIR, "BASELINE THE DEVELOPER STARTS FROM (templates/)"
    return None, ""


def _code_view_order(paths: list[Path]) -> list[Path]:
    """Tests first, then everything else alphabetically.

    THE TESTS ARE THE CONTRACT. They are what the Developer's `run_tests` gate
    actually enforces, so they are the part of the view the design must not
    contradict -- and under a plain `sorted()` they are also the part that gets
    dropped, because `memory_system/` sorts ahead of `tests/` and
    `memory_system/store.py` alone is 22k characters. CODE_VIEW_MAX_CHARS is now
    sized to fit the whole baseline so nothing is dropped at all, but that is a
    budget anyone can lower and any workspace can outgrow. Ordering is what makes
    the truncation safe whenever it does bite: the Architect seeing an
    implementation with none of the assertions binding it is strictly worse than
    seeing nothing, because it invites a confident redesign of an API whose
    contract is invisible.
    """
    return sorted(paths, key=lambda p: (0 if "tests" in p.parts else 1, p.as_posix()))


def _read_only_codebase_view(workspace: Path) -> str:
    """Inline a size-bounded view of the code the Developer will be working in."""
    source, label = _code_view_source(workspace)
    if source is None:
        return "(no implementation yet, and no template to seed from -- this is iteration 1)"

    candidates = [
        path for path in source.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sql"}
        and not any(part in {".sessions", "__pycache__", ".cordis"} for part in path.parts)
    ]

    preamble = (
        f"[{label}]\n"
        "The files under tests/ are the contract the Developer's test gate enforces.\n"
        "A design that renames or re-signatures what they import cannot be built:\n"
        "the Developer must keep them passing. Extend this code; do not replace it.\n"
    )
    chunks: list[str] = [preamble]
    budget = CODE_VIEW_MAX_CHARS
    for path in _code_view_order(candidates):
        text = path.read_text(encoding="utf-8", errors="replace")
        header = f"\n----- {path.relative_to(source)} ({len(text)} chars) -----\n"
        if budget - len(header) <= 0:
            chunks.append("\n[... code view truncated ...]")
            break
        body = text[: max(0, budget - len(header))]
        chunks.append(header + body)
        budget -= len(header) + len(body)
    return "".join(chunks) if len(chunks) > 1 else "(workspace is empty)"


# ======================================================================
# the Developer's build failure, as a critique of the design
# ======================================================================
#
# WHY THIS EXISTS. `routers.route_after_developer` sends an unbuildable design
# back to the Architect on the explicit grounds that a build which never
# compiled is evidence about the DESIGN. That edge was, until now, a one-bit
# channel: the Architect saw `halt_reason` nowhere in its prompt and re-derived
# its proposal from the same critique, the same metrics and the same code view
# it had last time -- with nothing in front of it saying the last design had
# been rejected by reality. The predictable result is iteration N+1 restating
# iteration N's work order and dying on the same assertion.
#
# The Critic cannot cover this case: it runs after the Judge, and a failed build
# never reaches either. So a failed build produces NO critique at all, and this
# block is the only feedback the Architect gets for that whole iteration.

DEV_FAILURE_MAX_CHARS = config.ARCHITECT_DEV_FAILURE_MAX_CHARS

# Advice that does not depend on which gate failed. Kept short and specific:
# the failing observations above it are the evidence, and this is only the
# framing that stops the Architect reading them as a Developer performance
# problem it can ignore.
_DEV_FAILURE_GUIDANCE = """
### WHAT THIS MEANS FOR THIS ITERATION
- Only the design can fix this. The Developer already spent its entire retry
  budget on the work order you wrote; handing it the same one costs another
  whole iteration and produces the same failure.
- The files under `tests/` are the contract and the Developer cannot change them
  to suit a design. If a step above failed because a test forbids what the work
  order asked for, the work order is what changes.
- An unbuilt design scores nothing at all -- no evaluation, no judge, no
  critique. A narrower change that compiles is worth more than a broader one
  that does not, so cut scope before you add to it.
- Say what you changed in response under `dev_failure_mitigations` in the JSON
  block, one entry per failure you are addressing.
"""


def _gate_summary(report: dict[str, Any]) -> str:
    """The unmet MANDATORY gates, each named with the tool that sets it."""
    gate_tools = report.get("gate_tools") or {}
    missing = [str(name) for name in (report.get("missing_gates") or [])]
    if not missing:
        return "(all three mandatory gates were green; the episode still did not finish)"
    return ", ".join(f"{name} (set by `{gate_tools.get(name, '?')}`)" for name in missing)


def _repeat_warning(state: OrchestratorState, report: dict[str, Any]) -> str:
    """Escalate when the SAME failure has now killed more than one iteration.

    This is the case the loop most needs pointed out. A signature recurring
    means the previous redesign did not touch whatever actually broke, and an
    Architect reading one failure report in isolation has no way to know that --
    it looks like a first occurrence every time.
    """
    signature = str(report.get("signature") or "")
    if not signature:
        return ""
    iterations = [
        entry.get("iteration")
        for entry in (state.get("dev_failure_history") or [])
        if isinstance(entry, dict) and entry.get("signature") == signature
    ]
    if len(iterations) < 2:
        return ""
    where = ", ".join(str(i) for i in iterations)
    return (
        "\n### THIS FAILURE HAS NOW REPEATED\n"
        f"Signature {signature} has broken the build in {len(iterations)} iterations "
        f"({where}). The redesign after the first occurrence did not address it. "
        "Do not restate the same approach a third time -- change the mechanism, "
        "or drop the part of the design that keeps failing and land the rest.\n"
    )


def _failure_lines(report: dict[str, Any]) -> list[str]:
    """One rendered stanza per DISTINCT retry-charged failure, most useful first.

    Build failures are ordered ahead of loop failures because they are the ones
    a design can be changed in response to; a repeated idempotent call says the
    episode stalled, not that the schema is wrong.

    IDENTICAL FAILURES ARE COLLAPSED, for the same reason the evaluator's
    circuit breaker collapses signatures: a stuck ReAct loop re-runs the same
    tool until its retries are gone, so the untouched form of this section is
    the same 900-character pytest report five times over. Printed in full it
    crowds out every other failure and buries the one fact the repetition
    actually adds -- that nothing the Developer tried moved it. The count says
    that in a clause.
    """
    entries = [e for e in (report.get("failures") or []) if isinstance(e, dict)]
    ordered = sorted(entries, key=lambda e: (0 if e.get("kind") == "build" else 1, e.get("step", 0)))

    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in ordered:
        key = (entry.get("kind"), entry.get("tool"), entry.get("signature"),
               str(entry.get("error") or ""))
        group = groups.get(key)
        if group is None:
            groups[key] = {"entry": entry, "steps": [entry.get("step")]}
        else:
            group["steps"].append(entry.get("step"))

    lines: list[str] = []
    for group in groups.values():
        entry = group["entry"]
        steps = [str(step) for step in group["steps"] if step is not None]
        args = entry.get("args") or {}
        args_text = f" {args}" if args else ""
        if len(steps) > 1:
            where = f"steps {', '.join(steps)} -- {len(steps)} attempts, identical result"
        else:
            where = f"step {steps[0] if steps else '?'}"
        head = (
            f"- {where} -- `{entry.get('tool')}`{args_text}"
            f"{'' if entry.get('kind') == 'build' else '  [loop stall, not a design fault]'}"
        )
        error = str(entry.get("error") or "").strip()
        body = str(entry.get("observation") or "").strip()
        stanza = head
        if error:
            stanza += f"\n  {error}"
        if body:
            stanza += "\n" + "\n".join(f"  | {line}" for line in body.splitlines())
        lines.append(stanza)
    return lines


def _developer_failure_block(state: OrchestratorState) -> str:
    """Render `dev_failure_report` for the Architect, or "" if the build was green.

    Size-bounded like the code view, and bounded in the same shape: the header,
    the gate summary and the recurrence warning are never dropped, because they
    are the parts that change what the Architect does. The per-step excerpts are
    the detail, so they are what falls off the end.
    """
    report = state.get("dev_failure_report") or {}
    if not isinstance(report, dict) or not report:
        return ""

    header = (
        "\n## THE DEVELOPER COULD NOT BUILD YOUR PREVIOUS DESIGN\n"
        "Read this as a critique of the design, not as a status report: the build "
        "failed, so this iteration produced no evaluation, no MGS and no Critic "
        "feedback at all. The metrics above are from the last iteration that did "
        "build.\n\n"
        f"reason: {report.get('reason', 'unknown')}\n"
        f"failure_signature: {report.get('signature') or '(none)'}\n"
        f"unmet mandatory gates: {_gate_summary(report)}\n"
        f"retries: {report.get('retries_used', 0)}/{report.get('retry_cap', 0)}"
        f"{' (exhausted)' if report.get('exhausted') else ''}\n"
        f"local test pass rate: {float(report.get('pass_rate', 0.0)):.3f}\n"
        f"workspace started from: {report.get('workspace_from', 'unknown')}\n"
        f"files the Developer wrote: "
        f"{', '.join(report.get('files_written') or []) or '(none -- it never wrote anything)'}\n"
    )
    repeat = _repeat_warning(state, report)

    counts = (
        f"\n### WHERE IT FAILED\n"
        f"{report.get('n_build_failures', 0)} build failure(s) and "
        f"{report.get('n_loop_failures', 0)} loop stall(s) spent the retry budget.\n"
    )

    # The guidance is reserved out of the budget rather than appended after it,
    # so a long pytest failure can never push the instruction that tells the
    # Architect what to DO with all of this off the end.
    chunks = [header, repeat, counts]
    budget = DEV_FAILURE_MAX_CHARS - sum(len(c) for c in chunks) - len(_DEV_FAILURE_GUIDANCE)

    dropped = 0
    for stanza in _failure_lines(report):
        if len(stanza) + 1 > budget:
            dropped += 1
            continue
        chunks.append(stanza + "\n")
        budget -= len(stanza) + 1
    if dropped:
        chunks.append(f"[... {dropped} further distinct failure(s) omitted for length ...]\n")

    trace = str(report.get("last_stack_trace") or "").strip()
    if trace and len(trace) + 40 <= budget:
        chunks.append(f"\n### THE TRACE IT DIED ON\n```\n{trace}\n```\n")

    chunks.append(_DEV_FAILURE_GUIDANCE)
    return "".join(chunks)


def _work_order_text(steps: list[Any]) -> str:
    return "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))


def _as_sql(value: Any) -> str:
    """Coerce a `schema_ddl` / `migration_sql` field to a SQL string.

    Models emit these as a string most of the time and as a LIST of statements
    the rest of the time. `str(a_list)` yields a Python repr -- `["CREATE ...",
    ...]` -- which is not SQL, fails to execute, and reads in the artifact like
    the model produced nonsense. Joining is what was meant.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        statements = [str(item).strip() for item in value if str(item).strip()]
        return ";\n".join(s.rstrip(";") for s in statements) + (";" if statements else "")
    return str(value)


_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.DOTALL)


def _fence_tags(text: str) -> str:
    """The fence languages present in a reply, for diagnosing a failed parse."""
    return ",".join(tag.lower() or "(untagged)" for tag, _ in _FENCE_RE.findall(text or ""))


def _sql_from_prose(text: str) -> str:
    """Last-resort DDL salvage: the ```sql fences in the design document.

    Only CREATE/ALTER statements are kept. A design's SQL fences also contain
    illustrative SELECTs for the retrieval loop, and feeding those to
    `sql_exec` as schema would fail on tables the DDL has not created yet.
    """
    statements: list[str] = []
    for tag, body in _FENCE_RE.findall(text or ""):
        if tag.lower() != "sql":
            continue
        for statement in body.split(";"):
            head = statement.strip().lstrip("(").strip().upper()
            if head.startswith(("CREATE ", "ALTER ")):
                statements.append(statement.strip() + ";")
    return "\n".join(statements)


async def architect_node(state: OrchestratorState) -> dict[str, Any]:
    iteration = int(state.get("iteration_count", 0)) + 1
    phase = str(state.get("current_curriculum_phase") or config.CURRICULUM_PHASES[0])
    workspace = Path(state.get("memory_codebase") or config.iteration_dir(iteration) / "workspace")

    async with node_span("architect", iteration, phase) as span:
        # ---- (a) prior-art survey, required before proposing ----
        search = get_search_client()
        hits: list[str] = []
        for query in PRIOR_ART_QUERIES:
            for hit in await search.search(query, max_results=3):
                line = f"- {hit.title} ({hit.url}): {hit.snippet}"
                if line not in hits:
                    hits.append(line)
        prior_art = "\n".join(hits) or "(web search unavailable; proposing without prior art)"

        # STRIPPED BEFORE IT IS PASTED IN. A critique that carries tool-call
        # markup teaches the Architect to emit tool-call markup: in runs_5iter
        # iteration 3 the Architect reproduced the exact `OpenFile` invocation
        # it had been shown in iteration 2's critique, produced no design, and
        # the iteration changed nothing.
        critique = strip_tool_call_markup(str(state.get("critique") or ""))
        attribution = state.get("attribution") or {}
        previous_schema = str(state.get("sql_schema") or "")

        # An unbuilt design never reaches the Judge or the Critic, so on this
        # path `critique` is stale (or empty) and this block is the ONLY
        # feedback about the iteration that just failed.
        dev_failure = _developer_failure_block(state)

        # The `dominant_term` marker is what the mock (and a real Critic-aware
        # Architect) keys its targeted migration off; keep the token stable.
        feedback_block = ""
        if critique:
            feedback_block = (
                "\n## CRITIQUE FROM THE PREVIOUS ITERATION\n"
                f"dominant_term: {attribution.get('dominant_term', 'unknown')}\n"
                f"dominant failing metric: {attribution.get('dominant_term', 'unknown')}\n"
                f"marginal contributions: {attribution.get('marginal', {})}\n\n"
                f"{critique}\n"
            )

        # Asked for only when there is something to mitigate: a key the model is
        # told to fill in with nothing to fill it from invites invention, and
        # `design.json` would then record fictional responses to a build that
        # never failed.
        mitigation_key = (
            "\nAlso include `dev_failure_mitigations`: an array of strings, one per\n"
            "failure above, each naming the design change you made in response."
            if dev_failure else ""
        )

        task = f"""You are designing iteration {iteration} of the memory system.

## CURRENT CURRICULUM PHASE
{phase}

## MEASURED PERFORMANCE SO FAR
U (utility_accuracy)        = {state.get('utility_score', 0.0):.4f}
A (privacy_leakage_rate)    = {state.get('access_violation_rate', 1.0):.4f}
F (deletion_leakage_rate)   = {state.get('forgetting_failure_rate', 1.0):.4f}
MGS (compliance_utility_score) = {state.get('mgs_score', 0.0):.4f}   target {config.MGS_TARGET}
Developer local pass rate   = {state.get('dev_set_pass_rate', 0.0):.3f}
{dev_failure}{feedback_block}
## PRIOR ART (from web search -- cite what you use)
{prior_art}

## CURRENT SQL SCHEMA
```sql
{previous_schema or '(none yet -- propose the initial DDL)'}
```

## CURRENT IMPLEMENTATION (read-only)
{_read_only_codebase_view(workspace)}

## YOUR TASK
Produce the design document and the Developer's work order.

THE WORK ORDER IS A DELTA AGAINST THE IMPLEMENTATION ABOVE, NOT A REBUILD OF IT.
The Developer inherits that exact workspace, already compiling and already
passing its tests, and its episode is REJECTED if it finishes without changing a
source file -- so a step like "implement `retrieve()`" against code that already
has one buys nothing and costs the iteration. Every step must name a change:
what file, what function, and what it does differently afterwards. If a
mechanism is already present and correct, do not restate it as work; spend the
iteration on the term the measurements above say is losing.

Remember the storage constraint (SQL/SQLite only) and both MUST requirements in
your instructions.
End with a single ```json fenced block containing schema_ddl, migration_sql,
retrieval_loop, forgetting_mechanism, work_order, targets_metric and
expected_tradeoff.{mitigation_key}
"""

        # Explicit budget: this is one open-ended design turn over a large
        # context, not a tight tool loop, and it does not fit the general
        # per-node default.
        result, block = await agent_call_json(
            ARCHITECT_PROFILE, task, workspace.parent,
            timeout_s=int(config.DSH_ARCHITECT_TIMEOUT_S),
        )
        span["tokens"] = result.usage.get("total_tokens", 0)

        if not result.ok:
            # A failed Architect is fatal to the iteration -- there is nothing
            # for the Developer to build. Halt rather than let the Developer
            # invent its own spec.
            log.error("architect failed: %s", result.error)
            return {
                "halt_reason": f"architect invocation failed: {result.error}",
                "iteration_count": iteration,
                "node_timings": [span],
            }

        # The design document crosses into the Developer's work order and the
        # Critic's prompt, so it is stripped for the same reason the critique is.
        design = strip_tool_call_markup(result.text)
        if looks_like_tool_call_markup(result.text):
            log.error(
                "architect emitted tool-call markup and no design even after a repair "
                "pass; it has no tools on this call and the code view is inlined in its "
                "task. The Developer is about to be handed a salvaged spec."
            )

        # A design whose JSON block did not parse is the most expensive silent
        # failure in this loop -- see the regression note on `extract_json_block`.
        # `extract_json_block` now repairs the dialects models actually emit, so
        # reaching here means the reply had no recoverable object at all. Say so
        # at ERROR, and salvage the DDL from the prose rather than handing the
        # Developer an empty spec and a zero-byte migration.
        if not block:
            log.error(
                "architect produced NO parseable json block (%d chars of prose, "
                "fences=%s); salvaging DDL from the prose",
                len(design), _fence_tags(design) or "(none)",
            )

        schema_ddl = _as_sql(block.get("schema_ddl")) or _sql_from_prose(design) or previous_schema
        migration_sql = _as_sql(block.get("migration_sql"))
        work_order = block.get("work_order") or []
        instructions = _work_order_text(work_order) if isinstance(work_order, list) else str(work_order)

        if not instructions.strip():
            # Fall back to the prose rather than handing the Developer nothing;
            # a vague work order is recoverable, an empty one is not.
            instructions = design
            log.warning("architect returned no parseable work_order; falling back to prose")

        if not schema_ddl.strip():
            # Iteration 1 with no DDL from anywhere means `sql_exec` has no
            # schema to apply and the Developer cannot clear its migration gate.
            # That is a design failure, and it belongs to the node that owns the
            # design -- surfacing it here beats three silent retries downstream.
            log.error("architect produced no schema DDL and none exists from a "
                      "previous iteration; the Developer has nothing to migrate")

        iter_dir = config.iteration_dir(iteration)
        write_artifact(iter_dir / "design.md", design)
        write_artifact(iter_dir / "migration.sql", migration_sql)
        write_artifact(
            iter_dir / "design.json",
            {
                "targets_metric": block.get("targets_metric"),
                "expected_tradeoff": block.get("expected_tradeoff"),
                "retrieval_loop": block.get("retrieval_loop"),
                "forgetting_mechanism": block.get("forgetting_mechanism"),
                "work_order": work_order,
                "prior_art": hits,
                # Recorded so a run's history shows whether a redesign actually
                # responded to the build failure that forced it, or merely
                # followed it. `responds_to_dev_failure` is the signature the
                # Architect was shown, which is what a repeat is detected on.
                "dev_failure_mitigations": block.get("dev_failure_mitigations"),
                "responds_to_dev_failure": (state.get("dev_failure_report") or {}).get("signature"),
            },
        )

        return {
            "proposed_design": design,
            "sql_schema": schema_ddl,
            "migration_sql": migration_sql,
            "dev_instructions": instructions,
            "iteration_count": iteration,
            # A fresh iteration starts with a clean evaluation slate: `eval_results`
            # has an `operator.add` reducer, so it can only be reset by the one
            # node that runs alone at the top of the iteration.
            "eval_stage": "dev",
            "eval_results": RESET,
            "predictions_path": "",
            "failure_signature": RESET,
            "halt_reason": None,
            "token_usage": usage_delta(result.usage),
            "node_timings": [span],
        }
