"""Architect node -- the Proposer.

Reads: the Critic's latest `critique.md`, its own notebook of every iteration
before that one (`runs/critique_summary.md`), the previous iteration's DEVELOPER
BUILD FAILURE if there was one, the current SQL schema, a read-only view of the
codebase, and a web survey of prior art.
Writes: `proposed_design`, `sql_schema`, `migration_sql`, `dev_instructions`,
`critique_digest`, and the notebook itself.

IT IS ALSO THE LOOP'S HISTORIAN, and the ORDER IN WHICH IT DOES THAT IS THE
POINT.  Iteration i+1's turn runs in exactly four steps:

    1. read iteration i's `critique.md` (it is `state["critique"]` verbatim --
       the Critic writes that text and nothing else into the file);
    2. read `critique_summary.md`, the notebook of iterations 1..i-1;
    3. write the design, the schema and the Developer's work order from the two;
    4. ONLY THEN append a summary of iteration i's critique to the notebook.

Step 4 comes last so that the notebook a turn reads is the history BEFORE the
critique it is answering: fold step 4 into step 2 and the newest critique arrives
twice, once in full and once as a summary of itself, with nothing marking which
of the two the design is supposed to be a response to.

The Architect is the right node to keep that notebook because it is the only one
that reads a critique and writes the design answering it in the same turn, so the
summary comes back in the same JSON block as the design -- no extra model call,
and no second reader that could disagree with the first about what the critique
said.  See nodes/_recap.py.

Privilege: `ARCHITECT_PROFILE` mounts no file-write plugin and no shell, so the
Architect physically cannot edit the codebase.  Its read-only view of the code
is inlined into the task text below rather than granted as a tool, because the
harness's file-tool plugin is a single read+write surface -- passing content is
safe where passing the capability is not.
"""

from __future__ import annotations

import logging
import re
import json
from pathlib import Path
from typing import Any

import config
import scoreboard
from harness.dsh_client import looks_like_tool_call_markup, strip_tool_call_markup
from harness.profiles import ARCHITECT_PROFILE
from nodes._common import node_span, usage_delta, write_artifact
from nodes._recap import (
    append_to_notebook,
    dev_failure_summary,
    digest_entry,
    fallback_critique_summary,
    load_notebook,
    render_notebook_prompt,
)
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

# The design-failure advice above is wrong for the other two classifications:
# "only the design can fix this" is false of a network outage and false of a
# work order too vague to act on, and following it produces exactly the
# unnecessary redesign that both cases must avoid.
_CLASSIFICATION_GUIDANCE: dict[str, str] = {
    "infrastructure": """
### WHAT THIS MEANS FOR THIS ITERATION
- The design was never actually tested. Do not treat any gate above as evidence
  about it, and do not change the schema in response to a transport failure.
- Keep the previous design's intent. If you still believe in it, restate it as a
  SHORTER work order -- the fewer turns an episode needs, the less of it a flaky
  endpoint can eat.
- Do not spend this iteration on a new mechanism. The cheapest useful outcome
  here is a clean measurement of the design you already had.
- Say under `dev_failure_mitigations` how you shortened the work order.
""",
    "inconclusive": """
### WHAT THIS MEANS FOR THIS ITERATION
- Nothing failed. The Developer never got as far as running a gate, which means
  the work order did not tell it concretely enough what to change.
- Rewrite the WORK ORDER, not the design. Every step must name a file, a
  function, and what that function does differently afterwards. A step like
  "implement retrieve()" against code that already has one is not actionable and
  is what produces an episode of reading.
- Cut the number of steps. An episode that lands three concrete edits is worth
  more than one that reads the whole workspace deciding where to start.
- Say under `dev_failure_mitigations` which steps you made concrete.
""",
}


# How the Architect should READ a failed build, per classification. The framing
# is the difference between a redesign and a re-run, and it used to be one
# sentence for all three: "Read this as a critique of the design."
#
# In run-8cf58d33b311 that sentence was wrong. Iteration 2's episode never
# called `run_tests` or `sql_exec` -- it died of OpenRouter timeouts -- and the
# Architect, told to read it as a design critique, produced the work order
# ("remove the top_k stop") that cost the run 0.095 MGS when iteration 3 built
# it. A gate that never ran is not a gate that failed, and the Architect must be
# told which of the two it is looking at before it is asked to respond.
_CLASSIFICATION_FRAMING: dict[str, str] = {
    "design": (
        "Read this as a critique of the design: a mandatory gate RAN and FAILED. "
        "The build produced no evaluation, no MGS and no Critic feedback, so this "
        "block is the whole of what this iteration measured. The metrics above "
        "are from the last iteration that did build.\n"
    ),
    "infrastructure": (
        "DO NOT REDESIGN IN RESPONSE TO THIS. The episode died of INFRASTRUCTURE "
        "-- transport timeouts, empty completions, a dead endpoint -- not of "
        "anything about your design. The graph has already re-run this iteration "
        "as many times as its budget allows and it still could not get a clean "
        "episode, so you are seeing it. The right response is the SMALLEST "
        "possible work order that still moves the metric, so that the next "
        "episode needs fewer turns to land it. Changing the schema will not make "
        "the network work.\n"
    ),
    "inconclusive": (
        "THIS IS A WORK-ORDER PROBLEM, NOT A SCHEMA PROBLEM. No mandatory gate "
        "failed -- the Developer simply never ran one. It read, listed and "
        "thought its way to the ceiling without building anything, which is what "
        "an episode does when the work order does not say concretely enough what "
        "to change. Do not rewrite the design in response. Rewrite the WORK "
        "ORDER: fewer steps, each naming one file, one function, and what that "
        "function does differently afterwards.\n"
    ),
}


def _turn_budget_note(state: OrchestratorState) -> str:
    """How much of its turn budget the previous Developer episode spent.

    Rendered as a warning when the episode came close to the ceiling, because
    "43 of 60" and "60 of 60" are the difference between a work order that fit
    and one that cost the whole iteration. Silent on iteration 1, which has no
    previous episode and would otherwise be handed a fabricated 0/0.
    """
    used = int(state.get("dev_turns_used", 0) or 0)
    ceiling = int(state.get("dev_turn_ceiling", 0) or 0)
    if not used or not ceiling:
        return ""
    line = (
        f"Previous Developer episode used {used}/{ceiling} turns"
        f" for a {len(_work_order_steps(state))}-step work order."
    )
    if used >= ceiling:
        return (
            line
            + "\n!! IT RAN OUT OF TURNS. An episode that hits the ceiling before running"
            "\n!! `compile_check`, `run_tests` and `sql_exec` is scored as a FAILED BUILD"
            "\n!! and every edit it made is discarded -- the iteration produces nothing."
            "\n!! YOUR WORK ORDER WAS TOO BIG. Cut it to the smallest change that still"
            "\n!! moves the metric: fewer steps, each naming one file, one function, and"
            "\n!! what that function does differently afterwards."
        )
    if used >= ceiling * 0.75:
        return (
            line
            + f"\n!! That is {used / ceiling:.0%} of the budget. Another step or two and the"
            "\n!! episode would have hit the ceiling and scored nothing. Keep this one"
            "\n!! the same size or smaller."
        )
    return line


def _work_order_steps(state: OrchestratorState) -> list[str]:
    """The previous work order's steps, for reporting its size back."""
    text = str(state.get("dev_instructions") or "")
    return [line for line in text.splitlines() if line.strip()]


def _gate_summary(report: dict[str, Any]) -> str:
    """The mandatory gates, saying for each whether it FAILED or never RAN.

    THESE ARE NOT THE SAME FACT and the previous version of this function
    printed them as one. `missing_gates` is every gate whose boolean is False,
    and the boolean is False both when the tool ran and failed and when the tool
    was never called at all. The Architect was handed `unmet mandatory gates:
    tests_ok, migration_ok` for an episode in which neither `run_tests` nor
    `sql_exec` had ever been invoked, and it redesigned against a test failure
    that had not happened.
    """
    gate_tools = report.get("gate_tools") or {}
    status = report.get("gate_status") or {}
    if not status:
        # A report from before gate provenance existed (a resumed run, a fixture).
        missing = [str(name) for name in (report.get("missing_gates") or [])]
        if not missing:
            return "mandatory gates: all three green; the episode still did not finish"
        return "unmet mandatory gates: " + ", ".join(
            f"{name} (set by `{gate_tools.get(name, '?')}`)" for name in missing
        )

    failed = [name for name, value in sorted(status.items()) if value == "failed"]
    never = [name for name, value in sorted(status.items()) if value == "never_ran"]
    parts: list[str] = []
    if failed:
        parts.append(
            "gates that RAN AND FAILED: "
            + ", ".join(f"{name} (`{gate_tools.get(name, '?')}`)" for name in failed)
        )
    if never:
        parts.append(
            "gates that were NEVER RUN (this is not evidence that they would fail): "
            + ", ".join(f"{name} (`{gate_tools.get(name, '?')}` was never called)"
                        for name in never)
        )
    if not parts:
        return "mandatory gates: all three green; the episode still did not finish"
    return "\n".join(parts)


def _transport_clause(report: dict[str, Any]) -> str:
    """How much of the retry budget the network spent, when any of it did."""
    count = int(report.get("n_transport_failures", 0) or 0)
    if not count:
        return ""
    errors = [str(e) for e in (report.get("transport_errors") or []) if e]
    detail = f" ({errors[0]})" if errors else ""
    return f", of which {count} were TRANSPORT failures, not build failures{detail}"


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

    classification = str(report.get("classification") or "design")
    header = (
        f"\n## THE DEVELOPER COULD NOT BUILD YOUR PREVIOUS DESIGN ({classification})\n"
        f"{_CLASSIFICATION_FRAMING.get(classification, _CLASSIFICATION_FRAMING['design'])}\n"
        f"reason: {report.get('reason', 'unknown')}\n"
        f"why it failed: {report.get('classification_reason') or '(unclassified)'}\n"
        f"failure_signature: {report.get('signature') or '(none)'}\n"
        f"{_gate_summary(report)}\n"
        f"retries: {report.get('retries_used', 0)}/{report.get('retry_cap', 0)}"
        f"{' (exhausted)' if report.get('exhausted') else ''}"
        f"{_transport_clause(report)}\n"
        f"local test pass rate: {float(report.get('pass_rate', 0.0)):.3f}\n"
        f"workspace started from: {report.get('workspace_from', 'unknown')}\n"
        f"files the Developer wrote: "
        f"{', '.join(report.get('files_written') or []) or '(none -- it never wrote anything)'}\n"
    )
    repeat = _repeat_warning(state, report)

    counts = (
        f"\n### WHERE IT FAILED\n"
        f"{report.get('n_build_failures', 0)} build failure(s), "
        f"{report.get('n_loop_failures', 0)} loop stall(s) and "
        f"{report.get('n_transport_failures', 0)} transport failure(s) "
        f"spent the retry budget.\n"
    )

    # The advice that matches what actually happened. See the note above
    # `_CLASSIFICATION_GUIDANCE`: telling an Architect that "only the design can
    # fix this" after a network outage is how a redesign gets written in answer
    # to a timeout.
    guidance = _CLASSIFICATION_GUIDANCE.get(classification, _DEV_FAILURE_GUIDANCE)

    # The guidance is reserved out of the budget rather than appended after it,
    # so a long pytest failure can never push the instruction that tells the
    # Architect what to DO with all of this off the end.
    chunks = [header, repeat, counts]
    budget = DEV_FAILURE_MAX_CHARS - sum(len(c) for c in chunks) - len(guidance)

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

    chunks.append(guidance)
    return "".join(chunks)


# ======================================================================
# the notebook of everything before this iteration
# ======================================================================
#
# WHY THE ARCHITECT KEEPS IT. `state["critique"]` holds exactly one critique --
# the most recent -- so iteration 4 was handed iteration 3's diagnosis and had no
# record whatsoever of what iterations 1 and 2 had already found and already
# tried. In runs_test2 that showed up as iterations 1, 2 and 4 all blaming the
# same `sanitize_and_decide` branch ordering, each without any sign of knowing
# an earlier one had already been there.
#
# The Architect is the right node to summarise it because it is the only one that
# reads a critique and writes the design answering it in the same turn, and it
# does so in the JSON block it is already producing -- so the notebook costs no
# extra model call and there is no second reader to disagree about what the
# critique said. The rows it adds go to `runs/critique_summary.md` AND to
# `critique_digest` in macro-graph state: the file is what the next turn reads,
# and the state list is what survives a wiped or repointed RUNS_DIR.


def _critique_source_iteration(state: OrchestratorState, iteration: int) -> int:
    """Which iteration the critique currently in state was written by.

    NOT always `iteration - 1`: a build that never compiled never reaches the
    Judge, so no Critic runs and `critique` stays as whatever the last iteration
    that DID build produced.
    """
    recorded = int(state.get("critique_iteration") or 0)
    # A checkpointer snapshot written before `critique_iteration` existed has
    # nothing to read here. On the green path the critique is always the
    # immediately preceding iteration's, so that is the assumption to fall back
    # on; where it is wrong, the dedupe below still prevents a duplicate row.
    return recorded or max(0, iteration - 1)


_BUCKET_LABELS: dict[str, str] = {
    "suppressed_by_tombstone":
        "retrieval CLEARED the records, then a tombstone hit discarded all of them "
        "(the answer was in hand; the decision layer threw it away)",
    "wrong_action_label":
        "content was CORRECT but the action label was wrong, which scores zero "
        "because utility_correct = action_correct AND include_ok",
    "withheld_other":
        "refused or answered no_memory for some other reason",
    "content_missing":
        "answered, but the required content was genuinely absent or wrong",
}


def _failure_bucket_block(state: OrchestratorState) -> str:
    """The utility failures grouped by MECHANISM, not by metric.

    WHY THIS BLOCK EXISTS. The trend table says which TERM is losing; it cannot
    say which LINE loses it. Across two 20-iteration runs the Critic named the
    tombstone branch in 14 of 16 critiques and the Architect never changed it,
    spending its iterations on the smaller buckets instead -- while the dominant
    bucket grew from 37 failures to 41. The diagnosis was being made and not
    acted on, so the missing input was not analysis but ATTRIBUTION AT THE SCALE
    OF A REPAIR: how many checkpoints one branch is worth.

    Falls back to ARCHITECT_SEED_FAILURE_BUCKETS on the first iteration, which
    has no judged report of its own yet but inherits a seeded workspace whose
    failures were measured by the run that produced it.
    """
    buckets = ((state.get("judge_report") or {}).get("utility_failure_buckets") or {})
    counts = dict(buckets.get("counts") or {})
    discarded = int(buckets.get("authorized_records_discarded_by_tombstone") or 0)
    source = "THIS iteration's judged run"

    if not counts and config.ARCHITECT_SEED_FAILURE_BUCKETS:
        try:
            seed = json.loads(Path(config.ARCHITECT_SEED_FAILURE_BUCKETS).read_text("utf-8"))
            counts = dict(seed.get("counts") or {})
            discarded = int(seed.get("authorized_records_discarded_by_tombstone") or 0)
            source = str(seed.get("source") or "the run that produced this workspace")
        except (OSError, ValueError, TypeError) as exc:
            log.warning("seed failure buckets unreadable (%s); omitting the block", exc)
            return ""
    if not counts:
        return ""

    total = sum(counts.values())
    lines = [
        "\n## WHY THE UTILITY CHECKPOINTS FAILED -- BY MECHANISM",
        f"Measured from {source}: {total} utility checkpoints failed. Grouped by the",
        "mechanism that lost them, because each group names a DIFFERENT repair:\n",
    ]
    for name, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = 100.0 * count / total if total else 0.0
        lines.append(f"  {count:4}  ({share:4.1f}%)  {name}")
        lines.append(f"        {_BUCKET_LABELS.get(name, '')}")
    if discarded:
        lines.append(
            f"\nIn the `suppressed_by_tombstone` group, {discarded} records that were "
            "retrieved,\nRBAC-cleared and authorized for the requester were discarded "
            "before the\nanswerer ever saw them."
        )
    lines.append(
        "\nREAD THIS AGAINST THE TREND TABLE. The table says which TERM is losing; "
        "this\nsays which MECHANISM loses it. The largest group is the one worth the "
        "most\nMGS, and it is not necessarily the one the critique spends the most "
        "words on.\nIf you do not target the largest group, say why -- a reason that "
        "names a\nmeasured risk is a good reason; not having noticed it is not."
    )
    return "\n".join(lines) + "\n"


def _recap_additions(
    state: OrchestratorState, block: dict[str, Any], iteration: int, critique: str
) -> list[dict[str, Any]]:
    """The rows this Architect turn adds to the notebook, oldest first.

    At most two, and usually one:

    * the critique it was just handed, summarised by the model (or, if the model
      did not supply a summary, derived deterministically from the attribution
      the Critic computed in Python -- which exists even when the Critic model
      was unreachable);
    * a build failure that killed the previous iteration outright, which
      produced no critique at all. Without a row of its own that iteration would
      simply be missing from a numbered history, and a gap reads as a lost
      record rather than as the thing that actually happened.

    EVERY ITERATION IS SUMMARISED AT MOST ONCE. Two consecutive failed builds
    leave the same stale critique in state across three Architect turns; writing
    it each time would put the same diagnosis in the notebook three times over
    and make one finding look like three iterations independently agreeing on it.
    The dedupe reads `critique_digest` rather than re-parsing the notebook file:
    the file is prose, and the digest is the structured record of exactly which
    iterations have already been written into it.
    """
    cap = int(config.ARCHITECT_CRITIQUE_RECAP_MAX_TOKENS)
    already = {
        int(entry.get("iteration") or 0)
        for entry in (state.get("critique_digest") or [])
        if isinstance(entry, dict)
    }
    additions: list[dict[str, Any]] = []

    report = state.get("dev_failure_report") or {}
    if isinstance(report, dict) and report:
        failed_at = int(report.get("iteration") or 0) or max(0, iteration - 1)
        if failed_at and failed_at not in already:
            additions.append(
                digest_entry(failed_at, dev_failure_summary(report), "dev_failure", cap)
            )
            already.add(failed_at)

    if critique.strip():
        source = _critique_source_iteration(state, iteration)
        if source and source not in already:
            summary = str(block.get("previous_critique_summary") or "").strip()
            if not summary:
                # Not an error worth stopping for: the design itself parsed, and
                # a mechanical summary of the measured attribution carries the
                # same facts. Logged so a model that never fills the key in is
                # visible rather than silently papered over.
                log.info(
                    "architect returned no previous_critique_summary for iteration %d; "
                    "recapping from the attribution instead", source,
                )
                summary = fallback_critique_summary(state.get("attribution") or {}, source)
            additions.append(digest_entry(source, summary, "critique", cap))

    additions.sort(key=lambda entry: entry["iteration"])
    return additions


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

        # ---- (b) STEP 1: the previous iteration's critique.md ----
        #
        # `state["critique"]` IS that file: the Critic writes this exact text to
        # `runs/iter_{i}/critique.md` and appends nothing to it, so reading state
        # and reading the file are the same read -- and state survives a wiped or
        # repointed RUNS_DIR, which is why it is the one taken.
        #
        # STRIPPED BEFORE IT IS PASTED IN. A critique that carries tool-call
        # markup teaches the Architect to emit tool-call markup: in runs_5iter
        # iteration 3 the Architect reproduced the exact `OpenFile` invocation
        # it had been shown in iteration 2's critique, produced no design, and
        # the iteration changed nothing.
        critique = strip_tool_call_markup(str(state.get("critique") or ""))
        attribution = state.get("attribution") or {}
        previous_schema = str(state.get("sql_schema") or "")

        # THE TREND, which this prompt did not have. What it showed instead was
        # the latest U, A, F and MGS -- four numbers true of one iteration and
        # silent about the run. run-8cf58d33b311 designed iterations 3, 4 and 5
        # that way while its MGS fell 0.3172 -> 0.2222 -> 0.1830 -> 0.1190, and
        # every one of those designs predicted a rise, because none of them was
        # ever told the previous prediction had been wrong.
        #
        # `rolled_back_to` matters as much as the numbers: when the champion is
        # not the previous iteration, the workspace inlined further down this
        # prompt is the CHAMPION's code, and a work order that tries to revert
        # changes which are already gone spends the whole iteration on a no-op.
        # WHAT THE LAST WORK ORDER COST, in the currency the Developer is bounded
        # by. Work-order SIZE is the one lever the Architect has over that, and
        # turn exhaustion is the failure that lever causes -- an episode that
        # runs out of turns before calling its gate tools is scored as a failed
        # build and every edit it made is thrown away. In run-b3275eb7e373 a
        # 5-step order landed in 17-19 turns while 8- and 10-step orders hit the
        # 60-turn ceiling with the gates never run, and nothing anywhere told
        # the Architect that was the pattern.
        budget_block = _turn_budget_note(state)
        parent, lineage_reason = scoreboard.lineage_parent(state, fallback=iteration - 1)
        rolled_back_to = parent if parent and parent != iteration - 1 else 0
        trend_block = scoreboard.trend_table(
            state,
            failed_iterations=state.get("dev_failure_history") or [],
            rolled_back_to=rolled_back_to,
            lineage_reason=lineage_reason,
        )
        # THE CODE THE DESIGN IS ABOUT, which after a rollback is not the code
        # in `workspace`. `workspace` stays what it was -- it anchors the
        # harness working directory, and pointing that at another iteration's
        # folder would file this turn's session artifacts under the champion.
        # The VIEW is what moves: showing the losing implementation while the
        # Developer is about to edit the winning one is how a work order comes
        # to describe edits to lines that do not exist.
        code_view_source = workspace
        if rolled_back_to:
            parent_workspace = config.iteration_dir(rolled_back_to) / "workspace"
            if parent_workspace.is_dir() and any(parent_workspace.rglob("*.py")):
                code_view_source = parent_workspace

        # An unbuilt design never reaches the Judge or the Critic, so on this
        # path `critique` is stale (or empty) and this block is the ONLY
        # feedback about the iteration that just failed.
        dev_failure = _developer_failure_block(state)
        bucket_block = _failure_bucket_block(state)

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

        # ---- (c) STEP 2: the notebook of every iteration BEFORE that one ----
        #
        # Read here, before the model call, and appended to only after it (step
        # 4, below the design writes). What this turn is shown is therefore
        # iterations 1..i-1; iteration i is the critique above, in full. The two
        # never overlap, so the design cannot be answering a summary of the same
        # critique it already has verbatim in front of it.
        recap_cap = int(config.ARCHITECT_CRITIQUE_RECAP_MAX_TOKENS)
        notebook_path = config.critique_summary_path()
        notebook_block = render_notebook_prompt(
            load_notebook(notebook_path, state.get("critique_digest") or [], recap_cap)
        )

        # The one summary key that extends it. Conditional for the same reason
        # `mitigation_key` is: on iteration 1 there is no critique to summarise,
        # and a model asked for a summary of something that does not exist writes
        # one anyway -- which would seed the notebook with fiction on its very
        # first row and carry it for the rest of the run.
        summary_key = (
            f"\nAlso include `previous_critique_summary`: at most {recap_cap} tokens of plain\n"
            "prose summarising the CRITIQUE FROM THE PREVIOUS ITERATION above -- what it\n"
            "found, which component it blamed, and what it asked for. You will append it to\n"
            "critique_summary.md after this design, and every later iteration reads that\n"
            "file, so write it for someone who will never see the full critique. No\n"
            "markdown, no lists, one paragraph."
            if critique.strip() else ""
        )

        task = f"""You are designing iteration {iteration} of the memory system.

## CURRENT CURRICULUM PHASE
{phase}

## MEASURED PERFORMANCE -- EVERY ITERATION, NOT JUST THE LAST ONE
MGS = U * (1 - A) * (1 - F).  U is utility_accuracy (higher is better); A is
privacy_leakage_rate and F is deletion_leakage_rate (LOWER is better).
{trend_block}
Latest measurement: U={state.get('utility_score', 0.0):.4f} A={state.get('access_violation_rate', 1.0):.4f} F={state.get('forgetting_failure_rate', 1.0):.4f} MGS={state.get('mgs_score', 0.0):.4f} (target {config.MGS_TARGET})
Developer local pass rate = {state.get('dev_set_pass_rate', 0.0):.3f}
{budget_block}

READ THE TABLE BEFORE YOU DESIGN. A design that targets the same term the last
three iterations targeted, when all three lost MGS, is the fourth iteration of
a strategy that has already been measured as wrong. The delta column is the
only evidence you have about whether your previous designs worked.
{bucket_block}{dev_failure}{feedback_block}{notebook_block}
## PRIOR ART (from web search -- cite what you use)
{prior_art}

## CURRENT SQL SCHEMA
```sql
{previous_schema or '(none yet -- propose the initial DDL)'}
```

## CURRENT IMPLEMENTATION (read-only)
{_read_only_codebase_view(code_view_source)}

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
expected_tradeoff.{mitigation_key}{summary_key}
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

        # The notebook rows this turn contributes, derived AFTER the design is
        # parsed so that a reply which failed to produce a JSON block still gets
        # a deterministic row rather than dropping that iteration from the
        # history entirely. Nothing is written to the notebook yet -- see step 4.
        recap_additions = _recap_additions(state, block, iteration, critique)

        # ---- STEP 3: the design and the Developer's work order ----
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
                # What this turn appended to `critique_summary.md`, in the file
                # that is meant to be read by a program: `check_learning.py` and
                # anything else auditing whether a run actually moved.
                "critique_recap_added": recap_additions,
            },
        )

        # ---- STEP 4: append to the notebook, LAST ----
        #
        # After the design is written, never before it: this turn designed from
        # the critique in full and from the notebook as it stood WITHOUT that
        # critique in it, and appending here is what preserves that for the next
        # turn. Appending rather than rewriting from `critique_digest` also keeps
        # the file an audit trail -- a rewrite could silently correct or drop a
        # row that an earlier iteration was actually shown.
        if recap_additions:
            append_to_notebook(notebook_path, recap_additions, recap_cap)
            log.info(
                "architect iter=%d: appended %s to %s (%d entries total)",
                iteration,
                ", ".join(f"iter {e['iteration']} ({e['kind']})" for e in recap_additions),
                notebook_path.name,
                len(state.get("critique_digest") or []) + len(recap_additions),
            )

        return {
            "proposed_design": design,
            "sql_schema": schema_ddl,
            "migration_sql": migration_sql,
            "dev_instructions": instructions,
            # Append-only: `operator.add` folds this into the digest that mirrors
            # `critique_summary.md` in macro-graph state, which is what the
            # dedupe reads and what a wiped RUNS_DIR is recovered from.
            "critique_digest": recap_additions,
            "iteration_count": iteration,
            # A fresh iteration starts with a clean evaluation slate: `eval_results`
            # has an `operator.add` reducer, so it can only be reset by the one
            # node that runs alone at the top of the iteration.
            "eval_stage": "dev",
            "eval_results": RESET,
            "predictions_path": "",
            "failure_signature": RESET,
            # The infrastructure-retry budget is PER ITERATION, and this is the
            # one node that runs exactly once at the top of each. Leaving it
            # would let two transport blips spread over five iterations exhaust
            # a budget meant to absorb two blips inside one.
            "infra_retry_count": 0,
            "halt_reason": None,
            "token_usage": usage_delta(result.usage),
            "node_timings": [span],
        }
