"""The Architect's notebook: one iteration's memory of the ones before it.

WHY THIS EXISTS
---------------
`critique.md` is written fresh every iteration and nothing carried across them.
Iteration 4's Architect saw iteration 3's critique and had no idea what
iterations 1 and 2 had already tried, because the only record of those was two
files nobody opens again. That is how a loop re-proposes a change it made and
measured two iterations ago.

So the Architect keeps a notebook -- `runs/critique_summary.md` -- and it is the
node that both reads and writes it. Iteration i+1's Architect opens iteration
i's `critique.md`, opens the notebook summarising iterations 1..i-1, writes the
design and the Developer's work order from the two of them, and only THEN
appends its summary of iteration i's critique to the notebook. That order is the
whole point: the notebook a turn reads is the history BEFORE the critique it is
answering, so the critique in front of it and the summary of it never arrive as
one undifferentiated blob.

WHY THE ARCHITECT OWNS THE FILE
-------------------------------
It is the only node that reads a critique and writes the design answering it in
the same turn, so the summary comes back in the JSON block it is already
producing -- no extra model call, and no second reader that could disagree with
the first about what the critique said. The Critic writes `critique.md` and
nothing else; the notebook is not its file to keep.

WHY THE CAP IS PER-ENTRY AND NOT PER-FILE
-----------------------------------------
The notebook is appended to, never rewritten, so an uncapped entry makes both
the file and the Architect's prompt grow without bound.  One capped entry per
iteration makes that growth linear and predictable -- about
`ARCHITECT_CRITIQUE_RECAP_MAX_TOKENS` per iteration, ~1k tokens over a full
MAX_ITERATIONS=10 run.  Capping the notebook as a whole instead would force
every entry to shrink as the run went on, and the earliest iterations -- the
ones whose lessons are most likely to have been forgotten -- are exactly the
ones that squeezing would erase first.

WHAT GETS SUMMARISED IS THE RAW CRITIQUE, NOT THE NOTEBOOK
----------------------------------------------------------
`state["critique"]` is the Critic's own text, and it is byte-identical to the
`critique.md` the Critic wrote for that iteration.  Summarising the notebook
instead would mean re-summarising the previous summaries every iteration, and a
summary of a summary of a summary is where a feedback loop quietly stops
carrying information.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

# Two rules of thumb for one estimate.  See `approx_tokens`.
_CHARS_PER_TOKEN = 4.0
_TOKENS_PER_WORD = 1.3

NOTEBOOK_FILENAME = "critique_summary.md"
NOTEBOOK_HEADING = "# CRITIQUE SUMMARY"


def approx_tokens(text: str) -> int:
    """Estimated BPE tokens, without taking a tokenizer dependency.

    Two rules of thumb, and the LARGER of the two wins: ~4 characters per token
    (right for English prose, under-counts identifier-dense text) and ~1.3
    tokens per whitespace-separated word (right for long words, under-counts
    punctuation).  Neither is exact and this is not trying to be -- the cap
    exists to bound how fast the notebook grows, and a few tokens either way on
    one entry does not change that.  Taking the max rather than the mean puts
    the error on the side of a shorter entry, which is the side that cannot
    hurt.
    """
    body = str(text or "")
    if not body.strip():
        return 0
    return max(
        math.ceil(len(body) / _CHARS_PER_TOKEN),
        math.ceil(len(body.split()) * _TOKENS_PER_WORD),
    )


def cap_tokens(text: str, max_tokens: int) -> str:
    """One paragraph of at most `max_tokens` estimated tokens.

    Whitespace is collapsed first, deliberately: an entry is rendered as a
    single markdown list item, and a model that answers with a bulleted list of
    its own would otherwise break the list it is being placed inside.

    Truncation is by whole words, and the trailing marker is inside the budget
    rather than added after it -- an ellipsis is a token too.
    """
    body = " ".join(str(text or "").split())
    if not body or approx_tokens(body) <= max_tokens:
        return body

    words = body.split(" ")
    # Start from the word count the budget implies rather than from the top:
    # a full critique is ~2000 words and a 100-token cap keeps ~75 of them, so
    # counting down from `len(words)` would rebuild the string 1900 times for
    # no reason.
    keep = min(len(words), max(1, int(max_tokens / _TOKENS_PER_WORD)))
    while keep > 0:
        candidate = " ".join(words[:keep]) + " ..."
        if approx_tokens(candidate) <= max_tokens:
            return candidate
        keep -= 1
    return ""


def digest_entry(iteration: int, summary: str, kind: str, max_tokens: int) -> dict[str, Any]:
    """One capped notebook row.  `kind` is `critique` or `dev_failure`."""
    return {
        "iteration": int(iteration),
        "kind": str(kind),
        "summary": cap_tokens(summary, max_tokens),
    }


def _recap_lines(digest: list[dict[str, Any]]) -> list[str]:
    """The rows, oldest first, whatever order they were accumulated in."""
    rows = [entry for entry in (digest or []) if isinstance(entry, dict)]
    rows.sort(key=lambda entry: int(entry.get("iteration") or 0))
    lines: list[str] = []
    for entry in rows:
        kind = str(entry.get("kind") or "critique")
        label = "build failure" if kind == "dev_failure" else "critique"
        summary = str(entry.get("summary") or "").strip() or "(no summary recorded)"
        lines.append(f"- **iteration {entry.get('iteration', '?')}** ({label}): {summary}")
    return lines


# ======================================================================
# the notebook on disk -- `runs/critique_summary.md`
# ======================================================================
#
# APPEND-ONLY, and one file for the whole run rather than one per iteration.
# A per-iteration copy would be a rewrite of the same history N times over, and
# the thing the Architect actually needs -- "what has this run already found?"
# -- would then depend on knowing which iteration's copy is the newest.


def notebook_header(max_tokens: int) -> str:
    """The preamble written once, when the notebook is first created."""
    return (
        f"{NOTEBOOK_HEADING}\n\n"
        "The Architect's running notebook. One entry per iteration, appended by the\n"
        "Architect turn that read that iteration's `critique.md` -- so the entry for\n"
        f"iteration i is written at the start of iteration i+1 and capped at {max_tokens}\n"
        "estimated tokens, which makes this file grow linearly with the run.\n\n"
        "An iteration whose build never compiled produced no critique at all; it still\n"
        "gets a row, marked `build failure`, because a gap in a numbered history reads\n"
        "as a lost record rather than as the thing that actually happened.\n"
    )


def render_notebook_entries(entries: list[dict[str, Any]]) -> str:
    """The block appended to the notebook for the rows written this turn."""
    lines = _recap_lines(entries)
    return ("\n" + "\n".join(lines) + "\n") if lines else ""


def read_notebook(path: Path) -> str:
    """The notebook as it stands, or "" when the run has not written one yet."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def load_notebook(path: Path, digest: list[dict[str, Any]], max_tokens: int) -> str:
    """The notebook text the Architect is shown.

    Read from disk, because that file is the deliverable and reading anything
    else would let the two drift silently. `critique_digest` is the fallback
    and not the source: a resumed run, a moved `RUNS_DIR` or a wiped artifacts
    directory would otherwise cost the Architect the whole run's history, and
    the digest carries exactly the same rows in macro-graph state.
    """
    text = read_notebook(path)
    if text.strip():
        return text
    lines = _recap_lines(digest)
    if not lines:
        return ""
    return notebook_header(max_tokens) + "\n" + "\n".join(lines) + "\n"


def append_to_notebook(
    path: Path, entries: list[dict[str, Any]], max_tokens: int
) -> str:
    """Append this turn's rows, creating the file with its header if need be.

    Returns the text appended, "" when there was nothing to add.  Appending
    rather than rewriting is what makes the file an audit trail: a rewrite from
    `critique_digest` would silently correct or drop a row that an earlier
    iteration was actually shown.
    """
    block = render_notebook_entries(entries)
    if not block:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "" if path.is_file() else notebook_header(max_tokens)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(header + block)
    return header + block


def render_notebook_prompt(notebook: str) -> str:
    """The notebook, as the Architect sees it in its task text.

    Empty string when there is nothing yet: an empty section under a heading
    reads to a model as "this was checked and there was nothing", which is a
    different and more misleading claim than saying nothing at all.
    """
    body = (notebook or "").strip()
    if not body:
        return ""
    return (
        "\n## YOUR NOTEBOOK -- WHAT EARLIER ITERATIONS ALREADY FOUND (critique_summary.md)\n"
        "This is the file you keep. One line per earlier iteration, summarised at the time\n"
        "by the Architect turn that read it. These are things that have ALREADY been\n"
        "diagnosed and acted on -- do not re-propose a change an earlier iteration already\n"
        "made, and if a failure here keeps recurring, say why this design attacks it\n"
        "differently. It does NOT yet contain the critique above; you will add that after\n"
        "this design is written.\n\n"
        f"{body}\n"
    )


def fallback_critique_summary(attribution: dict[str, Any], iteration: int) -> str:
    """The deterministic summary, used when the model did not supply one.

    Built from `attribution`, which is computed in Python by the Critic and is
    therefore available even on the path where the Critic model was unreachable
    and the critique itself is the attribution-only fallback.  A mechanical
    summary of real numbers beats an empty entry, and beats asking a second
    model call for one.
    """
    if not isinstance(attribution, dict) or not attribution:
        return f"iteration {iteration}: a critique was produced but no attribution was recorded."
    mechanisms = ", ".join(str(m) for m in (attribution.get("observed_mechanisms") or [])) or "none recorded"
    component = str(attribution.get("component") or "unattributed")
    gain = attribution.get("dominant_gain")
    gain_text = f"{float(gain):+.4f}" if isinstance(gain, (int, float)) else "unknown"
    return (
        f"Dominant failing term {attribution.get('dominant_term', '?')} "
        f"(perfecting it was worth {gain_text} MGS), traced to {component}. "
        f"Observed mechanisms: {mechanisms}. "
        f"{len(attribution.get('proposals') or [])} proposal(s) were made."
    )


def dev_failure_summary(report: dict[str, Any]) -> str:
    """The notebook row for an iteration that never reached the Critic at all.

    A failed build produces no critique, so without this the notebook would
    simply skip that iteration number -- and a gap in a numbered history reads
    as a lost record rather than as the thing that actually happened.
    Deterministic on purpose: nothing here is inferred, it is the report's own
    fields.
    """
    gates = ", ".join(str(name) for name in (report.get("missing_gates") or [])) or "none"
    return (
        f"No critique: the Developer could not build the design "
        f"({report.get('reason', 'unknown')}). Unmet mandatory gates: {gates}. "
        f"Retries {report.get('retries_used', 0)}/{report.get('retry_cap', 0)}, "
        f"local pass rate {float(report.get('pass_rate', 0.0)):.3f}, "
        f"signature {report.get('signature') or '(none)'}."
    )
