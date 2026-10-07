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


HISTORY_FILENAME = "notebook_history.md"

NOTEBOOK_PREAMBLE = (
    "The Architect's notebook. It is REWRITTEN every iteration, by the Architect,\n"
    "from the previous version plus what the latest round found -- so it is a\n"
    "curated set of standing notes, not a transcript. A note survives because the\n"
    "Architect judged it still worth knowing; one that has been superseded is\n"
    "meant to be dropped or rewritten rather than left to contradict a newer one.\n\n"
    "`notebook_history.md` beside it keeps every version, append-only, so what a\n"
    "given iteration was shown can still be reconstructed.\n"
)


def notebook_header() -> str:
    """The preamble, rewritten with the file each time."""
    return f"{NOTEBOOK_HEADING}\n\n{NOTEBOOK_PREAMBLE}"


def cap_words(text: str, max_words: int) -> str:
    """One note, trimmed to `max_words` whole words.

    Words rather than tokens because the budget is also stated to the model, and
    a model can approximate a word count while it cannot count BPE tokens. The
    cap is a backstop for a note that ignores the instruction, not the mechanism
    the budget is meant to work through.
    """
    words = " ".join(str(text or "").split()).split(" ")
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]) + " ..."


def cap_notes(notes: list[str], max_notes: int, max_words: int) -> list[str]:
    """The note list, bounded in both directions.

    WHOLE NOTES ARE DROPPED, NEVER HALF OF ONE. The previous scheme truncated
    each per-iteration entry at 100 estimated tokens, which cut 63% of them
    mid-sentence -- and because a summary runs diagnosis, then proposal, then
    what was decided, the clause that got cut was systematically the decision:
    exactly what the notebook exists to carry. Dropping a whole note at the tail
    loses one note and keeps every surviving one readable.
    """
    cleaned = [cap_words(note, max_words) for note in notes if str(note or "").strip()]
    return cleaned[:max_notes]


def render_notebook(notes: list[str]) -> str:
    """The notebook file, in full."""
    body = "\n".join(f"- {note}" for note in notes)
    return f"{notebook_header()}\n{body}\n" if body else ""


def read_notebook(path: Path) -> str:
    """The notebook as it stands, or "" when the run has not written one yet."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def notes_from_notebook(text: str) -> list[str]:
    """The note list parsed back out of the file.

    The file is the source of truth -- state is the fallback -- so a resumed run
    recovers its notes by reading what it wrote rather than by trusting a
    snapshot that may be older.
    """
    return [
        line.lstrip("-").strip()
        for line in (text or "").splitlines()
        if line.lstrip().startswith("- ")
    ]


def load_notes(path: Path, state_notes: list[str] | None) -> list[str]:
    """The notes the Architect is shown: the file, or state if it is gone."""
    from_file = notes_from_notebook(read_notebook(path))
    if from_file:
        return from_file
    return [str(n) for n in (state_notes or []) if str(n).strip()]


def write_notebook(path: Path, notes: list[str]) -> str:
    """Replace the notebook with `notes`, and return what was written.

    REFUSES TO WRITE AN EMPTY FILE. A model that returns no notes -- a timeout, a
    reply with no json block, a key it forgot -- must not be able to erase the
    run's memory, which under a rewrite-in-place scheme is a single bad response
    away. The caller keeps the previous notebook in that case.
    """
    if not notes:
        return ""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_notebook(notes)
    path.write_text(text, encoding="utf-8")
    return text


def append_history(path: Path, iteration: int, notes: list[str]) -> None:
    """Append this iteration's version of the notebook to the audit log.

    The notebook is now rewritten rather than appended to, which loses the
    property the old append-only file had for free: being able to see what a
    given iteration was actually shown. This log keeps it. Nothing reads it --
    it is for the post-mortem, and for telling a curation from a loss.
    """
    if not notes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "" if path.is_file() else (
        "# NOTEBOOK HISTORY\n\nEvery version of `critique_summary.md`, appended as it "
        "was written. The notebook itself is curated and rewritten; this is not.\n"
    )
    block = f"\n## after iteration {iteration}\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(header + block)


def render_notebook_prompt(notes: list[str], max_notes: int, max_words: int) -> str:
    """The notebook, and the instruction to curate it, as the Architect sees it.

    The instruction lives with the content on purpose: the Architect is being
    asked to return the NEXT version of this exact list, and the rules for doing
    that are unreadable away from the thing they apply to.
    """
    if notes:
        body = "\n".join(f"- {note}" for note in notes)
        current = (
            "\n## YOUR NOTEBOOK -- WHAT THIS RUN HAS LEARNED SO FAR\n"
            "You wrote this. It is the only memory that survives an iteration; the\n"
            "designs and critiques of earlier rounds are not shown to you again.\n\n"
            f"{body}\n"
        )
    else:
        current = (
            "\n## YOUR NOTEBOOK -- WHAT THIS RUN HAS LEARNED SO FAR\n"
            "Empty: this is the first iteration to write it.\n"
        )
    return current + (
        "\n### REWRITING IT\n"
        "Return `notebook_notes`: the FULL list as it should stand after this\n"
        "iteration -- not an addition to it. Whatever you leave out is forgotten.\n\n"
        f"- At most {max_notes} notes, each at most {max_words} words.\n"
        "- Keep a note while it still changes what a later iteration would do.\n"
        "- Rewrite a note that a newer measurement has refined; DROP one that has\n"
        "  been superseded or settled. Two notes that disagree cost you the reader.\n"
        "- Record what was TRIED and what it MEASURED, not only what was wrong: a\n"
        "  change that was made and did not help is the note that stops the run\n"
        "  making it again.\n"
        "- Write each note so it stands alone. A later iteration sees this list and\n"
        "  nothing else from this round.\n"
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
