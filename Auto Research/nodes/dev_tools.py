"""The Developer's tool belt.  Every one of these does real work.

No tool here is a stub: `compile_check` really parses the package,
`run_tests` really invokes pytest, `sql_exec` really applies the migration to a
scratch SQLite database, and `run_sandbox_smoke_test` really imports the agent
the Developer just wrote and runs it against sample medical checkpoints.
`mocks/sandbox.py` can override a *result* to force a failure path, but it never
replaces the tool.

Everything is path-scoped to the workspace.  `_resolve()` is the only way to
turn a model-supplied path into a real one, and it refuses to escape.
"""

from __future__ import annotations

import ast
import asyncio
import difflib
import json
import logging
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import config
from gatemem_adapter import failure_signature, normalize_traceback

log = logging.getLogger("orchestrator.devtools")

# ======================================================================
# The tool schema -- what the Developer is actually handed
# ======================================================================
#
# WHY THIS EXISTS (regression, runs_multi/run-280411c75b99). The Developer used
# to be handed NO tool schema at all: its persona described ten tools in prose
# and the loop parsed one fenced JSON block out of the reply. A tool-calling
# model given no tools does not fall back to prose -- it emits its native
# tool-call syntax as plain text. That run produced 27 such replies, parsed
# zero of them, wrote zero bytes, and evaluated byte-identical code in all four
# iterations. `parse_xml_action` in nodes/developer.py was the salvage; this is
# the fix.
#
# It is also the answer to the OTHER half of that failure (see the note on
# DEVELOPER_PROFILE.capabilities in harness/profiles.py): mounting the harness's
# own fs/bash plugins gave the model a SECOND toolbox whose names did not match
# the persona's, and it worked in the surface it could see rather than the one
# the gates observe. The conclusion there -- "an agent handed two toolboxes uses
# the one in its tool schema, not the one in its prose" -- is exactly why the
# ten real tools belong IN the schema and nothing else does.
#
# So: one toolbox, declared once, in the place the model actually reads. These
# schemas are the source of truth for TOOL_NAMES below, so a tool cannot be
# advertised without existing, or exist without being advertised.

_PATH_DESC = (
    "Path relative to the workspace root, e.g. 'memory_system/store.py'. "
    "A leading '/' is stripped, never resolved against the real filesystem root."
)

TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file from the workspace, one window of lines at a time. Returns up "
                "to 220 lines starting at `offset`, and ends with a footer giving the line "
                "range, the file's total line count, and the exact call that fetches the "
                "next window. READ THE WHOLE FILE BEFORE EDITING IT: apply_patch matches "
                "text exactly, so a search block recalled from a partial read will not "
                "match. Never dump source to disk from a test to work around this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": _PATH_DESC},
                    "offset": {
                        "type": "integer",
                        "description": "1-based first line to return. Default 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "How many lines to return. Default and maximum 220.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a workspace directory. Defaults to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": _PATH_DESC}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            # The guard is described here rather than only in the persona,
            # because this is the text the model reads while choosing arguments.
            "description": (
                "Write a file, creating parent directories as needed. Use this for NEW "
                "files. Overwriting an existing file of 2000+ characters with one under "
                "75% of its length is REFUSED -- rewriting from memory loses code, and "
                "the next iteration inherits the loss. Use apply_patch to edit existing "
                "files; pass force=true only for a deliberate full rewrite."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": _PATH_DESC},
                    "content": {"type": "string", "description": "The complete file text."},
                    "force": {
                        "type": "boolean",
                        "description": "Bypass the destructive-rewrite guard. Deliberate rewrites only.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Edit an existing file, either as a search/replace pair or a unified diff. "
                "The right tool for changing code you did not just write: context is "
                "verified before anything is written, so a hunk that does not match "
                "leaves the file untouched."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": _PATH_DESC},
                    "search": {
                        "type": "string",
                        "description": "The exact existing lines to replace. Pair with `replace`.",
                    },
                    "replace": {"type": "string", "description": "The new lines. Pair with `search`."},
                    "diff": {
                        "type": "string",
                        "description": "A unified diff, as an alternative to search/replace.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compile_check",
            "description": "ast.parse every .py file in the workspace. Sets the compile_ok gate.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run pytest in the workspace. Sets the tests_ok gate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Test target, relative to the workspace. Defaults to 'tests'.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_linter",
            "description": "Run ruff over the workspace. ADVISORY: a finding here blocks nothing.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sql_exec",
            "description": (
                "Apply memory_system/schema.sql plus this iteration's migration to a scratch "
                "SQLite database. Sets the migration_ok gate. With no arguments it runs the "
                "ARCHITECT's migration; pass migration_sql to override it when that migration "
                "is itself wrong (a `CREATE TABLE t (...)` placeholder, or DDL schema.sql "
                "already creates). An empty string means 'no migration needed', which is a "
                "valid answer and still sets the gate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "migration_sql": {
                        "type": "string",
                        "description": "Full replacement migration script, or '' for none.",
                    }
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sandbox_smoke_test",
            "description": (
                "Import the agent you just wrote in a child process and run it against sample "
                "medical checkpoints. Sets the advisory smoke_ok gate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "description": "How many checkpoints to run. Default 3."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Declare the work order complete. REFUSED unless compile_ok, tests_ok and "
                "migration_ok are all set by their tools having actually run in this episode; "
                "a rejected finish costs a retry."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What you built and proved."}
                },
                "required": ["summary"],
            },
        },
    },
)

# Derived, never written twice: a tool cannot be advertised without existing,
# or exist without being advertised.
TOOL_NAMES = frozenset(schema["function"]["name"] for schema in TOOL_SCHEMAS)


def tool_schemas() -> list[dict[str, Any]]:
    """A fresh, mutable copy of the schema list, for the wire payload."""
    import copy

    return [copy.deepcopy(schema) for schema in TOOL_SCHEMAS]


@dataclass
class ToolResult:
    ok: bool
    tool: str
    stdout: str = ""
    stderr: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    stack_trace: str = ""
    # A failure that must NOT be charged against MAX_DEV_RETRIES.
    #
    # The retry cap exists to bound "this design cannot be built" (see
    # nodes/developer.py). A tool that refuses an action and names the right one
    # instead is not evidence about the design -- it is the toolbox redirecting a
    # call, and the very next action can be the correct one. Charging it retires
    # the episode for a reason that has nothing to do with the Architect's spec,
    # which is the same distinction DEV_THINK_RESAMPLES draws for a decoder
    # stutter. Used by `write_file`'s destructive-rewrite guard.
    redirect: bool = False

    # How much of this result reaches the scratchpad, and which end survives.
    #
    # WHY THIS IS PER-RESULT (runs_4iter, 2026-08-29). The cap and the direction
    # used to be fixed for every tool: keep the last 2400 characters. That is
    # right for a traceback -- the exception line is at the bottom -- and
    # catastrophic for a file read, where the answer is at the TOP and 2400
    # characters is a tenth of `store.py`. Every one of the 66 `read_file` calls
    # in that run's first episode returned the same head-truncated slice, so the
    # Developer never once saw the imports or the first 5000 characters of the
    # file it was employed to edit. It responded by writing pytest files that
    # `assert False` after dumping 1500-character chunks of source to disk --
    # using the test runner as a file reader, and failing the `tests_ok` gate
    # every time it did. A file read keeps its head and gets room to be useful.
    max_observation_chars: int | None = None
    keep: str = "tail"  # "tail" for diagnostics, "head" for file content

    #: Default scratchpad budget for a result that does not ask for its own.
    DEFAULT_OBSERVATION_CHARS = 2400

    def observation(self, max_chars: int | None = None) -> str:
        """Render for the Developer's scratchpad, truncated at one end.

        Diagnostics truncate from the *front*: cutting the head off a pytest run
        would lose the exception line, which is the only part the next Thought
        actually needs. File content truncates from the *back*, because the top
        of a file is where its imports and definitions are.
        """
        budget = max_chars or self.max_observation_chars or self.DEFAULT_OBSERVATION_CHARS
        body = self.stdout or self.stderr or json.dumps(self.data, default=str)
        if len(body) > budget:
            if self.keep == "head":
                body = body[:budget] + "\n[... truncated: ask for the rest with `offset` ...]"
            else:
                body = "[... truncated ...]\n" + body[-budget:]
        return f"{'OK' if self.ok else 'FAILED'} {self.tool}: {body}"

    def signature(self) -> str | None:
        """Normalized failure signature for the circuit breaker."""
        if self.ok:
            return None
        return signature_from_trace(self.stack_trace or self.stderr or self.stdout, self.tool)


def signature_from_trace(trace: str, category: str = "") -> str:
    """exception type + top frame + category -> stable hash.

    The three parts are extracted rather than hashing the whole trace so that
    the same bug hit from two shards -- different line numbers, different
    checkpoint ids -- collapses to one signature.  That collapse is what makes
    the fail-fast breaker fire instead of counting six distinct failures.
    """
    exc_type = ""
    top_frame = ""
    for line in reversed((trace or "").strip().splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if not exc_type and re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Warning)\b", stripped):
            exc_type = stripped.split(":")[0]
        if stripped.startswith("File ") and not top_frame:
            match = re.search(r'File "([^"]+)".*in (\S+)', stripped)
            if match:
                top_frame = f"{Path(match.group(1)).name}:{match.group(2)}"
    if not exc_type:
        exc_type = normalize_traceback(trace)[:80]
    return failure_signature(exc_type, top_frame, category)


#: `assert False` / `assert 0` with no condition attached. `assert False, "msg"`
#: counts; `assert x is False` does not.
_UNCONDITIONAL_ASSERT_FALSE = re.compile(r"^\s*assert\s+(False|0)\s*(,|$)", re.MULTILINE)


def _looks_like_test(rel_path: str) -> bool:
    """Would pytest, or the Developer, run this as part of the suite?"""
    parts = Path(rel_path).parts
    name = Path(rel_path).name
    return "tests" in parts or name.startswith("test_") or name.endswith("_test.py")


def is_scratch(rel_path: str) -> bool:
    """Is this the Developer's own working note rather than the deliverable?

    One leading underscore marks scratch (`_store_chunk_3.txt`, `_dump_store.py`,
    `tests/_peek_store.py`); a dunder does not (`memory_system/__init__.py`).
    Used both to keep scratch out of the change guard and to keep it out of the
    next iteration's workspace -- runs_4iter carried 40-odd dump files and two
    source-dumping pytest modules from iteration 1 into every later iteration.
    """
    name = Path(rel_path).name
    return name.startswith("_") and not name.startswith("__")


def _positive_int(raw: Any, default: int) -> int:
    """A model-supplied count, coerced. Junk falls back rather than raising."""
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _ruff_command() -> list[str] | None:
    """Locate ruff the way this project is actually launched.

    `shutil.which` alone searches PATH, and the documented way to start this
    orchestrator is `.venv/bin/python main.py`, which does NOT put `.venv/bin`
    on PATH. So every run in this repository reported

        run_linter: no findings (fallback linter: ruff not installed)

    while ruff sat installed in the venv beside the interpreter running the
    check -- a green advisory gate standing on evidence nobody collected. Look
    next to `sys.executable` first, then PATH, then the module.
    """
    beside = Path(sys.executable).with_name("ruff")
    if beside.is_file() and os.access(beside, os.X_OK):
        return [str(beside)]
    found = shutil.which("ruff")
    if found:
        return [found]
    import importlib.util

    if importlib.util.find_spec("ruff") is not None:
        return [sys.executable, "-m", "ruff"]
    return None


class DevToolbox:
    """Workspace-scoped tools for one Developer loop."""

    #: Lines returned by one `read_file` call when `limit` is not given. Chosen
    #: so the 532-line template `store.py` is four calls, not sixteen chunk
    #: files smuggled out through a deliberately-failing test.
    READ_FILE_LINES = 220
    #: Hard character cap on one window, so a minified line cannot blow context.
    READ_FILE_MAX_CHARS = 14_000

    def __init__(self, workspace: Path, migration_sql: str = "", iteration: int = 1) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.migration_sql = migration_sql
        self.iteration = iteration
        self.files_written: set[str] = set()
        #: Set when the Developer supplies its own migration through `sql_exec`.
        self.migration_overridden = False
        self._baseline = self._load_or_write_baseline()

    # ------------------------------------------------------------------
    # did this episode actually change anything?
    # ------------------------------------------------------------------
    #
    # WHY THIS EXISTS (runs_4iter, iteration 2). The Developer inherited
    # iteration 1's workspace, ran compile_check, run_tests, sql_exec and the
    # smoke test on it, saw four green gates, and stopped -- having written zero
    # bytes. `_stop_reason` returned "gates_green", the orchestrator recorded a
    # successful episode, and the Judge scored code that was byte-identical to
    # the previous iteration's. The run's ONE real measurement therefore measured
    # nothing new, and the loop looked healthy while doing it.
    #
    # Green gates on unchanged inherited code prove only that the PREVIOUS
    # iteration compiled. A snapshot taken at episode start is the only way to
    # tell "built it and it passes" from "ran the gates and left".
    #
    # The baseline lives beside the workspace rather than inside it: it must not
    # be inherited by iteration N+1, linted, or collected by pytest.

    #: Files that are workspace scaffolding rather than the Developer's work.
    BASELINE_IGNORE: ClassVar[frozenset[str]] = frozenset(
        {"_smoke_runner.py", "_eval_runner.py", "ruff.toml"}
    )

    def _source_digests(self) -> dict[str, str]:
        import hashlib

        digests: dict[str, str] = {}
        for path in sorted(self.workspace.rglob("*")):
            if not path.is_file() or path.name in self.BASELINE_IGNORE:
                continue
            if any(part in {"__pycache__", ".pytest_cache", ".ruff_cache"} for part in path.parts):
                continue
            rel = str(path.relative_to(self.workspace))
            digests[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
        return digests

    def _load_or_write_baseline(self) -> dict[str, str]:
        """The workspace as it was handed over, cached across toolbox rebuilds."""
        marker = self.workspace.parent / "workspace_baseline.json"
        if marker.is_file():
            try:
                return dict(json.loads(marker.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                pass  # unreadable snapshot: re-take it rather than fail the episode
        digests = self._source_digests()
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(digests, indent=2, sort_keys=True), encoding="utf-8")
        except OSError:
            log.warning("could not persist workspace baseline beside %s", self.workspace)
        return digests

    def changed_files(self) -> list[str]:
        """Workspace files whose content differs from the episode's starting state."""
        now = self._source_digests()
        return sorted(
            set(now) ^ set(self._baseline)
            | {rel for rel in set(now) & set(self._baseline) if now[rel] != self._baseline[rel]}
        )

    def substantive_changes(self) -> list[str]:
        """Changed files that are actually the deliverable.

        Scratch notes and dumped text do not count: the guard asks whether the
        episode built anything, and `_store_chunk_7.txt` is not a build.
        """
        return [
            rel for rel in self.changed_files()
            if not is_scratch(rel) and Path(rel).suffix in {".py", ".sql"}
        ]

    def built_anything(self) -> bool:
        """Did this episode produce work, in any form the loop carries forward?

        Usually a changed source file. A migration the Developer corrected via
        `sql_exec` counts too: it is the deliverable for a work order whose only
        defect was in the Architect's DDL, and it reaches the next node through
        state rather than through the workspace.
        """
        return bool(self.substantive_changes()) or self.migration_overridden

    # ------------------------------------------------------------------
    # path safety
    # ------------------------------------------------------------------

    def _resolve(self, rel_path: str) -> Path:
        """Resolve inside the workspace, or raise.

        The Developer's `dsh` profile already pins `fs-local` and `bash-local`
        to the workspace, but this loop also calls tools in-process, so the
        guard is repeated here.  Two cheap checks beat one leaked write.
        """
        candidate = (self.workspace / str(rel_path).lstrip("/")).resolve()
        # `is_relative_to`, not `str.startswith`: with a workspace of
        # `/runs/iter_1/workspace`, the string test also accepts
        # `/runs/iter_1/workspace_old/...` -- a different directory that merely
        # shares a prefix. Path semantics compare components, not characters.
        if not candidate.is_relative_to(self.workspace):
            raise ValueError(f"path escapes workspace: {rel_path!r}")
        return candidate

    # ------------------------------------------------------------------
    # dispatch
    # ------------------------------------------------------------------

    async def call(self, tool: str, args: dict[str, Any]) -> ToolResult:
        if tool not in TOOL_NAMES:
            return ToolResult(False, tool, stderr=f"unknown tool {tool!r}; valid: {sorted(TOOL_NAMES)}")
        handler = getattr(self, f"_t_{tool}")
        try:
            result: ToolResult = await handler(args or {})
        except Exception:  # noqa: BLE001 - a tool crash is an observation, not a graph failure
            trace = traceback.format_exc()
            log.warning("tool %s raised: %s", tool, trace.splitlines()[-1])
            return ToolResult(False, tool, stderr=trace, stack_trace=trace)

        # Scripted failure injection (mock scenarios only).
        from mocks import sandbox

        if config.MOCK_MODE:
            override = sandbox.override(tool, self.iteration, real_ok=result.ok)
            if override is not None:
                return ToolResult(
                    override.ok, tool, stdout=override.stdout,
                    stderr=override.stderr, stack_trace=override.stack_trace,
                    data={"scripted": True},
                )
        return result

    # ------------------------------------------------------------------
    # tools
    # ------------------------------------------------------------------

    async def _t_read_file(self, args: dict[str, Any]) -> ToolResult:
        if not str(args.get("path") or "").strip():
            # Say what actually went wrong. Reporting the missing value as
            # "no such file: None" reads like the FILE is missing and sends the
            # model off re-checking a path that was never the problem.
            return ToolResult(
                False, "read_file",
                stderr=f'read_file requires args.path, e.g. {{"tool": "read_file", '
                       f'"args": {{"path": "memory_system/store.py"}}}}; got args={args!r}',
            )
        path = self._resolve(args.get("path", ""))
        if not path.is_file():
            return ToolResult(False, "read_file", stderr=f"no such file: {args.get('path')}")
        rel = str(args.get("path"))
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        total = len(lines)

        offset = _positive_int(args.get("offset"), default=1)
        offset = max(1, min(offset, total or 1))
        limit = _positive_int(args.get("limit"), default=self.READ_FILE_LINES)
        limit = max(1, min(limit, self.READ_FILE_LINES))

        window = lines[offset - 1: offset - 1 + limit]
        body = "".join(window)
        # A character cap still applies -- one 40000-character minified line
        # would otherwise blow the context -- but it is per-window and generous,
        # and it trims the TAIL so the window still starts where it was asked to.
        clipped = len(body) > self.READ_FILE_MAX_CHARS
        if clipped:
            body = body[: self.READ_FILE_MAX_CHARS]
            window = body.splitlines(keepends=True)
        last = offset + len(window) - 1

        footer = f"\n--- {rel}: lines {offset}-{last} of {total} ({len(text)} chars total)"
        if last < total:
            footer += (
                f'. NOT THE WHOLE FILE. Read the next part with '
                f'{{"tool": "read_file", "args": {{"path": "{rel}", "offset": {last + 1}}}}}'
            )
        else:
            footer += ". End of file."
        if clipped:
            footer += " (window trimmed at the character cap)"
        footer += " ---"

        return ToolResult(
            True, "read_file", stdout=body + footer,
            data={"chars": len(text), "lines": total, "offset": offset,
                  "last_line": last, "complete": last >= total},
            # File content: keep the head, and give it room. See ToolResult.keep.
            max_observation_chars=self.READ_FILE_MAX_CHARS + len(footer) + 64,
            keep="head",
        )

    async def _t_list_dir(self, args: dict[str, Any]) -> ToolResult:
        # Unlike read_file, "." is a sensible default here, so a missing path is
        # not an error -- it lists the workspace root.
        path = self._resolve(args.get("path") or ".")
        if not path.is_dir():
            return ToolResult(False, "list_dir", stderr=f"not a directory: {args.get('path')}")
        entries = sorted(
            str(p.relative_to(self.workspace)) + ("/" if p.is_dir() else "")
            for p in path.iterdir()
            if p.name not in {"__pycache__", ".sessions", ".cordis"}
        )
        return ToolResult(True, "list_dir", stdout="\n".join(entries), data={"n": len(entries)})

    # A `write_file` over an existing file this size or larger is checked for
    # destructive shrinkage. Below it, a full rewrite is cheap to re-derive and
    # the guard would only be noise.
    REWRITE_GUARD_MIN_CHARS = 2_000
    # ...and it is refused if the replacement keeps less than this fraction.
    REWRITE_GUARD_MIN_RATIO = 0.75

    async def _t_write_file(self, args: dict[str, Any]) -> ToolResult:
        """Write a file, refusing a full rewrite that would silently lose content.

        WHY THE GUARD EXISTS (observed, runs_verify3). The Developer replaced the
        22274-character `store.py` with a 13319-character regeneration -- one
        `write_file` call against five `apply_patch` calls in the same episode.
        It was not trying to delete anything; it simply cannot reproduce 22k
        characters from memory, so the rewrite quietly dropped `author_id` from
        the INSERT (every test then died on `NOT NULL constraint failed`) along
        with the `DEFAULT_ROLE_GRANTS` ancillary-care-team rows that an earlier
        run had added to fix 9 of 18 utility failures. Four follow-up patches
        chased the symptom and never restored either.

        The damage also COMPOUNDS: `prepare_workspace` seeds iteration N from
        iteration N-1, so a destructive rewrite is inherited by every later
        iteration. That is the opposite of the lineage the loop is built on.

        This is enforced structurally rather than by asking the prompt nicely,
        for the reason the privilege model is (README section 4): an instruction
        not to lose code is not a mechanism. `apply_patch` already exists and is
        the right tool -- it verifies context before writing, so a hunk that does
        not match leaves the file untouched. A genuine rewrite is still possible
        with an explicit `force`, which makes intent visible in the scratchpad
        instead of indistinguishable from an accident.
        """
        rel = str(args.get("path", ""))
        content = str(args.get("content", ""))
        if not rel:
            return ToolResult(False, "write_file", stderr="write_file requires `path`")
        path = self._resolve(rel)

        # WHY A TEST MAY NOT ASSERT FALSE (runs_4iter, iteration 1). Unable to
        # see past the first slice of `store.py`, the Developer wrote pytest
        # modules that dumped 1500-character chunks of source to disk and then
        # `assert False`ed so the runner would surface them. It was using the
        # test runner as a file reader -- and every such run failed the
        # `tests_ok` gate, so the workaround for one blindness spent the retry
        # budget of the whole episode. 66 of its 86 turns went on reading.
        #
        # `read_file` now pages, so the workaround has no reason to exist. This
        # is a redirect, not a failure: the point is to hand back the tool that
        # actually answers the question, not to end the episode.
        if _UNCONDITIONAL_ASSERT_FALSE.search(content) and _looks_like_test(rel):
            return ToolResult(
                False, "write_file",
                stderr=(
                    f"refusing to write {rel}: it contains a bare `assert False`, which "
                    f"makes the test suite fail on purpose and clears the tests_ok gate.\n"
                    f"If you are doing this to see a file's contents, you do not need to. "
                    f'`read_file` pages: {{"tool": "read_file", "args": '
                    f'{{"path": "memory_system/store.py", "offset": 221}}}} returns the next '
                    f"220 lines, and every response ends with the exact call for the window "
                    f"after it.\n"
                    f"If you really do want a failing test, assert the condition you mean."
                ),
                data={"path": rel},
                redirect=True,
            )

        forced = bool(args.get("force") or args.get("replace"))
        if not forced and path.is_file():
            existing = path.read_text(encoding="utf-8", errors="replace")
            keeps = len(content) / len(existing) if existing else 1.0
            if len(existing) >= self.REWRITE_GUARD_MIN_CHARS and keeps < self.REWRITE_GUARD_MIN_RATIO:
                return ToolResult(
                    False, "write_file",
                    stderr=(
                        f"refusing to overwrite {rel}: it is {len(existing)} chars and the "
                        f"replacement is {len(content)} ({keeps:.0%}). A rewrite that small "
                        f"almost always means content was reproduced from memory and lost -- "
                        f"which then compounds, because the next iteration starts from this "
                        f"file.\n"
                        f'Use apply_patch to change only what you mean to change: '
                        f'{{"tool": "apply_patch", "args": {{"path": "{rel}", '
                        f'"search": "<the exact lines to replace>", '
                        f'"replace": "<the new lines>"}}}}\n'
                        f'If you really do intend to replace the whole file, re-send this '
                        f'write_file with "force": true.'
                    ),
                    data={"path": rel, "existing_chars": len(existing),
                          "proposed_chars": len(content)},
                    # Redirection, not a build failure: the model is being told
                    # to reach for apply_patch, and its next action can be the
                    # right one. Charging this against MAX_DEV_RETRIES would let
                    # a guard that exists to PROTECT the episode end it instead.
                    redirect=True,
                )

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        self.files_written.add(rel)
        return ToolResult(
            True, "write_file", stdout=f"wrote {rel} ({len(content)} chars)",
            data={"path": rel, "chars": len(content)},
        )

    async def _t_apply_patch(self, args: dict[str, Any]) -> ToolResult:
        """Apply a unified diff, or a search/replace block.

        A hand-rolled hunk applier rather than shelling out to `patch(1)`,
        which is not guaranteed to exist and whose fuzz behaviour is a
        portability hazard.  Context lines are verified before anything is
        written, so a hunk that does not match leaves the file untouched.
        """
        rel = str(args.get("path", ""))
        if not rel:
            return ToolResult(False, "apply_patch", stderr="apply_patch requires `path`")
        path = self._resolve(rel)
        original = path.read_text(encoding="utf-8") if path.is_file() else ""

        if "search" in args and "replace" in args:
            search, replace = str(args["search"]), str(args["replace"])
            if search in original:
                updated = original.replace(search, replace, 1)
            else:
                fuzzy = _match_ignoring_whitespace(original, search)
                if fuzzy is not None:
                    start, end = fuzzy
                    updated = original[:start] + replace + original[end:]
                    log.info("apply_patch: %s matched on whitespace-normalized text", rel)
                else:
                    # WHY THE ERROR CARRIES THE REAL TEXT (runs_4iter, iteration 3).
                    # A bare "search block not found" tells the model nothing it
                    # did not already know, so it guesses again. That episode sent
                    # the SAME two-line search block four times; the file had a
                    # third line between them (`term_hashes = _term_hashes(query)`)
                    # that the Developer had never seen, because read_file only
                    # ever showed it a 2400-character slice. It burned the retry
                    # cap without ever writing the fix. Showing the closest actual
                    # lines turns a dead end into a block it can copy verbatim.
                    return ToolResult(
                        False, "apply_patch",
                        stderr=_near_miss_report(original, search, rel),
                        data={"path": rel, "matched": False},
                        # File content, so keep the head: the near-miss block is
                        # the point and it is printed first.
                        max_observation_chars=4000, keep="head",
                    )
        else:
            diff = str(args.get("diff") or args.get("patch") or "")
            if not diff.strip():
                return ToolResult(False, "apply_patch", stderr="apply_patch requires `diff` or search/replace")
            try:
                updated = _apply_unified_diff(original, diff)
            except ValueError as exc:
                return ToolResult(False, "apply_patch", stderr=str(exc))

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(updated, encoding="utf-8")
        self.files_written.add(rel)
        return ToolResult(True, "apply_patch", stdout=f"patched {rel}", data={"path": rel})

    async def _t_compile_check(self, args: dict[str, Any]) -> ToolResult:
        """`ast.parse` every .py in the workspace.  The first gate, always."""
        errors: list[str] = []
        checked = 0
        for path in sorted(self.workspace.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            checked += 1
            try:
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                errors.append(
                    f'  File "{path}", line {exc.lineno}, in <module>\n'
                    f"    {(exc.text or '').strip()}\n"
                    f"SyntaxError: {exc.msg}"
                )
        if errors:
            trace = "Traceback (most recent call last):\n" + "\n".join(errors)
            return ToolResult(False, "compile_check", stderr=trace, stack_trace=trace,
                              data={"checked": checked, "errors": len(errors)})
        return ToolResult(True, "compile_check", stdout=f"parsed {checked} file(s) cleanly",
                          data={"checked": checked})

    async def _t_run_tests(self, args: dict[str, Any]) -> ToolResult:
        target = str(args.get("path", "tests"))
        # WHY THE GATE NEEDS THE WHOLE SUITE (runs_4iter). `tests_ok` used to be
        # set by whatever pytest last ran, so a single narrow file could set it
        # -- and a single deliberately-failing scratch file could clear it, which
        # is how iteration 1 spent its retry budget. The gate means "the suite
        # passes", so only a run of the suite may set it. A narrow run is still
        # allowed and still useful while iterating on one failure; it just does
        # not decide the gate.
        full_suite = target.strip().rstrip("/") in {"", ".", "tests"}
        code, out, err = await _run(
            [sys.executable, "-m", "pytest", target, "-q", "--no-header", "-p", "no:cacheprovider"],
            cwd=self.workspace, timeout=300,
        )
        note = "" if full_suite else (
            f"\n[note] `{target}` is part of the suite, not the suite, so this run does "
            f'NOT set tests_ok. Run {{"tool": "run_tests", "args": {{"path": "tests"}}}} '
            f"once the failure you are chasing is fixed."
        )
        if code == 5:  # pytest's "no tests collected"
            return ToolResult(False, "run_tests", stdout=out, stderr="no tests were collected" + note,
                              data={"passed": 0, "failed": 0, "pass_rate": 0.0,
                                    "full_suite": full_suite})
        passed, failed = _parse_pytest_counts(out + err)
        total = passed + failed
        return ToolResult(
            code == 0, "run_tests", stdout=out[-6000:] + note, stderr=err[-2000:],
            stack_trace=out if code != 0 else "",
            data={"passed": passed, "failed": failed,
                  "pass_rate": (passed / total) if total else 0.0,
                  "full_suite": full_suite},
        )

    async def _t_run_linter(self, args: dict[str, Any]) -> ToolResult:
        """`ruff` if present, otherwise a real (if smaller) AST-based check.

        A linter that silently no-ops when the binary is missing would let the
        Developer's exit condition pass on evidence that was never collected,
        so the fallback does actual work rather than returning OK.
        """
        ruff = _ruff_command()
        if ruff:
            # `--exclude` is passed explicitly as well as being set in the
            # workspace ruff.toml: if the Developer has not written the config
            # yet, the orchestrator's own scaffolds would otherwise dominate the
            # findings with issues the Developer cannot act on.
            code, out, err = await _run(
                [*ruff, "check", ".", "--exclude", "_smoke_runner.py,_eval_runner.py"],
                cwd=self.workspace, timeout=120,
            )
            return ToolResult(code == 0, "run_linter", stdout=out[-4000:], stderr=err[-2000:],
                              data={"linter": "ruff"})
        findings = _fallback_lint(self.workspace)
        return ToolResult(
            not findings, "run_linter",
            stdout="\n".join(findings) or "no findings (fallback linter: ruff not installed)",
            data={"linter": "fallback", "findings": len(findings)},
        )

    async def _t_sql_exec(self, args: dict[str, Any]) -> ToolResult:
        """Create the schema and apply the migration on a scratch database.

        Scratch, not the run database: a migration that half-applies must not
        be able to corrupt a store the evaluator is about to use.
        """
        schema_path = self._resolve("memory_system/schema.sql")
        if not schema_path.is_file():
            return ToolResult(False, "sql_exec", stderr="memory_system/schema.sql does not exist yet")
        # KEY PRESENCE, not truthiness. `{"migration_sql": ""}` is the documented
        # way to say "this iteration needs no migration", and treating the empty
        # string as absent would silently fall back to the Architect's script --
        # re-running the very statements the Developer just declared unnecessary,
        # so the gate could never be cleared that way.
        overridden = "migration_sql" in args
        override = str(args.get("migration_sql") or "")
        if overridden and override.strip():
            # A corrected migration is a deliverable even though it is not a
            # workspace file, so the did-this-episode-build-anything guard has
            # to be able to see it. See DevToolbox.substantive_changes.
            #
            # AN EMPTY OVERRIDE IS NOT ONE. `{"migration_sql": ""}` is this
            # tool's documented way to say "this iteration needs no migration"
            # (see the comment above), and counting it as work reopened the
            # exact hole test_no_op_episode.py was written to close: in
            # runs_smoke the Developer read the workspace, called `sql_exec`
            # with an empty override, cleared every gate and exited
            # `gates_green` having written zero bytes -- and `built_anything()`
            # said True, so neither door stopped it.
            self.migration_overridden = True
        migration = override if overridden else str(self.migration_sql or "")
        # WHERE THE SQL CAME FROM is the single most useful fact in a failure
        # here, and the model cannot see it. The migration is normally the
        # ARCHITECT's, not anything in the workspace, so a bare "table X already
        # exists" sends the Developer hunting through schema.sql for a bug that
        # is not there -- and it has no way to know it may supply its own.
        source = "args.migration_sql" if overridden else "the Architect's migration_sql"

        scratch = Path(tempfile.mkdtemp(prefix="gm_migrate_")) / "scratch.sqlite"
        failed_statement = ""
        try:
            conn = sqlite3.connect(str(scratch))
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            applied = 0
            if migration.strip():
                # Statement at a time so the error message names the statement
                # that actually failed rather than the whole script.
                for statement in _split_sql(migration):
                    failed_statement = statement
                    conn.execute(statement)
                    applied += 1
                conn.commit()
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
            indexes = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            conn.close()
            return ToolResult(
                True, "sql_exec",
                stdout=f"schema created; {applied} migration statement(s) applied\n"
                       f"tables: {tables}\nindexes: {indexes}",
                data={"tables": tables, "indexes": indexes, "applied": applied},
            )
        except sqlite3.Error as exc:
            trace = (
                f"Traceback (most recent call last):\n"
                f'  File "{schema_path}", line 1, in sql_exec\n'
                f"sqlite3.{type(exc).__name__}: {exc}"
            )
            detail = (
                f"{exc}\n\n"
                f"failing statement (from {source}):\n  {failed_statement.strip()[:600]}\n\n"
                f"schema.sql was applied first and succeeded, so this statement conflicts "
                f"with -- or is invalid against -- the schema already in memory_system/"
                f"schema.sql. If the migration itself is wrong (a placeholder such as "
                f'`CREATE TABLE t (...)`, or DDL that schema.sql already creates), pass a '
                f'corrected script yourself: {{"tool": "sql_exec", "args": '
                f'{{"migration_sql": "<full corrected SQL, or an empty string if no '
                f'migration is needed>"}}}}. Editing schema.sql with write_file is the other '
                f"valid fix when the SCHEMA is what is wrong."
            )
            return ToolResult(False, "sql_exec", stderr=detail, stack_trace=trace)
        finally:
            shutil.rmtree(scratch.parent, ignore_errors=True)

    async def _t_run_sandbox_smoke_test(self, args: dict[str, Any]) -> ToolResult:
        """Run the freshly written agent against sample medical checkpoints.

        Executed in a child process, not in-process: importing model-authored
        code into the orchestrator would let a bad edit take the research loop
        down with it, and a stale module in `sys.modules` would make the next
        iteration test the previous iteration's code.
        """
        n = int(args.get("n", 3))
        runner = self.workspace / "_smoke_runner.py"
        runner.write_text(_SMOKE_RUNNER, encoding="utf-8")
        code, out, err = await _run(
            [sys.executable, str(runner), str(n)], cwd=self.workspace, timeout=180,
            env_extra={"GATEMEM_ORCHESTRATOR_ROOT": str(config.PROJECT_ROOT),
                       "MOCK_MODE": "true" if config.MOCK_MODE else "false",
                       "GATEMEM_REPO": str(config.GATEMEM_REPO)},
        )
        payload: dict[str, Any] = {}
        for line in out.splitlines():
            if line.startswith("SMOKE_RESULT="):
                try:
                    payload = json.loads(line[len("SMOKE_RESULT="):])
                except json.JSONDecodeError:
                    pass
        ok = code == 0 and bool(payload.get("ok"))
        return ToolResult(ok, "run_sandbox_smoke_test", stdout=out[-4000:], stderr=err[-3000:],
                          stack_trace=err if not ok else "", data=payload)

    async def _t_finish(self, args: dict[str, Any]) -> ToolResult:
        return ToolResult(True, "finish", stdout=str(args.get("summary", "done")), data=dict(args))


# ======================================================================
# helpers
# ======================================================================


async def _run(
    argv: list[str], cwd: Path, timeout: int, env_extra: dict[str, str] | None = None
) -> tuple[int, str, str]:
    """Async subprocess with a hard timeout.  Never blocks the event loop."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(cwd), env.get("PYTHONPATH", "")) if p
    )
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=str(cwd), env=env,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        return 124, "", f"timed out after {timeout}s: {' '.join(argv[:3])}"
    return process.returncode or 0, stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


def _match_ignoring_whitespace(original: str, search: str) -> tuple[int, int] | None:
    """Character span in `original` matching `search` up to whitespace, or None.

    Deliberately conservative: it matches whole lines, compares them stripped,
    and gives up unless the match is UNIQUE. Fuzz that picks one of several
    candidates is how `patch(1)` corrupts files, and this loop's whole premise
    is that iteration N inherits iteration N-1's workspace -- a patch applied in
    the wrong place is inherited too.
    """
    needle = [line.strip() for line in search.strip("\n").splitlines()]
    if not needle:
        return None
    lines = original.splitlines(keepends=True)
    haystack = [line.strip() for line in lines]
    starts = [
        i for i in range(len(haystack) - len(needle) + 1)
        if haystack[i: i + len(needle)] == needle
    ]
    if len(starts) != 1:
        return None
    begin = starts[0]
    start_char = sum(len(line) for line in lines[:begin])
    end_char = start_char + sum(len(line) for line in lines[begin: begin + len(needle)])
    # Trailing newline belongs to the surrounding text, not the replacement,
    # unless the search block asked for it.
    if not search.endswith("\n") and original[start_char:end_char].endswith("\n"):
        end_char -= 1
    return start_char, end_char


def _near_miss_report(original: str, search: str, rel: str) -> str:
    """Why the search block did not match, with the real text to copy."""
    needle = search.strip("\n").splitlines()
    lines = original.splitlines()
    if not needle or not lines:
        return f"search block not found in {rel} (file is empty)"

    width = len(needle)
    best_start, best_score = 0, -1.0
    for i in range(max(1, len(lines) - width + 1)):
        window = lines[i: i + width]
        score = difflib.SequenceMatcher(
            None, [ln.strip() for ln in needle], [ln.strip() for ln in window]
        ).ratio()
        if score > best_score:
            best_start, best_score = i, score

    pad = 3
    lo = max(0, best_start - pad)
    hi = min(len(lines), best_start + width + pad)
    actual = "\n".join(lines[lo:hi])
    return (
        f"search block not found in {rel}. The closest text in the file is at "
        f"lines {lo + 1}-{hi} ({best_score:.0%} similar). This is what is ACTUALLY "
        f"there -- copy the exact lines you want into `search`, including "
        f"indentation, and do not retype them from memory:\n"
        f"----- {rel} lines {lo + 1}-{hi} -----\n{actual}\n----- end -----\n"
        f"Re-sending the same search block will fail the same way."
    )


_PYTEST_COUNTS = re.compile(r"(\d+) (passed|failed|error|errors)")


def _parse_pytest_counts(text: str) -> tuple[int, int]:
    passed = failed = 0
    for count, kind in _PYTEST_COUNTS.findall(text):
        if kind == "passed":
            passed = max(passed, int(count))
        else:
            failed = max(failed, int(count))
    return passed, failed


def _split_sql(script: str) -> list[str]:
    """Split a migration into statements on `;`, ignoring comments and literals.

    WHY THIS IS NOT `script.split(";")` (observed, runs_verify4). The previous
    version dropped only lines that *start* with `--`, so a TRAILING comment
    survived -- and the Architect writes them constantly:

        ALTER TABLE records ADD COLUMN search_tokens TEXT NOT NULL DEFAULT '';
            -- tokenized routine body for relevance; speeds INSTR scans vs full body

    That comment contains a semicolon, so the naive split cut inside it and
    handed sql_exec the fragment `speeds INSTR scans vs full body ALTER TABLE
    records ADD COLUMN content_hash TEXT`. The Developer got
    `near "speeds": syntax error` -- an error about the Architect's PROSE rather
    than its DDL -- and spent one of MAX_DEV_RETRIES working out that the
    migration was fine all along.

    A semicolon inside a string literal (`DEFAULT 'a;b'`) is the same bug with a
    rarer trigger, so quotes are tracked rather than assumed absent.
    """
    statements: list[str] = []
    current: list[str] = []
    index, length = 0, len(script)

    while index < length:
        char = script[index]

        # Inside a literal nothing is punctuation. '' is an escaped quote.
        if char == "'":
            current.append(char)
            index += 1
            while index < length:
                current.append(script[index])
                if script[index] == "'":
                    if index + 1 < length and script[index + 1] == "'":
                        current.append(script[index + 1])
                        index += 2
                        continue
                    index += 1
                    break
                index += 1
            continue

        if char == "-" and script.startswith("--", index):
            newline = script.find("\n", index)
            index = length if newline == -1 else newline
            continue

        if char == "/" and script.startswith("/*", index):
            close = script.find("*/", index + 2)
            index = length if close == -1 else close + 2
            continue

        if char == ";":
            statements.append("".join(current).strip())
            current = []
            index += 1
            continue

        current.append(char)
        index += 1

    statements.append("".join(current).strip())
    return [statement for statement in statements if statement]


def _apply_unified_diff(original: str, diff: str) -> str:
    """Minimal but strict unified-diff applier.

    Strict on purpose: context mismatches raise instead of fuzzing, because a
    fuzzy patch that lands in the wrong place produces code that compiles and
    is wrong -- the most expensive failure mode in this loop.
    """
    src = original.splitlines(keepends=True)
    out: list[str] = []
    cursor = 0
    hunk_header = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

    lines = diff.splitlines()
    index = 0
    saw_hunk = False
    while index < len(lines):
        header = hunk_header.match(lines[index])
        if not header:
            index += 1
            continue
        saw_hunk = True
        start = int(header.group(1)) - 1
        if start < cursor:
            raise ValueError(f"hunk at line {start + 1} overlaps a previous hunk")
        out.extend(src[cursor:start])
        cursor = start
        index += 1
        while index < len(lines) and not hunk_header.match(lines[index]):
            line = lines[index]
            if line.startswith("+++") or line.startswith("---"):
                index += 1
                continue
            if line.startswith("+"):
                out.append(line[1:] + "\n")
            elif line.startswith("-"):
                if cursor >= len(src) or src[cursor].rstrip("\n") != line[1:].rstrip("\n"):
                    got = src[cursor].rstrip("\n") if cursor < len(src) else "<eof>"
                    raise ValueError(
                        f"context mismatch at line {cursor + 1}: expected {line[1:]!r}, found {got!r}"
                    )
                cursor += 1
            elif line.startswith(" ") or line == "":
                if cursor < len(src):
                    out.append(src[cursor])
                    cursor += 1
            index += 1
    if not saw_hunk:
        raise ValueError("diff contained no @@ hunk headers")
    out.extend(src[cursor:])
    return "".join(out)


def _fallback_lint(workspace: Path) -> list[str]:
    """AST-based checks used when ruff is unavailable: unused imports, bare except."""
    findings: list[str] = []
    for path in sorted(workspace.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "_smoke_runner.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue  # compile_check owns syntax; do not double-report
        source = path.read_text(encoding="utf-8")
        imported: dict[str, int] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported[(alias.asname or alias.name).split(".")[0]] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                # `from __future__ import annotations` is a compiler directive,
                # not a name the module uses; flagging it as unused is the
                # classic naive-linter false positive.
                if node.module == "__future__":
                    continue
                for alias in node.names:
                    if alias.name != "*":
                        imported[alias.asname or alias.name] = node.lineno
            elif isinstance(node, ast.ExceptHandler) and node.type is None:
                findings.append(f"{path.name}:{node.lineno}: E722 bare except")
        for name, lineno in imported.items():
            if len(re.findall(rf"\b{re.escape(name)}\b", source)) <= 1:
                findings.append(f"{path.name}:{lineno}: F401 unused import {name!r}")
    return findings


# The smoke-test child process.  Written into the workspace at call time so it
# imports the Developer's code, not the orchestrator's copy of it.
_SMOKE_RUNNER = '''"""Smoke test: run the freshly written agent on sample medical checkpoints."""
import json, os, sys, traceback

sys.path.insert(0, os.environ.get("GATEMEM_ORCHESTRATOR_ROOT", ""))
sys.path.insert(0, os.getcwd())


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    try:
        from memory_system.agent import GateMemAgent
    except Exception:
        print("SMOKE_RESULT=" + json.dumps({"ok": False, "error": "import failed"}))
        traceback.print_exc()
        return 1

    try:
        import config as orch_config
        from gatemem_adapter import (VALID_ACTIONS, assert_no_hidden_fields,
                                     load_medical_dataset, strip_hidden_fields)
        if orch_config.MOCK_MODE:
            from mocks.dataset import build_mock_dataset
            dataset = build_mock_dataset()
        else:
            dataset = load_medical_dataset(orch_config.GATEMEM_DATA_DIR)
    except Exception:
        print("SMOKE_RESULT=" + json.dumps({"ok": False, "error": "dataset load failed"}))
        traceback.print_exc()
        return 1

    checkpoints = sorted(dataset.checkpoints, key=lambda c: c["checkpoint_id"])[:n]
    results, failures = [], []
    for cp in checkpoints:
        try:
            episode = dataset.episodes_by_id[cp["episode_id"]]
            agent = GateMemAgent(":memory:")
            agent.reset(episode)
            for turn in dataset.turns_up_to(cp["episode_id"], cp["as_of_turn_id"]):
                agent.ingest(turn)
            safe = strip_hidden_fields(cp)
            assert_no_hidden_fields(safe, where="smoke checkpoint")
            out = agent.query(safe)
            assert out.get("action") in VALID_ACTIONS, "invalid action: %r" % out.get("action")
            assert isinstance(out.get("answer"), str), "answer must be a string"
            results.append({"checkpoint_id": cp["checkpoint_id"], "action": out["action"]})
        except Exception as exc:
            failures.append({"checkpoint_id": cp["checkpoint_id"], "error": repr(exc)})
            traceback.print_exc()

    payload = {"ok": not failures, "n": len(checkpoints),
               "results": results, "failures": failures}
    print("SMOKE_RESULT=" + json.dumps(payload))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
'''
