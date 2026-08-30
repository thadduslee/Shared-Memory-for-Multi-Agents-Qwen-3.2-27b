"""One async abstraction over the DeepSeek Harness (`dsh`).

WHY THIS FILE EXISTS
--------------------
The brief requires every graph node to be a `dsh` harness invocation rather
than a raw chat completion, with per-node least privilege.  `dsh` is a
Node/TypeScript plugin harness, so "least privilege" is expressed as a *Cordis
plugin composition*: a node's tools are exactly the plugins its config file
mounts.  This module turns a `DSHProfile` into such a composition, launches the
harness, and returns a structured result.

WHAT IS CONFIRMED vs ASSUMED
----------------------------
CONFIRMED by reading github.com/deepseek-ai/deepseek-harness @ master:

* The repo DOES ship a first-party Python binding at `python/sdk`, published as
  `deepseek-harness-sdk` and imported as `deepseek_harness`.  Per the brief's
  instruction ("verify whether the repo's python/ directory exposes a
  first-party Python binding; if it does, use it"), that is the primary path
  here -- the subprocess CLI path is the fallback, not the default.
* Public surface: `DeepSeekHarness(provider=..., model=..., max_tokens=...,
  cwd=..., runtime_cwd=..., session_root=..., cordis=..., env=...,
  runtime_bin=..., launch_args_override=..., request_timeout_seconds=...,
  shutdown_timeout_seconds=..., base_url=..., api_key=...)`, and
  `.run(input, *, session_id=None, on_notification=None) -> RunResult`, where
  `RunResult = (session_id, final_response, finish_reason, events,
  notifications, session_root)`.
* The SDK is SYNCHRONOUS -- it drives a subprocess over JSON-RPC stdio.  Since
  every graph node is `async def`, every call is dispatched through
  `asyncio.to_thread` so the event loop is never blocked.  This is not a
  stylistic choice: a blocking `.run()` inside a `Send` fan-out would serialise
  the entire cluster.
* The runtime reads `DEEPSEEK_BASE_URL` / `DEEPSEEK_API_KEY`; the SDK README
  states callers may point those at another endpoint. VERIFIED end to end on
  2026-08-24: `DeepSeekHarness(provider="deepseek-official",
  model="deepseek/deepseek-v4-flash-0731", base_url="https://openrouter.ai/api/v1",
  api_key=<OpenRouter key>)` returns `finish_reason="completed"` with real
  content. The `provider` stays the registered Cordis route name; only the
  endpoint changes.
* Plugin package names and config keys used below are copied verbatim from
  `examples/jsonrpc-agent/cordis.yml` and
  `python/sdk-runtime/.../runtime/cordis.yml`, including
  `@deepseek-ai/dsh-agent-spine-demo`'s `persona` key -- which is where a
  system prompt goes -- and its `DSH_SYSTEM_PROMPT` environment fallback.

# VERIFIED (was an ASSUMPTION): `max_turns` DOES NOT TAKE EFFECT.  Neither
#   `maxTurns` nor `DSH_MAX_TURNS` appears anywhere in the pinned runtime
#   binary, and `dsh-agent-spine-demo` -- the loop that would honour it --
#   accepts only persona / workspaceContext / skills / toolBash / toolJobs /
#   dshHome.  Nothing reads the variable, so a profile's `max_turns` is
#   documentation, not a limit: the agent loop runs until it finishes or until
#   the caller's wall-clock timeout kills it.  Do not treat it as a safety
#   bound.  `temperature` is exported the same way and is equally unconfirmed.
#   This is why the Developer -- whose composition mounts bash, a write surface
#   and subagents -- ran past 630s with `max_turns=40` nominally set.
# ASSUMPTION: no first-party web-search plugin was confirmed in the repo, so the
#   Architect's required web search is NOT routed through dsh.  It goes through
#   `tools/websearch.py`, which is a separate, explicitly-mocked adapter.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import config

log = logging.getLogger("orchestrator.dsh")


# ==========================================================================
# Tool vocabulary -> real Cordis plugins
# ==========================================================================

# Our abstract capability names, mapped to the plugin blocks that grant them.
# A node that does not list a capability simply does not get those plugins
# mounted, which is a structural guarantee rather than a prompt-level request:
# the Architect cannot write a file because no file-writing tool exists in its
# composition, not because we asked it nicely.
CAPABILITY_PLUGINS: dict[str, list[dict[str, Any]]] = {
    "fs_read": [
        {"id": "fs-local", "name": "@deepseek-ai/dsh-fs-local",
         "config": {"cwd": "${DSH_CWD}"}},
        {"id": "fs-observation-policy", "name": "@deepseek-ai/dsh-fs-observation-policy"},
    ],
    # tool-fs is the model-facing file tool surface; fs-local alone does not
    # expose file tools to the model (per the upstream comment in cordis.yml).
    "fs_write": [{"id": "tool-fs", "name": "@deepseek-ai/dsh-tool-fs"}],
    "bash": [
        {"id": "subprocess", "name": "@deepseek-ai/dsh-subprocess-local"},
        {"id": "bash", "name": "@deepseek-ai/dsh-bash-local",
         "config": {"cwd": "${DSH_CWD}", "timeoutMs": 120000}},
    ],
    "todo": [{"id": "tool-todo", "name": "@deepseek-ai/dsh-tool-todo",
              "config": {"allowParallelInProgress": True}}],
    "subagent": [
        {"id": "subagent", "name": "@deepseek-ai/dsh-subagent"},
        {"id": "subagent-spawn-in-process",
         "name": "@deepseek-ai/dsh-subagent-spawn-in-process",
         "config": {"providerName": "spawn"}},
        {"id": "tool-subagent", "name": "@deepseek-ai/dsh-tool-subagent",
         "config": {"provider": "spawn", "toolName": "subagent",
                    "enableRunInBackground": False}},
    ],
}

# Mounted for every profile regardless of capability: the JSON-RPC transport
# the SDK talks to, the agent spine, the LLM adapter, persistence, and token
# accounting.  Removing any of these breaks the harness rather than restricting
# the agent.
BASE_PLUGINS: list[dict[str, Any]] = [
    {"id": "sdk-jsonrpc-server", "name": "@deepseek-ai/dsh-sdk-jsonrpc-server"},
    {"id": "llm-deepseek", "name": "@deepseek-ai/dsh-llm-deepseek"},
    {"id": "sessions", "name": "@deepseek-ai/dsh-session-persistence-jsonl",
     "config": {"root": "${DSH_SESSION_ROOT}"}},
    {"id": "session-checkpoints", "name": "@deepseek-ai/dsh-session-checkpoint-policy"},
    {"id": "token-meter", "name": "@deepseek-ai/dsh-token-meter"},
    {"id": "compaction-basic", "name": "@deepseek-ai/dsh-compaction-basic",
     "config": {"thresholdRatio": 0.8, "retainRatio": 0.16, "maxTokens": 8192,
                "compactionRetries": 1}},
]


@dataclass(frozen=True)
class DSHProfile:
    """Per-node harness profile.  The unit of least privilege.

    Attributes map to the harness as follows:

    * `model`, `provider`, `max_tokens` -> `DeepSeekHarnessConfig` fields.
      `max_tokens=None` means uncapped and is passed through as None.
    * `system_prompt`   -> `agent-spine-demo`'s `persona` config key.
    * `capabilities`    -> which plugin blocks get mounted (the tool set).
    * `workdir`         -> `cwd`, which both `fs-local` and `bash-local` are
                           pinned to, so the agent cannot reach outside it.
    * `max_turns`, `temperature` -> exported as env vars, and NEITHER IS READ
                           by the harness on the SDK path. VERIFIED against the
                           installed `deepseek_harness`: `DeepSeekHarnessConfig`
                           has exactly provider, model, max_tokens, cwd,
                           runtime_cwd, session_root, cordis, env, runtime_bin,
                           launch_args_override, request_timeout_seconds,
                           shutdown_timeout_seconds, base_url and api_key --
                           there is no temperature field to pass one to, and a
                           real run's `request/header` confirms the wire config
                           is {provider, model, maxTokens, reasoningEffort}.
                           `temperature` IS honoured on the `http` transport,
                           where `nodes/_transport.py` puts it in the request
                           body. Do not "fix" this by adding a temperature key
                           to the `llm-deepseek` plugin block below: that block
                           is mounted by EVERY profile, and an unrecognised
                           config key there risks failing every node at launch
                           to control a parameter this loop does not depend on.
    """

    name: str
    model: str
    system_prompt: str
    capabilities: frozenset[str] = frozenset()
    max_turns: int = 24
    temperature: float = 0.2
    # `None` = uncapped: BOTH transports omit the parameter rather than
    # sending a number, so the model runs until it emits a stop token.
    # `DeepSeekHarnessConfig.max_tokens` is already `int | None` and its
    # client drops `maxTokens` from the wire payload when it is None
    # (VERIFIED: deepseek_harness/client.py, `if max_tokens is not None`).
    max_tokens: int | None = None

    # `provider` is a CORDIS ROUTE NAME, not a vendor. It must name a provider
    # route the mounted composition actually registers -- the bundled default
    # registers exactly one, `deepseek-official`. Which *endpoint* that route
    # talks to is decided by `base_url`, so a profile pointed at OpenRouter or
    # at a local vLLM server still uses the registered route name. Putting a
    # vendor name here ("openai", "vllm-local") makes the harness fail to
    # resolve the adapter at launch.
    provider: str = config.DSH_PROVIDER

    # `route` names the HTTP serving route used when this profile runs over the
    # `http` transport. Declared explicitly rather than reverse-mapped from
    # `base_url`: two profiles may legitimately share one base URL (one
    # OpenRouter key serving both the agentic models and the judge), and a
    # reverse map silently collapses them onto whichever entry was defined last.
    route: str = "openrouter"

    base_url: str | None = None
    api_key_env: str | None = None
    workdir: Path | None = None

    def with_workdir(self, workdir: Path) -> "DSHProfile":
        """Scope this profile to a directory.  Profiles are frozen, so this
        returns a copy -- the module-level profiles stay immutable and shared."""
        return DSHProfile(**{**self.__dict__, "workdir": Path(workdir)})

    def plugins(self) -> list[dict[str, Any]]:
        blocks = list(BASE_PLUGINS)
        blocks.append(
            {
                "id": "agent-core",
                "name": "@deepseek-ai/dsh-agent-spine-demo",
                "config": {
                    "persona": self.system_prompt,
                    "workspaceContext": {"maxBytes": 65536},
                    "skills": {"enabled": False},
                    # Bash is a *tool-surface* switch on the spine as well as a
                    # mounted plugin; both must agree or the agent advertises a
                    # tool it cannot call.
                    "toolBash": {"enableRunInBackground": False}
                    if "bash" in self.capabilities
                    else False,
                    "toolJobs": False,
                },
            }
        )
        for capability in sorted(self.capabilities):
            blocks.extend(CAPABILITY_PLUGINS.get(capability, []))
        return blocks


@dataclass
class DSHResult:
    """Structured result of one harness invocation."""

    ok: bool
    text: str
    profile: str
    finish_reason: str | None = None
    session_id: str | None = None
    duration_s: float = 0.0
    stderr: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    error: str | None = None
    # The two fields a tool-calling turn needs, carried so that `agent_call` can
    # stay transport-blind (see nodes/_transport.py). The dsh path never
    # populates them: the harness runs its OWN tool loop over mounted Cordis
    # plugins, so it has no seam through which to accept our schema or hand back
    # an unexecuted call. `content` falls back to `text` there, which is the
    # same thing when nothing was rendered into it.
    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)

    def json_block(self) -> dict[str, Any] | None:
        """Extract the first fenced JSON object from the agent's response.

        Nodes ask the harness for Markdown prose *plus* a machine-parseable
        JSON block; this pulls the block out.  Returns None rather than raising
        so a node can fall back to its prose output.
        """
        return extract_json_block(self.text)


# --------------------------------------------------------------------------
# JSON block extraction
#
# Every node's machine-readable deliverable arrives as a ```json fence inside
# Markdown prose, so this function is the seam the whole loop's structured
# output passes through.  It is deliberately forgiving, because the failure it
# guards against is SILENT: a block that does not parse is not an error anyone
# sees, it is an Architect design with no migration, a Judge with no verdict and
# a Critic with no proposals -- each of which looks like a working run.
#
# REGRESSION (runs_smoke/iter_1). The Architect emitted a syntactically perfect
# design document whose JSON block contained a JavaScript-style expression:
#
#     "migration_sql": "-- Initial creation...\nPRAGMA foreign_keys = ON;\n" + schema_ddl,
#
# `json.loads` failed at character 1775, the brace-matching fallback failed too,
# and `extract_json_block` returned None. The Architect then wrote a 154-byte
# design.json with every field null and a ZERO-BYTE migration.sql, and handed
# the Developer an empty work order -- which is why that Developer sat in a
# repetition loop emitting "<thought Let's read the files." until it timed out.
# One malformed comma cost the entire iteration.
#
# So a failed parse is now retried against a repair pass rather than discarded.
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n", re.IGNORECASE)

# Bare identifiers spliced into a JSON value by a model writing JavaScript:
#   "a": "text" + schema_ddl,   ->   "a": "text",
# Only ever applied OUTSIDE a string, by `_repair_json` below.
_CONCAT_IDENT_RE = re.compile(r"\+\s*[A-Za-z_][A-Za-z0-9_.]*\s*(?=[,}\]\n])")

_PY_LITERALS = {"True": "true", "False": "false", "None": "null"}

# A bare, unquoted word outside a string. Cannot appear in valid JSON, so
# whatever it is, it is a mistake -- `_repair_json` decides which one.
_BARE_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _repair_json(text: str) -> str:
    """Best-effort repair of the JSON dialects models actually emit.

    Walks the text once, tracking string state, so every repair is applied only
    where it is unambiguous.  A naive regex pass over the whole document would
    corrupt the payload -- `schema_ddl` and `migration_sql` are SQL strings that
    legitimately contain braces, commas, quotes and the word `True`.

    Repairs, all outside strings except where noted:

    * `"a" + "b"`  -> `"ab"`          (string concatenation)
    * `"a" + ident` -> `"a"`          (concatenation with a variable; the
                                       variable is unresolvable, so it is
                                       dropped rather than guessed at)
    * trailing commas before `}` or `]`
    * `//` and `/* */` comments
    * Python's `True` / `False` / `None`
    * a raw newline or tab INSIDE a string, escaped rather than left to fail
    """
    out: list[str] = []
    index = 0
    length = len(text)
    in_string = False
    quote = '"'
    while index < length:
        char = text[index]

        if in_string:
            if char == "\\" and index + 1 < length:
                # `\'` is legal inside a single-quoted string and illegal inside
                # the double-quoted one we are rewriting it into.
                if quote == "'" and text[index + 1] == "'":
                    out.append("'")
                else:
                    out.append(text[index : index + 2])
                index += 2
                continue
            if char == "\n":
                out.append("\\n")
                index += 1
                continue
            if char == "\r":
                out.append("\\r")
                index += 1
                continue
            if char == "\t":
                out.append("\\t")
                index += 1
                continue
            if char == '"' and quote == "'":
                # A double quote inside a single-quoted string was data; it has
                # to be escaped now that the delimiters are double quotes.
                out.append('\\"')
                index += 1
                continue
            if char == quote:
                # Look ahead for `"  +  "` and splice the two strings into one
                # by dropping both the closing quote and the reopening quote.
                ahead = index + 1
                while ahead < length and text[ahead] in " \t\r\n":
                    ahead += 1
                if ahead < length and text[ahead] == "+":
                    after = ahead + 1
                    while after < length and text[after] in " \t\r\n":
                        after += 1
                    if after < length and text[after] == quote:
                        index = after + 1
                        continue
                in_string = False
                # Always close with a double quote: the delimiter may have been
                # a single quote on the way in, and emitting it verbatim would
                # produce `"tool': ...` -- broken in a new way.
                out.append('"')
                index += 1
                continue
            out.append(char)
            index += 1
            continue

        # --- outside a string ---
        if char in "\"'":
            # Normalize a single-quoted string to a double-quoted one.
            in_string = True
            quote = char
            out.append('"')
            index += 1
            continue
        if text.startswith("//", index):
            index = text.find("\n", index)
            if index == -1:
                break
            continue
        if text.startswith("/*", index):
            closed = text.find("*/", index)
            index = length if closed == -1 else closed + 2
            continue
        concat = _CONCAT_IDENT_RE.match(text, index)
        if concat:
            index = concat.end()
            continue
        if char == ",":
            ahead = index + 1
            while ahead < length and text[ahead] in " \t\r\n":
                ahead += 1
            if ahead < length and text[ahead] in "}]":
                index += 1  # trailing comma
                continue
        word = _BARE_WORD_RE.match(text, index)
        if word:
            token = word.group(0)
            end = word.end()
            if token in _PY_LITERALS:
                out.append(_PY_LITERALS[token])
            elif token in {"true", "false", "null"}:
                out.append(token)
            else:
                # An unquoted bare word. In valid JSON this cannot occur, so the
                # only question is which mistake it is:
                #   {tool: "x"}     an unquoted KEY   -> quote it
                #   {"a": ident}    an unresolvable   -> null, because the value
                #                   VALUE                it named is not in the
                #                                        document to recover
                ahead = end
                while ahead < length and text[ahead] in " \t\r\n":
                    ahead += 1
                is_key = ahead < length and text[ahead] == ":"
                out.append(f'"{token}"' if is_key else "null")
            index = end
            continue
        if text.startswith("...", index):
            # A literal ellipsis placeholder -- models write it where they mean
            # "and so on". There is nothing to recover, so it becomes null.
            out.append("null")
            index += 3
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _loads_object(candidate: str) -> dict[str, Any] | None:
    """Parse one candidate as a JSON object, repairing it if the first try fails."""
    candidate = candidate.strip()
    if not candidate:
        return None
    for attempt in (candidate, _repair_json(candidate)):
        try:
            parsed = json.loads(attempt)
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _fenced_candidates(text: str) -> list[str]:
    """Every fenced block body, ```json ones first, in document order.

    Prefers `json`-tagged fences because a design document routinely contains
    ```sql and ```python fences too, and the deliverable is the tagged one.
    Untagged and other-language fences are still tried afterwards, since a
    model that forgets the tag has still produced the block.
    """
    tagged: list[str] = []
    untagged: list[str] = []
    for match in _FENCE_RE.finditer(text):
        body_start = match.end()
        closing = text.find("```", body_start)
        body = text[body_start:] if closing == -1 else text[body_start:closing]
        (tagged if match.group(1).lower() == "json" else untagged).append(body)
    return tagged + untagged


def _balanced_objects(text: str) -> list[str]:
    """Every balanced top-level `{...}` span, in document order.

    String-aware, so a brace inside a SQL string does not desynchronise the
    depth count -- which is exactly what happens to a naive scanner on a
    `schema_ddl` value containing `CHECK (json_valid(x))`.
    """
    spans: list[str] = []
    depth = 0
    begin = -1
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                begin = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and begin != -1:
                spans.append(text[begin : index + 1])
                begin = -1
    return spans


# --------------------------------------------------------------------------
# Tool-call markup leaking into prose
#
# A model that wants to open a file and has no tool schema to do it with does
# not say so -- it EMITS THE CALL AS TEXT, in whatever markup its own training
# uses, and stops as though it were now waiting for a result that is never
# coming. On DeepSeek that markup is `<|DSML|>tool_calls`; other vendors emit
# `<tool_call>`, `<function_calls>` or a bare `<invoke name="...">`.
#
# WHAT IT COST (runs_5iter, 2026-08-29). The Critic's task said "READ THE SOURCE
# FIRST ... Open the files that own the mechanisms", and the Critic runs over
# AGENT_TRANSPORT=http, which advertises no tools. All three critiques in that
# run were four lines of `<|DSML|>tool_calls` markup and nothing else: no
# prose, no ```json fence, so `component: null`, `mechanisms: []` and ZERO
# proposals every iteration -- a loop with no feedback in it. `result.ok` was
# true throughout, because a reply full of markup is not an empty reply.
#
# Then it spread. The critique is pasted verbatim into the next Architect's
# prompt, and in iteration 3 the Architect imitated the markup it had been
# shown: 181 characters of `<|DSML|>invoke name="OpenFile"`, no design, an
# empty work order, and an iteration that changed not one byte.
#
# So: detect it (a reply that is mostly markup is a FAILED reply, whatever its
# finish_reason), and strip it before any text crosses into another node's
# prompt.
# --------------------------------------------------------------------------

# Any tag whose NAME is a tool-call verb, with or without a vendor's decoration
# around it: `<|DSML|>tool_calls>`, `</|DSML|>parameter>`, `<invoke name="Read">`,
# `<parameter ...>`, `<tool_call>`. The name has to sit in tag-name
# position, so a sentence that merely uses the word "parameter" is untouched.
_TOOL_MARKUP_RE = re.compile(
    r"</?\s*(?:[|｜]{1,2}\s*[A-Za-z_]{0,16}\s*[|｜]{1,2}\s*)?(?:antml:)?"
    r"(?:tool_calls?|function_calls|invoke|parameter)\b[^<>]{0,400}?>",
    re.IGNORECASE,
)

# A tag-shaped opener is enough on its own; these are the names models reach for
# when they are trying to call a tool they have not been given.
_TOOL_MARKUP_SIGNALS = (
    "tool_calls", "tool_call", "function_calls", "<invoke", "invoke name=",
    "DSML", "\u2758",
)


def strip_tool_call_markup(text: str) -> str:
    """Remove emitted tool-call syntax, keeping whatever prose surrounds it.

    Used on every text that crosses from one node into another node's prompt,
    because markup that reaches a downstream model is markup that model may
    imitate.
    """
    if not text:
        return text
    cleaned = _TOOL_MARKUP_RE.sub("", text)
    # The markup above is line-oriented in practice; drop the blank scaffolding
    # it leaves behind rather than shipping a document made of empty lines.
    lines = [line for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def looks_like_tool_call_markup(text: str, *, threshold: float = 0.5) -> bool:
    """True when a reply is mostly an attempt to call a tool it does not have.

    `threshold` is the share of the reply that has to be markup. A design
    document that merely MENTIONS `<tool_call>` in a sentence keeps almost all
    of its length after stripping and is not flagged; a reply that is four
    invoke tags loses nearly all of it and is.
    """
    body = (text or "").strip()
    if not body:
        return False
    if not any(signal in body for signal in _TOOL_MARKUP_SIGNALS):
        return False
    remaining = strip_tool_call_markup(body)
    return len(remaining) < threshold * len(body)


def extract_json_block(text: str) -> dict[str, Any] | None:
    """Find the deliverable JSON object in a model's Markdown reply.

    Tries, in order: every ```json fence, every other fence, then every
    balanced top-level `{...}` span -- and each candidate is parsed twice, once
    as written and once through `_repair_json`. Returns the first object that
    parses. Returns None only when nothing in the text is recoverable.
    """
    if not text:
        return None
    for candidate in _fenced_candidates(text):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    for candidate in _balanced_objects(text):
        parsed = _loads_object(candidate)
        if parsed is not None:
            return parsed
    return None


# ==========================================================================
# Cordis config rendering
# ==========================================================================


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text.startswith("${") and text.endswith("}"):
        # Turn our placeholder into the harness's own env-var idiom, which the
        # reference compositions express as a `!!js` node.
        env_name = text[2:-1]
        return f"!!js process.env.{env_name} ?? process.cwd()"
    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def _yaml_dump_config(value: Any, indent: int) -> list[str]:
    pad = " " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{pad}{key}:")
                lines.extend(_yaml_dump_config(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {_yaml_scalar(item)}")
    else:
        lines.append(f"{pad}{_yaml_scalar(value)}")
    return lines


def render_cordis_config(profile: DSHProfile, out_dir: Path | None = None) -> Path:
    """Write the Cordis composition for `profile` and return its path.

    Written with a plain emitter rather than PyYAML because the harness's config
    dialect uses `!!js` nodes that PyYAML would quote into uselessness.
    """
    out_dir = Path(out_dir or config.DSH_CORDIS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{profile.name}.cordis.yml"

    lines = [
        f"# Cordis composition for the {profile.name!r} node -- generated, do not edit.",
        "# Tools are granted structurally: a plugin that is not mounted here is a",
        "# tool this node physically cannot call.",
        f"# capabilities: {sorted(profile.capabilities) or ['<none>']}",
        "# stdout is reserved for JSON-RPC; do not add a console logger.",
        "",
    ]
    for block in profile.plugins():
        lines.append(f"- id: {block['id']}")
        lines.append(f"  name: '{block['name']}'")
        cfg = block.get("config")
        if cfg:
            lines.append("  config:")
            lines.extend(_yaml_dump_config(cfg, 4))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ==========================================================================
# Client interface
# ==========================================================================


class _HarnessAborted(RuntimeError):
    """Raised inside the worker thread when its harness was torn down."""


class _HarnessHandle:
    """Lets the event loop tear down a harness that is blocking a worker thread.

    `asyncio.wait_for` cancels the coroutine that is *awaiting*; it can never
    cancel the thread behind `asyncio.to_thread`, because Python threads are not
    cancellable. So a bare `wait_for(to_thread(...))` timeout does not stop the
    work -- it only stops us watching it. The abandoned thread keeps driving its
    dsh subprocess, keeps emitting progress lines into the same log, and because
    `to_thread` uses non-daemon workers it also prevents the interpreter from
    exiting. Three timed-out attempts produced three concurrent Architects and a
    process that had printed its summary but could not die.

    `HarnessClient.close()` is the real lever: it terminates the runtime
    subprocess and fails every in-flight waiter, which is what actually unblocks
    `harness.run()` in the worker. This handle just carries the harness across
    the thread boundary so the loop can reach it.
    """

    def __init__(self, profile_name: str) -> None:
        self._lock = threading.Lock()
        self._harness: Any = None
        self._aborted = False
        self.profile_name = profile_name

    def publish(self, harness: Any) -> bool:
        """Called from the worker. False means "abort already fired, stand down"."""
        with self._lock:
            if self._aborted:
                return False
            self._harness = harness
            return True

    def abort(self) -> None:
        """Called from the loop (via a thread -- close() itself blocks)."""
        with self._lock:
            self._aborted = True
            harness = self._harness
            self._harness = None
        if harness is None:
            return
        try:
            harness.close()
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort
            log.warning("dsh: %s harness close() failed: %s", self.profile_name, exc)


class DSHClientProtocol(Protocol):
    async def run(
        self, profile: DSHProfile, task: str, workdir: Path, timeout_s: int
    ) -> DSHResult: ...


class RealDSHClient:
    """Drives the actual harness.  Used when `MOCK_MODE` is False.

    Prefers the first-party Python SDK; falls back to the `npx @deepseek-ai/dsh`
    CLI when the SDK is not importable.
    """

    def __init__(self) -> None:
        self._sdk: Any = None
        self._sdk_checked = False

    def _load_sdk(self) -> Any:
        if not self._sdk_checked:
            self._sdk_checked = True
            try:
                import deepseek_harness  # type: ignore

                self._sdk = deepseek_harness
                log.info("dsh: using first-party Python SDK (deepseek_harness)")
            except ImportError:
                self._sdk = None
                log.warning(
                    "dsh: `deepseek_harness` not importable under %s "
                    "-- falling back to the CLI subprocess path", sys.executable,
                )
        return self._sdk

    def _profile_env(self, profile: DSHProfile, workdir: Path, cordis_path: Path) -> dict[str, str]:
        env = {
            "DSH_CWD": str(workdir),
            "DSH_CORDIS_CONFIG": str(cordis_path),
            "DSH_SESSION_ROOT": str(workdir / ".sessions"),
            # The persona is inlined into the generated config; this env var is
            # the reference composition's fallback and keeps the two agreeing.
            "DSH_SYSTEM_PROMPT": profile.system_prompt,
            # See the module-level ASSUMPTION about these two.
            "DSH_MAX_TURNS": str(profile.max_turns),
            "DSH_TEMPERATURE": str(profile.temperature),
        }
        if profile.base_url:
            env[config.DSH_BASE_URL_ENV] = profile.base_url
        if profile.api_key_env:
            key = os.environ.get(profile.api_key_env, "")
            if key:
                env[config.DSH_API_KEY_ENV] = key
        return env

    async def run(
        self, profile: DSHProfile, task: str, workdir: Path, timeout_s: int
    ) -> DSHResult:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        cordis_path = render_cordis_config(profile)
        started = time.monotonic()

        sdk = self._load_sdk()
        last_error = "no attempt made"
        for attempt in range(config.DSH_MAX_RETRIES + 1):
            attempt_started = time.monotonic()
            try:
                if sdk is not None:
                    result = await self._run_sdk_guarded(
                        sdk, profile, task, workdir, cordis_path, timeout_s
                    )
                else:
                    result = await self._run_cli(profile, task, workdir, cordis_path, timeout_s)
                # Per-attempt, not cumulative: a success on attempt 2 should
                # report its own turn, not that turn plus the dead one before it.
                result.duration_s = time.monotonic() - attempt_started
                if result.ok:
                    return result
                last_error = result.error or "harness returned not-ok"
            except asyncio.TimeoutError:
                # Deliberately NOT retried. A hard timeout means the budget was
                # wrong, and the next attempt gets the same budget and the same
                # work -- it restarts from scratch and dies at the same wall.
                # Retries exist for flaky transport; burning N x timeout_s to
                # rediscover a known-too-small budget only delays the diagnosis.
                last_error = f"dsh hard timeout after {timeout_s}s"
                log.error(
                    "dsh timeout profile=%s after %ds (not retried -- raise its budget)",
                    profile.name, timeout_s,
                )
                break
            except Exception as exc:  # noqa: BLE001 - the harness is a subprocess; anything can come back
                last_error = f"{type(exc).__name__}: {exc}"
                log.exception("dsh invocation failed profile=%s", profile.name)

            if attempt < config.DSH_MAX_RETRIES:
                delay = min(2.0 * (2**attempt), 20.0) * (0.5 + random.random())
                await asyncio.sleep(delay)

        return DSHResult(
            ok=False, text="", profile=profile.name,
            duration_s=time.monotonic() - started, error=last_error,
        )

    async def _run_sdk_guarded(
        self, sdk: Any, profile: DSHProfile, task: str, workdir: Path,
        cordis_path: Path, timeout_s: int,
    ) -> DSHResult:
        """Run the blocking SDK call under a timeout that can actually stop it.

        On timeout: close the harness out-of-band, then wait for the worker to
        unwind before letting the caller retry. Overlapping a retry with a still
        -live previous attempt is what produced interleaved progress logs and
        duplicate billed turns.
        """
        handle = _HarnessHandle(profile.name)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._run_sdk_blocking, sdk, profile, task, workdir,
                cordis_path, timeout_s, handle,
            )
        )
        try:
            # shield: wait_for would cancel `worker`, which does nothing to the
            # thread but does lose our only handle on it. We want the task intact.
            return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout_s + 30)
        except asyncio.TimeoutError:
            log.warning("dsh: tearing down timed-out %s harness", profile.name)
            await asyncio.to_thread(handle.abort)
            try:
                await asyncio.wait_for(
                    asyncio.shield(worker), timeout=config.DSH_TEARDOWN_GRACE_S
                )
            except asyncio.TimeoutError:
                # Nothing further we can do from here; say so rather than let a
                # silent non-daemon thread hold the interpreter open at exit.
                log.error(
                    "dsh: %s worker did not stop within %.0fs after abort -- leaked",
                    profile.name, config.DSH_TEARDOWN_GRACE_S,
                )
            except Exception as exc:  # noqa: BLE001 - abort broke the transport
                log.debug("dsh: %s unwound after abort: %s", profile.name, exc)
            raise

    def _run_sdk_blocking(
        self, sdk: Any, profile: DSHProfile, task: str, workdir: Path,
        cordis_path: Path, timeout_s: int, handle: _HarnessHandle,
    ) -> DSHResult:
        """Synchronous SDK call.  ALWAYS invoked via `asyncio.to_thread`."""
        harness_cls = sdk.DeepSeekHarness
        kwargs: dict[str, Any] = {
            "provider": profile.provider,
            "model": profile.model,
            "max_tokens": profile.max_tokens,
            "cwd": str(workdir),
            "session_root": str(workdir / ".sessions"),
            "cordis": str(cordis_path),
            "env": self._profile_env(profile, workdir, cordis_path),
            "request_timeout_seconds": float(timeout_s),
        }
        if profile.base_url:
            kwargs["base_url"] = profile.base_url
        if profile.api_key_env:
            key = os.environ.get(profile.api_key_env)
            if key:
                kwargs["api_key"] = key

        progress = _ProgressLogger(profile.name)
        harness = harness_cls(**kwargs)
        if not handle.publish(harness):
            # Timed out during construction -- never start the turn.
            harness.close()
            raise _HarnessAborted(f"{profile.name} aborted before start")
        try:
            run_result = harness.run(task, on_notification=progress)
        finally:
            handle.abort()  # idempotent; also the normal-path close
        # Only on a real finish: a torn-down attempt raises above, so this no
        # longer prints "finished in 931s" for a turn that was killed at 330s.
        progress.done()

        events = list(run_result.events or [])
        text = collect_assistant_text(events) or (run_result.final_response or "")
        # `.strip()`, not truthiness (regression, runs_smoke/iter_1). A Developer
        # turn came back as a single space character: `bool(" ")` is True, so it
        # was reported as a SUCCESSFUL turn carrying an unparseable reply. The
        # caller then spent a retry on the no-parseable-action fallback instead
        # of retrying the transport, which is what an empty reply deserves.
        substantive = bool(text.strip())
        return DSHResult(
            ok=substantive and run_result.finish_reason != "error",
            text=text,
            profile=profile.name,
            finish_reason=run_result.finish_reason,
            session_id=run_result.session_id,
            events=events,
            usage=_usage_from_events(events),
            error=None if substantive else f"empty response ({run_result.finish_reason})",
        )

    async def _run_cli(
        self, profile: DSHProfile, task: str, workdir: Path, cordis_path: Path, timeout_s: int
    ) -> DSHResult:
        """Non-interactive CLI fallback.

        # ASSUMPTION: the upstream README documents only `npx @deepseek-ai/dsh web`
        #   and `--no-open`; no headless/JSON-output flags are documented.  The
        #   argv below is therefore a GUESS and is quarantined in this one
        #   function so that a single edit fixes it.  VERIFY with
        #   `npx @deepseek-ai/dsh --help`.  Installing `deepseek-harness-sdk`
        #   avoids this path entirely and is the supported route.
        """
        npx = shutil.which("npx")
        if not npx:
            # Name the interpreter: the overwhelmingly common cause of this is
            # running under a different virtualenv than the one the project's
            # dependencies were installed into, and "not available" alone sends
            # you looking for a missing package instead of a missing activation.
            return DSHResult(
                ok=False, text="", profile=profile.name,
                error=(
                    f"dsh is unreachable. `deepseek_harness` is not importable under "
                    f"{sys.executable}, and `npx` is not on PATH so the CLI fallback "
                    f"cannot run either. Either activate the environment where "
                    f"`pip install -r requirements.txt` was run, or set "
                    f"AGENT_TRANSPORT=http to bypass the harness entirely."
                ),
            )

        argv = [
            npx, "--yes", "@deepseek-ai/dsh", "run",
            "--prompt", task,
            "--model", profile.model,
            "--provider", profile.provider,
            "--cwd", str(workdir),
            "--config", str(cordis_path),
            "--output", "json",
            "--no-open",
        ]
        env = {**os.environ, **self._profile_env(profile, workdir, cordis_path)}
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=str(workdir), env=env,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise

        out = stdout.decode("utf-8", "replace")
        err = stderr.decode("utf-8", "replace")
        if process.returncode != 0:
            return DSHResult(ok=False, text=out, profile=profile.name, stderr=err,
                             error=f"dsh CLI exit {process.returncode}: {err[:500]}")
        parsed = extract_json_block(out) or {}
        text = str(parsed.get("final_response") or parsed.get("result") or out)
        return DSHResult(ok=True, text=text, profile=profile.name, stderr=err,
                         finish_reason=parsed.get("finish_reason"))


# Content-block types that carry the model's private scratchpad rather than its
# reply.  Deliberately NOT collected: a reasoning trace routinely contains a
# *draft* ```json block, and `extract_json_block` returns the first fence it
# finds -- folding reasoning into the text would let a discarded draft outrank
# the real answer.  These tokens are already accounted for separately as
# `reasoningTokens` in `_usage_from_events`.
_REASONING_BLOCKS = frozenset(
    {"thinking", "redacted_thinking", "reasoning", "reasoning_content"}
)

# Every spelling of "this block is a tool call" observed across adapters.  The
# SDK does not document its content-block shape and it varies by provider, so
# recognition is liberal: a block we fail to recognise is a block whose tokens
# vanish, which is the exact failure this function exists to prevent.
_TOOL_CALL_BLOCKS = frozenset(
    {"tool_use", "tool_call", "toolcall", "function_call", "functioncall", "invoke"}
)
_TOOL_NAME_KEYS = ("name", "toolName", "tool_name", "tool")
_TOOL_ARG_KEYS = ("input", "args", "arguments", "parameters", "params")


def _tool_call_action(block: dict[str, Any]) -> dict[str, Any] | None:
    """Read one tool-call content block as a `{"tool": ..., "args": {...}}` action.

    Returns None for any block that is not a recognisable tool call.
    """
    kind = str(block.get("type") or "").replace("-", "_").lower()
    if kind not in _TOOL_CALL_BLOCKS:
        return None
    # OpenAI-style blocks nest the payload one level down under `function`;
    # Anthropic-style blocks carry it inline.  Read both from one mapping.
    nested = block.get("function")
    source: dict[str, Any] = {**block, **nested} if isinstance(nested, dict) else block

    name = next(
        (source[key].strip() for key in _TOOL_NAME_KEYS
         if isinstance(source.get(key), str) and source[key].strip()),
        "",
    )
    if not name:
        return None

    raw = next((source[key] for key in _TOOL_ARG_KEYS if source.get(key) is not None), {})
    if isinstance(raw, str):
        # `arguments` is a JSON *string* in the OpenAI shape.  A call whose
        # arguments failed to decode is still worth surfacing by name: the
        # Developer's next STATUS block re-anchors it, and an unknown-tool
        # error is a cheaper turn than a silent no-op.
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    return {"tool": name, "args": raw if isinstance(raw, dict) else {}}


def collect_assistant_text(events: list[dict[str, Any]]) -> str:
    """Concatenate EVERY assistant text message in the run, in order.

    WHY NOT `RunResult.final_response`: the SDK documents that field as "the
    last committed root-session assistant text in the interval". That is the
    right answer for a chat turn and the wrong one for document production. An
    agent with tools naturally emits: (1) the deliverable, (2) a tool call,
    (3) a short "done" acknowledgement -- and `final_response` returned only (3).

    This was not hypothetical. The Architect's first real run produced three
    assistant messages of 0, 15231 and 494 characters; `final_response` returned
    the 494-character acknowledgement, so a complete design document was
    discarded and `migration.sql` was written empty.

    Joined with blank lines so a fenced ```json block that opened in one message
    and closed in another still parses.

    TOOL-CALL BLOCKS ARE RECOVERED, NOT DROPPED (regression, runs_smoke/iter_1).
    This used to keep only blocks with `type == "text"`.  The Developer profile
    mounts no capabilities, so the model is handed an agentic persona with an
    empty tool schema -- and it answers in its native tool-call format anyway.
    Those blocks matched no mounted tool, were discarded here, and `dev_think`
    saw a bare thought with no action: "developer produced no parseable action".
    The token accounting is what proved it. The Architect reconciles exactly
    (4898 output - 2353 reasoning = 2545 content tokens ~ its 10260 captured
    characters), while two Developer turns reported 233 and 44 content tokens
    and landed 59 and 1 characters here. Roughly nine tenths of what the model
    said was being thrown away before any parser saw it.

    Recovered calls are re-emitted as fenced ```json action blocks -- the shape
    `dev_think` already parses -- and are appended AFTER the prose rather than
    interleaved. That ordering is load-bearing: `extract_json_block` takes the
    first fence in the text, so a node that produced a real deliverable still
    wins, and only a turn that produced no fence of its own falls through to a
    recovered call.
    """
    parts: list[str] = []
    recovered: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("type") != "assistant/message":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        message = data.get("message") if isinstance(data.get("message"), dict) else data
        content = message.get("content")
        if not isinstance(content, list):
            continue
        chunks: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = str(block.get("type") or "").replace("-", "_").lower()
            if kind in _REASONING_BLOCKS:
                continue
            action = _tool_call_action(block)
            if action is not None:
                recovered.append(action)
                continue
            # Any remaining block carrying text: `text` is the documented type,
            # but adapters also emit `output_text` and friends, and a block whose
            # only payload is prose reads the same either way.
            if isinstance(block.get("text"), str):
                chunks.append(block["text"])
        text = "".join(chunks)
        if text.strip():
            parts.append(text)

    if recovered:
        log.info(
            "dsh: recovered %d tool-call block(s) the text stream would have "
            "dropped: %s", len(recovered), ", ".join(a["tool"] for a in recovered),
        )
        parts.extend(
            "```json\n" + json.dumps(action, ensure_ascii=False) + "\n```"
            for action in recovered
        )
    return "\n\n".join(parts)


class _ProgressLogger:
    """Heartbeat for a long harness call.

    A node that legitimately takes four minutes is indistinguishable from a hung
    one when the only log lines are "entered" and "exited". This prints a line
    at most every `interval_s` seconds so a real run is watchable.

    Invoked from the SDK's own thread (the call is wrapped in
    `asyncio.to_thread`); `logging` is thread-safe, and nothing here touches the
    event loop.
    """

    def __init__(self, profile_name: str, interval_s: float = 20.0) -> None:
        self.profile_name = profile_name
        self.interval_s = interval_s
        self.started = time.monotonic()
        self.last = self.started
        self.events = 0
        self.steps = 0
        self.tools: list[str] = []

    def __call__(self, notification: Any) -> None:
        self.events += 1
        payload = getattr(notification, "payload", None) or {}
        event = payload.get("event") if isinstance(payload, dict) else None
        if isinstance(event, dict):
            kind = event.get("type")
            if kind == "step/start":
                self.steps += 1
            elif kind == "tool/call":
                data = event.get("data") or {}
                name = data.get("name") or (data.get("call") or {}).get("name")
                if name:
                    self.tools.append(str(name))

        now = time.monotonic()
        if now - self.last >= self.interval_s:
            self.last = now
            log.info(
                "   %s still working: %.0fs elapsed, step %d, %d events%s",
                self.profile_name, now - self.started, self.steps, self.events,
                f", tools: {', '.join(self.tools[-3:])}" if self.tools else "",
            )

    def done(self) -> None:
        log.info(
            "   %s finished in %.0fs (%d steps, %d events%s)",
            self.profile_name, time.monotonic() - self.started, self.steps, self.events,
            f", tools: {', '.join(self.tools)}" if self.tools else "",
        )


# Token-usage key names, most authoritative first.
#
# VERIFIED against real session logs (runs_smoke/iter_1), replacing an
# ASSUMPTION that was wrong in a way that silently disarmed the budget guard.
# The harness carries usage on `assistant/message` under `data.usage`, in
# CAMEL CASE:
#
#     {"inputTokens": 692, "outputTokens": 4898,
#      "cacheReadTokens": 0, "reasoningTokens": 2353}
#
# The old code looked only for `prompt_tokens` / `completion_tokens` /
# `input_tokens` / `output_tokens`, so every dsh call accounted as ZERO. That
# made `MAX_TOTAL_TOKENS` unreachable for the Architect, Developer and Critic --
# a cost cap that cannot fire is not a cap -- and reported `tokens=0` on every
# node span. The snake_case spellings are kept as fallbacks so the http-shaped
# payload still reads correctly.
_INPUT_TOKEN_KEYS = ("inputTokens", "prompt_tokens", "input_tokens")
_OUTPUT_TOKEN_KEYS = ("outputTokens", "completion_tokens", "output_tokens")
_TOTAL_TOKEN_KEYS = ("totalTokens", "total_tokens")


def _first_int(usage: dict[str, Any], keys: tuple[str, ...]) -> int:
    """The first key present, as an int.  Missing or unparseable reads as 0."""
    for key in keys:
        if key in usage:
            try:
                return int(usage[key] or 0)
            except (TypeError, ValueError):
                return 0
    return 0


def _usage_from_events(events: list[dict[str, Any]]) -> dict[str, int]:
    """Sum token usage out of the harness's event stream.

    `reasoningTokens` is deliberately NOT added on top of `outputTokens`: the
    real logs show it is a SUBSET of it, not a sibling. The Architect's turn
    reported 4898 output and 2353 reasoning, and its captured text was ~2545
    tokens -- 2353 + 2545 = 4898 exactly. Adding them would have inflated that
    node's billed output by 92%.

    `cacheReadTokens` is likewise not added to the input: providers count cached
    reads inside `inputTokens` already.
    """
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for event in events:
        data = event.get("data") if isinstance(event, dict) else None
        usage = data.get("usage") if isinstance(data, dict) else None
        if not isinstance(usage, dict):
            continue
        totals["input_tokens"] += _first_int(usage, _INPUT_TOKEN_KEYS)
        totals["output_tokens"] += _first_int(usage, _OUTPUT_TOKEN_KEYS)
        totals["total_tokens"] += _first_int(usage, _TOTAL_TOKEN_KEYS)
    if not totals["total_tokens"]:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    return totals


class MockDSHClient:
    """Offline stand-in with the identical interface.

    Returns canned, schema-valid output per node role so that the full graph --
    including the Developer's ReAct loop and the evaluator fan-out -- runs end
    to end with no network, no GPUs and no Node runtime.  Content comes from
    `mocks/dsh_responses.py` so that the canned text lives next to the rest of
    the mock fixtures rather than being buried in the client.
    """

    def __init__(self, latency_s: float = 0.0) -> None:
        self.latency_s = latency_s
        self.calls: list[dict[str, Any]] = []

    async def run(
        self, profile: DSHProfile, task: str, workdir: Path, timeout_s: int
    ) -> DSHResult:
        from mocks.dsh_responses import canned_response

        self.calls.append({"profile": profile.name, "task": task[:400], "workdir": str(workdir)})
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        # Rendering the cordis file even in mock mode keeps the least-privilege
        # composition under test: a profile that generates invalid YAML fails
        # offline instead of in production.
        render_cordis_config(profile, Path(workdir) / ".cordis")
        text, ok = canned_response(profile.name, task)
        return DSHResult(
            ok=ok, text=text, profile=profile.name, finish_reason="completed" if ok else "error",
            session_id=f"mock-{profile.name}-{len(self.calls)}", duration_s=self.latency_s,
            usage={"input_tokens": 1200, "output_tokens": 400, "total_tokens": 1600},
            error=None if ok else "mock failure injected",
        )


# ==========================================================================
# Process-wide accessor + the brief's required free function
# ==========================================================================

_DSH_CLIENT: DSHClientProtocol | None = None


def get_dsh_client() -> DSHClientProtocol:
    """The only place that decides mock-vs-real for the harness."""
    global _DSH_CLIENT
    if _DSH_CLIENT is None:
        _DSH_CLIENT = MockDSHClient() if config.MOCK_MODE else RealDSHClient()
        log.info("dsh client: %s", type(_DSH_CLIENT).__name__)
    return _DSH_CLIENT


def reset_dsh_client() -> None:
    global _DSH_CLIENT
    _DSH_CLIENT = None


async def run_dsh(
    agent_profile: DSHProfile,
    task: str,
    workdir: Path,
    timeout_s: int = int(config.DSH_DEFAULT_TIMEOUT_S),
) -> DSHResult:
    """The single async abstraction every node calls (brief section 3.1)."""
    profile = agent_profile if agent_profile.workdir else agent_profile.with_workdir(workdir)
    return await get_dsh_client().run(profile, task, Path(workdir), timeout_s)
