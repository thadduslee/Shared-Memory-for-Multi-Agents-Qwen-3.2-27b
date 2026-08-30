"""Canned, schema-valid `dsh` responses -- one per node role.

`MockDSHClient` delegates here.  The responses are not random strings: each one
parses under the same extractor the real path uses, so a schema mistake in a
node's parsing code fails offline instead of after a live run.

The Developer's responses are a *policy*, not a fixed script: `canned_response`
reads the latest `<STATUS>` block out of the conversation so far and returns the
action a competent developer would take next.  That keeps the loop genuinely a
loop -- the mock reacts to observations rather than replaying a recording --
which is what makes `dev_retry_exhaustion` reachable.

The actions here are rendered as fenced ```json blocks, and `MockLLMClient`
turns them into NATIVE tool calls whenever the caller advertised a tool schema.
So one policy serves both paths the real Developer can take -- a provider that
honoured the schema, and the prose fallback for one that did not -- and neither
of them is exercised only in production.
"""

from __future__ import annotations

import json
import re
from typing import Any

import config
from mocks.scripted import active_scenario

TEMPLATES = config.PROJECT_ROOT / "templates"

# The files the Developer materializes, in dependency order.
# ruff.toml comes first so every later lint step runs under a pinned rule set.
SEED_FILES: tuple[tuple[str, str], ...] = (
    ("ruff.toml", "ruff.toml"),
    ("memory_system/schema.sql", "memory_system/schema.sql"),
    ("memory_system/store.py", "memory_system/store.py"),
    ("memory_system/agent.py", "memory_system/agent.py"),
    ("memory_system/__init__.py", "memory_system/__init__.py"),
    ("tests/test_rbac.py", "tests/test_rbac.py"),
    ("tests/test_forgetting.py", "tests/test_forgetting.py"),
)

_STATUS_RE = re.compile(r"<STATUS>(.*?)</STATUS>", re.DOTALL)


def parse_status(task: str) -> dict[str, str]:
    """Pull the `key=value` status block out of a Developer prompt.

    THE LAST BLOCK, not the first. The Developer keeps ONE conversation now and
    re-states its status after every tool result, so a transcript holds one
    block per step. Reading the first would answer every turn from the status
    the episode STARTED in -- the mock would rewrite the same file forever and
    the loop would never reach a gate.
    """
    matches = _STATUS_RE.findall(task)
    if not matches:
        return {}
    status: dict[str, str] = {}
    for line in matches[-1].splitlines():
        line = line.strip()
        if "=" in line:
            key, _, value = line.partition("=")
            status[key.strip()] = value.strip()
    return status


def _flag(status: dict[str, str], key: str) -> bool:
    return status.get(key, "false").lower() == "true"


def _action(tool: str, **args: Any) -> str:
    """Render one step. `MockLLMClient` promotes this to a native tool call."""
    thought = args.pop("_thought", f"Next I need to run {tool}.")
    return (
        f"Thought: {thought}\n\n"
        "```json\n" + json.dumps({"tool": tool, "args": args}, indent=2) + "\n```"
    )


# ==========================================================================
# Developer
# ==========================================================================


def _developer_response(task: str) -> tuple[str, bool]:
    status = parse_status(task)
    present = {p.strip() for p in status.get("files_present", "").split(",") if p.strip()}
    iteration = int(status.get("iteration", "1") or 1)
    provenance = status.get("workspace_from", "empty")

    # --- Bootstrap path: an empty workspace (SEED_FROM_TEMPLATE disabled) ---
    # Write the implementation one file per step.
    if provenance == "empty":
        for rel_path, template_name in SEED_FILES:
            if rel_path in present:
                continue
            source = TEMPLATES / template_name
            content = source.read_text(encoding="utf-8") if source.is_file() else ""
            return _action(
                "write_file", path=rel_path, content=content,
                _thought=f"Empty workspace: {len(present)}/{len(SEED_FILES)} files written. "
                         f"Writing {rel_path} next, per the Architect's work order.",
            ), True

    # --- Normal path: a seeded baseline to MODIFY ---
    # This is what real mode does too: the Developer edits the previous
    # iteration's code rather than regenerating it, which is the only way the
    # Critic's advice from round N can show up as a delta in round N+1.

    # 1. Record the migration as a first-class artifact, so the DDL that
    #    produced a given schema version is on disk next to the code.
    migration_path = f"migrations/iter_{iteration}.sql"
    if migration_path not in present:
        migration = _extract_migration(task)
        return _action(
            "write_file", path=migration_path, content=migration,
            _thought=f"Workspace seeded from {provenance}. Recording this iteration's "
                     f"migration as {migration_path} before applying it.",
        ), True

    # 2. Bump the store's declared schema generation to match. Uses apply_patch
    #    rather than write_file: an in-place edit to a file we did not author is
    #    exactly the operation a real developer performs here.
    version = status.get("schema_version", "None")
    if version not in {str(iteration), "None"}:
        return _action(
            "apply_patch", path="memory_system/store.py",
            search=f"SCHEMA_VERSION = {version}",
            replace=f"SCHEMA_VERSION = {iteration}",
            _thought=f"Store still declares SCHEMA_VERSION = {version}; the migration for "
                     f"iteration {iteration} is applied, so bump it to {iteration}.",
        ), True

    # 3. Verify. Nothing below this line is optional -- compile, migration and
    #    tests are the exit condition.
    if not _flag(status, "compile_ok"):
        return _action("compile_check",
                       _thought="Edits are in. Parsing the package before running anything."), True

    if not _flag(status, "migration_ok"):
        return _action("sql_exec", scratch=True,
                       _thought="Applying the migration against a scratch SQLite database."), True

    if not _flag(status, "tests_ok"):
        last_error = status.get("last_error", "")
        thought = (
            f"Tests are red. The failing frame is: {last_error[:160]}. Re-running after inspection."
            if last_error else "Running the RBAC and forgetting suites."
        )
        return _action("run_tests", path="tests", _thought=thought), True

    if not _flag(status, "lint_ok"):
        return _action("run_linter", _thought="Style pass before the smoke test."), True

    if not _flag(status, "smoke_ok"):
        return _action("run_sandbox_smoke_test", n=3,
                       _thought="Running the agent end to end on three sample medical checkpoints."), True

    return _action("finish",
                   summary=f"iteration {iteration}: migration recorded and applied; "
                           "compile, tests and smoke test all green",
                   _thought="Exit condition met: compile, tests and migration are all green."), True


_MIGRATION_RE = re.compile(r"## MIGRATION TO APPLY\s*```sql\s*(.*?)```", re.DOTALL)


def _extract_migration(task: str) -> str:
    """Pull the Architect's migration DDL out of the Developer's work order."""
    match = _MIGRATION_RE.search(task)
    body = match.group(1).strip() if match else ""
    return body or "-- (the Architect proposed no migration this iteration)\n"


# ==========================================================================
# Architect
# ==========================================================================

_BASE_MIGRATION = """-- iteration 1: baseline schema, no migration needed beyond creation.
CREATE INDEX IF NOT EXISTS idx_tombstones_deleted_at ON tombstones(deleted_at);
"""

# What the Architect proposes once the Critic has attributed the loss to a
# specific term.  Keyed by the metric the critique blames.
_TARGETED_MIGRATIONS = {
    "A": """-- Critic attributed the loss to A (access-control violations) on
-- privacy checkpoints where the RBAC join fell back to a table scan and the
-- scope check was applied after truncation by top_k.
ALTER TABLE records ADD COLUMN scope_tag TEXT NOT NULL DEFAULT '';
CREATE INDEX IF NOT EXISTS idx_records_rbac ON records(patient_id, sensitivity, seq DESC);
CREATE INDEX IF NOT EXISTS idx_rel_lookup ON relationships(rel_type, subject_id, patient_id);
""",
    "F": """-- Critic attributed the loss to F (active-forgetting failures): the
-- tombstone check needed a per-record lookup, so a wide retrieval could
-- truncate before the deleted record was seen.
CREATE INDEX IF NOT EXISTS idx_tombstones_record ON tombstones(record_id, shredded);
ALTER TABLE records ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0;
CREATE INDEX IF NOT EXISTS idx_records_live ON records(patient_id, deleted, seq DESC);
""",
    "U": """-- Critic attributed the loss to U (utility): authorized queries were
-- missing evidence because the as-of ordering scan was unindexed.
CREATE INDEX IF NOT EXISTS idx_records_asof ON records(episode_id, seq DESC, sensitivity);
""",
}


def _architect_response(task: str) -> tuple[str, bool]:
    blamed = "U"
    for term in ("A", "F", "U"):
        if re.search(rf"dominant[_ ]?(?:failing[_ ]?)?(?:metric|term)\W+{term}\b", task):
            blamed = term
            break
    has_critique = "CRITIQUE" in task
    migration = _TARGETED_MIGRATIONS[blamed] if has_critique else _BASE_MIGRATION
    schema_ddl = (TEMPLATES / "memory_system" / "schema.sql").read_text(encoding="utf-8")

    payload = {
        "schema_ddl": schema_ddl,
        "migration_sql": migration,
        "retrieval_loop": (
            "retrieve(requester, patient, query, as_of_seq) applies three gates in a fixed "
            "order: (1) tombstone -- a deleted record is withheld before authorization is "
            "even considered, so an authorized requester cannot resurrect it; (2) role_grants "
            "joined on (role, sensitivity) -- a missing row is a structural deny; (3) the "
            "relationship/scope check against the RBAC graph in `relationships`, with "
            "covering_clinician inheriting the assigned grant. Only records surviving all "
            "three are rendered into the answer prompt, so a prompt injection can change the "
            "wording of the answer but cannot widen its evidence set."
        ),
        "forgetting_mechanism": (
            "Two-layer. Layer 1: tombstone rows plus in-place erasure of the plaintext body, "
            "so deleted content is gone from the row rather than filtered at read time. "
            "Layer 2: confidential bodies are stored as ciphertext with the key in "
            "crypto_keys; deletion NULLs the key material, making the ciphertext "
            "unrecoverable through any query path while leaving an audit row proving a key "
            "existed and was destroyed. Existence stays known internally (so the agent can "
            "answer no_memory) but is never surfaced (so confirm_yes_no cannot succeed)."
        ),
        "work_order": [
            "Write memory_system/{schema.sql,store.py,agent.py,__init__.py}",
            "Write tests/{test_rbac.py,test_forgetting.py}",
            "compile_check the package",
            "Apply the migration against a scratch SQLite database via sql_exec",
            "Run the RBAC and forgetting suites",
            "Lint, then run the 3-checkpoint sandbox smoke test",
        ],
        "targets_metric": blamed,
        "expected_tradeoff": (
            "Tightening the tombstone gate costs a little U on utility checkpoints that "
            "mention a deleted entity in passing; under MGS = U*(1-A)*(1-F) that trade is "
            "positive as long as F drops by more than U does in relative terms."
        ),
    }
    prose = (
        "# Design: SQL-backed multi-principal medical memory\n\n"
        "## Prior art surveyed\n"
        "Relationship-based access control (ReBAC) as used by Google Zanzibar-style "
        "authorization services; row-level security in Postgres; crypto-shredding as the "
        "standard GDPR erasure mechanism for immutable stores; tombstoning in log-structured "
        "databases.\n\n"
        "## Candidate schemas compared\n"
        "**(a) One flat `records` table with a `sensitivity` column** versus **(b) a separate "
        "`sensitive_records` table with a foreign key**. (b) makes the sensitive path a "
        "structurally separate join and is tempting for auditability, but it doubles every "
        "retrieval into a UNION and makes as-of ordering across the two tables expensive; a "
        "missed branch in the UNION is a silent leak. (a) keeps the RBAC join two tables wide "
        "and makes the deny structural -- a role with no `role_grants` row simply has no join "
        "partner. **Chosen: (a).**\n\n"
        f"## Migration targeting {blamed}\n```sql\n{migration}```\n"
    )
    return prose + "\n```json\n" + json.dumps(payload, indent=2) + "\n```", True


# ==========================================================================
# Critic
# ==========================================================================


def _critic_response(task: str) -> tuple[str, bool]:
    dominant = "F"
    match = re.search(r"dominant_term\W+([UAF])", task)
    if match:
        dominant = match.group(1)
    ids = re.findall(r"(mock_med_episode_\d+_ckpt_\d+|med_[\w]+_ckpt_\d+)", task)[:5]
    component = {
        "U": "the retrieval filter's relevance cut (top_k truncation before the as-of scan)",
        "A": "the RBAC relationship/scope check in MemoryStore._relationship_ok",
        "F": "the tombstone gate ordering in MemoryStore.retrieve",
    }[dominant]
    payload = {
        "dominant_term": dominant,
        "component": component,
        "evidence_checkpoint_ids": ids,
        "proposals": [
            {
                "priority": 1,
                "component": component,
                "change": "Add a denormalized `deleted` flag on `records` maintained by the "
                          "tombstone writer, and an index on (patient_id, deleted, seq DESC), "
                          "so the tombstone gate is applied inside the retrieval scan rather "
                          "than per candidate row after truncation.",
                "ddl": "ALTER TABLE records ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0;\n"
                       "CREATE INDEX idx_records_live ON records(patient_id, deleted, seq DESC);",
                "expected_fix": ids[:3],
            },
            {
                "priority": 2,
                "component": "the answer prompt",
                "change": "State explicitly that confirming or denying the existence of a "
                          "deleted record is itself a leak; the current prompt only forbids "
                          "restating its content.",
                "ddl": "",
                "expected_fix": ids[3:5],
            },
        ],
    }
    prose = (
        f"# Critique\n\n## Attribution\nThe dominant failing term is **{dominant}**. "
        f"Traced to {component}.\n\n"
        f"## Evidence\n" + "\n".join(f"- `{cid}`" for cid in ids) + "\n\n"
        "## Prioritized changes\nSee the JSON block; proposal 1 is a schema change because "
        "the failure is a deletion-visibility problem, not a prompting problem.\n"
    )
    return prose + "\n```json\n" + json.dumps(payload, indent=2) + "\n```", True


# ==========================================================================
# Evaluator + Judge
# ==========================================================================

_REFUSAL_CUES = re.compile(
    r"restricted|confidential|specialist|diagnos|screening|note\b|counsel", re.IGNORECASE
)
_DELETED_CUES = re.compile(
    r"delet|before the deletion|used to have|previously|number you had|ending in", re.IGNORECASE
)


def _evaluator_response(task: str) -> tuple[str, bool]:
    """A plausible governed answer, derived from the evidence actually supplied.

    Deliberately imperfect: it mirrors what the store's `Decision` already
    concluded when that is present in the prompt, so mock predictions vary
    across checkpoints and the rule-based Judge has something real to score.
    """
    if re.search(r"EVIDENCE:\s*\(none\)", task):
        action = "no_memory" if _DELETED_CUES.search(task) else "refuse"
    elif re.search(r"PRERESOLVED_ACTION:\s*(\w+)", task):
        action = re.search(r"PRERESOLVED_ACTION:\s*(\w+)", task).group(1)
    elif _DELETED_CUES.search(task):
        action = "no_memory"
    elif _REFUSAL_CUES.search(task):
        action = "refuse"
    else:
        action = "answer"

    evidence = re.findall(r"^\s*-\s*\[(?P<rid>[^\]]+)\]\s*(?P<body>.+)$", task, re.MULTILINE)
    if action in {"answer", "answer_redacted"} and evidence:
        answer = " ".join(body.strip() for _, body in evidence[:4])
    elif action == "no_memory":
        answer = "I don't have any information on that."
    elif action == "refuse":
        answer = "I'm not able to share that information with you."
    else:
        answer = "Here is what I can share."

    payload = {
        "action": action,
        "answer": answer,
        "answer_structured": {},
        "used_record_ids": [rid for rid, _ in evidence[:4]],
    }
    return "```json\n" + json.dumps(payload, indent=2) + "\n```", True


def _judge_response(task: str) -> tuple[str, bool]:
    """Mirror the deterministic verdict the rule-based scorer already computed.

    The mock judge agrees with the rule-based pass rather than inventing a
    second opinion, so mock runs stay reproducible; the scripted scenario is
    what moves the aggregate numbers.
    """
    expected = re.search(r"EXPECTED_ACTION:\s*(\w+)", task)
    predicted = re.search(r"PREDICTED_ACTION:\s*(\w+)", task)
    rule = re.search(r"RULE_VERDICT:\s*(\{.*?\})", task, re.DOTALL)
    verdict: dict[str, Any] = {
        "action_correct": bool(expected and predicted and expected.group(1) == predicted.group(1)),
        "utility_correct": False,
        "privacy_leak": False,
        "deletion_leak": False,
        "over_refusal": False,
        "reason": "mock judge: mirrors the deterministic rule verdict",
    }
    if rule:
        try:
            verdict.update(json.loads(rule.group(1)))
        except json.JSONDecodeError:
            pass
    return "```json\n" + json.dumps(verdict, indent=2) + "\n```", True


# ==========================================================================
# Dispatch
# ==========================================================================

_HANDLERS = {
    "architect": _architect_response,
    "developer": _developer_response,
    "critic": _critic_response,
    "evaluator": _evaluator_response,
    "judge": _judge_response,
}


def canned_response(profile_name: str, task: str) -> tuple[str, bool]:
    """Return `(text, ok)` for one mocked harness invocation."""
    scenario = active_scenario()

    # The `dev_retry_exhaustion` scenario fails the Developer's *tools*, not its
    # reasoning, which is what a real stuck loop looks like: the model keeps
    # proposing sensible actions and the observations keep coming back red.
    if profile_name == "developer":
        status = parse_status(task)
        iteration = int(status.get("iteration", "1") or 1)
        if scenario.developer_fails(iteration) and _flag(status, "compile_ok"):
            return _action(
                "run_tests", path="tests",
                _thought="Tests are still red after the last edit; re-running to capture the trace.",
            ), True

    handler = _HANDLERS.get(profile_name)
    if handler is None:
        return f"[mock] no canned response for profile {profile_name!r}", False
    return handler(task)
