"""GateMem's own prompt files, rendered for our Evaluator and Judge.

WHY THE FILES AND NOT OUR OWN WORDING
-------------------------------------
The Evaluator and the Judge are not ours to design.  The Evaluator IS the system
under test as the benchmark defines it, and the Judge IS the benchmark's scoring
instrument; every number this loop reports means "GateMem's metric" only insofar
as both of them behave the way GateMem's own harness makes them behave.  A
prompt we wrote ourselves -- however much better it scores -- silently redefines
the measurement, and a result measured against a redefined instrument cannot be
compared with the paper.

So these two prompts are LOADED FROM `bench/prompts/`, not written here:

    query_prompt.txt   the multi-party assistant under evaluation  -> Evaluator
    judge_prompt.txt   the impartial grader                        -> Judge

`judge_prompt_gatemem.txt` is deliberately NOT used.  It is the longer,
GateMem-specific variant; `judge_prompt.txt` is the one this project scores
against, and mixing the two would make our numbers comparable to neither.

The Architect, Developer and Critic are the opposite case: they are OUR research
loop, they have no counterpart in the paper, and their prompts live with the
nodes that own them.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not paraphrase, repair or extend the files.  The only transformations
are the ones GateMem's own runner performs -- splitting the query template on
`[REQUEST CONTEXT]` into a system and a user half, and substituting the
documented `{placeholders}`.  When a file changes upstream, this module changes
with it by construction.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import config

# ==========================================================================
# Loading
# ==========================================================================

_QUERY_PROMPT = "query_prompt.txt"
# NOT judge_prompt_gatemem.txt. See the module docstring.
_JUDGE_PROMPT = "judge_prompt.txt"


def _prompts_dir() -> Path:
    return Path(config.GATEMEM_REPO) / "bench" / "prompts"


@lru_cache(maxsize=8)
def _load(filename: str, override: str) -> str:
    """The prompt text, from the override path when one is set.

    Cached because the Evaluator renders one of these per checkpoint -- 579 per
    iteration -- and re-reading the file each time buys nothing.
    """
    path = Path(override) if override else _prompts_dir() / filename
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(
            f"GateMem prompt {path} is unreadable ({exc}). Set GATEMEM_REPO to a "
            f"checkout containing bench/prompts/, or point "
            f"{'EVALUATOR_PROMPT_FILE' if filename == _QUERY_PROMPT else 'JUDGE_PROMPT_FILE'} "
            f"at the file you want used."
        ) from exc


def split_system_user(template: str) -> tuple[str, str]:
    """GateMem's own split, reproduced exactly.

    Mirrors `BaseAgent._split_system_user` in bench/agents/base.py: everything
    before `[REQUEST CONTEXT]` is the system half with the `[SYSTEM]` marker
    stripped, and the rest -- marker included -- is the user half. A template
    without both markers is sent whole as the user message, which is what the
    benchmark does too.
    """
    if "[SYSTEM]" in template and "[REQUEST CONTEXT]" in template:
        before, after = template.split("[REQUEST CONTEXT]", 1)
        return before.replace("[SYSTEM]", "").strip(), ("[REQUEST CONTEXT]" + after).strip()
    return "", template.strip()


# ==========================================================================
# Evaluator -- bench/prompts/query_prompt.txt
# ==========================================================================


def _bench_domain_helpers() -> Any:
    """`bench.domains`, imported lazily so the orchestrator runs without it.

    The policy text and the domain label are part of the measured prompt, so
    they are taken from the benchmark rather than restated here.
    """
    root = str(Path(config.GATEMEM_REPO))
    if root not in sys.path:
        sys.path.insert(0, root)
    from bench import domains

    return domains


def format_relationship_facts(relationships: list[dict[str, Any]] | None) -> str:
    """The requester-relevant relationship lines, in GateMem's format.

    The FILTERING is done upstream, in the evaluation shard, because that is
    where the episode is loaded; this only formats what it is handed. Mirrors
    `BaseAgent._format_relationship_facts`, including its `(none)` sentinel --
    an empty block would read to the model as "this requester has no
    relationships", which is a different and stronger claim than "none were
    supplied".
    """
    if not relationships:
        return "(none)"
    domains = _bench_domain_helpers()
    return "\n".join(domains.format_relationship_fact(rel or {}) for rel in relationships)


def format_memory_block(evidence: list[dict[str, Any]] | None) -> str:
    """The retrieved-memory block, in GateMem's format.

    Mirrors `BaseAgent._format_retrieved_memory_block`: one numbered line per
    record, speaker attributed, newlines flattened so one record cannot look
    like several.
    """
    if not evidence:
        return "(none)"
    lines: list[str] = []
    for index, item in enumerate(evidence, 1):
        speaker = str(item.get("principal_id") or item.get("speaker") or "unknown")
        text = str(item.get("text") or "").strip().replace("\n", " ")
        lines.append(f"Memory {index} (speaker={speaker}): {text}")
    return "\n".join(lines)


def render_query_prompt(record: dict[str, Any]) -> tuple[str, str]:
    """`(system, user)` for one checkpoint, from query_prompt.txt.

    `record` is an evaluation-shard row: asker, query_text, evidence, and the
    `relationships` the shard filtered for this requester.
    """
    domains = _bench_domain_helpers()
    domain_key = domains.normalize_domain_name(record.get("domain") or "medical")
    asker = record.get("asker") or {}

    variables = {
        "asker_principal_id": asker.get("principal_id", "?"),
        "asker_role": asker.get("role", "?"),
        "relationship_facts_block": format_relationship_facts(record.get("relationships")),
        "retrieved_memory_block": format_memory_block(record.get("evidence")),
        "query_text": record.get("query_text", ""),
        "domain_label": domains.get_domain_label(domain_key),
        "global_access_policy_block": domains.get_query_policy_block(domain_key),
    }
    system_template, user_template = split_system_user(
        _load(_QUERY_PROMPT, config.EVALUATOR_PROMPT_FILE)
    )
    try:
        return system_template.format(**variables), user_template.format(**variables)
    except KeyError as exc:
        raise ValueError(
            f"query_prompt.txt wants a variable this adapter does not supply: {exc.args[0]!r}"
        ) from exc


# ==========================================================================
# Judge -- bench/prompts/judge_prompt.txt
# ==========================================================================

# The keys judge_prompt.txt asks for. It returns `notes`, not `reason`, and asks
# for NO over_refusal at all -- see `normalise_verdict`.
JUDGE_KEYS = ("action_ok", "utility_ok", "privacy_leak", "deletion_leak", "notes")


def render_judge_prompt(
    prediction: dict[str, Any],
    annotation: dict[str, Any],
    visible: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """`(system, user)` for one verdict, from judge_prompt.txt.

    The file is one document with no `[SYSTEM]` marker, so it is sent whole as
    the user message and the system half is empty -- which is what
    `split_system_user` does with it, and what the benchmark does with it.

    TWO SOURCES, because the fields the prompt asks for live in two places.
    `annotation` is the HIDDEN subset -- query_type, attack_type,
    expected_action, judge_spec, leak_targets -- which only the Judge may read.
    `visible` is the rest of the checkpoint, the part the agent under test also
    saw: the requester, the question, the as-of turn. GateMem's prompt shows
    both, and without `visible` the CASE block degrades to "requester: ? (?)"
    with an empty question, which is a materially easier grading problem than
    the benchmark poses.
    """
    visible = visible or {}
    output = prediction.get("output") or {}
    spec = annotation.get("judge_spec") if isinstance(annotation.get("judge_spec"), dict) else {}
    try:
        structured = json.dumps(output.get("answer_structured") or {}, ensure_ascii=False)
    except (TypeError, ValueError):
        structured = str(output.get("answer_structured") or {})

    variables = {
        "query_type": annotation.get("query_type"),
        "attack_type": annotation.get("attack_type") or "(none)",
        "as_of_turn_id": visible.get("as_of_turn_id") or "(unknown)",
        "asker_principal_id": (visible.get("asker") or {}).get("principal_id", "?"),
        "asker_role": (visible.get("asker") or {}).get("role", "?"),
        "query_text": visible.get("query_text", ""),
        "pred_action": output.get("action"),
        "pred_answer": output.get("answer"),
        "pred_answer_structured": structured,
        "judge_spec_json": json.dumps(spec or {}, ensure_ascii=False, indent=1),
        "leak_targets_json": json.dumps(annotation.get("leak_targets") or [], ensure_ascii=False),
    }
    system_template, user_template = split_system_user(
        _load(_JUDGE_PROMPT, config.JUDGE_PROMPT_FILE)
    )
    try:
        return system_template.format(**variables), user_template.format(**variables)
    except KeyError as exc:
        raise ValueError(
            f"judge_prompt.txt wants a variable this adapter does not supply: {exc.args[0]!r}"
        ) from exc


def normalise_verdict(raw: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any]:
    """GateMem's verdict shape, mapped onto the one our merge expects.

    TWO GENUINE DIFFERENCES, and neither is a judgement call we get to make:

    * `notes` instead of `reason`. Renamed, not rewritten. The file also forbids
      the Judge from quoting sensitive values in it, so it is a category label
      rather than a quotation -- which is what we want in an artifact anyway.
    * NO `over_refusal`. GateMem's judge does not score it; our `OR` metric and
      the merge both expect it. It is taken from the RULE pass, which computes
      it deterministically from the expected and predicted actions, so the
      number keeps a definition rather than being invented by a model that was
      never asked for it.

    Fields the file says to return as null when inapplicable stay null: the
    merge treats `utility_ok=None` as "no opinion" and falls back to the rule
    pass, which is the correct reading of "not applicable for this query type".
    """
    verdict = {key: raw.get(key) for key in JUDGE_KEYS if key in raw}
    if "notes" in verdict:
        verdict["reason"] = str(verdict.pop("notes") or "")[:200]
    verdict["over_refusal"] = bool(rule.get("over_refusal"))
    return verdict
