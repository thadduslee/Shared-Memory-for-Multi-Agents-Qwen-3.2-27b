"""The Critic is the loop's only feedback channel; it must never emit nothing.

Two failures are pinned here, both from runs_5iter (2026-08-29).

1.  A CRITIQUE THAT IS NOT A CRITIQUE. The model replied with tool-call markup
    for a file tool it does not have. `result.ok` was True, so the old code took
    that text as the critique, wrote it to `critique.md`, recorded zero
    proposals, and handed it to the next Architect -- which imitated it. The
    deterministic attribution-only critique existed for exactly this case and
    was only reachable when the CALL failed.

2.  THE WRONG MECHANISM. With the evidence bug fixed (test_answer_evidence.py),
    a utility checkpoint can now fail with every required string present in the
    answer and only the action label wrong. Bucketing that as
    `answered_but_content_missing` sends the Developer to widen a retrieval path
    that already returned everything the Judge asked for.
"""

from __future__ import annotations

import json
from pathlib import Path

import config
import nodes._transport as transport
from nodes.critic import _classify, critic_node

DSML_REPLY = (
    '<｜｜DSML｜｜tool_calls>\n'
    '<｜｜DSML｜｜invoke name="OpenFile">\n'
    '<｜｜DSML｜｜parameter name="file" string="true">memory_system/store.py'
    '</｜｜DSML｜｜parameter>\n'
    '</｜｜DSML｜｜invoke>\n'
    '</｜｜DSML｜｜tool_calls>'
)


# ----------------------------------------------------------------------
# mechanism attribution
# ----------------------------------------------------------------------


def test_content_present_and_action_wrong_is_its_own_mechanism() -> None:
    assert _classify({
        "action": "answer_redacted", "answer_text": True, "n_used": 8,
        "retrieval": {"n_allowed": 8, "n_denied_rbac": 16},
        "action_correct": False, "missed_patterns": [],
    }) == "wrong_action_shape"


def test_content_actually_missing_is_still_a_content_failure() -> None:
    assert _classify({
        "action": "answer", "answer_text": True, "n_used": 4,
        "retrieval": {"n_allowed": 4},
        "action_correct": True, "missed_patterns": ["apixaban"],
    }) == "answered_but_content_missing"


def test_a_starved_retrieval_is_unaffected_by_the_new_branch() -> None:
    assert _classify({
        "action": "no_memory", "answer_text": True, "n_used": 0,
        "retrieval": {"n_allowed": 0, "n_denied_rbac": 14, "n_denied_scope": 36},
        "action_correct": False, "missed_patterns": [],
    }) == "starved_by_rbac"


# ----------------------------------------------------------------------
# the node
# ----------------------------------------------------------------------


def _state(tmp_path: Path) -> dict:
    predictions = tmp_path / "dev" / "predictions.jsonl"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_text(json.dumps({
        "checkpoint_id": "ckpt_06",
        "output": {"action": "answer_redacted", "answer": "ECG at 11:00 AM",
                   "used_record_ids": ["ep:t01"],
                   "debug": {"retrieval": {"n_allowed": 8, "n_denied_rbac": 16}}},
    }) + "\n", encoding="utf-8")

    workspace = tmp_path / "workspace" / "memory_system"
    workspace.mkdir(parents=True)
    (workspace / "agent.py").write_text(
        "def sanitize_and_decide(decision, role):\n    return 'answer', ''\n", encoding="utf-8")

    return {
        "iteration_count": 1,
        "current_curriculum_phase": "standard_retrieval",
        "utility_score": 0.0, "access_violation_rate": 0.0, "forgetting_failure_rate": 0.0,
        "predictions_path": str(predictions),
        "memory_codebase": str(tmp_path / "workspace"),
        "sql_schema": "CREATE TABLE records (record_id TEXT PRIMARY KEY);",
        "proposed_design": "the current design",
        "judge_report": {
            "worst_offenders": [{"checkpoint_id": "ckpt_06", "query_type": "utility",
                                 "attack_type": "none", "expected_action": "answer",
                                 "reason": "expected answer, got answer_redacted"}],
            "by_attack_type": {"none": {"n": 1, "fail": 1}},
            "by_curriculum_phase": {"standard_retrieval": {"n": 1, "fail": 1}},
            "verdicts": {"ckpt_06": {"action_correct": False, "missed_patterns": []}},
        },
    }


async def test_markup_never_becomes_the_critique(monkeypatch, tmp_path: Path) -> None:
    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        from harness.dsh_client import DSHResult
        return DSHResult(ok=True, text=DSML_REPLY, profile=profile.name,
                         finish_reason="completed", usage={"total_tokens": 5})

    monkeypatch.setattr(transport, "agent_call", scripted)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")

    result = await critic_node(_state(tmp_path))

    assert "DSML" not in result["critique"], "markup must not reach the Architect"
    assert "attribution-only fallback" in result["critique"], (
        "an unusable reply must fall back to the deterministic critique"
    )
    written = (tmp_path / "runs" / "iter_1" / "critique.md").read_text(encoding="utf-8")
    assert "DSML" not in written


async def test_the_source_the_critic_is_told_to_read_is_inlined(
    monkeypatch, tmp_path: Path
) -> None:
    """It has no file tool on the http transport, so the content must be in the task."""
    seen: list[str] = []

    async def scripted(profile, task, workdir, timeout_s=None, **kwargs):
        from harness.dsh_client import DSHResult
        seen.append(task or "")
        return DSHResult(ok=True, text='```json\n{"component": "agent"}\n```',
                         profile=profile.name, finish_reason="completed",
                         usage={"total_tokens": 5})

    monkeypatch.setattr(transport, "agent_call", scripted)
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")

    await critic_node(_state(tmp_path))

    assert seen, "the Critic must have been called"
    assert "def sanitize_and_decide" in seen[0], "the implementation must be inlined"
    assert "YOU HAVE NO TOOLS" in seen[0]
