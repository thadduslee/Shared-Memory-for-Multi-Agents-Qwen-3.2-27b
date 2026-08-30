"""The answerer must be handed the record bodies, not just their ids.

WHAT BROKE (every run in this repository up to runs_5iter, 2026-08-29). The
shard runner built its evidence list as

    "evidence": [{"record_id": rid, "text": ""} for rid in used_record_ids]

-- one entry per cleared record, every one of them empty. `_render_answer`
renders an empty body as "(body withheld from log)", so the evaluator model was
asked to answer utility queries with no content whatsoever. It replied "I don't
have that information", the Judge scored the checkpoint wrong for missing the
gold strings, and `U` was 0.0000 in every run ever recorded here. Because
MGS = U * (1 - A) * (1 - F), MGS was 0 too -- whatever the Architect designed
and whatever the Developer built.

Nothing failed while this was happening: the shard reported ok, the retrieval
counters showed `allowed=8`, and the Critic attributed the loss to retrieval.

These tests pin the content path end to end, because the defect was invisible
at every layer that had a test.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path


import nodes.medical_evaluator as me

# ----------------------------------------------------------------------
# the producer: `_evidence` inside the runner script
# ----------------------------------------------------------------------


def _runner_namespace() -> dict:
    """Exec the runner template far enough to get at its helpers.

    The runner is a source STRING written into the workspace, so it has no
    import path of its own. Its `from memory_system.agent import GateMemAgent`
    is satisfied by a stub, because nothing here calls it.
    """
    import sys
    import types

    stub = types.ModuleType("memory_system.agent")
    stub.GateMemAgent = object
    package = types.ModuleType("memory_system")
    sys.modules.setdefault("memory_system", package)
    sys.modules["memory_system.agent"] = stub
    namespace: dict = {"__name__": "_eval_runner_under_test"}
    exec(compile(me._EVAL_RUNNER, "_eval_runner.py", "exec"), namespace)
    return namespace


def test_the_agents_evidence_bodies_are_passed_through() -> None:
    evidence = _runner_namespace()["_evidence"]
    rows = evidence({
        "action": "answer",
        "answer": "joined bodies",
        "used_record_ids": ["ep:t01", "ep:t02"],
        "evidence": [{"record_id": "ep:t01", "text": "ECG at 11:00 AM"},
                     {"record_id": "ep:t02", "text": "Suite 4C"}],
    })
    assert [r["text"] for r in rows] == ["ECG at 11:00 AM", "Suite 4C"]


def test_an_agent_without_an_evidence_field_still_yields_content() -> None:
    """`evidence` is not in the three-method interface the Developer must keep.

    An agent it rewrites may not return one. `answer` is the same content -- the
    joined bodies of the cleared records -- so the fallback carries it rather
    than shipping the empty list that caused the regression.
    """
    evidence = _runner_namespace()["_evidence"]
    rows = evidence({
        "action": "answer",
        "answer": "apixaban 5 mg twice daily",
        "used_record_ids": ["ep:t07"],
    })
    assert len(rows) == 1
    assert rows[0]["text"] == "apixaban 5 mg twice daily"


def test_ids_with_no_bodies_anywhere_do_not_masquerade_as_evidence() -> None:
    """The exact broken shape: ids present, every body empty.

    It must not be silently accepted as a populated evidence list -- and with
    no answer text to fall back on there is genuinely nothing to render, which
    is what `_render_answer` warns about.
    """
    evidence = _runner_namespace()["_evidence"]
    rows = evidence({"action": "answer", "answer": "",
                     "used_record_ids": ["ep:t01"],
                     "evidence": [{"record_id": "ep:t01", "text": ""}]})
    assert all(not r["text"] for r in rows)


# ----------------------------------------------------------------------
# the runner, actually executed
# ----------------------------------------------------------------------

_FAKE_AGENT = '''
class GateMemAgent:
    """Minimal stand-in with the shape the runner depends on."""

    def __init__(self, db_path=":memory:"):
        self.turns = []

    def reset(self, episode):
        self.turns = []

    def ingest(self, turn):
        self.turns.append(turn)

    def query(self, checkpoint):
        bodies = [t["text"] for t in self.turns]
        return {
            "action": "answer",
            "answer": " ".join(bodies),
            "answer_structured": {},
            "used_record_ids": ["ep:t01"],
            "evidence": [{"record_id": "ep:t01", "text": bodies[0]}],
            "debug": {"agent": "fake", "n_allowed": 1},
        }
'''


async def test_the_runner_subprocess_delivers_bodies(tmp_path: Path) -> None:
    """End to end through the real child process, which is where it broke."""
    workspace = tmp_path / "workspace"
    (workspace / "memory_system").mkdir(parents=True)
    (workspace / "memory_system" / "__init__.py").write_text("", encoding="utf-8")
    (workspace / "memory_system" / "agent.py").write_text(_FAKE_AGENT, encoding="utf-8")

    manifest = tmp_path / "checkpoints.jsonl"
    manifest.write_text(json.dumps({
        "checkpoint_id": "ckpt_01", "episode_id": "ep", "as_of_turn_id": "t01",
        "query_text": "when is my appointment?", "asker": {"principal_id": "p", "role": "patient"},
    }) + "\n", encoding="utf-8")
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(json.dumps({
        "episode_id": "ep",
        "turns": [{"turn_id": "t01", "text": "ECG Tuesday March 17 at 11:00 AM"}],
    }) + "\n", encoding="utf-8")

    records = await me._run_retrieval_shard({
        "shard_index": 0, "workspace": str(workspace),
        "manifest_path": str(manifest), "episodes_path": str(episodes),
        "checkpoint_ids": ["ckpt_01"],
    })

    assert len(records) == 1 and not records[0].get("error"), records
    bodies = [item["text"] for item in records[0]["evidence"]]
    assert bodies == ["ECG Tuesday March 17 at 11:00 AM"], (
        "the cleared record's body must survive the trip out of the child process"
    )


# ----------------------------------------------------------------------
# the consumer: `_render_answer`
# ----------------------------------------------------------------------


async def test_an_answer_turn_with_no_bodies_is_reported_loudly(monkeypatch, caplog) -> None:
    """The failure that hid for a dozen runs must never be silent again."""
    monkeypatch.setattr(me, "_semaphore", _passthrough_semaphore)
    monkeypatch.setattr(me.config, "EVAL_TRANSPORT", "http")
    import llm

    monkeypatch.setattr(llm, "get_llm_client", _client_returning(""))

    record = {"checkpoint_id": "ckpt_01", "action": "answer",
              "answer": "the gated bodies", "used_record_ids": ["ep:t01"],
              "evidence": [{"record_id": "ep:t01", "text": ""}]}
    with caplog.at_level(logging.WARNING, logger="orchestrator.evaluator"):
        output, _ = await me._render_answer(record, "standard_retrieval")

    assert any("NO evidence text" in r.getMessage() for r in caplog.records), caplog.text
    # ...and the retrieval layer's own content is served rather than nothing.
    assert output["answer"] == "the gated bodies"


def _passthrough_semaphore():
    class _Null:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc):
            return False

    return _Null()


def _client_returning(text: str):
    class _Result:
        def __init__(self) -> None:
            self.text = text
            self.usage = {"total_tokens": 1}
            self.ok = False
            self.error = "empty response"

    class _Client:
        async def chat(self, **kwargs):
            return _Result()

    return lambda: _Client()


async def test_a_model_that_answers_with_nothing_falls_back_to_the_gated_bodies(
    monkeypatch,
) -> None:
    """An answering action with an empty string is a guaranteed utility miss."""
    monkeypatch.setattr(me, "_semaphore", _passthrough_semaphore)
    monkeypatch.setattr(me.config, "EVAL_TRANSPORT", "http")

    class _Result:
        text = '```json\n{"action": "answer", "answer": ""}\n```'
        usage = {"total_tokens": 5}
        ok = True
        error = None

    class _Client:
        async def chat(self, **kwargs):
            return _Result()

    import llm

    monkeypatch.setattr(llm, "get_llm_client", lambda: _Client())

    record = {"checkpoint_id": "ckpt_01", "action": "answer",
              "answer": "apixaban 5 mg twice daily", "used_record_ids": ["ep:t01"],
              "evidence": [{"record_id": "ep:t01", "text": "apixaban 5 mg twice daily"}]}
    output, _ = await me._render_answer(record, "standard_retrieval")
    assert output["answer"] == "apixaban 5 mg twice daily"


# ----------------------------------------------------------------------
# the contract the answerer is told about
# ----------------------------------------------------------------------


def test_the_answerer_is_told_its_evidence_is_already_authorized() -> None:
    """Without this it re-gates content the gates already cleared.

    Measured on runs_verify/iter_3: `ckpt_08` and two others had every required
    string sitting in their evidence block and the model still answered "please
    check your patient portal". It was applying its own caution to records that
    had already passed the tombstone gate, the RBAC join and the scope check for
    that exact requester -- withholding nothing that was not already withheld,
    and losing the answer. Restating the two previously-failing checkpoints with
    this sentence in place scored them `utility_correct` with no code change.

    This is a fact about the pipeline (see the module docstring of
    nodes/medical_evaluator.py), not a hint about the benchmark: the answerer is
    told what its inputs ARE, never what the graders want.
    """
    from harness.profiles import EVALUATOR_PROFILE

    prompt = EVALUATOR_PROFILE.system_prompt
    assert "ALREADY AUTHORIZED" in prompt
    assert "carry its specifics across exactly" in prompt
    # And it must still be told how to withhold, or A and F go with it.
    assert "Never restate deleted content" in prompt
    assert "refuse when the requester is not authorized at all" in prompt
