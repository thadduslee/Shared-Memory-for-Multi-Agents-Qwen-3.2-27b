"""A relative `RUNS_DIR` must not silently kill every evaluation shard.

WHAT HAPPENED. `run-86c51cd90e1e` was launched as

    RUNS_DIR=runs_real_100iter_v2 .venv/bin/python main.py --real ...

with `RUNS_DIR` *relative*. `config.RUNS_DIR` stored it verbatim, so
`stage_dir()` produced `runs_real_100iter_v2/iter_N/dev/`, and the dispatcher
wrote that relative string into every shard spec as `manifest_path`. The
orchestrator's own cwd is the project root, so nothing complained -- but
`_run_retrieval_shard` launches `_eval_runner.py` with `cwd=workspace`, several
directories deep inside the run, and there the path resolves to nothing:

    FileNotFoundError: [Errno 2] No such file or directory:
        'runs_real_100iter_v2/iter_6/dev/checkpoints.stripped.jsonl'

All five shards died identically, the circuit breaker tripped on the shared
signature, `route_after_collect` fail-fasted back to the Architect, and the Judge
was never reached -- so the run produced no `judge_report.json`, no scoreboard
row and no critique, while still paying full price for an Architect and a
Developer episode every iteration. Seven iterations, ~2M tokens, zero signal.

Worse, it was invisible: `log.error(..., str(exc)[:200])` truncated from the
FRONT, keeping the useless banner and cutting the exception off mid-word at
`sys.exit(ma`. The log said nothing 231 lines running.

These tests pin all three halves of the fix: the path is absolute, the shard
spec is absolute, and a failure message keeps the part that explains it.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import config
from nodes import medical_evaluator as ev

ROOT = Path(__file__).resolve().parent.parent


# ======================================================================
# 1. config.RUNS_DIR
# ======================================================================


def _runs_dir_under(env_value: str, cwd: Path) -> Path:
    """Import `config` in a fresh interpreter and report where RUNS_DIR landed.

    A subprocess rather than a reload: `RUNS_DIR` is `Final` and read at import
    time by several modules, so mutating it in-process would test the wrong
    thing.
    """
    out = subprocess.run(
        [sys.executable, "-c", "import config; print(config.RUNS_DIR)"],
        cwd=str(cwd), env={**os.environ, "RUNS_DIR": env_value, "PYTHONPATH": str(ROOT)},
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def test_relative_runs_dir_becomes_absolute():
    resolved = _runs_dir_under("runs_relative_probe", ROOT)
    assert resolved.is_absolute(), f"RUNS_DIR stayed relative: {resolved}"
    assert resolved == ROOT / "runs_relative_probe"


def test_relative_runs_dir_is_cwd_relative_not_project_relative(tmp_path):
    """The shell's meaning is kept: `RUNS_DIR=x` means `$PWD/x`."""
    resolved = _runs_dir_under("runs_relative_probe", tmp_path)
    assert resolved == tmp_path / "runs_relative_probe"


def test_absolute_runs_dir_is_left_alone(tmp_path):
    assert _runs_dir_under(str(tmp_path / "elsewhere"), ROOT) == tmp_path / "elsewhere"


def test_derived_paths_are_absolute():
    """Everything a shard is handed comes from these two."""
    assert config.iteration_dir(3).is_absolute()
    assert config.stage_dir(3, "dev").is_absolute()
    assert config.critique_summary_path().is_absolute()


# ======================================================================
# 2. The shard spec
# ======================================================================


def _dispatch(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "EXPECTED_DEV_CHECKPOINTS", 4)
    state = {
        "iteration_count": 1,
        "eval_stage": "dev",
        "current_curriculum_phase": config.CURRICULUM_PHASES[0],
        "memory_codebase": str(tmp_path / "runs" / "iter_1" / "workspace"),
    }
    return asyncio.run(ev.eval_dispatch_node(state))["shard_plan"]


def test_dispatch_writes_absolute_paths(tmp_path, monkeypatch):
    """The child process runs with `cwd=workspace`, so relative is unreadable."""
    shards = _dispatch(tmp_path, monkeypatch)
    assert shards, "dispatch produced no shards"
    for shard in shards:
        for key in ("manifest_path", "episodes_path", "workspace"):
            assert Path(shard[key]).is_absolute(), f"shard {shard['shard_index']} {key} is relative"


def test_shard_paths_readable_from_the_workspace(tmp_path, monkeypatch):
    """The exact failure: open() the manifest with the child's cwd, not ours."""
    shards = _dispatch(tmp_path, monkeypatch)
    shard = shards[0]
    workspace = Path(shard["workspace"])
    workspace.mkdir(parents=True, exist_ok=True)

    code = (
        "import json,sys;"
        "rows=[json.loads(l) for l in open(sys.argv[1],encoding='utf-8') if l.strip()];"
        "print(len(rows))"
    )
    for key in ("manifest_path", "episodes_path"):
        done = subprocess.run(
            [sys.executable, "-c", code, shard[key]],
            cwd=str(workspace), capture_output=True, text=True, check=False,
        )
        assert done.returncode == 0, f"{key} unreadable from the workspace:\n{done.stderr}"
        assert int(done.stdout.strip()) > 0, f"{key} was empty"


def test_manifest_is_where_the_spec_says_it_is(tmp_path, monkeypatch):
    shards = _dispatch(tmp_path, monkeypatch)
    manifest = Path(shards[0]["manifest_path"])
    assert manifest.is_file()
    ids = {json.loads(line)["checkpoint_id"] for line in manifest.read_text().splitlines() if line.strip()}
    assert set(shards[0]["checkpoint_ids"]) <= ids


# ======================================================================
# 3. The message that should have told us
# ======================================================================


CHILD_FAILURE = (
    "retrieval shard 0 produced no result\n"
    "Traceback (most recent call last):\n"
    '  File "/x/_eval_runner.py", line 89, in <module>\n'
    "    sys.exit(main())\n"
    '  File "/x/_eval_runner.py", line 50, in main\n'
    "    (json.loads(l) for l in open(spec[\"manifest_path\"], encoding=\"utf-8\") if l.strip())}\n"
    "FileNotFoundError: [Errno 2] No such file or directory: "
    "'runs_real_100iter_v2/iter_6/dev/checkpoints.stripped.jsonl'\n"
)


def test_diagnosis_keeps_the_exception_line():
    line = ev._diagnosis(RuntimeError(CHILD_FAILURE))
    assert "FileNotFoundError" in line, f"the cause was truncated away: {line}"
    assert "produced no result" in line, "lost the context"
    assert "\n" not in line, "a log line must stay one line"


def test_diagnosis_stays_within_its_limit():
    line = ev._diagnosis(RuntimeError("head " + "x" * 5000 + "\ntail " + "y" * 5000), limit=120)
    assert len(line) <= 120


def test_diagnosis_handles_a_single_line_and_an_empty_one():
    assert ev._diagnosis(RuntimeError("boom")) == "boom"
    assert "ValueError" in ev._diagnosis(ValueError(""))


def test_clip_keeps_the_tail():
    clipped = ev._clip_keeping_tail(CHILD_FAILURE * 20, 1200)
    assert len(clipped) <= 1200
    assert "checkpoints.stripped.jsonl" in clipped, "the filename was cut off mid-path again"
    assert clipped.startswith("retrieval shard 0"), "lost the head"


def test_clip_is_a_no_op_when_it_fits():
    assert ev._clip_keeping_tail("short", 1200) == "short"
