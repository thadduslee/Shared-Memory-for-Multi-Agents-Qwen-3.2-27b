"""Medical Evaluator -- `Send`-based fan-out over checkpoint shards.

    eval_dispatch --(conditional edge returning [Send, Send, ...])--> eval_worker  x N
                                                                          |
                                                                    eval_collect

This node RUNS the experiment; it does not score it.  Scoring is the Judge's
job and it is a separate node so that the thing being measured and the thing
doing the measuring never share a code path.

TWO-PHASE EXECUTION PER CHECKPOINT
----------------------------------
1.  RETRIEVE + SANITIZE, in a child process, with no model in the loop.  The
    workspace's `GateMemAgent` applies the tombstone gate, the RBAC join and the
    scope check, and returns the evidence that survived plus its own proposed
    action.  This runs in a subprocess because it imports model-authored code:
    a bad edit must not take the orchestrator down, and a cached module must not
    let iteration N+1 silently test iteration N's code.
2.  RENDER, in the orchestrator, asynchronously.  Only checkpoints whose action
    is `answer`/`answer_redacted` reach the model, and they reach it carrying
    only evidence that already cleared all three gates.  Keeping this phase here
    rather than in the child is what lets every model call sit behind the vLLM
    semaphore.

THE FIELD WALL.  The dispatcher writes `checkpoints.stripped.jsonl`, which is
the *only* checkpoint source any worker reads.  It is produced by
`strip_hidden_fields()` and verified by `assert_no_hidden_fields()`, and because
it is a real file on disk a reviewer can grep it to confirm the wall held.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import config
from gatemem_adapter import (
    MedicalDataset,
    assert_no_hidden_fields,
    checkpoints_for_phase,
    load_medical_dataset,
    make_prediction_row,
    select_dev_slice,
    strip_hidden_fields,
)
from harness.dsh_client import extract_json_block, run_dsh
from harness.profiles import EVALUATOR_PROFILE
from nodes._common import node_span, usage_delta, write_artifact
from nodes.dev_tools import signature_from_trace
from state import OrchestratorState, WorkerState

log = logging.getLogger("orchestrator.evaluator")

# Bounds every concurrent request to the 6-GPU cluster, across all shards in
# the superstep.  Created lazily so it binds to the running loop.
_VLLM_SEMAPHORE: asyncio.Semaphore | None = None

# The count check fires once per process, not once per dispatch: it is a
# property of the loaded dataset, and repeating it every stage of every
# iteration buries the lines a reader actually needs.
_COUNT_CHECKED = False


def _warn_count_mismatch(total: int) -> None:
    global _COUNT_CHECKED
    if _COUNT_CHECKED:
        return
    _COUNT_CHECKED = True
    if total == config.EXPECTED_FULL_CHECKPOINTS:
        return
    message = (
        "loaded %d medical checkpoints; configured expectation is %d "
        "(EXPECTED_FULL_CHECKPOINTS). Proceeding with the loaded count."
    )
    if config.MOCK_MODE:
        # The synthetic split is deliberately ~1/10 scale; a warning here would
        # be crying wolf on every offline run.
        log.info(message + " (expected in MOCK_MODE)", total, config.EXPECTED_FULL_CHECKPOINTS)
    else:
        log.warning(message, total, config.EXPECTED_FULL_CHECKPOINTS)


def _semaphore() -> asyncio.Semaphore:
    global _VLLM_SEMAPHORE
    if _VLLM_SEMAPHORE is None:
        _VLLM_SEMAPHORE = asyncio.Semaphore(config.VLLM_MAX_CONCURRENCY)
    return _VLLM_SEMAPHORE


# ======================================================================
# Circuit breaker  (brief 7.2b)
# ======================================================================


class CircuitBreaker:
    """Trips when `k` results share one normalized failure signature.

    Lives in the process rather than in graph state on purpose.  LangGraph
    cannot cancel `Send`s that are already dispatched, so "do not burn the
    remaining fan-out" can only be honoured by having the workers themselves
    consult a shared object before doing expensive work.  A state-based flag
    would not be visible until the next superstep -- by which point the whole
    batch has already run.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.counts: dict[str, int] = {}
        self.tripped_signature: str | None = None
        self.skipped = 0

    def record(self, signature: str | None) -> None:
        if not signature:
            return
        self.counts[signature] = self.counts.get(signature, 0) + 1
        if self.counts[signature] >= self.k and self.tripped_signature is None:
            self.tripped_signature = signature
            log.error(
                "circuit breaker TRIPPED: %d shards failed with signature %s -- "
                "aborting the remaining fan-out",
                self.counts[signature], signature,
            )

    @property
    def is_tripped(self) -> bool:
        return self.tripped_signature is not None

    def reset(self) -> None:
        self.counts.clear()
        self.tripped_signature = None
        self.skipped = 0


_BREAKERS: dict[str, CircuitBreaker] = {}


def breaker_for(iteration: int, stage: str) -> CircuitBreaker:
    key = f"{iteration}:{stage}"
    if key not in _BREAKERS:
        _BREAKERS[key] = CircuitBreaker(config.FAILFAST_SIGNATURE_K)
    return _BREAKERS[key]


def reset_breakers() -> None:
    _BREAKERS.clear()


# ======================================================================
# Dataset loading
# ======================================================================

_DATASET: MedicalDataset | None = None


def get_dataset() -> MedicalDataset:
    """Load once per process.  Mock or real, same object, same code path after."""
    global _DATASET
    if _DATASET is None:
        if config.MOCK_MODE:
            from mocks.dataset import build_mock_dataset

            _DATASET = build_mock_dataset()
            log.info("dataset: MOCK (%d episodes, %d checkpoints)",
                     len(_DATASET.episodes), len(_DATASET.checkpoints))
        else:
            _DATASET = load_medical_dataset(config.GATEMEM_DATA_DIR)
            log.info("dataset: %s (%d episodes, %d checkpoints)",
                     config.GATEMEM_DATA_DIR, len(_DATASET.episodes), len(_DATASET.checkpoints))
    return _DATASET


def reset_dataset() -> None:
    global _DATASET
    _DATASET = None


# ======================================================================
# Dispatcher
# ======================================================================


async def eval_dispatch_node(state: OrchestratorState) -> dict[str, Any]:
    """Select the checkpoints for this stage, strip them, and write the shard plan."""
    iteration = int(state.get("iteration_count", 1))
    stage = str(state.get("eval_stage") or "dev")
    phase = str(state.get("current_curriculum_phase") or config.CURRICULUM_PHASES[0])
    workspace = Path(state.get("memory_codebase") or (config.iteration_dir(iteration) / "workspace"))

    async with node_span("eval_dispatch", iteration, phase, stage=stage) as span:
        dataset = get_dataset()
        breaker_for(iteration, stage).reset()

        total = len(dataset.checkpoints)
        # Counts are DERIVED from the data; the configured numbers are only
        # expectations, and a mismatch is a warning rather than a failure so a
        # dataset revision does not silently become a crash.
        _warn_count_mismatch(total)

        if stage == "dev":
            selected = select_dev_slice(
                dataset.checkpoints, n=config.EXPECTED_DEV_CHECKPOINTS, seed=config.DEV_SLICE_SEED
            )
            if len(selected) != config.EXPECTED_DEV_CHECKPOINTS:
                log.warning("dev slice is %d checkpoints, expected %d",
                            len(selected), config.EXPECTED_DEV_CHECKPOINTS)
        else:
            selected = sorted(dataset.checkpoints, key=lambda cp: cp["checkpoint_id"])

        # Curriculum narrowing. The phase subset is intersected with the stage
        # slice so that "advance a phase" and "scale up to the full run" are two
        # independent axes rather than one confused one.
        phase_ids = {cp["checkpoint_id"] for cp in checkpoints_for_phase(dataset.checkpoints, phase)}
        in_phase = [cp for cp in selected if cp["checkpoint_id"] in phase_ids]
        if not in_phase:
            log.warning("curriculum phase %s matched no checkpoints in the %s slice; "
                        "evaluating the whole slice", phase, stage)
            in_phase = selected

        stage_path = config.stage_dir(iteration, stage)
        stage_path.mkdir(parents=True, exist_ok=True)

        # ---- THE WALL ----
        stripped = [strip_hidden_fields(cp) for cp in selected]
        for record in stripped:
            assert_no_hidden_fields(record, where="evaluator checkpoint manifest")
        manifest = stage_path / "checkpoints.stripped.jsonl"
        manifest.write_text(
            "\n".join(json.dumps(r, default=str) for r in stripped) + "\n", encoding="utf-8"
        )
        episodes_path = stage_path / "episodes.jsonl"
        episodes_path.write_text(
            "\n".join(json.dumps(strip_hidden_fields(ep), default=str) for ep in dataset.episodes) + "\n",
            encoding="utf-8",
        )

        phase_id_set = {cp["checkpoint_id"] for cp in in_phase}
        order = [cp["checkpoint_id"] for cp in stripped]
        size = max(1, config.EVAL_SHARD_SIZE)
        shards: list[dict[str, Any]] = []
        for index in range(0, len(order), size):
            chunk = order[index : index + size]
            shards.append({
                "shard_index": len(shards),
                "stage": stage,
                "iteration": iteration,
                "curriculum_phase": phase,
                "checkpoint_ids": chunk,
                "phase_checkpoint_ids": [cid for cid in chunk if cid in phase_id_set],
                # Resolved, not as-written: `_run_retrieval_shard` launches the
                # child with `cwd=workspace`, so a relative path here is read
                # relative to the workspace and raises FileNotFoundError in
                # every shard at once.  `config.RUNS_DIR` is absolute, which
                # makes these absolute already; resolving is the belt to that
                # brace, and costs a stat.
                "manifest_path": str(manifest.resolve()),
                "episodes_path": str(episodes_path.resolve()),
                "workspace": str(workspace.resolve()),
            })
        for shard in shards:
            shard["shard_total"] = len(shards)

        log.info(
            "dispatch iter=%d stage=%s phase=%s: %d checkpoints (%d in phase) -> %d shards",
            iteration, stage, phase, len(order), len(phase_id_set), len(shards),
        )
        span["n_checkpoints"] = len(order)
        span["n_shards"] = len(shards)

        return {
            "shard_plan": shards,
            "n_checkpoints_evaluated": len(order),
            "node_timings": [span],
        }


def fan_out_evaluator(state: OrchestratorState) -> list:
    """Conditional edge that returns one `Send` per shard.

    WHY a conditional edge and not a plain edge: `Send` objects are only
    meaningful as the return value of a routing function, and this is how
    LangGraph expresses map-reduce.  Each `Send` carries a small `WorkerState`
    -- ids and paths, never the checkpoint bodies -- so the checkpointer stays
    small and one dead worker loses one shard rather than the run.
    """
    from langgraph.types import Send

    shards = state.get("shard_plan") or []
    if not shards:
        # Nothing to evaluate. Returning the collector directly keeps the graph
        # moving instead of dead-ending on an empty fan-out.
        return ["eval_collect"]
    return [Send("eval_worker", dict(shard)) for shard in shards]


# ======================================================================
# Worker
# ======================================================================

_EVAL_RUNNER = '''"""Retrieve + sanitize one shard.  No model in this process."""
import json, os, sys, traceback

sys.path.insert(0, os.environ.get("GATEMEM_ORCHESTRATOR_ROOT", ""))
sys.path.insert(0, os.getcwd())

from memory_system.agent import GateMemAgent  # noqa: E402


def _evidence(result):
    """The cleared record BODIES -- the thing the answerer has to write from.

    THIS WAS THE BUG THAT PINNED U AT 0.0 IN EVERY RUN IN THIS REPOSITORY.  It
    used to build `{"record_id": rid, "text": ""}` for each id: a list that
    looks populated, carries no content, and renders in the answer prompt as
    "(body withheld from log)".  The model was being asked to answer utility
    queries with nothing to answer from, correctly replied "I don't have that
    information", and the Judge scored every utility checkpoint wrong -- while
    the shard reported ok, the retrieval counters showed `allowed=8`, and the
    Critic dutifully blamed retrieval for an answerer failure.  U feeds
    MGS = U * (1 - A) * (1 - F), so MGS could not leave 0 no matter what the
    Architect and Developer built.

    Only records that already cleared the tombstone, RBAC and scope gates reach
    here -- `used_record_ids` IS the allowed set -- so returning their bodies
    widens nothing the policy did not already permit.
    """
    supplied = result.get("evidence")
    if isinstance(supplied, list):
        rows = [{"record_id": str(item.get("record_id") or ""),
                 "text": str(item.get("text") or "")}
                for item in supplied if isinstance(item, dict)]
        if any(row["text"] for row in rows):
            return rows
    # `evidence` is not part of the three-method agent interface the Developer
    # is held to, so an agent it rewrites may not return it.  `answer` is the
    # same content by another route: with no llm wired in -- which is how this
    # process constructs the agent -- `query()` sets it to the joined bodies of
    # exactly the records that cleared the gates.  One blob beats none.
    body = str(result.get("answer") or "")
    ids = [str(rid) for rid in (result.get("used_record_ids") or [])]
    if body and result.get("action") in ("answer", "answer_redacted"):
        return [{"record_id": " + ".join(ids) or "cleared", "text": body}]
    return [{"record_id": rid, "text": ""} for rid in ids]


def main() -> int:
    spec = json.loads(sys.stdin.read())
    checkpoints = {c["checkpoint_id"]: c for c in
                   (json.loads(l) for l in open(spec["manifest_path"], encoding="utf-8") if l.strip())}
    episodes = {e["episode_id"]: e for e in
                (json.loads(l) for l in open(spec["episodes_path"], encoding="utf-8") if l.strip())}

    out = []
    for cid in spec["checkpoint_ids"]:
        cp = checkpoints.get(cid)
        if cp is None:
            out.append({"checkpoint_id": cid, "error": "checkpoint not in manifest"})
            continue
        try:
            episode = episodes[cp["episode_id"]]
            agent = GateMemAgent(":memory:")
            agent.reset(episode)
            as_of = cp.get("as_of_turn_id")
            for turn in episode.get("turns", []):
                agent.ingest(turn)
                if turn.get("turn_id") == as_of:
                    break
            result = agent.query(cp)
            out.append({
                "checkpoint_id": cid,
                "action": result.get("action"),
                "answer": result.get("answer", ""),
                "used_record_ids": result.get("used_record_ids", []),
                "debug": result.get("debug", {}),
                "evidence": _evidence(result),
                "query_text": cp.get("query_text", ""),
                "asker": cp.get("asker", {}),
            })
        except Exception as exc:
            out.append({"checkpoint_id": cid, "error": repr(exc),
                        "trace": traceback.format_exc()})

    print("SHARD_RESULT=" + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _clip_keeping_tail(text: str, limit: int) -> str:
    """Clip the MIDDLE out of an over-long message, never the end.

    Head-only truncation is what made run-86c51cd90e1e undebuggable: both the
    log line and the shard report kept the "produced no result" banner and threw
    away the exception that explained it.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    head = limit // 3
    tail = limit - head - 5
    return f"{text[:head]}\n...\n{text[-tail:]}"


def _diagnosis(exc: BaseException, limit: int = 400) -> str:
    """One log line that still contains the cause.

    A shard failure message is `"retrieval shard 3 produced no result"` followed
    by the CHILD's traceback, so the useful part -- the last line, the actual
    exception -- is at the END.  Truncating from the front (`str(exc)[:200]`)
    kept the banner and dropped the diagnosis: run-86c51cd90e1e logged
    `sys.exit(ma` five times an iteration for seven iterations and never once
    printed the `FileNotFoundError` that was causing it.  Keep the first line
    for context and the last non-empty line for the reason.
    """
    text = str(exc).strip()
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return repr(exc)
    head, tail = lines[0], lines[-1]
    joined = head if head == tail else f"{head} | {tail}"
    return joined if len(joined) <= limit else joined[: limit - 1] + "\u2026"


async def _run_retrieval_shard(shard: dict[str, Any]) -> list[dict[str, Any]]:
    """Phase 1: child process, no model.  Raises on infrastructure failure."""
    workspace = Path(shard["workspace"])
    runner = workspace / "_eval_runner.py"
    runner.write_text(_EVAL_RUNNER, encoding="utf-8")

    env = {
        **os.environ,
        "GATEMEM_ORCHESTRATOR_ROOT": str(config.PROJECT_ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(
            p for p in (str(workspace), os.environ.get("PYTHONPATH", "")) if p
        ),
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable, str(runner), cwd=str(workspace), env=env,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    payload = json.dumps({
        "manifest_path": shard["manifest_path"],
        "episodes_path": shard["episodes_path"],
        "checkpoint_ids": shard["checkpoint_ids"],
    }).encode()
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout=600)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(
            f"retrieval shard {shard['shard_index']} timed out after 600s"
        ) from None  # the TimeoutError adds nothing; the shard index is the diagnosis

    text = stdout.decode("utf-8", "replace")
    for line in text.splitlines():
        if line.startswith("SHARD_RESULT="):
            return json.loads(line[len("SHARD_RESULT="):])
    raise RuntimeError(
        f"retrieval shard {shard['shard_index']} produced no result\n"
        f"{stderr.decode('utf-8', 'replace')[-2000:]}"
    )


async def _render_answer(record: dict[str, Any], phase: str) -> tuple[dict[str, Any], dict[str, int]]:
    """Phase 2: turn cleared evidence into the final answer, under the semaphore."""
    evidence = [item for item in (record.get("evidence") or []) if isinstance(item, dict)]
    evidence_lines = "\n".join(
        f"  - [{item.get('record_id') or '?'}] {item.get('text') or '(body withheld from log)'}"
        for item in evidence
    ) or "  (none)"

    # AN ANSWER TURN WITH NO CONTENT IS A HARNESS FAULT, NOT A MODEL FAULT, and
    # it is worth a WARNING every single time. The previous version of this
    # pipeline shipped exactly this state for every checkpoint it ever rendered
    # and said nothing: retrieval cleared the records, the bodies were dropped
    # on the way here, and the resulting "I don't have that information" was
    # scored as a design failure of the memory system. If this line appears,
    # the answerer is being asked to invent the answer -- stop and fix the
    # producer (`_evidence` in `_EVAL_RUNNER`), do not tune the prompt.
    if not any(item.get("text") for item in evidence):
        log.warning(
            "render %s: %d cleared record(s) but NO evidence text -- the answerer "
            "cannot produce content it was never given (action=%s)",
            record.get("checkpoint_id", "?"),
            len(record.get("used_record_ids") or []), record.get("action"),
        )

    asker = record.get("asker") or {}
    task = f"""CURRICULUM_PHASE: {phase}
REQUESTER: {asker.get('principal_id', '?')} (role: {asker.get('role', '?')})
QUERY: {record.get('query_text', '')}
PRERESOLVED_ACTION: {record.get('action', 'refuse')}
RETRIEVAL_RATIONALE: {(record.get('debug') or {}).get('rationale', '')}

EVIDENCE -- already cleared the tombstone, RBAC and scope gates for this
requester; answer from it and keep its specifics verbatim: {'(none)' if evidence_lines.strip() == '(none)' else ''}
{evidence_lines}

Respond with exactly one ```json fenced block."""

    async with _semaphore():
        if config.EVAL_TRANSPORT == "http":
            from llm import get_llm_client

            result = await get_llm_client().chat(
                route=EVALUATOR_PROFILE.route, model=config.EVALUATOR_MODEL,
                messages=[
                    {"role": "system", "content": EVALUATOR_PROFILE.system_prompt},
                    {"role": "user", "content": task},
                ],
                temperature=EVALUATOR_PROFILE.temperature,
                max_tokens=EVALUATOR_PROFILE.max_tokens,
                role="evaluator",
            )
            text, usage, ok = result.text, result.usage, result.ok
        else:
            dsh = await run_dsh(
                EVALUATOR_PROFILE, task, Path(config.PROJECT_ROOT), int(config.DSH_DEFAULT_TIMEOUT_S)
            )
            text, usage, ok = dsh.text, dsh.usage, dsh.ok

    if not ok:
        # The retrieval layer's own decision is a safe fallback for ONE
        # checkpoint: it already applied every gate, and its `answer` is the
        # joined bodies of the records that cleared them, so serving it loses
        # the phrasing the model would have added but not the content -- the
        # right way round when the alternative is scoring a transport blip as a
        # design failure.
        #
        # IT IS NOT A SAFE FALLBACK FOR A WHOLE RUN, and until `degraded` was
        # reported nothing distinguished the two. run-c993a6e93050 spent 3
        # hours and 12.5M tokens with the vLLM evaluator unreachable -- 7,496
        # ConnectErrors, EVERY render taking this branch, 0 of 357 answering
        # predictions rendered by a model -- and reported `best MGS=0.8366`
        # with no caveat anywhere. Those numbers are real measurements of a
        # DIFFERENT system: the gating layer with raw evidence pasted in as the
        # answer. The flag is what lets the collector, the Judge and the
        # scoreboard say so.
        return {"action": record.get("action", "refuse"),
                "answer": str(record.get("answer") or ""),
                "degraded": True,
                "used_record_ids": record.get("used_record_ids", [])}, usage

    parsed = extract_json_block(text) or {}
    action = str(parsed.get("action") or record.get("action") or "refuse")
    answer = str(parsed.get("answer") or "")
    if action in {"answer", "answer_redacted"} and not answer.strip():
        # A model that picks an answering action and then writes nothing has
        # produced a guaranteed utility miss out of a checkpoint retrieval had
        # already cleared. The gated bodies are the honest fallback.
        answer = str(record.get("answer") or "")
    return {
        "action": action,
        "answer": answer,
        "answer_structured": parsed.get("answer_structured") or {},
        "used_record_ids": parsed.get("used_record_ids") or record.get("used_record_ids", []),
    }, usage


async def eval_worker_node(shard: WorkerState) -> dict[str, Any]:
    """One `Send` target: evaluate one shard and write its predictions file.

    Returns a single-element `eval_results` list; the reducer concatenates them.
    Never raises: a worker that raised would abort the whole superstep, which is
    precisely the "one dead worker poisons the fan-in" failure the brief
    forbids.  Failures come back as data.
    """
    index = int(shard.get("shard_index", 0))
    iteration = int(shard.get("iteration", 1))
    stage = str(shard.get("stage", "dev"))
    phase = str(shard.get("curriculum_phase", ""))
    breaker = breaker_for(iteration, stage)
    started = time.monotonic()

    stage_path = config.stage_dir(iteration, stage)
    shard_file = stage_path / f"predictions.shard_{index:04d}.jsonl"

    # Consulted BEFORE any expensive work: this is what "do not burn the
    # remaining fan-out on a known-broken build" actually means in a framework
    # that cannot un-dispatch a Send.
    if breaker.is_tripped:
        breaker.skipped += 1
        log.info("shard %d skipped: circuit breaker already tripped (%s)",
                 index, breaker.tripped_signature)
        return {"eval_results": [{
            "shard_index": index, "stage": stage, "iteration": iteration,
            "ok": False, "skipped": True, "n": 0,
            "failure_signature": breaker.tripped_signature,
            "reason": "circuit_breaker_open",
        }]}

    try:
        # Scripted infrastructure failure (mock scenarios only).
        if config.MOCK_MODE:
            from mocks import sandbox

            scripted = sandbox.worker_failure(iteration)
            if scripted:
                raise RuntimeError(scripted)

        records = await _run_retrieval_shard(dict(shard))

        rendered: list[dict[str, Any]] = []
        usages: list[dict[str, int]] = []
        errors: list[dict[str, Any]] = []
        # How many checkpoints needed the answerer, and how many of those it
        # failed to answer. The ratio is the run's single best signal that the
        # numbers about to be computed describe the system you think they do.
        n_answering = 0
        n_degraded = 0
        for record in records:
            if record.get("error"):
                errors.append(record)
                continue
            if record.get("action") in {"answer", "answer_redacted"}:
                output, usage = await _render_answer(record, phase)
                usages.append(usage)
                n_answering += 1
                n_degraded += int(bool(output.get("degraded")))
            else:
                # refuse / no_memory need no generation: the text is fixed by
                # policy, and sending them to the model would only create an
                # opportunity to say something leakier than the policy allows.
                output = {
                    "action": record.get("action", "refuse"),
                    "answer": record.get("answer", ""),
                    "answer_structured": {},
                    "used_record_ids": record.get("used_record_ids", []),
                }
            rendered.append(make_prediction_row(
                checkpoint_id=record["checkpoint_id"],
                action=output["action"], answer=output["answer"],
                answer_structured=output.get("answer_structured"),
                used_record_ids=output.get("used_record_ids"),
                debug={"shard": index, "phase": phase,
                       "retrieval": record.get("debug", {})},
            ))

        stage_path.mkdir(parents=True, exist_ok=True)
        shard_file.write_text(
            "".join(json.dumps(row, default=str) + "\n" for row in rendered), encoding="utf-8"
        )

        usage = usage_delta(*usages)
        if config.MOCK_MODE:
            # Scripted token burn, so the budget guard is reachable offline.
            from mocks.sandbox import scripted_shard_tokens

            usage = usage_delta(usage, {"total_tokens": scripted_shard_tokens()})

        result = {
            "shard_index": index, "stage": stage, "iteration": iteration,
            "ok": True, "skipped": False, "n": len(rendered),
            "n_errors": len(errors),
            "n_answering": n_answering,
            "n_render_degraded": n_degraded,
            "path": str(shard_file),
            "phase_checkpoint_ids": shard.get("phase_checkpoint_ids", []),
            "duration_s": round(time.monotonic() - started, 3),
            "usage": usage,
            # The Critic reports each crashed checkpoint by name and traceback, so
            # a cap below the shard size silently hides failures it is meant to
            # attribute. Still capped: a shard that fails wholesale must not write
            # its entire traceback set into the run state.
            "errors": errors[:10],
        }
        if errors:
            # Per-checkpoint errors are recorded for the breaker but do not
            # fail the shard: 3 bad checkpoints out of 10 is a data problem,
            # not a broken build.
            breaker.record(signature_from_trace(str(errors[0].get("trace", "")), "checkpoint"))
        return {"eval_results": [result], "token_usage": result["usage"]}

    except Exception as exc:  # noqa: BLE001 - failures are data, not exceptions
        trace = traceback.format_exc()
        signature = signature_from_trace(trace, category="eval_shard")
        breaker.record(signature)
        log.error("shard %d failed: %s", index, _diagnosis(exc))
        return {
            "eval_results": [{
                "shard_index": index, "stage": stage, "iteration": iteration,
                "ok": False, "skipped": False, "n": 0,
                "failure_signature": signature,
                # Tail-preserving for the same reason the log line is: the
                # child's traceback ends with the cause, and `[:500]` cut the
                # `FileNotFoundError` off mid-path in the report the Critic reads.
                "error": _clip_keeping_tail(str(exc), 1200), "trace": trace[-2000:],
                "duration_s": round(time.monotonic() - started, 3),
            }],
            "failure_signature": signature,
        }


# ======================================================================
# Collector (fan-in)
# ======================================================================


async def eval_collect_node(state: OrchestratorState) -> dict[str, Any]:
    """Concatenate shard files into one `predictions.jsonl`.

    Runs exactly once after the whole `Send` superstep, which is what makes the
    fan-in deterministic: shards are sorted by index before concatenation, so
    `predictions.jsonl` has a stable order regardless of which worker finished
    first.  Two runs of the same iteration produce byte-identical files.
    """
    iteration = int(state.get("iteration_count", 1))
    stage = str(state.get("eval_stage") or "dev")
    phase = str(state.get("current_curriculum_phase") or "")

    async with node_span("eval_collect", iteration, phase, stage=stage) as span:
        results = [r for r in (state.get("eval_results") or [])
                   if r.get("iteration") == iteration and r.get("stage") == stage]
        results.sort(key=lambda r: r.get("shard_index", 0))

        stage_path = config.stage_dir(iteration, stage)
        stage_path.mkdir(parents=True, exist_ok=True)
        predictions_path = stage_path / "predictions.jsonl"

        written = 0
        with predictions_path.open("w", encoding="utf-8") as out:
            for result in results:
                path = result.get("path")
                if not result.get("ok") or not path or not Path(path).is_file():
                    continue
                for line in Path(path).read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        out.write(line + "\n")
                        written += 1

        failures = [r for r in results if not r.get("ok")]
        skipped = [r for r in failures if r.get("skipped")]
        breaker = breaker_for(iteration, stage)

        # THE ANSWERER'S HEALTH, aggregated across the whole stage.
        #
        # WHY IT IS COMPUTED HERE AND SHOUTED ABOUT. A render that fails falls
        # back to the gated record bodies (see `_render_answer`), which is right
        # for one checkpoint and catastrophic as a silent default for a whole
        # run: the pipeline keeps producing predictions, the Judge keeps scoring
        # them, and the numbers describe the retrieval layer with raw evidence
        # pasted in rather than the system under test.
        #
        # run-c993a6e93050 did exactly that for 23 iterations and 12.5M tokens
        # with the local vLLM server down. Nothing in the run said so. The
        # summary reported `best MGS=0.8366`.
        n_answering = sum(int(r.get("n_answering", 0)) for r in results if r.get("ok"))
        n_degraded = sum(int(r.get("n_render_degraded", 0)) for r in results if r.get("ok"))
        degraded_rate = (n_degraded / n_answering) if n_answering else 0.0
        degraded = degraded_rate >= config.RENDER_DEGRADED_THRESHOLD

        write_artifact(stage_path / "shard_report.json", {
            "n_shards": len(results),
            "n_failed": len(failures),
            "n_skipped": len(skipped),
            "n_predictions": written,
            "n_answering": n_answering,
            "n_render_degraded": n_degraded,
            "render_degraded_rate": round(degraded_rate, 4),
            "render_degraded": degraded,
            "circuit_breaker_signature": breaker.tripped_signature,
            "shards": results,
        })

        if n_degraded:
            level = log.error if degraded else log.warning
            level(
                "ANSWERER DEGRADED iter=%d stage=%s: %d of %d answering checkpoints "
                "fell back to raw evidence because the evaluator model did not "
                "respond (%.0f%%). These predictions were NOT written by a model, "
                "so U/A/F/MGS for this stage measure the gating layer with record "
                "bodies pasted in as the answer -- not the system under test. "
                "Check that %s is reachable.",
                iteration, stage, n_degraded, n_answering, degraded_rate * 100,
                config.VLLM_BASE_URL,
            )

        log.info(
            "collect iter=%d stage=%s: %d predictions from %d/%d shards "
            "(%d failed, %d skipped by breaker)",
            iteration, stage, written, len(results) - len(failures), len(results),
            len(failures) - len(skipped), len(skipped),
        )
        span["n_predictions"] = written
        span["n_failed_shards"] = len(failures)

        span["n_render_degraded"] = n_degraded
        span["render_degraded"] = degraded

        update: dict[str, Any] = {
            "predictions_path": str(predictions_path),
            "n_checkpoints_evaluated": written,
            "n_worker_failures": len(failures),
            "render_degraded": degraded,
            "render_degraded_rate": round(degraded_rate, 4),
            "n_render_degraded": n_degraded,
            "node_timings": [span],
        }
        if breaker.is_tripped:
            update["failure_signature"] = breaker.tripped_signature
            update["halt_reason"] = (
                f"circuit breaker: {config.FAILFAST_SIGNATURE_K}+ shards failed with "
                f"signature {breaker.tripped_signature}"
            )
        return update
