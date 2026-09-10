#!/usr/bin/env python3
"""Entrypoint for the GateMem self-improving research loop.

Runs offline out of the box:

    python main.py                              # MOCK_MODE, happy_path scenario
    python main.py --list-scenarios             # every forceable routing path
    python main.py --scenario dev_gate_fail     # force one of them
    python main.py --real                       # real dsh / OpenRouter / vLLM / OpenAI

Exit code is 0 when the run terminated for a normal reason (target reached,
iteration budget spent) and 1 when it halted on a failure (budget guard,
circuit breaker, unbuildable design), so CI can tell the difference.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from pathlib import Path

# The project root must be importable before anything else, because the package
# lives in a directory whose name contains a space and cannot be a package name.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402
import scoreboard  # noqa: E402
from nodes._common import setup_logging, write_artifact  # noqa: E402
from state import initial_state  # noqa: E402

log = logging.getLogger("orchestrator.main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", default=None,
                        help="mock scenario name; see --list-scenarios")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="print the forceable routing paths and exit")
    parser.add_argument("--real", action="store_true",
                        help="disable MOCK_MODE and use the real dsh/OpenRouter/vLLM/OpenAI clients")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--dev-checkpoints", type=int, default=None,
                        help="size of the dev slice (default 50); lower it to make a real run cheap")
    parser.add_argument("--no-full", action="store_true",
                        help="never scale up to the full 579-checkpoint stage, whatever the gate says")
    parser.add_argument("--smoke", action="store_true",
                        help="cheapest useful real run: 6-checkpoint dev slice, 1 iteration, no full "
                             "stage. Use this to verify connectivity before spending a real budget.")
    parser.add_argument("--mgs-target", type=float, default=None)
    parser.add_argument("--dev-gate", type=float, default=None)
    parser.add_argument("--thread-id", default=None,
                        help="checkpointer thread id; reuse one to resume an interrupted run")
    parser.add_argument("--log-level", default=None, choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--preflight", action="store_true",
                        help="probe every configured endpoint with a tiny request, then exit")
    parser.add_argument("--print-graph", action="store_true",
                        help="print the compiled graph's mermaid source and exit")
    return parser.parse_args(argv)


def apply_overrides(args: argparse.Namespace) -> None:
    """CLI overrides land on the config module.

    Mutating module attributes rather than threading a settings object through
    every node keeps `config` the single source of truth that the brief asks
    for; nodes read `config.X` at call time, so the override is seen everywhere.
    """
    if args.real:
        config.MOCK_MODE = False
    if args.scenario:
        config.MOCK_SCENARIO = args.scenario
    if args.max_iterations is not None:
        config.MAX_ITERATIONS = args.max_iterations
    if args.mgs_target is not None:
        config.MGS_TARGET = args.mgs_target
    if args.dev_gate is not None:
        config.DEV_GATE_MGS = args.dev_gate
    if args.dev_checkpoints is not None:
        config.EXPECTED_DEV_CHECKPOINTS = args.dev_checkpoints
    if args.no_full:
        config.SKIP_FULL_STAGE = True
    if args.smoke:
        # Deliberately tiny: this run exists to prove every endpoint answers,
        # not to produce a number anyone should quote.
        config.EXPECTED_DEV_CHECKPOINTS = args.dev_checkpoints or 6
        config.MAX_ITERATIONS = args.max_iterations or 1
        config.SKIP_FULL_STAGE = True


def check_transport_deps() -> str | None:
    """Refuse to start a real run whose transport cannot possibly work.

    Without this the run gets as far as the Architect, spends its wall clock on
    retries, and reports a halt reason -- an expensive way to discover that the
    wrong virtualenv is active. Returns an error string, or None if the
    configured transport is usable.
    """
    import importlib.util
    import shutil
    import sys

    if config.AGENT_TRANSPORT != "dsh":
        return None
    if importlib.util.find_spec("deepseek_harness") is not None:
        return None
    if shutil.which("npx"):
        return None
    return (
        f"AGENT_TRANSPORT=dsh, but the harness is unreachable:\n"
        f"  interpreter : {sys.executable}\n"
        f"  deepseek_harness : NOT importable here\n"
        f"  npx              : not on PATH (so the CLI fallback cannot run either)\n"
        f"Fix by either:\n"
        f"  1. activating the project environment:  "
        f"source \"{config.PROJECT_ROOT}/.venv/bin/activate\"\n"
        f"  2. or bypassing the harness:            AGENT_TRANSPORT=http python main.py ...\n"
        f"Run `python main.py --preflight --real` to check every endpoint at once."
    )


async def preflight() -> int:
    """Verify every configured endpoint answers, before spending a real budget.

    Sends one minimal completion per route actually in use. The whole check
    costs a handful of tokens, and it catches the three failures that account
    for nearly every broken first run: a base URL pointing somewhere else, a
    model slug the provider does not recognise, and a missing key.
    """
    import os

    from harness.profiles import (
        ARCHITECT_PROFILE, EVALUATOR_PROFILE, JUDGE_PROFILE,
    )
    from llm.client import AsyncLLMClient

    print("\n" + "=" * 78)
    print(f"PREFLIGHT  (MOCK_MODE={config.MOCK_MODE})")
    print("=" * 78)

    ok = True

    # --- dataset -----------------------------------------------------
    data_file = config.GATEMEM_DATA_DIR / "checkpoints.jsonl"
    if data_file.is_file():
        n = sum(1 for line in data_file.open(encoding="utf-8") if line.strip())
        print(f"  [ ok ] GateMem medical data      {n} checkpoints at {config.GATEMEM_DATA_DIR}")
        if n != config.EXPECTED_FULL_CHECKPOINTS:
            print(f"         note: expected {config.EXPECTED_FULL_CHECKPOINTS}")
    else:
        ok = False
        print(f"  [FAIL] GateMem medical data      not found at {config.GATEMEM_DATA_DIR}")

    # --- one probe per distinct (route, model) pair -------------------
    probes = [
        ("architect/developer/critic", ARCHITECT_PROFILE.route, config.ARCHITECT_MODEL,
         ARCHITECT_PROFILE.api_key_env),
        ("evaluator", EVALUATOR_PROFILE.route, config.EVALUATOR_MODEL,
         EVALUATOR_PROFILE.api_key_env),
        ("judge", JUDGE_PROFILE.route, config.JUDGE_MODEL, JUDGE_PROFILE.api_key_env),
    ]
    seen: set[tuple[str, str]] = set()
    client = AsyncLLMClient()
    try:
        for label, route, model, key_env in probes:
            if (route, model) in seen:
                print(f"  [ ok ] {label:<28} shares the probe above ({route}/{model})")
                continue
            seen.add((route, model))
            if key_env and not os.environ.get(key_env) and route != "vllm":
                ok = False
                print(f"  [FAIL] {label:<28} {key_env} is not set")
                continue
            result = await client.chat(
                route=route, model=model,
                messages=[{"role": "user", "content": "Reply with the single word: ok"}],
                temperature=0.0, max_tokens=16, role="preflight",
            )
            if result.ok:
                print(f"  [ ok ] {label:<28} {route}/{model}  "
                      f"({result.usage.get('total_tokens', 0)} tokens)")
            else:
                ok = False
                print(f"  [FAIL] {label:<28} {route}/{model}")
                print(f"         {str(result.error)[:200]}")
    finally:
        await client.aclose()

    # --- dsh runtime, only if the agentic nodes will use it ------------
    if config.AGENT_TRANSPORT == "dsh":
        try:
            import deepseek_harness  # noqa: F401

            print("  [ ok ] dsh Python SDK             importable "
                  "(AGENT_TRANSPORT=dsh will use the harness)")
            print("         OpenRouter via the harness's DEEPSEEK_BASE_URL is verified working.")
            print("         If the Architect still fails, AGENT_TRANSPORT=http is an "
                  "equivalent fallback.")
        except ImportError:
            print("  [warn] dsh Python SDK             not importable; the CLI fallback is a guess.")
            print("         Either `pip install deepseek-harness-sdk` or use AGENT_TRANSPORT=http.")

    print("=" * 78)
    print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED -- fix the above before a real run")
    if ok:
        # A POINT-IN-TIME CHECK, and worth saying so. run-c993a6e93050 passed
        # this and then lost its evaluator an hour in: 7,496 ConnectErrors,
        # every render falling back to raw evidence, and 23 iterations scored
        # against it. The in-run detection is what covers that (see
        # `render_degraded` in nodes/medical_evaluator.py); this line is here so
        # nobody reads a green preflight as a guarantee about the next 3 hours.
        print("  note: this is a snapshot. An endpoint that dies mid-run is caught by the")
        print("        evaluator's own degradation check, which halts rather than scoring")
        print("        answers the model never wrote (HALT_ON_DEGRADED_EVAL).")
    print("=" * 78 + "\n")
    return 0 if ok else 1


async def run(args: argparse.Namespace) -> int:
    from graph import build_graph, default_checkpointer

    graph = build_graph(checkpointer=default_checkpointer())

    if args.print_graph:
        print(graph.get_graph().draw_mermaid())
        return 0

    thread_id = args.thread_id or f"run-{uuid.uuid4().hex[:12]}"
    workspace = config.iteration_dir(1) / "workspace"
    state = initial_state(workspace=str(workspace), started_at=time.monotonic())

    log.info("=" * 78)
    log.info("GateMem self-improving orchestrator | thread=%s", thread_id)
    log.info("mode=%s scenario=%s", "MOCK" if config.MOCK_MODE else "REAL", config.MOCK_SCENARIO)
    log.info("MGS_TARGET=%.2f DEV_GATE_MGS=%.2f MAX_ITERATIONS=%d MAX_DEV_RETRIES=%d",
             config.MGS_TARGET, config.DEV_GATE_MGS, config.MAX_ITERATIONS, config.MAX_DEV_RETRIES)
    log.info("dev slice=%d checkpoints | full stage=%s | seed=%s",
             config.EXPECTED_DEV_CHECKPOINTS,
             "DISABLED (--no-full/--smoke)" if config.SKIP_FULL_STAGE else "enabled at the gate",
             "templates/" if config.SEED_FROM_TEMPLATE else "empty workspace")
    log.info("models: architect=%s evaluator=%s judge=%s",
             config.ARCHITECT_MODEL, config.EVALUATOR_MODEL, config.JUDGE_MODEL)
    log.info("=" * 78)

    final = await graph.ainvoke(
        state,
        config={
            "recursion_limit": config.RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        },
    )

    summary = {
        "thread_id": thread_id,
        "mode": "mock" if config.MOCK_MODE else "real",
        "scenario": config.MOCK_SCENARIO if config.MOCK_MODE else None,
        "iterations": final.get("iteration_count", 0),
        "halt_reason": final.get("halt_reason"),
        "final_curriculum_phase": final.get("current_curriculum_phase"),
        # THE LAST ITERATION'S NUMBERS. Read them beside `scoreboard` below,
        # not instead of it: on a run that regressed these describe the WORST
        # code the run produced, because they describe the most recent code.
        "final_metrics": {
            "U_utility_accuracy": final.get("utility_score"),
            "A_privacy_leakage_rate": final.get("access_violation_rate"),
            "F_deletion_leakage_rate": final.get("forgetting_failure_rate"),
            "MGS_compliance_utility_score": final.get("mgs_score"),
        },
        # Retained under the old key so anything that parsed these summaries
        # before still finds them; `final_metrics` is the honest name.
        "metrics": {
            "U_utility_accuracy": final.get("utility_score"),
            "A_privacy_leakage_rate": final.get("access_violation_rate"),
            "F_deletion_leakage_rate": final.get("forgetting_failure_rate"),
            "MGS_compliance_utility_score": final.get("mgs_score"),
        },
        # THE WHOLE TRAJECTORY, and which iteration actually won. A run summary
        # that reports only the final measurement cannot distinguish a loop that
        # improved for five iterations from one that collapsed for five, and
        # run-8cf58d33b311 -- 0.3172 -> 0.2222 -> 0.1830 -> 0.1190 -- looked
        # exactly like a successful run in its own summary file. `best_iteration`
        # is also the answer to "which workspace should I actually ship?".
        "scoreboard": scoreboard.summary(final),
        # Named for what it is. `dev_set_pass_rate` read like a score on the
        # benchmark's dev set; it is the Developer's own unit-test suite in its
        # own workspace, and runs_4iter reported it as 1.0 directly above an MGS
        # of 0.0 -- two true numbers that together said something false.
        "dev_unit_test_pass_rate": final.get("dev_set_pass_rate"),
        # Which source files the last Developer episode actually changed. Empty
        # next to green gates means the episode built nothing and the metrics
        # below describe inherited code.
        "dev_files_changed": final.get("dev_files_changed", []),
        "predictions_path": final.get("predictions_path"),
        "n_checkpoints_evaluated": final.get("n_checkpoints_evaluated"),
        "n_worker_failures": final.get("n_worker_failures"),
        "failure_signature": final.get("failure_signature"),
        # One entry per iteration whose build never went green. A run that
        # reached MAX_ITERATIONS with an empty history spent its budget on real
        # evaluations; one with three entries spent it on designs that could not
        # be built, and the two need telling apart without reading every log.
        "dev_failure_history": final.get("dev_failure_history", []),
        # The run's own narrative: one capped line per iteration, written by the
        # Architect that read that iteration's critique. This is the structured
        # mirror of `runs/critique_summary.md`, the notebook the Architect keeps.
        # Reading it beside `metrics` is how you tell a run that diagnosed three
        # different things from one that diagnosed the same thing three times.
        "critique_digest": final.get("critique_digest", []),
        "token_usage": final.get("token_usage"),
        "node_timings": final.get("node_timings", []),
    }
    write_artifact(config.RUNS_DIR / f"summary_{thread_id}.json", summary)

    print("\n" + "=" * 78)
    print(json.dumps({k: v for k, v in summary.items() if k != "node_timings"}, indent=2, default=str))
    print("=" * 78)
    # THE ONE LINE A HUMAN ACTUALLY READS. A run that ended below its own best
    # must not be able to scroll past looking like every other run: the final
    # workspace is not the one worth keeping, and which one IS worth keeping is
    # a question the summary can answer in a sentence.
    board = summary["scoreboard"]
    if board["regressed_from_best"]:
        print(
            f"!! THIS RUN REGRESSED. Best MGS={board['best_mgs']:.4f} at iteration "
            f"{board['best_iteration']}; the run ended at {board['final_mgs']:.4f} "
            f"(iteration {board['final_iteration']}).\n"
            f"   The code worth keeping is "
            f"{config.iteration_dir(board['best_iteration']) / 'workspace'}"
        )
    elif board["best_iteration"]:
        print(f"best MGS={board['best_mgs']:.4f} at iteration {board['best_iteration']} "
              f"(also the final iteration)"
              if board["best_iteration"] == board["final_iteration"]
              else f"best MGS={board['best_mgs']:.4f} at iteration {board['best_iteration']}")
    print(f"artifacts: {config.RUNS_DIR}")

    reason = str(final.get("halt_reason") or "")
    clean = reason.startswith("target reached") or reason.startswith("iteration budget")
    return 0 if clean else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    apply_overrides(args)
    setup_logging(args.log_level)

    if args.list_scenarios:
        from mocks.scripted import scenario_table

        print(f"{'scenario':<24} {'default MGS':>12}  description")
        print("-" * 100)
        for row in scenario_table():
            print(f"{row['name']:<24} {row['default_mgs']:>12.4f}  {row['description']}")
        return 0

    if not config.MOCK_MODE:
        import os

        from harness.profiles import ALL_PROFILES

        # Derived from the profiles actually in use, not a fixed list: with
        # JUDGE_ROUTE=openrouter there is no OpenAI key to miss, and warning
        # about one would send you looking for a problem you do not have.
        # The local vLLM route is exempt -- it accepts any bearer token.
        missing = sorted({
            profile.api_key_env for profile in ALL_PROFILES.values()
            if profile.api_key_env and profile.route != "vllm"
            and not os.environ.get(profile.api_key_env)
        })
        if missing:
            log.warning("running with --real but these env vars are unset: %s", ", ".join(missing))

    if args.preflight:
        return asyncio.run(preflight())

    if not config.MOCK_MODE:
        problem = check_transport_deps()
        if problem:
            log.error("%s", problem)
            return 1

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        log.warning("interrupted; artifacts under %s are intact", config.RUNS_DIR)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
