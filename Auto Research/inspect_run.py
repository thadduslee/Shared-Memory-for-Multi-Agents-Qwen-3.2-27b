#!/usr/bin/env python3
"""Print one digest per iteration: what each of the five nodes actually produced.

    python inspect_run.py runs_5iter                 # every iteration
    python inspect_run.py runs_5iter --iteration 3   # just one
    python inspect_run.py runs_5iter --full          # + the notebook, unabridged

WHY THIS EXISTS.  A five-iteration run leaves ~40 files across five directories,
and the question you actually have -- "did the loop learn anything between
iteration 3 and iteration 4" -- is answered by six numbers scattered across four
of them.  `check_learning.py` grades the run against pass/fail channels; this
just shows you what happened, node by node, so you can read it yourself.

It reads ONLY artifacts on disk, so it works on a finished run, an interrupted
one, and a run that is still going.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

BOLD, DIM, GREEN, YELLOW, RED, CYAN, OFF = (
    "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[36m", "\033[0m"
)

GATE_TOOLS = {
    "compile_check": "compile", "run_tests": "tests", "sql_exec": "migration",
    "run_linter": "lint", "run_sandbox_smoke_test": "smoke",
}


def _load(path: Path) -> Any:
    """Missing and half-written artifacts are expected: a run can still be going."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _kb(path: Path) -> str:
    return f"{path.stat().st_size / 1000:.1f}k" if path.is_file() else "MISSING"


def _wrap(body: str, indent: str = "    ", width: int = 96) -> str:
    """One paragraph, hard-wrapped, so a 100-token recap stays readable."""
    words, lines, line = body.split(), [], ""
    for word in words:
        if len(line) + len(word) + 1 > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    lines.append(line)
    return "\n".join(indent + l for l in lines if l)


def _is_scratch(name: str) -> bool:
    """Mirrors `nodes.dev_tools.is_scratch`, without importing the orchestrator."""
    return name.startswith("_") and not name.startswith("__")


def _episode_changes(iter_dir: Path) -> tuple[list[str], list[str]]:
    """(new, modified) source files, against the baseline the episode inherited.

    `codebase_delta.txt` is a file INVENTORY, not a diff, and `dev_files_changed`
    only reaches disk for the final iteration -- so the per-iteration answer has
    to be recomputed here from `workspace_baseline.json`, which records a sha256
    per inherited file.
    """
    baseline = _load(iter_dir / "workspace_baseline.json")
    workspace = iter_dir / "workspace"
    if not isinstance(baseline, dict) or not workspace.is_dir():
        return [], []
    new, modified = [], []
    for path in sorted(workspace.rglob("*")):
        # Anything hidden is tooling residue -- `.ruff_cache`, `.pytest_cache`,
        # the per-iteration `.cordis` compositions -- and none of it is the
        # Developer's work.
        if not path.is_file() or any(
            part == "__pycache__" or part.startswith(".") for part in path.parts
        ):
            continue
        rel = path.relative_to(workspace).as_posix()
        # The Developer's own scratch, by the same rule `nodes.dev_tools`
        # applies -- one leading underscore, but not a dunder. `ruff.toml` and
        # the two runner shims are toolbox scaffolding, not the deliverable.
        if _is_scratch(path.name) or rel == "ruff.toml":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if rel not in baseline:
            new.append(rel)
        elif baseline[rel] != digest:
            modified.append(rel)
    return new, modified


# ======================================================================
# one node, one block
# ======================================================================


def architect(iter_dir: Path) -> None:
    design_json = _load(iter_dir / "design.json") or {}
    work_order = design_json.get("work_order") or []
    print(f"  {BOLD}ARCHITECT{OFF}  design.md {_kb(iter_dir / 'design.md')} · "
          f"migration.sql {_kb(iter_dir / 'migration.sql')} · "
          f"targets {CYAN}{design_json.get('targets_metric')}{OFF} · "
          f"{len(work_order)} work-order step(s)")

    for i, step in enumerate(work_order[:4], 1):
        print(f"    {DIM}{i}. {str(step)[:88]}{OFF}")
    if len(work_order) > 4:
        print(f"    {DIM}... {len(work_order) - 4} more{OFF}")

    added = design_json.get("critique_recap_added") or []
    if added:
        rows = ", ".join(f"iter {e.get('iteration')} ({e.get('kind')})" for e in added)
        print(f"    {GREEN}appended to critique_summary.md after this design:{OFF} {rows}")
    else:
        print(f"    {DIM}appended to critique_summary.md after this design: (nothing){OFF}")

    mitigations = design_json.get("dev_failure_mitigations") or []
    for m in mitigations:
        print(f"    {YELLOW}mitigating the build failure:{OFF} {str(m)[:86]}")


def developer(iter_dir: Path) -> None:
    steps = _load(iter_dir / "developer_scratchpad.json") or []
    failure = _load(iter_dir / "developer_failure.json")
    ok = sum(1 for s in steps if s.get("ok") is True or s.get("ok") == "True")

    verdict = f"{RED}FAILED{OFF}" if failure else f"{GREEN}green{OFF}"
    print(f"  {BOLD}DEVELOPER{OFF}  {verdict} · {len(steps)} step(s), {ok} ok / "
          f"{len(steps) - ok} failed · delta {_kb(iter_dir / 'codebase_delta.txt')}")

    gates = {}
    for step in steps:
        action = step.get("action")
        tool = action.get("tool") if isinstance(action, dict) else None
        if tool in GATE_TOOLS:
            gates[GATE_TOOLS[tool]] = bool(step.get("ok") is True or step.get("ok") == "True")
    if gates:
        rendered = "  ".join(
            f"{GREEN if v else RED}{k}{OFF}" for k, v in gates.items())
        print(f"    gates reached: {rendered}")

    new, modified = _episode_changes(iter_dir)
    if new or modified:
        parts = []
        if modified:
            parts.append(f"modified {', '.join(modified[:4])}"
                         + (f" (+{len(modified) - 4})" if len(modified) > 4 else ""))
        if new:
            parts.append(f"new {', '.join(new[:4])}"
                         + (f" (+{len(new) - 4})" if len(new) > 4 else ""))
        print(f"    files changed: {' · '.join(parts)}")
    else:
        # Green gates and unchanged code are both true of a no-op episode, and
        # only the first shows up in the run summary. Say the second out loud.
        print(f"    {YELLOW}files changed: NONE -- this episode inherited its code unchanged{OFF}")

    if failure:
        print(f"    {RED}reason:{OFF} {str(failure.get('reason'))[:88]}")
        print(f"    {RED}unmet mandatory gates:{OFF} "
              f"{', '.join(failure.get('missing_gates') or []) or 'none'} · "
              f"retries {failure.get('retries_used')}/{failure.get('retry_cap')} · "
              f"pass rate {float(failure.get('pass_rate', 0.0)):.3f}")


def evaluator(iter_dir: Path) -> None:
    ran = False
    for stage in ("dev", "full"):
        stage_dir = iter_dir / stage
        report = _load(stage_dir / "shard_report.json")
        if report is None:
            continue
        ran = True
        breaker = report.get("circuit_breaker_signature")
        print(f"  {BOLD}EVALUATOR{OFF}  {stage:<4} {report.get('n_predictions', 0)} prediction(s) "
              f"from {report.get('n_shards', 0)} shard(s) · "
              f"{report.get('n_failed', 0)} failed, {report.get('n_skipped', 0)} skipped"
              + (f" · {RED}circuit breaker: {str(breaker)[:30]}{OFF}" if breaker else ""))
        actions: dict[str, int] = {}
        for line in _text(stage_dir / "predictions.jsonl").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                # A malformed line is the Judge's finding to report, not this
                # script's -- `n_malformed_lines` above already counts it.
                continue
            action = str((row.get("output") or {}).get("action"))
            actions[action] = actions.get(action, 0) + 1
        if actions:
            spread = "  ".join(f"{k}={v}" for k, v in sorted(actions.items(), key=lambda kv: -kv[1]))
            print(f"    {DIM}actions: {spread}{OFF}")
    if not ran:
        print(f"  {BOLD}EVALUATOR{OFF}  {DIM}never ran (the build failed before it){OFF}")


def judge(iter_dir: Path, mgs_target: float, dev_gate: float) -> None:
    report = _load(iter_dir / "judge_report.json")
    if report is None:
        print(f"  {BOLD}JUDGE{OFF}      {DIM}never ran{OFF}")
        return
    mgs = float(report.get("MGS", 0.0))
    colour = GREEN if mgs >= mgs_target else (YELLOW if mgs >= dev_gate else RED)
    print(f"  {BOLD}JUDGE{OFF}      U={report.get('U', 0.0):.4f}  "
          f"A={report.get('A', 1.0):.4f}  F={report.get('F', 1.0):.4f}  ->  "
          f"{colour}MGS={mgs:.4f}{OFF}  {DIM}(gate {dev_gate}, target {mgs_target}){OFF}")
    print(f"    {DIM}scored {report.get('n_scored', 0)} "
          f"(U:{report.get('n_utility', 0)} A:{report.get('n_privacy', 0)} "
          f"F:{report.get('n_safety', 0)}) · malformed {report.get('n_malformed_lines', 0)} · "
          f"missing {report.get('n_missing_predictions', 0)}{OFF}")
    # THE COMPARISON, not just the number. An iteration's own MGS says nothing
    # about whether the change that produced it helped, and reading four of
    # these in a row without the comparison is how run-8cf58d33b311 looked like
    # four ordinary iterations while it lost 0.2 MGS.
    verdict = str(report.get("verdict_vs_best") or "")
    if verdict:
        best, best_iter = report.get("best_mgs_so_far"), report.get("best_iteration_so_far")
        label = {
            "improved": f"{GREEN}NEW BEST{OFF}",
            "first": f"{DIM}first measurement{OFF}",
            "tied": f"{YELLOW}TIED with the best -- this code is not adopted{OFF}",
            "regression": f"{RED}REGRESSION{OFF}",
        }.get(verdict, verdict)
        against = (f"  {DIM}(best so far: {float(best):.4f} at iteration {best_iter}){OFF}"
                   if best is not None and verdict != "first" else "")
        print(f"    {label}{against}")
    worst = sorted(
        ((k, v) for k, v in (report.get("by_attack_type") or {}).items() if v.get("n")),
        key=lambda kv: -(kv[1]["fail"] / kv[1]["n"]))[:3]
    if worst:
        spread = "  ".join(f"{k} {v['fail']}/{v['n']}" for k, v in worst)
        print(f"    {DIM}worst attack types: {spread}{OFF}")


def critic(iter_dir: Path) -> None:
    attribution = _load(iter_dir / "attribution.json")
    critique_path = iter_dir / "critique.md"
    if attribution is None:
        print(f"  {BOLD}CRITIC{OFF}     {DIM}never ran (no critique this iteration){OFF}")
        return
    print(f"  {BOLD}CRITIC{OFF}     dominant {CYAN}{attribution.get('dominant_term')}{OFF} "
          f"({float(attribution.get('dominant_gain', 0.0)):+.4f} MGS if perfected) · "
          f"{len(attribution.get('proposals') or [])} proposal(s) · "
          f"critique.md {_kb(critique_path)}")
    print(f"    {DIM}component: {str(attribution.get('component'))[:88]}{OFF}")
    mechanisms = attribution.get("observed_mechanisms") or []
    if mechanisms:
        print(f"    {DIM}observed mechanisms: {', '.join(str(m) for m in mechanisms)}{OFF}")


def notebook(root: Path, show_full: bool) -> None:
    """`critique_summary.md` -- the Architect's notebook, one file for the run.

    Printed once, after the per-iteration blocks, because that is what it is:
    the accumulated history rather than an artifact of any one iteration. Each
    row was appended by the Architect turn AFTER the one whose critique it
    summarises, which is why the last iteration's critique has no row yet.
    """
    path = root / "critique_summary.md"
    if not path.is_file():
        print(f"  {YELLOW}no critique_summary.md -- the Architect kept no notebook{OFF}")
        return
    rows = [line[2:] for line in _text(path).splitlines() if line.startswith("- ")]
    print(f"  {GREEN}critique_summary.md {_kb(path)} · {len(rows)} iteration(s) summarised{OFF}")
    for row in rows:
        if show_full:
            print(_wrap(row, indent="    "))
        else:
            print(f"    {row[:94]}")


# ======================================================================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", help="a runs directory, e.g. runs_5iter")
    parser.add_argument("--iteration", type=int, default=None, help="only this one")
    parser.add_argument("--full", action="store_true",
                        help="print the notebook's entries unabridged")
    parser.add_argument("--mgs-target", type=float, default=0.85)
    parser.add_argument("--dev-gate", type=float, default=0.80)
    args = parser.parse_args(argv)

    root = Path(args.runs_dir).expanduser()
    if not root.is_dir():
        print(f"{RED}no such directory: {root}{OFF}")
        return 2

    iterations = sorted(
        (p for p in root.glob("iter_*") if p.is_dir()),
        key=lambda p: int(p.name.split("_")[1]),
    )
    if args.iteration:
        iterations = [p for p in iterations if p.name == f"iter_{args.iteration}"]
    if not iterations:
        print(f"{RED}no iter_* directories under {root}{OFF}")
        return 2

    for iter_dir in iterations:
        print()
        print("=" * 98)
        print(f"{BOLD}{iter_dir.name.upper().replace('_', ' ')}{OFF}   {DIM}{iter_dir}{OFF}")
        print("=" * 98)
        architect(iter_dir)
        developer(iter_dir)
        evaluator(iter_dir)
        judge(iter_dir, args.mgs_target, args.dev_gate)
        critic(iter_dir)

    # The Architect's notebook: one file for the whole run, not per iteration.
    print()
    print("=" * 98)
    print(f"{BOLD}THE ARCHITECT'S NOTEBOOK{OFF}   {DIM}{root / 'critique_summary.md'}{OFF}")
    print("=" * 98)
    notebook(root, args.full)

    # The run's own narrative, end to end.
    summaries = sorted(root.glob("summary_*.json"))
    if summaries:
        summary = _load(summaries[-1]) or {}
        print()
        print("=" * 98)
        print(f"{BOLD}RUN SUMMARY{OFF}   {DIM}{summaries[-1].name}{OFF}")
        print("=" * 98)
        metrics = summary.get("metrics") or {}
        board = summary.get("scoreboard") or {}
        print(f"  halt reason      : {summary.get('halt_reason')}")
        print(f"  iterations       : {summary.get('iterations')}")
        print(f"  final MGS        : {metrics.get('MGS_compliance_utility_score')}")
        # WHICH WORKSPACE IS ACTUALLY WORTH KEEPING. On a run that regressed,
        # the final MGS above describes the LAST code, which is not the best
        # code -- and the last code is what a reader assumes "the result" means.
        if board:
            colour = RED if board.get("regressed_from_best") else GREEN
            print(f"  best MGS         : {colour}{board.get('best_mgs')}"
                  f" at iteration {board.get('best_iteration')}{OFF}")
            if board.get("regressed_from_best"):
                print(f"  {RED}{BOLD}!! this run ended BELOW its own best "
                      f"({board.get('regression_streak')} iteration(s) in a row){OFF}")
                best = root / f"iter_{int(board.get('best_iteration') or 0)}" / "workspace"
                print(f"  {RED}   keep: {best}{OFF}")
        print(f"  tokens           : {(summary.get('token_usage') or {}).get('total_tokens')}")
        digest = summary.get("critique_digest") or []
        print(f"  {BOLD}recap digest ({len(digest)} entries){OFF}")
        for entry in digest:
            print(f"    {CYAN}iteration {entry.get('iteration')} ({entry.get('kind')}){OFF}")
            print(_wrap(str(entry.get("summary") or ""), indent="      "))
    return 0


if __name__ == "__main__":
    sys.exit(main())
