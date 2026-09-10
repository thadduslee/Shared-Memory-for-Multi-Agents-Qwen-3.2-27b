#!/usr/bin/env python
"""Deterministic U/A/F/MGS for a workspace, with no model in the loop.

    python bench_template.py templates
    python bench_template.py runs_real5_notebook/iter_1/workspace --slice full
    python bench_template.py templates --compare runs_real5_notebook/iter_5/workspace

WHY THIS EXISTS
===============
The evaluator runs each checkpoint in two phases (see the header of
`nodes/medical_evaluator.py`):

    1. RETRIEVE + SANITIZE -- `store.retrieve()` then `sanitize_and_decide()`.
       No model. This phase alone decides the ACTION LABEL.
    2. RENDER -- a model turns the cleared evidence into prose.

The Judge scores utility as `action_correct and include_ok`, so phase 1 fully
determines one of the two conjuncts and phase 2 only ever determines the other.
That means **the action-shape half of U is measurable offline, exactly,
deterministically, in seconds, for free** -- and it is the half that
run-8cf58d33b311 lost. That run spent 4.9 million tokens across five iterations
to discover a regression this script would have reported in three seconds:

    $ python bench_template.py runs_real5_notebook/iter_1/workspace
    $ python bench_template.py runs_real5_notebook/iter_5/workspace

WHAT THE NUMBERS MEAN, AND WHAT THEY DO NOT
-------------------------------------------
With no model wired in, `GateMemAgent.query` sets the answer to the joined
bodies of the records that cleared the gates. So:

  * The ACTION LABEL is exact -- it is the same code path the real run takes.
  * `A` and `F` are close to exact: both are dominated by "did this answer at
    all when it should have refused", which is a property of the label, and by
    literal leak-target matching against evidence the real answerer would also
    have been handed.
  * `U` is a LOWER BOUND. A rendered answer can drop a required string that the
    raw bodies contain; it cannot invent one that they do not. Treat the U here
    as the ceiling the answerer is allowed to work under.

Use it to compare two workspaces, not to predict a run's headline score.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from gatemem_adapter import memory_governance_score, select_dev_slice

HIDDEN = {"query_type", "attack_type", "expected_action", "judge_spec", "leak_targets",
          "gold_answer_structured", "gold_refusal_category", "policy_snapshot"}


def _parse(value: Any) -> Any:
    """The dataset stores dicts and lists as their Python reprs in some dumps."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _load_dataset(which: str) -> tuple[list[dict], dict[str, dict]]:
    data = config.GATEMEM_DATA_DIR
    checkpoints = [json.loads(line) for line in
                   (data / "checkpoints.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes = {
        episode["episode_id"]: episode
        for episode in (json.loads(line) for line in
                        (data / "episodes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    }
    if which == "dev":
        checkpoints = select_dev_slice(
            checkpoints, n=config.EXPECTED_DEV_CHECKPOINTS, seed=config.DEV_SLICE_SEED
        )
    return checkpoints, episodes


def evaluate(workspace: str | Path, which: str = "dev") -> dict[str, Any]:
    """Score one workspace. Returns metrics plus a per-checkpoint detail list.

    The workspace's `memory_system` package is imported fresh each call --
    every module under that name is evicted first -- so two workspaces can be
    compared in one process without the first one's classes answering for the
    second. Getting this wrong produces two identical scores and a very
    confident wrong conclusion.
    """
    from nodes.judge import rule_score

    path = Path(workspace).resolve()
    if not (path / "memory_system" / "agent.py").is_file():
        raise SystemExit(f"no memory_system package at {path}")

    sys.path.insert(0, str(path))
    for module in [name for name in list(sys.modules) if name.startswith("memory_system")]:
        del sys.modules[module]
    try:
        from memory_system.agent import GateMemAgent

        checkpoints, episodes = _load_dataset(which)
        buckets: dict[str, list[bool]] = collections.defaultdict(list)
        detail: list[dict[str, Any]] = []

        for checkpoint in checkpoints:
            episode = episodes[checkpoint["episode_id"]]
            agent = GateMemAgent(":memory:")
            agent.reset(episode)
            for turn in episode.get("turns", []):
                agent.ingest(turn)
                if turn.get("turn_id") == checkpoint.get("as_of_turn_id"):
                    break
            visible = {k: v for k, v in checkpoint.items() if k not in HIDDEN}
            result = agent.query(visible)
            verdict = rule_score({"output": result}, {
                "query_type": checkpoint["query_type"],
                "expected_action": checkpoint["expected_action"],
                "judge_spec": _parse(checkpoint.get("judge_spec")) or {},
                "leak_targets": _parse(checkpoint.get("leak_targets")) or [],
            })
            query_type = checkpoint["query_type"]
            if query_type == "utility":
                buckets["utility"].append(bool(verdict["utility_correct"]))
            elif query_type == "privacy":
                buckets["privacy"].append(bool(verdict["privacy_leak"]))
            elif query_type == "safety":
                buckets["safety"].append(bool(verdict["deletion_leak"]))
            detail.append({
                "checkpoint_id": checkpoint["checkpoint_id"],
                "query_type": query_type,
                "attack_type": checkpoint.get("attack_type") or "none",
                "expected_action": checkpoint["expected_action"],
                "action": result.get("action"),
                "action_correct": result.get("action") == checkpoint["expected_action"],
                **{k: v for k, v in verdict.items() if isinstance(v, bool)},
            })
    finally:
        sys.path.remove(str(path))

    def mean(values: list[bool]) -> float:
        return (sum(1 for value in values if value) / len(values)) if values else 0.0

    utility, access, forgetting = (
        mean(buckets["utility"]), mean(buckets["privacy"]), mean(buckets["safety"])
    )
    return {
        "workspace": str(path),
        "slice": which,
        "n": len(detail),
        "U": utility, "A": access, "F": forgetting,
        "MGS": memory_governance_score(utility, access, forgetting),
        "n_utility": len(buckets["utility"]),
        "n_utility_correct": sum(1 for v in buckets["utility"] if v),
        "n_privacy": len(buckets["privacy"]),
        "n_privacy_leaks": sum(1 for v in buckets["privacy"] if v),
        "n_safety": len(buckets["safety"]),
        "n_deletion_leaks": sum(1 for v in buckets["safety"] if v),
        "n_action_correct": sum(1 for d in detail if d["action_correct"]),
        "confusions": collections.Counter(
            (d["expected_action"], d["action"]) for d in detail if not d["action_correct"]
        ),
        "detail": detail,
    }


def render(result: dict[str, Any]) -> str:
    lines = [
        f"{result['workspace']}  [{result['slice']}, {result['n']} checkpoints]",
        f"  U   = {result['U']:.4f}   ({result['n_utility_correct']}/{result['n_utility']} utility correct)",
        f"  A   = {result['A']:.4f}   ({result['n_privacy_leaks']}/{result['n_privacy']} privacy leaks)",
        f"  F   = {result['F']:.4f}   ({result['n_deletion_leaks']}/{result['n_safety']} deletion leaks)",
        f"  MGS = {result['MGS']:.4f}",
        f"  action labels correct: {result['n_action_correct']}/{result['n']}",
    ]
    if result["confusions"]:
        lines.append("  confusions (expected -> got):")
        lines += [f"    {want:16s} -> {got:16s}  {count}"
                  for (want, got), count in result["confusions"].most_common()]
    return "\n".join(lines)


def compare(before: dict[str, Any], after: dict[str, Any]) -> str:
    """Per-checkpoint, because an aggregate can improve while cases regress."""
    index = {d["checkpoint_id"]: d for d in before["detail"]}
    scored = {"utility": "utility_correct", "privacy": "privacy_leak", "safety": "deletion_leak"}
    better: collections.Counter = collections.Counter()
    worse: list[str] = []
    for row in after["detail"]:
        old = index.get(row["checkpoint_id"])
        if old is None:
            continue
        key = scored[row["query_type"]]
        # For utility, True is good; for privacy and safety, True is a LEAK.
        was_good = old[key] if row["query_type"] == "utility" else not old[key]
        now_good = row[key] if row["query_type"] == "utility" else not row[key]
        if now_good and not was_good:
            better[row["query_type"]] += 1
        elif was_good and not now_good:
            worse.append(f"{row['checkpoint_id']} [{row['query_type']}] "
                         f"expected={row['expected_action']} "
                         f"was={old['action']} now={row['action']}")
    terms = (
        f"  U {before['U']:.4f} -> {after['U']:.4f}   "
        f"A {before['A']:.4f} -> {after['A']:.4f}   "
        f"F {before['F']:.4f} -> {after['F']:.4f}"
    )
    lines = [
        "",
        f"MGS {before['MGS']:.4f} -> {after['MGS']:.4f}  ({after['MGS'] - before['MGS']:+.4f})",
        terms,
        f"improved: {dict(better) or '(none)'}",
        f"REGRESSED: {len(worse)}",
    ]
    lines += [f"  {row}" for row in worse[:25]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workspace", help="a directory containing memory_system/")
    parser.add_argument("--slice", default="dev", choices=("dev", "full"),
                        help="the seeded 50-checkpoint dev slice, or all 579")
    parser.add_argument("--compare", metavar="WORKSPACE",
                        help="score this one too and diff them per checkpoint")
    parser.add_argument("--json", metavar="PATH", help="write the per-checkpoint detail here")
    args = parser.parse_args(argv)

    if not (config.GATEMEM_DATA_DIR / "checkpoints.jsonl").is_file():
        print(f"no GateMem checkout at {config.GATEMEM_DATA_DIR}; set GATEMEM_REPO",
              file=sys.stderr)
        return 2

    result = evaluate(args.workspace, args.slice)
    if args.compare:
        baseline = evaluate(args.compare, args.slice)
        print(render(baseline))
        print()
        print(render(result))
        print(compare(baseline, result))
    else:
        print(render(result))
    if args.json:
        Path(args.json).write_text(
            "\n".join(json.dumps(row) for row in result["detail"]), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
