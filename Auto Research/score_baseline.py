#!/usr/bin/env python3
"""Score a GateMem baseline's predictions with THIS project's judge.

WHY THIS EXISTS
---------------
`outputs/` holds the paper's baseline systems -- remem_i, mem0, a_mem,
rag_naive, rag_policy, long_context -- each with the `predictions.jsonl` it
produced over all 579 medical checkpoints.  Their published numbers were
produced by a DIFFERENT grader: `openai/gpt-4.1`, and with `gated_by_action`
false, where ours gates utility on the action label and runs a local
Qwen3.8-27B.  Comparing our loop's MGS against those numbers therefore compares
two systems AND two graders at once, and there is no way to tell the two apart
afterwards.

Scoring their stored predictions with our judge removes the second difference.
The predictions are fixed -- no baseline is re-run, nothing is re-generated --
so the only thing that changes is the instrument, and the resulting table is
the honest one to put our own runs next to.

    python score_baseline.py outputs/medical_qwen38_27b_remem_i__judge-gpt41
    python score_baseline.py outputs/*__judge-gpt41 --out baseline_scores.json
    python score_baseline.py outputs/... --limit 50        # a cheap smoke

WHAT IT DOES NOT SETTLE.  Which of `judge_prompt.txt` and
`judge_prompt_gatemem.txt` the published runs used is not recorded anywhere in
`outputs/` -- not in summary.json, not in judge_scores.jsonl -- and the two ask
for identical keys, so the stored verdicts cannot distinguish them.  This
script uses whichever file `config.BENCHMARK_PROMPTS` selects, and reports it,
rather than guessing on the reader's behalf.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

import config
import gatemem_adapter as g
from nodes.judge import _aggregate, _llm_verdict, _merge_one, _score_one


def load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    """`predictions.jsonl` -> {checkpoint_id: row}, malformed lines skipped."""
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = row.get("checkpoint_id")
        if cid:
            rows[str(cid)] = row
    return rows


async def score(
    predictions: dict[str, dict[str, Any]],
    annotations: dict[str, dict[str, Any]],
    visible: dict[str, dict[str, Any]],
    *,
    use_llm: bool,
) -> dict[str, Any]:
    """The rule pass, then the LLM pass, then our own aggregation.

    Deliberately the SAME functions `nodes.judge` calls on a live run. Anything
    reimplemented here would be a second scorer to keep in step, and the whole
    point is that the baselines and our runs go through one.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    for cid, prediction in predictions.items():
        annotation = annotations.get(cid)
        if annotation is None:
            continue
        verdicts[cid] = _score_one(prediction, annotation)

    if use_llm and verdicts:
        ids = list(verdicts)
        outcomes = await asyncio.gather(
            *(
                _llm_verdict(predictions[cid], annotations[cid], verdicts[cid], visible.get(cid))
                for cid in ids
            ),
            return_exceptions=True,
        )
        fallbacks = 0
        for cid, outcome in zip(ids, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                fallbacks += 1
                continue
            llm_verdict, _usage = outcome
            if not llm_verdict:
                fallbacks += 1
            verdicts[cid] = _merge_one(verdicts[cid], llm_verdict)
        if fallbacks:
            print(f"  ! {fallbacks} checkpoint(s) fell back to the rule verdict", file=sys.stderr)

    missing = sorted(set(annotations) - set(predictions))
    for cid in missing:
        verdicts[cid] = {
            "action_correct": False, "utility_correct": False,
            "privacy_leak": True, "deletion_leak": True, "over_refusal": False,
            "reason": "no prediction in this baseline's output", "missing": True,
        }
    return _aggregate(verdicts, annotations, 0, len(missing))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("targets", nargs="+", help="outputs/<run> directories, or predictions.jsonl paths")
    parser.add_argument("--out", help="write the table here as JSON")
    parser.add_argument("--out-dir", help="write per-baseline summary.json and verdicts here")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N checkpoints")
    parser.add_argument("--rules-only", action="store_true", help="skip the LLM pass (free, and a floor)")
    args = parser.parse_args()

    dataset = g.load_medical_dataset(config.GATEMEM_DATA_DIR)
    annotations = dataset.annotations_by_id()
    visible = {cp["checkpoint_id"]: cp for cp in dataset.checkpoints}

    print(f"judge:   {config.JUDGE_MODEL} via {config.JUDGE_ROUTE}")
    print(f"prompts: BENCHMARK_PROMPTS={config.BENCHMARK_PROMPTS} "
          f"(gatemem = bench/prompts/judge_prompt.txt)")
    print(f"scorer:  JUDGE_IMPL={config.JUDGE_IMPL}, utility gated on the action label\n")

    table: dict[str, Any] = {}
    for target in args.targets:
        path = Path(target)
        predictions_path = path / "predictions.jsonl" if path.is_dir() else path
        if not predictions_path.is_file():
            print(f"skip {target}: no predictions.jsonl", file=sys.stderr)
            continue

        predictions = load_predictions(predictions_path)
        scored_annotations = annotations
        if args.limit:
            # The ANNOTATIONS are narrowed too. Narrowing only the predictions
            # would leave the rest counted as missing, and a missing prediction
            # scores as both a privacy and a deletion leak -- so a 40-checkpoint
            # smoke reported A=0.94 and F=1.00 for a system that had answered
            # all 579.
            predictions = dict(list(predictions.items())[: args.limit])
            scored_annotations = {
                cid: annotations[cid] for cid in predictions if cid in annotations
            }

        name = predictions_path.parent.name
        started = time.monotonic()
        report = await score(
            predictions, scored_annotations, visible, use_llm=not args.rules_only,
        )
        elapsed = time.monotonic() - started

        published: dict[str, Any] = {}
        summary_path = predictions_path.parent / "summary.json"
        if summary_path.is_file():
            try:
                published = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                published = {}

        row = {
            "U": report["U"], "A": report["A"], "F": report["F"],
            "MGS": report["MGS"], "OR": report["OR"],
            "n_scored": report["n_scored"],
            "n_missing": report["n_missing_predictions"],
            "published_U": published.get("utility_accuracy"),
            "published_MGS": published.get("compliance_utility_score"),
            "seconds": round(elapsed, 1),
        }
        table[name] = row

        if args.out_dir:
            # One directory per baseline, mirroring outputs/ so the two can be
            # read side by side. `summary.json` carries the SAME metric keys the
            # published one does -- plus `scored_by`, because a summary that
            # does not name its grader is how two runs get compared across two
            # instruments without anyone noticing.
            dest = Path(args.out_dir) / name
            dest.mkdir(parents=True, exist_ok=True)
            summary = {
                "n_checkpoints": report["n_scored"],
                "n_utility": report["n_utility"],
                "n_privacy": report["n_privacy"],
                "n_safety": report["n_safety"],
                "utility_accuracy": report["U"],
                "privacy_leakage_rate": report["A"],
                "deletion_leakage_rate": report["F"],
                "over_refusal_rate": report["OR"],
                "compliance_utility_score": report["MGS"],
                "gated_by_action": True,
                "by_attack_type": report["by_attack_type"],
                "by_curriculum_phase": report["by_curriculum_phase"],
                "utility_failure_buckets": report.get("utility_failure_buckets"),
                "scored_by": {
                    "judge_model": config.JUDGE_MODEL,
                    "judge_route": config.JUDGE_ROUTE,
                    "judge_impl": config.JUDGE_IMPL,
                    "benchmark_prompts": config.BENCHMARK_PROMPTS,
                    "judge_prompt_file": (
                        config.JUDGE_PROMPT_FILE
                        or str(Path(config.GATEMEM_REPO) / "bench/prompts/judge_prompt.txt")
                    ),
                    "predictions_from": str(predictions_path),
                },
                "published_for_reference": {
                    "utility_accuracy": published.get("utility_accuracy"),
                    "compliance_utility_score": published.get("compliance_utility_score"),
                    "gated_by_action": published.get("gated_by_action"),
                    "judge": "openai/gpt-4.1" if "judge-gpt41" in name else "matcher only",
                },
            }
            (dest / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            with (dest / "judge_scores.jsonl").open("w", encoding="utf-8") as handle:
                for cid, verdict in report["verdicts"].items():
                    handle.write(json.dumps({"checkpoint_id": cid, "judge": verdict}) + "\n")

        print(f"{name}")
        print(f"  ours      U={row['U']:.4f} A={row['A']:.4f} F={row['F']:.4f} "
              f"MGS={row['MGS']:.4f} OR={row['OR']:.4f}  ({row['n_scored']} scored, {elapsed:.0f}s)")
        if published:
            print(f"  published U={published.get('utility_accuracy', float('nan')):.4f} "
                  f"MGS={published.get('compliance_utility_score', float('nan')):.4f}"
                  f"   (judge: gpt-4.1, gated_by_action="
                  f"{published.get('gated_by_action')})")
        print()

    if args.out and table:
        Path(args.out).write_text(json.dumps(table, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
