#!/usr/bin/env python3
"""Re-judge existing GateMem predictions with a swappable judge model.

Agent inference and judging are independent stages. Once ``predictions.jsonl``
exists, changing the judge does NOT require re-running the agents or the GPU
server: this script reads the predictions you already have, re-grades them
with a different judge, and rewrites the metrics.

Single run:
  python bench/scripts/rejudge.py \
    --run_dir outputs/medical_qwen32b_rag_naive \
    --judge gpt41

All seven baselines at once:
  python bench/scripts/rejudge.py \
    --out_root outputs --domain medical --all_baselines \
    --judge gpt41 --concurrency 8

Offline plumbing check (no network, no cost):
  python bench/scripts/rejudge.py --run_dir outputs/... --judge mock

By default results are written to a NEW directory
``<run_dir>__judge-<judge_key>`` so your original judgements survive.
Pass --in_place to overwrite the run directory instead.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from bench.eval.runner import dump_jsonl, load_jsonl  # noqa: E402
from bench.eval.scorer import score_predictions  # noqa: E402
from bench.eval.validator import validate_dataset  # noqa: E402
from bench.eval.verifiers import AuthError, build_verifier, list_judges  # noqa: E402

BASELINE_AGENTS = [
    "a_mem",
    "remem_i",
    "remem_s",
    "mem0",
    "long_context",
    "rag_naive",
    "rag_policy",
]
DOMAINS = ["medical", "office", "education", "household"]


def _rate(numer: int, denom: int) -> float:
    return float(numer) / float(denom) if denom > 0 else 0.0


def summarize_judge_rows(
    rows: List[Dict[str, Any]],
    *,
    gate_by_action: bool,
    judge_desc: Dict[str, Any],
    latency_total: float,
    tokens: Dict[str, int],
) -> Dict[str, Any]:
    """Reproduce the judge summary block emitted by bench/eval/judge.py."""
    action_rows = [r for r in rows if isinstance(r.get("judge"), dict)]
    util = [r for r in rows if r.get("query_type") == "utility"]
    priv = [r for r in rows if r.get("query_type") == "privacy"]
    safe = [r for r in rows if r.get("query_type") == "safety"]

    us = [r for r in util if isinstance(r.get("judge"), dict) and r["judge"].get("utility_ok") is not None]
    ps = [r for r in priv if isinstance(r.get("judge"), dict) and r["judge"].get("privacy_leak") is not None]
    ss = [r for r in safe if isinstance(r.get("judge"), dict) and r["judge"].get("deletion_leak") is not None]

    return {
        "n_judged": len(rows),
        "n_action_scored": len(action_rows),
        "n_utility": len(util),
        "n_utility_scored": len(us),
        "n_privacy": len(priv),
        "n_privacy_scored": len(ps),
        "n_safety": len(safe),
        "n_safety_scored": len(ss),
        "judge_action_ok_rate": _rate(
            sum(1 for r in action_rows if r["judge"].get("action_ok") is True), len(action_rows)
        ),
        "judge_utility_accuracy": _rate(
            sum(1 for r in us if r["judge"].get("utility_ok") is True), len(us)
        ),
        "judge_effective_utility_accuracy": _rate(
            sum(
                1
                for r in us
                if r["judge"].get("utility_ok") is True and r["judge"].get("action_ok") is True
            ),
            len(us),
        ),
        "judge_privacy_leakage_rate": _rate(
            sum(1 for r in ps if r["judge"].get("privacy_leak") is True), len(ps)
        ),
        "judge_deletion_leakage_rate": _rate(
            sum(1 for r in ss if r["judge"].get("deletion_leak") is True), len(ss)
        ),
        "judge_parse_failure_rate": _rate(
            sum(1 for r in action_rows if r["judge"].get("parse_ok") is False), len(action_rows)
        ),
        "judge_llm_action_agreement_rate": _rate(
            sum(
                1
                for r in action_rows
                if r["judge"].get("action_ok_llm") is not None
                and r["judge"].get("action_ok_llm") == r["judge"].get("action_ok")
            ),
            sum(1 for r in action_rows if r["judge"].get("action_ok_llm") is not None),
        ),
        "gated_by_action": bool(gate_by_action),
        "llm": {
            "latency_s_total": latency_total,
            "tokens": {
                **tokens,
                "avg_total_tokens_per_case": (tokens["total_tokens"] / len(rows)) if rows else 0,
            },
            "provider": "verifier",
            "model": judge_desc.get("model"),
            "base_url": judge_desc.get("base_url"),
        },
        "verifier": judge_desc,
    }


def promote_judge_metrics(
    summary: Dict[str, Any], judge_summary: Dict[str, Any]
) -> Dict[str, Any]:
    """Promote judge metrics to top-level fields (same math as run_eval.py)."""
    action_acc = float(judge_summary.get("judge_action_ok_rate") or 0.0)
    utility_acc = float(judge_summary.get("judge_effective_utility_accuracy") or 0.0)
    privacy_leak = float(judge_summary.get("judge_privacy_leakage_rate") or 0.0)
    deletion_leak = float(judge_summary.get("judge_deletion_leakage_rate") or 0.0)

    promoted = dict(summary)
    promoted.update(
        {
            "n_checkpoints": int(judge_summary.get("n_judged") or summary.get("n_checkpoints") or 0),
            "n_utility": int(judge_summary.get("n_utility") or summary.get("n_utility") or 0),
            "action_accuracy": action_acc,
            "utility_accuracy": utility_acc,
            "privacy_leakage_rate": privacy_leak,
            "deletion_leakage_rate": deletion_leak,
            "privacy_answer_leakage_rate": privacy_leak,
            "deletion_answer_leakage_rate": deletion_leak,
            "over_refusal_rate": float(summary.get("over_refusal_rate") or 0.0),
            # MGS = U * (1 - A) * (1 - F)
            "compliance_utility_score": utility_acc
            * (1.0 - privacy_leak)
            * (1.0 - deletion_leak),
            "llm_judge": judge_summary,
        }
    )
    return promoted


def rejudge_run(
    *,
    run_dir: Path,
    data_dir: Path,
    out_dir: Path,
    judge_key: str,
    judge_kwargs: Dict[str, Any],
    concurrency: int,
    resume: bool,
    gate_by_action: bool,
    limit: Optional[int],
) -> Dict[str, Any]:
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"No predictions.jsonl in {run_dir}")

    episodes = load_jsonl(str(data_dir / "episodes.jsonl"))
    checkpoints = load_jsonl(str(data_dir / "checkpoints.jsonl"))
    errors, warnings = validate_dataset(episodes=episodes, checkpoints=checkpoints, strict=True)
    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        for e in errors:
            print(f"[ERROR] {e}")
        raise SystemExit("Dataset validation failed.")

    predictions = load_jsonl(str(pred_path))
    ckpt_by_id = {str(c.get("checkpoint_id")): c for c in checkpoints}

    out_dir.mkdir(parents=True, exist_ok=True)
    judge_path = out_dir / "judge_scores.jsonl"

    done: set[str] = set()
    existing: List[Dict[str, Any]] = []
    if resume and judge_path.exists():
        existing = load_jsonl(str(judge_path), ignore_errors=True)
        done = {str(r.get("checkpoint_id")) for r in existing if r.get("checkpoint_id")}
        print(f"  resume: {len(done)} checkpoints already judged")
    elif judge_path.exists():
        backup = judge_path.with_suffix(f".jsonl.bak.{int(time.time())}")
        shutil.copy2(judge_path, backup)
        judge_path.unlink()
        print(f"  backed up previous judge_scores.jsonl -> {backup.name}")

    todo = [
        p
        for p in predictions
        if str(p.get("checkpoint_id")) in ckpt_by_id
        and str(p.get("checkpoint_id")) not in done
    ]
    if limit:
        todo = todo[:limit]

    verifier = build_verifier(judge_key, gate_by_action=gate_by_action, **judge_kwargs)
    print(f"  judge: {verifier.model} @ {verifier.base_url}")
    verifier.preflight()
    print(f"  {len(todo)} checkpoints to judge (concurrency={concurrency})")

    rows: List[Dict[str, Any]] = list(existing)
    tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    latency_total = 0.0
    write_lock = threading.Lock()
    acc_lock = threading.Lock()
    n_done = 0
    t_start = time.perf_counter()

    def _work(pred: Dict[str, Any]) -> Dict[str, Any]:
        ckpt = ckpt_by_id[str(pred.get("checkpoint_id"))]
        return verifier.verify(ckpt, pred)

    def _record(row: Dict[str, Any]) -> None:
        nonlocal n_done, latency_total
        with write_lock:
            with open(judge_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.append(row)
        with acc_lock:
            n_done += 1
            usage = (row.get("llm") or {}).get("usage") or {}
            for k in tokens:
                tokens[k] += int(usage.get(k) or 0)
            latency_total += float((row.get("llm") or {}).get("latency_s") or 0.0)
            if n_done % 25 == 0 or n_done == len(todo):
                el = time.perf_counter() - t_start
                print(f"    judged {n_done}/{len(todo)}  ({el:.0f}s)", flush=True)

    if concurrency <= 1:
        for pred in todo:
            _record(_work(pred))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(_work, p): p for p in todo}
            try:
                for fut in as_completed(futures):
                    _record(fut.result())
            except BaseException:
                # Cancel queued work so a bad key does not fire 579 doomed calls.
                for f in futures:
                    f.cancel()
                raise

    # Rule-based scores stay part of summary.json, exactly as run_eval.py does.
    scores, rule_summary = score_predictions(
        episodes=episodes,
        checkpoints=checkpoints,
        predictions=predictions,
        gate_by_action=gate_by_action,
    )
    dump_jsonl(str(out_dir / "scores.jsonl"), scores)

    all_rows = load_jsonl(str(judge_path), ignore_errors=True)
    judge_summary = summarize_judge_rows(
        all_rows,
        gate_by_action=gate_by_action,
        judge_desc={**verifier.describe(), "judge_key": judge_key},
        latency_total=latency_total,
        tokens=tokens,
    )

    summary = dict(rule_summary)
    summary["rule_based"] = dict(rule_summary)
    summary = promote_judge_metrics(summary, judge_summary)
    summary["judge_key"] = judge_key
    summary["rejudged_from"] = str(run_dir)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not (out_dir / "predictions.jsonl").exists():
        shutil.copy2(pred_path, out_dir / "predictions.jsonl")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-judge GateMem predictions with a swappable judge model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available judges:\n" + list_judges(),
    )
    ap.add_argument("--run_dir", help="A single run directory containing predictions.jsonl")
    ap.add_argument("--out_root", default="outputs", help="Root of run directories")
    ap.add_argument("--domain", choices=DOMAINS, help="Domain for --all_baselines")
    ap.add_argument("--model_key", default="qwen32b", help="Model key used in run names")
    ap.add_argument("--all_baselines", action="store_true", help="Loop all 7 baselines")
    ap.add_argument("--data_dir", help="Override dataset dir (else inferred from domain)")

    ap.add_argument("--judge", default="gpt41", help="Judge profile key")
    ap.add_argument("--judge_model", default=None, help="Override the model slug")
    ap.add_argument("--judge_base_url", default=None, help="Override the endpoint")
    ap.add_argument("--judge_api_key_env", default=None, help="Override the API-key env var")
    ap.add_argument("--judge_temperature", type=float, default=None)
    ap.add_argument("--judge_max_output_tokens", type=int, default=None)
    ap.add_argument("--judge_timeout_s", type=float, default=None)
    ap.add_argument("--no_json_mode", action="store_true", help="Disable response_format")

    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--resume", action="store_true", help="Skip already-judged checkpoints")
    ap.add_argument("--gate_by_action", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Judge only the first N (pilot)")
    ap.add_argument("--in_place", action="store_true", help="Overwrite the run dir")
    ap.add_argument("--suffix", default=None, help="Output suffix (default __judge-<key>)")
    ap.add_argument("--list_judges", action="store_true")
    ap.add_argument(
        "--export_prompt",
        metavar="PATH",
        default=None,
        help="Write JUDGE_PROMPT_GATEMEM to PATH and exit. The result is usable "
        "as run_eval.py --judge_prompt_path (brace escaping is preserved).",
    )
    args = ap.parse_args()

    if args.list_judges:
        print(list_judges())
        return

    if args.export_prompt:
        from bench.eval.verifiers import JUDGE_PROMPT_GATEMEM

        out = Path(args.export_prompt)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(JUDGE_PROMPT_GATEMEM, encoding="utf-8")
        print(f"wrote prompt -> {out}")
        return

    repo_root = Path(REPO_ROOT)
    judge_kwargs: Dict[str, Any] = {
        "model": args.judge_model,
        "base_url": args.judge_base_url,
        "api_key_env": args.judge_api_key_env,
        "temperature": args.judge_temperature,
        "max_output_tokens": args.judge_max_output_tokens,
        "timeout_s": args.judge_timeout_s,
        "json_mode": False if args.no_json_mode else None,
    }

    run_dirs: List[Path] = []
    if args.run_dir:
        run_dirs = [Path(args.run_dir)]
    elif args.all_baselines:
        if not args.domain:
            raise SystemExit("--all_baselines requires --domain")
        for agent in BASELINE_AGENTS:
            d = Path(args.out_root) / f"{args.domain}_{args.model_key}_{agent}"
            if d.exists():
                run_dirs.append(d)
            else:
                print(f"[skip] missing run dir: {d}")
    else:
        raise SystemExit("Provide --run_dir, or --all_baselines with --domain.")

    if not run_dirs:
        raise SystemExit("No run directories found.")

    suffix = args.suffix or f"__judge-{args.judge}"
    results: Dict[str, Dict[str, Any]] = {}

    for run_dir in run_dirs:
        domain = args.domain or run_dir.name.split("_")[0]
        data_dir = Path(args.data_dir) if args.data_dir else repo_root / "bench" / "data" / domain
        if not (data_dir / "checkpoints.jsonl").exists():
            raise SystemExit(f"Bad --data_dir: {data_dir}")

        out_dir = run_dir if args.in_place else run_dir.parent / (run_dir.name + suffix)
        print(f"\n=== {run_dir.name}  ->  {out_dir.name} ===")
        summary = rejudge_run(
            run_dir=run_dir,
            data_dir=data_dir,
            out_dir=out_dir,
            judge_key=args.judge,
            judge_kwargs=judge_kwargs,
            concurrency=args.concurrency,
            resume=args.resume,
            gate_by_action=args.gate_by_action,
            limit=args.limit,
        )
        results[run_dir.name] = summary
        print(
            f"  U={summary['utility_accuracy']:.4f}  "
            f"A={summary['privacy_leakage_rate']:.4f}  "
            f"F={summary['deletion_leakage_rate']:.4f}  "
            f"MGS={summary['compliance_utility_score']:.4f}  "
            f"parse_fail={summary['llm_judge']['judge_parse_failure_rate']:.4f}"
        )

    print("\n" + "=" * 96)
    print(f"{'run':<44}{'U':>10}{'A':>10}{'F':>10}{'MGS':>10}{'OR':>10}")
    print("-" * 96)
    for name, s in results.items():
        print(
            f"{name:<44}{s['utility_accuracy']:>10.4f}{s['privacy_leakage_rate']:>10.4f}"
            f"{s['deletion_leakage_rate']:>10.4f}{s['compliance_utility_score']:>10.4f}"
            f"{float(s.get('over_refusal_rate') or 0.0):>10.4f}"
        )


if __name__ == "__main__":
    try:
        main()
    except AuthError as exc:
        print(f"\nAUTHENTICATION FAILED\n{exc}\n", file=sys.stderr)
        raise SystemExit(2)