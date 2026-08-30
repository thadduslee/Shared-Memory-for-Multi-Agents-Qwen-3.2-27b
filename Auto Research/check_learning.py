#!/usr/bin/env python3
"""Audit whether a multi-iteration run actually LEARNED, or merely repeated.

A rising MGS is not proof of learning -- it can be sampling noise on a small
dev slice, or a lucky Developer retry on an unchanged design. This script
checks the four channels that are supposed to carry knowledge from iteration
N to N+1, and reports each one separately so a break is attributable:

  1. WORKSPACE LINEAGE   iter N+1's code is seeded from iter N's, not reset to
                         the template. (nodes/developer.py: prepare_workspace)
  2. CRITIQUE -> DESIGN  the Critic names a dominant failing term in
                         attribution.json; the next design.json should target
                         THAT term in `targets_metric`. This is the sharpest
                         single test: it is the only channel where a specific
                         claim made in iteration N must reappear as a specific
                         decision in N+1.
  3. SCHEMA EVOLUTION    migration.sql changes between iterations, so the
                         Architect is proposing something new rather than
                         re-emitting its baseline DDL.
  4. METRIC TRAJECTORY   U / A / F / MGS per iteration, plus the failure
                         signature, which should not repeat verbatim.

Usage:  python check_learning.py [RUNS_DIR]
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

GREEN, RED, YELLOW, DIM, BOLD, OFF = (
    "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"
)


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _source_files(workspace: Path) -> dict[str, str]:
    """Content of every source file, keyed by workspace-relative path."""
    if not workspace.is_dir():
        return {}
    skip = {".sessions", "__pycache__", ".cordis", "migrations"}
    scaffolds = {"_smoke_runner.py", "_eval_runner.py"}
    out = {}
    for p in sorted(workspace.rglob("*")):
        if not p.is_file() or p.suffix not in {".py", ".sql", ".toml"}:
            continue
        if any(part in skip for part in p.parts) or p.name in scaffolds:
            continue
        out[str(p.relative_to(workspace))] = p.read_text(errors="replace")
    return out


def main(argv: list[str]) -> int:
    runs = Path(argv[1] if len(argv) > 1 else "runs").resolve()
    if not runs.is_dir():
        print(f"{RED}no such runs dir: {runs}{OFF}")
        return 2

    iters = sorted(
        (int(p.name.split("_")[1]), p)
        for p in runs.glob("iter_*") if p.name.split("_")[-1].isdigit()
    )
    if not iters:
        print(f"{RED}no iter_* directories under {runs}{OFF}")
        return 2

    print(f"\n{BOLD}Learning audit: {runs}{OFF}")
    print(f"{DIM}{len(iters)} iteration(s) found{OFF}\n")

    # ---- metric trajectory ------------------------------------------------
    print(f"{BOLD}METRIC TRAJECTORY{OFF}  {DIM}(MGS = U * (1-A) * (1-F)){OFF}")
    print(f"  {'iter':<6}{'U':>8}{'A':>8}{'F':>8}{'MGS':>9}{'  delta':>9}   phase")
    previous_mgs = None
    seen_mgs: dict[int, float] = {}
    scripted_iters: list[int] = []
    for n, d in iters:
        report = _load(d / "judge_report.json")
        if not report:
            print(f"  {n:<6}{DIM}      -- no judge_report.json "
                  f"(iteration did not reach the Judge){OFF}")
            continue
        u, a, f = report.get("U", 0.0), report.get("A", 1.0), report.get("F", 1.0)
        mgs = report.get("MGS", 0.0)
        seen_mgs[n] = mgs
        if report.get("scripted"):
            scripted_iters.append(n)
        if previous_mgs is None:
            delta = "      --"
        else:
            diff = mgs - previous_mgs
            colour = GREEN if diff > 0 else (RED if diff < 0 else YELLOW)
            delta = f"{colour}{diff:+8.4f}{OFF}"
        print(f"  {n:<6}{u:>8.3f}{a:>8.3f}{f:>8.3f}{mgs:>9.4f}{delta}   "
              f"{report.get('curriculum_phase', '?')}")
        previous_mgs = mgs
    print()

    # PROVENANCE GATE. Under MOCK_MODE the Judge overwrites the headline
    # U/A/F/MGS with fixture values from the scenario and files the real
    # aggregate under `measured` (nodes/judge.py). A scripted trajectory can
    # climb beautifully while the system under test does nothing, so this must
    # be checked BEFORE any of the trajectory is believed -- it is the one
    # failure mode where every other channel can look green and the numbers
    # still mean nothing.
    if scripted_iters:
        print(f"{RED}{BOLD}!! SCRIPTED METRICS in iteration(s) "
              f"{', '.join(map(str, scripted_iters))}{OFF}")
        print(f"{RED}   These U/A/F/MGS values are scenario fixtures, not measurements."
              f"{OFF}")
        print(f"{RED}   The trajectory above is FICTION. Re-run with --real.{OFF}")
        for n, d in iters:
            report = _load(d / "judge_report.json") or {}
            measured = report.get("measured")
            if measured:
                print(f"{DIM}   iter {n} actually measured: "
                      f"U={measured.get('U', 0):.3f} A={measured.get('A', 1):.3f} "
                      f"F={measured.get('F', 1):.3f} MGS={measured.get('MGS', 0):.4f}{OFF}")
        print()

    # ---- per-transition channels -----------------------------------------
    checks: list[tuple[str, bool | None]] = []
    for (prev_n, prev_d), (n, d) in itertools.pairwise(iters):
        print(f"{BOLD}ITERATION {prev_n} -> {n}{OFF}")

        # 1. workspace lineage
        before, after = _source_files(prev_d / "workspace"), _source_files(d / "workspace")
        if not before or not after:
            print(f"  {YELLOW}? lineage      one workspace missing; cannot compare{OFF}")
            checks.append(("lineage", None))
        else:
            changed = sorted(k for k in after if before.get(k) != after.get(k))
            added = sorted(set(after) - set(before))
            carried = sorted(set(before) & set(after))
            if not carried:
                print(f"  {RED}FAIL lineage   iter {n} shares NO file with iter {prev_n} "
                      f"-- it was reset, not seeded{OFF}")
                checks.append(("lineage", False))
            elif not changed:
                print(f"  {RED}FAIL lineage   {len(carried)} files carried but NONE changed "
                      f"-- the Developer wrote nothing{OFF}")
                checks.append(("lineage", False))
            else:
                print(f"  {GREEN}OK   lineage{OFF}   {len(carried)} file(s) carried forward, "
                      f"{len(changed)} modified, {len(added)} new")
                for k in changed[:5]:
                    delta = len(after[k]) - len(before.get(k, ""))
                    print(f"                 {DIM}{k}  {delta:+d} chars{OFF}")
                checks.append(("lineage", True))

        # 2. critique -> design  (the sharpest test)
        attribution = _load(prev_d / "attribution.json")
        design = _load(d / "design.json")
        if not attribution:
            print(f"  {YELLOW}? critique     iter {prev_n} has no attribution.json "
                  f"(Critic never ran){OFF}")
            checks.append(("critique->design", None))
        elif not design:
            print(f"  {YELLOW}? critique     iter {n} has no design.json{OFF}")
            checks.append(("critique->design", None))
        else:
            dominant = attribution.get("dominant_term")
            targeted = design.get("targets_metric")
            gain = attribution.get("dominant_gain", 0.0)
            if dominant and targeted and str(targeted).strip().upper().startswith(
                str(dominant).strip().upper()[:1]
            ):
                print(f"  {GREEN}OK   critique{OFF}   iter {prev_n} Critic: dominant={dominant} "
                      f"(gain {gain:+.4f})  ->  iter {n} Architect targets {targeted}")
                checks.append(("critique->design", True))
            else:
                print(f"  {RED}FAIL critique  iter {prev_n} Critic said the dominant failing "
                      f"term was {dominant} (gain {gain:+.4f}),{OFF}")
                print(f"                 {RED}but iter {n} targets {targeted} -- the critique "
                      f"did not steer the design{OFF}")
                checks.append(("critique->design", False))
            component = attribution.get("component")
            if component:
                print(f"                 {DIM}named component: {str(component)[:78]}{OFF}")

        # 3. schema evolution
        prev_sql = (prev_d / "migration.sql")
        cur_sql = (d / "migration.sql")
        if prev_sql.is_file() and cur_sql.is_file():
            if prev_sql.read_text().strip() == cur_sql.read_text().strip():
                print(f"  {RED}FAIL schema    migration.sql is byte-identical to iter "
                      f"{prev_n} -- the Architect re-emitted its baseline{OFF}")
                checks.append(("schema", False))
            else:
                print(f"  {GREEN}OK   schema{OFF}    migration.sql differs from iter {prev_n} "
                      f"({len(prev_sql.read_text())} -> {len(cur_sql.read_text())} chars)")
                checks.append(("schema", True))
        else:
            print(f"  {YELLOW}? schema       migration.sql missing on one side{OFF}")
            checks.append(("schema", None))
        print()

    # ---- verdict ----------------------------------------------------------
    if len(iters) < 2:
        print(f"{YELLOW}Only one iteration: nothing to compare. Raise --max-iterations "
              f"and keep MGS_TARGET above the reachable score.{OFF}\n")
        return 1

    if scripted_iters:
        checks.append(("measured-metrics", False))
    failed = [name for name, ok in checks if ok is False]
    unknown = [name for name, ok in checks if ok is None]
    if failed:
        print(f"{RED}{BOLD}NOT LEARNING{OFF} -- broken channel(s): "
              f"{', '.join(sorted(set(failed)))}")
        rc = 1
    elif unknown:
        print(f"{YELLOW}{BOLD}INCONCLUSIVE{OFF} -- unverified channel(s): "
              f"{', '.join(sorted(set(unknown)))}")
        rc = 1
    else:
        print(f"{GREEN}{BOLD}LEARNING{OFF} -- every channel carried knowledge forward.")
        rc = 0
    if len(seen_mgs) >= 2:
        first, last = seen_mgs[min(seen_mgs)], seen_mgs[max(seen_mgs)]
        arrow = GREEN + "improved" if last > first else RED + "did NOT improve"
        print(f"MGS {first:.4f} -> {last:.4f}: {arrow}{OFF}")
    print()
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
