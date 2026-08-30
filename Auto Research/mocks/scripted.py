"""Deterministic scripted outcomes -- the reviewer's control panel.

Every routing path in the graph is reachable offline by setting
`MOCK_SCENARIO`.  Each scenario is a pure data description of what the mocked
world does on each (iteration, stage), so a reviewer can force a path without
editing code and a test can assert on it without mocking mocks.

    MOCK_SCENARIO=happy_path            dev gate passes -> full run -> hits MGS_TARGET
    MOCK_SCENARIO=immediate_success     terminates on iteration 1
    MOCK_SCENARIO=dev_gate_fail         dev MGS < 0.80 -> full run SKIPPED -> Critic
    MOCK_SCENARIO=dev_retry_exhaustion  Developer burns MAX_DEV_RETRIES -> Architect
    MOCK_SCENARIO=failfast_signature    every shard dies identically -> batch aborted
    MOCK_SCENARIO=curriculum_fail       the active phase fails -> later phases halted
    MOCK_SCENARIO=budget_exhausted      budget guard trips -> END with halt_reason
    MOCK_SCENARIO=max_iterations        never reaches target -> stops at MAX_ITERATIONS

Run `python main.py --list-scenarios` to print this table at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import config


@dataclass(frozen=True)
class RoundSpec:
    """The scores a scripted round hands back to the Judge."""

    utility: float
    access: float          # A -- violation rate, lower is better
    forgetting: float      # F -- failure rate, lower is better
    dev_pass_rate: float = 1.0

    @property
    def mgs(self) -> float:
        return self.utility * (1.0 - self.access) * (1.0 - self.forgetting)


@dataclass
class Scenario:
    """A complete scripted world."""

    name: str
    description: str
    default: RoundSpec
    # (iteration, stage) -> scores.  Missing keys fall back to `default`.
    rounds: dict[tuple[int, str], RoundSpec] = field(default_factory=dict)
    # Iterations on which the Developer's tools never go green, forcing the
    # Developer loop to exhaust MAX_DEV_RETRIES.
    developer_fail_iterations: frozenset[int] = frozenset()
    # Iteration -> the identical stack trace every shard raises, which is what
    # the circuit breaker is supposed to notice.
    worker_failure: dict[int, str] = field(default_factory=dict)
    # Iterations on which the current curriculum phase scores below threshold.
    curriculum_fail_iterations: frozenset[int] = frozenset()
    # Pretend this many tokens have been spent per evaluation shard, to make the
    # budget guard reachable without an eight-hour run.
    tokens_per_shard: int = 1600

    def scores(self, iteration: int, stage: str) -> RoundSpec:
        return self.rounds.get((iteration, stage), self.default)

    def developer_fails(self, iteration: int) -> bool:
        return iteration in self.developer_fail_iterations

    def worker_stack_trace(self, iteration: int) -> str | None:
        return self.worker_failure.get(iteration)

    def curriculum_fails(self, iteration: int) -> bool:
        return iteration in self.curriculum_fail_iterations


# --------------------------------------------------------------------------
# A representative traceback for the fail-fast scenario.  All shards raise this
# same text; `gatemem_adapter.normalize_traceback` must reduce the per-shard
# variation (line numbers, ids) away so the signatures collide.
# --------------------------------------------------------------------------
_IDENTICAL_TRACE = (
    'Traceback (most recent call last):\n'
    '  File "/runs/iter_1/workspace/memory_system/retrieval.py", line 148, in retrieve\n'
    '    rows = self._conn.execute(sql, params).fetchall()\n'
    "sqlite3.OperationalError: no such column: r.principal_scope"
)


SCENARIOS: dict[str, Scenario] = {
    # ---- 1. The full happy path: gate opens on iteration 1, target on 2. ----
    "happy_path": Scenario(
        name="happy_path",
        description="dev gate passes, full run executes, MGS_TARGET reached on iteration 2",
        default=RoundSpec(utility=0.90, access=0.03, forgetting=0.02),
        rounds={
            # Iteration 1: clears DEV_GATE_MGS (0.80) so the full run happens,
            # but the full 579 is harder than the 50 and lands below MGS_TARGET.
            (1, "dev"): RoundSpec(utility=0.89, access=0.05, forgetting=0.04, dev_pass_rate=0.94),
            (1, "full"): RoundSpec(utility=0.86, access=0.09, forgetting=0.07, dev_pass_rate=0.94),
            # Iteration 2: the Critic's schema fix lands; both stages clear 0.85.
            (2, "dev"): RoundSpec(utility=0.94, access=0.02, forgetting=0.01, dev_pass_rate=1.0),
            (2, "full"): RoundSpec(utility=0.93, access=0.03, forgetting=0.02, dev_pass_rate=1.0),
        },
    ),
    # ---- 2. Terminates on the first iteration. ----
    "immediate_success": Scenario(
        name="immediate_success",
        description="MGS_TARGET reached on the first full run; loop exits after one iteration",
        default=RoundSpec(utility=0.95, access=0.02, forgetting=0.01),
    ),
    # ---- 3. Dev gate closed: the 579-checkpoint run must be SKIPPED. ----
    "dev_gate_fail": Scenario(
        name="dev_gate_fail",
        description="dev-slice MGS stays under 0.80, so the full run is never entered",
        default=RoundSpec(utility=0.62, access=0.22, forgetting=0.18, dev_pass_rate=0.71),
    ),
    # ---- 4. Developer cannot get to green. ----
    "dev_retry_exhaustion": Scenario(
        name="dev_retry_exhaustion",
        description="Developer exhausts MAX_DEV_RETRIES on iteration 1 and routes back to Architect",
        default=RoundSpec(utility=0.80, access=0.10, forgetting=0.10),
        developer_fail_iterations=frozenset({1}),
    ),
    # ---- 5. Circuit breaker: identical failures across shards. ----
    "failfast_signature": Scenario(
        name="failfast_signature",
        description="every evaluation shard raises the same error; the batch is aborted early",
        default=RoundSpec(utility=0.88, access=0.05, forgetting=0.04),
        worker_failure={1: _IDENTICAL_TRACE},
    ),
    # ---- 6. Curriculum phase failure halts the remaining phases. ----
    "curriculum_fail": Scenario(
        name="curriculum_fail",
        description="the iteration-1 curriculum phase fails, halting later phases and feeding back to the Architect",
        default=RoundSpec(utility=0.87, access=0.06, forgetting=0.05),
        curriculum_fail_iterations=frozenset({1}),
    ),
    # ---- 7. Budget guard. ----
    "budget_exhausted": Scenario(
        name="budget_exhausted",
        description="token budget is consumed mid-run; the guard routes to END with a halt_reason",
        default=RoundSpec(utility=0.70, access=0.15, forgetting=0.10),
        tokens_per_shard=5_000_000,
    ),
    # ---- 8. Never converges. ----
    "max_iterations": Scenario(
        name="max_iterations",
        description="MGS improves but never reaches MGS_TARGET; the loop stops at MAX_ITERATIONS",
        default=RoundSpec(utility=0.83, access=0.08, forgetting=0.06),
    ),
}


def active_scenario() -> Scenario:
    """The scenario named by `MOCK_SCENARIO`, defaulting to happy_path."""
    return SCENARIOS.get(config.MOCK_SCENARIO, SCENARIOS["happy_path"])


def scenario_table() -> list[dict[str, Any]]:
    """Rows for `main.py --list-scenarios`."""
    return [
        {"name": s.name, "default_mgs": round(s.default.mgs, 4), "description": s.description}
        for s in SCENARIOS.values()
    ]
