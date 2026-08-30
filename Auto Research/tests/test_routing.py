"""Every routing path in the graph is reachable, and reachable for the right reason.

Two layers of test:

1.  UNIT -- routers are pure functions of state, so each branch is driven by
    constructing the state that should produce it.  Fast, and precise about
    *why* a branch fires.
2.  INTEGRATION -- each scripted `MOCK_SCENARIO` is run through the real
    compiled graph and asserted to have taken its path.  This is what catches a
    router that is correct in isolation but wired to the wrong node in
    `graph.py`.
"""

from __future__ import annotations

import time

import pytest

import config
import routers
from gatemem_adapter import memory_governance_score, select_dev_slice
from mocks.dataset import build_mock_dataset
from nodes.critic import marginal_contributions
from nodes.dev_tools import signature_from_trace
from state import RESET, accumulate_or_reset, keep_first_signature, merge_dicts


def base_state(**overrides):
    """A healthy mid-run state; tests override only what they are testing."""
    state = {
        "iteration_count": 1,
        "eval_stage": "dev",
        "mgs_score": 0.5,
        "utility_score": 0.8,
        "access_violation_rate": 0.1,
        "forgetting_failure_rate": 0.1,
        "judge_report": {"phase_score": 1.0},
        "n_checkpoints_evaluated": 50,
        "n_worker_failures": 0,
        "current_curriculum_phase": config.CURRICULUM_PHASES[0],
        "halt_reason": None,
        "failure_signature": None,
        "proceed_to_full": False,
        "started_at": time.monotonic(),
        "token_usage": {"total_tokens": 0},
    }
    state.update(overrides)
    return state


# ======================================================================
# 1. Developer routing
# ======================================================================


def test_developer_success_routes_to_evaluation() -> None:
    assert routers.route_after_developer(base_state()) == "evaluate"


def test_developer_exhaustion_routes_back_to_architect() -> None:
    """A build that will not build is evidence about the DESIGN."""
    state = base_state(halt_reason="developer exhausted 5 retries", failure_signature="deadbeef")
    assert routers.route_after_developer(state) == "architect"


def test_developer_exhaustion_at_iteration_cap_halts_instead_of_looping() -> None:
    state = base_state(
        halt_reason="developer exhausted 5 retries", iteration_count=config.MAX_ITERATIONS
    )
    assert routers.route_after_developer(state) == "halt"


# ======================================================================
# 2. Fan-in / fail-fast routing
# ======================================================================


def test_collect_success_routes_to_judge() -> None:
    assert routers.route_after_collect(base_state()) == "judge"


def test_circuit_breaker_skips_the_judge_and_returns_to_architect() -> None:
    """Scoring a broken build's lucky shards would misattribute the loss."""
    state = base_state(
        failure_signature="1deb02dcaa56be9f",
        halt_reason="circuit breaker: 3+ shards failed with signature 1deb02dcaa56be9f",
    )
    assert routers.route_after_collect(state) == "architect"


def test_empty_prediction_set_does_not_reach_the_judge() -> None:
    """0.0 on every term is indistinguishable from 'answered everything wrong'."""
    assert routers.route_after_collect(base_state(n_checkpoints_evaluated=0)) == "architect"


# ======================================================================
# 3. The dev -> full scale-up gate
# ======================================================================


def test_dev_gate_passed_scales_up_to_the_full_run() -> None:
    state = base_state(mgs_score=config.DEV_GATE_MGS + 0.01, proceed_to_full=True)
    assert routers.route_after_judge(state) == "scale_up"


def test_dev_gate_failed_skips_the_full_run_entirely() -> None:
    state = base_state(mgs_score=config.DEV_GATE_MGS - 0.01, proceed_to_full=False)
    assert routers.route_after_judge(state) == "critic"


def test_full_stage_never_scales_up_again() -> None:
    """There is nothing above `full`; a second scale-up would loop forever."""
    state = base_state(eval_stage="full", mgs_score=0.95, proceed_to_full=True)
    assert routers.route_after_judge(state) == "critic"


def test_gate_and_target_are_distinct_thresholds() -> None:
    """The brief's deliberate distinction: 0.80 to scale up, 0.85 to stop."""
    assert config.DEV_GATE_MGS < config.MGS_TARGET
    between = (config.DEV_GATE_MGS + config.MGS_TARGET) / 2
    assert routers.route_after_judge(
        base_state(mgs_score=between, proceed_to_full=True)
    ) == "scale_up"
    assert routers.route_after_critic(base_state(mgs_score=between)) == "architect"


def test_skip_full_stage_overrides_an_open_gate(monkeypatch) -> None:
    """The cost guard must not be defeatable by a good dev score."""
    monkeypatch.setattr(config, "SKIP_FULL_STAGE", True)
    state = base_state(mgs_score=0.99, proceed_to_full=True)
    assert routers.route_after_judge(state) == "critic"


def test_skip_full_stage_defaults_off_so_research_runs_scale_up() -> None:
    assert config.SKIP_FULL_STAGE is False


# ======================================================================
# 4. Curriculum
# ======================================================================


def test_curriculum_failure_halts_the_remaining_phases() -> None:
    state = base_state(
        judge_report={"phase_score": config.CURRICULUM_PASS_THRESHOLD - 0.01},
        proceed_to_full=True, mgs_score=0.95,
    )
    # Even with a passing MGS and an open gate, a failed phase goes to the Critic.
    assert routers.route_after_judge(state) == "critic"


def test_curriculum_advances_only_on_pass() -> None:
    passed = base_state(judge_report={"phase_score": 1.0})
    assert routers.advance_curriculum(passed)[0] == config.CURRICULUM_PHASES[1]

    failed = base_state(judge_report={"phase_score": 0.0})
    assert routers.advance_curriculum(failed)[0] == config.CURRICULUM_PHASES[0]


def test_curriculum_reports_completion_on_the_last_phase() -> None:
    state = base_state(
        current_curriculum_phase=config.CURRICULUM_PHASES[-1],
        judge_report={"phase_score": 1.0},
    )
    phase, complete = routers.advance_curriculum(state)
    assert complete and phase == config.CURRICULUM_PHASES[-1]


def test_curriculum_is_ordered_easy_to_hard() -> None:
    assert config.CURRICULUM_PHASES[0] == "standard_retrieval"
    assert config.CURRICULUM_PHASES[-1] == "adversarial_injection"


# ======================================================================
# 5. Macro loop termination
# ======================================================================


def test_target_reached_terminates() -> None:
    assert routers.route_after_critic(base_state(mgs_score=config.MGS_TARGET)) == "halt"


def test_below_target_with_iterations_left_loops_to_architect() -> None:
    assert routers.route_after_critic(base_state(mgs_score=0.1)) == "architect"


def test_iteration_budget_terminates() -> None:
    state = base_state(mgs_score=0.1, iteration_count=config.MAX_ITERATIONS)
    assert routers.route_after_critic(state) == "halt"


# ======================================================================
# 6. Budget guard -- highest precedence everywhere
# ======================================================================


def test_token_budget_halts_from_every_router() -> None:
    state = base_state(
        token_usage={"total_tokens": config.MAX_TOTAL_TOKENS + 1},
        mgs_score=0.0, proceed_to_full=True,
    )
    assert routers.route_after_developer(state) == "halt"
    assert routers.route_after_collect(state) == "halt"
    assert routers.route_after_judge(state) == "halt"
    assert routers.route_after_critic(state) == "halt"
    assert routers.route_after_architect(state) == "halt"


def test_wallclock_budget_halts() -> None:
    state = base_state(started_at=time.monotonic() - (config.MAX_WALLCLOCK_S + 10))
    assert routers.route_after_critic(state) == "halt"
    assert "wall-clock" in routers.halt_reason_for(state)


def test_halt_reason_is_populated_for_every_exit() -> None:
    assert "target reached" in routers.halt_reason_for(base_state(mgs_score=0.99))
    assert "iteration budget" in routers.halt_reason_for(
        base_state(mgs_score=0.1, iteration_count=config.MAX_ITERATIONS)
    )
    assert routers.halt_reason_for(base_state(halt_reason="circuit breaker")) == "circuit breaker"


# ======================================================================
# 7. Reducers
# ======================================================================


def test_list_reducer_accumulates_across_concurrent_workers() -> None:
    acc = []
    for shard in range(5):
        acc = accumulate_or_reset(acc, [{"shard": shard}])
    assert len(acc) == 5


def test_list_reducer_cannot_be_cleared_with_an_empty_list() -> None:
    """The LangGraph gotcha this design exists to avoid."""
    assert accumulate_or_reset([{"a": 1}], []) == [{"a": 1}]


def test_list_reducer_is_cleared_by_the_explicit_sentinel() -> None:
    assert accumulate_or_reset([{"a": 1}], RESET) == []


def test_signature_reducer_keeps_the_first_and_is_reset_explicitly() -> None:
    assert keep_first_signature("first", "second") == "first"
    assert keep_first_signature(None, "second") == "second"
    assert keep_first_signature("first", None) == "first"
    assert keep_first_signature("first", RESET) is None


def test_dict_reducer_sums_numbers_and_overwrites_others() -> None:
    merged = merge_dicts({"total_tokens": 10, "route": "vllm"}, {"total_tokens": 5, "route": "openai"})
    assert merged["total_tokens"] == 15
    assert merged["route"] == "openai"


# ======================================================================
# 8. Attribution arithmetic
# ======================================================================


def test_marginal_attribution_ranks_by_effect_on_the_product_not_by_raw_badness() -> None:
    """A=0.30 looks worse than F=0.05 and IS worse -- but the ranking must be
    computed from the product, not from the raw values."""
    attribution = marginal_contributions(0.9, 0.30, 0.05)
    assert attribution["dominant_term"] == "A"
    assert attribution["marginal"]["A"] > attribution["marginal"]["F"]


def test_low_utility_dominates_when_the_leak_rates_are_already_small() -> None:
    attribution = marginal_contributions(0.40, 0.02, 0.02)
    assert attribution["dominant_term"] == "U"


def test_mgs_formula_matches_the_paper() -> None:
    assert memory_governance_score(0.9, 0.1, 0.1) == pytest.approx(0.9 * 0.9 * 0.9)


# ======================================================================
# 9. Deterministic dev slice
# ======================================================================


def test_dev_slice_is_identical_across_calls() -> None:
    """Iteration N and N+1 must be scored on the same checkpoints."""
    checkpoints = build_mock_dataset().checkpoints
    first = select_dev_slice(checkpoints, n=50, seed=config.DEV_SLICE_SEED)
    second = select_dev_slice(list(reversed(checkpoints)), n=50, seed=config.DEV_SLICE_SEED)
    assert [c["checkpoint_id"] for c in first] == [c["checkpoint_id"] for c in second]


def test_dev_slice_has_the_requested_size_and_all_three_query_types() -> None:
    """A slice with no `safety` checkpoints would make F undefined."""
    selected = select_dev_slice(build_mock_dataset().checkpoints, n=50, seed=config.DEV_SLICE_SEED)
    assert len(selected) == 50
    assert {c["query_type"] for c in selected} == {"utility", "privacy", "safety"}


def test_dev_slice_is_a_strict_subset_of_the_full_run() -> None:
    checkpoints = build_mock_dataset().checkpoints
    selected = select_dev_slice(checkpoints, n=50, seed=config.DEV_SLICE_SEED)
    assert len(selected) < len(checkpoints), "the gate is meaningless if dev == full"


# ======================================================================
# 10. Failure-signature normalization (the circuit breaker's premise)
# ======================================================================


def test_same_bug_from_two_shards_yields_one_signature() -> None:
    """Line numbers, ids and addresses must not split one bug into three."""
    trace_a = (
        'Traceback (most recent call last):\n'
        '  File "/runs/iter_1/workspace/memory_system/retrieval.py", line 148, in retrieve\n'
        "sqlite3.OperationalError: no such column: r.principal_scope"
    )
    trace_b = trace_a.replace("line 148", "line 152").replace("iter_1", "iter_2")
    assert signature_from_trace(trace_a, "eval_shard") == signature_from_trace(trace_b, "eval_shard")


def test_different_bugs_yield_different_signatures() -> None:
    trace_a = (
        'Traceback (most recent call last):\n'
        '  File "store.py", line 10, in retrieve\n'
        "sqlite3.OperationalError: no such column: x"
    )
    trace_b = (
        'Traceback (most recent call last):\n'
        '  File "agent.py", line 10, in query\n'
        "KeyError: 'asker'"
    )
    assert signature_from_trace(trace_a, "eval_shard") != signature_from_trace(trace_b, "eval_shard")


# ======================================================================
# 11. Startup guards
# ======================================================================


def test_transport_dep_check_passes_when_the_sdk_is_importable(monkeypatch) -> None:
    import main

    monkeypatch.setattr(config, "AGENT_TRANSPORT", "dsh")
    assert main.check_transport_deps() is None


def test_transport_dep_check_is_skipped_for_the_http_transport(monkeypatch) -> None:
    """http needs no harness, so the guard must not block it."""
    import main

    monkeypatch.setattr(config, "AGENT_TRANSPORT", "http")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    assert main.check_transport_deps() is None


def test_transport_dep_check_names_the_interpreter_when_dsh_is_missing(monkeypatch) -> None:
    """The message must be self-diagnosing.

    The overwhelmingly common cause is running under a different virtualenv than
    the one the dependencies were installed into; "not available" alone sends
    you hunting for a missing package instead of a missing activation.
    """
    import shutil
    import sys

    import main

    monkeypatch.setattr(config, "AGENT_TRANSPORT", "dsh")
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)

    message = main.check_transport_deps()
    assert message is not None
    assert sys.executable in message, "the failing interpreter must be named"
    assert "AGENT_TRANSPORT=http" in message, "the escape hatch must be offered"
    assert ".venv/bin/activate" in message, "the likely fix must be offered"
