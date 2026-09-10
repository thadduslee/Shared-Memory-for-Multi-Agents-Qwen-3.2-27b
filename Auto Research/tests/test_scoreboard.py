"""The loop's memory of its own scores, and everything derived from it.

WHAT THESE TESTS ARE PINNING. `scoreboard.py` exists because run-8cf58d33b311
scored 0.3172, 0.2222, 0.1830, 0.1190 across five iterations and the loop could
not tell. Each function here answers one of the four questions that failure
turned on:

  * `best_of`          -- what WAS the best? (the halt reason claimed 0.1190)
  * `champion_iteration` -- whose code should the next iteration inherit?
  * `verdict_for`      -- did the last change help?
  * `term_deltas`      -- which term moved, as opposed to which is worth most?

The run's real numbers are used as the fixture throughout, so a regression in
any of these is a regression against a failure that actually happened rather
than against an invented one.
"""

from __future__ import annotations

import config
import scoreboard


def row(iteration, u, a, f, *, stage="dev", workspace=""):
    """One score row, MGS computed rather than passed, as the Judge computes it."""
    return scoreboard.score_row(
        iteration=iteration, stage=stage, phase="standard_retrieval",
        utility=u, access=a, forgetting=f, mgs=u * (1 - a) * (1 - f),
        n_checkpoints=50, workspace=workspace,
    )


#: run-8cf58d33b311, exactly. Iteration 2 is absent because its build failed and
#: it never reached the Judge -- which is itself load-bearing: a failed build
#: must not be eligible to become anyone's parent.
REAL_RUN = [
    row(1, 0.4444444444444444, 0.17647058823529413, 0.13333333333333333),
    row(3, 0.2222222222222222, 0.0, 0.0),
    row(4, 0.2222222222222222, 0.11764705882352941, 0.06666666666666667),
    row(5, 0.16666666666666666, 0.17647058823529413, 0.13333333333333333),
]


# ======================================================================
# rows
# ======================================================================


def test_a_row_is_json_safe_and_rounded() -> None:
    """It goes into graph state and therefore into every checkpointer snapshot."""
    import json

    entry = row(1, 0.4444444444444444, 0.1, 0.1)
    assert json.loads(json.dumps(entry)) == entry
    assert entry["U"] == 0.444444
    assert entry["stage"] == "dev"


def test_rows_are_returned_in_iteration_order_whatever_order_they_arrived() -> None:
    shuffled = [REAL_RUN[2], REAL_RUN[0], REAL_RUN[3], REAL_RUN[1]]
    assert [r["iteration"] for r in scoreboard.rows_for_stage(shuffled)] == [1, 3, 4, 5]


def test_a_malformed_entry_cannot_take_the_run_down() -> None:
    """History is rehydrated from a checkpointer; one bad row must not raise.

    Every caller of this module sits on a routing path, so an exception here
    would end the run rather than degrade one decision.
    """
    history = [*REAL_RUN, "not a dict", None, {"no": "iteration"}]
    assert scoreboard.best_of(history) == (0.317211, 1)
    assert scoreboard.champion_iteration(history, fallback=99) == 1


def test_the_full_stage_is_kept_but_never_mixed_with_the_dev_stage() -> None:
    """50 checkpoints and 579 checkpoints are not the same measurement.

    A full-stage row scoring higher than every dev row must not be able to
    become the champion: the champion decides which WORKSPACE is inherited, and
    picking it by comparing scores over different populations is a measurement
    error dressed as a decision.
    """
    history = [*REAL_RUN, row(5, 0.9, 0.0, 0.0, stage="full")]
    assert scoreboard.best_of(history) == (0.317211, 1)
    assert scoreboard.best_of(history, "full")[1] == 5
    assert [r["iteration"] for r in scoreboard.rows_for_stage(history, "full")] == [5]


# ======================================================================
# best_of -- the number the halt reason got wrong
# ======================================================================


def test_best_of_returns_the_maximum_not_the_most_recent() -> None:
    """THE BUG, stated as an assertion.

    `halt_reason_for` formatted "best MGS" out of `state["mgs_score"]`, which is
    the LAST score. On this history that produced `best MGS=0.1190` when the
    best was 0.3172 -- the run's headline number reporting its worst result.
    """
    best_mgs, best_iteration = scoreboard.best_of({"score_history": REAL_RUN})
    assert best_iteration == 1
    assert round(best_mgs, 4) == 0.3172
    assert round(REAL_RUN[-1]["MGS"], 4) == 0.1190  # what it used to report


def test_best_of_is_zero_before_anything_has_been_judged() -> None:
    assert scoreboard.best_of({}) == (0.0, 0)
    assert scoreboard.best_of({"score_history": []}) == (0.0, 0)


def test_a_tie_keeps_the_earlier_iteration_as_best() -> None:
    """See the module docstring: a tie is drift with a better cover story."""
    history = [row(1, 0.5, 0.0, 0.0), row(2, 0.5, 0.0, 0.0)]
    assert scoreboard.best_of(history) == (0.5, 1)


# ======================================================================
# champion_iteration -- which workspace the next Developer inherits
# ======================================================================


def test_the_champion_is_the_previous_iteration_while_the_run_is_improving() -> None:
    """The rollback must be INVISIBLE on a healthy run.

    If it were not, this change would be altering the behaviour of every run
    rather than only the ones that regressed.
    """
    history = [row(1, 0.3, 0.1, 0.1), row(2, 0.5, 0.1, 0.1), row(3, 0.7, 0.1, 0.1)]
    assert scoreboard.champion_iteration({"score_history": history}, fallback=3) == 3


def test_the_champion_is_the_best_iteration_once_the_run_regresses() -> None:
    """THE FIX, on the real run's numbers.

    Iteration 6 of run-8cf58d33b311 would have been seeded from iteration 5's
    code -- the worst the run produced. It is seeded from iteration 1's instead.
    """
    assert scoreboard.champion_iteration({"score_history": REAL_RUN}, fallback=5) == 1


def test_a_failed_build_can_never_become_the_parent_of_the_next_iteration() -> None:
    """Iteration 2 wrote nothing and was never judged; iteration 3 inherited it.

    With no row of its own it is not a candidate at all, so this cannot recur.
    """
    history = [REAL_RUN[0]]  # only iteration 1 has been judged
    assert scoreboard.champion_iteration({"score_history": history}, fallback=2) == 1


def test_the_fallback_is_used_when_nothing_has_been_judged() -> None:
    """Iteration 1, and any run whose every build has failed so far."""
    assert scoreboard.champion_iteration({}, fallback=0) == 0
    assert scoreboard.champion_iteration({"score_history": []}, fallback=3) == 3


def test_rollback_can_be_switched_off(monkeypatch) -> None:
    """`ROLLBACK_TO_BEST=False` restores the old unconditional N-1 lineage."""
    monkeypatch.setattr(config, "ROLLBACK_TO_BEST", False)
    assert scoreboard.champion_iteration({"score_history": REAL_RUN}, fallback=5) == 5


def test_the_champion_row_carries_the_workspace_that_produced_it() -> None:
    history = [row(1, 0.5, 0.0, 0.0, workspace="/runs/iter_1/workspace"),
               row(2, 0.2, 0.0, 0.0, workspace="/runs/iter_2/workspace")]
    assert scoreboard.champion_workspace({"score_history": history}) == "/runs/iter_1/workspace"


# ======================================================================
# verdict_for / regression_streak
# ======================================================================


def test_every_verdict_on_the_real_run() -> None:
    """Iteration 1 is the first measurement; 3, 4 and 5 are all regressions."""
    assert scoreboard.verdict_for(REAL_RUN, 1) == scoreboard.VERDICT_FIRST
    assert scoreboard.verdict_for(REAL_RUN, 3) == scoreboard.VERDICT_REGRESSION
    assert scoreboard.verdict_for(REAL_RUN, 4) == scoreboard.VERDICT_REGRESSION
    assert scoreboard.verdict_for(REAL_RUN, 5) == scoreboard.VERDICT_REGRESSION


def test_a_verdict_is_measured_against_the_best_not_against_the_predecessor() -> None:
    """A partial recovery is still a regression.

    Iteration 3 falls off a cliff; iteration 4 climbs back some of the way and
    beats iteration 3 comfortably -- while still sitting well below iteration 1.
    Comparing against the PREDECESSOR would call that an improvement, make
    iteration 4 the champion, and adopt code the run has already measured as
    0.2 MGS worse than what it had. Once a rollback is in play, iteration N's
    parent IS the champion, so the champion is also the honest comparison.
    """
    history = [row(1, 0.80, 0.0, 0.0), row(2, 0.20, 0.0, 0.0), row(3, 0.60, 0.0, 0.0)]
    assert history[2]["MGS"] > history[1]["MGS"]            # 3 beat 2 by a lot
    assert scoreboard.verdict_for(history, 3) == scoreboard.VERDICT_REGRESSION
    assert scoreboard.champion_iteration({"score_history": history}, fallback=3) == 1


def test_an_improvement_is_recognised_as_one() -> None:
    history = [row(1, 0.3, 0.1, 0.1), row(2, 0.6, 0.1, 0.1)]
    assert scoreboard.verdict_for(history, 2) == scoreboard.VERDICT_IMPROVED


def test_an_exact_tie_is_its_own_verdict() -> None:
    history = [row(1, 0.5, 0.0, 0.0), row(2, 0.5, 0.0, 0.0)]
    assert scoreboard.verdict_for(history, 2) == scoreboard.VERDICT_TIED


def test_the_regression_streak_counts_back_from_the_end() -> None:
    assert scoreboard.regression_streak(REAL_RUN) == 3
    assert scoreboard.regression_streak([REAL_RUN[0]]) == 0
    recovered = [*REAL_RUN, row(6, 0.9, 0.0, 0.0)]
    assert scoreboard.regression_streak(recovered) == 0


def test_the_tolerance_widens_what_counts_as_an_improvement(monkeypatch) -> None:
    """A margin can be demanded on a noisy slice; a tie never clears it."""
    history = [row(1, 0.50, 0.0, 0.0), row(2, 0.51, 0.0, 0.0)]
    assert scoreboard.verdict_for(history, 2) == scoreboard.VERDICT_IMPROVED
    monkeypatch.setattr(config, "ROLLBACK_TOLERANCE", 0.05)
    assert scoreboard.verdict_for(history, 2) == scoreboard.VERDICT_TIED


# ======================================================================
# term_deltas -- the question marginal_contributions cannot answer
# ======================================================================


def test_term_deltas_name_the_term_that_actually_moved() -> None:
    """Iteration 1 -> 3: A and F went to ZERO and U halved.

    `marginal_contributions` looked at iteration 3 and reported U as dominant
    with a gain of +0.7778 -- which is true and useless, because A and F were
    already perfect and U was the only term left to name. The movement says the
    thing that matters: U fell 0.222, and that fall cost more MGS than the
    improvements in A and F bought.
    """
    deltas = scoreboard.term_deltas(REAL_RUN[0], REAL_RUN[1])
    assert deltas["moved"]["U"] < 0        # U fell
    assert deltas["moved"]["A"] < 0        # A improved (lower is better)
    assert deltas["moved"]["F"] < 0        # F improved
    assert deltas["worst_term"] == "U"
    assert deltas["delta_mgs_from"]["U"] < 0
    assert deltas["delta_mgs_from"]["A"] > 0
    assert deltas["delta_mgs_from"]["F"] > 0
    assert deltas["delta_mgs"] < 0         # and the net was still a loss


def test_term_deltas_decompose_the_total_up_to_the_interaction_term() -> None:
    """A product has cross terms; the residual is reported, not hidden."""
    deltas = scoreboard.term_deltas(REAL_RUN[0], REAL_RUN[3])
    total = sum(deltas["delta_mgs_from"].values()) + deltas["interaction"]
    assert abs(total - deltas["delta_mgs"]) < 1e-9


def test_a_term_that_did_not_move_contributes_exactly_zero() -> None:
    a = row(1, 0.5, 0.2, 0.1)
    b = row(2, 0.4, 0.2, 0.1)
    deltas = scoreboard.term_deltas(a, b)
    assert deltas["delta_mgs_from"]["A"] == 0.0
    assert deltas["delta_mgs_from"]["F"] == 0.0
    assert deltas["worst_term"] == "U"


def test_term_deltas_are_empty_without_two_rows_to_compare() -> None:
    assert scoreboard.term_deltas(None, REAL_RUN[0]) == {}
    assert scoreboard.term_deltas(REAL_RUN[0], None) == {}


def test_an_iteration_that_traded_privacy_for_utility_is_attributed_to_privacy() -> None:
    """The case `marginal_contributions` structurally cannot surface.

    U rose, so "fix U" is still the highest marginal gain -- and the run got
    WORSE, because A rose further. Only the movement identifies A.
    """
    before = row(1, 0.40, 0.05, 0.05)
    after = row(2, 0.45, 0.40, 0.05)
    deltas = scoreboard.term_deltas(before, after)
    assert deltas["moved"]["U"] > 0
    assert deltas["worst_term"] == "A"
    assert deltas["delta_mgs"] < 0


# ======================================================================
# regression_report
# ======================================================================


def test_the_regression_report_is_empty_when_there_is_nothing_to_explain() -> None:
    assert scoreboard.regression_report({"score_history": []}) == {}
    assert scoreboard.regression_report({"score_history": [REAL_RUN[0]]}) == {}
    improving = [row(1, 0.3, 0.1, 0.1), row(2, 0.6, 0.1, 0.1)]
    assert scoreboard.regression_report({"score_history": improving}) == {}


def test_the_regression_report_names_the_champion_the_delta_and_the_streak() -> None:
    report = scoreboard.regression_report({"score_history": REAL_RUN})
    assert report["verdict"] == scoreboard.VERDICT_REGRESSION
    assert report["iteration"] == 5
    assert report["champion_iteration"] == 1
    assert round(report["champion_mgs"], 4) == 0.3172
    assert round(report["current_mgs"], 4) == 0.1190
    assert round(report["delta"], 4) == -0.1983
    assert report["streak"] == 3
    assert report["terms"]["worst_term"] == "U"


# ======================================================================
# trend_table -- the block the Architect reads
# ======================================================================


def test_the_trend_table_shows_every_iteration_with_its_delta() -> None:
    table = scoreboard.trend_table({"score_history": REAL_RUN})
    assert "0.3172" in table and "0.1190" in table
    assert "-0.0950" in table                     # iteration 3's delta
    assert "BEST SO FAR: iteration 1" in table
    assert "REGRESSION" in table


def test_the_trend_table_shouts_about_a_streak() -> None:
    table = scoreboard.trend_table({"score_history": REAL_RUN})
    assert "3 ITERATIONS IN A ROW" in table


def test_a_failed_build_gets_a_row_saying_so_rather_than_a_gap() -> None:
    """A gap in a numbered history reads as a lost record, not as a failure."""
    table = scoreboard.trend_table(
        {"score_history": REAL_RUN},
        failed_iterations=[{"iteration": 2, "missing_gates": ["tests_ok"]}],
    )
    assert "build failed, never evaluated" in table
    lines = [line for line in table.splitlines() if line.strip().startswith("2 ")]
    assert lines, table


def test_the_table_tells_the_architect_its_workspace_was_rolled_back() -> None:
    """Without this the Architect writes a work order to revert changes that are
    already gone, and the whole iteration is a no-op."""
    table = scoreboard.trend_table({"score_history": REAL_RUN}, rolled_back_to=1)
    assert "ALREADY GONE" in table
    assert "rolled back to iteration 1's code" in table
    assert "Design a" in table and "DIFFERENT approach" in table


def test_without_a_rollback_the_table_offers_a_revert_as_the_work_order() -> None:
    table = scoreboard.trend_table({"score_history": REAL_RUN})
    assert "Reverting the change that caused this is a legitimate work order" in table


def test_the_first_iteration_gets_an_honest_empty_table() -> None:
    assert "nothing has been judged yet" in scoreboard.trend_table({})


# ======================================================================
# summary -- what lands in summary_run-*.json
# ======================================================================


def test_the_summary_reports_both_the_best_and_the_final() -> None:
    board = scoreboard.summary({"score_history": REAL_RUN})
    assert board["best_iteration"] == 1
    assert round(board["best_mgs"], 4) == 0.3172
    assert board["final_iteration"] == 5
    assert round(board["final_mgs"], 4) == 0.1190
    assert board["regressed_from_best"] is True
    assert board["regression_streak"] == 3
    assert len(board["history"]) == 4


def test_a_healthy_run_is_not_reported_as_regressed() -> None:
    board = scoreboard.summary(
        {"score_history": [row(1, 0.3, 0.1, 0.1), row(2, 0.6, 0.1, 0.1)]}
    )
    assert board["regressed_from_best"] is False
    assert board["best_iteration"] == board["final_iteration"] == 2
