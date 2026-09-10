"""The loop can now say "that made it worse" -- and is told to.

THREE NODES HAD TO CHANGE, because the loop was blind to its own regressions in
three different places at once, and fixing any one of them alone leaves the
other two able to reproduce the failure:

  * THE JUDGE could measure a fall and not record it. `mgs_score` was a scalar
    overwritten every iteration, so nothing anywhere held the previous value.
  * THE ARCHITECT was shown four scalars under the heading "MEASURED PERFORMANCE
    SO FAR" -- true of one iteration, silent about the run. It designed
    iterations 3, 4 and 5 of run-8cf58d33b311 without ever being told that its
    previous three designs had each lost MGS.
  * THE CRITIC had no vocabulary for it. `marginal_contributions` ranks terms by
    what perfecting each is worth, and under MGS = U*(1-A)*(1-F) with small A
    and F that names U on essentially every round. It named U four times out of
    four while the actual story was that iteration 3's edit had halved U.

These tests pin the rendered text rather than only the numbers, because the text
is the interface: a fact computed correctly and phrased so the model reads past
it is a fact the loop still does not have.
"""

from __future__ import annotations

import scoreboard
from nodes.critic import (
    _fallback_critique,
    _regression_block,
    _regression_json_keys,
    _regression_task_step,
    marginal_contributions,
)


def rows(*specs, stage="dev"):
    return [
        scoreboard.score_row(iteration=i, stage=stage, phase="standard_retrieval",
                             utility=u, access=a, forgetting=f, mgs=u * (1 - a) * (1 - f))
        for i, u, a, f in specs
    ]


#: run-8cf58d33b311's dev-stage scores.
REAL_RUN = rows(
    (1, 0.4444444444444444, 0.17647058823529413, 0.13333333333333333),
    (3, 0.2222222222222222, 0.0, 0.0),
    (4, 0.2222222222222222, 0.11764705882352941, 0.06666666666666667),
    (5, 0.16666666666666666, 0.17647058823529413, 0.13333333333333333),
)


# ======================================================================
# the structural bias the regression block exists to counteract
# ======================================================================


def test_the_marginal_ranking_names_u_on_every_iteration_of_the_real_run() -> None:
    """NOT A BUG IN `marginal_contributions` -- a property of the metric.

    This is why "dominant_term = U" carried no information for four consecutive
    critiques, and why the regression block has to outrank it in the prompt.
    """
    for row in REAL_RUN:
        attribution = marginal_contributions(row["U"], row["A"], row["F"])
        assert attribution["dominant_term"] == "U", row


def test_the_ranking_names_u_even_when_u_is_the_term_that_improved() -> None:
    """The case that makes the ranking actively misleading.

    U rose and A rose further, so MGS fell -- and "fix U" is still the largest
    marginal gain. Only the movement identifies the term that did the damage.
    """
    before, after = rows((1, 0.40, 0.05, 0.05), (2, 0.45, 0.40, 0.05))
    assert marginal_contributions(after["U"], after["A"], after["F"])["dominant_term"] == "U"
    assert scoreboard.term_deltas(before, after)["worst_term"] == "A"


# ======================================================================
# the Critic's regression block
# ======================================================================


def test_there_is_no_regression_block_when_there_is_no_regression() -> None:
    assert _regression_block({}) == ""
    assert _regression_task_step({}) == ""
    assert _regression_json_keys({}) == ""


def test_the_regression_block_leads_with_the_comparison() -> None:
    block = _regression_block(scoreboard.regression_report({"score_history": REAL_RUN}))
    assert block.startswith("## !! REGRESSION -- THIS IS THE FINDING")
    assert "0.1190" in block and "0.3172" in block
    assert "-0.1983" in block


def test_the_regression_block_decomposes_the_movement_by_term() -> None:
    block = _regression_block(scoreboard.regression_report({"score_history": REAL_RUN}))
    assert "WHICH TERM MOVED" in block
    assert "The costliest movement was U." in block


def test_the_regression_block_escalates_on_a_streak() -> None:
    block = _regression_block(scoreboard.regression_report({"score_history": REAL_RUN}))
    assert "3 CONSECUTIVE ITERATIONS" in block
    assert "already measured as wrong" in block


def test_a_single_regression_does_not_shout_about_a_streak() -> None:
    history = rows((1, 0.8, 0.0, 0.0), (2, 0.4, 0.0, 0.0))
    block = _regression_block(scoreboard.regression_report({"score_history": history}))
    assert "CONSECUTIVE ITERATIONS" not in block


def test_a_tie_is_explained_as_a_tie_and_says_the_code_is_not_adopted() -> None:
    history = rows((1, 0.5, 0.0, 0.0), (2, 0.5, 0.0, 0.0))
    block = _regression_block(scoreboard.regression_report({"score_history": history}))
    assert "TIED rather than fell" in block
    assert "not being adopted" in block


def test_the_task_step_makes_revert_a_first_class_answer() -> None:
    """The verdict the Critic could not previously give.

    A model asked only for "prioritized changes" gives changes. `revert` has to
    be spelled out as an allowed and complete answer or it is not one.
    """
    step = _regression_task_step(scoreboard.regression_report({"score_history": REAL_RUN}))
    assert "REVERT" in step
    assert "complete and\n         legitimate recommendation" in step
    assert "KEEP AND FIX" in step
    assert "NOT THE CAUSE" in step
    assert "regression_verdict" in step


def test_the_json_contract_asks_for_the_verdict_and_the_cause() -> None:
    keys = _regression_json_keys(scoreboard.regression_report({"score_history": REAL_RUN}))
    assert "regression_verdict" in keys
    assert "revert" in keys and "keep_and_fix" in keys and "not_the_cause" in keys
    assert "regression_cause" in keys


# ======================================================================
# the deterministic fallback
# ======================================================================


def test_the_regression_survives_an_unreachable_critic_model() -> None:
    """It is computed in Python from the score history, so a dead model is no
    reason for the Architect not to be told."""
    attribution = marginal_contributions(0.1667, 0.1765, 0.1333)
    critique = _fallback_critique(
        attribution, cited=[], worst_attacks=[], census="",
        regression=scoreboard.regression_report({"score_history": REAL_RUN}),
    )
    assert "## !! REGRESSION" in critique
    assert "Recommended action: REVERT" in critique
    assert "Costliest movement: U" in critique


def test_the_fallback_without_a_regression_is_unchanged() -> None:
    attribution = marginal_contributions(0.5, 0.1, 0.1)
    critique = _fallback_critique(attribution, cited=[], worst_attacks=[], census="")
    assert "REGRESSION" not in critique
    assert "## Attribution" in critique


# ======================================================================
# the Architect's trend block
# ======================================================================


def test_the_architect_sees_the_whole_trajectory_not_the_last_row() -> None:
    """THE FOUR SCALARS, REPLACED. What the prompt used to carry was
    U/A/F/MGS for one iteration and nothing else."""
    table = scoreboard.trend_table({"score_history": REAL_RUN})
    for value in ("0.3172", "0.2222", "0.1830", "0.1190"):
        assert value in table, value
    assert "delta" in table


def test_the_architect_is_told_which_iteration_it_has_to_beat() -> None:
    table = scoreboard.trend_table({"score_history": REAL_RUN})
    assert "BEST SO FAR: iteration 1" in table
    assert "MOST RECENT: iteration 5" in table


def test_a_rolled_back_architect_is_told_not_to_write_a_revert() -> None:
    """The interaction bug the rollback would otherwise introduce.

    With the workspace already rolled back, "revert iteration 5's change" is a
    work order to undo edits that are not in the code -- a whole iteration spent
    on a no-op. The Architect has to be told the rollback already happened.
    """
    table = scoreboard.trend_table({"score_history": REAL_RUN}, rolled_back_to=1)
    assert "THE LOSING CHANGES ARE ALREADY GONE" in table
    assert "Do NOT write a work order that reverts" in table


def test_without_a_rollback_a_revert_is_recommended_instead() -> None:
    """The two messages are mutually exclusive and must never both appear."""
    table = scoreboard.trend_table({"score_history": REAL_RUN})
    assert "Reverting the change that caused this is a legitimate work order" in table
    assert "ALREADY GONE" not in table
