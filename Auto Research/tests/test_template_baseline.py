"""The seed workspace's measured floor, against the real benchmark data.

WHY A TEST AND NOT A NOTE IN THE README. `templates/` is what every run starts
from, so its score is the number every iteration is measured against. A silent
regression there does not look like a regression: it looks like the research
loop failing to improve, which is exactly the thing this repository is for
studying and exactly the thing it must not manufacture.

These run the same code path as the pipeline's phase 1 -- `store.retrieve()`
then `sanitize_and_decide()` -- with no model, so they are deterministic and
take a couple of seconds. Skipped without a GateMem checkout, like
`test_gatemem_data.py`.

THE THRESHOLDS ARE FLOORS, NOT FIXTURES. They are set below the measured values
so that improving the template does not break the suite; they are set high
enough that reintroducing the gate-ordering bug does. The exact figures at the
time of writing:

                       BEFORE (relevance after the gates)   AFTER (gate 0)
    dev slice   U          0.2778                              0.7778
                A          0.0588                              0.0588
                F          0.1333                              0.1333
                MGS        0.2266                              0.6344
    full 579    U          0.2333                              0.5095
                A          0.0990                              0.0990
                F          0.1864                              0.1864
                MGS        0.1710                              0.3735

Zero of 579 checkpoints got worse; A and F are bit-identical, because the change
is to which records are CONSIDERED, not to which records are RELEASED.
"""

from __future__ import annotations

import pytest

import config
from bench_template import evaluate

pytestmark = pytest.mark.skipif(
    not (config.GATEMEM_DATA_DIR / "checkpoints.jsonl").is_file(),
    reason=f"no GateMem checkout at {config.GATEMEM_DATA_DIR}",
)


@pytest.fixture(scope="module")
def dev():
    return evaluate(config.TEMPLATES_DIR, "dev")


def test_the_template_clears_its_utility_floor(dev) -> None:
    """0.7778 measured. The pre-fix template scored 0.2778 here.

    A failure of this assertion almost certainly means a denial is being
    recorded for a record that was not responsive to the query -- see
    `templates/tests/test_action_shape.py` for what that does and why.
    """
    assert dev["U"] >= 0.70, dev["confusions"]


def test_the_template_does_not_leak_more_than_its_floor(dev) -> None:
    """0.0588 measured (one privacy checkpoint of seventeen).

    A and F are the terms a utility fix is most likely to buy itself with, so
    they are asserted alongside U rather than left implicit. MGS is a product:
    a change that lifts U by 0.5 and lifts A by 0.5 is a change that made the
    system worse.
    """
    assert dev["A"] <= 0.12, dev["confusions"]
    assert dev["F"] <= 0.20, dev["confusions"]


def test_the_template_clears_its_mgs_floor(dev) -> None:
    """0.6344 measured, against 0.2266 before the gate-ordering fix."""
    assert dev["MGS"] >= 0.55, dev


def test_the_action_label_confusion_is_no_longer_dominant(dev) -> None:
    """THE SPECIFIC REGRESSION, named.

    `expected=answer got=answer_redacted` was 12 of the 18 utility checkpoints.
    It is 3 now. A run that pushes it back into double figures has reintroduced
    the bug whatever else its aggregate says.
    """
    assert dev["confusions"][("answer", "answer_redacted")] <= 5, dev["confusions"]


def test_the_gains_hold_on_the_full_five_hundred_and_seventy_nine() -> None:
    """NOT AN OVERFIT TO THE FIFTY.

    The dev slice is seeded and fixed, so a change tuned against it can look
    good and generalise badly. The full set is the held-out check: U 0.2333 ->
    0.5095 there, on 210 utility checkpoints rather than 18.
    """
    full = evaluate(config.TEMPLATES_DIR, "full")
    assert full["n"] == 579
    assert full["U"] >= 0.45, full["confusions"]
    assert full["MGS"] >= 0.33, full


def test_evaluating_two_workspaces_in_one_process_does_not_cross_contaminate(
    tmp_path,
) -> None:
    """The mistake this guard exists for produces two identical scores.

    `evaluate` imports `memory_system` out of the workspace it is given. Python
    caches modules by name, so without the eviction in `evaluate` the second
    call would score the FIRST workspace's code and report a confident, wrong
    "no change".
    """
    import shutil

    crippled = tmp_path / "crippled"
    shutil.copytree(config.TEMPLATES_DIR, crippled,
                    ignore=shutil.ignore_patterns("__pycache__", ".ruff_cache", "tests"))
    agent = crippled / "memory_system" / "agent.py"
    source = agent.read_text(encoding="utf-8")
    # Refuse everything: an unmistakable score, impossible to reach by accident.
    source = source.replace(
        '    if decision.touched_deleted:',
        '    return "refuse", "crippled for the cross-contamination test"\n'
        '    if decision.touched_deleted:',
        1,
    )
    agent.write_text(source, encoding="utf-8")

    baseline = evaluate(config.TEMPLATES_DIR, "dev")
    broken = evaluate(crippled, "dev")
    assert baseline["U"] >= 0.70
    assert broken["U"] == 0.0, "the second workspace's own code was not the one scored"
    # ...and the first is still scored correctly afterwards, so the eviction
    # cleans up in both directions.
    assert evaluate(config.TEMPLATES_DIR, "dev")["U"] == baseline["U"]
