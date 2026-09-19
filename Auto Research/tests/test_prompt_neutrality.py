"""The prompts may say WHAT WAS MEASURED. They may not say WHICH COMPONENT IS AT FAULT.

WHY THIS IS A TEST AND NOT A CONVENTION
---------------------------------------
This loop's headline claim is that it diagnoses and repairs a system by itself.
That claim is only worth as much as the prompts are innocent of the answer, and
the prompts drifted the other way for a long time without anyone noticing:

* `_COMPONENT_HYPOTHESES` handed the Critic a per-metric list of suspects --
  the retrieval filter for U, the role_grants join for A, the tombstone gate
  for F -- as priors "to confirm or replace".
* A census label read "Implicates ONE function: the Decision -> action mapping,
  `sanitize_and_decide` in memory_system/agent.py. A retrieval, schema or index
  change cannot fix these" -- the file, the function, the repair, and the ruled
  out alternatives.
* A failure-group legend read "this is a decision-layer repair, and no retrieval
  change will touch it".

Each was added in good faith to make a real failure legible. Together they meant
a reported finding could be a finding the prompt supplied. Deleting them once
does not keep them gone, because the pressure that produced them -- a run going
nowhere, and an obvious sentence that would fix it -- recurs every time.

WHAT IS STILL ALLOWED, and the line is not subtle: counters (`allowed=8`,
`denied_tombstone=26`), counts per group, checkpoint ids, the Judge's reasons,
the source itself. Those are observations. What may not appear is a claim about
which component is responsible, or which class of change would fix it.
"""

from __future__ import annotations

import pytest

# Identifiers from the system under test. Naming one inside an agent's
# instructions points at a component before the agent has looked.
IMPLEMENTATION_NAMES = (
    "sanitize_and_decide",
    "role_grants",
    "top_k",
    "memory_system/agent.py",
    "memory_system/store.py",
    "retrieve()",
)

# Phrasings that assert a cause or prescribe a repair class.
VERDICT_PHRASES = (
    "implicates",
    "is the fix",
    "cannot fix",
    "will not move",
    "the only group",
    "repair, and no",
    "most plausibly",
    "hypothesised component",
    "component hypotheses",
)


def _agent_prompt_text() -> dict[str, str]:
    """Every instruction string the Architect, Developer and Critic can read."""
    from harness.profiles import ARCHITECT_PROFILE, CRITIC_PROFILE, DEVELOPER_PROFILE
    from nodes import critic, developer

    census = " ".join(critic._MECHANISM_MEANING.values())
    return {
        "architect persona": ARCHITECT_PROFILE.system_prompt,
        "developer persona": DEVELOPER_PROFILE.system_prompt,
        "critic persona": CRITIC_PROFILE.system_prompt,
        "developer task note": developer.BENCHMARK_TOOL_NOTE,
        "critic mechanism labels": census,
    }


@pytest.mark.parametrize("name", IMPLEMENTATION_NAMES)
def test_no_prompt_names_a_component_of_the_system_under_test(name: str) -> None:
    for where, text in _agent_prompt_text().items():
        assert name not in text, (
            f"{where} names {name!r}. The agents must locate components from the "
            f"evidence, not from their instructions."
        )


@pytest.mark.parametrize("phrase", VERDICT_PHRASES)
def test_no_prompt_asserts_a_cause_or_prescribes_a_repair(phrase: str) -> None:
    for where, text in _agent_prompt_text().items():
        assert phrase not in text.lower(), (
            f"{where} contains {phrase!r}, which states a conclusion the loop is "
            f"supposed to reach on its own."
        )


def test_the_measured_material_is_still_there() -> None:
    """The guard must not have been satisfied by deleting the evidence.

    Neutrality is only worth having if the Critic can still see what happened,
    so this asserts the observations survived the removal of the conclusions.
    """
    from nodes import critic

    labels = " ".join(critic._MECHANISM_MEANING.values()).lower()
    for observation in ("counters", "answer", "denied", "allowed", "scan"):
        assert observation in labels, f"the mechanism labels no longer describe {observation}"
    assert "crashed_or_missing" in critic._MECHANISM_MEANING
    assert not hasattr(critic, "_COMPONENT_HYPOTHESES")
