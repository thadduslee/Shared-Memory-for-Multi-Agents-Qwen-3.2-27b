"""Contract tests against a real GateMem checkout.

Skipped when `GATEMEM_REPO` does not point at one, so the suite still passes
offline -- but when the data IS present these are the tests that catch the
adapter drifting from the benchmark.
"""

from __future__ import annotations

import pytest

import config
from gatemem_adapter import (
    assert_no_hidden_fields,
    checkpoints_for_phase,
    load_medical_dataset,
    select_dev_slice,
    strip_hidden_fields,
)

pytestmark = pytest.mark.skipif(
    not (config.GATEMEM_DATA_DIR / "checkpoints.jsonl").is_file(),
    reason=f"no GateMem checkout at {config.GATEMEM_DATA_DIR}",
)


@pytest.fixture(scope="module")
def dataset():
    return load_medical_dataset(config.GATEMEM_DATA_DIR)


def test_medical_split_has_the_documented_size(dataset) -> None:
    assert len(dataset.checkpoints) == config.EXPECTED_FULL_CHECKPOINTS == 579
    assert len(dataset.episodes) == 21


def test_domain_filter_uses_the_episode_not_the_checkpoint(dataset) -> None:
    """`domain` lives on the EPISODE; filtering the checkpoint key returns
    everything, silently."""
    assert all(ep.get("domain") == "medical" for ep in dataset.episodes)
    assert not any("domain" in cp for cp in dataset.checkpoints)


def test_curriculum_partitions_every_checkpoint_exactly_once(dataset) -> None:
    """A phase that drops checkpoints makes its own pass rate meaningless."""
    seen: set[str] = set()
    for phase in config.CURRICULUM_PHASES:
        ids = {cp["checkpoint_id"] for cp in checkpoints_for_phase(dataset.checkpoints, phase)}
        assert not (ids & seen), f"{phase} overlaps an earlier phase"
        seen |= ids
    assert len(seen) == len(dataset.checkpoints), "some checkpoints belong to no phase"


def test_field_wall_holds_on_all_579_real_checkpoints(dataset) -> None:
    for checkpoint in dataset.checkpoints:
        assert_no_hidden_fields(
            strip_hidden_fields(checkpoint), where=checkpoint["checkpoint_id"]
        )


def test_dev_slice_is_stratified_and_deterministic_on_real_data(dataset) -> None:
    selected = select_dev_slice(
        dataset.checkpoints, n=config.EXPECTED_DEV_CHECKPOINTS, seed=config.DEV_SLICE_SEED
    )
    assert len(selected) == 50
    assert {cp["query_type"] for cp in selected} == {"utility", "privacy", "safety"}
    again = select_dev_slice(
        list(reversed(dataset.checkpoints)), n=50, seed=config.DEV_SLICE_SEED
    )
    assert [c["checkpoint_id"] for c in selected] == [c["checkpoint_id"] for c in again]


def test_as_of_turn_slicing_respects_the_incremental_protocol(dataset) -> None:
    """An agent may only use information available up to the checkpoint turn."""
    for checkpoint in dataset.checkpoints[:20]:
        turns = dataset.turns_up_to(checkpoint["episode_id"], checkpoint["as_of_turn_id"])
        assert turns, checkpoint["checkpoint_id"]
        assert turns[-1]["turn_id"] == checkpoint["as_of_turn_id"]


def test_expected_actions_are_all_in_the_documented_vocabulary(dataset) -> None:
    from gatemem_adapter import VALID_ACTIONS

    assert {cp["expected_action"] for cp in dataset.checkpoints} <= VALID_ACTIONS


def test_mock_dataset_mirrors_the_real_schema(dataset) -> None:
    """If the offline fixture and the real data disagree, the fixture is wrong."""
    from mocks.dataset import build_mock_dataset

    mock = build_mock_dataset()
    real_keys = set(dataset.checkpoints[0]) | {"attack_type"}
    for checkpoint in mock.checkpoints:
        assert set(checkpoint) <= real_keys, f"mock has keys the real data lacks: {set(checkpoint) - real_keys}"
    assert set(mock.episodes[0]) == set(dataset.episodes[0])
    assert {c["query_type"] for c in mock.checkpoints} == {c["query_type"] for c in dataset.checkpoints}
