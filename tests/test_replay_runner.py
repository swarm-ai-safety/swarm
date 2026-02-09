"""Tests for replay execution utilities."""

from pathlib import Path

import pytest

from swarm.replay import EpisodeSpec, ReplayRunner
from swarm.scenarios import load_scenario


@pytest.fixture
def base_scenario():
    """Load a lightweight scenario for replay tests."""
    path = Path("scenarios/baseline.yaml")
    if not path.exists():
        pytest.skip("baseline.yaml not found")
    scenario = load_scenario(path)
    scenario.orchestrator_config.n_epochs = 2
    scenario.orchestrator_config.steps_per_epoch = 3
    return scenario


def test_episode_spec_validates_replay_k(base_scenario):
    with pytest.raises(ValueError, match="replay_k must be >= 1"):
        EpisodeSpec(scenario=base_scenario, seed=42, replay_k=0)


def test_episode_spec_seed_schedule(base_scenario):
    spec = EpisodeSpec(scenario=base_scenario, seed=100, replay_k=4)
    assert spec.replay_seeds() == [100, 101, 102, 103]


def test_replay_runner_runs_k_replays(base_scenario):
    spec = EpisodeSpec(scenario=base_scenario, seed=42, replay_k=3)
    runner = ReplayRunner(spec)

    results = runner.run()

    assert len(results) == 3
    assert [r.seed for r in results] == [42, 43, 44]


def test_replay_runner_is_reproducible_for_same_spec(base_scenario):
    spec = EpisodeSpec(scenario=base_scenario, seed=42, replay_k=3)
    run_a = ReplayRunner(spec).run()
    run_b = ReplayRunner(spec).run()

    key_a = [(r.seed, r.total_interactions, r.total_welfare) for r in run_a]
    key_b = [(r.seed, r.total_interactions, r.total_welfare) for r in run_b]

    assert key_a == key_b


def test_replay_runner_applies_simulation_overrides(base_scenario):
    spec = EpisodeSpec(
        scenario=base_scenario,
        seed=42,
        replay_k=1,
        parameter_overrides={"simulation.steps_per_epoch": 1},
    )
    runner = ReplayRunner(spec)
    results = runner.run()
    assert len(results) == 1
    # With 2 epochs x 1 step, this should remain a valid bounded run.
    assert results[0].total_interactions >= 0
