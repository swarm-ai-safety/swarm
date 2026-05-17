"""Tests for the personality distribution sweep scenarios and runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.runner import DeliveryScenarioRunner

SCENARIO_DIR = Path(__file__).resolve().parent.parent / "scenarios"

PERSONALITY_SCENARIOS = [
    "simworld_delivery_personality_conscientious.yaml",
    "simworld_delivery_personality_aggressive.yaml",
    "simworld_delivery_personality_cautious.yaml",
    "simworld_delivery_personality_opportunistic.yaml",
]


@pytest.fixture(params=PERSONALITY_SCENARIOS)
def scenario(request):
    path = SCENARIO_DIR / request.param
    with open(path) as f:
        return yaml.safe_load(f)


class TestPersonalityScenarios:
    """Verify personality scenario YAMLs are valid and runnable."""

    def test_scenario_has_required_fields(self, scenario):
        assert "scenario_id" in scenario
        assert "agents" in scenario
        assert "simulation" in scenario
        assert "delivery" in scenario
        assert scenario["motif"] == "simworld_delivery"

    def test_agent_count_is_eight(self, scenario):
        total = sum(a.get("count", 1) for a in scenario["agents"])
        assert total == 8, f"Expected 8 agents, got {total}"

    def test_all_policies_are_valid(self, scenario):
        valid = {"conscientious", "aggressive", "cautious", "opportunistic"}
        for agent_spec in scenario["agents"]:
            assert agent_spec["policy"] in valid

    def test_scenario_runs_one_epoch(self, scenario):
        delivery_cfg = scenario.get("delivery", {})
        config = DeliveryConfig.from_dict({**delivery_cfg, "seed": 0})

        runner = DeliveryScenarioRunner(
            config=config,
            agent_specs=scenario.get("agents", []),
            n_epochs=1,
            steps_per_epoch=5,
            seed=0,
        )
        metrics = runner.run()
        assert len(metrics) == 1
        assert metrics[0].orders_created > 0


class TestParetoFront:
    """Test the Pareto front computation."""

    def test_pareto_front_simple(self):
        from examples.run_simworld_personality_sweep import compute_pareto_front

        points = [(0.5, 0.9), (0.7, 0.8), (0.9, 0.7), (0.6, 0.6)]
        front = compute_pareto_front(points)
        # (0.5, 0.9), (0.7, 0.8), (0.9, 0.7) are non-dominated
        # (0.6, 0.6) is dominated by (0.7, 0.8)
        assert len(front) == 3
        assert (0.6, 0.6) not in front

    def test_pareto_front_single_point(self):
        from examples.run_simworld_personality_sweep import compute_pareto_front

        front = compute_pareto_front([(0.5, 0.5)])
        assert front == [(0.5, 0.5)]

    def test_pareto_front_all_dominated(self):
        from examples.run_simworld_personality_sweep import compute_pareto_front

        # Only (0.9, 0.9) dominates everything
        points = [(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]
        front = compute_pareto_front(points)
        assert (0.9, 0.9) in front
