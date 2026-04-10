"""Tests for autoresearch plateau detection."""

import random

from swarm.analysis.autoresearch import _mutate_governance
from swarm.core.orchestrator import OrchestratorConfig
from swarm.governance.config import GovernanceConfig


def test_mutate_governance_produces_change():
    """_mutate_governance should change at least one parameter."""
    config = OrchestratorConfig()
    config.governance_config = GovernanceConfig()

    # Create a minimal scenario-like object
    class FakeScenario:
        orchestrator_config = config

    scenario = FakeScenario()
    rng = random.Random(42)

    param, old, new = _mutate_governance(scenario, rng)
    assert param  # non-empty string
    assert old != new  # something changed


def test_multiple_mutations_change_multiple_params():
    """Applying _mutate_governance multiple times can change multiple params."""
    config = OrchestratorConfig()
    config.governance_config = GovernanceConfig()

    class FakeScenario:
        orchestrator_config = config

    scenario = FakeScenario()
    rng = random.Random(42)

    params_changed = set()
    for _ in range(10):
        param, old, new = _mutate_governance(scenario, rng)
        params_changed.add(param)

    # With 10 mutations and random seed, we should hit at least 2 different params
    assert len(params_changed) >= 2
