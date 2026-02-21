"""Scenario-level tests for Letta integration.

Tests loading a YAML scenario with Letta agents, verifying agent creation,
and running 1 epoch with a mocked client.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from swarm.agents.letta_agent import LettaAgent
from swarm.scenarios.loader import create_agents, load_scenario

LETTA_SCENARIO_YAML = """\
scenario_id: letta_test
description: "Test scenario with Letta agents"

letta:
  enabled: true
  base_url: "http://localhost:8283"
  server_mode: external
  default_model: "anthropic/claude-sonnet-4-20250514"
  shared_governance_block: true

agents:
  - type: letta
    count: 2
    letta:
      persona: "You are a cooperative agent."
      archetype: cooperative

  - type: honest
    count: 2

simulation:
  n_epochs: 1
  steps_per_epoch: 3
  seed: 42
"""


class TestLettaScenarioLoading:
    def test_load_scenario_with_letta(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(LETTA_SCENARIO_YAML)
            f.flush()
            scenario = load_scenario(Path(f.name))

        assert scenario.scenario_id == "letta_test"
        assert scenario.orchestrator_config.letta_config is not None
        assert scenario.orchestrator_config.letta_config.enabled is True
        assert (
            scenario.orchestrator_config.letta_config.server_mode == "external"
        )

    def test_create_letta_agents(self):
        agent_specs = [
            {
                "type": "letta",
                "count": 3,
                "letta": {
                    "persona": "Test persona",
                    "archetype": "strategic",
                },
            }
        ]
        agents = create_agents(agent_specs, seed=42)
        assert len(agents) == 3
        for agent in agents:
            assert isinstance(agent, LettaAgent)
            assert agent._persona == "Test persona"
            assert agent._archetype == "strategic"
            assert not agent._initialized

    def test_create_mixed_agents(self):
        agent_specs = [
            {
                "type": "letta",
                "count": 2,
                "letta": {"persona": "Cooperative"},
            },
            {"type": "honest", "count": 3},
        ]
        agents = create_agents(agent_specs, seed=42)
        assert len(agents) == 5
        letta_agents = [a for a in agents if isinstance(a, LettaAgent)]
        assert len(letta_agents) == 2


class TestLettaScenarioRun:
    def test_run_with_mocked_letta(self):
        """Run 1 epoch with Letta agents backed by a mock client."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(LETTA_SCENARIO_YAML)
            f.flush()
            scenario = load_scenario(Path(f.name))

        from swarm.scenarios.loader import build_orchestrator

        orchestrator = build_orchestrator(scenario)

        # Mock the Letta lifecycle so it doesn't need a real server
        mock_lifecycle = MagicMock()
        mock_lifecycle.start.return_value = True
        mock_lifecycle.governance_block_id = "mock-gov-block"
        mock_lifecycle.client = MagicMock()
        mock_lifecycle.client.create_agent.return_value = "mock-letta-id"
        mock_lifecycle.client.send_message.return_value = '{"action": "noop"}'
        mock_lifecycle.client.update_core_memory.return_value = None
        mock_lifecycle.client.get_core_memory.return_value = "{}"
        mock_lifecycle.client.attach_shared_block.return_value = None
        mock_lifecycle.client.insert_archival.return_value = None
        mock_lifecycle.memory_mapper = MagicMock()
        mock_lifecycle.memory_mapper.observation_to_memory_blocks.return_value = {
            "swarm_state": "{}"
        }
        mock_lifecycle.memory_mapper.observation_to_message.return_value = (
            "Test step message"
        )
        mock_lifecycle.memory_mapper.extract_trust_updates.return_value = {}
        mock_lifecycle.response_parser = MagicMock()
        mock_lifecycle.response_parser.parse.return_value = MagicMock(
            action_type=MagicMock(value="noop"),
        )

        # Replace the lifecycle manager
        orchestrator._letta_lifecycle = mock_lifecycle

        # Run
        metrics = orchestrator.run()
        assert len(metrics) == 1  # 1 epoch
        assert mock_lifecycle.start.called
        assert mock_lifecycle.shutdown.called
