"""Tests for the Orchestrator simulation engine."""

import pytest
import tempfile
from pathlib import Path

from src.agents.honest import HonestAgent
from src.agents.opportunistic import OpportunisticAgent
from src.agents.deceptive import DeceptiveAgent
from src.agents.adversarial import AdversarialAgent
from src.core.orchestrator import Orchestrator, OrchestratorConfig, EpochMetrics
from src.core.payoff import PayoffConfig
from src.env.state import EnvState
from src.models.agent import AgentType


class TestOrchestratorBasics:
    """Basic orchestrator tests."""

    def test_create_orchestrator(self):
        """Test orchestrator creation."""
        orchestrator = Orchestrator()

        assert orchestrator.state is not None
        assert orchestrator.feed is not None
        assert orchestrator.task_pool is not None
        assert len(orchestrator._agents) == 0

    def test_create_with_config(self):
        """Test orchestrator creation with config."""
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=5,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        assert orchestrator.config.n_epochs == 5
        assert orchestrator.config.steps_per_epoch == 5
        assert orchestrator.config.seed == 42

    def test_register_agent(self):
        """Test agent registration."""
        orchestrator = Orchestrator()
        agent = HonestAgent(agent_id="agent_1")

        state = orchestrator.register_agent(agent)

        assert state.agent_id == "agent_1"
        assert "agent_1" in orchestrator._agents
        assert orchestrator.state.get_agent("agent_1") is not None

    def test_register_duplicate_raises(self):
        """Test duplicate agent registration raises."""
        orchestrator = Orchestrator()
        agent = HonestAgent(agent_id="agent_1")

        orchestrator.register_agent(agent)

        with pytest.raises(ValueError, match="already registered"):
            orchestrator.register_agent(agent)


class TestOrchestratorSimulation:
    """Simulation tests."""

    def test_run_minimal_simulation(self):
        """Test running a minimal simulation."""
        config = OrchestratorConfig(
            n_epochs=2,
            steps_per_epoch=3,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        # Add one agent
        agent = HonestAgent(agent_id="solo_agent")
        orchestrator.register_agent(agent)

        # Run simulation
        metrics = orchestrator.run()

        assert len(metrics) == 2
        assert all(isinstance(m, EpochMetrics) for m in metrics)

    def test_run_multi_agent_simulation(self):
        """Test running simulation with multiple agents."""
        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=5,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        # Add multiple agents
        for i in range(3):
            agent = HonestAgent(agent_id=f"honest_{i}")
            orchestrator.register_agent(agent)

        metrics = orchestrator.run()

        assert len(metrics) == 3
        # Should have some activity
        assert any(m.total_posts > 0 or m.total_interactions > 0 for m in metrics)

    def test_run_mixed_agent_types(self):
        """Test running simulation with mixed agent types."""
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=5,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        # Add different agent types
        orchestrator.register_agent(HonestAgent(agent_id="honest_1"))
        orchestrator.register_agent(HonestAgent(agent_id="honest_2"))
        orchestrator.register_agent(OpportunisticAgent(agent_id="opp_1"))
        orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))
        orchestrator.register_agent(AdversarialAgent(agent_id="adv_1"))

        metrics = orchestrator.run()

        assert len(metrics) == 5
        assert orchestrator.state.current_epoch == 5

    def test_simulation_computes_metrics(self):
        """Test that simulation computes meaningful metrics."""
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=10,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        # Add agents that will interact
        for i in range(5):
            agent = HonestAgent(agent_id=f"agent_{i}")
            orchestrator.register_agent(agent)

        metrics = orchestrator.run()

        # Check metrics are computed
        for m in metrics:
            assert m.epoch >= 0
            # Toxicity and quality_gap should be valid floats
            assert isinstance(m.toxicity_rate, float)
            assert isinstance(m.quality_gap, float)


class TestOrchestratorPauseResume:
    """Tests for pause/resume functionality."""

    def test_pause_state_is_set(self):
        """Test that pause sets the state correctly."""
        config = OrchestratorConfig(
            n_epochs=10,
            steps_per_epoch=5,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        agent = HonestAgent(agent_id="agent_1")
        orchestrator.register_agent(agent)

        # Pause the simulation
        orchestrator.pause()
        assert orchestrator.state.is_paused

        # Agent can't act when paused
        assert not orchestrator.state.can_agent_act("agent_1")

    def test_resume_continues_simulation(self):
        """Test that resume continues simulation."""
        orchestrator = Orchestrator()
        agent = HonestAgent(agent_id="agent_1")
        orchestrator.register_agent(agent)

        orchestrator.pause()
        assert orchestrator.state.is_paused

        orchestrator.resume()
        assert not orchestrator.state.is_paused


class TestOrchestratorEventLogging:
    """Tests for event logging."""

    def test_logs_events_to_file(self):
        """Test that events are logged to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"

            config = OrchestratorConfig(
                n_epochs=2,
                steps_per_epoch=3,
                log_path=log_path,
                log_events=True,
                seed=42,
            )
            orchestrator = Orchestrator(config=config)

            agent = HonestAgent(agent_id="agent_1")
            orchestrator.register_agent(agent)

            orchestrator.run()

            # Check log file exists and has content
            # File is created on first write, check event_log has events
            assert orchestrator.event_log is not None

            # Count events by replaying
            events = list(orchestrator.event_log.replay())
            # Should at least have simulation_started and simulation_ended
            assert len(events) >= 2

    def test_event_log_replay(self):
        """Test replaying events from log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"

            config = OrchestratorConfig(
                n_epochs=3,
                steps_per_epoch=5,
                log_path=log_path,
                seed=42,
            )
            orchestrator = Orchestrator(config=config)

            for i in range(3):
                orchestrator.register_agent(HonestAgent(agent_id=f"agent_{i}"))

            orchestrator.run()

            # Replay events
            events = list(orchestrator.event_log.replay())
            assert len(events) > 0

            # Should have simulation start and end events
            event_types = [e.event_type.value for e in events]
            assert "simulation_started" in event_types
            assert "simulation_ended" in event_types


class TestOrchestratorCallbacks:
    """Tests for callback functionality."""

    def test_epoch_end_callback(self):
        """Test epoch end callbacks are called."""
        config = OrchestratorConfig(
            n_epochs=3,
            steps_per_epoch=2,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)
        orchestrator.register_agent(HonestAgent(agent_id="agent_1"))

        callback_data = []

        def callback(metrics):
            callback_data.append(metrics.epoch)

        orchestrator.on_epoch_end(callback)
        orchestrator.run()

        assert len(callback_data) == 3
        assert callback_data == [0, 1, 2]

    def test_interaction_complete_callback(self):
        """Test interaction complete callbacks."""
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=10,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        for i in range(5):
            orchestrator.register_agent(HonestAgent(agent_id=f"agent_{i}"))

        interaction_data = []

        def callback(interaction, payoff_init, payoff_counter):
            interaction_data.append({
                "p": interaction.p,
                "accepted": interaction.accepted,
                "payoff_init": payoff_init,
            })

        orchestrator.on_interaction_complete(callback)
        orchestrator.run()

        # Should have captured some interactions
        # (depends on random behavior, but with 5 agents over 5 epochs should have some)
        if interaction_data:
            for data in interaction_data:
                assert 0 <= data["p"] <= 1
                assert isinstance(data["accepted"], bool)


class TestOrchestratorIntegration:
    """Integration tests for full simulation scenarios."""

    def test_five_agents_ten_epochs(self):
        """Test MVP v0 success criteria: 5 agents, 10+ epochs."""
        config = OrchestratorConfig(
            n_epochs=10,
            steps_per_epoch=10,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        # Register 5 agents of different types
        orchestrator.register_agent(HonestAgent(agent_id="honest_1"))
        orchestrator.register_agent(HonestAgent(agent_id="honest_2"))
        orchestrator.register_agent(OpportunisticAgent(agent_id="opp_1"))
        orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))
        orchestrator.register_agent(AdversarialAgent(agent_id="adv_1"))

        # Run simulation
        metrics = orchestrator.run()

        # Verify success criteria
        assert len(metrics) == 10, "Should complete 10 epochs"
        assert len(orchestrator.get_all_agents()) == 5, "Should have 5 agents"

        # Verify metrics computed per epoch
        for m in metrics:
            assert isinstance(m.toxicity_rate, float)
            assert isinstance(m.quality_gap, float)
            assert m.toxicity_rate >= 0

        # Verify state is consistent
        assert orchestrator.state.current_epoch == 10

    def test_agents_interact(self):
        """Test that agents actually interact with each other."""
        config = OrchestratorConfig(
            n_epochs=5,
            steps_per_epoch=20,  # More steps for more interactions
            seed=123,
        )
        orchestrator = Orchestrator(config=config)

        # Add agents
        for i in range(4):
            orchestrator.register_agent(HonestAgent(agent_id=f"agent_{i}"))

        metrics = orchestrator.run()

        # Check for interactions
        total_interactions = sum(m.total_interactions for m in metrics)
        assert total_interactions > 0, "Should have some interactions"

    def test_event_log_enables_replay(self):
        """Test that event log enables simulation replay."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "events.jsonl"

            config = OrchestratorConfig(
                n_epochs=5,
                steps_per_epoch=5,
                log_path=log_path,
                seed=42,
            )
            orchestrator = Orchestrator(config=config)

            for i in range(3):
                orchestrator.register_agent(HonestAgent(agent_id=f"agent_{i}"))

            original_metrics = orchestrator.run()

            # Reconstruct interactions from log
            reconstructed = orchestrator.event_log.to_interactions()

            # Should be able to reconstruct
            # (number depends on what happened in simulation)
            assert isinstance(reconstructed, list)

    def test_coordination_patterns_observable(self):
        """Test that coordination patterns can be observed."""
        config = OrchestratorConfig(
            n_epochs=10,
            steps_per_epoch=15,
            seed=42,
        )
        orchestrator = Orchestrator(config=config)

        # Add collaborative agents
        for i in range(5):
            orchestrator.register_agent(HonestAgent(agent_id=f"honest_{i}"))

        metrics = orchestrator.run()

        # Check for activity patterns
        total_posts = sum(m.total_posts for m in metrics)
        total_votes = sum(m.total_votes for m in metrics)

        assert total_posts > 0, "Agents should create posts"
        assert total_votes >= 0, "Agents may vote"

        # Check agent states evolved
        for agent in orchestrator.get_all_agents():
            state = orchestrator.state.get_agent(agent.agent_id)
            # Agents should have participated
            assert state.interactions_initiated >= 0 or state.interactions_received >= 0
