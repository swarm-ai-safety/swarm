"""Tests for cross-run memory: loading/saving trust priors across simulation runs.

This module tests the full lifecycle of cross-run memory:
1. Agents load prior snapshots before network initialization
2. Trust priors (counterparty_memory) are initialized from prior snapshots
3. Interaction history is NOT loaded (ephemeral per run)
4. Snapshots are saved at run end with current trust state
5. Round-trip: save from run A → load into run B → verify trust priors carry
"""

from pathlib import Path
from typing import List

import pytest

from swarm.agents.base import BaseAgent
from swarm.agents.honest import HonestAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.knowledge.graph_memory import (
    AgentMemorySnapshot,
    GraphMemoryStore,
    RelationshipEdge,
)


class TestGraphMemoryStore:
    """Tests for GraphMemoryStore basic operations."""

    def test_save_and_load_snapshot(self, tmp_path: Path) -> None:
        """Test saving and loading a single agent's memory snapshot."""
        store_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(store_path))

        # Create and save a snapshot
        snapshot = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 0.8, "agent_3": 0.3},
            interaction_summaries=[],
            total_interactions=5,
            total_payoff=2.5,
            run_id="run_1",
            epoch=9,
            timestamp="2026-01-01T00:00:00",
        )
        store.save("agent_1", snapshot)

        # Load and verify
        loaded = store.load("agent_1")
        assert loaded is not None
        assert loaded.agent_id == "agent_1"
        assert loaded.agent_type == "honest"
        assert loaded.counterparty_trust == {"agent_2": 0.8, "agent_3": 0.3}
        assert loaded.total_interactions == 5
        assert loaded.total_payoff == 2.5

    def test_load_most_recent_snapshot(self, tmp_path: Path) -> None:
        """Test that load() returns the most recent snapshot by default."""
        store_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(store_path))

        # Save multiple snapshots for same agent
        snap1 = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 0.5},
            interaction_summaries=[],
            total_interactions=1,
            total_payoff=0.5,
            run_id="run_1",
            epoch=0,
            timestamp="2026-01-01T00:00:00",
        )
        snap2 = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 0.9},  # Updated trust
            interaction_summaries=[],
            total_interactions=5,
            total_payoff=2.5,
            run_id="run_2",
            epoch=4,
            timestamp="2026-01-02T00:00:00",
        )

        store.save("agent_1", snap1)
        store.save("agent_1", snap2)

        # Load most recent (index -1, the default)
        latest = store.load("agent_1")
        assert latest is not None
        assert latest.counterparty_trust["agent_2"] == 0.9
        assert latest.run_id == "run_2"

    def test_load_nonexistent_snapshot(self, tmp_path: Path) -> None:
        """Test that load() returns None for nonexistent agent."""
        store_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(store_path))

        loaded = store.load("nonexistent_agent")
        assert loaded is None

    def test_relationship_edge_validation(self, tmp_path: Path) -> None:
        """Test that trust scores in relationships are validated."""
        store_path = tmp_path / "memory.json"
        GraphMemoryStore(str(store_path))

        # Attempt to add relationship with invalid trust
        with pytest.raises(ValueError):
            edge = RelationshipEdge(
                agent_a="agent_1",
                agent_b="agent_2",
                trust_a_to_b=1.5,  # Out of bounds
            )
            edge.validate()

    def test_save_all_multiple_agents(self, tmp_path: Path) -> None:
        """Test bulk saving of snapshots for multiple agents."""
        # This requires BaseAgent instances, so we'll test through the helper
        store_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(store_path))

        snapshots = [
            AgentMemorySnapshot(
                agent_id=f"agent_{i}",
                agent_type="honest",
                counterparty_trust={},
                interaction_summaries=[],
                total_interactions=0,
                total_payoff=0.0,
                run_id="run_1",
                epoch=9,
                timestamp="2026-01-01T00:00:00",
            )
            for i in range(3)
        ]

        for snap in snapshots:
            store.save(snap.agent_id, snap)

        # Load all
        all_snaps = store.load_all()
        assert len(all_snaps) == 3
        for i in range(3):
            assert f"agent_{i}" in all_snaps

    def test_metadata_tracking(self, tmp_path: Path) -> None:
        """Test that metadata is tracked (creation time, last update)."""
        store_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(store_path))

        metadata = store.get_metadata()
        assert "created" in metadata
        assert "last_updated" in metadata

        # Save something and check last_updated changes
        snapshot = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={},
            interaction_summaries=[],
            total_interactions=0,
            total_payoff=0.0,
            run_id="run_1",
            epoch=0,
            timestamp="2026-01-01T00:00:00",
        )
        store.save("agent_1", snapshot)

        metadata_after = store.get_metadata()
        assert metadata_after["last_updated"] >= metadata["last_updated"]


class TestAgentLoadPriorMemory:
    """Tests for BaseAgent.load_prior_memory() method."""

    def test_load_prior_memory_updates_trust(self) -> None:
        """Test that load_prior_memory updates _counterparty_memory."""
        agent = HonestAgent(agent_id="agent_1", rng=None)

        # Initial state: neutral trust (0.5) for unknown agents
        assert agent._counterparty_memory.get("agent_2", 0.5) == 0.5

        # Create prior snapshot with specific trust values
        snapshot = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 0.8, "agent_3": 0.2},
            interaction_summaries=[],
            total_interactions=10,
            total_payoff=3.0,
            run_id="run_1",
            epoch=9,
            timestamp="2026-01-01T00:00:00",
        )

        agent.load_prior_memory(snapshot)

        # Verify trust was loaded
        assert agent._counterparty_memory["agent_2"] == 0.8
        assert agent._counterparty_memory["agent_3"] == 0.2

    def test_load_prior_memory_does_not_restore_history(self) -> None:
        """Test that load_prior_memory does NOT restore interaction history."""
        agent = HonestAgent(agent_id="agent_1", rng=None)

        # Verify initial history is empty
        initial_history_len = len(agent._interaction_history)

        # Create snapshot with interaction summaries (which should be ignored)
        snapshot = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 0.8},
            interaction_summaries=[
                {
                    "interaction_id": "int_1",
                    "counterparty": "agent_2",
                    "p": 0.9,
                    "accepted": True,
                    "type": "soft",
                }
            ],
            total_interactions=10,
            total_payoff=3.0,
            run_id="run_1",
            epoch=9,
            timestamp="2026-01-01T00:00:00",
        )

        agent.load_prior_memory(snapshot)

        # Verify history was NOT restored
        assert len(agent._interaction_history) == initial_history_len

    def test_load_prior_memory_type_validation(self) -> None:
        """Test that load_prior_memory validates snapshot type."""
        agent = HonestAgent(agent_id="agent_1", rng=None)

        # Attempt to load with invalid type
        with pytest.raises(TypeError):
            agent.load_prior_memory({"agent_id": "agent_1"})  # type: ignore

    def test_load_prior_memory_validates_snapshot(self) -> None:
        """Test that load_prior_memory validates the snapshot before loading."""
        agent = HonestAgent(agent_id="agent_1", rng=None)

        # Create invalid snapshot (trust out of bounds)
        snapshot = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 1.5},  # Invalid!
            interaction_summaries=[],
            total_interactions=0,
            total_payoff=0.0,
            run_id="run_1",
            epoch=0,
            timestamp="2026-01-01T00:00:00",
        )

        # Should raise ValueError during validation
        with pytest.raises(ValueError):
            agent.load_prior_memory(snapshot)


class TestOrchestratorMemoryBoundary:
    """Tests for Orchestrator loading/saving memory at run boundaries."""

    @pytest.fixture
    def basic_config(self, tmp_path: Path) -> OrchestratorConfig:
        """Create a minimal OrchestratorConfig for testing."""
        memory_path = tmp_path / "memory.json"
        return OrchestratorConfig(
            scenario_id="test_scenario",
            n_epochs=2,
            steps_per_epoch=2,
            seed=42,
            run_id="test_run_1",
            graph_memory_path=str(memory_path),
        )

    def test_orchestrator_initializes_graph_memory(
        self, basic_config: OrchestratorConfig
    ) -> None:
        """Test that Orchestrator initializes GraphMemoryStore when path is provided."""
        orch = Orchestrator(basic_config)
        assert orch._graph_memory is not None
        assert isinstance(orch._graph_memory, GraphMemoryStore)

    def test_orchestrator_no_memory_when_path_none(
        self, tmp_path: Path
    ) -> None:
        """Test that Orchestrator skips memory when graph_memory_path is None."""
        config = OrchestratorConfig(
            scenario_id="test_scenario",
            n_epochs=1,
            steps_per_epoch=1,
            seed=42,
            run_id="test_run_1",
            graph_memory_path=None,
        )
        orch = Orchestrator(config)
        assert orch._graph_memory is None

    def test_orchestrator_loads_prior_at_start(
        self, tmp_path: Path
    ) -> None:
        """Test that Orchestrator loads prior snapshots before network init.

        We verify this by:
        1. Create memory store with prior snapshots
        2. Create orchestrator with same memory path
        3. Create agents manually and verify they loaded the trust priors
        """
        memory_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(memory_path))

        # Pre-populate store with prior snapshots
        prior_snap_1 = AgentMemorySnapshot(
            agent_id="agent_1",
            agent_type="honest",
            counterparty_trust={"agent_2": 0.9},
            interaction_summaries=[],
            total_interactions=5,
            total_payoff=2.0,
            run_id="prior_run",
            epoch=9,
            timestamp="2026-01-01T00:00:00",
        )
        prior_snap_2 = AgentMemorySnapshot(
            agent_id="agent_2",
            agent_type="honest",
            counterparty_trust={"agent_1": 0.8},
            interaction_summaries=[],
            total_interactions=4,
            total_payoff=1.5,
            run_id="prior_run",
            epoch=9,
            timestamp="2026-01-01T00:00:00",
        )
        store.save("agent_1", prior_snap_1)
        store.save("agent_2", prior_snap_2)

        # Create orchestrator with same memory path
        config = OrchestratorConfig(
            scenario_id="test_scenario",
            n_epochs=1,
            steps_per_epoch=1,
            seed=42,
            run_id="new_run",
            graph_memory_path=str(memory_path),
        )
        orch = Orchestrator(config)

        # Manually create and register agents (simplified)
        agent1 = HonestAgent(agent_id="agent_1", rng=None)
        agent2 = HonestAgent(agent_id="agent_2", rng=None)
        orch._agents = [agent1, agent2]

        # Simulate the memory loading that happens in run()
        # (We're testing the logic directly since we can't easily call run() with minimal setup)
        if orch._graph_memory is not None:
            prior_snapshots = orch._graph_memory.load_all()
            for agent in orch._agents:
                if agent.agent_id in prior_snapshots:
                    snapshot = prior_snapshots[agent.agent_id]
                    agent.load_prior_memory(snapshot)

        # Verify trust priors were loaded
        assert agent1._counterparty_memory["agent_2"] == 0.9
        assert agent2._counterparty_memory["agent_1"] == 0.8

    def test_orchestrator_saves_snapshots_at_end(
        self, tmp_path: Path
    ) -> None:
        """Test that Orchestrator saves memory snapshots at run end.

        We verify this by:
        1. Create agents with modified trust state
        2. Simulate saving (using the save_all pattern)
        3. Verify snapshots are persisted to disk
        """
        memory_path = tmp_path / "memory.json"
        store = GraphMemoryStore(str(memory_path))

        # Create agents with specific trust state
        agents: List[BaseAgent] = [
            HonestAgent(agent_id="agent_1", rng=None),
            HonestAgent(agent_id="agent_2", rng=None),
        ]

        # Simulate agents learning trust (manual update for testing)
        agents[0]._counterparty_memory["agent_2"] = 0.75
        agents[1]._counterparty_memory["agent_1"] = 0.85

        # Save using the pattern from orchestrator
        run_id = "test_run"
        final_epoch = 9
        store.save_all(agents, run_id, final_epoch)

        # Verify persistence by creating new store instance
        store2 = GraphMemoryStore(str(memory_path))
        loaded_snap_1 = store2.load("agent_1")
        loaded_snap_2 = store2.load("agent_2")

        assert loaded_snap_1 is not None
        assert loaded_snap_2 is not None
        assert loaded_snap_1.counterparty_trust["agent_2"] == 0.75
        assert loaded_snap_2.counterparty_trust["agent_1"] == 0.85
        assert loaded_snap_1.run_id == "test_run"
        assert loaded_snap_2.epoch == 9


class TestRoundTripMemory:
    """Tests for full round-trip: save in run A, load in run B."""

    def test_round_trip_trust_priors(self, tmp_path: Path) -> None:
        """Test complete round-trip: save → load → verify trust carries over."""
        memory_path = tmp_path / "memory.json"

        # --- Run A: Save snapshots ---
        run_a_agents = [
            HonestAgent(agent_id="agent_1", rng=None),
            HonestAgent(agent_id="agent_2", rng=None),
        ]

        # Simulate learning in run A
        run_a_agents[0]._counterparty_memory["agent_2"] = 0.7
        run_a_agents[1]._counterparty_memory["agent_1"] = 0.6

        # Save run A snapshots
        store_a = GraphMemoryStore(str(memory_path))
        store_a.save_all(run_a_agents, run_id="run_a", epoch=9)

        # --- Run B: Load and verify ---
        run_b_agents = [
            HonestAgent(agent_id="agent_1", rng=None),
            HonestAgent(agent_id="agent_2", rng=None),
        ]

        # Initially, agents have neutral trust
        assert run_b_agents[0]._counterparty_memory.get("agent_2", 0.5) == 0.5
        assert run_b_agents[1]._counterparty_memory.get("agent_1", 0.5) == 0.5

        # Load prior snapshots
        store_b = GraphMemoryStore(str(memory_path))
        prior_snapshots = store_b.load_all()

        for agent in run_b_agents:
            if agent.agent_id in prior_snapshots:
                snapshot = prior_snapshots[agent.agent_id]
                agent.load_prior_memory(snapshot)

        # Verify trust carried over from run A to run B
        assert run_b_agents[0]._counterparty_memory["agent_2"] == 0.7
        assert run_b_agents[1]._counterparty_memory["agent_1"] == 0.6

    def test_round_trip_multiple_runs(self, tmp_path: Path) -> None:
        """Test round-trip across three runs: A → B → C."""
        memory_path = tmp_path / "memory.json"

        # --- Run A ---
        run_a_agents = [
            HonestAgent(agent_id="agent_1", rng=None),
        ]
        run_a_agents[0]._counterparty_memory["agent_2"] = 0.5

        store_a = GraphMemoryStore(str(memory_path))
        store_a.save_all(run_a_agents, run_id="run_a", epoch=0)

        # --- Run B ---
        run_b_agents = [
            HonestAgent(agent_id="agent_1", rng=None),
        ]
        store_b = GraphMemoryStore(str(memory_path))
        prior_b = store_b.load_all()
        if "agent_1" in prior_b:
            run_b_agents[0].load_prior_memory(prior_b["agent_1"])

        # Verify load from A
        assert run_b_agents[0]._counterparty_memory["agent_2"] == 0.5

        # Learn more in run B
        run_b_agents[0]._counterparty_memory["agent_2"] = 0.8
        store_b.save_all(run_b_agents, run_id="run_b", epoch=0)

        # --- Run C ---
        run_c_agents = [
            HonestAgent(agent_id="agent_1", rng=None),
        ]
        store_c = GraphMemoryStore(str(memory_path))
        prior_c = store_c.load_all()
        if "agent_1" in prior_c:
            run_c_agents[0].load_prior_memory(prior_c["agent_1"])

        # Verify most recent (from B) was loaded
        assert run_c_agents[0]._counterparty_memory["agent_2"] == 0.8
