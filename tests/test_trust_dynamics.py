"""Tests for swarm/analysis/trust_dynamics.py."""

import tempfile
from datetime import datetime
from pathlib import Path

from swarm.analysis.trust_dynamics import TrustDynamicsAnalyzer
from swarm.knowledge.graph_memory import AgentMemorySnapshot, GraphMemoryStore


class TestGetTrustTimeline:
    """Test trust timeline extraction."""

    def test_empty_snapshots(self):
        """Timeline should be empty when agent has no snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            timeline = analyzer.get_trust_timeline("agent1", "agent2")
            assert timeline == []

    def test_single_snapshot(self):
        """Extract trust from single snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Create and save a snapshot
            snapshot = AgentMemorySnapshot(
                agent_id="agent1",
                agent_type="honest",
                counterparty_trust={"agent2": 0.7},
                epoch=0,
                timestamp=datetime.now().isoformat(),
            )
            store.save("agent1", snapshot)

            timeline = analyzer.get_trust_timeline("agent1", "agent2")
            assert timeline == [(0, 0.7)]

    def test_multiple_snapshots_sorted(self):
        """Timeline should be sorted by epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Create snapshots in non-sequential order
            snapshots = [
                AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": 0.5},
                    epoch=2,
                    timestamp=datetime.now().isoformat(),
                ),
                AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": 0.6},
                    epoch=0,
                    timestamp=datetime.now().isoformat(),
                ),
                AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": 0.8},
                    epoch=1,
                    timestamp=datetime.now().isoformat(),
                ),
            ]

            for snapshot in snapshots:
                store.save("agent1", snapshot)

            timeline = analyzer.get_trust_timeline("agent1", "agent2")
            # Should be sorted by epoch
            assert timeline == [(0, 0.6), (1, 0.8), (2, 0.5)]

    def test_missing_counterparty(self):
        """Timeline should skip epochs where counterparty is not in trust dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            snapshots = [
                AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": 0.5},
                    epoch=0,
                    timestamp=datetime.now().isoformat(),
                ),
                AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent3": 0.7},  # No agent2
                    epoch=1,
                    timestamp=datetime.now().isoformat(),
                ),
                AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": 0.8},
                    epoch=2,
                    timestamp=datetime.now().isoformat(),
                ),
            ]

            for snapshot in snapshots:
                store.save("agent1", snapshot)

            timeline = analyzer.get_trust_timeline("agent1", "agent2")
            # Should skip epoch 1
            assert timeline == [(0, 0.5), (2, 0.8)]


class TestComputeConvergenceRate:
    """Test convergence rate calculation."""

    def test_insufficient_snapshots(self):
        """Convergence should be 0.0 with fewer than 4 snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Add only 2 snapshots
            for epoch in range(2):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": 0.5 + epoch * 0.1},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            convergence = analyzer.compute_convergence_rate("agent1")
            assert convergence == 0.0

    def test_fast_converging_agent(self):
        """Fast convergence: small deltas in second half, large in first half."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Create pattern: large swings, then stable
            trust_values = [0.2, 0.8, 0.5, 0.55, 0.56, 0.57]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            convergence = analyzer.compute_convergence_rate("agent1")
            # Second half should have much lower variance than first half
            assert convergence > 0.5

    def test_volatile_agent(self):
        """Volatile agent: increasing volatility (divergence)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Create pattern: small swings then large swings (divergence)
            trust_values = [0.49, 0.51, 0.48, 0.52, 0.2, 0.8]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            convergence = analyzer.compute_convergence_rate("agent1")
            # Should be low or negative - volatility increases
            assert convergence < 0.3

    def test_already_converged(self):
        """Already converged: zero variance in first half."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # No change in first half, no change in second half -> convergence = 1.0
            trust_values = [0.5, 0.5, 0.5, 0.5]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            convergence = analyzer.compute_convergence_rate("agent1")
            assert convergence == 1.0


class TestComputeTrustDecayRate:
    """Test decay rate (linear regression slope)."""

    def test_insufficient_snapshots(self):
        """Decay rate should be 0.0 with fewer than 2 snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            snapshot = AgentMemorySnapshot(
                agent_id="agent1",
                agent_type="honest",
                counterparty_trust={"agent2": 0.5},
                epoch=0,
                timestamp=datetime.now().isoformat(),
            )
            store.save("agent1", snapshot)

            decay = analyzer.compute_trust_decay_rate("agent1")
            assert decay == 0.0

    def test_decaying_trust(self):
        """Decaying trust: negative slope."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Trust decreases over epochs
            trust_values = [0.9, 0.8, 0.7, 0.6, 0.5]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust, "agent3": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            decay = analyzer.compute_trust_decay_rate("agent1")
            # Slope should be negative
            assert decay < 0.0

    def test_growing_trust(self):
        """Growing trust: positive slope."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Trust increases over epochs
            trust_values = [0.1, 0.2, 0.3, 0.4, 0.5]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust, "agent3": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            decay = analyzer.compute_trust_decay_rate("agent1")
            # Slope should be positive
            assert decay > 0.0

    def test_stable_trust(self):
        """Stable trust: near-zero slope."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Trust stays constant
            trust_values = [0.5, 0.5, 0.5, 0.5]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            decay = analyzer.compute_trust_decay_rate("agent1")
            # Slope should be very close to 0
            assert abs(decay) < 1e-10


class TestGetSystemDynamicsSummary:
    """Test system-wide dynamics summary."""

    def test_empty_store(self):
        """Summary for empty store should have zero values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            summary = analyzer.get_system_dynamics_summary()

            assert summary["mean_convergence_rate"] == 0.0
            assert summary["median_convergence_rate"] == 0.0
            assert summary["mean_decay_rate"] == 0.0
            assert summary["agents_converged_count"] == 0
            assert summary["agents_volatile_count"] == 0
            assert summary["trust_stability_index"] == 0.0

    def test_multiple_agents_summary(self):
        """Summary should aggregate metrics across agents."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Agent 1: converging
            for epoch in range(6):
                trust = [0.2, 0.8, 0.5, 0.55, 0.56, 0.57][epoch]
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            # Agent 2: volatile
            for epoch in range(6):
                trust = [0.3, 0.7, 0.4, 0.8, 0.35, 0.75][epoch]
                snapshot = AgentMemorySnapshot(
                    agent_id="agent2",
                    agent_type="opportunistic",
                    counterparty_trust={"agent1": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent2", snapshot)

            summary = analyzer.get_system_dynamics_summary()

            # Should have summaries with reasonable values
            assert "mean_convergence_rate" in summary
            assert "agents_converged_count" in summary
            assert summary["agents_converged_count"] >= 0

    def test_summary_has_all_keys(self):
        """Summary should have all required keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Add at least one agent with snapshots
            snapshot = AgentMemorySnapshot(
                agent_id="agent1",
                agent_type="honest",
                counterparty_trust={"agent2": 0.5},
                epoch=0,
                timestamp=datetime.now().isoformat(),
            )
            store.save("agent1", snapshot)

            summary = analyzer.get_system_dynamics_summary()

            expected_keys = {
                "mean_convergence_rate",
                "median_convergence_rate",
                "mean_decay_rate",
                "agents_converged_count",
                "agents_volatile_count",
                "trust_stability_index",
            }
            assert set(summary.keys()) == expected_keys


class TestDetectTrustShocks:
    """Test trust shock detection."""

    def test_no_shocks(self):
        """No shocks when changes are below threshold."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Small changes
            for epoch in range(3):
                trust = 0.5 + epoch * 0.05
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            shocks = analyzer.detect_trust_shocks("agent1", threshold=0.2)
            assert shocks == []

    def test_single_shock(self):
        """Detect a single trust shock."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Stable, then big drop
            trust_values = [0.5, 0.5, 0.1]
            for epoch, trust in enumerate(trust_values):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust={"agent2": trust},
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            shocks = analyzer.detect_trust_shocks("agent1", threshold=0.2)

            assert len(shocks) == 1
            shock = shocks[0]
            assert shock["epoch"] == 2
            assert shock["counterparty"] == "agent2"
            assert shock["delta"] == 0.4
            assert shock["trust_before"] == 0.5
            assert shock["trust_after"] == 0.1

    def test_multiple_shocks(self):
        """Detect multiple shocks across counterparties."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Multiple shocks in different counterparties
            snapshots_data = [
                {"agent2": 0.5, "agent3": 0.6},
                {"agent2": 0.1, "agent3": 0.9},  # Both shock
                {"agent2": 0.1, "agent3": 0.9},
            ]

            for epoch, cp_trust in enumerate(snapshots_data):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust=cp_trust,
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            shocks = analyzer.detect_trust_shocks("agent1", threshold=0.2)

            # Should detect shocks for both agent2 and agent3 at epoch 1
            assert len(shocks) == 2
            assert all(s["epoch"] == 1 for s in shocks)
            counterparties = {s["counterparty"] for s in shocks}
            assert counterparties == {"agent2", "agent3"}

    def test_shocks_sorted_by_epoch(self):
        """Shocks should be sorted by epoch."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            # Shocks at different epochs
            snapshots_data = [
                {"agent2": 0.5},
                {"agent2": 0.1},  # Shock at epoch 1
                {"agent2": 0.7},  # Shock at epoch 2
                {"agent2": 0.8},
            ]

            for epoch, cp_trust in enumerate(snapshots_data):
                snapshot = AgentMemorySnapshot(
                    agent_id="agent1",
                    agent_type="honest",
                    counterparty_trust=cp_trust,
                    epoch=epoch,
                    timestamp=datetime.now().isoformat(),
                )
                store.save("agent1", snapshot)

            shocks = analyzer.detect_trust_shocks("agent1", threshold=0.2)

            assert len(shocks) == 2
            epochs = [s["epoch"] for s in shocks]
            assert epochs == sorted(epochs)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_counterparty_trust(self):
        """Handle agents with empty counterparty_trust dicts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            snapshot = AgentMemorySnapshot(
                agent_id="agent1",
                agent_type="honest",
                counterparty_trust={},
                epoch=0,
                timestamp=datetime.now().isoformat(),
            )
            store.save("agent1", snapshot)

            # Should not crash
            timeline = analyzer.get_trust_timeline("agent1", "agent2")
            assert timeline == []

            convergence = analyzer.compute_convergence_rate("agent1")
            assert convergence == 0.0

            decay = analyzer.compute_trust_decay_rate("agent1")
            assert decay == 0.0

    def test_single_snapshot_per_agent(self):
        """System summary with agents having single snapshot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            for agent_id in ["agent1", "agent2"]:
                snapshot = AgentMemorySnapshot(
                    agent_id=agent_id,
                    agent_type="honest",
                    counterparty_trust={"agent3": 0.5},
                    epoch=0,
                    timestamp=datetime.now().isoformat(),
                )
                store.save(agent_id, snapshot)

            summary = analyzer.get_system_dynamics_summary()
            # Should not crash and return valid metrics
            assert "mean_convergence_rate" in summary
            assert "mean_decay_rate" in summary

    def test_nonexistent_agent(self):
        """Queries for nonexistent agent should return empty/zero results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = GraphMemoryStore(str(Path(tmpdir) / "memory.json"))
            analyzer = TrustDynamicsAnalyzer(store)

            timeline = analyzer.get_trust_timeline("nonexistent", "other")
            assert timeline == []

            convergence = analyzer.compute_convergence_rate("nonexistent")
            assert convergence == 0.0

            decay = analyzer.compute_trust_decay_rate("nonexistent")
            assert decay == 0.0

            shocks = analyzer.detect_trust_shocks("nonexistent")
            assert shocks == []
