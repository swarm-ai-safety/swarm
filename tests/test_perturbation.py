"""Tests for PerturbationEngine."""

from unittest.mock import MagicMock

import pytest

from swarm.core.perturbation import (
    AgentDropoutConfig,
    CorruptionMode,
    NetworkPartitionConfig,
    ParameterShocksConfig,
    ParameterShockSpec,
    PartitionMode,
    PerturbationConfig,
    PerturbationEngine,
    ResourceShockConfig,
    ResourceShockMode,
    ScheduleSpec,
    ShockTrigger,
    SignalCorruptionConfig,
)
from swarm.core.proxy import ProxyObservables
from swarm.models.agent import AgentState, AgentType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(agent_specs=None):
    """Build a minimal mock EnvState with agents dict."""
    state = MagicMock()
    agents = {}
    for spec in (agent_specs or []):
        a = AgentState(
            agent_id=spec["id"],
            agent_type=AgentType(spec.get("type", "honest")),
            resources=spec.get("resources", 100.0),
        )
        agents[spec["id"]] = a
    state.agents = agents
    return state


def _make_network(agent_ids, edges=None):
    """Build a mock network with get_neighbors, add_edge, remove_edge."""
    adjacency = {a: {} for a in agent_ids}
    for a, b, w in (edges or []):
        adjacency[a][b] = w
        adjacency[b][a] = w

    net = MagicMock()

    def get_neighbors(a):
        return dict(adjacency.get(a, {}))

    def remove_edge(a, b):
        adjacency.get(a, {}).pop(b, None)
        adjacency.get(b, {}).pop(a, None)

    def add_edge(a, b, weight=1.0):
        if a not in adjacency:
            adjacency[a] = {}
        if b not in adjacency:
            adjacency[b] = {}
        adjacency[a][b] = weight
        adjacency[b][a] = weight

    net.get_neighbors = get_neighbors
    net.remove_edge = remove_edge
    net.add_edge = add_edge
    net._adjacency = adjacency  # for assertions
    return net


def _make_governance(**kwargs):
    """Build a mock governance engine with a config object."""
    gov = MagicMock()
    config = MagicMock()
    for k, v in kwargs.items():
        setattr(config, k, v)
    gov.config = config
    return gov


# ===========================================================================
# TestPerturbationConfig
# ===========================================================================


class TestPerturbationConfig:
    def test_defaults_all_disabled(self):
        cfg = PerturbationConfig()
        assert not cfg.agent_dropout.enabled
        assert not cfg.signal_corruption.enabled
        assert not cfg.parameter_shocks.enabled
        assert not cfg.network_partition.enabled
        assert not cfg.resource_shock.enabled

    def test_seed_propagates(self):
        cfg = PerturbationConfig(seed=42)
        engine = PerturbationEngine(cfg)
        # Engine RNG should be deterministic
        v1 = engine._rng.random()
        engine2 = PerturbationEngine(PerturbationConfig(seed=42))
        v2 = engine2._rng.random()
        assert v1 == v2


# ===========================================================================
# TestAgentDropout
# ===========================================================================


class TestAgentDropout:
    def test_dropout_probability(self):
        cfg = PerturbationConfig(
            seed=42,
            agent_dropout=AgentDropoutConfig(
                enabled=True,
                probability_per_step=1.0,  # always drop
                min_duration_steps=1,
                max_duration_steps=1,
            ),
        )
        state = _make_state([{"id": "a1"}, {"id": "a2"}])
        engine = PerturbationEngine(cfg, state=state)
        engine.on_step_start(0, 0)
        assert engine.get_dropped_agents() == {"a1", "a2"}

    def test_no_dropout_when_disabled(self):
        cfg = PerturbationConfig(
            agent_dropout=AgentDropoutConfig(enabled=False)
        )
        state = _make_state([{"id": "a1"}])
        engine = PerturbationEngine(cfg, state=state)
        engine.on_step_start(0, 0)
        assert engine.get_dropped_agents() == set()

    def test_exempt_types(self):
        cfg = PerturbationConfig(
            seed=0,
            agent_dropout=AgentDropoutConfig(
                enabled=True,
                probability_per_step=1.0,
                min_duration_steps=1,
                max_duration_steps=1,
                exempt_types=["honest"],
            ),
        )
        state = _make_state([
            {"id": "a1", "type": "honest"},
            {"id": "a2", "type": "adversarial"},
        ])
        engine = PerturbationEngine(cfg, state=state)
        engine.on_step_start(0, 0)
        dropped = engine.get_dropped_agents()
        assert "a1" not in dropped
        assert "a2" in dropped

    def test_duration_bounds(self):
        cfg = PerturbationConfig(
            seed=42,
            agent_dropout=AgentDropoutConfig(
                enabled=True,
                probability_per_step=1.0,
                min_duration_steps=2,
                max_duration_steps=2,
            ),
        )
        state = _make_state([{"id": "a1"}])
        engine = PerturbationEngine(cfg, state=state)
        # Step 0: roll dropout (duration=2)
        engine.on_step_start(0, 0)
        assert "a1" in engine.get_dropped_agents()
        # Step 1: tick (2->1), still dropped
        engine.on_step_start(0, 1)
        assert "a1" in engine.get_dropped_agents()
        # Step 2: tick (1->0, removed), but re-rolls because prob=1.0
        # To test expiry, disable further rolls
        engine.config.agent_dropout.probability_per_step = 0.0
        engine.on_step_start(0, 2)
        assert "a1" not in engine.get_dropped_agents()

    def test_deterministic_with_seed(self):
        cfg = PerturbationConfig(
            seed=99,
            agent_dropout=AgentDropoutConfig(
                enabled=True,
                probability_per_step=0.5,
                min_duration_steps=1,
                max_duration_steps=3,
            ),
        )
        agents = [{"id": f"a{i}"} for i in range(10)]

        engine1 = PerturbationEngine(cfg, state=_make_state(agents))
        engine1.on_step_start(0, 0)
        dropped1 = engine1.get_dropped_agents()

        engine2 = PerturbationEngine(
            PerturbationConfig(
                seed=99,
                agent_dropout=AgentDropoutConfig(
                    enabled=True,
                    probability_per_step=0.5,
                    min_duration_steps=1,
                    max_duration_steps=3,
                ),
            ),
            state=_make_state(agents),
        )
        engine2.on_step_start(0, 0)
        dropped2 = engine2.get_dropped_agents()
        assert dropped1 == dropped2


# ===========================================================================
# TestSignalCorruption
# ===========================================================================


class TestSignalCorruption:
    def test_zero_out(self):
        cfg = PerturbationConfig(
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                targets=["task_progress_delta"],
                mode=CorruptionMode.ZERO_OUT,
            )
        )
        engine = PerturbationEngine(cfg)
        obs = ProxyObservables(task_progress_delta=0.8, rework_count=2)
        result = engine.perturb_observables(obs)
        assert result.task_progress_delta == 0.0
        assert result.rework_count == 2  # not targeted

    def test_invert(self):
        cfg = PerturbationConfig(
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                targets=["task_progress_delta", "counterparty_engagement_delta"],
                mode=CorruptionMode.INVERT,
            )
        )
        engine = PerturbationEngine(cfg)
        obs = ProxyObservables(
            task_progress_delta=0.5, counterparty_engagement_delta=-0.3
        )
        result = engine.perturb_observables(obs)
        assert result.task_progress_delta == -0.5
        assert result.counterparty_engagement_delta == pytest.approx(0.3)

    def test_random(self):
        cfg = PerturbationConfig(
            seed=42,
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                targets=["task_progress_delta"],
                mode=CorruptionMode.RANDOM,
            ),
        )
        engine = PerturbationEngine(cfg)
        obs = ProxyObservables(task_progress_delta=0.5)
        result = engine.perturb_observables(obs)
        assert result.task_progress_delta != 0.5  # very unlikely same
        assert -1.0 <= result.task_progress_delta <= 1.0

    def test_sticky(self):
        cfg = PerturbationConfig(
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                targets=["task_progress_delta"],
                mode=CorruptionMode.STICKY,
            )
        )
        engine = PerturbationEngine(cfg)
        obs1 = ProxyObservables(task_progress_delta=0.7)
        r1 = engine.perturb_observables(obs1)
        assert r1.task_progress_delta == 0.7  # first call captures

        obs2 = ProxyObservables(task_progress_delta=0.2)
        r2 = engine.perturb_observables(obs2)
        assert r2.task_progress_delta == 0.7  # stuck at first value

    def test_disabled(self):
        cfg = PerturbationConfig(
            signal_corruption=SignalCorruptionConfig(enabled=False)
        )
        engine = PerturbationEngine(cfg)
        obs = ProxyObservables(task_progress_delta=0.5)
        result = engine.perturb_observables(obs)
        assert result.task_progress_delta == 0.5

    def test_schedule_activation(self):
        cfg = PerturbationConfig(
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                schedule=[ScheduleSpec(start_epoch=2, end_epoch=4)],
            )
        )
        engine = PerturbationEngine(cfg)
        assert not engine.is_signal_corruption_active(0)
        assert not engine.is_signal_corruption_active(1)
        assert engine.is_signal_corruption_active(2)
        assert engine.is_signal_corruption_active(3)
        assert engine.is_signal_corruption_active(4)
        assert not engine.is_signal_corruption_active(5)

    def test_integer_target_zero_out(self):
        cfg = PerturbationConfig(
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                targets=["rework_count"],
                mode=CorruptionMode.ZERO_OUT,
            )
        )
        engine = PerturbationEngine(cfg)
        obs = ProxyObservables(rework_count=5)
        result = engine.perturb_observables(obs)
        assert result.rework_count == 0


# ===========================================================================
# TestParameterShocks
# ===========================================================================


class TestParameterShocks:
    def test_epoch_trigger(self):
        gov = _make_governance(transaction_tax_rate=0.05)
        cfg = PerturbationConfig(
            parameter_shocks=ParameterShocksConfig(
                enabled=True,
                shocks=[
                    ParameterShockSpec(
                        trigger=ShockTrigger.EPOCH,
                        at_epoch=2,
                        params={"transaction_tax_rate": 0.50},
                    )
                ],
            )
        )
        engine = PerturbationEngine(cfg, governance_engine=gov)
        engine.on_epoch_start(0)
        assert gov.config.transaction_tax_rate == 0.05
        engine.on_epoch_start(2)
        assert gov.config.transaction_tax_rate == 0.50

    def test_revert_after_epochs(self):
        gov = _make_governance(transaction_tax_rate=0.05)
        cfg = PerturbationConfig(
            parameter_shocks=ParameterShocksConfig(
                enabled=True,
                shocks=[
                    ParameterShockSpec(
                        trigger=ShockTrigger.EPOCH,
                        at_epoch=1,
                        params={"transaction_tax_rate": 0.90},
                        revert_after_epochs=2,
                    )
                ],
            )
        )
        engine = PerturbationEngine(cfg, governance_engine=gov)
        engine.on_epoch_start(1)
        assert gov.config.transaction_tax_rate == 0.90
        engine.on_epoch_start(2)
        assert gov.config.transaction_tax_rate == 0.90  # not yet
        engine.on_epoch_start(3)
        assert gov.config.transaction_tax_rate == 0.05  # reverted

    def test_condition_trigger(self):
        gov = _make_governance(bandwidth_cap=10)
        cfg = PerturbationConfig(
            parameter_shocks=ParameterShocksConfig(
                enabled=True,
                shocks=[
                    ParameterShockSpec(
                        trigger=ShockTrigger.CONDITION,
                        when="toxicity_rate > 0.5",
                        params={"bandwidth_cap": 2},
                    )
                ],
            )
        )
        engine = PerturbationEngine(cfg, governance_engine=gov)
        engine.check_condition({"toxicity_rate": 0.3, "epoch": 0})
        assert gov.config.bandwidth_cap == 10
        engine.check_condition({"toxicity_rate": 0.8, "epoch": 1})
        assert gov.config.bandwidth_cap == 2


# ===========================================================================
# TestNetworkPartition
# ===========================================================================


class TestNetworkPartition:
    def test_bisect(self):
        agents = [{"id": f"a{i}"} for i in range(4)]
        state = _make_state(agents)
        edges = [
            ("a0", "a1", 1.0),
            ("a0", "a2", 1.0),
            ("a0", "a3", 1.0),
            ("a1", "a2", 1.0),
            ("a1", "a3", 1.0),
            ("a2", "a3", 1.0),
        ]
        net = _make_network(["a0", "a1", "a2", "a3"], edges)
        cfg = PerturbationConfig(
            seed=42,
            network_partition=NetworkPartitionConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=0,
                mode=PartitionMode.BISECT,
            ),
        )
        engine = PerturbationEngine(cfg, state=state, network=net)
        engine.on_epoch_start(0)
        # Some cross-group edges should be removed
        assert engine._partition_active

    def test_isolate_type(self):
        state = _make_state([
            {"id": "h1", "type": "honest"},
            {"id": "h2", "type": "honest"},
            {"id": "a1", "type": "adversarial"},
        ])
        edges = [("h1", "a1", 1.0), ("h2", "a1", 1.0), ("h1", "h2", 1.0)]
        net = _make_network(["h1", "h2", "a1"], edges)
        cfg = PerturbationConfig(
            network_partition=NetworkPartitionConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=0,
                mode=PartitionMode.ISOLATE_TYPE,
                isolate_type="adversarial",
            )
        )
        engine = PerturbationEngine(cfg, state=state, network=net)
        engine.on_epoch_start(0)
        # Adversarial agent should have no cross-type neighbors
        assert "h1" not in net.get_neighbors("a1")
        assert "h2" not in net.get_neighbors("a1")
        # Honest-honest edge preserved
        assert "h2" in net.get_neighbors("h1")

    def test_heal(self):
        state = _make_state([{"id": "a1"}, {"id": "a2"}])
        edges = [("a1", "a2", 1.0)]
        net = _make_network(["a1", "a2"], edges)
        cfg = PerturbationConfig(
            seed=42,
            network_partition=NetworkPartitionConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=0,
                mode=PartitionMode.BISECT,
                heal_after_epochs=2,
            ),
        )
        engine = PerturbationEngine(cfg, state=state, network=net)
        engine.on_epoch_start(0)
        assert engine._partition_active
        engine.on_epoch_start(1)
        assert engine._partition_active
        engine.on_epoch_start(2)
        assert not engine._partition_active
        # Edge should be restored
        assert "a2" in net.get_neighbors("a1")


# ===========================================================================
# TestResourceShock
# ===========================================================================


class TestResourceShock:
    def test_drain_all(self):
        state = _make_state([
            {"id": "a1", "resources": 100.0},
            {"id": "a2", "resources": 200.0},
        ])
        cfg = PerturbationConfig(
            resource_shock=ResourceShockConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=1,
                mode=ResourceShockMode.DRAIN_ALL,
                magnitude=0.5,
            )
        )
        engine = PerturbationEngine(cfg, state=state)
        engine.on_epoch_start(0)
        assert state.agents["a1"].resources == 100.0
        engine.on_epoch_start(1)
        assert state.agents["a1"].resources == pytest.approx(50.0)
        assert state.agents["a2"].resources == pytest.approx(100.0)

    def test_redistribute(self):
        state = _make_state([
            {"id": "a1", "resources": 200.0},
            {"id": "a2", "resources": 0.0},
        ])
        cfg = PerturbationConfig(
            resource_shock=ResourceShockConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=0,
                mode=ResourceShockMode.REDISTRIBUTE,
                magnitude=1.0,  # full redistribution
            )
        )
        engine = PerturbationEngine(cfg, state=state)
        engine.on_epoch_start(0)
        # Equal share = 100 each with magnitude 1.0
        assert state.agents["a1"].resources == pytest.approx(100.0)
        assert state.agents["a2"].resources == pytest.approx(100.0)

    def test_inflate(self):
        state = _make_state([
            {"id": "a1", "resources": 100.0},
        ])
        cfg = PerturbationConfig(
            resource_shock=ResourceShockConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=0,
                mode=ResourceShockMode.INFLATE,
                magnitude=0.5,
            )
        )
        engine = PerturbationEngine(cfg, state=state)
        engine.on_epoch_start(0)
        assert state.agents["a1"].resources == pytest.approx(150.0)


# ===========================================================================
# TestPerturbationEngineIntegration
# ===========================================================================


class TestPerturbationEngineIntegration:
    def test_full_lifecycle(self):
        """Multiple perturbation types active simultaneously."""
        state = _make_state([
            {"id": "a1", "resources": 100.0},
            {"id": "a2", "resources": 100.0},
        ])
        gov = _make_governance(transaction_tax_rate=0.05)
        net = _make_network(["a1", "a2"], [("a1", "a2", 1.0)])

        cfg = PerturbationConfig(
            seed=42,
            agent_dropout=AgentDropoutConfig(
                enabled=True,
                probability_per_step=0.5,
                min_duration_steps=1,
                max_duration_steps=1,
            ),
            signal_corruption=SignalCorruptionConfig(
                enabled=True,
                targets=["task_progress_delta"],
                mode=CorruptionMode.ZERO_OUT,
            ),
            parameter_shocks=ParameterShocksConfig(
                enabled=True,
                shocks=[
                    ParameterShockSpec(
                        trigger=ShockTrigger.EPOCH,
                        at_epoch=1,
                        params={"transaction_tax_rate": 0.50},
                        revert_after_epochs=2,
                    )
                ],
            ),
            resource_shock=ResourceShockConfig(
                enabled=True,
                trigger=ShockTrigger.EPOCH,
                at_epoch=0,
                mode=ResourceShockMode.DRAIN_ALL,
                magnitude=0.1,
            ),
        )

        engine = PerturbationEngine(
            cfg,
            state=state,
            network=net,
            governance_engine=gov,
        )

        # Epoch 0: resource shock fires
        engine.on_epoch_start(0)
        assert state.agents["a1"].resources == pytest.approx(90.0)

        # Step: dropout may fire
        engine.on_step_start(0, 0)

        # Signal corruption always active
        obs = ProxyObservables(task_progress_delta=0.9)
        result = engine.perturb_observables(obs)
        assert result.task_progress_delta == 0.0

        # Epoch 1: parameter shock fires
        engine.on_epoch_start(1)
        assert gov.config.transaction_tax_rate == 0.50

        # Epoch 3: parameter shock reverts
        engine.on_epoch_start(3)
        assert gov.config.transaction_tax_rate == 0.05

    def test_no_crash_with_none_refs(self):
        """Engine with None state/network/governance shouldn't crash."""
        cfg = PerturbationConfig(
            agent_dropout=AgentDropoutConfig(enabled=True),
            signal_corruption=SignalCorruptionConfig(enabled=True),
            parameter_shocks=ParameterShocksConfig(
                enabled=True,
                shocks=[
                    ParameterShockSpec(
                        trigger=ShockTrigger.EPOCH, at_epoch=0,
                        params={"x": 1},
                    )
                ],
            ),
            network_partition=NetworkPartitionConfig(
                enabled=True, at_epoch=0
            ),
            resource_shock=ResourceShockConfig(enabled=True, at_epoch=0),
        )
        engine = PerturbationEngine(cfg)
        # Should not raise
        engine.on_epoch_start(0)
        engine.on_step_start(0, 0)
        assert engine.get_dropped_agents() == set()
