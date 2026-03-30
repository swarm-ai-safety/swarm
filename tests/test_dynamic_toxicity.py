"""Tests for dynamic toxicity feedback middleware."""

import pytest

from swarm.core.dynamic_toxicity import (
    ProxyCalibrationDriftMiddleware,
    QualityContagionMiddleware,
    TrustErosionMiddleware,
)
from swarm.core.middleware import MiddlewareContext
from swarm.core.proxy import ProxyComputer
from swarm.logging.event_bus import EventBus
from swarm.metrics.soft_metrics import SoftMetrics
from swarm.models.interaction import SoftInteraction


def _make_interactions(p_values, accepted=True):
    """Create interactions with given p values."""
    interactions = []
    for i, p in enumerate(p_values):
        interactions.append(
            SoftInteraction(
                interaction_id=f"int-{i}",
                initiator=f"agent-{i % 3}",
                counterparty=f"agent-{(i + 1) % 3}",
                p=p,
                accepted=accepted,
            )
        )
    return interactions


def _make_context(interactions, agents=None, frozen=None):
    """Build a minimal MiddlewareContext."""
    from swarm.env.state import EnvState

    state = EnvState()
    state.completed_interactions = list(interactions)
    if frozen:
        state.frozen_agents = set(frozen)

    ctx = MiddlewareContext(
        state=state,
        config=None,
        agents=agents or {},
        event_bus=EventBus(),
        metrics_calculator=SoftMetrics(),
    )
    return ctx


# -----------------------------------------------------------------------
# Proxy Calibration Drift
# -----------------------------------------------------------------------


class TestProxyCalibrationDrift:
    def test_sigmoid_k_decreases_with_toxicity(self):
        """High toxicity should reduce sigmoid_k."""
        proxy = ProxyComputer(sigmoid_k=2.0)
        bus = EventBus()
        mw = ProxyCalibrationDriftMiddleware(proxy, bus, alpha=0.5, k_floor=0.5)

        # Epoch with high toxicity (low p values)
        interactions = _make_interactions([0.2, 0.3, 0.1, 0.25])
        ctx = _make_context(interactions)

        initial_k = proxy.sigmoid_k
        mw.on_epoch_end(ctx)

        assert proxy.sigmoid_k < initial_k
        assert proxy.sigmoid_k >= 0.5  # floor

    def test_low_toxicity_minimal_drift(self):
        """Low toxicity should barely affect sigmoid_k."""
        proxy = ProxyComputer(sigmoid_k=2.0)
        bus = EventBus()
        mw = ProxyCalibrationDriftMiddleware(proxy, bus, alpha=0.5, k_floor=0.5)

        # Epoch with low toxicity (high p values)
        interactions = _make_interactions([0.9, 0.95, 0.85, 0.92])
        ctx = _make_context(interactions)

        mw.on_epoch_end(ctx)
        # Toxicity ~ 0.1, cumulative = 0.1, decay = 1 - 0.5*0.1 = 0.95
        assert proxy.sigmoid_k == pytest.approx(2.0 * 0.95, abs=0.05)

    def test_k_floor_respected(self):
        """sigmoid_k should never go below k_floor."""
        proxy = ProxyComputer(sigmoid_k=2.0)
        bus = EventBus()
        mw = ProxyCalibrationDriftMiddleware(proxy, bus, alpha=2.0, k_floor=0.8)

        # Many epochs of high toxicity
        for _ in range(20):
            interactions = _make_interactions([0.1, 0.2, 0.15])
            ctx = _make_context(interactions)
            mw.on_epoch_end(ctx)

        assert proxy.sigmoid_k >= 0.8

    def test_cumulative_toxicity_tracked(self):
        """Drift should accumulate across epochs."""
        proxy = ProxyComputer(sigmoid_k=2.0)
        bus = EventBus()
        mw = ProxyCalibrationDriftMiddleware(proxy, bus, alpha=0.5, k_floor=0.5)

        for _ in range(3):
            interactions = _make_interactions([0.5, 0.5, 0.5])
            ctx = _make_context(interactions)
            mw.on_epoch_end(ctx)

        # toxicity per epoch = 0.5, cumulative = 1.5
        # decay = 1 - 0.5 * 1.5 = 0.25
        assert proxy.sigmoid_k == pytest.approx(2.0 * 0.25, abs=0.01)


# -----------------------------------------------------------------------
# Trust Erosion
# -----------------------------------------------------------------------


class TestTrustErosion:
    def _make_honest_agent(self, agent_id):
        from swarm.agents.honest import HonestAgent
        from swarm.models.agent import AgentType

        agent = HonestAgent(agent_id=agent_id)
        agent.agent_type = AgentType.HONEST
        return agent

    def _make_deceptive_agent(self, agent_id):
        from swarm.agents.deceptive import DeceptiveAgent
        from swarm.models.agent import AgentType

        agent = DeceptiveAgent(agent_id=agent_id)
        agent.agent_type = AgentType.DECEPTIVE
        return agent

    def test_no_exit_below_threshold(self):
        """No agents should exit when toxicity is below threshold."""
        bus = EventBus()
        mw = TrustErosionMiddleware(
            bus, beta=0.3, exit_threshold=0.5, window=1, seed=42
        )

        agents = {f"h{i}": self._make_honest_agent(f"h{i}") for i in range(5)}
        # Low toxicity
        interactions = _make_interactions([0.8, 0.9, 0.85])
        ctx = _make_context(interactions, agents=agents)

        mw.on_epoch_end(ctx)
        assert len(ctx.state.frozen_agents) == 0

    def test_exit_above_threshold(self):
        """Agents should start exiting when rolling toxicity exceeds threshold."""
        bus = EventBus()
        mw = TrustErosionMiddleware(
            bus, beta=1.0, exit_threshold=0.3, window=1, min_honest=0, seed=42
        )

        agents = {f"h{i}": self._make_honest_agent(f"h{i}") for i in range(5)}
        # High toxicity (p=0.1 → toxicity=0.9)
        interactions = _make_interactions([0.1, 0.1, 0.1])
        ctx = _make_context(interactions, agents=agents)

        mw.on_epoch_end(ctx)
        # With beta=1.0 and excess=0.6, exit_prob=0.6 — some should exit
        assert len(ctx.state.frozen_agents) > 0

    def test_min_honest_respected(self):
        """At least min_honest agents should always remain."""
        bus = EventBus()
        mw = TrustErosionMiddleware(
            bus, beta=10.0, exit_threshold=0.0, window=1, min_honest=2, seed=42
        )

        agents = {f"h{i}": self._make_honest_agent(f"h{i}") for i in range(3)}
        interactions = _make_interactions([0.1, 0.1, 0.1])
        ctx = _make_context(interactions, agents=agents)

        mw.on_epoch_end(ctx)
        remaining_honest = len(agents) - len(ctx.state.frozen_agents)
        assert remaining_honest >= 2

    def test_only_honest_agents_exit(self):
        """Deceptive agents should not exit."""
        bus = EventBus()
        mw = TrustErosionMiddleware(
            bus, beta=10.0, exit_threshold=0.0, window=1, min_honest=0, seed=42
        )

        agents = {}
        agents["h0"] = self._make_honest_agent("h0")
        agents["d0"] = self._make_deceptive_agent("d0")

        interactions = _make_interactions([0.1, 0.1])
        ctx = _make_context(interactions, agents=agents)

        mw.on_epoch_end(ctx)
        # Only h0 could exit, not d0
        assert "d0" not in ctx.state.frozen_agents


# -----------------------------------------------------------------------
# Quality Contagion
# -----------------------------------------------------------------------


class TestQualityContagion:
    def _make_agent_with_memory(self, agent_id):
        from swarm.agents.honest import HonestAgent

        agent = HonestAgent(agent_id=agent_id)
        return agent

    def test_low_quality_interactions_reduce_trust(self):
        """Interacting with low-p counterparts should reduce trust memory."""
        bus = EventBus()
        mw = QualityContagionMiddleware(bus, gamma=0.5, neutral=0.5)

        agent = self._make_agent_with_memory("a0")
        agent._counterparty_memory["a1"] = 0.7  # initially trusts a1

        agents = {"a0": agent}

        # a0 initiates low-quality interaction with a1
        interactions = [
            SoftInteraction(
                interaction_id="int-0",
                initiator="a0",
                counterparty="a1",
                p=0.1,  # very low quality
                accepted=True,
            )
        ]
        ctx = _make_context(interactions, agents=agents)
        mw.on_epoch_end(ctx)

        # Trust in a1 should have decreased
        # delta = 0.5 * (0.1 - 0.5) = -0.2
        # new trust = 0.7 + (-0.2) = 0.5
        assert agent._counterparty_memory["a1"] < 0.7

    def test_high_quality_interactions_increase_trust(self):
        """Interacting with high-p counterparts should increase trust."""
        bus = EventBus()
        mw = QualityContagionMiddleware(bus, gamma=0.5, neutral=0.5)

        agent = self._make_agent_with_memory("a0")
        agent._counterparty_memory["a1"] = 0.5

        agents = {"a0": agent}

        interactions = [
            SoftInteraction(
                interaction_id="int-0",
                initiator="a0",
                counterparty="a1",
                p=0.9,
                accepted=True,
            )
        ]
        ctx = _make_context(interactions, agents=agents)
        mw.on_epoch_end(ctx)

        # delta = 0.5 * (0.9 - 0.5) = 0.2
        assert agent._counterparty_memory["a1"] > 0.5

    def test_rejected_interactions_no_contagion(self):
        """Rejected interactions should not cause contagion."""
        bus = EventBus()
        mw = QualityContagionMiddleware(bus, gamma=0.5, neutral=0.5)

        agent = self._make_agent_with_memory("a0")
        agent._counterparty_memory["a1"] = 0.7

        agents = {"a0": agent}

        interactions = [
            SoftInteraction(
                interaction_id="int-0",
                initiator="a0",
                counterparty="a1",
                p=0.1,
                accepted=False,  # rejected
            )
        ]
        ctx = _make_context(interactions, agents=agents)
        mw.on_epoch_end(ctx)

        assert agent._counterparty_memory["a1"] == 0.7

    def test_bias_clamp(self):
        """Accumulated bias should be clamped to [-0.5, 0.5]."""
        bus = EventBus()
        mw = QualityContagionMiddleware(bus, gamma=1.0, neutral=0.5)

        agent = self._make_agent_with_memory("a0")
        agents = {"a0": agent}

        # Many epochs of extreme low-quality interactions
        for _ in range(20):
            interactions = [
                SoftInteraction(
                    interaction_id="int-0",
                    initiator="a0",
                    counterparty="a1",
                    p=0.0,
                    accepted=True,
                )
            ]
            ctx = _make_context(interactions, agents=agents)
            mw.on_epoch_end(ctx)

        assert mw._agent_bias["a0"] >= -0.5
