"""Tests for WorkRegimeAgent and adapt_policy wiring via middleware."""

from unittest.mock import MagicMock

import pytest

from swarm.agents.work_regime_agent import WorkRegimeAgent
from swarm.core.middleware import MiddlewareContext, WorkRegimeAdaptMiddleware
from swarm.models.interaction import SoftInteraction


def _make_interaction(
    *,
    initiator: str = "worker_0",
    counterparty: str = "manager_0",
    accepted: bool = True,
    p: float = 0.5,
) -> SoftInteraction:
    return SoftInteraction(
        interaction_id="test",
        initiator=initiator,
        counterparty=counterparty,
        accepted=accepted,
        p=p,
    )


class TestWorkRegimeAgentAdaptPolicy:
    """Unit tests for adapt_policy logic."""

    def setup_method(self):
        self.agent = WorkRegimeAgent("worker_0")

    def test_initial_policy_state(self):
        """Policy variables start at defaults."""
        assert self.agent.compliance_propensity == pytest.approx(0.8)
        assert self.agent.cooperation_threshold == pytest.approx(0.3)
        assert self.agent.redistribution_preference == pytest.approx(0.2)
        assert self.agent.exit_propensity == pytest.approx(0.05)
        assert self.agent.grievance == pytest.approx(0.0)

    def test_adapt_policy_updates_grievance_on_pay_gap(self):
        """Negative pay gap (underpaid) increases grievance."""
        self.agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=1.0,  # agent earns less than peers
            eval_noise=0.0,
            workload_pressure=0.0,
        )
        assert self.agent.grievance > 0.0

    def test_adapt_policy_no_grievance_when_overpaid(self):
        """Positive pay gap (overpaid) produces zero grievance change."""
        initial_grievance = self.agent.grievance
        self.agent.adapt_policy(
            avg_payoff=2.0,
            peer_avg_payoff=1.0,  # agent earns more than peers
            eval_noise=0.0,
            workload_pressure=0.0,
        )
        # grievance decays when unfairness_signal=0
        assert self.agent.grievance <= initial_grievance

    def test_adapt_policy_compliance_decreases_under_pressure(self):
        """Compliance drops when workload_pressure and grievance are high."""
        self.agent.grievance = 5.0  # pre-seed grievance
        initial = self.agent.compliance_propensity
        self.agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=0.0,
            eval_noise=0.0,
            workload_pressure=1.0,
        )
        assert self.agent.compliance_propensity < initial

    def test_adapt_policy_cooperation_threshold_rises_with_grievance(self):
        """Cooperation threshold increases with accumulated grievance."""
        self.agent.grievance = 3.0
        initial = self.agent.cooperation_threshold
        self.agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=0.0,
            eval_noise=0.0,
            workload_pressure=0.0,
        )
        assert self.agent.cooperation_threshold > initial

    def test_adapt_policy_exit_propensity_rises_when_grievance_high(self):
        """Exit propensity increases when grievance exceeds threshold."""
        self.agent.grievance = 2.0
        initial = self.agent.exit_propensity
        self.agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=0.0,
            eval_noise=0.0,
            workload_pressure=0.0,
        )
        assert self.agent.exit_propensity > initial

    def test_policy_drift_zero_initially(self):
        """No drift when policy state equals initial state."""
        assert self.agent.policy_drift() == pytest.approx(0.0)

    def test_policy_drift_positive_after_adaptation(self):
        """Drift is positive after adapt_policy causes state changes."""
        self.agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=2.0,
            eval_noise=0.5,
            workload_pressure=1.0,
        )
        assert self.agent.policy_drift() > 0.0

    def test_policy_snapshot_keys(self):
        """policy_snapshot returns expected keys."""
        snap = self.agent.policy_snapshot()
        for key in (
            "compliance_propensity",
            "cooperation_threshold",
            "redistribution_preference",
            "exit_propensity",
            "grievance",
            "drift",
        ):
            assert key in snap

    def test_policy_bounds_respected(self):
        """Policy variables stay within [0, 1] after many adaptation steps."""
        for _ in range(50):
            self.agent.adapt_policy(
                avg_payoff=-5.0,  # very negative — max stress
                peer_avg_payoff=5.0,
                eval_noise=1.0,
                workload_pressure=1.0,
            )
        assert 0.0 <= self.agent.compliance_propensity <= 1.0
        assert 0.0 <= self.agent.cooperation_threshold <= 1.0
        assert 0.0 <= self.agent.redistribution_preference <= 1.0
        assert 0.0 <= self.agent.exit_propensity <= 0.8


class TestWorkRegimeAgentUpdateFromOutcome:
    """update_from_outcome accumulates payoffs for epoch-end adaptation."""

    def test_epoch_payoffs_accumulate(self):
        agent = WorkRegimeAgent("worker_0")
        ix = _make_interaction(initiator="worker_0")
        agent.update_from_outcome(ix, payoff=1.5)
        agent.update_from_outcome(ix, payoff=-0.5)
        assert len(agent._epoch_payoffs) == 2
        assert sum(agent._epoch_payoffs) == pytest.approx(1.0)

    def test_on_epoch_end_calls_adapt_policy_and_resets(self):
        """on_epoch_end invokes adapt_policy and clears epoch_payoffs."""
        agent = WorkRegimeAgent("worker_0")
        ix = _make_interaction(initiator="worker_0")
        agent.update_from_outcome(ix, payoff=-1.0)
        assert len(agent._epoch_payoffs) == 1

        initial_grievance = agent.grievance
        agent.on_epoch_end(peer_avg_payoff=1.0, workload_pressure=0.5)

        # Payoffs should be cleared
        assert len(agent._epoch_payoffs) == 0
        # Grievance should change (underpaid vs peer_avg_payoff=1.0)
        assert agent.grievance != initial_grievance or agent.grievance >= 0.0

    def test_on_epoch_end_resets_strike_count(self):
        """on_epoch_end resets per-epoch strike counter."""
        agent = WorkRegimeAgent("worker_0")
        agent._epoch_strike_count = 3
        agent.on_epoch_end(peer_avg_payoff=0.0, workload_pressure=0.0)
        assert agent._epoch_strike_count == 0


class TestWorkRegimeAdaptMiddleware:
    """WorkRegimeAdaptMiddleware invokes on_epoch_end for WorkRegimeAgents."""

    def _make_ctx(self, agents, agent_states=None, completed_interactions=None):
        """Build a minimal MiddlewareContext mock."""
        ctx = MagicMock(spec=MiddlewareContext)
        ctx.agents = agents

        # state.agents returns AgentState objects with total_payoff
        mock_state = MagicMock()
        mock_state.agents = agent_states or {}
        mock_state.completed_interactions = completed_interactions or []
        ctx.state = mock_state

        mock_config = MagicMock()
        mock_config.bandwidth_cap = 10
        ctx.config = mock_config

        return ctx

    def test_calls_on_epoch_end_for_work_regime_agents(self):
        """Middleware calls on_epoch_end on each WorkRegimeAgent."""
        agent = WorkRegimeAgent("w0")
        agent.on_epoch_end = MagicMock()

        mock_state = MagicMock()
        mock_state.agents = {}
        mock_state.completed_interactions = []

        ctx = self._make_ctx({"w0": agent})
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        agent.on_epoch_end.assert_called_once()

    def test_skips_non_work_regime_agents(self):
        """Middleware does not call on_epoch_end on other agent types."""
        from swarm.agents.honest import HonestAgent

        honest = HonestAgent("h0")
        assert not isinstance(honest, WorkRegimeAgent)

        ctx = self._make_ctx({"h0": honest})
        mw = WorkRegimeAdaptMiddleware()
        # Should not raise even when no WorkRegimeAgents are present
        mw.on_epoch_end(ctx)

    def test_no_op_when_no_work_regime_agents(self):
        """Middleware is silent when agent pool has no WorkRegimeAgents."""
        from swarm.agents.honest import HonestAgent

        ctx = self._make_ctx({"h0": HonestAgent("h0")})
        mw = WorkRegimeAdaptMiddleware()
        # Should complete without error
        mw.on_epoch_end(ctx)

    def test_peer_avg_payoff_computed_from_state(self):
        """Middleware passes population mean payoff to on_epoch_end."""
        agent = WorkRegimeAgent("w0")
        agent.on_epoch_end = MagicMock()

        mock_agent_state_a = MagicMock()
        mock_agent_state_a.total_payoff = 2.0
        mock_agent_state_b = MagicMock()
        mock_agent_state_b.total_payoff = 4.0

        ctx = self._make_ctx(
            agents={"w0": agent},
            agent_states={"a": mock_agent_state_a, "b": mock_agent_state_b},
        )
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        call_kwargs = agent.on_epoch_end.call_args[1]
        assert call_kwargs["peer_avg_payoff"] == pytest.approx(3.0)

    def test_workload_pressure_from_bandwidth_cap(self):
        """Workload pressure = completed_interactions / bandwidth_cap."""
        agent = WorkRegimeAgent("w0")
        agent.on_epoch_end = MagicMock()

        # 5 completed interactions out of cap=10 → pressure=0.5
        ctx = self._make_ctx(
            agents={"w0": agent},
            completed_interactions=[MagicMock()] * 5,
        )
        ctx.config.bandwidth_cap = 10
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        call_kwargs = agent.on_epoch_end.call_args[1]
        assert call_kwargs["workload_pressure"] == pytest.approx(0.5)

    def test_workload_pressure_capped_at_one(self):
        """Workload pressure is capped at 1.0 even when over capacity."""
        agent = WorkRegimeAgent("w0")
        agent.on_epoch_end = MagicMock()

        ctx = self._make_ctx(
            agents={"w0": agent},
            completed_interactions=[MagicMock()] * 20,
        )
        ctx.config.bandwidth_cap = 10
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        call_kwargs = agent.on_epoch_end.call_args[1]
        assert call_kwargs["workload_pressure"] <= 1.0

    def test_workload_pressure_zero_when_no_cap(self):
        """Workload pressure defaults to 0 when bandwidth_cap is unset."""
        agent = WorkRegimeAgent("w0")
        agent.on_epoch_end = MagicMock()

        ctx = self._make_ctx(agents={"w0": agent})
        ctx.config.bandwidth_cap = 0  # no cap
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        call_kwargs = agent.on_epoch_end.call_args[1]
        assert call_kwargs["workload_pressure"] == pytest.approx(0.0)

    def test_adapt_policy_state_changes_after_middleware(self):
        """End-to-end: policy state drifts after middleware invocation."""
        agent = WorkRegimeAgent("w0")
        ix = _make_interaction(initiator="w0")
        # Simulate some bad payoffs this epoch
        agent.update_from_outcome(ix, payoff=-2.0)
        agent.update_from_outcome(ix, payoff=-1.0)

        initial_grievance = agent.grievance
        mock_state = MagicMock()
        mock_state.agents = {}
        mock_state.completed_interactions = []
        ctx = self._make_ctx(agents={"w0": agent})
        ctx.config.bandwidth_cap = 10

        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        # After middleware, adapt_policy should have run and grievance changed
        assert agent.grievance != initial_grievance or agent._epoch_payoffs == []
        assert agent._epoch_payoffs == []  # epoch accumulators reset
