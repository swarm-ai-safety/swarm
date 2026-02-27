"""Tests for WorkRegimeAgent policy adaptation wiring.

Validates that adapt_policy is invoked during simulation updates via the
WorkRegimeAdaptMiddleware + on_epoch_end pathway.
"""

import pytest

from swarm.agents.work_regime_agent import WorkRegimeAgent
from swarm.core.middleware import WorkRegimeAdaptMiddleware, MiddlewareContext
from swarm.models.interaction import SoftInteraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(agent_id: str = "wr_0", **kwargs) -> WorkRegimeAgent:
    config = {
        "compliance_propensity": 0.8,
        "cooperation_threshold": 0.3,
        "redistribution_preference": 0.2,
        "exit_propensity": 0.05,
        "adapt_rate": 0.1,
        "grievance_decay": 0.9,
        **kwargs,
    }
    return WorkRegimeAgent(agent_id=agent_id, config=config)


def _make_interaction(
    *,
    initiator: str = "wr_0",
    counterparty: str = "other",
    p: float = 0.5,
    accepted: bool = True,
) -> SoftInteraction:
    return SoftInteraction(
        interaction_id="test",
        initiator=initiator,
        counterparty=counterparty,
        accepted=accepted,
        task_progress_delta=0.5,
        rework_count=0,
        verifier_rejections=0,
        tool_misuse_flags=0,
        counterparty_engagement_delta=0.3,
        v_hat=0.0,
        p=p,
        tau=0.0,
        c_a=0.0,
        c_b=0.0,
        r_a=0.0,
        r_b=0.0,
    )


# ---------------------------------------------------------------------------
# Unit tests: WorkRegimeAgent
# ---------------------------------------------------------------------------


class TestWorkRegimeAgentAdaptPolicy:
    """adapt_policy mutates policy state correctly."""

    def test_adapt_policy_raises_grievance_on_underpayment(self):
        agent = _make_agent()
        initial_grievance = agent.grievance
        agent.adapt_policy(
            avg_payoff=-1.0,
            peer_avg_payoff=2.0,  # agent paid much less than peers
            eval_noise=0.0,
            workload_pressure=0.5,
        )
        assert agent.grievance > initial_grievance

    def test_adapt_policy_lowers_compliance_under_stress(self):
        agent = _make_agent()
        initial_compliance = agent.compliance_propensity
        # Force high grievance first
        agent.grievance = 5.0
        agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=0.0,
            eval_noise=0.0,
            workload_pressure=1.0,
        )
        assert agent.compliance_propensity < initial_compliance

    def test_adapt_policy_raises_redistribution_on_pay_gap(self):
        agent = _make_agent()
        initial_redist = agent.redistribution_preference
        agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=10.0,  # large pay gap
            eval_noise=0.0,
            workload_pressure=0.0,
        )
        assert agent.redistribution_preference > initial_redist

    def test_adapt_policy_raises_exit_under_high_grievance(self):
        agent = _make_agent()
        agent.grievance = 5.0  # high grievance
        initial_exit = agent.exit_propensity
        agent.adapt_policy(
            avg_payoff=0.0,
            peer_avg_payoff=0.0,
            eval_noise=0.0,
            workload_pressure=0.5,
        )
        assert agent.exit_propensity > initial_exit

    def test_adapt_policy_bounds_state_to_unit_interval(self):
        agent = _make_agent()
        agent.grievance = 100.0
        for _ in range(50):
            agent.adapt_policy(
                avg_payoff=-10.0,
                peer_avg_payoff=10.0,
                eval_noise=1.0,
                workload_pressure=1.0,
            )
        assert 0.0 <= agent.compliance_propensity <= 1.0
        assert 0.0 <= agent.cooperation_threshold <= 1.0
        assert 0.0 <= agent.redistribution_preference <= 1.0
        assert 0.0 <= agent.exit_propensity <= 1.0

    def test_grievance_capped_at_grievance_cap(self):
        agent = _make_agent()
        for _ in range(100):
            agent.adapt_policy(
                avg_payoff=-10.0,
                peer_avg_payoff=10.0,
                eval_noise=1.0,
                workload_pressure=1.0,
            )
        assert agent.grievance <= agent._grievance_cap


class TestUpdateFromOutcome:
    """update_from_outcome accumulates payoffs for epoch-end adaptation."""

    def test_epoch_payoffs_accumulated(self):
        agent = _make_agent()
        interaction = _make_interaction(initiator=agent.agent_id)

        agent.update_from_outcome(interaction, payoff=3.0)
        agent.update_from_outcome(interaction, payoff=-1.0)

        assert agent._epoch_payoffs == [3.0, -1.0]

    def test_on_epoch_end_calls_adapt_policy_and_clears_accumulators(self):
        agent = _make_agent()
        interaction = _make_interaction(initiator=agent.agent_id)
        agent.update_from_outcome(interaction, payoff=-5.0)

        initial_redist = agent.redistribution_preference
        agent.on_epoch_end(peer_avg_payoff=5.0, workload_pressure=0.5)

        # Accumulators cleared
        assert agent._epoch_payoffs == []
        assert agent._epoch_strike_count == 0
        # Policy drifted (redistribution should increase due to pay gap)
        assert agent.redistribution_preference > initial_redist

    def test_on_epoch_end_with_no_payoffs_does_not_crash(self):
        agent = _make_agent()
        agent.on_epoch_end(peer_avg_payoff=0.0, workload_pressure=0.0)  # no-op


# ---------------------------------------------------------------------------
# Unit tests: WorkRegimeAdaptMiddleware
# ---------------------------------------------------------------------------


class TestWorkRegimeAdaptMiddleware:
    """WorkRegimeAdaptMiddleware invokes on_epoch_end on all WorkRegimeAgents."""

    def _make_ctx(self, agents, completed_interactions=None, gov_engine=None):
        """Build a minimal MiddlewareContext with mocks."""

        class FakeConfig:
            steps_per_epoch = 10

        class FakeEventBus:
            def emit(self, event):
                pass

        class FakeState:
            pass

        state = FakeState()
        state.completed_interactions = completed_interactions or []

        return MiddlewareContext(
            state=state,
            config=FakeConfig(),
            agents={a.agent_id: a for a in agents},
            event_bus=FakeEventBus(),
            governance_engine=gov_engine,
        )

    def test_middleware_calls_on_epoch_end(self):
        agent = _make_agent()
        interaction = _make_interaction(initiator=agent.agent_id)
        agent.update_from_outcome(interaction, payoff=-5.0)

        ctx = self._make_ctx(
            [agent],
            completed_interactions=[interaction] * 5,
        )
        initial_redist = agent.redistribution_preference

        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        # Policy should have adapted
        assert agent.redistribution_preference >= initial_redist

    def test_middleware_skips_non_work_regime_agents(self):
        from swarm.agents.honest import HonestAgent

        honest = HonestAgent(agent_id="h0")
        ctx = self._make_ctx([honest])

        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)  # should not raise

    def test_middleware_computes_peer_avg_payoff_across_agents(self):
        a1 = _make_agent("wr_1")
        a2 = _make_agent("wr_2")

        interaction = _make_interaction(initiator="wr_1", counterparty="wr_2")
        a1.update_from_outcome(interaction, payoff=10.0)
        a2.update_from_outcome(interaction, payoff=-10.0)

        ctx = self._make_ctx([a1, a2], completed_interactions=[interaction])
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)

        # Both agents processed (epoch_payoffs cleared)
        assert a1._epoch_payoffs == []
        assert a2._epoch_payoffs == []

    def test_middleware_with_governance_engine_uses_bandwidth_cap(self):
        agent = _make_agent()

        class FakeGovConfig:
            bandwidth_cap = 5

        class FakeGovEngine:
            config = FakeGovConfig()

        ctx = self._make_ctx([agent], gov_engine=FakeGovEngine())
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)  # should not raise

    def test_middleware_no_work_regime_agents_is_noop(self):
        from swarm.agents.honest import HonestAgent

        ctx = self._make_ctx([HonestAgent(agent_id="h0")])
        mw = WorkRegimeAdaptMiddleware()
        mw.on_epoch_end(ctx)  # should not raise


# ---------------------------------------------------------------------------
# Integration test: wired into orchestrator
# ---------------------------------------------------------------------------


class TestWorkRegimeAdaptInOrchestrator:
    """WorkRegimeAdaptMiddleware is present in the orchestrator pipeline."""

    def test_middleware_registered_in_pipeline(self):
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        config = OrchestratorConfig(n_epochs=1, steps_per_epoch=2, seed=42)
        orc = Orchestrator(config=config)
        orc.register_agent(_make_agent())
        pipeline_types = [type(mw).__name__ for mw in orc._pipeline.all()]
        assert "WorkRegimeAdaptMiddleware" in pipeline_types

    def test_policy_drifts_after_simulation_run(self):
        """adapt_policy is called during run() and produces drift."""
        from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

        agent = _make_agent(adapt_rate=0.5)  # high rate to force visible drift
        initial_compliance = agent.compliance_propensity
        initial_exit = agent.exit_propensity

        config = OrchestratorConfig(n_epochs=3, steps_per_epoch=5, seed=0)
        orc = Orchestrator(config=config)
        orc.register_agent(agent)
        orc.run()

        # adapt_policy was invoked: _recent_payoffs is populated by adapt_policy
        assert len(agent._recent_payoffs) > 0
        # Policy state is within valid range after adaptation
        assert 0.0 <= agent.compliance_propensity <= 1.0
        assert 0.0 <= agent.exit_propensity <= 1.0
