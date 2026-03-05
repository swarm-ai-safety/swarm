"""Tests for governance modules."""

import pytest

from swarm_gym.governance.tax import TaxPolicy
from swarm_gym.governance.audits import AuditPolicy
from swarm_gym.governance.circuit_breaker import CircuitBreakerPolicy
from swarm_gym.utils.types import Action


class TestTaxPolicy:
    def test_tax_applied_to_trades(self):
        policy = TaxPolicy(rate=0.10)
        actions = [
            Action(agent_id="a0", type="trade", asset="X", qty=2.0, price=10.0),
        ]
        result_actions, interventions, events = policy.apply({}, actions)
        assert result_actions[0].metadata["tax_paid"] == pytest.approx(2.0)
        assert result_actions[0].metadata["post_tax_value"] == pytest.approx(18.0)
        assert len(interventions) == 1
        assert interventions[0].type == "tax_applied"

    def test_no_tax_on_non_trades(self):
        policy = TaxPolicy(rate=0.10)
        actions = [Action(agent_id="a0", type="cooperate")]
        result_actions, interventions, events = policy.apply({}, actions)
        assert "tax_paid" not in result_actions[0].metadata
        assert len(interventions) == 0

    def test_state_tracks_revenue(self):
        policy = TaxPolicy(rate=0.05)
        actions = [Action(agent_id="a0", type="trade", asset="X", qty=1.0, price=100.0)]
        policy.apply({}, actions)
        state = policy.get_state()
        assert state["tax_revenue"] == pytest.approx(5.0)

    def test_reset_clears_revenue(self):
        policy = TaxPolicy(rate=0.05)
        actions = [Action(agent_id="a0", type="trade", asset="X", qty=1.0, price=100.0)]
        policy.apply({}, actions)
        policy.reset()
        assert policy.get_state()["tax_revenue"] == 0.0


class TestAuditPolicy:
    def test_suspicious_action_detected(self):
        policy = AuditPolicy(p_audit=1.0, penalty=5.0)  # Always audit
        policy.reset(seed=42)
        actions = [Action(agent_id="a0", type="hide")]
        _, interventions, events = policy.apply({}, actions)
        # Should detect and penalize
        assert actions[0].metadata.get("audit_penalty") == 5.0
        assert any(e.outcome == "FAIL" for e in events)

    def test_honest_action_passes(self):
        policy = AuditPolicy(p_audit=1.0, penalty=5.0, false_positive_rate=0.0)
        policy.reset(seed=42)
        actions = [Action(agent_id="a0", type="cooperate")]
        _, interventions, events = policy.apply({}, actions)
        # Should pass audit
        passing = [e for e in events if e.outcome == "PASS"]
        assert len(passing) >= 1

    def test_budget_limits_audits(self):
        policy = AuditPolicy(p_audit=1.0, penalty=5.0, budget_per_episode=1)
        policy.reset(seed=42)
        actions = [
            Action(agent_id="a0", type="hide"),
            Action(agent_id="a1", type="hide"),
            Action(agent_id="a2", type="hide"),
        ]
        policy.apply({}, actions)
        assert policy.get_state()["audits_used"] <= 1


class TestCircuitBreakerPolicy:
    def test_blocks_terminal_actions(self):
        policy = CircuitBreakerPolicy(block_terminal=True, terminal_actions=["nuke"])
        actions = [Action(agent_id="a0", type="nuke")]
        result, interventions, events = policy.apply({}, actions)
        assert result[0].type == "noop"
        assert result[0].metadata["blocked_by"] == "circuit_breaker"
        assert len(interventions) == 1
        assert interventions[0].type == "terminal_action_blocked"

    def test_escalation_threshold(self):
        policy = CircuitBreakerPolicy(threshold=0.5)
        world_state = {"escalation_risk": 0.8}
        actions = [Action(agent_id="a0", type="escalate", level="threaten")]
        result, interventions, events = policy.apply(world_state, actions)
        assert result[0].type == "noop"
        assert any(i.type == "escalation_breaker_tripped" for i in interventions)

    def test_allows_below_threshold(self):
        policy = CircuitBreakerPolicy(threshold=0.9)
        world_state = {"escalation_risk": 0.3}
        actions = [Action(agent_id="a0", type="escalate", level="signal")]
        result, interventions, events = policy.apply(world_state, actions)
        assert result[0].type == "escalate"

    def test_cooldown_freezes_agent(self):
        policy = CircuitBreakerPolicy(
            block_terminal=True, terminal_actions=["nuke"], cooldown_steps=2,
        )
        # First action: blocked (nuke)
        actions = [Action(agent_id="a0", type="nuke")]
        policy.apply({}, actions)

        # Second action: still frozen
        actions = [Action(agent_id="a0", type="cooperate")]
        result, _, _ = policy.apply({}, actions)
        assert result[0].type == "noop"
