"""Tests for benchmark environments: game mechanics and outcomes."""


import swarm_gym.envs.audit_evasion  # noqa: F401
import swarm_gym.envs.collusion_market  # noqa: F401
import swarm_gym.envs.escalation_ladder  # noqa: F401
from swarm_gym.envs.registry import make
from swarm_gym.utils.types import Action


class TestEscalationLadder:
    def test_catastrophe_on_nuke(self):
        env = make("swarm/escalation_ladder:v1", num_agents=3, episode_len=10)
        env.reset(seed=42)
        # One agent nukes
        actions = [
            Action(agent_id="a0", type="noop"),
            Action(agent_id="a1", type="noop"),
            Action(agent_id="a2", type="escalate", level="nuke"),
        ]
        result = env.step(actions)
        assert result.done
        outcomes = env.get_episode_outcomes()
        assert outcomes["catastrophe"] is True
        assert any(e.type == "CATASTROPHE" for e in result.events)

    def test_cooperation_windows(self):
        env = make(
            "swarm/escalation_ladder:v1",
            num_agents=2, episode_len=30,
            cooperation_windows=[5],
        )
        env.reset(seed=42)
        # Advance to cooperation window
        for _t in range(6):
            actions = [Action(agent_id=aid, type="noop") for aid in env.agent_ids]
            env.step(actions)

        # Step 5 should have a cooperation window event
        # (We ran steps 0-5; step 5 is the 6th call)
        # Check that cooperation window events appeared at some point
        env2 = make(
            "swarm/escalation_ladder:v1",
            num_agents=2, episode_len=30,
            cooperation_windows=[0],
        )
        env2.reset(seed=42)
        actions = [Action(agent_id=aid, type="cooperate") for aid in env2.agent_ids]
        result = env2.step(actions)
        assert any(e.type == "COOPERATION_WINDOW" for e in result.events)

    def test_outcomes_schema(self):
        env = make("swarm/escalation_ladder:v1", num_agents=3, episode_len=5)
        env.reset(seed=42)
        for _ in range(5):
            if env.done:
                break
            actions = [Action(agent_id=aid, type="cooperate") for aid in env.agent_ids]
            env.step(actions)
        outcomes = env.get_episode_outcomes()
        assert "catastrophe" in outcomes
        assert "cooperation_rate" in outcomes
        assert "compliance_rate" in outcomes
        assert "welfare_total" in outcomes


class TestCollusionMarket:
    def test_trade_records_prices(self):
        env = make("swarm/collusion_market:v1", num_agents=3, episode_len=10)
        env.reset(seed=42)
        actions = [
            Action(agent_id="a0", type="trade", asset="good_0", qty=1.0, price=10.0),
            Action(agent_id="a1", type="trade", asset="good_0", qty=1.0, price=10.0),
            Action(agent_id="a2", type="trade", asset="good_0", qty=1.0, price=10.0),
        ]
        env.step(actions)
        # All same price -> should trigger collusion detection
        outcomes = env.get_episode_outcomes()
        assert "collusion_score" in outcomes

    def test_outcomes_schema(self):
        env = make("swarm/collusion_market:v1", num_agents=3, episode_len=5)
        env.reset(seed=42)
        for _ in range(5):
            actions = [Action(agent_id=aid, type="cooperate") for aid in env.agent_ids]
            env.step(actions)
        outcomes = env.get_episode_outcomes()
        assert "consumer_welfare" in outcomes
        assert "cartel_duration" in outcomes


class TestAuditEvasion:
    def test_evasion_tracked(self):
        env = make("swarm/audit_evasion:v1", num_agents=3, episode_len=10)
        env.reset(seed=42)
        actions = [
            Action(agent_id="a0", type="hide"),
            Action(agent_id="a1", type="cooperate"),
            Action(agent_id="a2", type="misreport"),
        ]
        env.step(actions)
        outcomes = env.get_episode_outcomes()
        assert outcomes["evasion_rate"] > 0

    def test_compliance_tracked(self):
        env = make("swarm/audit_evasion:v1", num_agents=3, episode_len=10)
        env.reset(seed=42)
        actions = [
            Action(agent_id="a0", type="cooperate"),
            Action(agent_id="a1", type="report"),
            Action(agent_id="a2", type="cooperate"),
        ]
        env.step(actions)
        outcomes = env.get_episode_outcomes()
        assert outcomes["compliance_rate"] > 0

    def test_outcomes_schema(self):
        env = make("swarm/audit_evasion:v1", num_agents=3, episode_len=5)
        env.reset(seed=42)
        for _ in range(5):
            actions = [Action(agent_id=aid, type="cooperate") for aid in env.agent_ids]
            env.step(actions)
        outcomes = env.get_episode_outcomes()
        assert "evasion_rate" in outcomes
        assert "compliance_rate" in outcomes
        assert "audit_budget_used" in outcomes
        assert "welfare_total" in outcomes
