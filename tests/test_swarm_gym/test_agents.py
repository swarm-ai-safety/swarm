"""Tests for scripted agent policies."""

import pytest

from swarm_gym.agents.scripted import (
    RandomPolicy, HonestPolicy, GreedyPolicy,
    AggressivePolicy, MixedPopulation,
)
from swarm_gym.utils.types import Action, AgentId, Observation


def _make_observations(agent_ids: list[str]) -> dict[str, Observation]:
    return {
        aid: Observation(
            agent_id=aid,
            step=0,
            visible_agents=agent_ids,
            own_resources=100.0,
            own_reputation=0.5,
            governance_state={"escalation_risk": 0.3},
        )
        for aid in agent_ids
    }


AGENT_IDS = ["a0", "a1", "a2", "a3"]


class TestRandomPolicy:
    def test_returns_actions_for_all_agents(self):
        policy = RandomPolicy()
        policy.reset(AGENT_IDS, seed=42)
        obs = _make_observations(AGENT_IDS)
        actions = policy.act(obs)
        assert len(actions) == len(AGENT_IDS)
        assert all(isinstance(a, Action) for a in actions)

    def test_deterministic_with_seed(self):
        p1 = RandomPolicy()
        p2 = RandomPolicy()
        p1.reset(AGENT_IDS, seed=99)
        p2.reset(AGENT_IDS, seed=99)
        obs = _make_observations(AGENT_IDS)
        a1 = p1.act(obs)
        a2 = p2.act(obs)
        assert [a.type for a in a1] == [a.type for a in a2]


class TestHonestPolicy:
    def test_never_escalates(self):
        policy = HonestPolicy()
        policy.reset(AGENT_IDS, seed=42)
        obs = _make_observations(AGENT_IDS)
        # Run many times to check no escalation
        for _ in range(20):
            actions = policy.act(obs)
            for a in actions:
                assert a.type in ("cooperate", "trade", "message")


class TestGreedyPolicy:
    def test_returns_valid_actions(self):
        policy = GreedyPolicy()
        policy.reset(AGENT_IDS, seed=42)
        obs = _make_observations(AGENT_IDS)
        actions = policy.act(obs)
        assert all(a.type in ("cooperate", "trade", "defect", "noop") for a in actions)


class TestAggressivePolicy:
    def test_includes_escalation(self):
        policy = AggressivePolicy()
        policy.reset(AGENT_IDS, seed=42)
        obs = _make_observations(AGENT_IDS)
        types_seen = set()
        for _ in range(50):
            actions = policy.act(obs)
            for a in actions:
                types_seen.add(a.type)
        assert "escalate" in types_seen


class TestMixedPopulation:
    def test_assigns_different_types(self):
        policy = MixedPopulation()
        policy.reset(AGENT_IDS, seed=42)
        records = policy.get_agent_records()
        types = {r.type for r in records}
        assert len(types) >= 2  # Should have at least 2 different types

    def test_custom_distribution(self):
        policy = MixedPopulation(distribution={"Honest": 2, "Greedy": 2})
        policy.reset(AGENT_IDS, seed=42)
        records = policy.get_agent_records()
        type_counts = {}
        for r in records:
            type_counts[r.type] = type_counts.get(r.type, 0) + 1
        assert type_counts.get("Honest", 0) == 2
        assert type_counts.get("Greedy", 0) == 2

    def test_act_returns_actions(self):
        policy = MixedPopulation()
        policy.reset(AGENT_IDS, seed=42)
        obs = _make_observations(AGENT_IDS)
        actions = policy.act(obs)
        assert len(actions) == len(AGENT_IDS)
