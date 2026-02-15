"""Tests for the PettingZoo multi-agent bridge."""

import pytest

try:
    import gymnasium  # noqa: F401
    import pettingzoo  # noqa: F401

    HAS_PETTINGZOO = True
except ImportError:
    HAS_PETTINGZOO = False

pytestmark = pytest.mark.skipif(
    not HAS_PETTINGZOO,
    reason="pettingzoo / gymnasium not installed",
)


@pytest.fixture
def env():
    from swarm.bridges.pettingzoo.environment import SwarmParallelEnv

    return SwarmParallelEnv()


@pytest.fixture
def env_small():
    from swarm.bridges.pettingzoo.config import PettingZooConfig
    from swarm.bridges.pettingzoo.environment import SwarmParallelEnv

    config = PettingZooConfig(
        n_agents=4,
        max_steps=10,
        agent_types={"honest": 0.5, "deceptive": 0.5},
    )
    return SwarmParallelEnv(config=config)


class TestSwarmParallelEnv:
    def test_reset_returns_observations_for_all_agents(self, env):
        observations, infos = env.reset(seed=42)
        assert set(observations.keys()) == set(env.agents)
        assert set(infos.keys()) == set(env.agents)
        for obs in observations.values():
            assert obs.shape[0] > 0
            assert obs.dtype.name.startswith("float")

    def test_agents_match_possible_agents_after_reset(self, env):
        env.reset(seed=1)
        assert env.agents == env.possible_agents

    def test_step_returns_correct_keys(self, env):
        env.reset(seed=42)
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        # All dicts should have entries for agents that were alive
        for a in rewards:
            assert isinstance(rewards[a], float)
            assert isinstance(terms[a], bool)
            assert isinstance(truncs[a], bool)

    def test_observation_space_valid(self, env):
        env.reset(seed=42)
        observations, _ = env.reset(seed=42)
        for agent, obs in observations.items():
            space = env.observation_space(agent)
            assert space.contains(obs), f"obs for {agent} not in space"

    def test_action_space_is_discrete_4(self, env):
        for agent in env.possible_agents:
            space = env.action_space(agent)
            assert space.n == 4

    def test_episode_truncates_at_max_steps(self, env_small):
        env_small.reset(seed=42)
        for _ in range(20):  # more than max_steps=10
            if not env_small.agents:
                break
            actions = {
                a: env_small.action_space(a).sample() for a in env_small.agents
            }
            _, _, _, truncs, _ = env_small.step(actions)
        # Episode should have ended
        assert env_small.agents == [] or all(
            truncs.get(a, False) for a in env_small.possible_agents
        )

    def test_reject_action_produces_no_accepted_interaction(self, env_small):
        env_small.reset(seed=42)
        # All agents reject
        actions = dict.fromkeys(env_small.agents, 0)
        env_small.step(actions)
        interactions = env_small.get_interactions()
        for ix in interactions:
            assert not ix.accepted

    def test_p_invariant_holds(self, env):
        """p must remain in [0, 1] across all interactions."""
        env.reset(seed=42)
        for _ in range(5):
            if not env.agents:
                break
            actions = {a: env.action_space(a).sample() for a in env.agents}
            env.step(actions)
        for ix in env.get_interactions():
            assert 0.0 <= ix.p <= 1.0, f"p={ix.p} out of bounds"

    def test_metrics_available(self, env):
        env.reset(seed=42)
        for _ in range(3):
            if not env.agents:
                break
            actions = dict.fromkeys(env.agents, 1)  # all accept
            env.step(actions)
        metrics = env.get_metrics()
        assert "toxicity_rate" in metrics
        assert "quality_gap" in metrics
        assert "mean_p" in metrics
        assert metrics["n_interactions"] > 0

    def test_events_recorded(self, env_small):
        env_small.reset(seed=42)
        actions = dict.fromkeys(env_small.agents, 1)
        env_small.step(actions)
        events = env_small.get_events()
        assert len(events) > 0
        types = {e.event_type.value for e in events}
        assert "episode_reset" in types
        assert "step_completed" in types

    def test_render_ansi(self):
        from swarm.bridges.pettingzoo.config import PettingZooConfig
        from swarm.bridges.pettingzoo.environment import SwarmParallelEnv

        config = PettingZooConfig(n_agents=3, agent_types={"honest": 1.0})
        env = SwarmParallelEnv(config=config, render_mode="ansi")
        env.reset(seed=42)
        output = env.render()
        assert "SWARM PettingZoo" in output
        assert "Alive:" in output

    def test_state_returns_global_vector(self, env):
        env.reset(seed=42)
        s = env.state()
        assert s.dtype.name.startswith("float")
        # 2 floats per agent + 3 ecosystem metrics
        expected_len = len(env.possible_agents) * 2 + 3
        assert len(s) == expected_len

    def test_seed_reproducibility(self):
        from swarm.bridges.pettingzoo.environment import SwarmParallelEnv

        env1 = SwarmParallelEnv()
        env2 = SwarmParallelEnv()

        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)

        for a in env1.agents:
            assert (obs1[a] == obs2[a]).all()

    def test_exploit_action_sets_misuse_flag(self, env_small):
        env_small.reset(seed=42)
        # All agents exploit
        actions = dict.fromkeys(env_small.agents, 2)
        env_small.step(actions)
        interactions = env_small.get_interactions()
        assert any(ix.tool_misuse_flags > 0 for ix in interactions)

    def test_signal_action_costs_resources(self, env_small):
        env_small.reset(seed=42)
        initial_resources = dict(env_small._resources)
        actions = dict.fromkeys(env_small.agents, 3)  # all signal
        env_small.step(actions)
        for a in env_small.possible_agents:
            if a in initial_resources:
                # Resources should decrease (signal cost) minus any payoff
                assert env_small._resources.get(a, 0.0) != initial_resources[a]


class TestPettingZooConfig:
    def test_default_config_valid(self):
        from swarm.bridges.pettingzoo.config import PettingZooConfig

        config = PettingZooConfig()
        assert config.n_agents == 5
        assert config.max_steps == 50

    def test_rejects_single_agent(self):
        from swarm.bridges.pettingzoo.config import PettingZooConfig

        with pytest.raises(ValueError, match="n_agents"):
            PettingZooConfig(n_agents=1)

    def test_rejects_bad_fractions(self):
        from swarm.bridges.pettingzoo.config import PettingZooConfig

        with pytest.raises(ValueError, match="sum to 1"):
            PettingZooConfig(agent_types={"honest": 0.3, "deceptive": 0.3})
