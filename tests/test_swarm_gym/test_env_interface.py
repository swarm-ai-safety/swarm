"""Tests for the frozen SwarmEnv interface: reset/step contract."""

import pytest

import swarm_gym.envs.audit_evasion  # noqa: F401
import swarm_gym.envs.collusion_market  # noqa: F401

# Ensure envs are registered
import swarm_gym.envs.escalation_ladder  # noqa: F401
from swarm_gym.envs.base import ResetResult, StepResult
from swarm_gym.envs.registry import list_envs, make
from swarm_gym.utils.types import Action

ALL_ENV_IDS = [
    "swarm/escalation_ladder:v1",
    "swarm/collusion_market:v1",
    "swarm/audit_evasion:v1",
]


@pytest.mark.parametrize("env_id", ALL_ENV_IDS)
class TestSwarmEnvInterface:
    """Test the frozen reset/step interface across all environments."""

    def test_reset_returns_reset_result(self, env_id):
        env = make(env_id)
        result = env.reset(seed=42)
        assert isinstance(result, ResetResult)
        assert len(result.observations) > 0
        assert "env_id" in result.info

    def test_reset_sets_agent_ids(self, env_id):
        env = make(env_id)
        env.reset(seed=42)
        assert len(env.agent_ids) > 0
        assert env.current_step == 0
        assert not env.done

    def test_step_returns_step_result(self, env_id):
        env = make(env_id)
        env.reset(seed=42)
        actions = [Action(agent_id=aid, type="noop") for aid in env.agent_ids]
        result = env.step(actions)
        assert isinstance(result, StepResult)
        assert len(result.observations) > 0
        assert isinstance(result.rewards, dict)
        assert isinstance(result.done, bool)
        assert result.metrics is not None
        assert isinstance(result.events, list)
        assert result.governance is not None

    def test_step_after_done_raises(self, env_id):
        env = make(env_id)
        env.reset(seed=42)
        # Run until done
        for _ in range(env.max_steps + 1):
            if env.done:
                break
            actions = [Action(agent_id=aid, type="noop") for aid in env.agent_ids]
            env.step(actions)
        assert env.done
        with pytest.raises(RuntimeError, match="done"):
            env.step([Action(agent_id=env.agent_ids[0], type="noop")])

    def test_deterministic_with_same_seed(self, env_id):
        """Same seed produces same observations."""
        env1 = make(env_id)
        env2 = make(env_id)
        r1 = env1.reset(seed=123)
        r2 = env2.reset(seed=123)
        for aid in env1.agent_ids:
            assert r1.observations[aid].own_resources == r2.observations[aid].own_resources

    def test_step_increments(self, env_id):
        env = make(env_id)
        env.reset(seed=42)
        assert env.current_step == 0
        actions = [Action(agent_id=aid, type="noop") for aid in env.agent_ids]
        env.step(actions)
        assert env.current_step == 1

    def test_action_space_nonempty(self, env_id):
        env = make(env_id)
        space = env.get_action_space()
        assert len(space) > 0
        assert "noop" in space


class TestRegistry:
    def test_list_envs(self):
        envs = list_envs()
        assert len(envs) >= 3
        for env_id in ALL_ENV_IDS:
            assert env_id in envs

    def test_make_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            make("swarm/nonexistent:v99")
