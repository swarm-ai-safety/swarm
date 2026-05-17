"""Mixed population: assigns different sub-policies to agents."""

from __future__ import annotations

import random as _random
from typing import Dict, List, Optional, Type

from swarm_gym.agents.base import AgentPolicy
from swarm_gym.utils.seeding import derive_seed
from swarm_gym.utils.types import Action, AgentId, AgentRecord, Observation


class MixedPopulation(AgentPolicy):
    """A population of agents with heterogeneous policies.

    By default, distributes agents across Honest, Greedy, Aggressive,
    and Random sub-policies. Custom distributions can be specified.
    """

    name = "MixedPop"

    def __init__(
        self,
        distribution: Optional[Dict[str, int]] = None,
    ):
        """
        Args:
            distribution: Mapping of policy name -> count. If None,
                uses a default mix. Policy names must match registered
                sub-policy classes.
        """
        self._distribution = distribution
        self._agent_ids: List[AgentId] = []
        self._agent_policies: Dict[AgentId, AgentPolicy] = {}
        self._agent_types: Dict[AgentId, str] = {}
        self._rng = _random.Random(0)

    def _get_sub_policies(self) -> Dict[str, Type[AgentPolicy]]:
        # Lazy import to avoid circular deps
        from swarm_gym.agents.scripted.random import RandomPolicy
        from swarm_gym.agents.scripted.honest import HonestPolicy
        from swarm_gym.agents.scripted.greedy import GreedyPolicy
        from swarm_gym.agents.scripted.aggressive import AggressivePolicy

        return {
            "Honest": HonestPolicy,
            "Greedy": GreedyPolicy,
            "Aggressive": AggressivePolicy,
            "Random": RandomPolicy,
        }

    def act(self, observations: Dict[AgentId, Observation]) -> List[Action]:
        actions: List[Action] = []
        # Group observations by sub-policy
        policy_obs: Dict[str, Dict[AgentId, Observation]] = {}
        for aid, obs in observations.items():
            ptype = self._agent_types.get(aid, "Random")
            if ptype not in policy_obs:
                policy_obs[ptype] = {}
            policy_obs[ptype][aid] = obs

        # Get actions from each sub-policy
        for ptype, obs_group in policy_obs.items():
            # Find any agent of this type to get its policy
            sample_aid = next(iter(obs_group))
            policy = self._agent_policies.get(sample_aid)
            if policy:
                actions.extend(policy.act(obs_group))
            else:
                # Fallback: noop
                for aid in obs_group:
                    actions.append(Action(agent_id=aid, type="noop"))

        return actions

    def reset(self, agent_ids: List[AgentId], seed: int = 0) -> None:
        self._agent_ids = list(agent_ids)
        self._rng = _random.Random(seed)
        self._agent_policies.clear()
        self._agent_types.clear()

        sub_policies = self._get_sub_policies()

        if self._distribution:
            # Assign according to specified distribution
            assignments: List[str] = []
            for pname, count in self._distribution.items():
                assignments.extend([pname] * count)
            # Pad or truncate to match agent count
            while len(assignments) < len(agent_ids):
                assignments.append(self._rng.choice(list(self._distribution.keys())))
            assignments = assignments[: len(agent_ids)]
            self._rng.shuffle(assignments)
        else:
            # Default: roughly equal mix
            types = list(sub_policies.keys())
            assignments = [types[i % len(types)] for i in range(len(agent_ids))]
            self._rng.shuffle(assignments)

        # Create sub-policy instances and assign agents
        policy_instances: Dict[str, AgentPolicy] = {}
        policy_agents: Dict[str, List[AgentId]] = {}

        for aid, ptype in zip(agent_ids, assignments):
            self._agent_types[aid] = ptype
            if ptype not in policy_instances:
                cls = sub_policies.get(ptype)
                if cls is None:
                    raise ValueError(f"Unknown sub-policy: {ptype}")
                policy_instances[ptype] = cls()
                policy_agents[ptype] = []
            self._agent_policies[aid] = policy_instances[ptype]
            policy_agents[ptype].append(aid)

        # Reset each sub-policy with its assigned agents
        for ptype, policy in policy_instances.items():
            sub_seed = derive_seed(seed, ptype)
            policy.reset(policy_agents[ptype], sub_seed)

    def get_agent_records(self) -> List[AgentRecord]:
        return [
            AgentRecord(agent_id=aid, type=self._agent_types.get(aid, "Unknown"))
            for aid in self._agent_ids
        ]
