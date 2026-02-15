"""PettingZoo multi-agent environment bridge for SWARM.

Exposes a SWARM simulation as a PettingZoo ParallelEnv where each agent
in the population is an independent RL actor with its own observation
and action space.  Rewards are SWARM soft payoffs.

Requires: ``pip install pettingzoo>=1.24.0``
"""

__all__ = [
    "SwarmParallelEnv",
    "PettingZooConfig",
]
