"""SWARM-SciAgentGym Bridge.

Connects SWARM's governance and metrics framework to SciAgentGym's
multi-step scientific tool-use benchmarking environment.

SciAgentGym provides 1780+ scientific tools across Physics, Chemistry,
Materials Science, Life Science, and Astronomy domains, enabling agents
to solve complex scientific problems through sequential tool invocation.

Architecture:
    SciAgentGymEnvironment
        ├── creates isolated task environments
        ├── registers domain-specific toolkits
        ├── mounts workspace filesystems
        └── handles environment teardown
    SciAgentGymClient
        └── interfaces with SciAgentGym API
    SciAgentGymMapper
        └── converts SciAgentGym traces → SoftInteractions

Requires: SciAgentGym installation
"""

try:
    # Try importing SciAgentGym core modules
    from gym.env import MinimalSciEnv  # noqa: F401
    from gym.toolbox import Toolbox  # noqa: F401

    SCIAGENTGYM_AVAILABLE = True
except ImportError:
    SCIAGENTGYM_AVAILABLE = False

__all__ = [
    "SCIAGENTGYM_AVAILABLE",
]
