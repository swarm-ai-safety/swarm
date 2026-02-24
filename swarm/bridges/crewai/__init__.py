"""SWARM-CrewAI Bridge.

Integrates CrewAI crews as SWARM participants, mapping crew task
completion events to ``SoftInteraction`` objects for governance testing.

This bridge is distinct from ``swarm.agents.crewai_adapter``:

- **Agent adapter** (``swarm.agents.crewai_adapter``): Wraps a crew as a
  single SWARM agent that produces one action per simulation step.
- **Bridge** (this module): Wraps crew execution as a SWARM interaction
  source that generates multiple interactions from a single crew run,
  one per task assigned/completed.  Suitable for studying how governance
  mechanisms interact with collaborative AI team dynamics.

Architecture::

    CrewAI Crew
        └── CrewAIBridge  (this module)
                ├── _task_to_interaction  (task result → SoftInteraction)
                ├── ProxyComputer         (observables → v_hat → p)
                ├── SoftPayoffEngine      (p → payoffs)
                └── EventLog              (append-only audit trail)

Task result mapping::

    task_progress   ← task completed without error
    rework_count    ← agent delegation depth (nested calls)
    engagement      ← output length normalised to [0, 1]
    verifier_score  ← optional external quality score

Requires: ``pip install crewai``
"""

from swarm.bridges.crewai.bridge import CrewAIBridge, CrewAIBridgeError, TaskResult
from swarm.bridges.crewai.config import CrewAIBridgeConfig

__all__ = [
    "CrewAIBridge",
    "CrewAIBridgeConfig",
    "CrewAIBridgeError",
    "TaskResult",
]
