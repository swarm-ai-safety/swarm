"""SWARM-AutoGPT Bridge.

Connects SWARM governance and metrics to AutoGPT-style autonomous agents.
Wraps the AutoGPT thought/command/result loop and maps each action cycle
to a ``SoftInteraction`` scored by ``ProxyComputer``.

Architecture::

    AutoGPT Agent (thought → command → observation loop)
        └── AutoGPTBridge  (this module)
                ├── _extract_observables  (action cycle → ProxyObservables)
                ├── ProxyComputer         (observables → v_hat → p)
                ├── SoftPayoffEngine      (p → payoffs)
                └── EventLog              (append-only audit trail)

Thought/Command mapping::

    task_progress   ← command execution success rate
    rework_count    ← criticisms count / thought confidence
    engagement      ← reasoning depth (thought length)
    verifier_score  ← optional external verifier

AutoGPT action schema (``AutoGPTAction``)::

    {
        "thoughts": {
            "text": str,           # reasoning
            "reasoning": str,      # chain-of-thought
            "plan": str,           # bullet list
            "criticism": str,      # self-critique
            "speak": str           # human-readable summary
        },
        "command": {
            "name": str,           # command name
            "args": dict           # command arguments
        }
    }

Requires: no external dependencies (protocol-level bridge)
"""

from swarm.bridges.autogpt.bridge import (
    AutoGPTAction,
    AutoGPTBridge,
    AutoGPTBridgeError,
)
from swarm.bridges.autogpt.config import AutoGPTBridgeConfig

__all__ = [
    "AutoGPTBridge",
    "AutoGPTAction",
    "AutoGPTBridgeConfig",
    "AutoGPTBridgeError",
]
