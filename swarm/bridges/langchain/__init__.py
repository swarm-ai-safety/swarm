"""SWARM-LangChain Bridge.

Wraps LangChain chains and agent executors as SWARM participants,
enabling governance testing on LLM agent pipelines.

Architecture::

    LangChain Chain / AgentExecutor
        └── LangChainBridge  (this module)
                ├── _extract_observables  (chain output → ProxyObservables)
                ├── ProxyComputer         (observables → v_hat → p)
                ├── SoftPayoffEngine      (p → payoffs)
                └── EventLog              (append-only audit trail)

Observable extraction heuristics::

    task_progress   ← did chain complete without error?  (1.0 / 0.0)
    rework_count    ← number of intermediate steps (retries / tool calls)
    engagement      ← output length as a proxy for thoroughness
    verifier_score  ← optional: caller-supplied quality score

Requires: ``pip install langchain langchain-core``
"""

from swarm.bridges.langchain.bridge import LangChainBridge, LangChainBridgeError
from swarm.bridges.langchain.config import LangChainBridgeConfig

__all__ = [
    "LangChainBridge",
    "LangChainBridgeConfig",
    "LangChainBridgeError",
]
