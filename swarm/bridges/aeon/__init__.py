"""SWARM ↔ Aeon bridge.

Reads Aeon's append-only agent-first JSONL ledgers (tasks, proofs, reviews)
from a local Aeon checkout — and optionally GitHub Actions skill-run
conclusions via the ``gh`` CLI — translating each record into a SWARM
``SoftInteraction`` so Aeon's *real* autonomous multi-agent activity can be
scored with the same soft-metrics as simulated scenarios.

Unlike the Gitlawb bridge (live GraphQL subscription), this is a filesystem
source: no network transport, no async client deps.

    Aeon checkout (memory/agent-first/*.jsonl)
        └── AeonClient        (reads ledgers + optional `gh run list`)
                └── AeonMapper        (record -> SoftInteraction, soft-labeled)
                        └── AeonMetrics       (SWARM soft-metrics report)
                                ↑ orchestrated by AeonRunner (oneshot | watch)
"""

from swarm.bridges.aeon.client import AeonClient
from swarm.bridges.aeon.config import AeonConfig
from swarm.bridges.aeon.mapper import AeonMapper
from swarm.bridges.aeon.metrics import AeonMetrics, AeonMetricsReport
from swarm.bridges.aeon.runner import AeonRunner

__all__ = [
    "AeonClient",
    "AeonConfig",
    "AeonMapper",
    "AeonMetrics",
    "AeonMetricsReport",
    "AeonRunner",
]
