"""Coasean Singularity Markets (CSM) benchmark for SWARM.

Evaluates whether multi-agent market participation reduces transaction
costs and improves welfare without triggering classic equilibrium
failures (congestion, obfuscation arms races, adverse selection,
lock-in, identity/Sybil issues).

Based on:

    Shahidi, P., Rusak, G., Manning, B. S., Fradkin, A., & Horton, J. J.
    (2025). "The Coasean Singularity? Demand, Supply, and Market Design
    with AI Agents." In *The Economics of Transformative AI*, Chapter 6.
    University of Chicago Press / NBER.
    https://www.nber.org/system/files/chapters/c15309/c15309.pdf

Modules
-------
- search_purchase : E-commerce search under obfuscation
- matching        : Two-sided matching with preference elicitation
- negotiation     : Bilateral bargaining with compute budgets
- platform_access : Platform lock-in and BYO-agent degradation
- identity        : Sybil / proof-of-personhood stress tests

Each module can be run independently or composed into multi-market
benchmark suites.
"""

from swarm.csm.types import (
    AgentOwnership,
    AgentSpecialization,
    CSMEpisodeRecord,
    MarketModule,
    PreferenceDimensionality,
    TransactionCostRegime,
)

__all__ = [
    "AgentOwnership",
    "AgentSpecialization",
    "CSMEpisodeRecord",
    "MarketModule",
    "PreferenceDimensionality",
    "TransactionCostRegime",
]
