"""Contract screening and interaction protocol layer.

Implements a pre-play contract stage where agents opt into governance
protocols that enforce desirable interaction properties (truthfulness,
fairness, etc.). Agents who refuse are routed to a default market with
no special protections.

Key abstractions:
- Contract: abstract protocol agents can sign
- TruthfulAuctionContract: Vickrey/VCG-style truthful mechanism
- FairDivisionContract: envy-free allocation protocol
- DefaultMarket: baseline with no protections
- ContractMarket: orchestrates sign/refuse stage and routing
"""

from swarm.contracts.behavioral import (
    BehavioralContract,
    DriftDetector,
    InvariantCheck,
    Precondition,
    RecoveryPolicy,
    StageGuarantee,
    compute_pipeline_bound,
    compute_pipeline_bound_with_drift,
)
from swarm.contracts.contract import (
    Contract,
    ContractDecision,
    ContractType,
    DefaultMarket,
    FairDivisionContract,
    TruthfulAuctionContract,
)
from swarm.contracts.market import ContractMarket, ContractMarketConfig
from swarm.contracts.metrics import ContractMetrics

__all__ = [
    "BehavioralContract",
    "Contract",
    "ContractDecision",
    "ContractMarket",
    "ContractMarketConfig",
    "ContractMetrics",
    "ContractType",
    "DefaultMarket",
    "DriftDetector",
    "FairDivisionContract",
    "InvariantCheck",
    "Precondition",
    "RecoveryPolicy",
    "StageGuarantee",
    "TruthfulAuctionContract",
    "compute_pipeline_bound",
    "compute_pipeline_bound_with_drift",
]
