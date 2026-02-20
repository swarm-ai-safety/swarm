"""Contract interface and concrete contract types.

Each contract defines:
- A signing cost (bonds, fees, audits) that creates costly signaling
- An execution protocol that modifies interaction parameters
- A penalty mechanism for contract violations

The key design lever: signing must be costly or verifiable so that
cooperative agents find it worthwhile but adversarial strategies find
margins eaten. This is Spence-style costly signaling applied to
multi-agent governance.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict

from swarm.models.agent import AgentState, AgentType
from swarm.models.interaction import SoftInteraction


class ContractType(Enum):
    """Enumeration of available contract types."""

    TRUTHFUL_AUCTION = "truthful_auction"
    FAIR_DIVISION = "fair_division"
    DEFAULT_MARKET = "default_market"


@dataclass
class ContractDecision:
    """Record of an agent's contract sign/refuse decision."""

    agent_id: str
    agent_type: AgentType
    contract_chosen: ContractType
    signing_cost_paid: float
    epoch: int
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for event logging."""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "contract_chosen": self.contract_chosen.value,
            "signing_cost_paid": self.signing_cost_paid,
            "epoch": self.epoch,
            "reason": self.reason,
        }


class Contract(ABC):
    """Abstract base class for interaction protocols.

    A contract is a governance protocol that agents can opt into. It sits
    between the orchestrator's handoff logic and the actual agent-to-agent
    interaction, modifying payoff parameters and enforcing rules.

    Lifecycle: offer() -> sign(agent) -> execute(interaction) -> penalize()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Contract name for logging."""

    @property
    @abstractmethod
    def contract_type(self) -> ContractType:
        """Contract type enum value."""

    @abstractmethod
    def signing_cost(self, agent: AgentState) -> float:
        """Cost for this agent to sign the contract.

        This is the key screening lever. The cost must be calibrated so
        cooperative agents find it worthwhile but adversarial strategies
        find their margins eaten.

        Args:
            agent: The agent considering signing.

        Returns:
            Non-negative cost to sign.
        """

    @abstractmethod
    def execute(self, interaction: SoftInteraction) -> SoftInteraction:
        """Apply the contract protocol to an interaction.

        May modify payoff parameters (tau, c_a, c_b), add metadata,
        or transform the interaction in protocol-specific ways.

        Args:
            interaction: The raw interaction.

        Returns:
            Modified interaction with contract protocol applied.
        """

    @abstractmethod
    def penalize(self, agent_id: str, violation_p: float) -> float:
        """Compute penalty for a contract violation.

        Args:
            agent_id: The violating agent.
            violation_p: The p value that triggered the violation.

        Returns:
            Penalty amount (non-negative).
        """

    def should_audit(self, interaction: SoftInteraction, rng: random.Random) -> bool:
        """Whether this interaction should be audited under contract rules.

        Default: no audits. Override for contracts with audit mechanisms.

        Args:
            interaction: The interaction to potentially audit.
            rng: Random number generator for reproducibility.

        Returns:
            True if interaction should be audited.
        """
        return False


class TruthfulAuctionContract(Contract):
    """Vickrey/VCG-style truthful mechanism contract.

    Key properties:
    - Truthful bidding is a dominant strategy (second-price logic)
    - Staking requirement creates costly commitment
    - Random audits with slashing for detected manipulation
    - Protocol constraints that hinder common attack strategies

    Signing gives access to better counterparties and lower friction,
    but requires a bond and subjects the agent to audits.
    """

    def __init__(
        self,
        stake_fraction: float = 0.05,
        audit_probability: float = 0.2,
        audit_threshold_p: float = 0.4,
        penalty_multiplier: float = 3.0,
        surplus_bonus: float = 0.1,
    ):
        """Initialize truthful auction contract.

        Args:
            stake_fraction: Fraction of resources required as bond.
            audit_probability: Probability of random audit per interaction.
            audit_threshold_p: p threshold below which audit triggers penalty.
            penalty_multiplier: Multiplier for violation penalties.
            surplus_bonus: Bonus added to surplus share for signed agents.
        """
        if not 0.0 <= stake_fraction <= 1.0:
            raise ValueError("stake_fraction must be in [0, 1]")
        if not 0.0 <= audit_probability <= 1.0:
            raise ValueError("audit_probability must be in [0, 1]")
        if not 0.0 <= audit_threshold_p <= 1.0:
            raise ValueError("audit_threshold_p must be in [0, 1]")
        if penalty_multiplier < 0:
            raise ValueError("penalty_multiplier must be non-negative")
        if surplus_bonus < 0:
            raise ValueError("surplus_bonus must be non-negative")

        self.stake_fraction = stake_fraction
        self.audit_probability = audit_probability
        self.audit_threshold_p = audit_threshold_p
        self.penalty_multiplier = penalty_multiplier
        self.surplus_bonus = surplus_bonus

    @property
    def name(self) -> str:
        return "truthful_auction"

    @property
    def contract_type(self) -> ContractType:
        return ContractType.TRUTHFUL_AUCTION

    def signing_cost(self, agent: AgentState) -> float:
        """Bond proportional to agent's resources."""
        return max(0.0, agent.resources * self.stake_fraction)

    def execute(self, interaction: SoftInteraction) -> SoftInteraction:
        """Apply truthful mechanism: reduce governance cost, add surplus bonus.

        Under the truthful protocol:
        - Lower governance friction (c_a, c_b reduced) because the protocol
          handles dispute resolution
        - Surplus bonus rewards participation in the governed pool
        - Metadata tracks contract provenance
        """
        # Create modified copy via model_copy
        updates: Dict[str, Any] = {
            "c_a": max(0.0, interaction.c_a * 0.5),  # Lower friction
            "c_b": max(0.0, interaction.c_b * 0.5),
            "r_a": interaction.r_a + self.surplus_bonus,  # Reputation bonus
            "metadata": {
                **interaction.metadata,
                "contract": self.name,
                "contract_type": self.contract_type.value,
                "audit_eligible": True,
            },
        }
        return interaction.model_copy(update=updates)

    def penalize(self, agent_id: str, violation_p: float) -> float:
        """Penalty proportional to how far below audit threshold."""
        if violation_p >= self.audit_threshold_p:
            return 0.0
        shortfall = self.audit_threshold_p - violation_p
        return shortfall * self.penalty_multiplier

    def should_audit(self, interaction: SoftInteraction, rng: random.Random) -> bool:
        """Probabilistic audit based on contract audit rate."""
        return rng.random() < self.audit_probability


class FairDivisionContract(Contract):
    """Envy-free allocation protocol contract.

    Key properties:
    - Guarantees proportional fairness of allocation
    - Lower signing cost than truthful auction (less costly signaling)
    - No audit mechanism (fairness is enforced structurally)
    - Good for resource sharing and task distribution

    Important nuance: envy-free != strategyproof. This contract guarantees
    fairness of the allocation given expressed preferences, but does NOT
    guarantee that truthful reporting is a dominant strategy. Agents can
    still manipulate their reported preferences.
    """

    def __init__(
        self,
        entry_fee: float = 2.0,
        fairness_bonus: float = 0.05,
        redistribution_rate: float = 0.1,
    ):
        """Initialize fair division contract.

        Args:
            entry_fee: Fixed fee to participate (flat, not proportional).
            fairness_bonus: Reputation bonus for fair interactions.
            redistribution_rate: Fraction of surplus redistributed equally.
        """
        if entry_fee < 0:
            raise ValueError("entry_fee must be non-negative")
        if fairness_bonus < 0:
            raise ValueError("fairness_bonus must be non-negative")
        if not 0.0 <= redistribution_rate <= 1.0:
            raise ValueError("redistribution_rate must be in [0, 1]")

        self.entry_fee = entry_fee
        self.fairness_bonus = fairness_bonus
        self.redistribution_rate = redistribution_rate

    @property
    def name(self) -> str:
        return "fair_division"

    @property
    def contract_type(self) -> ContractType:
        return ContractType.FAIR_DIVISION

    def signing_cost(self, agent: AgentState) -> float:
        """Flat entry fee."""
        return self.entry_fee

    def execute(self, interaction: SoftInteraction) -> SoftInteraction:
        """Apply fair division: redistribute surplus, add fairness bonus.

        Under fair division:
        - Tau is adjusted toward 0 (more equal surplus split)
        - Small fairness reputation bonus for both parties
        - Metadata tracks contract provenance
        """
        # Move tau toward 0 by redistribution_rate
        adjusted_tau = interaction.tau * (1.0 - self.redistribution_rate)

        updates: Dict[str, Any] = {
            "tau": adjusted_tau,
            "r_a": interaction.r_a + self.fairness_bonus,
            "r_b": interaction.r_b + self.fairness_bonus,
            "metadata": {
                **interaction.metadata,
                "contract": self.name,
                "contract_type": self.contract_type.value,
                "redistribution_applied": True,
            },
        }
        return interaction.model_copy(update=updates)

    def penalize(self, agent_id: str, violation_p: float) -> float:
        """Minimal penalty - fairness is enforced structurally."""
        return 0.0


class DefaultMarket(Contract):
    """Baseline market with no special protections.

    Agents who refuse all contracts are routed here. No signing cost,
    no protocol enforcement, no audits. Higher friction and no
    reputation bonus compared to governed contracts.

    In theory, if contract screening works, adversarial agents will
    concentrate here and end up targeting each other (the containment-
    through-self-selection hypothesis).
    """

    def __init__(
        self,
        friction_premium: float = 0.05,
    ):
        """Initialize default market.

        Args:
            friction_premium: Additional governance cost added to
                interactions in the unprotected pool.
        """
        if friction_premium < 0:
            raise ValueError("friction_premium must be non-negative")
        self.friction_premium = friction_premium

    @property
    def name(self) -> str:
        return "default_market"

    @property
    def contract_type(self) -> ContractType:
        return ContractType.DEFAULT_MARKET

    def signing_cost(self, agent: AgentState) -> float:
        """No cost to enter the default market."""
        return 0.0

    def execute(self, interaction: SoftInteraction) -> SoftInteraction:
        """Default market adds friction, no protections."""
        updates: Dict[str, Any] = {
            "c_a": interaction.c_a + self.friction_premium,
            "c_b": interaction.c_b + self.friction_premium,
            "metadata": {
                **interaction.metadata,
                "contract": self.name,
                "contract_type": self.contract_type.value,
            },
        }
        return interaction.model_copy(update=updates)

    def penalize(self, agent_id: str, violation_p: float) -> float:
        """No penalty in the default market."""
        return 0.0
