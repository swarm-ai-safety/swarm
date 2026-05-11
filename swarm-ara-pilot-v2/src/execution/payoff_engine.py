"""
Soft Payoff Engine: Distributional Payoff Computation and Governance Application

Implements payoff decomposition with multiple governance levers:
1. Soft surplus: S_soft = p * s_plus - (1-p) * s_minus
2. Transaction tax: tau * S_soft
3. Externality internalization: rho * (1-p) * h
4. Audit sampling and reputation dynamics
5. Circuit breaker hard-stop

Implementation conforms to Algorithm section (Eq. 4–11) and Architecture section.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class GovernanceParams:
    """
    Governance lever parameters: tau, rho, theta_CB, lambda, p_audit.

    All validated against bounds from training.md and heuristics.md.
    """
    tau: float = 0.10  # Transaction tax rate, [0.0, 0.30]
    rho: float = 0.10  # Externality internalization, [0.0, 1.0]
    theta_CB: float = 0.35  # Circuit breaker threshold, [0.2, 0.8]
    lambda_decay: float = 0.85  # Reputation decay, (0.0, 1.0)
    p_audit: float = 0.10  # Audit probability, [0.0, 0.5]

    def validate(self):
        """Check bounds on all parameters."""
        assert 0.0 <= self.tau <= 0.30, f"tau out of range: {self.tau}"
        assert 0.0 <= self.rho <= 1.0, f"rho out of range: {self.rho}"
        assert 0.2 <= self.theta_CB <= 0.8, f"theta_CB out of range: {self.theta_CB}"
        assert 0.0 < self.lambda_decay < 1.0, f"lambda_decay out of range: {self.lambda_decay}"
        assert 0.0 <= self.p_audit <= 0.5, f"p_audit out of range: {self.p_audit}"


@dataclass
class PayoffStructure:
    """
    Payoff parameters per scenario.

    From model.md and Table 3:
    - s_plus: surplus if interaction beneficial
    - s_minus: loss if interaction harmful (typically 0 or negative)
    - h: ecosystem harm externality cost
    """
    s_plus: float = 1.0
    s_minus: float = 0.0
    h: float = 0.5

    def validate(self):
        """Check payoff structure bounds."""
        assert self.s_plus > 0, f"s_plus must be positive: {self.s_plus}"
        assert self.h > 0, f"h must be positive: {self.h}"


class SoftPayoffEngine:
    """
    Computes expected payoffs under distributional uncertainty (soft labels)
    and applies governance levers.

    Payoff decomposition (Eq. 8–11, Algorithm section):
    Π_soft = S_soft
           - tau * S_soft              (transaction tax)
           - rho * (1-p) * h           (externality internalization)
           + L_rep                     (reputation incentive)
           - 𝟙[p < theta_CB] * ∞      (circuit breaker)

    Key design choices (Architecture section):
    - Payoff decomposes additively (A6): each lever is independent cost term
    - Circuit breaker is hard constraint applied after payoff computation
    - Reputation accumulates over time with decay; audit outcomes sampled randomly
    - Complexity: O(H) per interaction where H = reputation history length
    """

    def __init__(
        self,
        governance_params: GovernanceParams = None,
        payoff_structure: PayoffStructure = None,
        random_state: np.random.Generator = None
    ):
        """
        Initialize payoff engine with governance and payoff parameters.

        Args:
            governance_params: GovernanceParams object; if None, uses defaults
            payoff_structure: PayoffStructure object; if None, uses defaults
            random_state: RNG for audit sampling; if None, creates new RNG
        """
        self.governance = governance_params or GovernanceParams()
        self.governance.validate()

        self.payoff_structure = payoff_structure or PayoffStructure()
        self.payoff_structure.validate()

        self.random_state = random_state or np.random.default_rng(seed=42)

    def compute_soft_surplus(self, p: float) -> float:
        """
        Compute expected surplus under uncertainty.

        Implements Eq. 4–5 (Algorithm section):
        S_soft = p * s_plus - (1-p) * s_minus

        Args:
            p: Soft label (probability beneficial), in [0, 1]

        Returns:
            Expected surplus S_soft
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        s_plus = self.payoff_structure.s_plus
        s_minus = self.payoff_structure.s_minus

        return p * s_plus - (1 - p) * s_minus

    def compute_soft_externality(self, p: float) -> float:
        """
        Compute expected harm externality.

        Implements Eq. 5 (Algorithm section):
        E_soft = (1-p) * h

        Args:
            p: Soft label, in [0, 1]

        Returns:
            Expected externality cost E_soft
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        h = self.payoff_structure.h
        return (1 - p) * h

    def _apply_tax(self, s_soft: float) -> float:
        """
        Apply transaction tax to surplus.

        Cost = tau * S_soft, where tau in [0.0, 0.30] (H07).

        Args:
            s_soft: Expected surplus from compute_soft_surplus

        Returns:
            Tax cost (positive value to be subtracted from payoff)
        """
        tau = self.governance.tau
        return tau * s_soft

    def _apply_externality_internalization(self, p: float) -> float:
        """
        Apply externality internalization cost.

        Cost = rho * (1-p) * h, where rho in [0.0, 1.0] (H07).

        Key finding (C03): aggressive rho >= 0.7 collapses welfare without
        reducing toxicity; must pair with adaptive theta_CB for Pareto improvement.

        Args:
            p: Soft label, in [0, 1]

        Returns:
            Externality cost (positive value to be subtracted from payoff)
        """
        rho = self.governance.rho
        h = self.payoff_structure.h
        return rho * (1 - p) * h

    def _apply_circuit_breaker(self, p: float) -> bool:
        """
        Apply circuit breaker hard-stop.

        Rejection criterion (H06): if p < theta_CB, reject (accept=False);
        otherwise, accept is determined by payoff sign.

        Optimal region (Table 6b): theta_CB in [0.35, 0.50];
        sensitivity High: ±0.05 shifts welfare and toxicity significantly.

        Args:
            p: Soft label, in [0, 1]

        Returns:
            True if accept, False if reject
        """
        theta_CB = self.governance.theta_CB
        return p >= theta_CB

    def _update_reputation(
        self,
        reputation_history: List[float],
        p: float
    ) -> float:
        """
        Update reputation based on audit outcome.

        Mechanism (H05):
        - Sample audit: u ~ Uniform[0,1]; if u < p_audit, audit_triggered=True
        - Audit outcome: if audit_triggered, audit_passes ~ Bernoulli(p)
        - Reputation update: if audit_passes, delta = +1.0; else delta = -0.5
        - Decay: reputation_t = reputation_{t-1} * lambda + delta

        Args:
            reputation_history: List of past reputation changes
            p: Soft label (audit pass probability correlates with p)

        Returns:
            Reputation delta (change in reputation from this interaction)
        """
        p_audit = self.governance.p_audit
        lambda_decay = self.governance.lambda_decay

        # Compute current reputation before update
        current_rep = 0.0
        if reputation_history:
            for i, past_delta in enumerate(reputation_history):
                # Decay: more recent updates weighted more heavily
                age = len(reputation_history) - i - 1
                current_rep += past_delta * (lambda_decay ** age)

        # Sample audit
        audit_triggered = self.random_state.random() < p_audit
        delta_rep = 0.0

        if audit_triggered:
            # Audit outcome correlated with p
            audit_passes = self.random_state.random() < p
            delta_rep = 1.0 if audit_passes else -0.5

        return delta_rep

    def compute_payoff(
        self,
        p: float,
        reputation_history: List[float] = None,
        return_components: bool = False
    ) -> Dict[str, float]:
        """
        Compute expected payoff under all governance levers.

        Full decomposition (Eq. 8–11, Algorithm section):
        Π_soft = S_soft
               - tau * S_soft              (transaction tax)
               - rho * (1-p) * h           (externality internalization)
               + L_rep                     (reputation incentive)
               - 𝟙[p < theta_CB] * ∞      (circuit breaker: reject if p < CB)

        Args:
            p: Soft label (probability beneficial), in [0, 1]
            reputation_history: List of past reputation deltas; if None, assumed empty
            return_components: If True, return breakdown of payoff components

        Returns:
            Dict with keys:
            - payoff: Final payoff after all governance (0 if rejected)
            - accept: Boolean (True if p >= theta_CB and payoff > 0)
            - s_soft: Expected surplus
            - externality_cost: Internalized externality cost
            - tax_cost: Transaction tax cost
            - reputation_delta: Reputation change from audit
            - (optional) components: Full breakdown if return_components=True
        """
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")

        reputation_history = reputation_history or []

        # Compute payoff components
        s_soft = self.compute_soft_surplus(p)
        tax_cost = self._apply_tax(s_soft)
        externality_cost = self._apply_externality_internalization(p)
        reputation_delta = self._update_reputation(reputation_history, p)

        # Apply circuit breaker
        accept = self._apply_circuit_breaker(p)

        # Compute net payoff
        if accept:
            payoff_net = s_soft - tax_cost - externality_cost + reputation_delta
            # If payoff is negative after governance, still reject
            if payoff_net < 0:
                accept = False
                payoff_net = 0.0
        else:
            payoff_net = 0.0

        result = {
            'payoff': payoff_net,
            'accept': accept,
            's_soft': s_soft,
            'externality_cost': externality_cost,
            'tax_cost': tax_cost,
            'reputation_delta': reputation_delta
        }

        if return_components:
            result['components'] = {
                's_soft': s_soft,
                'tax_cost': tax_cost,
                'externality_cost': externality_cost,
                'reputation_delta': reputation_delta,
                'circuit_breaker_applied': not accept
            }

        return result
