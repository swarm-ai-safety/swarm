"""Diversity as Defense (DaD) governance lever.

Implements the governance principle: maintain sufficient agent heterogeneity
to keep error correlations below a controlled threshold; minimize tail risk
(correlation-amplified failure), not mean error.

The lever tracks pairwise error correlations across agent types and enforces:
  Rule 1 - Correlation cap: rho_bar(x) <= rho_max
  Rule 2 - Entropy floor: H(x) >= H_min
  Rule 3 - Adversarial fraction: sum of adversarial type weights >= alpha_min
  Rule 4 - Disagreement-triggered audit: D(t) >= tau triggers audit
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.agent import AgentType
from swarm.models.interaction import SoftInteraction


@dataclass
class DiversityMetrics:
    """Snapshot of diversity-related measurements.

    Captures the state of the population mix, error correlations, and
    rule compliance at a point in time.
    """

    # Population mix x_k for each agent type
    population_mix: Dict[str, float] = field(default_factory=dict)

    # Mean error rate across all agents
    mean_error_rate: float = 0.0

    # Mean pairwise error correlation (rho_bar)
    mean_correlation: float = 0.0

    # Per-type-pair correlation matrix rho_{k,l}
    type_correlation_matrix: Dict[Tuple[str, str], float] = field(
        default_factory=dict
    )

    # Shannon entropy of the population mix
    entropy: float = 0.0

    # Adversarial fraction in the mix
    adversarial_fraction: float = 0.0

    # Risk surrogate: p_bar * (1 - p_bar) * (1 + (N-1) * rho_bar)
    risk_surrogate: float = 0.0

    # Rule compliance flags
    correlation_cap_satisfied: bool = True
    entropy_floor_satisfied: bool = True
    adversarial_fraction_satisfied: bool = True

    # Last disagreement rate observed
    disagreement_rate: float = 0.0
    disagreement_audit_triggered: bool = False


class DiversityDefenseLever(GovernanceLever):
    """Diversity as Defense governance lever.

    Tracks agent error patterns by type and enforces diversity constraints
    to reduce correlation-amplified tail risk in swarm decisions.

    The lever monitors four governance rules:

    1. **Correlation cap** -- Penalises when the induced mean pairwise error
       correlation ``rho_bar(x)`` exceeds ``rho_max``.
    2. **Entropy floor** -- Penalises when the Shannon entropy ``H(x)`` of
       the agent-type population mix drops below ``H_min``.
    3. **Adversarial fraction** -- Warns when the fraction of adversarial
       probe agents drops below ``alpha_min``.
    4. **Disagreement-triggered audit** -- Applies an audit cost when the
       per-interaction disagreement rate exceeds ``tau``.
    """

    def __init__(self, config: GovernanceConfig):
        super().__init__(config)

        # Per-agent binary error history (agent_id -> list of 0/1)
        self._error_history: Dict[str, List[int]] = defaultdict(list)

        # Agent-to-type mapping (populated from EnvState)
        self._agent_types: Dict[str, str] = {}

        # Most recent metrics snapshot
        self._latest_metrics: Optional[DiversityMetrics] = None

    @property
    def name(self) -> str:
        return "diversity_defense"

    # ------------------------------------------------------------------
    # Core math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_population_mix(
        agent_types: Dict[str, str],
    ) -> Dict[str, float]:
        """Compute the population mix x_k from agent type assignments.

        Args:
            agent_types: Mapping from agent_id to type label.

        Returns:
            Dictionary mapping type label to fraction of population.
        """
        if not agent_types:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for type_label in agent_types.values():
            counts[type_label] += 1
        total = len(agent_types)
        return {k: v / total for k, v in counts.items()}

    @staticmethod
    def compute_shannon_entropy(mix: Dict[str, float]) -> float:
        """Compute Shannon entropy H(x) = -sum x_k log x_k.

        Args:
            mix: Population mix (values must sum to 1).

        Returns:
            Shannon entropy in nats.
        """
        entropy = 0.0
        for x_k in mix.values():
            if x_k > 0:
                entropy -= x_k * math.log(x_k)
        return entropy

    @staticmethod
    def compute_pairwise_correlation(
        errors_i: List[int],
        errors_j: List[int],
    ) -> float:
        """Compute Pearson correlation between two binary error sequences.

        Args:
            errors_i: Binary error indicators for agent i.
            errors_j: Binary error indicators for agent j.

        Returns:
            Pearson correlation coefficient, or 0.0 if undefined.
        """
        n = min(len(errors_i), len(errors_j))
        if n < 2:
            return 0.0

        ei = errors_i[-n:]
        ej = errors_j[-n:]

        mean_i = sum(ei) / n
        mean_j = sum(ej) / n

        var_i = sum((e - mean_i) ** 2 for e in ei) / n
        var_j = sum((e - mean_j) ** 2 for e in ej) / n

        if var_i < 1e-12 or var_j < 1e-12:
            return 0.0

        cov = sum((ei[k] - mean_i) * (ej[k] - mean_j) for k in range(n)) / n
        return cov / math.sqrt(var_i * var_j)

    def compute_type_correlations(
        self,
        window: int,
    ) -> Tuple[Dict[Tuple[str, str], float], float]:
        """Compute per-type-pair and mean pairwise error correlations.

        Groups agents by type and computes the average within-group and
        cross-group pairwise correlations, then derives the mix-weighted
        mean correlation rho_bar(x).

        Args:
            window: Number of most recent errors to use per agent.

        Returns:
            Tuple of (type-pair correlation dict, mean correlation).
        """
        # Group agents by type
        type_agents: Dict[str, List[str]] = defaultdict(list)
        for agent_id, type_label in self._agent_types.items():
            if agent_id in self._error_history and self._error_history[agent_id]:
                type_agents[type_label].append(agent_id)

        types = sorted(type_agents.keys())
        if not types:
            return {}, 0.0

        # Trim histories to window
        trimmed: Dict[str, List[int]] = {}
        for agent_id, history in self._error_history.items():
            trimmed[agent_id] = history[-window:]

        # Compute type-level correlations rho_{k,l}
        type_corr: Dict[Tuple[str, str], float] = {}
        for i, tk in enumerate(types):
            for j, tl in enumerate(types):
                if j < i:
                    type_corr[(tk, tl)] = type_corr[(tl, tk)]
                    continue
                agents_k = type_agents[tk]
                agents_l = type_agents[tl]
                pair_corrs: List[float] = []
                for ak in agents_k:
                    for al in agents_l:
                        if ak == al:
                            continue
                        rho = self.compute_pairwise_correlation(
                            trimmed[ak], trimmed[al]
                        )
                        pair_corrs.append(rho)
                type_corr[(tk, tl)] = (
                    sum(pair_corrs) / len(pair_corrs)
                    if pair_corrs
                    else 0.0
                )

        # Mix-weighted mean: rho_bar = sum_k,l x_k x_l rho_{k,l}
        mix = self.compute_population_mix(self._agent_types)
        rho_bar = 0.0
        for (tk, tl), rho_kl in type_corr.items():
            rho_bar += mix.get(tk, 0.0) * mix.get(tl, 0.0) * rho_kl

        return type_corr, rho_bar

    @staticmethod
    def compute_risk_surrogate(
        mean_error: float,
        mean_correlation: float,
        n_agents: int,
    ) -> float:
        """Compute the risk surrogate R(x).

        R(x) = p_bar * (1 - p_bar) * (1 + (N - 1) * rho_bar)

        This is proportional to the variance of the sum of errors, which
        drives the tail probability of collective failure under majority
        vote.

        Args:
            mean_error: Average per-agent error rate (p_bar).
            mean_correlation: Mean pairwise error correlation (rho_bar).
            n_agents: Number of agents (N).

        Returns:
            Risk surrogate value.
        """
        if n_agents < 1:
            return 0.0
        return (
            mean_error
            * (1 - mean_error)
            * (1 + (n_agents - 1) * mean_correlation)
        )

    @staticmethod
    def compute_disagreement_rate(
        decisions: List[int],
    ) -> float:
        """Compute disagreement rate D(t) for a set of agent decisions.

        D(t) = 1 - max_y (1/N) sum_i 1[y_i = y]

        Args:
            decisions: List of binary decisions (0 or 1) from N agents.

        Returns:
            Disagreement rate in [0, 1].
        """
        if not decisions:
            return 0.0
        n = len(decisions)
        count_1 = sum(decisions)
        count_0 = n - count_1
        majority_frac = max(count_1, count_0) / n
        return 1.0 - majority_frac

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------

    def _refresh_agent_types(self, state: EnvState) -> None:
        """Update agent type mapping from current environment state."""
        self._agent_types = {
            agent_id: agent_state.agent_type.value
            for agent_id, agent_state in state.agents.items()
        }

    def _prune_stale_agents(self, state: EnvState) -> None:
        """Remove error history for agents no longer in the environment."""
        stale = [
            aid for aid in self._error_history if aid not in state.agents
        ]
        for aid in stale:
            del self._error_history[aid]

    def _compute_full_metrics(self, state: EnvState) -> DiversityMetrics:
        """Compute a full metrics snapshot from current state."""
        self._refresh_agent_types(state)
        self._prune_stale_agents(state)

        mix = self.compute_population_mix(self._agent_types)
        entropy = self.compute_shannon_entropy(mix)

        adversarial_types = {AgentType.ADVERSARIAL.value}
        adv_fraction = sum(
            mix.get(t, 0.0) for t in adversarial_types
        )

        window = self.config.diversity_correlation_window
        type_corr, rho_bar = self.compute_type_correlations(window)

        # Mean error rate across all agents with history
        all_errors: List[int] = []
        for history in self._error_history.values():
            recent = history[-window:]
            all_errors.extend(recent)
        mean_error = sum(all_errors) / len(all_errors) if all_errors else 0.0

        n_agents = len(state.agents)
        risk = self.compute_risk_surrogate(mean_error, rho_bar, n_agents)

        metrics = DiversityMetrics(
            population_mix=mix,
            mean_error_rate=mean_error,
            mean_correlation=rho_bar,
            type_correlation_matrix=type_corr,
            entropy=entropy,
            adversarial_fraction=adv_fraction,
            risk_surrogate=risk,
            correlation_cap_satisfied=(
                rho_bar <= self.config.diversity_rho_max
            ),
            entropy_floor_satisfied=(
                entropy >= self.config.diversity_entropy_min
            ),
            adversarial_fraction_satisfied=(
                adv_fraction >= self.config.diversity_adversarial_fraction_min
            ),
        )

        self._latest_metrics = metrics
        return metrics

    def on_epoch_start(
        self,
        state: EnvState,
        epoch: int,
    ) -> LeverEffect:
        """Evaluate diversity rules at epoch boundary.

        Computes population mix, correlation, entropy, and adversarial
        fraction.  Applies cost penalties when correlation cap or entropy
        floor constraints are violated.

        Args:
            state: Current environment state.
            epoch: The epoch number starting.

        Returns:
            LeverEffect with costs for constraint violations.
        """
        if not self.config.diversity_enabled:
            return LeverEffect(lever_name=self.name)

        metrics = self._compute_full_metrics(state)

        cost_a = 0.0
        cost_b = 0.0
        violations: List[str] = []

        # Rule 1: Correlation cap
        if not metrics.correlation_cap_satisfied:
            excess = metrics.mean_correlation - self.config.diversity_rho_max
            penalty = excess * self.config.diversity_correlation_penalty_rate
            cost_a += penalty
            cost_b += penalty
            violations.append("correlation_cap")

        # Rule 2: Entropy floor
        if not metrics.entropy_floor_satisfied:
            deficit = self.config.diversity_entropy_min - metrics.entropy
            penalty = deficit * self.config.diversity_entropy_penalty_rate
            cost_a += penalty
            cost_b += penalty
            violations.append("entropy_floor")

        # Rule 3: Adversarial fraction (warning only, no cost)
        if not metrics.adversarial_fraction_satisfied:
            violations.append("adversarial_fraction_low")

        return LeverEffect(
            cost_a=cost_a,
            cost_b=cost_b,
            lever_name=self.name,
            details={
                "epoch": epoch,
                "population_mix": metrics.population_mix,
                "entropy": metrics.entropy,
                "mean_correlation": metrics.mean_correlation,
                "risk_surrogate": metrics.risk_surrogate,
                "adversarial_fraction": metrics.adversarial_fraction,
                "violations": violations,
                "correlation_cap_satisfied": metrics.correlation_cap_satisfied,
                "entropy_floor_satisfied": metrics.entropy_floor_satisfied,
                "adversarial_fraction_satisfied": (
                    metrics.adversarial_fraction_satisfied
                ),
            },
        )

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """Record errors and check disagreement rule.

        Each interaction records a binary error indicator for the
        initiator.  When ground truth is available, error = 1[p < threshold
        and ground_truth == -1] or 1[p >= threshold and ground_truth == +1].
        Without ground truth, error is simply 1[p < threshold].

        After recording, the lever checks the disagreement rate across
        recent agent decisions.  If the disagreement rate exceeds ``tau``,
        an audit cost is applied (Rule 4).

        Args:
            interaction: The completed interaction.
            state: Current environment state.

        Returns:
            LeverEffect with audit cost if disagreement threshold breached.
        """
        if not self.config.diversity_enabled:
            return LeverEffect(lever_name=self.name)

        # Only track agents registered in the environment to prevent
        # unbounded memory growth from arbitrary initiator strings.
        agent_id = interaction.initiator
        if agent_id not in state.agents:
            return LeverEffect(lever_name=self.name)

        # Record error for the initiator
        threshold = self.config.diversity_error_threshold_p
        is_error: int
        if interaction.ground_truth is not None:
            # With ground truth: error iff the soft label disagrees
            predicted_positive = interaction.p >= threshold
            actually_positive = interaction.ground_truth == 1
            is_error = 0 if predicted_positive == actually_positive else 1
        else:
            # Without ground truth: low p counts as error
            is_error = 1 if interaction.p < threshold else 0

        self._error_history[agent_id].append(is_error)

        # Trim to window
        window = self.config.diversity_correlation_window
        if len(self._error_history[agent_id]) > window:
            self._error_history[agent_id] = (
                self._error_history[agent_id][-window:]
            )

        # Rule 4: Disagreement-triggered audit
        # Collect most recent decisions from all agents with history
        recent_decisions: List[int] = []
        for _aid, history in self._error_history.items():
            if history:
                recent_decisions.append(history[-1])

        disagreement = self.compute_disagreement_rate(recent_decisions)

        cost_a = 0.0
        audit_triggered = False
        if disagreement >= self.config.diversity_disagreement_tau:
            cost_a = self.config.diversity_audit_cost
            audit_triggered = True

        if self._latest_metrics is not None:
            self._latest_metrics.disagreement_rate = disagreement
            self._latest_metrics.disagreement_audit_triggered = audit_triggered

        return LeverEffect(
            cost_a=cost_a,
            lever_name=self.name,
            details={
                "agent_id": agent_id,
                "is_error": is_error,
                "p": interaction.p,
                "disagreement_rate": disagreement,
                "audit_triggered": audit_triggered,
            },
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_metrics(self) -> Optional[DiversityMetrics]:
        """Return the most recent diversity metrics snapshot."""
        return self._latest_metrics

    def get_error_history(self) -> Dict[str, List[int]]:
        """Return per-agent error histories (deep copy)."""
        return {k: list(v) for k, v in self._error_history.items()}

    def get_agent_types(self) -> Dict[str, str]:
        """Return the current agent-to-type mapping."""
        return dict(self._agent_types)

    def clear_history(self) -> None:
        """Reset all tracked error histories."""
        self._error_history.clear()
        self._latest_metrics = None
