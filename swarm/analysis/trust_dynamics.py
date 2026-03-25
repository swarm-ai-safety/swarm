"""Temporal trust dynamics: convergence and decay analysis.

This module provides tools for analyzing how agent trust evolves over time,
including convergence rates, decay patterns, and trust shocks.
"""

from typing import Dict, List, Tuple

from swarm.knowledge.graph_memory import AgentMemorySnapshot, GraphMemoryStore


class TrustDynamicsAnalyzer:
    """Analyzes temporal dynamics of trust across agent snapshots.

    Takes a GraphMemoryStore and provides methods for extracting trust timelines,
    computing convergence rates, detecting decay patterns, and identifying shocks.
    """

    def __init__(self, memory_store: GraphMemoryStore):
        """Initialize analyzer with a graph memory store.

        Args:
            memory_store: GraphMemoryStore containing agent memory snapshots.
        """
        self.store = memory_store

    def get_trust_timeline(
        self, agent_id: str, counterparty_id: str
    ) -> List[Tuple[int, float]]:
        """Get trust score evolution for a directed relationship.

        Args:
            agent_id: Source agent ID
            counterparty_id: Target agent ID

        Returns:
            List of (epoch, trust_score) tuples, sorted by epoch.
            Empty if no snapshots found for agent.
        """
        snapshots = self.store._snapshots.get(agent_id, [])
        if not snapshots:
            return []

        timeline = []
        for snapshot_dict in snapshots:
            snapshot = AgentMemorySnapshot.from_dict(snapshot_dict)
            if counterparty_id in snapshot.counterparty_trust:
                trust = snapshot.counterparty_trust[counterparty_id]
                timeline.append((snapshot.epoch, trust))

        # Sort by epoch
        timeline.sort(key=lambda x: x[0])
        return timeline

    def compute_convergence_rate(self, agent_id: str) -> float:
        """Measure how quickly an agent's trust scores stabilize.

        Computes as 1 - (variance of deltas in second half / variance in first half).
        - Value near 1.0: fast convergence (trust stabilizes)
        - Value near 0.0: no convergence (still volatile)
        - Value < 0.0: increasing volatility (divergence)

        Args:
            agent_id: Agent ID

        Returns:
            Convergence rate in [-inf, 1.0]. Returns 0.0 if insufficient data.
        """
        snapshots = self.store._snapshots.get(agent_id, [])
        if len(snapshots) < 4:  # Need at least 4 points to split halves
            return 0.0

        # Collect all trust changes over time
        all_deltas: List[float] = []
        for i in range(1, len(snapshots)):
            prev_snapshot = AgentMemorySnapshot.from_dict(snapshots[i - 1])
            curr_snapshot = AgentMemorySnapshot.from_dict(snapshots[i])

            # Compute change in trust for each counterparty
            for counterparty_id in curr_snapshot.counterparty_trust:
                if counterparty_id in prev_snapshot.counterparty_trust:
                    prev_trust = prev_snapshot.counterparty_trust[counterparty_id]
                    curr_trust = curr_snapshot.counterparty_trust[counterparty_id]
                    delta = abs(curr_trust - prev_trust)
                    all_deltas.append(delta)

        if len(all_deltas) < 2:
            return 0.0

        # Split into halves
        split = len(all_deltas) // 2
        first_half = all_deltas[:split]
        second_half = all_deltas[split:]

        # Compute variances
        var_first = self._compute_variance(first_half)
        var_second = self._compute_variance(second_half)

        if var_first == 0.0:
            # No change in first half - already converged
            return 1.0

        # Convergence = 1 - (second_var / first_var)
        convergence = 1.0 - (var_second / var_first)
        return convergence

    def compute_trust_decay_rate(self, agent_id: str) -> float:
        """Measure average per-epoch trust change across all counterparties.

        Returns slope of linear regression on trust over epochs.
        - Negative: trust is decaying on average
        - Positive: trust is growing on average
        - Near 0: stable trust

        Args:
            agent_id: Agent ID

        Returns:
            Slope (trust change per epoch). Returns 0.0 if insufficient data.
        """
        snapshots = self.store._snapshots.get(agent_id, [])
        if len(snapshots) < 2:
            return 0.0

        # Collect (epoch, mean_trust) pairs
        epoch_mean_trust: List[Tuple[int, float]] = []
        for snapshot_dict in snapshots:
            snapshot = AgentMemorySnapshot.from_dict(snapshot_dict)
            if snapshot.counterparty_trust:
                mean_trust = sum(snapshot.counterparty_trust.values()) / len(
                    snapshot.counterparty_trust
                )
                epoch_mean_trust.append((snapshot.epoch, mean_trust))

        if len(epoch_mean_trust) < 2:
            return 0.0

        # Simple linear regression: slope = cov(x,y) / var(x)
        epochs: List[float] = [float(x[0]) for x in epoch_mean_trust]
        trusts = [x[1] for x in epoch_mean_trust]

        slope = self._linear_regression_slope(epochs, trusts)
        return slope

    def get_system_dynamics_summary(self) -> Dict[str, float]:
        """Summarize trust dynamics across all agents.

        Returns:
            Dict with keys:
                - mean_convergence_rate
                - median_convergence_rate
                - mean_decay_rate
                - agents_converged_count (convergence > 0.5)
                - agents_volatile_count (convergence < 0.1)
                - trust_stability_index (1 - normalized variance)
        """
        agent_ids = list(self.store._snapshots.keys())
        if not agent_ids:
            return {
                "mean_convergence_rate": 0.0,
                "median_convergence_rate": 0.0,
                "mean_decay_rate": 0.0,
                "agents_converged_count": 0,
                "agents_volatile_count": 0,
                "trust_stability_index": 0.0,
            }

        # Compute convergence and decay for each agent
        convergence_rates = []
        decay_rates = []
        all_trust_changes = []

        for agent_id in agent_ids:
            conv = self.compute_convergence_rate(agent_id)
            decay = self.compute_trust_decay_rate(agent_id)
            convergence_rates.append(conv)
            decay_rates.append(decay)

            # Collect trust changes for stability index
            snapshots = self.store._snapshots.get(agent_id, [])
            for i in range(1, len(snapshots)):
                prev_snap = AgentMemorySnapshot.from_dict(snapshots[i - 1])
                curr_snap = AgentMemorySnapshot.from_dict(snapshots[i])
                for cp_id in curr_snap.counterparty_trust:
                    if cp_id in prev_snap.counterparty_trust:
                        delta = abs(
                            curr_snap.counterparty_trust[cp_id]
                            - prev_snap.counterparty_trust[cp_id]
                        )
                        all_trust_changes.append(delta)

        # Compute summary metrics
        mean_convergence = (
            sum(convergence_rates) / len(convergence_rates)
            if convergence_rates
            else 0.0
        )
        median_convergence = self._median(convergence_rates)
        mean_decay = sum(decay_rates) / len(decay_rates) if decay_rates else 0.0

        converged_count = sum(1 for c in convergence_rates if c > 0.5)
        volatile_count = sum(1 for c in convergence_rates if c < 0.1)

        # Trust stability = 1 - normalized variance of changes
        if all_trust_changes:
            var_changes = self._compute_variance(all_trust_changes)
            mean_changes = sum(all_trust_changes) / len(all_trust_changes)
            # Normalize by mean to get coefficient of variation
            if mean_changes > 0:
                cv = var_changes**0.5 / mean_changes  # sqrt(variance) / mean
                stability = max(0.0, 1.0 - cv)
            else:
                stability = 1.0 if var_changes == 0.0 else 0.0
        else:
            stability = 0.0

        return {
            "mean_convergence_rate": mean_convergence,
            "median_convergence_rate": median_convergence,
            "mean_decay_rate": mean_decay,
            "agents_converged_count": converged_count,
            "agents_volatile_count": volatile_count,
            "trust_stability_index": stability,
        }

    def detect_trust_shocks(
        self, agent_id: str, threshold: float = 0.2
    ) -> List[Dict[str, float | int | str]]:
        """Detect epochs where trust changed significantly.

        Args:
            agent_id: Agent ID
            threshold: Minimum trust change to flag (default 0.2)

        Returns:
            List of dicts with keys: epoch, counterparty, delta, trust_before, trust_after
            Sorted by epoch.
        """
        snapshots = self.store._snapshots.get(agent_id, [])
        if len(snapshots) < 2:
            return []

        shocks = []
        for i in range(1, len(snapshots)):
            prev_snap = AgentMemorySnapshot.from_dict(snapshots[i - 1])
            curr_snap = AgentMemorySnapshot.from_dict(snapshots[i])

            for counterparty_id in curr_snap.counterparty_trust:
                if counterparty_id in prev_snap.counterparty_trust:
                    trust_before = prev_snap.counterparty_trust[counterparty_id]
                    trust_after = curr_snap.counterparty_trust[counterparty_id]
                    delta = abs(trust_after - trust_before)

                    if delta >= threshold:
                        shock: Dict[str, float | int | str] = {
                            "epoch": curr_snap.epoch,
                            "counterparty": counterparty_id,
                            "delta": delta,
                            "trust_before": trust_before,
                            "trust_after": trust_after,
                        }
                        shocks.append(shock)

        # Sort by epoch
        shocks.sort(key=lambda x: x["epoch"] if isinstance(x["epoch"], int) else 0)
        return shocks

    # ===== Private helper methods =====

    @staticmethod
    def _compute_variance(values: List[float]) -> float:
        """Compute sample variance.

        Args:
            values: List of numeric values

        Returns:
            Sample variance. Returns 0.0 if fewer than 2 values.
        """
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        sum_sq_diff = sum((v - mean) ** 2 for v in values)
        variance = sum_sq_diff / (len(values) - 1)
        return variance

    @staticmethod
    def _median(values: List[float]) -> float:
        """Compute median of a list.

        Args:
            values: List of numeric values

        Returns:
            Median value. Returns 0.0 if empty.
        """
        if not values:
            return 0.0

        sorted_vals = sorted(values)
        n = len(sorted_vals)

        if n % 2 == 1:
            return sorted_vals[n // 2]
        else:
            return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    @staticmethod
    def _linear_regression_slope(x_values: List[float], y_values: List[float]) -> float:
        """Compute slope of simple linear regression y = ax + b.

        Uses the formula: slope = cov(x,y) / var(x)

        Args:
            x_values: Independent variable values
            y_values: Dependent variable values

        Returns:
            Slope. Returns 0.0 if insufficient data or var(x) = 0.
        """
        if len(x_values) < 2 or len(x_values) != len(y_values):
            return 0.0

        n = len(x_values)
        mean_x = sum(x_values) / n
        mean_y = sum(y_values) / n

        # Compute covariance and variance of x
        cov_xy = sum(
            (x_values[i] - mean_x) * (y_values[i] - mean_y) for i in range(n)
        ) / (n - 1)
        var_x = (
            sum((x - mean_x) ** 2 for x in x_values) / (n - 1)
            if n > 1
            else 0.0
        )

        if var_x == 0.0:
            return 0.0

        slope = cov_xy / var_x
        return slope
