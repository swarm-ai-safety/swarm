"""Adverse selection detection via relationship graph analysis.

This module detects adverse selection patterns in agent interaction networks
by analyzing quality gaps and selection pressures in the GraphMemoryStore.

Adverse selection occurs when agents preferentially match with lower-quality
counterparts, leading to systematic quality degradation or exploitation
patterns.

Key concepts:
- Quality gap: E[p | accepted] - baseline (expected vs actual match quality)
- Selection pressure: ratio of actual avg_p to system avg_p (< 1.0 = adverse)
- Exploited agents: those with negative quality gaps (matching down)
- Exploiting agents: those with low avg_p but high payoffs (gaining from bad matches)
"""

from typing import Dict, List

import numpy as np

from swarm.knowledge.graph_memory import GraphMemoryStore


class AdverseSelectionDetector:
    """Detects adverse selection patterns in agent relationship networks.

    Analyzes relationships from a GraphMemoryStore to identify:
    - Agents matching with systematically lower-quality counterparts
    - Agents successfully exploiting lower-quality matches
    - System-wide selection pressure and quality degradation

    Attributes:
        store: GraphMemoryStore instance with agent relationships
    """

    def __init__(self, store: GraphMemoryStore):
        """Initialize the detector.

        Args:
            store: GraphMemoryStore instance with agent relationships
        """
        self.store = store
        self._relationships = store.get_all_relationships()

    def compute_quality_gap_by_agent(self) -> Dict[str, float]:
        """Compute quality gap for each agent as initiator.

        Quality gap is the difference between the agent's average match quality
        (avg_p of their accepted relationships) and the system-wide baseline.

        A negative quality gap indicates the agent is matching with
        systematically lower-quality counterparts (adverse selection).

        Returns:
            Dict mapping agent_id -> quality_gap (float).
            Only includes agents with at least one interaction.
        """
        if not self._relationships:
            return {}

        # Compute system baseline: overall average p
        system_avg_p = np.mean([rel.avg_p for rel in self._relationships])

        # Group relationships by initiator (agent_a)
        by_initiator: Dict[str, List[float]] = {}
        for rel in self._relationships:
            if rel.interaction_count > 0:  # Only accepted relationships
                if rel.agent_a not in by_initiator:
                    by_initiator[rel.agent_a] = []
                by_initiator[rel.agent_a].append(rel.avg_p)

        # Compute quality gap for each initiator
        result = {}
        for agent_id, avg_ps in by_initiator.items():
            agent_avg_p = np.mean(avg_ps)
            quality_gap = agent_avg_p - system_avg_p
            result[agent_id] = float(quality_gap)

        return result

    def identify_exploited_agents(self, threshold: float = -0.1) -> List[str]:
        """Identify agents experiencing adverse selection (being matched down).

        Agents are considered exploited if their quality gap falls below the
        threshold, meaning they consistently match with lower-quality
        counterparts than the system average.

        Args:
            threshold: Quality gap threshold (default -0.1).
                      Agents with gap < threshold are flagged as exploited.

        Returns:
            Sorted list of exploited agent IDs.
        """
        quality_gaps = self.compute_quality_gap_by_agent()
        exploited = [
            agent_id
            for agent_id, gap in quality_gaps.items()
            if gap < threshold
        ]
        return sorted(exploited)

    def identify_exploiting_agents(self, threshold: float = 0.1) -> List[str]:
        """Identify agents successfully exploiting lower-quality matches.

        An agent is considered an exploiter if:
        1. Their average match quality (avg_p in initiated relationships) is
           significantly below the system average (> 0.2 difference), AND
        2. Their total payoff is above the median initiator payoff.

        This identifies agents who gain disproportionate returns from
        matching with low-quality counterparts.

        Args:
            threshold: Payoff threshold percentile (default 0.1 = top 10%).
                      Agents above this threshold are flagged as exploiting.

        Returns:
            Sorted list of exploiting agent IDs.
        """
        if not self._relationships:
            return []

        system_avg_p = np.mean([rel.avg_p for rel in self._relationships])

        # Group relationships by initiator
        by_initiator: Dict[str, List[float]] = {}
        by_initiator_payoff: Dict[str, float] = {}

        for rel in self._relationships:
            if rel.interaction_count > 0:
                if rel.agent_a not in by_initiator:
                    by_initiator[rel.agent_a] = []
                    by_initiator_payoff[rel.agent_a] = 0.0

                by_initiator[rel.agent_a].append(rel.avg_p)
                by_initiator_payoff[rel.agent_a] += rel.total_payoff_a

        # Find agents with significantly below-average match quality
        # (> 0.2 difference suggests systematic exploitation)
        low_quality_initiators = []
        for agent_id, avg_ps in by_initiator.items():
            agent_avg_p = np.mean(avg_ps)
            if agent_avg_p < system_avg_p - 0.2:
                low_quality_initiators.append(agent_id)

        # Compute payoff threshold (top 1-threshold of low-quality initiators)
        if not low_quality_initiators:
            return []

        payoffs = [by_initiator_payoff[a] for a in low_quality_initiators]
        payoff_threshold = np.percentile(payoffs, (1 - threshold) * 100)

        # Flag those with high payoffs
        exploiting = [
            agent_id
            for agent_id in low_quality_initiators
            if by_initiator_payoff[agent_id] >= payoff_threshold
        ]

        return sorted(exploiting)

    def compute_selection_pressure(self) -> Dict[str, float]:
        """Compute selection pressure for each agent.

        Selection pressure is the ratio of an agent's average match quality
        to the system-wide average. Values < 1.0 indicate adverse selection
        pressure (matching with below-average quality).

        Args:
            (None)

        Returns:
            Dict mapping agent_id -> selection_pressure (float).
            Only includes agents with at least one interaction.
        """
        if not self._relationships:
            return {}

        system_avg_p = np.mean([rel.avg_p for rel in self._relationships])

        # Avoid division by zero
        if system_avg_p == 0:
            return {}

        # Group relationships by initiator
        by_initiator: Dict[str, List[float]] = {}
        for rel in self._relationships:
            if rel.interaction_count > 0:
                if rel.agent_a not in by_initiator:
                    by_initiator[rel.agent_a] = []
                by_initiator[rel.agent_a].append(rel.avg_p)

        # Compute selection pressure for each initiator
        result = {}
        for agent_id, avg_ps in by_initiator.items():
            agent_avg_p = np.mean(avg_ps)
            pressure = agent_avg_p / system_avg_p
            result[agent_id] = float(pressure)

        return result

    def get_adverse_selection_summary(self) -> Dict:
        """Get comprehensive adverse selection summary statistics.

        Provides a high-level view of adverse selection patterns in the system.

        Returns:
            Dict with keys:
                - system_avg_p: Overall average interaction quality
                - agents_exploited_count: Number of exploited agents
                - agents_exploiting_count: Number of exploiting agents
                - worst_quality_gap: Most negative quality gap
                - best_quality_gap: Most positive quality gap
                - selection_pressure_variance: Variance of selection pressure
                - selection_pressure_mean: Mean selection pressure (should be ~1.0)
        """
        if not self._relationships:
            return {
                "system_avg_p": 0.0,
                "agents_exploited_count": 0,
                "agents_exploiting_count": 0,
                "worst_quality_gap": 0.0,
                "best_quality_gap": 0.0,
                "selection_pressure_variance": 0.0,
                "selection_pressure_mean": 1.0,
            }

        system_avg_p = float(np.mean([rel.avg_p for rel in self._relationships]))

        exploited = self.identify_exploited_agents()
        exploiting = self.identify_exploiting_agents()

        quality_gaps = self.compute_quality_gap_by_agent()
        selection_pressures = self.compute_selection_pressure()

        if not quality_gaps:
            worst_gap = 0.0
            best_gap = 0.0
        else:
            worst_gap = float(min(quality_gaps.values()))
            best_gap = float(max(quality_gaps.values()))

        if not selection_pressures:
            pressure_variance = 0.0
            pressure_mean = 1.0
        else:
            pressures = list(selection_pressures.values())
            pressure_variance = float(np.var(pressures))
            pressure_mean = float(np.mean(pressures))

        return {
            "system_avg_p": system_avg_p,
            "agents_exploited_count": len(exploited),
            "agents_exploiting_count": len(exploiting),
            "worst_quality_gap": worst_gap,
            "best_quality_gap": best_gap,
            "selection_pressure_variance": pressure_variance,
            "selection_pressure_mean": pressure_mean,
        }
