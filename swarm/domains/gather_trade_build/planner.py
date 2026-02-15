"""Planner agent for the bilevel Planner-Workers loop.

The planner observes aggregate epoch statistics and updates the tax schedule
on a configurable cadence. Supports heuristic, bandit, and (stub) RL planners.
"""

from __future__ import annotations

import logging
import random
from typing import Dict, List, Optional

from swarm.domains.gather_trade_build.config import PlannerConfig, TaxBracket
from swarm.domains.gather_trade_build.tax_schedule import TaxSchedule

logger = logging.getLogger(__name__)


class PlannerAgent:
    """Bilevel planner that updates tax policy each epoch.

    Planner types:
      - heuristic: rule-based adjustments targeting a welfare objective
      - bandit: epsilon-greedy over rate perturbations
      - rl: placeholder for future RL training
    """

    def __init__(
        self,
        config: PlannerConfig,
        tax_schedule: TaxSchedule,
        seed: Optional[int] = None,
    ) -> None:
        self._config = config
        self._tax_schedule = tax_schedule
        self._rng = random.Random(seed)
        self._epoch_count = 0

        # Bandit state
        self._prev_welfare: Optional[float] = None
        self._prev_action: Optional[List[TaxBracket]] = None

    def should_update(self, epoch: int) -> bool:
        """Whether the planner should update this epoch."""
        return epoch > 0 and epoch % self._config.update_interval_epochs == 0

    def update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Observe aggregate stats and return new tax brackets.

        Args:
            stats: Aggregate stats from GTBEnvironment.get_aggregate_stats().
                   Expected keys: total_income, mean_income, gini,
                   total_tax_revenue, total_houses, n_workers.

        Returns:
            New list of TaxBracket to apply.
        """
        self._epoch_count += 1

        if self._config.planner_type == "heuristic":
            return self._heuristic_update(stats)
        elif self._config.planner_type == "bandit":
            return self._bandit_update(stats)
        else:
            # RL stub: no-op, keep current schedule
            logger.info("RL planner stub: keeping current schedule")
            return self._tax_schedule.brackets

    def _compute_welfare(self, stats: Dict[str, float]) -> float:
        """Compute welfare objective from stats."""
        prod = stats.get("mean_income", 0.0)
        gini = stats.get("gini", 0.0)
        return self._config.prod_weight * prod - self._config.ineq_weight * gini

    def _heuristic_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Rule-based planner: increase rates if inequality is high, decrease if productivity is low."""
        gini = stats.get("gini", 0.0)
        mean_income = stats.get("mean_income", 0.0)
        lr = self._config.learning_rate

        current = self._tax_schedule.brackets
        new_brackets = []

        for bracket in current:
            # If inequality is high, raise rates (especially upper brackets)
            # If productivity is low, lower rates
            gini_signal = (gini - 0.3) * 2.0  # positive when gini > 0.3
            prod_signal = -(max(0, 5.0 - mean_income) / 5.0)  # negative when low prod

            adjustment = lr * (gini_signal + prod_signal)
            new_rate = max(0.0, min(1.0, bracket.rate + adjustment))
            new_brackets.append(TaxBracket(
                threshold=bracket.threshold, rate=new_rate,
            ))

        self._tax_schedule.update_brackets(new_brackets)
        return self._tax_schedule.brackets

    def _bandit_update(self, stats: Dict[str, float]) -> List[TaxBracket]:
        """Epsilon-greedy bandit planner over rate perturbations."""
        welfare = self._compute_welfare(stats)
        current = self._tax_schedule.brackets

        # If previous action improved welfare, keep direction; otherwise reverse
        if self._prev_welfare is not None and self._prev_action is not None:
            if welfare < self._prev_welfare:
                # Revert to previous
                current = self._prev_action

        self._prev_welfare = welfare
        self._prev_action = list(current)

        # Epsilon-greedy: with probability epsilon, try random perturbation
        if self._rng.random() < self._config.exploration_rate:
            new_brackets = []
            for bracket in current:
                delta = self._rng.gauss(0, self._config.learning_rate)
                new_rate = max(0.0, min(1.0, bracket.rate + delta))
                new_brackets.append(TaxBracket(
                    threshold=bracket.threshold, rate=new_rate,
                ))
        else:
            new_brackets = list(current)

        self._tax_schedule.update_brackets(new_brackets)
        return self._tax_schedule.brackets

    @property
    def tax_schedule(self) -> TaxSchedule:
        return self._tax_schedule
