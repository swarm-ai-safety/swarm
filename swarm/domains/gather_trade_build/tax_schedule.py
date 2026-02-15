"""Tax schedule engine with piecewise brackets and U-shaped support."""

from __future__ import annotations

import math
from typing import List

from swarm.domains.gather_trade_build.config import TaxBracket, TaxScheduleConfig


class TaxSchedule:
    """Piecewise tax schedule with optional smoothing and non-monotone support.

    Supports:
      - Flat: single rate for all income
      - Piecewise progressive: increasing marginal rates
      - Non-monotone / U-shaped: rates can decrease then increase
      - Sigmoid smoothing at bracket edges to reduce bunching incentives
    """

    def __init__(self, config: TaxScheduleConfig) -> None:
        self._config = config
        self._brackets = sorted(config.brackets, key=lambda b: b.threshold)
        self._smoothing = config.smoothing
        self._validate()

    def _validate(self) -> None:
        if not self._brackets:
            raise ValueError("Tax schedule must have at least one bracket")
        if self._brackets[0].threshold != 0.0:
            raise ValueError("First bracket must start at threshold 0.0")
        for b in self._brackets:
            if b.rate < 0.0 or b.rate > 1.0:
                raise ValueError(f"Tax rate must be in [0, 1], got {b.rate}")
        if not self._config.allow_non_monotone:
            for i in range(1, len(self._brackets)):
                if self._brackets[i].rate < self._brackets[i - 1].rate - 1e-9:
                    raise ValueError(
                        "Non-monotone rates require allow_non_monotone=True"
                    )

    def compute_tax(self, income: float) -> float:
        """Compute total tax owed on the given income.

        Args:
            income: Gross (reported) income for the epoch.

        Returns:
            Total tax amount (non-negative).
        """
        if income <= 0:
            return 0.0
        if self._smoothing > 0:
            return self._compute_tax_smooth(income)
        return self._compute_tax_hard(income)

    def _compute_tax_hard(self, income: float) -> float:
        total_tax = 0.0
        for i, bracket in enumerate(self._brackets):
            if i + 1 < len(self._brackets):
                upper = self._brackets[i + 1].threshold
            else:
                upper = float("inf")
            if income <= bracket.threshold:
                break
            taxable = min(income, upper) - bracket.threshold
            if taxable > 0:
                total_tax += taxable * bracket.rate
        return total_tax

    def _compute_tax_smooth(self, income: float) -> float:
        sigma = self._smoothing
        n_steps = min(max(100, int(income * 10)), 10000)  # cap to prevent CPU bomb
        dx = income / n_steps
        total_tax = 0.0
        for step_i in range(n_steps):
            x = (step_i + 0.5) * dx
            marginal = self._marginal_rate_smooth(x, sigma)
            total_tax += marginal * dx
        return total_tax

    def _marginal_rate_smooth(self, x: float, sigma: float) -> float:
        if len(self._brackets) == 1:
            return self._brackets[0].rate
        rate = self._brackets[0].rate
        for i in range(1, len(self._brackets)):
            threshold = self._brackets[i].threshold
            rate_diff = self._brackets[i].rate - self._brackets[i - 1].rate
            z = (x - threshold) / max(sigma, 1e-6)
            z = max(-500.0, min(500.0, z))
            weight = 1.0 / (1.0 + math.exp(-z))
            rate += rate_diff * weight
        return max(0.0, min(1.0, rate))

    def marginal_rate(self, income: float) -> float:
        """Get the marginal tax rate at the given income level."""
        if income <= 0:
            return self._brackets[0].rate if self._brackets else 0.0
        for i in range(len(self._brackets) - 1, -1, -1):
            if income >= self._brackets[i].threshold:
                return self._brackets[i].rate
        return self._brackets[0].rate

    def effective_rate(self, income: float) -> float:
        """Get the effective (average) tax rate."""
        if income <= 0:
            return 0.0
        return self.compute_tax(income) / income

    @property
    def brackets(self) -> List[TaxBracket]:
        return list(self._brackets)

    @property
    def bracket_thresholds(self) -> List[float]:
        return [b.threshold for b in self._brackets]

    def update_brackets(self, new_brackets: List[TaxBracket]) -> None:
        """Update brackets (used by the planner at epoch boundaries).

        Applies damping if configured.
        """
        damping = self._config.damping
        if damping > 0 and len(new_brackets) == len(self._brackets):
            damped = []
            for old, new in zip(self._brackets, new_brackets, strict=True):
                damped_rate = old.rate + (1.0 - damping) * (new.rate - old.rate)
                damped_rate = max(0.0, min(1.0, damped_rate))
                damped.append(TaxBracket(threshold=new.threshold, rate=damped_rate))
            self._brackets = sorted(damped, key=lambda b: b.threshold)
        else:
            self._brackets = sorted(new_brackets, key=lambda b: b.threshold)
        self._validate()

    def to_dict(self) -> dict:
        return {
            "family": self._config.schedule_family,
            "brackets": [
                {"threshold": b.threshold, "rate": b.rate} for b in self._brackets
            ],
            "smoothing": self._smoothing,
            "damping": self._config.damping,
        }
