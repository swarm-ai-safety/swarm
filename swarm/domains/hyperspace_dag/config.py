"""Configuration for the Hyperspace DAG Planner domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DagProxyConfig:
    """Proxy weight overrides for DAG structural signals.

    DAG coherence (edge density, depth) is carried via the engagement
    channel, so we upweight it relative to the delivery baseline.
    """

    # Execution outcome is still the primary signal
    task_progress_weight: float = 0.35
    rework_weight: float = 0.20
    verifier_weight: float = 0.15
    # Structural coherence via engagement — upweighted from 0.2
    engagement_weight: float = 0.30

    def __post_init__(self) -> None:
        weights = [
            self.task_progress_weight,
            self.rework_weight,
            self.verifier_weight,
            self.engagement_weight,
        ]
        if any(w < 0 for w in weights):
            raise ValueError("All proxy weights must be non-negative")


@dataclass
class DagConfig:
    """Top-level configuration for the Hyperspace DAG domain."""

    proxy: DagProxyConfig = None  # type: ignore[assignment]
    acceptance_threshold: float = 0.5
    # Plans below this confidence are flagged for extra scrutiny
    confidence_floor: float = 0.3
    # Maximum retries before marking a plan as failed
    max_retries: int = 3
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.proxy is None:
            self.proxy = DagProxyConfig()
        if not 0.0 <= self.acceptance_threshold <= 1.0:
            raise ValueError("acceptance_threshold must be in [0, 1]")
        if not 0.0 <= self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DagConfig":
        """Parse a DagConfig from a YAML-sourced dict."""
        if not data:
            return cls()

        proxy_data = data.get("proxy", {})
        proxy_cfg = DagProxyConfig(**{
            k: proxy_data[k] for k in (
                "task_progress_weight", "rework_weight",
                "verifier_weight", "engagement_weight",
            ) if k in proxy_data
        })

        return cls(
            proxy=proxy_cfg,
            acceptance_threshold=data.get("acceptance_threshold", 0.5),
            confidence_floor=data.get("confidence_floor", 0.3),
            max_retries=data.get("max_retries", 3),
            seed=data.get("seed"),
        )
