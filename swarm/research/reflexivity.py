"""Reflexivity handling for recursive agent research.

Implements solutions from docs/research/reflexivity.md:
- Shadow simulations for drift measurement
- Publish-then-attack protocol
- Goodhart-resistant metrics
- Temporal checkpointing
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import numpy as np


class RobustnessClassification(Enum):
    """Classification of finding robustness to disclosure."""

    DISCLOSURE_ROBUST = "disclosure_robust"
    CONDITIONALLY_VALID = "conditionally_valid"
    FRAGILE = "fragile"
    UNKNOWN = "unknown"


@dataclass
class Finding:
    """A research finding that may be subject to reflexivity."""

    statement: str
    metric_name: str
    metric_value: float
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    conditions: list[str] = field(default_factory=list)
    robustness: RobustnessClassification = RobustnessClassification.UNKNOWN
    reflexivity_note: str = ""

    def with_reflexivity_note(self) -> str:
        """Generate finding statement with reflexivity disclosure."""
        base = self.statement
        if self.robustness == RobustnessClassification.DISCLOSURE_ROBUST:
            note = "This finding holds under full-knowledge conditions."
        elif self.robustness == RobustnessClassification.CONDITIONALLY_VALID:
            note = "This finding assumes agents do not have access to this finding. Under full-knowledge conditions, the result may degrade."
        elif self.robustness == RobustnessClassification.FRAGILE:
            note = "WARNING: This finding inverts under full-knowledge conditions and should be treated as intelligence, not science."
        else:
            note = "Reflexivity robustness has not been tested."

        return f"{base}\n\n[Reflexivity Note: {note}]"


@dataclass
class SimulationSnapshot:
    """Snapshot of simulation state at a point in time."""

    timestamp: datetime
    metrics: dict[str, float]
    agent_strategies: dict[str, str]
    config: dict[str, Any]
    random_seed: int
    label: str = ""  # e.g., "T0_baseline", "T1_post_publication"


@dataclass
class ShadowSimulationResult:
    """Result of running shadow simulation comparison."""

    treatment_metrics: dict[str, list[float]]  # metric -> values over time
    control_metrics: dict[str, list[float]]
    divergence: dict[str, float]  # metric -> divergence score
    overall_divergence: float
    reflexivity_magnitude: float
    finding_is_robust: bool
    details: str = ""


class ShadowSimulation:
    """Run parallel simulations to measure reflexivity magnitude.

    From reflexivity.md #2:
    Run two parallel simulations from the same seed:
    - Treatment: Findings are "published" (injected into agent observations)
    - Control: Findings are withheld

    The divergence measures how much knowing the results changes the results.

    Seeds are shared only for initialization. After the findings injection
    point, random state is allowed to diverge naturally so that measured
    divergence reflects both the causal effect and the stochastic
    amplification of disclosure.
    """

    def __init__(
        self,
        simulation_fn: Callable[[dict, list[str] | None], dict[str, float]],
        divergence_threshold: float = 0.2,
    ):
        """Initialize shadow simulation.

        Args:
            simulation_fn: Function that runs simulation given config and optional
                          published findings. Returns dict of metric -> value.
            divergence_threshold: Maximum acceptable divergence for robustness.
        """
        self.simulation_fn = simulation_fn
        self.divergence_threshold = divergence_threshold

    def run(
        self,
        config: dict[str, Any],
        findings: list[str],
        epochs: int = 10,
        seed: int = 42,
    ) -> ShadowSimulationResult:
        """Run shadow simulation comparison.

        Args:
            config: Simulation configuration.
            findings: List of finding statements to inject in treatment.
            epochs: Number of epochs to run.
            seed: Random seed for reproducibility.

        Returns:
            ShadowSimulationResult with divergence analysis.
        """
        treatment_metrics: dict[str, list[float]] = {}
        control_metrics: dict[str, list[float]] = {}

        # Run parallel simulations
        for epoch in range(epochs):
            # Control: same seed for initialization, no findings
            np.random.seed(seed + epoch)
            control_result = self.simulation_fn(config, None)

            # Treatment: different seed offset so post-injection noise diverges
            # The seed shares the same base but uses a disjoint range,
            # ensuring initialization is comparable but stochastic paths differ.
            np.random.seed(seed + epoch + epochs)
            treatment_result = self.simulation_fn(config, findings)

            # Collect metrics
            for metric, value in control_result.items():
                control_metrics.setdefault(metric, []).append(value)
            for metric, value in treatment_result.items():
                treatment_metrics.setdefault(metric, []).append(value)

        # Compute divergence per metric
        divergence = {}
        for metric in control_metrics:
            control_vals = np.array(control_metrics[metric])
            treatment_vals = np.array(treatment_metrics.get(metric, control_vals))

            # Normalized divergence: |mean_diff| / pooled_std
            mean_diff = abs(np.mean(treatment_vals) - np.mean(control_vals))
            pooled_std = np.sqrt(
                (np.var(control_vals) + np.var(treatment_vals)) / 2 + 1e-10
            )
            divergence[metric] = mean_diff / pooled_std

        overall_divergence = np.mean(list(divergence.values()))
        reflexivity_magnitude = overall_divergence

        return ShadowSimulationResult(
            treatment_metrics=treatment_metrics,
            control_metrics=control_metrics,
            divergence=divergence,
            overall_divergence=overall_divergence,
            reflexivity_magnitude=reflexivity_magnitude,
            finding_is_robust=overall_divergence < self.divergence_threshold,
            details=self._generate_details(divergence, overall_divergence),
        )

    def _generate_details(self, divergence: dict[str, float], overall: float) -> str:
        """Generate human-readable analysis."""
        lines = ["Shadow Simulation Analysis:", ""]
        for metric, div in sorted(divergence.items(), key=lambda x: -x[1]):
            status = "HIGH" if div > self.divergence_threshold else "OK"
            lines.append(f"  {metric}: {div:.3f} {status}")
        lines.append("")
        lines.append(f"Overall divergence: {overall:.3f}")
        lines.append(
            f"Reflexivity magnitude: {'HIGH' if overall > self.divergence_threshold else 'LOW'}"
        )
        return "\n".join(lines)


@dataclass
class AttackResult:
    """Result of adversarial attack on finding."""

    attack_strategy: str
    governance_still_holds: bool
    metric_degradation: dict[str, float]
    successful_evasion: bool
    details: str = ""


@dataclass
class PublishThenAttackResult:
    """Result of publish-then-attack protocol."""

    finding: str
    attacks_tested: list[AttackResult]
    governance_holds_under_attack: bool
    robustness_classification: RobustnessClassification
    summary: str = ""


class PublishThenAttack:
    """Publish-then-attack protocol for testing disclosure robustness.

    From reflexivity.md #4:
    Before publishing a finding, red-team it:
    "Given that every agent now knows this finding, what attack strategies become viable?"
    """

    # Default per-metric evasion thresholds. Positive means "increase is bad"
    # (toxicity), negative means "decrease is bad" (welfare, quality_gap).
    DEFAULT_EVASION_THRESHOLDS: dict[str, float] = {
        "toxicity": 0.1,
        "welfare": -50.0,
        "quality_gap": -0.15,
    }

    def __init__(
        self,
        simulation_fn: Callable[[dict, list[str], str | None], dict[str, float]],
        attack_strategies: list[str] | None = None,
        evasion_thresholds: dict[str, float] | None = None,
    ):
        """Initialize publish-then-attack protocol.

        Args:
            simulation_fn: Function that runs simulation given config, findings,
                          and optional attack strategy. Returns metrics.
            attack_strategies: List of attack strategy names to test.
            evasion_thresholds: Per-metric thresholds for detecting evasion.
                              Positive threshold: degradation > threshold is evasion.
                              Negative threshold: degradation < threshold is evasion.
        """
        self.simulation_fn = simulation_fn
        self.attack_strategies = attack_strategies or [
            "direct_evasion",
            "metric_gaming",
            "collusive_adaptation",
            "boundary_exploitation",
        ]
        self.evasion_thresholds = evasion_thresholds or self.DEFAULT_EVASION_THRESHOLDS

    def _is_evaded(self, degradation: dict[str, float]) -> bool:
        """Check if any metric was evaded beyond its threshold."""
        for metric, threshold in self.evasion_thresholds.items():
            deg = degradation.get(metric, 0)
            if threshold >= 0:
                # Positive threshold: metric increase is bad (e.g. toxicity)
                if deg > threshold:
                    return True
            else:
                # Negative threshold: metric decrease is bad (e.g. welfare)
                if deg < threshold:
                    return True
        return False

    def run(
        self,
        config: dict[str, Any],
        finding: str,
        baseline_metrics: dict[str, float],
        trials: int = 5,
    ) -> PublishThenAttackResult:
        """Run publish-then-attack protocol.

        Args:
            config: Simulation configuration.
            finding: The finding to test.
            baseline_metrics: Metrics from original simulation.
            trials: Number of trials per attack strategy.

        Returns:
            PublishThenAttackResult with robustness classification.
        """
        attack_results = []

        for strategy in self.attack_strategies:
            # Run simulation with knowledge-augmented adversaries
            degradations = []
            evasions = []

            for _trial in range(trials):
                result = self.simulation_fn(config, [finding], strategy)

                # Compute degradation from baseline
                degradation = {}
                for metric, baseline_val in baseline_metrics.items():
                    if metric in result:
                        degradation[metric] = result[metric] - baseline_val
                degradations.append(degradation)

                # Check if any metric was evaded
                evasions.append(self._is_evaded(degradation))

            # Aggregate results
            avg_degradation = {
                metric: np.mean([d.get(metric, 0) for d in degradations])
                for metric in baseline_metrics
            }
            evasion_rate = np.mean(evasions)

            attack_results.append(
                AttackResult(
                    attack_strategy=strategy,
                    governance_still_holds=bool(evasion_rate < 0.5),
                    metric_degradation={
                        k: float(v) for k, v in avg_degradation.items()
                    },
                    successful_evasion=bool(evasion_rate > 0.5),
                    details=f"Evasion rate: {evasion_rate:.1%}",
                )
            )

        # Determine overall robustness
        all_hold = all(r.governance_still_holds for r in attack_results)
        any_evaded = any(r.successful_evasion for r in attack_results)

        if all_hold and not any_evaded:
            classification = RobustnessClassification.DISCLOSURE_ROBUST
        elif any_evaded and not all_hold:
            classification = RobustnessClassification.FRAGILE
        else:
            classification = RobustnessClassification.CONDITIONALLY_VALID

        return PublishThenAttackResult(
            finding=finding,
            attacks_tested=attack_results,
            governance_holds_under_attack=all_hold,
            robustness_classification=classification,
            summary=self._generate_summary(attack_results, classification),
        )

    def _generate_summary(
        self,
        results: list[AttackResult],
        classification: RobustnessClassification,
    ) -> str:
        """Generate human-readable summary."""
        lines = [
            "Publish-Then-Attack Protocol Results:",
            "",
            f"Classification: {classification.value.upper()}",
            "",
            "Attack Strategies Tested:",
        ]
        for r in results:
            status = "HELD" if r.governance_still_holds else "EVADED"
            lines.append(f"  {r.attack_strategy}: {status}")
            if r.metric_degradation:
                for metric, deg in r.metric_degradation.items():
                    if abs(deg) > 0.01:
                        lines.append(f"    {metric}: {deg:+.3f}")
        return "\n".join(lines)


@dataclass
class TemporalCheckpoint:
    """Checkpoint for temporal drift analysis."""

    label: str  # T0, T1, T2, etc.
    timestamp: datetime
    metrics: dict[str, float]
    is_publication_event: bool = False


class TemporalCheckpointing:
    """Temporal checkpointing for measuring publication-induced drift.

    From reflexivity.md #5:
    Snapshot platform dynamics at regular intervals to measure
    drift attributable to publication vs natural evolution.

    Checkpoints are stored sorted by timestamp.
    """

    def __init__(self) -> None:
        self.checkpoints: list[TemporalCheckpoint] = []

    def checkpoint(
        self,
        label: str,
        metrics: dict[str, float],
        is_publication: bool = False,
    ) -> TemporalCheckpoint:
        """Create a checkpoint. Inserted in timestamp order."""
        cp = TemporalCheckpoint(
            label=label,
            timestamp=datetime.now(timezone.utc),
            metrics=metrics,
            is_publication_event=is_publication,
        )
        self.checkpoints.append(cp)
        # Maintain sorted order by timestamp
        self.checkpoints.sort(key=lambda c: c.timestamp)
        return cp

    def _find(self, label: str) -> TemporalCheckpoint | None:
        """Find a checkpoint by label."""
        return next((c for c in self.checkpoints if c.label == label), None)

    def compute_drift(
        self,
        from_label: str,
        to_label: str,
    ) -> dict[str, float]:
        """Compute drift between two checkpoints."""
        from_cp = self._find(from_label)
        to_cp = self._find(to_label)

        if not from_cp or not to_cp:
            return {}

        drift = {}
        for metric in from_cp.metrics:
            if metric in to_cp.metrics:
                drift[metric] = to_cp.metrics[metric] - from_cp.metrics[metric]
        return drift

    def compute_publication_effect(self) -> dict[str, float]:
        """Compute excess drift attributable to publication.

        Returns drift(T0, T2) - drift(T0, T1) where T1 is post-publication.
        Uses sorted checkpoint order, not insertion order.
        """
        if len(self.checkpoints) < 3:
            return {}

        # Find publication checkpoint
        pub_idx = next(
            (i for i, c in enumerate(self.checkpoints) if c.is_publication_event),
            None,
        )
        if pub_idx is None or pub_idx == 0 or pub_idx >= len(self.checkpoints) - 1:
            return {}

        t0 = self.checkpoints[0]
        t1 = self.checkpoints[pub_idx]
        t2 = self.checkpoints[-1]

        drift_t0_t1 = {
            m: t1.metrics.get(m, 0) - t0.metrics.get(m, 0) for m in t0.metrics
        }
        drift_t0_t2 = {
            m: t2.metrics.get(m, 0) - t0.metrics.get(m, 0) for m in t0.metrics
        }

        # Excess drift = total drift - pre-publication drift
        excess = {m: drift_t0_t2.get(m, 0) - drift_t0_t1.get(m, 0) for m in t0.metrics}
        return excess


class GoodhartResistantMetrics:
    """Goodhart-resistant metric strategies.

    From reflexivity.md #3:
    Once you publish a metric threshold, agents optimize against it.
    """

    def __init__(self) -> None:
        self._holdout_metrics: dict[str, Callable] = {}
        self._published_metrics: list[str] = []
        self._metric_ensemble: dict[str, list[Callable]] = {}

    def register_holdout_metric(self, name: str, compute_fn: Callable) -> None:
        """Register a metric that won't be published."""
        self._holdout_metrics[name] = compute_fn

    def register_ensemble(self, name: str, metrics: list[Callable]) -> None:
        """Register an ensemble of metrics measuring same property."""
        self._metric_ensemble[name] = metrics

    def compute_composite(
        self,
        data: Any,
        metric_fns: dict[str, Callable],
        weights: dict[str, float] | None = None,
    ) -> float:
        """Compute composite metric (hard to jointly optimize)."""
        weights = weights or dict.fromkeys(metric_fns, 1.0)
        total_weight = sum(weights.values())

        score = 0.0
        for name, fn in metric_fns.items():
            value = fn(data)
            score += weights.get(name, 1.0) * value

        return score / total_weight

    def detect_gaming(
        self,
        data: Any,
        published_values: dict[str, float],
    ) -> dict[str, bool]:
        """Detect if published metrics are being gamed.

        Gaming is detected when published metrics look good but holdout
        metrics reveal problems.
        """
        gaming_detected = {}

        for name, fn in self._holdout_metrics.items():
            holdout_value = fn(data)
            # If holdout is bad but published are good, gaming suspected
            published_avg = np.mean(list(published_values.values()))
            if holdout_value > 0.5 and published_avg < 0.3:  # Thresholds configurable
                gaming_detected[name] = True
            else:
                gaming_detected[name] = False

        return gaming_detected

    def check_ensemble_consistency(
        self,
        data: Any,
        ensemble_name: str,
    ) -> tuple[bool, float]:
        """Check if ensemble metrics are consistent.

        Inconsistency suggests gaming of individual metrics.
        """
        if ensemble_name not in self._metric_ensemble:
            return True, 0.0

        values = [fn(data) for fn in self._metric_ensemble[ensemble_name]]
        if not values:
            return True, 0.0

        variance = np.var(values)
        consistent = variance < 0.1  # Threshold configurable
        return consistent, variance


@dataclass
class ReflexivityAnalysis:
    """Complete reflexivity analysis for a finding."""

    finding: Finding
    shadow_simulation: ShadowSimulationResult | None = None
    publish_then_attack: PublishThenAttackResult | None = None
    temporal_drift: dict[str, float] = field(default_factory=dict)
    final_classification: RobustnessClassification = RobustnessClassification.UNKNOWN
    epistemic_note: str = ""

    def generate_disclosure(self) -> str:
        """Generate reflexivity disclosure for publication."""
        lines = ["## Reflexivity Analysis", ""]

        if self.shadow_simulation:
            lines.append(
                f"**Shadow Simulation Divergence**: {self.shadow_simulation.overall_divergence:.3f}"
            )
            lines.append(
                f"**Finding Robust to Self-Knowledge**: {'Yes' if self.shadow_simulation.finding_is_robust else 'No'}"
            )
            lines.append("")

        if self.publish_then_attack:
            lines.append(
                f"**Disclosure Robustness**: {self.publish_then_attack.robustness_classification.value}"
            )
            lines.append(
                f"**Governance Holds Under Attack**: {'Yes' if self.publish_then_attack.governance_holds_under_attack else 'No'}"
            )
            lines.append("")

        lines.append(
            f"**Final Classification**: {self.final_classification.value.upper()}"
        )
        lines.append("")
        lines.append(f"**Epistemic Note**: {self.epistemic_note}")

        return "\n".join(lines)


class ReflexivityAnalyzer:
    """Complete reflexivity analysis pipeline."""

    def __init__(
        self,
        simulation_fn: Callable,
        attack_simulation_fn: Callable | None = None,
    ):
        self.shadow_sim = ShadowSimulation(simulation_fn)
        self.publish_attack = PublishThenAttack(attack_simulation_fn or simulation_fn)
        self.temporal = TemporalCheckpointing()
        self.goodhart = GoodhartResistantMetrics()

    def analyze(
        self,
        finding: Finding,
        config: dict[str, Any],
        baseline_metrics: dict[str, float],
    ) -> ReflexivityAnalysis:
        """Run complete reflexivity analysis."""

        # 1. Shadow simulation
        shadow_result = self.shadow_sim.run(
            config=config,
            findings=[finding.statement],
            epochs=10,
        )

        # 2. Publish-then-attack
        attack_result = self.publish_attack.run(
            config=config,
            finding=finding.statement,
            baseline_metrics=baseline_metrics,
        )

        # 3. Determine final classification
        if (
            shadow_result.finding_is_robust
            and attack_result.governance_holds_under_attack
        ):
            classification = RobustnessClassification.DISCLOSURE_ROBUST
            note = "This finding holds under full-knowledge conditions."
        elif (
            not shadow_result.finding_is_robust
            and not attack_result.governance_holds_under_attack
        ):
            classification = RobustnessClassification.FRAGILE
            note = "WARNING: This finding inverts under full-knowledge conditions."
        else:
            classification = RobustnessClassification.CONDITIONALLY_VALID
            note = "This finding assumes agents do not have access to this finding."

        # Update finding
        finding.robustness = classification
        finding.reflexivity_note = note

        return ReflexivityAnalysis(
            finding=finding,
            shadow_simulation=shadow_result,
            publish_then_attack=attack_result,
            final_classification=classification,
            epistemic_note=note,
        )
