"""Parameter sweep for batch simulations."""

import csv
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from swarm.core.orchestrator import Orchestrator
from swarm.governance.config import GovernanceConfig
from swarm.scenarios import ScenarioConfig, build_orchestrator


@dataclass
class SweepParameter:
    """A parameter to sweep over."""

    name: str  # e.g., "governance.transaction_tax_rate"
    values: List[Any]  # Values to sweep

    def __post_init__(self):
        if not self.values:
            raise ValueError(f"Parameter {self.name} must have at least one value")


@dataclass
class SweepConfig:
    """Configuration for a parameter sweep."""

    # Base scenario to modify
    base_scenario: ScenarioConfig

    # Parameters to sweep
    parameters: List[SweepParameter] = field(default_factory=list)

    # Number of runs per parameter combination (for statistical significance)
    runs_per_config: int = 1

    # Random seed base (incremented for each run)
    seed_base: Optional[int] = 42

    def add_parameter(self, name: str, values: List[Any]) -> "SweepConfig":
        """Add a parameter to sweep (fluent API)."""
        self.parameters.append(SweepParameter(name=name, values=values))
        return self

    def total_runs(self) -> int:
        """Calculate total number of simulation runs."""
        if not self.parameters:
            return self.runs_per_config

        n_combinations = 1
        for param in self.parameters:
            n_combinations *= len(param.values)

        return n_combinations * self.runs_per_config


@dataclass
class SweepResult:
    """Result from a single simulation run."""

    # Parameter values for this run
    params: Dict[str, Any]
    run_index: int
    seed: int

    # Aggregate metrics
    total_interactions: int = 0
    accepted_interactions: int = 0
    avg_toxicity: float = 0.0
    avg_quality_gap: float = 0.0
    total_welfare: float = 0.0
    welfare_per_epoch: float = 0.0

    # Agent outcomes
    n_agents: int = 0
    n_frozen: int = 0
    avg_reputation: float = 0.0
    avg_payoff: float = 0.0
    min_payoff: float = 0.0
    max_payoff: float = 0.0

    # Per-type payoffs
    honest_avg_payoff: float = 0.0
    opportunistic_avg_payoff: float = 0.0
    deceptive_avg_payoff: float = 0.0
    adversarial_avg_payoff: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary for CSV export."""
        result = dict(self.params)
        result.update(
            {
                "run_index": self.run_index,
                "seed": self.seed,
                "total_interactions": self.total_interactions,
                "accepted_interactions": self.accepted_interactions,
                "avg_toxicity": self.avg_toxicity,
                "avg_quality_gap": self.avg_quality_gap,
                "total_welfare": self.total_welfare,
                "welfare_per_epoch": self.welfare_per_epoch,
                "n_agents": self.n_agents,
                "n_frozen": self.n_frozen,
                "avg_reputation": self.avg_reputation,
                "avg_payoff": self.avg_payoff,
                "min_payoff": self.min_payoff,
                "max_payoff": self.max_payoff,
                "honest_avg_payoff": self.honest_avg_payoff,
                "opportunistic_avg_payoff": self.opportunistic_avg_payoff,
                "deceptive_avg_payoff": self.deceptive_avg_payoff,
                "adversarial_avg_payoff": self.adversarial_avg_payoff,
            }
        )
        return result


def _set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Set a nested attribute using dot notation."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def _apply_params(scenario: ScenarioConfig, params: Dict[str, Any]) -> ScenarioConfig:
    """Apply parameter overrides to a scenario config.

    Supported parameter name patterns:
    - ``governance.<attr>`` — governance config attribute
    - ``payoff.<attr>`` — payoff config attribute
    - ``simulation.<attr>`` — orchestrator config attribute
    - ``agents.<type>.config.<attr>`` — agent config for all agents of ``<type>``
    - ``n_epochs``, ``steps_per_epoch`` — simulation parameters
    - Any direct orchestrator config attribute (fallback)
    """
    # Map parameter names to config paths
    config = scenario.orchestrator_config

    for name, value in params.items():
        if name.startswith("governance."):
            attr = name[len("governance."):]
            if config.governance_config is None:
                config.governance_config = GovernanceConfig()
            setattr(config.governance_config, attr, value)

        elif name.startswith("payoff."):
            attr = name[len("payoff."):]
            setattr(config.payoff_config, attr, value)

        elif name.startswith("simulation."):
            attr = name[len("simulation."):]
            setattr(config, attr, value)

        elif name.startswith("agents."):
            # agents.<type>.config.<attr>  e.g. agents.ldt.config.acausality_depth
            parts = name.split(".")
            if len(parts) >= 4 and parts[2] == "config":
                agent_type = parts[1]
                attr = ".".join(parts[3:])
                for spec in scenario.agent_specs:
                    if spec.get("type") == agent_type:
                        if "config" not in spec:
                            spec["config"] = {}
                        spec["config"][attr] = value

        elif name == "n_epochs":
            config.n_epochs = value

        elif name == "steps_per_epoch":
            config.steps_per_epoch = value

        else:
            # Try to set directly on orchestrator config
            if hasattr(config, name):
                setattr(config, name, value)

    return scenario


def _extract_results(
    orchestrator: Orchestrator,
    params: Dict[str, Any],
    run_index: int,
    seed: int,
) -> SweepResult:
    """Extract metrics from a completed simulation."""
    metrics_history = orchestrator.get_metrics_history()
    n_epochs = len(metrics_history)

    # Aggregate metrics
    total_interactions = sum(m.total_interactions for m in metrics_history)
    accepted_interactions = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / n_epochs if n_epochs else 0
    )
    avg_quality_gap = (
        sum(m.quality_gap for m in metrics_history) / n_epochs if n_epochs else 0
    )
    total_welfare = sum(m.total_welfare for m in metrics_history)
    welfare_per_epoch = total_welfare / n_epochs if n_epochs else 0

    # Agent outcomes
    agents = orchestrator.get_all_agents()
    n_agents = len(agents)
    n_frozen = len(orchestrator.state.frozen_agents)

    payoffs = []
    reputations = []
    payoffs_by_type: Dict[str, List[float]] = {
        "honest": [],
        "opportunistic": [],
        "deceptive": [],
        "adversarial": [],
    }

    for agent in agents:
        state = orchestrator.state.get_agent(agent.agent_id)
        if state:
            payoffs.append(state.total_payoff)
            reputations.append(state.reputation)
            agent_type = agent.agent_type.value
            if agent_type in payoffs_by_type:
                payoffs_by_type[agent_type].append(state.total_payoff)

    avg_payoff = sum(payoffs) / len(payoffs) if payoffs else 0
    avg_reputation = sum(reputations) / len(reputations) if reputations else 0
    min_payoff = min(payoffs) if payoffs else 0
    max_payoff = max(payoffs) if payoffs else 0

    def avg_or_zero(lst: List[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return SweepResult(
        params=params,
        run_index=run_index,
        seed=seed,
        total_interactions=total_interactions,
        accepted_interactions=accepted_interactions,
        avg_toxicity=avg_toxicity,
        avg_quality_gap=avg_quality_gap,
        total_welfare=total_welfare,
        welfare_per_epoch=welfare_per_epoch,
        n_agents=n_agents,
        n_frozen=n_frozen,
        avg_reputation=avg_reputation,
        avg_payoff=avg_payoff,
        min_payoff=min_payoff,
        max_payoff=max_payoff,
        honest_avg_payoff=avg_or_zero(payoffs_by_type["honest"]),
        opportunistic_avg_payoff=avg_or_zero(payoffs_by_type["opportunistic"]),
        deceptive_avg_payoff=avg_or_zero(payoffs_by_type["deceptive"]),
        adversarial_avg_payoff=avg_or_zero(payoffs_by_type["adversarial"]),
    )


class SweepRunner:
    """Runs parameter sweeps over simulations."""

    def __init__(
        self,
        config: SweepConfig,
        progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
    ):
        """
        Initialize the sweep runner.

        Args:
            config: Sweep configuration
            progress_callback: Optional callback(current, total, params) for progress updates
        """
        self.config = config
        self.progress_callback = progress_callback
        self.results: List[SweepResult] = []

    def run(self) -> List[SweepResult]:
        """
        Execute the parameter sweep.

        Returns:
            List of results for each run
        """
        self.results = []

        # Generate all parameter combinations
        if self.config.parameters:
            param_names = [p.name for p in self.config.parameters]
            param_values = [p.values for p in self.config.parameters]
            combinations = list(itertools.product(*param_values))
        else:
            param_names = []
            combinations = [()]

        total_runs = self.config.total_runs()
        current_run = 0

        for combo in combinations:
            params = dict(zip(param_names, combo, strict=True))

            for run_idx in range(self.config.runs_per_config):
                current_run += 1

                # Calculate seed
                seed = (
                    self.config.seed_base + current_run
                    if self.config.seed_base
                    else None
                )

                # Progress callback
                if self.progress_callback:
                    self.progress_callback(current_run, total_runs, params)

                # Run simulation
                result = self._run_single(params, run_idx, seed)
                self.results.append(result)

        return self.results

    def _run_single(
        self,
        params: Dict[str, Any],
        run_index: int,
        seed: Optional[int],
    ) -> SweepResult:
        """Run a single simulation with given parameters."""
        # Deep copy the base scenario
        import copy

        scenario = copy.deepcopy(self.config.base_scenario)

        # Apply parameter overrides
        scenario = _apply_params(scenario, params)

        # Set seed
        if seed is not None:
            scenario.orchestrator_config.seed = seed

        # Build and run
        orchestrator = build_orchestrator(scenario)
        orchestrator.run()

        # Extract results
        return _extract_results(orchestrator, params, run_index, seed or 0)

    # Canonical column names expected by downstream tools (/stats, /plot, papers).
    # Maps internal SweepResult field names → canonical names.
    COLUMN_ALIASES: Dict[str, str] = {
        "avg_toxicity": "toxicity_rate",
        "total_welfare": "welfare",
        "avg_quality_gap": "quality_gap",
        "honest_avg_payoff": "honest_payoff",
        "opportunistic_avg_payoff": "opportunistic_payoff",
        "deceptive_avg_payoff": "deceptive_payoff",
        "adversarial_avg_payoff": "adversarial_payoff",
    }

    def to_csv(self, path: Union[str, Path]) -> None:
        """Export results to CSV with canonical column names."""
        if not self.results:
            raise ValueError("No results to export. Run sweep first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [self._normalize_columns(r.to_dict()) for r in self.results]
        fieldnames = list(rows[0].keys())

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @classmethod
    def _normalize_columns(cls, row: Dict[str, Any]) -> Dict[str, Any]:
        """Rename internal field names to canonical names for CSV export."""
        out: Dict[str, Any] = {}
        for key, value in row.items():
            canonical = cls.COLUMN_ALIASES.get(key, key)
            out[canonical] = value
        return out

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics across all runs."""
        if not self.results:
            return {}

        # Group by parameter combination
        from collections import defaultdict

        groups: Dict[tuple, List[SweepResult]] = defaultdict(list)

        for r in self.results:
            key = tuple(sorted(r.params.items()))
            groups[key].append(r)

        summaries = []
        for param_key, runs in groups.items():
            params = dict(param_key)
            n_runs = len(runs)

            summary = {
                **params,
                "n_runs": n_runs,
                "mean_welfare": sum(r.total_welfare for r in runs) / n_runs,
                "mean_toxicity": sum(r.avg_toxicity for r in runs) / n_runs,
                "mean_frozen": sum(r.n_frozen for r in runs) / n_runs,
                "mean_honest_payoff": sum(r.honest_avg_payoff for r in runs) / n_runs,
                "mean_adversarial_payoff": sum(r.adversarial_avg_payoff for r in runs)
                / n_runs,
            }
            summaries.append(summary)

        return {
            "total_runs": len(self.results),
            "param_combinations": len(groups),
            "summaries": summaries,
        }


def quick_sweep(
    scenario: ScenarioConfig,
    param_name: str,
    values: List[Any],
    runs_per_config: int = 1,
    seed_base: int = 42,
) -> List[SweepResult]:
    """
    Convenience function for single-parameter sweep.

    Args:
        scenario: Base scenario
        param_name: Parameter to sweep (e.g., "governance.transaction_tax_rate")
        values: Values to test
        runs_per_config: Runs per parameter value
        seed_base: Base random seed

    Returns:
        List of sweep results
    """
    config = SweepConfig(
        base_scenario=scenario,
        parameters=[SweepParameter(name=param_name, values=values)],
        runs_per_config=runs_per_config,
        seed_base=seed_base,
    )
    runner = SweepRunner(config)
    return runner.run()
