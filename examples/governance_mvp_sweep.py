#!/usr/bin/env python3
"""
Governance MVP Sweep — 12 runs comparing individual governance mechanisms.

Each run isolates a single governance lever (or combination) against a
shared baseline to measure its effect on toxicity, welfare, quality gap,
and per-agent-type payoffs.

Runs:
  1. Baseline (no governance)
  2. Reputation decay
  3. Vote-weight normalization
  4. Bandwidth caps
  5. Transparency ledger
  6. Random audits
  7. Circuit breaker
  8. Transaction tax
  9. Staking / bonding
 10. Moderator agent
 11. Collusion detection
 12. Combined (defense-in-depth)

Usage:
    python examples/governance_mvp_sweep.py
    python examples/governance_mvp_sweep.py --output results/mvp_sweep.csv
    python examples/governance_mvp_sweep.py --scenario scenarios/baseline.yaml
    python examples/governance_mvp_sweep.py --runs-per-config 3
    python examples/governance_mvp_sweep.py --epochs 10
"""

import argparse
import copy
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis.sweep import SweepResult, _apply_params, _extract_results
from swarm.governance.config import GovernanceConfig
from swarm.scenarios import ScenarioConfig, build_orchestrator, load_scenario

# ── Named governance configurations ──────────────────────────────────────


def _governance_configs() -> List[Dict[str, Any]]:
    """
    Return the 12 named governance configurations for the MVP sweep.

    Each entry maps ``governance.*`` parameter names to their values.
    Unlisted parameters keep their baseline (disabled) defaults.
    """
    return [
        # 1. Baseline — all governance disabled
        {
            "label": "baseline",
            "params": {},
        },
        # 2. Reputation decay — reputations decay 10% per epoch
        {
            "label": "reputation_decay",
            "params": {
                "governance.reputation_decay_rate": 0.9,
            },
        },
        # 3. Vote-weight normalization — diminishing vote influence
        {
            "label": "vote_normalization",
            "params": {
                "governance.vote_normalization_enabled": True,
                "governance.max_vote_weight_per_agent": 5.0,
            },
        },
        # 4. Bandwidth caps — restrict interactions per step
        {
            "label": "bandwidth_caps",
            "params": {
                "governance.bandwidth_cap": 3,
            },
        },
        # 5. Transparency ledger — public visibility of interaction quality
        {
            "label": "transparency_ledger",
            "params": {
                "governance.transparency_enabled": True,
                "governance.transparency_bonus_rate": 0.15,
                "governance.transparency_threshold_p": 0.5,
            },
        },
        # 6. Random audits — probabilistic quality checks with penalties
        {
            "label": "random_audits",
            "params": {
                "governance.audit_enabled": True,
                "governance.audit_probability": 0.2,
                "governance.audit_penalty_multiplier": 2.0,
                "governance.audit_threshold_p": 0.5,
            },
        },
        # 7. Circuit breaker — freeze high-toxicity agents
        {
            "label": "circuit_breaker",
            "params": {
                "governance.circuit_breaker_enabled": True,
                "governance.freeze_threshold_toxicity": 0.6,
                "governance.freeze_threshold_violations": 2,
                "governance.freeze_duration_epochs": 2,
            },
        },
        # 8. Transaction tax — per-interaction tax on surplus
        {
            "label": "transaction_tax",
            "params": {
                "governance.transaction_tax_rate": 0.10,
                "governance.transaction_tax_split": 0.5,
            },
        },
        # 9. Staking / bonding — minimum stake to participate
        {
            "label": "staking_bonding",
            "params": {
                "governance.staking_enabled": True,
                "governance.min_stake_to_participate": 5.0,
                "governance.stake_slash_rate": 0.15,
            },
        },
        # 10. Moderator agent — continuous oversight with moderate penalties
        {
            "label": "moderator_agent",
            "params": {
                "governance.moderator_enabled": True,
                "governance.moderator_review_probability": 0.3,
                "governance.moderator_penalty_multiplier": 1.5,
                "governance.moderator_threshold_p": 0.4,
            },
        },
        # 11. Collusion detection — detect and penalize coordinated pairs
        {
            "label": "collusion_detection",
            "params": {
                "governance.collusion_detection_enabled": True,
                "governance.collusion_frequency_threshold": 2.0,
                "governance.collusion_correlation_threshold": 0.7,
                "governance.collusion_score_threshold": 0.5,
                "governance.collusion_penalty_multiplier": 1.0,
                "governance.collusion_realtime_penalty": True,
                "governance.collusion_realtime_rate": 0.1,
            },
        },
        # 12. Combined (defense-in-depth) — multiple mechanisms layered
        {
            "label": "combined_defense_in_depth",
            "params": {
                "governance.reputation_decay_rate": 0.9,
                "governance.transaction_tax_rate": 0.05,
                "governance.audit_enabled": True,
                "governance.audit_probability": 0.15,
                "governance.audit_penalty_multiplier": 2.0,
                "governance.audit_threshold_p": 0.5,
                "governance.circuit_breaker_enabled": True,
                "governance.freeze_threshold_toxicity": 0.7,
                "governance.freeze_threshold_violations": 3,
                "governance.freeze_duration_epochs": 2,
                "governance.transparency_enabled": True,
                "governance.transparency_bonus_rate": 0.1,
                "governance.moderator_enabled": True,
                "governance.moderator_review_probability": 0.2,
            },
        },
    ]


# ── Sweep runner ─────────────────────────────────────────────────────────


@dataclass
class NamedSweepResult:
    """Result from a named governance configuration run."""

    label: str
    sweep_result: SweepResult

    def to_dict(self) -> Dict[str, Any]:
        d = self.sweep_result.to_dict()
        d["governance_label"] = self.label
        return d


def run_governance_sweep(
    base_scenario: ScenarioConfig,
    *,
    runs_per_config: int = 1,
    seed_base: int = 42,
    n_epochs: Optional[int] = None,
    progress: bool = True,
) -> List[NamedSweepResult]:
    """
    Run the 12-configuration governance MVP sweep.

    Args:
        base_scenario: Base scenario to modify for each run
        runs_per_config: Number of runs per configuration
        seed_base: Base random seed (incremented per run)
        n_epochs: Override number of epochs (None = use scenario default)
        progress: Print progress to stdout

    Returns:
        List of NamedSweepResult for all runs
    """
    configs = _governance_configs()
    total_runs = len(configs) * runs_per_config
    results: List[NamedSweepResult] = []
    current_run = 0

    for config_entry in configs:
        label = config_entry["label"]
        params = config_entry["params"]

        for run_idx in range(runs_per_config):
            current_run += 1
            seed = seed_base + current_run

            if progress:
                print(
                    f"  [{current_run:>3}/{total_runs}] "
                    f"{label:<30} (run {run_idx + 1}/{runs_per_config}, seed={seed})"
                )

            # Deep copy and apply parameters
            scenario = copy.deepcopy(base_scenario)

            # Override epochs if requested
            if n_epochs is not None:
                scenario.orchestrator_config.n_epochs = n_epochs

            # Ensure governance config exists
            if scenario.orchestrator_config.governance_config is None:
                scenario.orchestrator_config.governance_config = GovernanceConfig()

            scenario = _apply_params(scenario, params)

            # Set seed
            scenario.orchestrator_config.seed = seed

            # Build and run
            orchestrator = build_orchestrator(scenario)
            orchestrator.run()

            # Extract results
            sweep_result = _extract_results(orchestrator, params, run_idx, seed)
            results.append(NamedSweepResult(label=label, sweep_result=sweep_result))

    return results


def export_csv(results: List[NamedSweepResult], path: Path) -> None:
    """Export sweep results to CSV."""
    if not results:
        raise ValueError("No results to export")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Collect union of all keys across results (different configs have
    # different governance parameter keys).
    all_keys: dict[str, None] = {}
    for result in results:
        for key in result.to_dict():
            all_keys[key] = None
    fieldnames = list(all_keys)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for result in results:
            writer.writerow(result.to_dict())


def print_summary(results: List[NamedSweepResult]) -> None:
    """Print a formatted summary table of the sweep results."""
    # Group by label
    from collections import defaultdict

    groups: Dict[str, List[NamedSweepResult]] = defaultdict(list)
    for r in results:
        groups[r.label].append(r)

    # Header
    print()
    print(
        f"{'Mechanism':<30} "
        f"{'Welfare':>10} "
        f"{'Toxicity':>10} "
        f"{'QualGap':>10} "
        f"{'Frozen':>7} "
        f"{'Honest':>10} "
        f"{'Opport':>10} "
        f"{'Decept':>10} "
        f"{'Advers':>10}"
    )
    print("-" * 127)

    for label in dict.fromkeys(r.label for r in results):
        runs = groups[label]
        n = len(runs)

        mean_welfare = sum(r.sweep_result.total_welfare for r in runs) / n
        mean_toxicity = sum(r.sweep_result.avg_toxicity for r in runs) / n
        mean_qgap = sum(r.sweep_result.avg_quality_gap for r in runs) / n
        mean_frozen = sum(r.sweep_result.n_frozen for r in runs) / n
        mean_honest = sum(r.sweep_result.honest_avg_payoff for r in runs) / n
        mean_oppo = sum(r.sweep_result.opportunistic_avg_payoff for r in runs) / n
        mean_decept = sum(r.sweep_result.deceptive_avg_payoff for r in runs) / n
        mean_advers = sum(r.sweep_result.adversarial_avg_payoff for r in runs) / n

        print(
            f"{label:<30} "
            f"{mean_welfare:>10.2f} "
            f"{mean_toxicity:>10.4f} "
            f"{mean_qgap:>10.4f} "
            f"{mean_frozen:>7.1f} "
            f"{mean_honest:>10.2f} "
            f"{mean_oppo:>10.2f} "
            f"{mean_decept:>10.2f} "
            f"{mean_advers:>10.2f}"
        )

    print()


# ── CLI entry point ──────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Governance MVP Sweep — 12 mechanism comparison runs"
    )
    parser.add_argument(
        "--scenario",
        "-s",
        type=Path,
        default=Path("scenarios/baseline.yaml"),
        help="Base scenario YAML (default: scenarios/baseline.yaml)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("logs/governance_mvp_sweep.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--runs-per-config",
        "-r",
        type=int,
        default=1,
        help="Number of runs per configuration (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        "-e",
        type=int,
        default=None,
        help="Override epoch count (default: use scenario value)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  Governance MVP Sweep — 12 Mechanism Comparison")
    print("=" * 70)
    print()

    # Load base scenario
    print(f"Base scenario:    {args.scenario}")
    base_scenario = load_scenario(args.scenario)

    n_configs = len(_governance_configs())
    total_runs = n_configs * args.runs_per_config
    print(f"Configurations:   {n_configs}")
    print(f"Runs per config:  {args.runs_per_config}")
    print(f"Total runs:       {total_runs}")
    if args.epochs is not None:
        print(f"Epochs override:  {args.epochs}")
    print(f"Seed base:        {args.seed}")
    print()

    # Run sweep
    print("Running sweep...")
    results = run_governance_sweep(
        base_scenario,
        runs_per_config=args.runs_per_config,
        seed_base=args.seed,
        n_epochs=args.epochs,
    )

    # Print summary
    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)
    print_summary(results)

    # Export CSV
    print(f"Results exported to: {args.output}")
    export_csv(results, args.output)
    print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
