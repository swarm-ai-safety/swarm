#!/usr/bin/env python
"""
Sweep: Cautious Reciprocator — governance ON vs OFF.

Compares the cautious_reciprocator scenario with all governance levers
enabled vs completely disabled, across multiple seeds.

Usage:
    python examples/cautious_governance_sweep.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.analysis import SweepConfig, SweepParameter, SweepRunner
from swarm.scenarios import load_scenario


def progress(current: int, total: int, params: dict) -> None:
    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
    print(f"  [{current}/{total}] {param_str}")


def main():
    scenario = load_scenario("scenarios/cautious_vs_adversaries.yaml")
    scenario.orchestrator_config.n_epochs = 15

    sweep_config = SweepConfig(
        base_scenario=scenario,
        parameters=[
            SweepParameter(
                name="governance.transaction_tax_rate",
                values=[0.0, 0.05],
            ),
            SweepParameter(
                name="governance.circuit_breaker_enabled",
                values=[False, True],
            ),
            SweepParameter(
                name="governance.audit_enabled",
                values=[False, True],
            ),
            SweepParameter(
                name="governance.reputation_decay_rate",
                values=[1.0, 0.95],
            ),
        ],
        runs_per_config=3,
        seed_base=42,
    )

    combos = sweep_config.total_runs()
    n_params = 1
    for p in sweep_config.parameters:
        n_params *= len(p.values)

    print("=" * 70)
    print("Cautious Reciprocator — Governance Sweep")
    print("=" * 70)
    print("  Scenario: cautious_vs_adversaries.yaml")
    print("  Agents: 3 cautious_reciprocator, 2 honest, 2 deceptive, 2 adversarial, 1 adaptive")
    print(f"  Parameter combinations: {n_params}")
    print(f"  Runs per config: {sweep_config.runs_per_config}")
    print(f"  Total runs: {combos}")
    print("  Epochs per run: 15")
    print()

    for p in sweep_config.parameters:
        print(f"  {p.name}: {p.values}")
    print()

    print("Running sweep...")
    print("-" * 70)
    runner = SweepRunner(sweep_config, progress_callback=progress)
    runner.run()

    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    summary = runner.summary()
    print(f"\nTotal runs: {summary['total_runs']}")
    print(f"Unique configs: {summary['param_combinations']}")
    print()

    # Header
    print(
        f"{'Tax':<6} {'CB':<5} {'Audit':<7} {'Decay':<7} "
        f"{'Welfare':>10} {'Toxicity':>10} {'Frozen':>8} "
        f"{'Honest$':>10} {'Advers$':>10}"
    )
    print("-" * 83)

    for s in summary["summaries"]:
        tax = s.get("governance.transaction_tax_rate", 0)
        cb = s.get("governance.circuit_breaker_enabled", False)
        audit = s.get("governance.audit_enabled", False)
        decay = s.get("governance.reputation_decay_rate", 1.0)
        print(
            f"{tax:<6.2f} {'Y' if cb else 'N':<5} {'Y' if audit else 'N':<7} {decay:<7.2f} "
            f"{s['mean_welfare']:>10.2f} {s['mean_toxicity']:>10.4f} {s['mean_frozen']:>8.1f} "
            f"{s['mean_honest_payoff']:>10.2f} {s['mean_adversarial_payoff']:>10.2f}"
        )

    print()

    # Find best and worst configs
    by_welfare = sorted(summary["summaries"], key=lambda s: s["mean_welfare"], reverse=True)
    by_toxicity = sorted(summary["summaries"], key=lambda s: s["mean_toxicity"])

    def config_label(s):
        tax = s.get("governance.transaction_tax_rate", 0)
        cb = "CB" if s.get("governance.circuit_breaker_enabled", False) else "noCB"
        audit = "audit" if s.get("governance.audit_enabled", False) else "noAudit"
        decay = s.get("governance.reputation_decay_rate", 1.0)
        return f"tax={tax:.2f}, {cb}, {audit}, decay={decay:.2f}"

    print("Best config by welfare:")
    best = by_welfare[0]
    print(f"  {config_label(best)}")
    print(f"  welfare={best['mean_welfare']:.2f}, toxicity={best['mean_toxicity']:.4f}")

    print()
    print("Best config by toxicity:")
    best_tox = by_toxicity[0]
    print(f"  {config_label(best_tox)}")
    print(f"  welfare={best_tox['mean_welfare']:.2f}, toxicity={best_tox['mean_toxicity']:.4f}")

    print()
    print("Worst config (highest toxicity):")
    worst = by_toxicity[-1]
    print(f"  {config_label(worst)}")
    print(f"  welfare={worst['mean_welfare']:.2f}, toxicity={worst['mean_toxicity']:.4f}")

    # Governance ON (all levers) vs OFF (no levers)
    print()
    print("=" * 70)
    print("Head-to-Head: Full Governance vs No Governance")
    print("=" * 70)

    gov_on = [s for s in summary["summaries"]
              if s.get("governance.transaction_tax_rate", 0) > 0
              and s.get("governance.circuit_breaker_enabled", False)
              and s.get("governance.audit_enabled", False)
              and s.get("governance.reputation_decay_rate", 1.0) < 1.0]

    gov_off = [s for s in summary["summaries"]
               if s.get("governance.transaction_tax_rate", 0) == 0
               and not s.get("governance.circuit_breaker_enabled", False)
               and not s.get("governance.audit_enabled", False)
               and s.get("governance.reputation_decay_rate", 1.0) >= 1.0]

    if gov_on and gov_off:
        on = gov_on[0]
        off = gov_off[0]

        print(f"\n{'Metric':<25} {'Gov OFF':>12} {'Gov ON':>12} {'Delta':>12}")
        print("-" * 65)
        for key, label in [
            ("mean_welfare", "Welfare"),
            ("mean_toxicity", "Toxicity"),
            ("mean_frozen", "Frozen agents"),
            ("mean_honest_payoff", "Honest payoff"),
            ("mean_adversarial_payoff", "Adversarial payoff"),
        ]:
            v_off = off.get(key, 0)
            v_on = on.get(key, 0)
            delta = v_on - v_off
            sign = "+" if delta > 0 else ""
            print(f"{label:<25} {v_off:>12.4f} {v_on:>12.4f} {sign}{delta:>11.4f}")
    else:
        print("  (Could not find exact full-on / full-off configs)")

    # Export
    out_path = Path("logs/cautious_governance_sweep.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.to_csv(out_path)
    print(f"\nExported CSV: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
