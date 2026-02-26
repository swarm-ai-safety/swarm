"""CLI entry point for running escalation sandbox scenarios.

Usage:
    python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_sandbox.yaml
    python -m swarm.domains.escalation_sandbox.run_scenario scenarios/escalation_sandbox.yaml --seed 42
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from swarm.domains.escalation_sandbox.config import EscalationConfig
from swarm.domains.escalation_sandbox.runner import EscalationRunner


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an Escalation Sandbox scenario",
    )
    parser.add_argument(
        "scenario",
        type=str,
        help="Path to scenario YAML file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides YAML seed)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load scenario
    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: scenario file not found: {scenario_path}", file=sys.stderr)
        sys.exit(1)

    with open(scenario_path) as f:
        raw = yaml.safe_load(f)

    # Parse domain config
    domain_data = raw.get("domain", raw)
    config = EscalationConfig.from_dict(domain_data)

    # Override seed if provided
    seed = args.seed if args.seed is not None else config.seed

    # Run
    runner = EscalationRunner(config, seed=seed)
    metrics = runner.run()

    # Export
    run_dir = runner.export(output_dir=args.output_dir)

    # Print summary
    print(f"\n{'=' * 60}")
    print("ESCALATION SANDBOX RESULTS")
    print(f"{'=' * 60}")
    print(f"Outcome:              {metrics.outcome}")
    print(f"Turns played:         {metrics.turns_played}")
    print(f"Max escalation:       {metrics.escalation_max}")
    print(f"Nuclear threshold:    turn {metrics.nuclear_threshold_turn}")
    print(f"Escalation velocity:  {metrics.escalation_velocity:.3f}")
    print(f"Signal-action div:    {metrics.signal_action_divergence:.3f}")
    print(f"Trust exploitation:   {metrics.trust_exploitation_index:.3f}")
    print(f"Accidental esc rate:  {metrics.accidental_escalation_rate:.3f}")
    print(f"Fog catastrophes:     {metrics.fog_catastrophe_count}")
    print(f"De-escalation rate:   {metrics.de_escalation_rate:.3f}")
    print(f"Collateral damage:    {metrics.collateral_damage:.1f}")
    print(f"Welfare composite:    {metrics.welfare_composite:.1f}")
    print(f"Governance backfire:  {metrics.governance_backfire_count}")
    print(f"Results exported to:  {run_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
