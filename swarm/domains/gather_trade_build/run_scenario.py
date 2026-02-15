"""CLI entry point for running GTB scenarios.

Usage:
    python -m swarm.domains.gather_trade_build.run_scenario scenarios/ai_economist_full.yaml
    python -m swarm.domains.gather_trade_build.run_scenario scenarios/ai_economist_full.yaml --seed 123 --epochs 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

from swarm.domains.gather_trade_build.config import GTBConfig
from swarm.domains.gather_trade_build.runner import GTBScenarioRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run an AI Economist GTB scenario")
    parser.add_argument("scenario", type=Path, help="Path to scenario YAML")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--epochs", type=int, default=None, help="Override n_epochs")
    parser.add_argument("--steps", type=int, default=None, help="Override steps_per_epoch")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args(argv)

    with open(args.scenario) as f:
        data = yaml.safe_load(f)

    domain_data = data.get("domain", {})
    config = GTBConfig.from_dict(domain_data)

    sim = data.get("simulation", {})
    n_epochs = args.epochs or sim.get("n_epochs", 10)
    steps = args.steps or sim.get("steps_per_epoch", 10)
    seed = args.seed or sim.get("seed", 42)

    # CLI seed overrides YAML; ensure env and runner use the same seed
    config.seed = seed

    agent_specs = data.get("agents", [{"policy": "honest", "count": 5}])

    runner = GTBScenarioRunner(
        config=config,
        agent_specs=agent_specs,
        n_epochs=n_epochs,
        steps_per_epoch=steps,
        seed=seed,
    )

    metrics = runner.run()
    run_dir = runner.export(args.output)

    logger.info("Scenario complete. Results in: %s", run_dir)
    logger.info(
        "Final epoch: welfare=%.3f gini=%.3f prod=%.2f",
        metrics[-1].welfare,
        metrics[-1].gini_coefficient,
        metrics[-1].total_production,
    )


if __name__ == "__main__":
    main()
