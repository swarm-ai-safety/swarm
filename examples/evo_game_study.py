#!/usr/bin/env python
"""Evolutionary game study runner.

Loads an evo-game scenario, runs the simulation, and renders
gamescape analysis comparing empirical population trajectory
with the replicator dynamics prediction.

Usage:
    python examples/evo_game_study.py                          # defaults
    python examples/evo_game_study.py --scenario path/to.yaml  # custom
    python examples/evo_game_study.py --dry-run                # no LLM
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the package is importable when run from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.scenarios.loader import build_orchestrator, load_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_SCENARIO = Path(__file__).resolve().parent.parent / "scenarios" / "evo_game_prisoners.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolutionary game study")
    parser.add_argument(
        "--scenario",
        type=Path,
        default=DEFAULT_SCENARIO,
        help="Path to scenario YAML (default: scenarios/evo_game_prisoners.yaml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without LLM calls (scripted agents only)",
    )
    args = parser.parse_args()

    # Load and build
    scenario = load_scenario(args.scenario)
    orchestrator = build_orchestrator(scenario)

    handler = orchestrator._evo_game_handler
    if handler is None:
        logger.error("Scenario does not have evo_game config enabled.")
        sys.exit(1)

    # Print initial state
    logger.info("=== Evolutionary Game Study ===")
    logger.info("Scenario: %s", scenario.scenario_id)
    logger.info("Game type: %s", handler.render_full_analysis().split("\n")[0])
    logger.info("Agents: %d", len(orchestrator._agents))
    logger.info(
        "Strategies: %s",
        {s.value: sum(1 for v in handler.strategies.values() if v == s)
         for s in set(handler.strategies.values())},
    )
    logger.info("")

    # Run simulation
    orchestrator.run()

    # Post-simulation analysis
    logger.info("\n=== Post-Simulation Analysis ===")

    # Empirical trajectory
    empirical = handler.get_population_trajectory()
    logger.info("Empirical cooperator trajectory: %s",
                [f"{x:.2f}" for x in empirical])

    # Replicator prediction
    predicted = handler.get_replicator_prediction(steps=len(empirical) * 200)
    # Sample at epoch boundaries
    step_size = max(1, len(predicted) // max(len(empirical), 1))
    predicted_sampled = [predicted[i * step_size] for i in range(len(empirical))
                         if i * step_size < len(predicted)]
    logger.info("Replicator prediction:           %s",
                [f"{x:.2f}" for x in predicted_sampled])

    # Full gamescape analysis
    logger.info("\n=== Gamescape Analysis ===")
    logger.info(handler.render_full_analysis())


if __name__ == "__main__":
    main()
