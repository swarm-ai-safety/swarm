#!/usr/bin/env python
"""
Run an AWM live-mode scenario against real database-backed servers.

Usage:
    python examples/run_awm_live.py
    python examples/run_awm_live.py scenarios/awm_demo.yaml --live
    python examples/run_awm_live.py scenarios/awm_live.yaml
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_prerequisites() -> bool:
    """Verify AWM prerequisites are available."""
    ok = True

    try:
        import httpx  # noqa: F401
    except ImportError:
        print("ERROR: httpx is not installed.")
        print("  Fix: pip install swarm-safety[awm]")
        ok = False

    envs_path = Path("external/awm-envs")
    if not envs_path.exists():
        print(f"ERROR: AWM environments not found at {envs_path}")
        print("  Fix: bash scripts/download_awm_envs.sh")
        ok = False

    return ok


def main():
    parser = argparse.ArgumentParser(description="Run AWM live-mode scenario")
    parser.add_argument(
        "scenario",
        nargs="?",
        default="scenarios/awm_live.yaml",
        help="Path to scenario YAML (default: scenarios/awm_live.yaml)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Override live_mode=True on any AWM scenario",
    )
    args = parser.parse_args()

    # Prerequisites check
    if not check_prerequisites():
        return 1

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: Scenario file not found: {scenario_path}")
        return 1

    print("=" * 60)
    print("AWM Live Mode Runner")
    print("=" * 60)
    print()

    from swarm.scenarios import build_orchestrator, load_scenario

    # Load scenario
    print(f"Loading scenario: {scenario_path}")
    scenario = load_scenario(scenario_path)

    # Override live_mode if --live flag is set
    awm_config = scenario.orchestrator_config.awm_config
    if args.live and awm_config is not None:
        awm_config.live_mode = True
        print("  [--live] Overriding live_mode=True")

    if awm_config is None:
        print("Error: Scenario has no AWM configuration")
        return 1

    print(f"  ID: {scenario.scenario_id}")
    print(f"  Live mode: {awm_config.live_mode}")
    print(f"  Step mode: {awm_config.step_mode}")
    print(f"  Environment: {awm_config.environment_id}")
    print(f"  Base port: {awm_config.base_port}")
    print()

    # Build orchestrator
    print("Building orchestrator...")
    orchestrator = build_orchestrator(scenario)

    print(f"\nRegistered {len(orchestrator.get_all_agents())} agents:")
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        print(
            f"  - {agent.agent_id} ({agent.agent_type.value}): "
            f"resources={state.resources:.0f}"
        )
    print()

    # Run simulation with cleanup
    config = scenario.orchestrator_config
    print(
        f"Running simulation: {config.n_epochs} epochs x "
        f"{config.steps_per_epoch} steps"
    )
    print("-" * 60)

    handler = getattr(orchestrator, "_awm_handler", None)

    try:
        metrics_history = orchestrator.run()
    finally:
        # Ensure servers are shut down even on error
        if (
            handler is not None
            and hasattr(handler, "_server_manager")
            and handler._server_manager is not None
        ):
            print("\nShutting down AWM servers...")
            asyncio.run(handler._server_manager.shutdown())

    print()
    print("Epoch-by-Epoch Metrics:")
    print("-" * 60)
    print(
        f"{'Epoch':<6} {'Interactions':<13} {'Accepted':<10} "
        f"{'Toxicity':<10} {'Welfare':<10}"
    )
    print("-" * 60)

    for m in metrics_history:
        print(
            f"{m.epoch:<6} "
            f"{m.total_interactions:<13} "
            f"{m.accepted_interactions:<10} "
            f"{m.toxicity_rate:<10.4f} "
            f"{m.total_welfare:<10.2f}"
        )

    print("-" * 60)
    print()

    # AWM-specific metrics
    if handler is not None:
        episodes = handler.get_completed_episodes()
        print("AWM Episode Summary:")
        print("-" * 60)
        print(f"  Total episodes: {len(episodes)}")

        if episodes:
            passed = sum(1 for ep in episodes if ep.verified)
            failed = len(episodes) - passed
            print(f"  Verification passed: {passed}")
            print(f"  Verification failed: {failed}")
            print()

            # Per-agent breakdown
            agent_episodes: dict[str, list] = {}
            for ep in episodes:
                agent_episodes.setdefault(ep.agent_id, []).append(ep)

            print("  Per-agent breakdown:")
            for agent_id, eps in sorted(agent_episodes.items()):
                agent_passed = sum(1 for ep in eps if ep.verified)
                total_calls = sum(ep.total_calls for ep in eps)
                total_errors = sum(ep.error_count for ep in eps)
                print(
                    f"    {agent_id}: {len(eps)} episodes, "
                    f"{agent_passed}/{len(eps)} passed, "
                    f"{total_calls} tool calls, "
                    f"{total_errors} errors"
                )
        print()

    # Final agent states
    print("Final Agent States:")
    print("-" * 60)
    frozen = orchestrator.state.frozen_agents
    for agent in orchestrator.get_all_agents():
        state = orchestrator.state.get_agent(agent.agent_id)
        status = " [FROZEN]" if agent.agent_id in frozen else ""
        print(
            f"  {agent.agent_id}: "
            f"rep={state.reputation:.2f}, "
            f"res={state.resources:.2f}, "
            f"payoff={state.total_payoff:.2f}"
            f"{status}"
        )
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
