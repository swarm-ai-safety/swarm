#!/usr/bin/env python3
"""Run a Claude Code bridge scenario from a SWARM YAML file."""

from __future__ import annotations

import argparse
import os
import re
import sys
from urllib.parse import urlparse
from pathlib import Path
from typing import Dict, List

from swarm.bridges.claude_code import BridgeConfig, ClaudeCodeBridge, ClientConfig
from swarm.bridges.claude_code.agent import ClaudeCodeAgent
from swarm.core.orchestrator import Orchestrator
from swarm.scenarios.loader import load_scenario


def _sanitize_agent_id(raw: str) -> str:
    """Normalize agent IDs to controller-safe characters."""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw.strip())
    cleaned = cleaned.strip("_")
    return cleaned or "agent"


def _is_loopback_url(url: str) -> bool:
    """Return True if the URL host is loopback."""
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        # Fallback for scheme-less values
        host = url.split("://")[-1].split("/")[0].split(":")[0]
    return host in ("localhost", "127.0.0.1", "::1")


def _build_agents(
    agent_specs: List[Dict],
    bridge: ClaudeCodeBridge,
) -> List[ClaudeCodeAgent]:
    agents: List[ClaudeCodeAgent] = []
    counters: Dict[str, int] = {}

    for spec in agent_specs:
        count = int(spec.get("count", 1))
        base_name = spec.get("name") or spec.get("type") or "agent"
        base_name = _sanitize_agent_id(base_name)
        config = spec.get("config", {})

        for _ in range(count):
            counters[base_name] = counters.get(base_name, 0) + 1
            agent_id = (
                base_name if count == 1 else f"{base_name}_{counters[base_name]}"
            )
            agent = ClaudeCodeAgent(
                agent_id=agent_id,
                bridge=bridge,
                config=config,
                name=agent_id,
            )
            agents.append(agent)

    return agents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a SWARM scenario using the Claude Code bridge."
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/claude_code_demo.yaml",
        help="Path to scenario YAML",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:3100",
        help="Claude Code controller base URL",
    )
    parser.add_argument(
        "--api-prefix",
        default="/api",
        help="API prefix for controller endpoints",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Bearer token for controller (prefer SWARM_BRIDGE_API_KEY)",
    )
    parser.add_argument(
        "--team-name",
        default="swarm",
        help="Controller session team name",
    )
    parser.add_argument(
        "--cwd",
        default=None,
        help="Working directory for controller agents",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Auto-approve plan/permission requests (loopback only)",
    )
    parser.add_argument(
        "--no-auto-approve",
        action="store_true",
        help="Deprecated; auto-approval is off by default",
    )

    args = parser.parse_args()

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: scenario file not found: {scenario_path}", file=sys.stderr)
        return 1

    scenario = load_scenario(scenario_path)

    # Avoid logging if path isn't writable
    if scenario.orchestrator_config.log_path is not None:
        log_parent = Path(scenario.orchestrator_config.log_path).parent
        if not os.access(log_parent, os.W_OK):
            scenario.orchestrator_config.log_path = None
            scenario.orchestrator_config.log_events = False

    orchestrator = Orchestrator(config=scenario.orchestrator_config)
    orchestrator.state.rate_limits = scenario.rate_limits

    cli_api_key = args.api_key
    if cli_api_key:
        print(
            "Warning: --api-key may be exposed via shell history or process listings. "
            "Prefer SWARM_BRIDGE_API_KEY.",
            file=sys.stderr,
        )
    api_key = cli_api_key or os.environ.get("SWARM_BRIDGE_API_KEY")

    auto_approve = args.auto_approve and not args.no_auto_approve
    if auto_approve and not _is_loopback_url(args.base_url):
        print(
            "Warning: auto-approve is disabled for non-loopback controller URLs.",
            file=sys.stderr,
        )
        auto_approve = False

    client_config = ClientConfig(
        base_url=args.base_url,
        api_prefix=args.api_prefix,
        api_key=api_key,
    )
    bridge_config = BridgeConfig(
        client_config=client_config,
        governance_config=scenario.orchestrator_config.governance_config,
        auto_respond_governance=auto_approve,
    )
    bridge = ClaudeCodeBridge(bridge_config, event_log=orchestrator.event_log)

    # Ensure controller session is ready
    bridge.init_session(team_name=args.team_name, cwd=args.cwd)

    # Register Claude Code agents from the scenario
    agents = _build_agents(scenario.agent_specs, bridge)
    for agent in agents:
        orchestrator.register_agent(agent)

    print("=" * 60)
    print("SWARM Claude Code Bridge Runner")
    print("=" * 60)
    print(f"Scenario:    {scenario.scenario_id}")
    print(f"Description: {scenario.description}")
    print(f"Agents:      {len(agents)}")
    print(f"Controller:  {args.base_url}{args.api_prefix}")
    print()
    print("Running simulation...")
    print("-" * 60)

    metrics_history = orchestrator.run()

    print()
    print(f"{'Epoch':<6} {'Interactions':<13} {'Accepted':<10} {'Toxicity':<10} {'Welfare':<10}")
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

    bridge.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
