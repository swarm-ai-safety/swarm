"""CLI entry point for the distributional-agi-safety framework.

Usage:
    python -m src run scenarios/baseline.yaml
    python -m src run scenarios/baseline.yaml --seed 42 --epochs 20
    python -m src run scenarios/baseline.yaml --export-json results.json
    python -m src run scenarios/baseline.yaml --export-csv output/
    python -m src list
"""

import argparse
import os
import sys
from pathlib import Path


def cmd_run(args: argparse.Namespace) -> int:
    """Run a simulation from a YAML scenario file."""
    from swarm.scenarios.loader import build_orchestrator, load_scenario

    scenario_path = Path(args.scenario)
    if not scenario_path.exists():
        print(f"Error: scenario file not found: {scenario_path}", file=sys.stderr)
        return 1

    # Load scenario
    scenario = load_scenario(scenario_path)

    # In constrained environments, scenario-configured event log paths
    # may not be writable. Fall back to in-memory-only execution.
    if scenario.orchestrator_config.log_path is not None:
        log_parent = Path(scenario.orchestrator_config.log_path).parent
        if not os.access(log_parent, os.W_OK):
            scenario.orchestrator_config.log_path = None
            scenario.orchestrator_config.log_events = False

    # Apply CLI overrides
    if args.seed is not None:
        scenario.orchestrator_config.seed = args.seed
    if args.epochs is not None:
        scenario.orchestrator_config.n_epochs = args.epochs
    if args.steps is not None:
        scenario.orchestrator_config.steps_per_epoch = args.steps

    if not args.quiet:
        print("=" * 60)
        print("Distributional AGI Safety Sandbox")
        print("=" * 60)
        print(f"Scenario:    {scenario.scenario_id}")
        print(f"Description: {scenario.description}")
        print(f"Epochs:      {scenario.orchestrator_config.n_epochs}")
        print(f"Steps/epoch: {scenario.orchestrator_config.steps_per_epoch}")
        print(f"Seed:        {scenario.orchestrator_config.seed}")
        print()

    # Build and run
    orchestrator = build_orchestrator(scenario)

    # CLI override: prompt audit logging for LLM agents
    if args.prompt_audit is not None:
        for agent in orchestrator.get_all_agents():
            llm_config = getattr(agent, "llm_config", None)
            if llm_config is None:
                continue
            if not hasattr(llm_config, "prompt_audit_path"):
                continue
            llm_config.prompt_audit_path = args.prompt_audit
            llm_config.prompt_audit_include_system_prompt = bool(
                args.prompt_audit_include_system
            )
            llm_config.prompt_audit_hash_system_prompt = True

    if not args.quiet:
        print(f"Agents: {len(orchestrator.get_all_agents())}")
        for agent in orchestrator.get_all_agents():
            print(f"  - {agent.agent_id} ({agent.agent_type.value})")
        print()
        print("Running simulation...")
        print("-" * 60)

    metrics_history = orchestrator.run()

    # Print results
    if not args.quiet:
        print()
        print(
            f"{'Epoch':<6} {'Interactions':<13} {'Accepted':<10} {'Toxicity':<10} {'Welfare':<10}"
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

    # Summary
    if metrics_history:
        total_interactions = sum(m.total_interactions for m in metrics_history)
        total_accepted = sum(m.accepted_interactions for m in metrics_history)
        avg_toxicity = sum(m.toxicity_rate for m in metrics_history) / len(
            metrics_history
        )

        if not args.quiet:
            print(f"Total interactions: {total_interactions}")
            print(f"Accepted:          {total_accepted}")
            print(f"Avg toxicity:      {avg_toxicity:.4f}")
            print(f"Final welfare:     {metrics_history[-1].total_welfare:.2f}")
            print()

    # Export if requested
    if args.export_json or args.export_csv:
        from swarm.analysis.aggregation import EpochSnapshot, SimulationHistory

        history = SimulationHistory(
            simulation_id=scenario.scenario_id,
            n_epochs=scenario.orchestrator_config.n_epochs,
            steps_per_epoch=scenario.orchestrator_config.steps_per_epoch,
            n_agents=len(orchestrator.get_all_agents()),
            seed=scenario.orchestrator_config.seed,
        )

        for m in metrics_history:
            snapshot = EpochSnapshot(
                epoch=m.epoch,
                total_interactions=m.total_interactions,
                accepted_interactions=m.accepted_interactions,
                rejected_interactions=m.total_interactions - m.accepted_interactions,
                toxicity_rate=m.toxicity_rate,
                quality_gap=m.quality_gap,
                total_welfare=m.total_welfare,
                avg_payoff=m.avg_payoff,
                n_agents=len(orchestrator.get_all_agents()),
            )
            history.add_epoch_snapshot(snapshot)

        if args.export_json:
            from swarm.analysis.export import export_to_json

            path = export_to_json(history, args.export_json)
            print(f"Exported JSON: {path}")

        if args.export_csv:
            from swarm.analysis.export import export_to_csv

            paths = export_to_csv(history, args.export_csv, prefix=scenario.scenario_id)
            for kind, path in paths.items():
                print(f"Exported CSV ({kind}): {path}")

    # Check success criteria
    if scenario.success_criteria and not args.quiet:
        _check_criteria(scenario.success_criteria, metrics_history)

    return 0


def _check_criteria(criteria: dict, metrics_history: list) -> None:
    """Print success criteria results."""
    print("Success Criteria:")
    print("-" * 60)

    avg_toxicity = (
        sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
        if metrics_history
        else 0
    )

    all_passed = True

    if "min_epochs" in criteria:
        passed = len(metrics_history) >= criteria["min_epochs"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(f"  {tag} Epochs: {len(metrics_history)} >= {criteria['min_epochs']}")
        all_passed = all_passed and passed

    if "min_agents" in criteria:
        # Can't check without orchestrator reference; skip
        pass

    if "toxicity_threshold" in criteria:
        passed = avg_toxicity <= criteria["toxicity_threshold"]
        tag = "[PASS]" if passed else "[FAIL]"
        print(
            f"  {tag} Toxicity: {avg_toxicity:.4f} <= {criteria['toxicity_threshold']}"
        )
        all_passed = all_passed and passed

    print()
    print(f"Result: {'ALL CRITERIA PASSED' if all_passed else 'SOME CRITERIA FAILED'}")


def cmd_list(args: argparse.Namespace) -> int:
    """List available scenario files."""
    scenarios_dir = Path(args.dir)
    if not scenarios_dir.is_dir():
        print(f"Error: directory not found: {scenarios_dir}", file=sys.stderr)
        return 1

    files = sorted(scenarios_dir.glob("*.yaml"))
    if not files:
        print(f"No YAML scenarios found in {scenarios_dir}")
        return 0

    print(f"Available scenarios in {scenarios_dir}/:")
    for f in files:
        print(f"  {f}")

    return 0


def cmd_agentrxiv(args: argparse.Namespace) -> int:
    """Handle AgentRxiv subcommands."""
    subcmd = args.agentrxiv_cmd

    if subcmd == "start":
        return _agentrxiv_start(args)
    elif subcmd == "search":
        return _agentrxiv_search(args)
    elif subcmd == "submit":
        return _agentrxiv_submit(args)
    elif subcmd == "status":
        return _agentrxiv_status(args)
    else:
        print(f"Unknown agentrxiv subcommand: {subcmd}", file=sys.stderr)
        return 1


def _agentrxiv_start(args: argparse.Namespace) -> int:
    """Start the AgentRxiv server."""
    from swarm.research.agentrxiv_server import AgentRxivServer, AgentRxivServerError

    port = getattr(args, "port", 5000) or 5000
    uploads_dir = getattr(args, "uploads_dir", "./agentrxiv_papers") or "./agentrxiv_papers"

    print(f"Starting AgentRxiv server on port {port}...")

    try:
        server = AgentRxivServer(port=port, uploads_dir=uploads_dir)
        server.start(wait=True)
        print(f"AgentRxiv server running at {server.base_url}")
        print("Press Ctrl+C to stop...")

        try:
            import time
            while server.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            server.stop()
            print("Server stopped.")
        return 0

    except AgentRxivServerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _agentrxiv_search(args: argparse.Namespace) -> int:
    """Search papers on AgentRxiv."""
    from swarm.research.platforms import AgentRxivClient

    base_url = getattr(args, "url", None)
    client = AgentRxivClient(base_url=base_url)

    if not client.health_check():
        print("Error: AgentRxiv server not running", file=sys.stderr)
        print("Start it with: python -m swarm agentrxiv start", file=sys.stderr)
        return 1

    query = args.query
    limit = getattr(args, "limit", 5) or 5

    print(f"Searching AgentRxiv for: {query}")
    print("-" * 60)

    result = client.search(query, limit=limit)

    if not result.papers:
        print("No papers found.")
        return 0

    for i, paper in enumerate(result.papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   ID: {paper.paper_id}")
        if paper.abstract:
            abstract = paper.abstract[:200] + "..." if len(paper.abstract) > 200 else paper.abstract
            print(f"   Abstract: {abstract}")

    print(f"\nTotal results: {result.total_count}")
    return 0


def _agentrxiv_submit(args: argparse.Namespace) -> int:
    """Submit a paper to AgentRxiv."""
    from swarm.research.platforms import AgentRxivClient, Paper

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF file not found: {pdf_path}", file=sys.stderr)
        return 1

    base_url = getattr(args, "url", None)
    client = AgentRxivClient(base_url=base_url)

    if not client.health_check():
        print("Error: AgentRxiv server not running", file=sys.stderr)
        print("Start it with: python -m swarm agentrxiv start", file=sys.stderr)
        return 1

    title = getattr(args, "title", None) or pdf_path.stem.replace("_", " ")

    print(f"Uploading: {pdf_path}")
    print(f"Title: {title}")

    paper = Paper(title=title, abstract="")
    result = client.submit(paper, pdf_path=str(pdf_path))

    if result.success:
        print(f"Success! Paper ID: {result.paper_id}")
        print("Triggering index update...")
        client.trigger_update()
        print("Done.")
        return 0
    else:
        print(f"Error: {result.message}", file=sys.stderr)
        return 1


def _agentrxiv_status(args: argparse.Namespace) -> int:
    """Check AgentRxiv server status."""
    from swarm.research.platforms import AgentRxivClient

    base_url = getattr(args, "url", None)
    client = AgentRxivClient(base_url=base_url)

    print(f"Checking AgentRxiv at {client.base_url}...")

    if client.health_check():
        print("Status: RUNNING")
        return 0
    else:
        print("Status: NOT RUNNING")
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description="Distributional AGI Safety Sandbox",
    )
    subparsers = parser.add_subparsers(dest="command")

    # run
    run_parser = subparsers.add_parser("run", help="Run a simulation scenario")
    run_parser.add_argument("scenario", help="Path to YAML scenario file")
    run_parser.add_argument(
        "--seed", type=int, default=None, help="Override random seed"
    )
    run_parser.add_argument(
        "--epochs", type=int, default=None, help="Override number of epochs"
    )
    run_parser.add_argument(
        "--steps", type=int, default=None, help="Override steps per epoch"
    )
    run_parser.add_argument(
        "--export-json", metavar="PATH", help="Export results to JSON file"
    )
    run_parser.add_argument(
        "--export-csv", metavar="DIR", help="Export results to CSV directory"
    )
    run_parser.add_argument(
        "--prompt-audit",
        metavar="PATH",
        help="Write LLM prompt/response audit JSONL to PATH (hashes system prompt by default)",
    )
    run_parser.add_argument(
        "--prompt-audit-include-system",
        action="store_true",
        help="Include full system prompt text in the audit log (more sensitive)",
    )
    run_parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress progress output"
    )

    # list
    list_parser = subparsers.add_parser("list", help="List available scenarios")
    list_parser.add_argument(
        "--dir", default="scenarios", help="Scenarios directory (default: scenarios)"
    )

    # agentrxiv
    arxiv_parser = subparsers.add_parser("agentrxiv", help="Manage AgentRxiv local preprint server")
    arxiv_subparsers = arxiv_parser.add_subparsers(dest="agentrxiv_cmd")

    # agentrxiv start
    arxiv_start = arxiv_subparsers.add_parser("start", help="Start local AgentRxiv server")
    arxiv_start.add_argument("--port", type=int, default=5000, help="Port to run server on (default: 5000)")
    arxiv_start.add_argument("--uploads-dir", default="./agentrxiv_papers", help="Directory for PDF storage")

    # agentrxiv search
    arxiv_search = arxiv_subparsers.add_parser("search", help="Search papers on AgentRxiv")
    arxiv_search.add_argument("query", help="Search query")
    arxiv_search.add_argument("--limit", type=int, default=5, help="Maximum results (default: 5)")
    arxiv_search.add_argument("--url", help="AgentRxiv server URL (default: http://127.0.0.1:5000)")

    # agentrxiv submit
    arxiv_submit = arxiv_subparsers.add_parser("submit", help="Submit a PDF to AgentRxiv")
    arxiv_submit.add_argument("pdf", help="Path to PDF file")
    arxiv_submit.add_argument("--title", help="Paper title (default: derived from filename)")
    arxiv_submit.add_argument("--url", help="AgentRxiv server URL (default: http://127.0.0.1:5000)")

    # agentrxiv status
    arxiv_status = arxiv_subparsers.add_parser("status", help="Check AgentRxiv server status")
    arxiv_status.add_argument("--url", help="AgentRxiv server URL (default: http://127.0.0.1:5000)")

    args = parser.parse_args()

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list":
        return cmd_list(args)
    elif args.command == "agentrxiv":
        if args.agentrxiv_cmd:
            return cmd_agentrxiv(args)
        else:
            arxiv_parser.print_help()
            return 0
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
