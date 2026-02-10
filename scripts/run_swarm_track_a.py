#!/usr/bin/env python3
"""Run SWARM Track A pipeline and optionally publish to AgentRxiv."""

from __future__ import annotations

import argparse
import sys

from swarm.research.swarm_papers.track_a import (
    TrackAConfig,
    TrackARunner,
    default_conditions,
)

try:  # optional
    from swarm.agents.llm_config import LLMConfig, LLMProvider
except Exception:  # pragma: no cover - optional
    LLMConfig = None  # type: ignore
    LLMProvider = None  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SWARM Track A benchmark")
    parser.add_argument("--tasks", type=int, default=200, help="Number of tasks")
    parser.add_argument("--difficulty", type=float, default=0.5, help="Task difficulty 0-1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        default="runs/swarm_collate",
        help="Output directory for run artifacts",
    )
    parser.add_argument(
        "--conditions",
        default="",
        help="Comma-separated condition names (default: all)",
    )
    parser.add_argument("--no-agentrxiv", action="store_true", help="Disable AgentRxiv")
    parser.add_argument("--agentrxiv-url", default=None, help="AgentRxiv base URL")
    parser.add_argument("--pdf", action="store_true", help="Render PDF via pdflatex")
    parser.add_argument("--publish", action="store_true", help="Publish to AgentRxiv")
    parser.add_argument(
        "--query",
        default="SWARM Track A verifiable reasoning",
        help="Query string for related work",
    )

    # LLM options
    parser.add_argument("--llm", action="store_true", help="Enable LLM solvers")
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "ollama"],
        default="anthropic",
    )
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conditions = default_conditions()
    if args.conditions:
        selected = {name.strip() for name in args.conditions.split(",") if name.strip()}
        conditions = [cond for cond in conditions if cond.name in selected]
        missing = selected - {cond.name for cond in conditions}
        if missing:
            print(f"Unknown conditions: {sorted(missing)}", file=sys.stderr)
            return 1

    llm_config = None
    if args.llm:
        if LLMConfig is None or LLMProvider is None:
            print("LLM dependencies missing. Install swarm-safety[llm].", file=sys.stderr)
            return 1
        llm_config = LLMConfig(
            provider=LLMProvider(args.provider),
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )

    config = TrackAConfig(
        n_tasks=args.tasks,
        seed=args.seed,
        difficulty=args.difficulty,
        output_dir=args.output_dir,
        conditions=conditions,
        enable_agentrxiv=not args.no_agentrxiv,
        agentrxiv_url=args.agentrxiv_url,
        enable_pdf=args.pdf,
        publish_to_agentrxiv=args.publish,
        query=args.query,
        llm_enabled=args.llm,
        llm_config=llm_config,
    )

    runner = TrackARunner(config)
    summary = runner.run()
    print(f"Run completed: {summary.run_id}")
    print(f"Outputs: {runner.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
