#!/usr/bin/env python
"""Ollama Model Size Comparison Study.

Compare how local Llama model size (3B vs 8B) affects agent behavior
in the SWARM safety sandbox across multiple seeds.

Prerequisites:
    ollama pull llama3.2
    ollama pull llama3.1:8b
    python -m pip install -e ".[llm,runtime]"

Usage:
    python examples/ollama_model_comparison.py
    python examples/ollama_model_comparison.py --seeds 3 --epochs 3 --steps 3
    python examples/ollama_model_comparison.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.agents.honest import HonestAgent
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMConfig, LLMProvider, PersonaType
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.governance.config import GovernanceConfig


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    """Scalar summary from one (model, seed) run."""

    model: str = ""
    seed: int = 0

    # Epoch-level aggregates
    total_interactions: int = 0
    accepted_interactions: int = 0
    total_posts: int = 0
    total_votes: int = 0
    toxicity_rate: float = 0.0
    quality_gap: float = 0.0
    avg_payoff: float = 0.0
    total_welfare: float = 0.0

    # Per-agent stats (LLM agents)
    llm_total_requests: int = 0
    llm_total_input_tokens: int = 0
    llm_total_output_tokens: int = 0
    llm_total_failures: int = 0

    # Per-agent payoff/reputation (mean across all agents)
    mean_reputation: float = 0.0
    mean_resources: float = 0.0
    mean_total_payoff: float = 0.0
    mean_interactions_initiated: float = 0.0

    # LLM agent-specific
    llm_mean_reputation: float = 0.0
    llm_mean_total_payoff: float = 0.0
    scripted_mean_reputation: float = 0.0
    scripted_mean_total_payoff: float = 0.0


# ---------------------------------------------------------------------------
# Orchestrator factory
# ---------------------------------------------------------------------------
def create_orchestrator(
    model: str,
    seed: int,
    n_epochs: int,
    steps_per_epoch: int,
    dry_run: bool = False,
) -> Orchestrator:
    """Build a 4-agent orchestrator: 2 LLM (open + strategic) + 2 scripted."""
    config = OrchestratorConfig(
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        seed=seed,
        payoff_config=PayoffConfig(
            s_plus=2.0,
            s_minus=1.0,
            h=2.0,
            theta=0.5,
        ),
        governance_config=GovernanceConfig(
            transaction_tax_rate=0.05,
            audit_enabled=True,
            audit_probability=0.1,
        ),
    )

    orchestrator = Orchestrator(config)

    # Two LLM agents with different personas
    for i, persona in enumerate([PersonaType.OPEN, PersonaType.STRATEGIC]):
        llm_config = LLMConfig(
            provider=LLMProvider.OLLAMA,
            model=model,
            persona=persona,
            temperature=0.7,
            max_tokens=512,
            cost_tracking=True,
            timeout=120.0,
        )
        agent = LLMAgent(
            agent_id=f"llm_{persona.value}_{i + 1}",
            llm_config=llm_config,
        )

        if dry_run:

            async def mock_call(*args, **kwargs):
                return ('{"action_type": "NOOP", "reasoning": "Dry run"}', 50, 20)

            agent._call_llm_async = mock_call

        orchestrator.register_agent(agent)

    # Two scripted honest agents
    for i in range(2):
        agent = HonestAgent(agent_id=f"honest_{i + 1}")
        orchestrator.register_agent(agent)

    return orchestrator


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
async def run_single(
    model: str,
    seed: int,
    n_epochs: int,
    steps_per_epoch: int,
    dry_run: bool = False,
) -> SeedResult:
    """Run one (model, seed) configuration and return scalar metrics."""
    orchestrator = create_orchestrator(
        model=model,
        seed=seed,
        n_epochs=n_epochs,
        steps_per_epoch=steps_per_epoch,
        dry_run=dry_run,
    )

    epoch_metrics = await orchestrator.run_async()

    # Aggregate epoch metrics
    total_interactions = sum(m.total_interactions for m in epoch_metrics)
    accepted_interactions = sum(m.accepted_interactions for m in epoch_metrics)
    total_posts = sum(m.total_posts for m in epoch_metrics)
    total_votes = sum(m.total_votes for m in epoch_metrics)
    total_welfare = sum(m.total_welfare for m in epoch_metrics)

    # Use final epoch for rates
    final = epoch_metrics[-1] if epoch_metrics else None
    toxicity_rate = final.toxicity_rate if final else 0.0
    quality_gap = final.quality_gap if final else 0.0
    avg_payoff = final.avg_payoff if final else 0.0

    # LLM usage
    usage_stats = orchestrator.get_llm_usage_stats()
    llm_requests = sum(s.get("total_requests", 0) for s in usage_stats.values())
    llm_input = sum(s.get("total_input_tokens", 0) for s in usage_stats.values())
    llm_output = sum(s.get("total_output_tokens", 0) for s in usage_stats.values())
    llm_failures = sum(s.get("total_failures", 0) for s in usage_stats.values())

    # Per-agent states
    agents = orchestrator.state.agents
    n_agents = len(agents) or 1

    mean_rep = sum(a.reputation for a in agents.values()) / n_agents
    mean_res = sum(a.resources for a in agents.values()) / n_agents
    mean_payoff = sum(a.total_payoff for a in agents.values()) / n_agents
    mean_init = sum(a.interactions_initiated for a in agents.values()) / n_agents

    # Split by type
    llm_agents = {k: v for k, v in agents.items() if k.startswith("llm_")}
    scripted_agents = {k: v for k, v in agents.items() if k.startswith("honest_")}

    n_llm = len(llm_agents) or 1
    n_scripted = len(scripted_agents) or 1

    llm_mean_rep = sum(a.reputation for a in llm_agents.values()) / n_llm
    llm_mean_pay = sum(a.total_payoff for a in llm_agents.values()) / n_llm
    scripted_mean_rep = sum(a.reputation for a in scripted_agents.values()) / n_scripted
    scripted_mean_pay = (
        sum(a.total_payoff for a in scripted_agents.values()) / n_scripted
    )

    return SeedResult(
        model=model,
        seed=seed,
        total_interactions=total_interactions,
        accepted_interactions=accepted_interactions,
        total_posts=total_posts,
        total_votes=total_votes,
        toxicity_rate=toxicity_rate,
        quality_gap=quality_gap,
        avg_payoff=avg_payoff,
        total_welfare=total_welfare,
        llm_total_requests=llm_requests,
        llm_total_input_tokens=llm_input,
        llm_total_output_tokens=llm_output,
        llm_total_failures=llm_failures,
        mean_reputation=mean_rep,
        mean_resources=mean_res,
        mean_total_payoff=mean_payoff,
        mean_interactions_initiated=mean_init,
        llm_mean_reputation=llm_mean_rep,
        llm_mean_total_payoff=llm_mean_pay,
        scripted_mean_reputation=scripted_mean_rep,
        scripted_mean_total_payoff=scripted_mean_pay,
    )


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------
def write_sweep_csv(results: List[SeedResult], path: Path) -> None:
    """Write one-row-per-(model, seed) summary CSV."""
    fieldnames = [f.name for f in fields(SeedResult)]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {}
            for fname in fieldnames:
                val = getattr(r, fname)
                if isinstance(val, float):
                    row[fname] = f"{val:.6f}"
                else:
                    row[fname] = val
            writer.writerow(row)


def compute_summary(
    results: List[SeedResult], model: str
) -> Dict[str, Dict[str, float]]:
    """Compute mean ± std for each numeric field, filtered to one model."""
    import numpy as np

    subset = [r for r in results if r.model == model]
    if not subset:
        return {}

    summary: Dict[str, Dict[str, float]] = {}
    for f in fields(SeedResult):
        if f.name in ("model", "seed"):
            continue
        vals = [getattr(r, f.name) for r in subset]
        arr = np.array(vals, dtype=float)
        summary[f.name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main(args: argparse.Namespace) -> int:
    models = ["llama3.2", "llama3.1:8b"]
    seeds = list(range(42, 42 + args.seeds))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or Path(f"runs/{timestamp}_ollama_model_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Ollama Model Size Comparison Study")
    print(f"  Models: {models}")
    print(f"  Seeds: {seeds}")
    print(f"  Epochs: {args.epochs}, Steps/epoch: {args.steps}")
    print("  Agents: 2 LLM + 2 scripted per run")
    print(f"  Dry run: {args.dry_run}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    # Save config for reproducibility
    with open(out_dir / "sweep_config.json", "w") as f:
        json.dump(
            {
                "models": models,
                "seeds": seeds,
                "epochs": args.epochs,
                "steps_per_epoch": args.steps,
                "timestamp": timestamp,
                "dry_run": args.dry_run,
            },
            f,
            indent=2,
        )

    all_results: List[SeedResult] = []
    total_runs = len(models) * len(seeds)
    run_idx = 0

    for model in models:
        print(f"\n{'─' * 70}")
        print(f"  Model: {model}")
        print(f"{'─' * 70}")

        for seed in seeds:
            run_idx += 1
            print(f"\n  [{run_idx}/{total_runs}] {model} seed={seed}...", flush=True)

            result = await run_single(
                model=model,
                seed=seed,
                n_epochs=args.epochs,
                steps_per_epoch=args.steps,
                dry_run=args.dry_run,
            )
            all_results.append(result)

            print(
                f"    interactions={result.total_interactions}, "
                f"accepted={result.accepted_interactions}, "
                f"welfare={result.total_welfare:.2f}, "
                f"toxicity={result.toxicity_rate:.3f}"
            )
            print(
                f"    llm_requests={result.llm_total_requests}, "
                f"llm_failures={result.llm_total_failures}, "
                f"llm_rep={result.llm_mean_reputation:.2f}, "
                f"scripted_rep={result.scripted_mean_reputation:.2f}"
            )

    # Write sweep CSV
    sweep_csv = out_dir / "sweep_results.csv"
    write_sweep_csv(all_results, sweep_csv)
    print(f"\n  -> {sweep_csv}")

    # Compute and write summary JSON
    summary: Dict[str, Any] = {}
    for model in models:
        summary[model] = compute_summary(all_results, model)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {out_dir / 'summary.json'}")

    # Print comparison table
    import numpy as np

    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    key_metrics = [
        "total_interactions",
        "accepted_interactions",
        "total_welfare",
        "toxicity_rate",
        "quality_gap",
        "total_posts",
        "total_votes",
        "llm_total_requests",
        "llm_total_failures",
        "llm_total_input_tokens",
        "llm_total_output_tokens",
        "mean_reputation",
        "mean_total_payoff",
        "mean_interactions_initiated",
        "llm_mean_reputation",
        "llm_mean_total_payoff",
        "scripted_mean_reputation",
        "scripted_mean_total_payoff",
    ]

    header = f"{'Metric':>35s}  {'llama3.2 (3B)':>20s}  {'llama3.1:8b':>20s}"
    print(header)
    print("─" * len(header))

    for metric in key_metrics:
        vals = {}
        for model in models:
            subset = [r for r in all_results if r.model == model]
            arr = np.array([getattr(r, metric) for r in subset], dtype=float)
            vals[model] = f"{arr.mean():.2f} ± {arr.std():.2f}"

        print(
            f"{metric:>35s}  {vals['llama3.2']:>20s}  {vals['llama3.1:8b']:>20s}"
        )

    print(f"\nAll outputs in: {out_dir}/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ollama model size comparison study"
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds (starting from 42)"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="Mock LLM calls for testing"
    )
    args = parser.parse_args()

    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
