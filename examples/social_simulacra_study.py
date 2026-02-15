#!/usr/bin/env python3
"""
Social Simulacra Study Runner — end-to-end pipeline for the simulacra scenario.

Executes the 5-phase pipeline defined in scenarios/social_simulacra.yaml:
  1. Load scenario — parse YAML, build CommunityConfig + PersonaSeed list
  2. Expand personas — PersonaExpander.expand(n=30) from 8 seeds
  3. Generate threads — ThreadGenerator.generate_threads(n=10)
  4. WhatIf injections — inject counterfactual personas into threads
  5. Multiverse analysis — MultiverseRunner across 12 universes / 6 temps

No LLM calls required — all generation uses deterministic synthetic fallbacks.

Usage:
    python examples/social_simulacra_study.py
    python examples/social_simulacra_study.py --scenario scenarios/social_simulacra.yaml
    python examples/social_simulacra_study.py --seed 42 --dry-run
    python examples/social_simulacra_study.py --output runs/my_study/
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from swarm.bridges.concordia.adapter import ConcordiaAdapter
from swarm.bridges.concordia.config import ConcordiaConfig
from swarm.bridges.concordia.multiverse import (
    MultiverseConfig,
    MultiverseResult,
    MultiverseRunner,
)
from swarm.bridges.concordia.simulacra import (
    CommunityConfig,
    ExpandedPersona,
    PersonaExpander,
    PersonaSeed,
    Thread,
    ThreadGenerator,
    WhatIfInjector,
    thread_to_narrative_samples,
)
from swarm.metrics.soft_metrics import SoftMetrics

# ── Scenario loading ─────────────────────────────────────────────────────


def load_scenario(path: Path) -> Dict[str, Any]:
    """Load and return the parsed YAML scenario."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_community(scenario: Dict[str, Any]) -> CommunityConfig:
    """Build CommunityConfig from scenario YAML."""
    c = scenario.get("community", {})
    return CommunityConfig(
        name=c.get("name", "Unnamed Community"),
        description=c.get("description", ""),
        goal=c.get("goal", ""),
        rules=c.get("rules", []),
        norms=c.get("norms", []),
    )


def build_seeds(scenario: Dict[str, Any]) -> List[PersonaSeed]:
    """Build PersonaSeed list from scenario YAML."""
    return [
        PersonaSeed(
            name=s["name"],
            description=s["description"],
            community_role=s.get("community_role", ""),
        )
        for s in scenario.get("seed_personas", [])
    ]


# ── Phase 1: Persona Expansion ──────────────────────────────────────────


def expand_personas(
    community: CommunityConfig,
    seeds: List[PersonaSeed],
    n: int,
    seed: int,
) -> List[ExpandedPersona]:
    """Expand seed personas into a larger population."""
    import random

    expander = PersonaExpander(community, seeds)
    rng = random.Random(seed)
    return expander.expand(n=n, rng=rng)


# ── Phase 2: Thread Generation ───────────────────────────────────────────


def generate_threads(
    community: CommunityConfig,
    personas: List[ExpandedPersona],
    scenario: Dict[str, Any],
    seed: int,
) -> List[Thread]:
    """Generate threads using scenario parameters."""
    import random

    tg_cfg = scenario.get("thread_generation", {})
    generator = ThreadGenerator(
        community=community,
        personas=personas,
        base_temperature=tg_cfg.get("base_temperature", 0.7),
        reply_mean=tg_cfg.get("reply_mean", 0.65),
        reply_stdev=tg_cfg.get("reply_stdev", 0.15),
        max_replies=tg_cfg.get("max_replies", 8),
        new_persona_prob=tg_cfg.get("new_persona_prob", 0.5),
    )
    rng = random.Random(seed)
    n_threads = tg_cfg.get("n_threads", 10)
    return generator.generate_threads(n_threads, rng=rng)


# ── Phase 3: WhatIf Injections ───────────────────────────────────────────


def run_whatif_injections(
    community: CommunityConfig,
    threads: List[Thread],
    scenario: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    """Run WhatIf counterfactual injections and compute pre/post metrics.

    Returns a list of dicts with injection name, baseline metrics, and
    post-injection metrics.
    """
    import random

    injections = scenario.get("whatif_injections", [])
    if not injections or not threads:
        return []

    rng = random.Random(seed)
    injector = WhatIfInjector(community)
    metrics = SoftMetrics()
    adapter = ConcordiaAdapter(config=ConcordiaConfig())

    results: List[Dict[str, Any]] = []

    for inj in injections:
        persona_name = inj["persona_name"]
        persona_desc = inj["persona_description"]

        # Pick a random thread for injection
        target_thread = rng.choice(threads)

        # Compute baseline metrics for the target thread
        baseline_interactions = _score_thread(adapter, target_thread)
        baseline_tox = metrics.toxicity_rate(baseline_interactions) if baseline_interactions else 0.0
        baseline_avg_p = (
            sum(i.p for i in baseline_interactions) / len(baseline_interactions)
            if baseline_interactions
            else 0.0
        )

        # Inject counterfactual persona
        injected_thread, injected_persona = injector.inject(
            target_thread,
            persona_desc,
            persona_name=persona_name,
            rng=rng,
        )

        # Compute post-injection metrics
        post_adapter = ConcordiaAdapter(config=ConcordiaConfig())
        post_interactions = _score_thread(post_adapter, injected_thread)
        post_tox = metrics.toxicity_rate(post_interactions) if post_interactions else 0.0
        post_avg_p = (
            sum(i.p for i in post_interactions) / len(post_interactions)
            if post_interactions
            else 0.0
        )

        results.append({
            "persona_name": persona_name,
            "persona_description": persona_desc,
            "target_thread_id": target_thread.thread_id,
            "baseline_toxicity": baseline_tox,
            "baseline_avg_p": baseline_avg_p,
            "baseline_n_posts": len(target_thread.posts),
            "post_toxicity": post_tox,
            "post_avg_p": post_avg_p,
            "post_n_posts": len(injected_thread.posts),
            "toxicity_delta": post_tox - baseline_tox,
        })

    return results


def _score_thread(adapter: ConcordiaAdapter, thread: Thread) -> list:
    """Score a thread through the adapter, returning interactions."""
    from swarm.models.interaction import SoftInteraction

    all_interactions: List[SoftInteraction] = []
    samples = thread_to_narrative_samples(thread)
    for agent_ids, narrative in samples:
        interactions = adapter.process_narrative(
            agent_ids=agent_ids,
            narrative_text=narrative,
            step=0,
        )
        all_interactions.extend(interactions)
    return all_interactions


# ── Phase 4: Multiverse Analysis ─────────────────────────────────────────


def run_multiverse(
    community: CommunityConfig,
    personas: List[ExpandedPersona],
    scenario: Dict[str, Any],
    seed: int,
) -> MultiverseResult:
    """Run multiverse analysis across universes and temperatures."""
    mv_cfg = scenario.get("multiverse", {})
    config = MultiverseConfig(
        n_universes=mv_cfg.get("n_universes", 12),
        temperatures=mv_cfg.get("temperatures", [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
        threads_per_universe=mv_cfg.get("threads_per_universe", 5),
        base_seed=seed,
    )
    runner = MultiverseRunner(community, personas, config)
    return runner.run()


# ── CSV Export Helpers ────────────────────────────────────────────────────


def export_personas_csv(personas: List[ExpandedPersona], path: Path) -> None:
    """Export expanded persona roster to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["name", "community_role", "generated", "seed_origin", "description"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in personas:
            writer.writerow({
                "name": p.name,
                "community_role": p.community_role,
                "generated": p.generated,
                "seed_origin": p.seed_origin,
                "description": p.description[:200],
            })


def export_threads_csv(threads: List[Thread], path: Path) -> None:
    """Export thread summaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["thread_id", "author", "reply_count", "participant_count", "participants"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for t in threads:
            root = t.root
            participants = t.participants
            writer.writerow({
                "thread_id": t.thread_id[:12],
                "author": root.author.name if root else "",
                "reply_count": len(t.replies),
                "participant_count": len(participants),
                "participants": "; ".join(p.name for p in participants),
            })


def export_whatif_csv(results: List[Dict[str, Any]], path: Path) -> None:
    """Export WhatIf comparison metrics to CSV."""
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "persona_name", "baseline_toxicity", "post_toxicity", "toxicity_delta",
        "baseline_avg_p", "post_avg_p", "baseline_n_posts", "post_n_posts",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})


def export_multiverse_csv(mv: MultiverseResult, path: Path) -> None:
    """Export per-universe metrics to CSV."""
    if not mv.universes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "universe_id", "temperature", "seed", "toxicity_rate", "quality_gap",
        "avg_p", "p_variance", "n_interactions", "n_threads",
        "avg_replies_per_thread", "n_unique_participants",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for u in mv.universes:
            writer.writerow({
                "universe_id": u.universe_id,
                "temperature": u.temperature,
                "seed": u.seed,
                "toxicity_rate": u.toxicity_rate,
                "quality_gap": u.quality_gap,
                "avg_p": u.avg_p,
                "p_variance": u.p_variance,
                "n_interactions": u.n_interactions,
                "n_threads": u.n_threads,
                "avg_replies_per_thread": u.avg_replies_per_thread,
                "n_unique_participants": u.n_unique_participants,
            })


def export_multiverse_summary_csv(mv: MultiverseResult, path: Path) -> None:
    """Export cross-universe summary stats to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = mv.to_dict()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)


def export_history(
    scenario: Dict[str, Any],
    personas: List[ExpandedPersona],
    threads: List[Thread],
    whatif_results: List[Dict[str, Any]],
    mv: MultiverseResult,
    seed: int,
    path: Path,
) -> None:
    """Export full reproducibility dump as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "scenario_id": scenario.get("scenario_id", "social_simulacra"),
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_personas": len(personas),
        "n_threads": len(threads),
        "whatif_results": whatif_results,
        "multiverse_summary": mv.to_dict(),
        "persona_names": [p.name for p in personas],
        "thread_summaries": [
            {
                "thread_id": t.thread_id[:12],
                "author": t.root.author.name if t.root else "",
                "n_posts": len(t.posts),
                "n_replies": len(t.replies),
            }
            for t in threads
        ],
        "universe_summaries": [
            {
                "universe_id": u.universe_id,
                "temperature": u.temperature,
                "toxicity_rate": u.toxicity_rate,
                "avg_p": u.avg_p,
            }
            for u in mv.universes
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── Plotting ─────────────────────────────────────────────────────────────


def generate_plots(
    personas: List[ExpandedPersona],
    threads: List[Thread],
    whatif_results: List[Dict[str, Any]],
    mv: MultiverseResult,
    plot_dir: Path,
) -> None:
    """Generate all 5 study plots. Skips gracefully if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Persona roles bar chart
    role_counts = Counter(p.community_role for p in personas)
    roles = list(role_counts.keys())
    counts = list(role_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(roles, counts, color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title(f"Community Roles ({len(personas)} personas)")
    fig.tight_layout()
    fig.savefig(plot_dir / "persona_roles.png", dpi=150)
    plt.close(fig)

    # 2. Thread depth histogram
    reply_counts = [len(t.replies) for t in threads]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(reply_counts, bins=range(0, max(reply_counts, default=1) + 2),
            color="salmon", edgecolor="black", align="left")
    ax.set_xlabel("Reply Count")
    ax.set_ylabel("Number of Threads")
    ax.set_title(f"Thread Depth Distribution ({len(threads)} threads)")
    fig.tight_layout()
    fig.savefig(plot_dir / "thread_depth.png", dpi=150)
    plt.close(fig)

    # 3. WhatIf toxicity comparison (grouped bar)
    if whatif_results:
        names = [r["persona_name"] for r in whatif_results]
        baseline_tox = [r["baseline_toxicity"] for r in whatif_results]
        post_tox = [r["post_toxicity"] for r in whatif_results]
        x = np.arange(len(names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x - width / 2, baseline_tox, width, label="Baseline", color="steelblue")
        ax.bar(x + width / 2, post_tox, width, label="Post-Injection", color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Toxicity Rate")
        ax.set_title("WhatIf Injection: Toxicity Impact")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "whatif_toxicity.png", dpi=150)
        plt.close(fig)

    # 4. Multiverse: avg_p vs temperature scatter
    if mv.universes:
        temps = [u.temperature for u in mv.universes]
        avg_ps = [u.avg_p for u in mv.universes]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(temps, avg_ps, s=60, alpha=0.7, color="steelblue")
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Average p")
        ax.set_title(
            f"Multiverse: avg_p vs Temperature "
            f"(r={mv.temperature_correlation:.3f})"
        )
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / "multiverse_p_by_temp.png", dpi=150)
        plt.close(fig)

    # 5. Multiverse bias-variance decomposition (stacked bar)
    if mv.universes:
        fig, ax = plt.subplots(figsize=(6, 5))
        components = ["Bias²", "Variance"]
        values = [mv.bias_squared, mv.variance]
        colors = ["#e74c3c", "#3498db"]
        bottom = 0.0
        for comp, val, col in zip(components, values, colors, strict=True):
            ax.bar("Total Error", val, bottom=bottom, label=f"{comp} = {val:.4f}",
                    color=col, width=0.4)
            bottom += val
        ax.set_ylabel("Error")
        ax.set_title(f"Bias-Variance Decomposition (total = {mv.total_error:.4f})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "multiverse_bias_variance.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {plot_dir}/")


# ── Summary Printing ─────────────────────────────────────────────────────


def print_summary(
    personas: List[ExpandedPersona],
    threads: List[Thread],
    whatif_results: List[Dict[str, Any]],
    mv: MultiverseResult,
) -> None:
    """Print a formatted summary of all phases."""
    print()
    print("=" * 70)
    print("  Results Summary")
    print("=" * 70)

    # Personas
    n_seed = sum(1 for p in personas if not p.generated)
    n_gen = sum(1 for p in personas if p.generated)
    print(f"\n  Personas: {len(personas)} total ({n_seed} seed + {n_gen} generated)")
    role_counts = Counter(p.community_role for p in personas)
    for role, count in role_counts.most_common(5):
        print(f"    {role:<25} {count}")

    # Threads
    reply_counts = [len(t.replies) for t in threads]
    avg_replies = sum(reply_counts) / len(reply_counts) if reply_counts else 0
    print(f"\n  Threads: {len(threads)} total, avg {avg_replies:.1f} replies/thread")

    # WhatIf
    if whatif_results:
        print("\n  WhatIf Injections:")
        print(f"    {'Persona':<20} {'Baseline Tox':>12} {'Post Tox':>12} {'Delta':>10}")
        print(f"    {'-' * 56}")
        for r in whatif_results:
            print(
                f"    {r['persona_name']:<20} "
                f"{r['baseline_toxicity']:>12.4f} "
                f"{r['post_toxicity']:>12.4f} "
                f"{r['toxicity_delta']:>+10.4f}"
            )

    # Multiverse
    if mv.universes:
        print(f"\n  Multiverse ({len(mv.universes)} universes):")
        print(f"    Toxicity:  mean={mv.toxicity_mean:.4f}  std={mv.toxicity_std:.4f}")
        print(f"    Avg p:     mean={mv.p_mean:.4f}  std={mv.p_std:.4f}")
        print(f"    Bias²={mv.bias_squared:.4f}  Variance={mv.variance:.4f}  Total={mv.total_error:.4f}")
        print(f"    Temp-toxicity correlation: {mv.temperature_correlation:.4f}")

    print()


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Social Simulacra Study — end-to-end pipeline runner"
    )
    parser.add_argument(
        "--scenario", type=Path,
        default=Path("scenarios/social_simulacra.yaml"),
        help="Path to scenario YAML (default: scenarios/social_simulacra.yaml)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output directory (default: runs/<timestamp>_social_simulacra/)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate scenario parsing without running the full pipeline",
    )
    args = parser.parse_args()

    # ── Load scenario ────────────────────────────────────────────────
    print("=" * 70)
    print("  Social Simulacra Study")
    print("=" * 70)
    print()

    if not args.scenario.exists():
        print(f"ERROR: Scenario file not found: {args.scenario}")
        return 1

    scenario = load_scenario(args.scenario)
    community = build_community(scenario)
    seeds = build_seeds(scenario)

    pe_cfg = scenario.get("persona_expansion", {})
    n_expand = pe_cfg.get("n_expanded", 30)
    tg_cfg = scenario.get("thread_generation", {})
    n_threads = tg_cfg.get("n_threads", 10)
    mv_cfg = scenario.get("multiverse", {})
    n_universes = mv_cfg.get("n_universes", 12)
    n_injections = len(scenario.get("whatif_injections", []))

    print(f"  Scenario:     {scenario.get('scenario_id', '?')}")
    print(f"  Community:    {community.name}")
    print(f"  Seed personas: {len(seeds)}")
    print(f"  Expand to:    {n_expand} generated personas")
    print(f"  Threads:      {n_threads}")
    print(f"  WhatIf inj:   {n_injections}")
    print(f"  Universes:    {n_universes}")
    print(f"  Seed:         {args.seed}")
    print()

    if args.dry_run:
        print("DRY RUN — validating scenario...")
        print(f"  Community config: OK ({len(community.rules)} rules, {len(community.norms)} norms)")
        print(f"  Seeds parsed:     OK ({len(seeds)} personas)")
        for s in seeds:
            print(f"    - {s.name} ({s.community_role})")
        injections = scenario.get("whatif_injections", [])
        print(f"  WhatIf configs:   OK ({len(injections)} injections)")
        for inj in injections:
            print(f"    - {inj['persona_name']}")
        print(f"  Multiverse:       OK ({n_universes} universes, "
              f"{len(mv_cfg.get('temperatures', []))} temperatures)")
        print()
        print("All configs valid.")
        return 0

    # ── Output directory ─────────────────────────────────────────────
    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        out_dir = Path("runs") / f"{ts}_social_simulacra"
    else:
        out_dir = args.output

    print(f"  Output:       {out_dir}")
    print()

    # ── Phase 1: Expand personas ─────────────────────────────────────
    print("Phase 1: Expanding personas...")
    personas = expand_personas(community, seeds, n=n_expand, seed=args.seed)
    print(f"  {len(personas)} personas ({sum(1 for p in personas if not p.generated)} seed + "
          f"{sum(1 for p in personas if p.generated)} generated)")

    # ── Phase 2: Generate threads ────────────────────────────────────
    print("Phase 2: Generating threads...")
    threads = generate_threads(community, personas, scenario, seed=args.seed)
    total_posts = sum(len(t.posts) for t in threads)
    print(f"  {len(threads)} threads, {total_posts} total posts")

    # ── Phase 3: WhatIf injections ───────────────────────────────────
    print("Phase 3: Running WhatIf injections...")
    whatif_results = run_whatif_injections(community, threads, scenario, seed=args.seed)
    print(f"  {len(whatif_results)} injections completed")

    # ── Phase 4: Multiverse analysis ─────────────────────────────────
    print("Phase 4: Running multiverse analysis...")
    mv = run_multiverse(community, personas, scenario, seed=args.seed)
    print(f"  {len(mv.universes)} universes completed")

    # ── Print summary ────────────────────────────────────────────────
    print_summary(personas, threads, whatif_results, mv)

    # ── Export ────────────────────────────────────────────────────────
    csv_dir = out_dir / "csv"
    print("Exporting results...")
    export_personas_csv(personas, csv_dir / "personas.csv")
    export_threads_csv(threads, csv_dir / "threads.csv")
    export_whatif_csv(whatif_results, csv_dir / "whatif_comparison.csv")
    export_multiverse_csv(mv, csv_dir / "multiverse.csv")
    export_multiverse_summary_csv(mv, csv_dir / "multiverse_summary.csv")
    export_history(scenario, personas, threads, whatif_results, mv, args.seed,
                   out_dir / "history.json")
    print(f"  CSV exported to {csv_dir}/")

    # ── Plots ────────────────────────────────────────────────────────
    generate_plots(personas, threads, whatif_results, mv, out_dir / "plots")

    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
