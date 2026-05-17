#!/usr/bin/env python
"""
Asymmetric Information Study — Transparency as Stabilizer or Destabilizer.

Research question:
    Does an intelligence advantage stabilize or destabilize escalation dynamics?
    When one agent can see through fog-of-war better than its opponent, does
    the informed agent exploit the gap, or does reduced uncertainty prevent
    accidental escalation?

Design:
    Sweep intelligence_quality asymmetry across strategy pairings.
    For each pairing, compare three conditions:
        symmetric_low:  (0.2, 0.2) — both agents in the dark
        symmetric_high: (0.8, 0.8) — both agents well-informed
        asymmetric:     (0.9, 0.2) — one agent has clear advantage

    Strategy pairings (6):
        dove vs dove          — cooperative baseline
        hawk vs hawk          — aggressive baseline
        tft vs tft            — reciprocal baseline
        calculating vs tft    — strategic deception vs reciprocator
        hawk vs dove          — aggressor vs pacifist
        calculating vs dove   — exploiter vs cooperative target

    Each condition runs N_EPISODES episodes with different seeds to smooth
    stochastic fog-of-war noise.

Metrics per condition:
    Escalation max, velocity, signal-action divergence, trust exploitation,
    accidental escalation rate, fog catastrophes, de-escalation rate,
    welfare composite, collateral damage, outcome distribution.

Usage:
    python examples/run_asymmetric_info_study.py [--episodes 20] [--turns 25]
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.domains.escalation_sandbox.config import (
    AgentConfig,
    CrisisConfig,
    EscalationConfig,
    FogOfWarConfig,
    GovernanceLeverConfig,
    SignalConfig,
)
from swarm.domains.escalation_sandbox.metrics import (
    EscalationMetrics,
    compute_sweep_statistics,
)
from swarm.domains.escalation_sandbox.runner import EscalationRunner

# ---------------------------------------------------------------------------
# Study parameters
# ---------------------------------------------------------------------------

N_EPISODES = 20  # episodes per condition (overridable via --episodes)
MAX_TURNS = 25   # turns per episode (overridable via --turns)
BASE_SEED = 42

# Intelligence quality conditions
IQ_CONDITIONS = {
    "symmetric_low":  (0.2, 0.2),
    "symmetric_high": (0.8, 0.8),
    "asymmetric":     (0.9, 0.2),
}

# Strategy pairings: (agent_a_persona, agent_b_persona)
STRATEGY_PAIRINGS = [
    ("dove", "dove"),
    ("hawk", "hawk"),
    ("tit_for_tat", "tit_for_tat"),
    ("calculating", "tit_for_tat"),
    ("hawk", "dove"),
    ("calculating", "dove"),
]


def _build_config(
    persona_a: str,
    persona_b: str,
    iq_a: float,
    iq_b: float,
    seed: int,
    max_turns: int,
) -> EscalationConfig:
    """Build an EscalationConfig for one episode."""
    return EscalationConfig(
        agents=[
            AgentConfig(
                agent_id="nation_a",
                name="Nation Alpha",
                agent_type="scripted",
                persona=persona_a,
                has_nuclear=True,
                has_second_strike=True,
                military_strength=100.0,
                economic_strength=100.0,
                intelligence_quality=iq_a,
            ),
            AgentConfig(
                agent_id="nation_b",
                name="Nation Beta",
                agent_type="scripted",
                persona=persona_b,
                has_nuclear=True,
                has_second_strike=True,
                military_strength=100.0,
                economic_strength=100.0,
                intelligence_quality=iq_b,
            ),
        ],
        crisis=CrisisConfig(
            template="border_dispute",
            timeline_turns=max_turns,
            initial_escalation=0,
            asymmetric=True,
            timeline_pressure=False,
        ),
        fog_of_war=FogOfWarConfig(
            enabled=True,
            noise_mu=0.0,
            noise_sigma=0.5,
            positive_skew=0.3,
            per_level_variance_scale=0.1,
            intelligence_reduction_factor=0.5,
        ),
        signals=SignalConfig(
            broadcast_before_action=True,
            trust_decay_rate=0.95,
            commitment_trap_enabled=True,
            commitment_reputation_cost=5.0,
        ),
        governance=GovernanceLeverConfig(
            mad_enabled=True,
            mad_retaliation_probability=0.9,
            treaty_max_level=5,
            treaty_defection_penalty=10.0,
            circuit_breaker_enabled=True,
            circuit_breaker_threshold=7,
            circuit_breaker_negotiation_turns=1,
            mediation_enabled=False,
            back_channel_enabled=False,
            de_escalation_friction_multiplier=1.0,
        ),
        seed=seed,
        max_turns=max_turns,
    )


def _run_condition(
    persona_a: str,
    persona_b: str,
    iq_a: float,
    iq_b: float,
    n_episodes: int,
    max_turns: int,
) -> list[EscalationMetrics]:
    """Run N episodes for a single (pairing, iq) condition."""
    all_metrics: list[EscalationMetrics] = []
    for ep in range(n_episodes):
        seed = BASE_SEED + ep * 7  # spread seeds
        config = _build_config(persona_a, persona_b, iq_a, iq_b, seed, max_turns)
        runner = EscalationRunner(config, seed=seed)
        metrics = runner.run()
        all_metrics.append(metrics)
    return all_metrics


def _per_agent_summary(
    metrics_list: list[EscalationMetrics], agent_id: str,
) -> dict[str, float]:
    """Average per-agent metrics across episodes."""
    n = len(metrics_list)
    if n == 0:
        return {}
    fields = [
        "final_level", "signal_action_divergence", "trust_score",
        "military_remaining", "economic_remaining", "welfare_remaining",
        "total_damage",
    ]
    result: dict[str, float] = {}
    for f in fields:
        vals = [
            m.per_agent.get(agent_id, {}).get(f, 0.0)
            for m in metrics_list
        ]
        result[f"mean_{f}"] = sum(vals) / n
    return result


def _outcome_distribution(metrics_list: list[EscalationMetrics]) -> dict[str, int]:
    """Count outcome types across episodes."""
    dist: dict[str, int] = {}
    for m in metrics_list:
        dist[m.outcome] = dist.get(m.outcome, 0) + 1
    return dist


def main() -> int:
    # Parse CLI args
    n_episodes = N_EPISODES
    max_turns = MAX_TURNS
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--episodes" and i + 1 < len(args):
            n_episodes = int(args[i + 1])
            i += 2
        elif args[i] == "--turns" and i + 1 < len(args):
            max_turns = int(args[i + 1])
            i += 2
        else:
            i += 1

    total_runs = len(STRATEGY_PAIRINGS) * len(IQ_CONDITIONS) * n_episodes
    print("=" * 78)
    print("Asymmetric Information Study")
    print("Transparency as Stabilizer or Destabilizer")
    print("=" * 78)
    print()
    print(f"Strategy pairings:    {len(STRATEGY_PAIRINGS)}")
    print(f"IQ conditions:        {len(IQ_CONDITIONS)}")
    print(f"Episodes per cond:    {n_episodes}")
    print(f"Turns per episode:    {max_turns}")
    print(f"Total episodes:       {total_runs}")
    print()

    # ---------------------------------------------------------------------------
    # Run sweep
    # ---------------------------------------------------------------------------

    # results[pairing_label][condition_name] = sweep_stats dict
    results: dict[str, dict[str, Any]] = {}
    # per_agent_results[pairing_label][condition_name] = {agent_id: summary}
    per_agent_results: dict[str, dict[str, dict[str, Any]]] = {}

    run_idx = 0
    for persona_a, persona_b in STRATEGY_PAIRINGS:
        pairing_label = f"{persona_a}_vs_{persona_b}"
        results[pairing_label] = {}
        per_agent_results[pairing_label] = {}

        print(f"--- {pairing_label} ---")

        for cond_name, (iq_a, iq_b) in IQ_CONDITIONS.items():
            run_idx += n_episodes
            print(
                f"  {cond_name} (IQ: {iq_a:.1f} vs {iq_b:.1f}) "
                f"[{run_idx}/{total_runs}]..."
            )

            episode_metrics = _run_condition(
                persona_a, persona_b, iq_a, iq_b, n_episodes, max_turns,
            )

            # Aggregate
            sweep_stats = compute_sweep_statistics(episode_metrics)
            outcomes = _outcome_distribution(episode_metrics)
            sweep_stats["outcome_distribution"] = outcomes

            results[pairing_label][cond_name] = sweep_stats

            # Per-agent
            pa = {}
            pa["nation_a"] = _per_agent_summary(episode_metrics, "nation_a")
            pa["nation_b"] = _per_agent_summary(episode_metrics, "nation_b")
            per_agent_results[pairing_label][cond_name] = pa

        print()

    # ---------------------------------------------------------------------------
    # Analysis 1: Condition comparison table per pairing
    # ---------------------------------------------------------------------------

    print("=" * 78)
    print("Analysis 1: Intelligence Asymmetry Effect by Strategy Pairing")
    print("=" * 78)
    print()

    key_metrics = [
        ("mean_escalation_max", "Esc Max", ".1f"),
        ("mean_escalation_velocity", "Esc Vel", ".3f"),
        ("mean_signal_action_divergence", "Diverge", ".3f"),
        ("mean_trust_exploitation_index", "TrustExp", ".3f"),
        ("mean_accidental_escalation_rate", "AccEsc", ".3f"),
        ("total_fog_catastrophes", "FogCat", "d"),
        ("mean_welfare_composite", "Welfare", ".1f"),
        ("mean_collateral_damage", "Collat", ".1f"),
        ("nuclear_threshold_rate", "NucRate", ".2f"),
        ("mutual_destruction_rate", "MutDest", ".2f"),
    ]

    for pairing_label, cond_results in results.items():
        print(f"  {pairing_label}")
        # Header
        header = f"    {'Metric':<12}"
        for cond_name in IQ_CONDITIONS:
            header += f"  {cond_name:<14}"
        print(header)
        print("    " + "-" * (12 + 16 * len(IQ_CONDITIONS)))

        for metric_key, metric_label, fmt in key_metrics:
            row = f"    {metric_label:<12}"
            for cond_name in IQ_CONDITIONS:
                val = cond_results[cond_name].get(metric_key, 0)
                if isinstance(val, int):
                    row += f"  {val:<14}"
                else:
                    row += f"  {val:<14{fmt}}"
            print(row)
        print()

    # ---------------------------------------------------------------------------
    # Analysis 2: Asymmetry advantage — per-agent breakdown
    # ---------------------------------------------------------------------------

    print("=" * 78)
    print("Analysis 2: Information Advantage — Per-Agent in Asymmetric Condition")
    print("=" * 78)
    print()

    agent_fields = [
        ("mean_final_level", "Final Lv", ".1f"),
        ("mean_signal_action_divergence", "Diverge", ".3f"),
        ("mean_trust_score", "Trust", ".3f"),
        ("mean_welfare_remaining", "Welfare", ".1f"),
        ("mean_total_damage", "Damage", ".1f"),
    ]

    for pairing_label, cond_data in per_agent_results.items():
        asym = cond_data.get("asymmetric", {})
        if not asym:
            continue

        print(f"  {pairing_label} (asymmetric: A=0.9, B=0.2)")
        header = f"    {'Metric':<12}  {'Nation A (IQ=0.9)':<20}  {'Nation B (IQ=0.2)':<20}"
        print(header)
        print("    " + "-" * 56)

        for field_key, label, fmt in agent_fields:
            val_a = asym.get("nation_a", {}).get(field_key, 0)
            val_b = asym.get("nation_b", {}).get(field_key, 0)
            row = f"    {label:<12}  {val_a:<20{fmt}}  {val_b:<20{fmt}}"
            print(row)
        print()

    # ---------------------------------------------------------------------------
    # Analysis 3: Stabilization score
    # ---------------------------------------------------------------------------

    print("=" * 78)
    print("Analysis 3: Does Transparency Stabilize or Destabilize?")
    print("=" * 78)
    print()

    # For each pairing, compute delta between asymmetric and symmetric_high
    # on key harm metrics. Positive delta = asymmetry makes things worse.
    harm_metrics = [
        "mean_escalation_max",
        "mean_accidental_escalation_rate",
        "mean_collateral_damage",
        "nuclear_threshold_rate",
        "mutual_destruction_rate",
    ]
    benefit_metrics = [
        "mean_welfare_composite",
        "mean_de_escalation_rate",
    ]

    print(f"  {'Pairing':<24}  {'Stab Score':<12}  {'Verdict':<20}")
    print("  " + "-" * 60)

    pairing_verdicts: dict[str, str] = {}
    for pairing_label, cond_results in results.items():
        sym_high = cond_results.get("symmetric_high", {})
        asym = cond_results.get("asymmetric", {})

        if not sym_high or not asym:
            continue

        # Stabilization score: sum of (symmetric_high - asymmetric) for harm
        # metrics + sum of (asymmetric - symmetric_high) for benefit metrics.
        # Positive = asymmetry is BETTER (stabilizing).
        score = 0.0
        for m in harm_metrics:
            sh = sym_high.get(m, 0) or 0
            am = asym.get(m, 0) or 0
            score += (sh - am)  # positive if asymmetric has less harm

        for m in benefit_metrics:
            sh = sym_high.get(m, 0) or 0
            am = asym.get(m, 0) or 0
            score += (am - sh)  # positive if asymmetric has more benefit

        if score > 0.5:
            verdict = "STABILIZING"
        elif score < -0.5:
            verdict = "DESTABILIZING"
        else:
            verdict = "NEUTRAL"

        pairing_verdicts[pairing_label] = verdict
        print(f"  {pairing_label:<24}  {score:<12.3f}  {verdict:<20}")

    print()

    # Summary
    stab_count = sum(1 for v in pairing_verdicts.values() if v == "STABILIZING")
    destab_count = sum(1 for v in pairing_verdicts.values() if v == "DESTABILIZING")
    neutral_count = sum(1 for v in pairing_verdicts.values() if v == "NEUTRAL")
    print(f"  Stabilizing: {stab_count}/{len(pairing_verdicts)}")
    print(f"  Destabilizing: {destab_count}/{len(pairing_verdicts)}")
    print(f"  Neutral: {neutral_count}/{len(pairing_verdicts)}")
    print()

    # ---------------------------------------------------------------------------
    # Analysis 4: Fog-of-war floor effect
    # ---------------------------------------------------------------------------

    print("=" * 78)
    print("Analysis 4: Fog-of-War Floor — Low IQ vs High IQ Symmetric")
    print("=" * 78)
    print()

    print(f"  {'Pairing':<24}  {'AccEsc(low)':<12}  {'AccEsc(high)':<13}  "
          f"{'FogCat(low)':<12}  {'FogCat(high)':<12}")
    print("  " + "-" * 78)

    for pairing_label, cond_results in results.items():
        sym_low = cond_results.get("symmetric_low", {})
        sym_high = cond_results.get("symmetric_high", {})

        acc_low = sym_low.get("mean_accidental_escalation_rate", 0) or 0
        acc_high = sym_high.get("mean_accidental_escalation_rate", 0) or 0
        fog_low = sym_low.get("total_fog_catastrophes", 0) or 0
        fog_high = sym_high.get("total_fog_catastrophes", 0) or 0

        print(
            f"  {pairing_label:<24}  {acc_low:<12.3f}  {acc_high:<13.3f}  "
            f"{fog_low:<12}  {fog_high:<12}"
        )

    print()

    # ---------------------------------------------------------------------------
    # Export
    # ---------------------------------------------------------------------------

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = Path("runs") / f"{timestamp}_asymmetric_info_seed{BASE_SEED}"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Main results JSON
    export_data: dict[str, Any] = {
        "study": "asymmetric_information",
        "n_episodes_per_condition": n_episodes,
        "max_turns": max_turns,
        "base_seed": BASE_SEED,
        "iq_conditions": {k: list(v) for k, v in IQ_CONDITIONS.items()},
        "strategy_pairings": [
            {"a": a, "b": b} for a, b in STRATEGY_PAIRINGS
        ],
        "results": {},
        "per_agent_asymmetric": {},
        "verdicts": pairing_verdicts,
    }

    for pairing_label, cond_results in results.items():
        export_data["results"][pairing_label] = {}
        for cond_name, stats in cond_results.items():
            # Convert outcome_distribution to serializable form
            serializable = {}
            for k, v in stats.items():
                if isinstance(v, (int, float, str, type(None))):
                    serializable[k] = v
                elif isinstance(v, dict):
                    serializable[k] = v
                else:
                    serializable[k] = str(v)
            export_data["results"][pairing_label][cond_name] = serializable

    for pairing_label, cond_data in per_agent_results.items():
        asym = cond_data.get("asymmetric", {})
        if asym:
            export_data["per_agent_asymmetric"][pairing_label] = asym

    results_path = export_dir / "asymmetric_info_results.json"
    with open(results_path, "w") as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported results: {results_path}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
