"""SimWorld Delivery × SWARM Screening Validation

Runs the delivery scenario, then maps each agent's behavioral profile
to a CapabilityManifest and evaluates separation quality through the
ScreeningProtocol.

Tests the hypothesis from theory.md: screening should produce
separation quality > 0.6 if the personality-strategy correspondence holds.

Usage:
    python examples/run_simworld_screening_validation.py [--seeds 5]
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from swarm.bridges.opensandbox.config import (
    AgentType,
    CapabilityManifest,
    GovernanceContract,
    InteractionPolicy,
    NetworkPolicy,
    OpenSandboxConfig,
)
from swarm.bridges.opensandbox.screener import ScreeningProtocol
from swarm.domains.simworld_delivery.config import DeliveryConfig
from swarm.domains.simworld_delivery.entities import AgentState
from swarm.domains.simworld_delivery.runner import DeliveryScenarioRunner


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def agent_to_manifest(agent: AgentState) -> CapabilityManifest:
    """Map a delivery agent's behavioral profile to a CapabilityManifest.

    The mapping uses observed behavior (not declared type) to infer
    the agent_type — this tests whether the screener can distinguish
    agents by their actions.
    """
    # Behavioral signals
    completion_rate = agent.deliveries_completed / max(
        agent.deliveries_completed + agent.deliveries_failed, 1
    )
    overbid_rate = agent.overbids / max(agent.total_bids, 1)

    # Map behavior to AgentType
    if completion_rate > 0.7 and overbid_rate < 0.15:
        agent_type = AgentType.COOPERATIVE
    elif overbid_rate > 0.25 or agent.deliveries_failed > agent.deliveries_completed:
        agent_type = AgentType.ADVERSARIAL
    elif agent.has_scooter and overbid_rate > 0.1:
        agent_type = AgentType.SELF_MODIFYING  # risk-taking, tool-investing
    else:
        agent_type = AgentType.STATIC  # conservative, stable

    # Resource requests scale with observed activity
    memory = 512 if agent.deliveries_completed > 3 else 256
    cpu = 1024 if agent.has_scooter else 512

    return CapabilityManifest(
        agent_id=agent.agent_id,
        agent_type=agent_type,
        capabilities=["python", "file_read"],
        requires_network=False,
        max_memory_mb=memory,
        max_cpu_shares=cpu,
        declared_intent=f"delivery_{agent.persona.value}",
        metadata={
            "persona": agent.persona.value,
            "completion_rate": completion_rate,
            "overbid_rate": overbid_rate,
            "reputation": agent.reputation,
            "earnings": agent.total_earnings,
        },
    )


def build_contracts() -> Dict[str, GovernanceContract]:
    """Create a 3-tier contract structure for the delivery domain.

    Tiers are designed to create meaningful separation:
    - Premium: high resources, full capabilities (cooperative agents)
    - Standard: moderate resources, basic capabilities (static agents)
    - Restricted: minimal resources, limited capabilities (adversarial)
    """
    return {
        "premium": GovernanceContract(
            contract_id="delivery_premium",
            tier="premium",
            capabilities=["python", "file_read", "file_write"],
            network=NetworkPolicy.ALLOWLIST,
            interaction=InteractionPolicy.SHARED_FS,
            max_memory_mb=1024,
            max_cpu_shares=2048,
            timeout_seconds=3600,
        ),
        "standard": GovernanceContract(
            contract_id="delivery_standard",
            tier="standard",
            capabilities=["python", "file_read"],
            network=NetworkPolicy.DENY_ALL,
            interaction=InteractionPolicy.MESSAGE_BUS_ONLY,
            max_memory_mb=400,
            max_cpu_shares=800,
            timeout_seconds=1800,
        ),
        "restricted": GovernanceContract(
            contract_id="delivery_restricted",
            tier="restricted",
            capabilities=["python"],
            network=NetworkPolicy.DENY_ALL,
            interaction=InteractionPolicy.NONE,
            max_memory_mb=200,
            max_cpu_shares=400,
            timeout_seconds=900,
        ),
    }


def compute_separation_quality(
    assignments: List[dict],
) -> float:
    """Compute separation quality: how well the screener sorts agent types.

    Separation quality = fraction of (cooperative → premium/standard) +
    (adversarial → restricted/rejected) assignments that are correct.
    """
    correct = 0
    total = 0

    for a in assignments:
        agent_type = a["agent_type"]
        tier = a["tier"]
        rejected = a.get("rejected", False)

        if agent_type == "cooperative":
            total += 1
            if tier in ("premium", "standard"):
                correct += 1
        elif agent_type == "adversarial":
            total += 1
            if tier == "restricted" or rejected:
                correct += 1
        elif agent_type == "self_modifying":
            total += 1
            if tier in ("standard", "restricted"):
                correct += 1
        elif agent_type == "static":
            total += 1
            if tier in ("standard", "premium"):
                correct += 1

    return correct / max(total, 1)


def run_validation(scenario_path: str, seeds: int) -> None:
    scenario = load_scenario(scenario_path)
    contracts = build_contracts()

    all_sep_quality: List[float] = []
    all_assignments: List[List[dict]] = []

    for seed in range(seeds):
        delivery_cfg = scenario.get("delivery", {})
        config = DeliveryConfig.from_dict({**delivery_cfg, "seed": seed})

        sim = scenario.get("simulation", {})
        runner = DeliveryScenarioRunner(
            config=config,
            agent_specs=scenario.get("agents", []),
            n_epochs=sim.get("n_epochs", 10),
            steps_per_epoch=sim.get("steps_per_epoch", 20),
            seed=seed,
        )
        runner.run()

        # Build manifests from behavioral data
        agents = runner.env.agents
        os_config = OpenSandboxConfig()
        screener = ScreeningProtocol(os_config)

        seed_assignments = []
        for agent_id, agent_state in agents.items():
            manifest = agent_to_manifest(agent_state)
            assignment = screener.evaluate(manifest, contracts)
            seed_assignments.append({
                "agent_id": agent_id,
                "persona": agent_state.persona.value,
                "agent_type": manifest.agent_type.value,
                "tier": assignment.tier,
                "score": assignment.score,
                "rejected": assignment.rejected,
                "completion_rate": manifest.metadata["completion_rate"],
                "overbid_rate": manifest.metadata["overbid_rate"],
                "reputation": manifest.metadata["reputation"],
            })

        sep_q = compute_separation_quality(seed_assignments)
        all_sep_quality.append(sep_q)
        all_assignments.append(seed_assignments)

    # Report
    print("\n" + "=" * 70)
    print("SCREENING PROTOCOL VALIDATION — SIMWORLD DELIVERY ECONOMY")
    print("=" * 70)
    print(f"Seeds: {seeds}")

    mean_sep = statistics.mean(all_sep_quality)
    std_sep = statistics.stdev(all_sep_quality) if len(all_sep_quality) > 1 else 0.0

    print(f"\nSeparation Quality: {mean_sep:.3f} ± {std_sep:.3f}")
    if mean_sep > 0.6:
        print("  [✓] PASS: separation quality > 0.6 threshold")
        print("      Personality-strategy correspondence confirmed")
    else:
        print("  [✗] FAIL: separation quality below 0.6 threshold")
        print("      Proxy signals may not capture personality-driven variation")

    # Detailed breakdown from last seed
    print("\n── Agent Assignments (Last Seed) ──")
    print(f"{'Agent':<35} {'Persona':<15} {'Inferred':<15} "
          f"{'Tier':<12} {'Score':>6} {'CompRate':>8} {'Overbid':>8}")
    print("-" * 105)
    for a in all_assignments[-1]:
        print(
            f"{a['agent_id']:<35} {a['persona']:<15} {a['agent_type']:<15} "
            f"{a['tier']:<12} {a['score']:>6.3f} "
            f"{a['completion_rate']:>8.3f} {a['overbid_rate']:>8.3f}"
        )

    # Tier distribution
    print("\n── Tier Distribution (All Seeds) ──")
    tier_counts: Dict[str, int] = {}
    type_tier: Dict[str, Dict[str, int]] = {}
    for seed_as in all_assignments:
        for a in seed_as:
            tier = a["tier"] or "rejected"
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            key = a["agent_type"]
            type_tier.setdefault(key, {})
            type_tier[key][tier] = type_tier[key].get(tier, 0) + 1

    for atype, tiers in sorted(type_tier.items()):
        parts = ", ".join(f"{t}={c}" for t, c in sorted(tiers.items()))
        print(f"  {atype:<15}: {parts}")

    print("\n" + "=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SimWorld Delivery × SWARM Screening Validation",
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument(
        "--scenario",
        default="scenarios/simworld_delivery_baseline.yaml",
    )
    args = parser.parse_args()
    run_validation(args.scenario, args.seeds)


if __name__ == "__main__":
    main()
