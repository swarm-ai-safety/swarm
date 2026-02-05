#!/usr/bin/env python
"""
MVP v0 Demo Script

Demonstrates the core simulation loop with 5 agents over 10+ epochs.
Validates the success criteria from the implementation plan.
"""

from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.honest import HonestAgent
from src.agents.opportunistic import OpportunisticAgent
from src.agents.deceptive import DeceptiveAgent
from src.agents.adversarial import AdversarialAgent
from src.core.orchestrator import Orchestrator, OrchestratorConfig
from src.core.payoff import PayoffConfig


def main():
    print("=" * 60)
    print("Distributional AGI Safety Sandbox - MVP v0 Demo")
    print("=" * 60)
    print()

    # Configure simulation
    config = OrchestratorConfig(
        n_epochs=10,
        steps_per_epoch=10,
        seed=42,
        payoff_config=PayoffConfig(
            s_plus=2.0,
            s_minus=1.0,
            h=2.0,
            theta=0.5,
            rho_a=0.0,
            rho_b=0.0,
            w_rep=1.0,
        ),
    )

    # Create orchestrator
    orchestrator = Orchestrator(config=config)

    # Register 5 agents of different types
    print("Registering agents...")
    agents = [
        HonestAgent(agent_id="honest_1"),
        HonestAgent(agent_id="honest_2"),
        HonestAgent(agent_id="honest_3"),
        OpportunisticAgent(agent_id="opportunistic_1"),
        DeceptiveAgent(agent_id="deceptive_1"),
    ]

    for agent in agents:
        state = orchestrator.register_agent(agent)
        print(f"  - {agent.agent_id} ({agent.agent_type.value}): reputation={state.reputation:.2f}")

    print()
    print(f"Running simulation: {config.n_epochs} epochs x {config.steps_per_epoch} steps")
    print("-" * 60)

    # Run simulation
    metrics_history = orchestrator.run()

    print()
    print("Epoch-by-Epoch Metrics:")
    print("-" * 60)
    print(f"{'Epoch':<6} {'Interactions':<13} {'Accepted':<10} {'Toxicity':<10} {'QualityGap':<12} {'Welfare':<10}")
    print("-" * 60)

    for m in metrics_history:
        print(
            f"{m.epoch:<6} "
            f"{m.total_interactions:<13} "
            f"{m.accepted_interactions:<10} "
            f"{m.toxicity_rate:<10.4f} "
            f"{m.quality_gap:<12.4f} "
            f"{m.total_welfare:<10.2f}"
        )

    print("-" * 60)
    print()

    # Summary statistics
    total_interactions = sum(m.total_interactions for m in metrics_history)
    total_accepted = sum(m.accepted_interactions for m in metrics_history)
    avg_toxicity = sum(m.toxicity_rate for m in metrics_history) / len(metrics_history)
    avg_quality_gap = sum(m.quality_gap for m in metrics_history) / len(metrics_history)
    total_welfare = sum(m.total_welfare for m in metrics_history)

    print("Summary Statistics:")
    print(f"  Total Interactions: {total_interactions}")
    print(f"  Accepted Interactions: {total_accepted}")
    print(f"  Average Toxicity Rate: {avg_toxicity:.4f}")
    print(f"  Average Quality Gap: {avg_quality_gap:.4f}")
    print(f"  Total Welfare: {total_welfare:.2f}")
    print()

    # Final agent states
    print("Final Agent States:")
    print("-" * 60)
    for agent in agents:
        state = orchestrator.state.get_agent(agent.agent_id)
        print(
            f"  {agent.agent_id}: "
            f"reputation={state.reputation:.2f}, "
            f"resources={state.resources:.2f}, "
            f"payoff={state.total_payoff:.2f}"
        )

    print()
    print("=" * 60)
    print("MVP v0 Success Criteria Validation:")
    print("=" * 60)

    # Validate success criteria
    success = True

    # Criterion 1: 5 agents interact over 10+ epochs
    if len(agents) >= 5 and len(metrics_history) >= 10:
        print("[PASS] 5 agents completed 10+ epochs")
    else:
        print(f"[FAIL] Only {len(agents)} agents or {len(metrics_history)} epochs")
        success = False

    # Criterion 2: Toxicity and conditional loss metrics computed per epoch
    if all(isinstance(m.toxicity_rate, float) for m in metrics_history):
        print("[PASS] Toxicity metrics computed per epoch")
    else:
        print("[FAIL] Toxicity metrics not computed")
        success = False

    if all(isinstance(m.quality_gap, float) for m in metrics_history):
        print("[PASS] Quality gap metrics computed per epoch")
    else:
        print("[FAIL] Quality gap metrics not computed")
        success = False

    # Criterion 3: Observable coordination patterns
    if total_interactions > 0:
        print(f"[PASS] Observable interactions: {total_interactions}")
    else:
        print("[WARN] No interactions observed (may need more steps)")

    print()
    if success:
        print("MVP v0 Success Criteria: ALL PASSED")
    else:
        print("MVP v0 Success Criteria: SOME FAILED")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
