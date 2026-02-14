#!/usr/bin/env python
"""Minimal SWARM example with decision-level replay disagreement.

Run:
    python examples/illusion_delta_minimal.py
"""

from __future__ import annotations

import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict

sys.path.insert(0, str(Path(__file__).parent.parent))

from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.logging.event_log import EventLog
from swarm.metrics.incoherence import disagreement_rate, illusion_delta
from swarm.models.events import EventType


def _run_with_log(seed: int) -> tuple[list[dict[str, object]], list[float]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / f"events_{seed}.jsonl"
        cfg = OrchestratorConfig(
            n_epochs=12,
            steps_per_epoch=8,
            seed=seed,
            log_path=log_path,
            log_events=True,
            observation_noise_probability=0.25,
            observation_noise_std=0.15,
        )
        orchestrator = Orchestrator(config=cfg)
        orchestrator.register_agent(HonestAgent(agent_id="honest_a", name="Alice"))
        orchestrator.register_agent(HonestAgent(agent_id="honest_b", name="Bob"))
        orchestrator.register_agent(
            DeceptiveAgent(agent_id="deceptive_x", name="Mallory")
        )
        orchestrator.run()

        events = list(EventLog(log_path).replay())

    decisions: list[dict[str, str | bool]] = []
    accepted_p_values: list[float] = []

    completed_acceptance: dict[str, bool] = {}
    for event in events:
        if (
            event.event_type == EventType.INTERACTION_COMPLETED
            and event.interaction_id is not None
        ):
            completed_acceptance[event.interaction_id] = bool(
                event.payload.get("accepted", False)
            )

    proposal_index = 0
    for event in events:
        if (
            event.event_type != EventType.INTERACTION_PROPOSED
            or event.interaction_id is None
            or event.initiator_id is None
            or event.counterparty_id is None
        ):
            continue

        decision_id = f"proposal_slot_{proposal_index}"
        proposal_index += 1

        accepted = completed_acceptance.get(event.interaction_id, False)
        interaction_type = str(event.payload.get("interaction_type", "unknown"))
        action_token = (
            f"{event.initiator_id}->{event.counterparty_id}:{interaction_type}:"
            f"{'accept' if accepted else 'reject'}"
        )
        decisions.append({"decision_id": decision_id, "action": action_token})

        if accepted:
            accepted_p_values.append(float(event.payload.get("p", 0.0)))

    return decisions, accepted_p_values


def main() -> int:
    seeds = [7, 8, 9, 10]

    all_decisions: list[list[dict[str, object]]] = []
    all_accepted_ps: list[list[float]] = []

    for seed in seeds:
        decisions, accepted_ps = _run_with_log(seed)
        all_decisions.append(decisions)
        all_accepted_ps.append(accepted_ps)

    decision_votes: DefaultDict[str, list[str]] = defaultdict(list)
    replay_counts: DefaultDict[str, int] = defaultdict(int)

    for decisions in all_decisions:
        seen_ids: set[str] = set()
        for row in decisions:
            decision_id = str(row["decision_id"])
            decision_votes[decision_id].append(str(row["action"]))
            if decision_id not in seen_ids:
                replay_counts[decision_id] += 1
                seen_ids.add(decision_id)

    disagreement_rates = [
        disagreement_rate(votes)
        for decision_id, votes in decision_votes.items()
        if replay_counts[decision_id] >= 2 and len(votes) >= 2
    ]

    p_values = all_accepted_ps[0]
    gap = illusion_delta(p_values=p_values, disagreement_rates=disagreement_rates)

    print("seed,accepted_interactions")
    for seed, accepted_ps in zip(seeds, all_accepted_ps, strict=True):
        print(f"{seed},{len(accepted_ps)}")

    print("\nsummary")
    print(f"accepted_samples_reference={len(p_values)}")
    print(f"decision_points_compared={len(disagreement_rates)}")
    print(f"perceived_coherence={gap.perceived_coherence:.4f}")
    print(f"distributed_coherence={gap.distributed_coherence:.4f}")
    print(f"illusion_delta={gap.illusion_delta:.4f}")

    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        print("\nInstall plotting deps to generate a figure:")
        print("python -m pip install -e '.[analysis]'")
        return 0

    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "illusion_delta_minimal.png"

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(disagreement_rates, bins=10, color="#c0392b", alpha=0.85)
    ax.set_title(f"Decision disagreement (illusion Δ={gap.illusion_delta:.3f})")
    ax.set_xlabel("decision-level disagreement rate")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(f"\nWrote plot: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
