"""Governance-aware run functions for benchmark experiments.

These run_fn callables simulate the effect of governance friction on agent
behavior. Each function receives a redacted TaskInstance and a governance
config dict, and returns a TaskResult degraded proportionally to the
governance overhead.

The key insight: governance doesn't change the task — it changes how well
agents can execute it. Tight governance adds latency (extra steps),
reduces coordination bandwidth, and may cause partial failures. The
oracle run has zero governance friction, so capability_ratio = 1.0
represents the governance-free upper bound.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from swarm.benchmarks.base import TaskInstance, TaskResult


def _governance_friction(gov_config: dict[str, Any], rng: np.random.Generator) -> float:
    """Compute a scalar friction factor in [0, 1] from governance config.

    0 = no friction (oracle-equivalent), 1 = total blockage.
    """
    friction = 0.0

    # Audit overhead: each audit adds latency
    audit_rate = float(gov_config.get("audit_rate", 0.0))
    friction += audit_rate * 0.25

    # Circuit breaker: chance of aborting in-progress work
    if gov_config.get("circuit_breaker_enabled", False):
        cb_sensitivity = float(gov_config.get("circuit_breaker_sensitivity", 0.5))
        friction += cb_sensitivity * 0.2

    # Staking: high stakes slow down agents (more cautious)
    stake = float(gov_config.get("min_stake", 0.0))
    friction += min(stake / 10.0, 0.3)

    # Bandwidth cap: low bandwidth = slow coordination
    bw_cap = float(gov_config.get("bandwidth_cap", 100))
    if bw_cap < 100:
        friction += (1.0 - bw_cap / 100.0) * 0.15

    # Confirmation gates: each gate adds a step
    n_gates = int(gov_config.get("confirmation_gates", 0))
    friction += min(n_gates * 0.1, 0.3)

    return min(friction, 0.95)  # cap at 95% — never total blockage


def _config_seed_offset(gov_config: dict[str, Any]) -> int:
    """Stable per-config offset for RNG seeding across Python processes."""
    config_id = str(gov_config.get("id", ""))
    digest = hashlib.blake2s(config_id.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def routing_run_fn(instance: TaskInstance, gov_config: dict[str, Any]) -> TaskResult:
    """Simulate governance-degraded routing.

    Governance friction manifests as:
    - Extra steps (audit/confirmation overhead)
    - Probability of payload corruption (noisy channels under constraint)
    - Possible routing detours (suboptimal path under bandwidth limits)
    """
    rng = np.random.default_rng(instance.seed + _config_seed_offset(gov_config))
    friction = _governance_friction(gov_config, rng)

    # Access routing-specific fields
    payload = getattr(instance, "payload", 0)
    source = getattr(instance, "source_agent", "")
    target = getattr(instance, "target_agent", "")
    network = getattr(instance, "network", None)

    # Oracle would find shortest path; friction causes detours
    if network is not None:
        from swarm.benchmarks.routing.message_routing import _bfs_path
        path = _bfs_path(network, source, target)
    else:
        path = [source, target]

    if not path:
        return TaskResult(completed=False, payload=None, steps_taken=0, agent_trace=[])

    # Friction adds extra steps (rerouting, confirmations)
    extra_steps = int(friction * len(path) * 3)
    actual_steps = len(path) - 1 + extra_steps

    # Payload corruption probability increases with friction
    # Near-frictionless configs should be highly stable: only meaningful
    # governance pressure should produce corruption events.
    effective_friction = max(0.0, friction - 0.05)
    corruption_prob = min(effective_friction * 0.45, 0.95)
    if rng.random() < corruption_prob:
        payload = int(rng.integers(0, 10_000))  # corrupted
        completed = True  # delivered, but wrong payload
    else:
        completed = True

    # Route detours: with high friction, agents take longer paths
    actual_trace = list(path)
    if friction > 0.3 and network is not None:
        # Add random intermediate hops
        all_agents = [f"agent_{i}" for i in range(instance.n_agents)]
        n_detours = int(friction * 3)
        for _ in range(n_detours):
            insert_pos = rng.integers(1, max(len(actual_trace), 2))
            detour = rng.choice(all_agents)
            actual_trace.insert(insert_pos, detour)

    return TaskResult(
        completed=completed,
        payload=payload,
        steps_taken=actual_steps,
        agent_trace=actual_trace,
    )


def coordination_run_fn(instance: TaskInstance, gov_config: dict[str, Any]) -> TaskResult:
    """Simulate governance-degraded coordination/allocation.

    Friction reduces coordination quality — agents can't communicate as
    freely, leading to suboptimal allocations.
    """
    rng = np.random.default_rng(instance.seed + _config_seed_offset(gov_config))
    friction = _governance_friction(gov_config, rng)

    target_total = getattr(instance, "target_total", 0.0)
    capacities = getattr(instance, "agent_capacities", {})

    if not capacities:
        return TaskResult(completed=False, payload={}, steps_taken=0, agent_trace=[])

    # Oracle does proportional allocation; friction adds noise
    total_cap = sum(capacities.values())
    ratio = target_total / total_cap if total_cap > 0 else 0.0

    allocation = {}
    for aid, cap in capacities.items():
        ideal = cap * ratio
        # Friction adds noise proportional to governance overhead
        noise = rng.normal(0, friction * ideal * 0.5)
        allocation[aid] = max(0.0, ideal + noise)

    # Extra coordination rounds needed
    steps = 1 + int(friction * 5)

    return TaskResult(
        completed=True,
        payload=allocation,
        steps_taken=steps,
        agent_trace=sorted(capacities.keys()),
    )


def auction_run_fn(instance: TaskInstance, gov_config: dict[str, Any]) -> TaskResult:
    """Simulate governance-degraded auction.

    Friction prevents agents from fully expressing valuations, leading to
    suboptimal resource assignments.
    """
    rng = np.random.default_rng(instance.seed + _config_seed_offset(gov_config))
    friction = _governance_friction(gov_config, rng)

    valuations = getattr(instance, "agent_valuations", {})
    n_resources = getattr(instance, "n_resources", 0)

    if not valuations:
        return TaskResult(completed=False, payload={}, steps_taken=0, agent_trace=[])

    agent_ids = list(valuations.keys())
    assignment: dict[int, str] = {}

    for r in range(n_resources):
        # Friction causes agents to misreport valuations (noisy bids)
        noisy_vals = {}
        for aid in agent_ids:
            true_val = valuations[aid][r]
            noise = rng.normal(0, friction * true_val * 0.8)
            noisy_vals[aid] = max(0.0, true_val + noise)

        # Assign to highest noisy bidder
        winner = max(agent_ids, key=lambda a: noisy_vals[a])
        assignment[r] = winner

    steps = 1 + int(friction * 3)

    return TaskResult(
        completed=True,
        payload=assignment,
        steps_taken=steps,
        agent_trace=sorted(set(assignment.values())),
    )


def pipeline_run_fn(instance: TaskInstance, gov_config: dict[str, Any]) -> TaskResult:
    """Simulate governance-degraded pipeline execution.

    Governance gates between stages add latency. High friction may cause
    agents to fail at intermediate stages (incomplete pipeline).
    """
    from swarm.benchmarks.long_horizon.pipeline_task import _stage_transform

    rng = np.random.default_rng(instance.seed + _config_seed_offset(gov_config))
    friction = _governance_friction(gov_config, rng)

    initial_payload = getattr(instance, "initial_payload", 0)
    stages = getattr(instance, "stages", [])

    if not stages:
        return TaskResult(completed=False, payload=0, steps_taken=0, agent_trace=[])

    payload = initial_payload
    trace: list[str] = []
    total_steps = 0

    for stage in stages:
        # Each governance gate has a chance of blocking the pipeline
        gate_failure_prob = friction * 0.15
        if rng.random() < gate_failure_prob:
            # Pipeline stalls at this stage
            return TaskResult(
                completed=True,  # attempted but incomplete
                payload=payload,  # intermediate state
                steps_taken=total_steps + 1,
                agent_trace=trace,
            )

        # Execute stage transform
        payload = _stage_transform(payload, stage.transform_key)
        trace.append(stage.agent_id)

        # Governance overhead: extra steps per gate
        gate_overhead = int(friction * 2)
        total_steps += 1 + gate_overhead

    return TaskResult(
        completed=True,
        payload=payload,
        steps_taken=total_steps,
        agent_trace=trace,
    )


# Registry mapping task_type to run_fn
RUN_FN_REGISTRY: dict[str, Any] = {
    "routing": routing_run_fn,
    "coordination": coordination_run_fn,
    "allocation": auction_run_fn,
    "long_horizon": pipeline_run_fn,
}
