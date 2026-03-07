---
description: "Contract screening, ABC behavioral contracts, drift detection, and pipeline compositionality."
---

# Contracts API

The contracts module implements pre-play contract screening and the ABC behavioral contracts formalism. Agents opt into governance protocols that enforce interaction properties (truthfulness, fairness). Agents who refuse are routed to a default market with no protections.

## Contract

Abstract base for interaction protocols. Each contract defines a signing cost (costly signaling), an execution protocol, and a penalty mechanism.

```python
from swarm.contracts import TruthfulAuctionContract, FairDivisionContract, DefaultMarket

# Vickrey/VCG-style truthful mechanism
auction = TruthfulAuctionContract(
    stake_fraction=0.05,
    audit_probability=0.2,
    audit_threshold_p=0.4,
    penalty_multiplier=3.0,
    surplus_bonus=0.1,
)

# Envy-free allocation protocol
fair = FairDivisionContract(
    entry_fee=2.0,
    fairness_bonus=0.05,
    redistribution_rate=0.1,
)

# Baseline with no protections (higher friction)
default = DefaultMarket(friction_premium=0.05)
```

### Contract Types

| Type | Signing Cost | Audit | Key Property |
|------|-------------|-------|-------------|
| `TruthfulAuctionContract` | Proportional bond (stake_fraction * resources) | Probabilistic | Truthful bidding is dominant strategy |
| `FairDivisionContract` | Flat entry fee | None | Proportional fairness (not strategyproof) |
| `DefaultMarket` | Free | None | No protections, higher friction |

### Contract Lifecycle

```
offer() → sign(agent) → execute(interaction) → penalize()
```

## ContractMarket

Orchestrates the signing stage and routes interactions through the appropriate contract.

```python
from swarm.contracts import ContractMarket, ContractMarketConfig

config = ContractMarketConfig(
    allow_switching=True,
    switching_cost_multiplier=0.5,
    belief_update_rate=0.3,
)

market = ContractMarket(config=config, seed=42)

# Each epoch: agents choose contracts
memberships = market.run_signing_stage(agents, epoch=0)

# Route interactions through chosen contract
modified = market.route_interaction(interaction)

# Update beliefs about pool quality
market.update_beliefs()

# Inspect pool composition
composition = market.get_pool_composition()
# {"truthful_auction": {"honest": 3}, "default_market": {"adversarial": 2}}
```

### ContractMarketConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `allow_switching` | bool | True | Can agents switch contracts between epochs |
| `switching_cost_multiplier` | float | 0.5 | Cost multiplier when changing contracts |
| `belief_update_rate` | float | 0.3 | How fast agents update pool quality beliefs |

Agent type preference weights control how likely each type is to sign each contract:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `honest_truthful_preference` | 0.8 | Honest agents' preference for truthful auction |
| `honest_fair_preference` | 0.6 | Honest agents' preference for fair division |
| `opportunistic_truthful_preference` | 0.3 | Opportunistic agents' preference for truthful auction |
| `opportunistic_fair_preference` | 0.4 | Opportunistic agents' preference for fair division |
| `adversarial_truthful_preference` | 0.1 | Adversarial agents' preference for truthful auction |
| `adversarial_fair_preference` | 0.15 | Adversarial agents' preference for fair division |

### Routing Logic

- Both agents in the same contract: use that contract
- Agents in different contracts: use default market (no protections granted to non-signers)
- Either agent has no membership: use default market

## BehavioralContract (ABC Formalism)

Implements the ABC framework where a contract is `C = (P, I, G, R)`:

- **P** — Preconditions: what must hold before an agent enters
- **I** — Invariants: runtime checks that must hold throughout
- **G** — Governance: execution, audit, and penalty logic (the wrapped `Contract`)
- **R** — Recovery: what happens when invariants are violated

`BehavioralContract` composes with (not subclasses) an existing `Contract`.

```python
from swarm.contracts import (
    BehavioralContract,
    TruthfulAuctionContract,
    Precondition,
    InvariantCheck,
    RecoveryPolicy,
)
from swarm.contracts.behavioral import (
    min_resources,
    min_trust_score,
    p_in_bounds,
    max_drift_rate,
    default_recovery,
)

# Wrap an existing contract with ABC layers
abc = BehavioralContract(
    governance=TruthfulAuctionContract(),
    preconditions=[
        min_resources(10.0),
        min_trust_score(0.3),
    ],
    invariants=[
        p_in_bounds(0.2, 1.0),
        max_drift_rate(0.15),
    ],
    recovery=default_recovery(),
)

# Check if an agent can enter
passed, failures = abc.check_preconditions(agent_state)

# Execute with invariant checking
modified = abc.execute(interaction)  # raises PermissionError if expelled

# Inspect violations
violations = abc.get_violations(agent_id="agent_1")

# Check expulsion
if abc.is_expelled("agent_1"):
    print("Agent was expelled due to invariant violations")
```

### Preconditions

Built-in precondition factories:

| Factory | Description |
|---------|-------------|
| `min_resources(threshold)` | Agent resources >= threshold |
| `min_trust_score(threshold)` | Agent reputation >= threshold |

Custom preconditions:

```python
Precondition(
    name="custom_check",
    check=lambda agent: agent.resources > 5 and agent.reputation > 0.5,
    description="Agent must have resources > 5 and reputation > 0.5",
)
```

### Invariants

Built-in invariant factories:

| Factory | Severity | Description |
|---------|----------|-------------|
| `p_in_bounds(low, high)` | 0.8 | p must stay within [low, high] |
| `max_drift_rate(threshold)` | 1.0 | Behavioral drift D* must stay below threshold |

Custom invariants:

```python
InvariantCheck(
    name="no_exploitation",
    check=lambda interaction: interaction.p > 0.3,
    severity=0.7,  # 0.0 to 1.0
    description="p must stay above exploitation threshold",
)
```

### Recovery Policy

The default escalating recovery maps violation severity to actions:

| Severity Threshold | Action | Effect |
|---------------------|--------|--------|
| >= 0.3 | `penalty` | Multiplied cost (1.5x) |
| >= 0.6 | `tier_downgrade` | Move to lower contract tier |
| >= 0.9 | `expel` | Permanently blocked from contract |

Only `expel` is enforced automatically by `BehavioralContract`. Other actions are tracked in the violation list for callers to act on.

## DriftDetector

Tracks behavioral drift D* across sessions. Detects agents whose behavior changes after building trust.

Uses a sliding window of p values per agent:

```
D* = mean(baseline_window) - mean(recent_window)
```

Positive D* indicates degradation (recent behavior worse than baseline). Only positive drift exceeding `drift_threshold` triggers a flag.

```python
from swarm.contracts import DriftDetector

detector = DriftDetector(
    window_size=20,       # Recent interactions for drift calculation
    baseline_size=50,     # Initial interactions for baseline
    drift_threshold=0.15, # D* above this flags the agent
)

# Record interactions
for p_value in agent_p_values:
    drift = detector.record("agent_1", p_value)
    # Returns D* if enough data, else None

# Check status
if detector.is_flagged("agent_1"):
    drift_rate = detector.get_drift("agent_1")
    print(f"Agent flagged with D* = {drift_rate:.3f}")

# All flagged agents
flagged = detector.get_flagged_agents()
# {"agent_1": 0.23, "agent_2": 0.18}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | 20 | Number of recent interactions for drift calculation |
| `baseline_size` | int | 50 | Number of initial interactions for baseline |
| `drift_threshold` | float | 0.15 | D* above this flags the agent |

### Key Properties

- Needs `baseline_size + window_size` observations before computing drift
- Baseline is computed once from the first `baseline_size` entries
- Always-adversarial agents are invisible to drift detection (no change from baseline) — use static screening to catch them
- History per agent is capped at `baseline_size + window_size` entries

## Pipeline Compositionality

Compute end-to-end compliance bounds for multi-stage pipelines.

Given per-stage contracts with compliance probability `p_i` and invariant violation bound `δ_i`:

- `p_pipeline = ∏(p_i)` — all stages must comply
- `δ_pipeline = min(1, Σ(δ_i))` — union bound on violations

Both degrade with more stages, arguing for fewer, stronger governance contracts.

```python
from swarm.contracts import StageGuarantee, compute_pipeline_bound

stages = [
    StageGuarantee(stage_name="screening", p=0.95, delta=0.02),
    StageGuarantee(stage_name="execution", p=0.90, delta=0.05),
    StageGuarantee(stage_name="audit", p=0.98, delta=0.01),
]

bound = compute_pipeline_bound(stages)
print(f"Pipeline compliance: {bound.p_pipeline:.4f}")  # 0.8379
print(f"Violation bound: {bound.delta_pipeline:.4f}")   # 0.08
```

### With Drift Degradation

Account for behavioral drift over time: `p_i(t) = clamp(p_i(0) - D* × t, 0, 1)`.

```python
from swarm.contracts import compute_pipeline_bound_with_drift

bound_t = compute_pipeline_bound_with_drift(
    stages=stages,
    drift_rate=0.01,   # D* per step
    time_steps=10,     # Project 10 steps ahead
)

print(f"Pipeline compliance at t=10: {bound_t.p_pipeline:.4f}")
# Each stage's p is reduced by 0.01 * 10 = 0.10
```

### PipelineBound

| Field | Type | Description |
|-------|------|-------------|
| `p_pipeline` | float | Product of per-stage compliance probabilities |
| `delta_pipeline` | float | Union bound on invariant violations |
| `n_stages` | int | Number of stages |
| `stage_details` | list | Per-stage breakdown |

## See also

- [Governance API](governance.md) — GovernanceEngine, levers, and configuration
- [Core API](core.md) — ProxyComputer, SoftPayoffEngine, SoftInteraction
