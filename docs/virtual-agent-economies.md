# Virtual Agent Economies

Components inspired by the "Virtual Agent Economies" paper (Tomasev et al., arXiv 2509.10147), implementing concepts from the paper that extend the simulation framework with fair allocation, collective coordination, high-frequency dynamics, boundary permeability, and identity infrastructure.

## Table of Contents

- [Dworkin-Style Auctions](#dworkin-style-auctions)
- [Mission Economies](#mission-economies)
- [High-Frequency Negotiation](#high-frequency-negotiation)
- [Permeability Model](#permeability-model)
- [Identity and Trust Infrastructure](#identity-and-trust-infrastructure)
- [Further Reading](#further-reading)

---

## Dworkin-Style Auctions

Fair resource allocation using auction mechanisms inspired by Ronald Dworkin's approach to distributive justice. Agents start with equal token endowments and bid on resource bundles. The mechanism uses tatonnement (iterative price adjustment) to find market-clearing prices, then verifies envy-freeness.

**Source:** `src/env/auction.py`

### How It Works

1. All agents receive equal token endowments
2. Agents submit bids expressing resource valuations
3. Tatonnement adjusts prices proportional to excess demand until convergence
4. Allocations are normalized so demand does not exceed supply
5. Envy-freeness is verified: no agent prefers another's bundle at clearing prices

### Quick Start

```python
from src.env.auction import DworkinAuction, AuctionConfig, AuctionBid

# Configure the auction
config = AuctionConfig(
    initial_endowment=100.0,
    max_rounds=50,
    price_adjustment_rate=0.1,
    convergence_tolerance=0.01,
    envy_tolerance=0.05,
)

auction = DworkinAuction(config)

# Define agent bids
bids = {
    "agent_1": AuctionBid(
        agent_id="agent_1",
        valuations={"compute": 2.0, "data": 1.0, "bandwidth": 0.5},
        budget=100.0,
    ),
    "agent_2": AuctionBid(
        agent_id="agent_2",
        valuations={"compute": 0.5, "data": 2.0, "bandwidth": 1.5},
        budget=100.0,
    ),
}

# Available resources
resources = {"compute": 10.0, "data": 20.0, "bandwidth": 15.0}

# Run the auction
result = auction.run_auction(bids, resources)

print(f"Converged: {result.converged} in {result.rounds_to_converge} rounds")
print(f"Envy-free: {result.is_envy_free}")
print(f"Total utility: {result.total_utility:.2f}")

# Check allocation fairness
gini = auction.compute_gini_coefficient(result.allocations)
print(f"Utility Gini: {gini:.3f}")  # 0 = perfect equality
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_endowment` | 100.0 | Equal token budget per agent |
| `max_rounds` | 50 | Maximum tatonnement iterations |
| `price_adjustment_rate` | 0.1 | Price change speed (0, 1] |
| `convergence_tolerance` | 0.01 | Max excess demand for convergence |
| `envy_tolerance` | 0.05 | Utility gap tolerance for envy-freeness |

### Connection to Soft Labels

Agent effective endowments can be modulated by reputation (derived from average `p` history), linking allocation fairness to the quality signal pipeline.

---

## Mission Economies

Collective goal coordination where agents align around shared societal missions with measurable criteria. Contributions are evaluated using the soft-label quality pipeline, and rewards are distributed proportional to individual contributions.

**Source:** `src/env/mission.py`

### How It Works

1. An agent proposes a mission with objectives, reward pool, and deadline
2. Other agents join the mission
3. Agents contribute interactions toward mission objectives
4. At deadline (or when objectives are met), the mission is evaluated
5. Rewards are distributed based on the configured strategy

### Quick Start

```python
from src.env.mission import MissionEconomy, MissionConfig, MissionObjective

config = MissionConfig(
    enabled=True,
    min_participants=2,
    max_active_missions=5,
    reward_distribution="proportional",  # "equal", "proportional", or "shapley"
)

economy = MissionEconomy(config)

# Propose a mission
objectives = [
    MissionObjective(
        description="Maintain high quality interactions",
        target_metric="avg_p",
        target_value=0.7,
        weight=1.0,
    ),
    MissionObjective(
        description="Reach interaction volume target",
        target_metric="total_count",
        target_value=50.0,
        weight=0.5,
    ),
]

mission = economy.propose_mission(
    coordinator_id="agent_1",
    name="Quality Improvement Initiative",
    objectives=objectives,
    reward_pool=100.0,
    deadline_epoch=20,
    current_epoch=0,
)

# Other agents join
economy.join_mission("agent_2", mission.mission_id)
economy.join_mission("agent_3", mission.mission_id)

# Record contributions (interactions submitted toward mission goals)
economy.record_contribution("agent_1", mission.mission_id, interaction)

# Evaluate mission
result = economy.evaluate_mission(mission.mission_id, all_interactions, current_epoch=20)
print(f"Score: {result['mission_score']:.2f}, Status: {result['status']}")

# Distribute rewards
rewards = economy.distribute_rewards(mission.mission_id, all_interactions)
for agent_id, amount in rewards.items():
    print(f"  {agent_id}: {amount:.2f}")

# Check for free-riding
gini = economy.free_rider_index(mission.mission_id)
print(f"Free-rider index (Gini): {gini:.3f}")
```

### Reward Distribution Strategies

| Strategy | Description |
|----------|-------------|
| **equal** | Equal split among all contributors |
| **proportional** | Weighted by contribution count and average `p` (quality-weighted) |
| **shapley** | Approximate Shapley values based on marginal quality improvement + volume share |

### Supported Objective Metrics

| Metric | Description |
|--------|-------------|
| `avg_p` | Average `p` across contributed interactions |
| `min_p` | Minimum `p` (worst-case quality) |
| `total_count` | Number of contributed interactions |
| `acceptance_rate` | Fraction of interactions that were accepted |
| `total_welfare` | Sum of expected surplus: `p * 2.0 - (1-p) * 1.0` |

---

## High-Frequency Negotiation

Models speed-based market dynamics where agents submit orders at high rates, with risk of flash crashes (sudden correlated quality collapses). Includes a flash crash detector and circuit breaker mechanism.

**Source:** `src/env/hfn.py`

### How It Works

1. Agents submit bid/ask/cancel orders to an order book
2. Orders are sorted by price priority and effective arrival time (including latency noise)
3. At batch intervals, matching bids and asks are cleared at midpoint price
4. A flash crash detector monitors price drops within a rolling window
5. Circuit breaker halts the market if a crash is detected

### Quick Start

```python
from src.env.hfn import HFNEngine, HFNConfig, HFNOrder

config = HFNConfig(
    tick_duration_ms=100.0,
    max_orders_per_tick=10,
    latency_noise_ms=10.0,
    batch_interval_ticks=5,
    halt_duration_ticks=20,
)

engine = HFNEngine(config, seed=42)

# Submit orders
engine.submit_order(HFNOrder(
    agent_id="agent_1",
    order_type="bid",
    resource_type="compute",
    quantity=5.0,
    price=1.2,
))

engine.submit_order(HFNOrder(
    agent_id="agent_2",
    order_type="ask",
    resource_type="compute",
    quantity=5.0,
    price=1.1,
))

# Process a tick (matches orders at batch intervals)
tick = engine.process_tick()
print(f"Tick {tick.tick_number}: price={tick.market_price:.2f}, "
      f"spread={tick.bid_ask_spread:.2f}, halted={tick.halted}")

# Check for flash crashes
crashes = engine.get_crash_history()
for crash in crashes:
    print(f"Flash crash at tick {crash.start_tick}: "
          f"drop={crash.price_drop_pct:.1%}, severity={crash.severity:.2f}")

# Fairness metric
gini = engine.speed_advantage_gini()
print(f"Speed advantage Gini: {gini:.3f}")
```

### Flash Crash Detection

The `FlashCrashDetector` monitors price changes within a rolling window:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `price_drop_threshold` | 0.1 | Fractional drop to trigger (10%) |
| `window_ticks` | 10 | Rolling window size |
| `volume_spike_factor` | 3.0 | Volume multiplier for detection |

When a crash is detected, the market halts via circuit breaker for `halt_duration_ticks`.

### HFN Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tick_duration_ms` | 100.0 | Duration of one tick |
| `max_orders_per_tick` | 10 | Per-agent rate limit |
| `latency_noise_ms` | 10.0 | Random latency added to orders |
| `priority_by_speed` | True | Faster orders get priority |
| `batch_interval_ticks` | 5 | Ticks between batch clearing |
| `halt_duration_ticks` | 20 | Market halt duration after crash |

### Connection to Soft Labels

Every executed trade generates an interaction with associated `p` values. Flash crashes manifest as sudden drops in the average `p` of accepted interactions.

---

## Permeability Model

Models sandbox boundaries as semi-permeable membranes with parameterized permeability (0 = fully sealed, 1 = fully open). Includes contagion modeling for how harmful interactions inside the sandbox propagate to the external world.

**Source:** `src/boundaries/permeability.py`

### How It Works

1. Effective permeability is computed from base level, threat, and agent trust
2. Individual flows are probabilistically allowed or blocked based on permeability and interaction quality
3. Contagion probability links harm level to boundary crossing: `prob = contagion_rate * (1-p) * permeability`
4. Spillover events track harm that escapes the sandbox, with external amplification

### Quick Start

```python
from src.boundaries.permeability import PermeabilityModel, PermeabilityConfig

config = PermeabilityConfig(
    base_permeability=0.5,
    contagion_rate=0.05,
    spillover_amplification=1.5,
    adaptive=True,
    threat_sensitivity=1.0,
)

model = PermeabilityModel(config, seed=42)

# Compute effective permeability under threat
perm = model.compute_effective_permeability(
    threat_level=0.3,
    agent_trust=0.8,
)
print(f"Effective permeability: {perm:.3f}")

# Simulate spillover from a batch of interactions
spillovers = model.simulate_spillover(interactions)
print(f"Spillovers: {len(spillovers)}, "
      f"blocked: {sum(1 for s in spillovers if s.blocked)}")

# Metrics
print(f"Containment rate: {model.containment_rate():.1%}")
print(f"Cumulative external harm: {model.state.cumulative_external_harm:.2f}")

# Find optimal permeability
optimal = model.optimal_permeability(
    interactions,
    external_harm_weight=1.0,
    n_samples=10,
)
print(f"Optimal permeability: {optimal:.2f}")
```

### Adaptive Permeability

When `adaptive=True`, effective permeability adjusts dynamically:

```
effective = base_permeability * threat_factor * trust_factor
```

Where:
- `threat_factor = 1.0 - (threat_level * threat_sensitivity)`, clamped to [0, 1]
- `trust_factor = 0.5 + 0.5 * agent_trust`

Higher threat tightens the boundary; higher trust loosens it.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_permeability` | 0.5 | Base permeability [0, 1] |
| `information_decay` | 0.1 | Quality degradation on crossing |
| `contagion_rate` | 0.05 | Base contagion probability |
| `spillover_amplification` | 1.5 | Harm amplification factor |
| `adaptive` | True | Enable adaptive permeability |
| `threat_sensitivity` | 1.0 | How strongly threat reduces permeability |

### Connection to Soft Labels

Contagion probability is directly proportional to `(1-p) * permeability`, linking boundary dynamics to the quality signal pipeline. Low-quality interactions (low `p`) are more likely to cause spillover harm.

---

## Identity and Trust Infrastructure

Implements verifiable credentials, Proof-of-Personhood, and Sybil detection for the simulation. These are abstracted versions of the cryptographic infrastructure proposed in the paper.

**Source:** `src/models/identity.py`, `src/governance/identity_lever.py`

### Components

| Component | Description |
|-----------|-------------|
| `VerifiableCredential` | Unforgeable claims with expiry and revocation |
| `AgentIdentity` | Extended identity with credentials and trust score |
| `CredentialIssuer` | System-level credential issuance and verification |
| `IdentityRegistry` | Identity management and Sybil detection |
| `SybilDetectionLever` | Governance lever for Sybil penalties |

### Quick Start

```python
from src.models.identity import (
    IdentityRegistry, IdentityConfig, CredentialIssuer,
)

# Configure identity infrastructure
config = IdentityConfig(
    identity_creation_cost=10.0,
    proof_of_personhood_required=False,
    credential_expiry_epochs=50,
    sybil_detection_enabled=True,
    max_identities_per_entity=1,
    behavioral_similarity_threshold=0.8,
)

registry = IdentityRegistry(config)

# Create identities
identity = registry.create_identity(
    "agent_1",
    entity_id="entity_a",
    proof_of_personhood=True,
    current_epoch=0,
)
print(f"Trust score: {identity.trust_score:.2f}")

# Issue credentials
issuer = CredentialIssuer(config)
cred = issuer.issue_reputation_credential("agent_1", reputation=0.85, current_epoch=5)
identity.credentials.append(cred)

# Recompute trust
score = identity.compute_trust_score(current_epoch=5)
print(f"Updated trust: {score:.2f}")

# Detect Sybil clusters
patterns = {
    "agent_1": {"target_a": 10, "target_b": 5},
    "agent_2": {"target_a": 10, "target_b": 5},  # Suspiciously similar
    "agent_3": {"target_c": 8, "target_d": 3},    # Different pattern
}
clusters = registry.detect_sybil_clusters(patterns)
print(f"Sybil clusters found: {len(clusters)}")
```

### Trust Score Computation

Trust is built from credentials and Proof-of-Personhood:

| Component | Score |
|-----------|-------|
| Base (new identity) | 0.3 |
| Proof-of-Personhood | +0.2 |
| Per valid credential | +0.1 (max 0.5 from credentials) |
| **Maximum** | **1.0** |

### Sybil Detection

Behavioral similarity analysis identifies clusters of agents that appear to be controlled by the same entity. The algorithm computes a combined Jaccard + cosine similarity score from interaction patterns:

1. **Jaccard similarity** of counterparty sets
2. **Cosine similarity** of normalized interaction frequency vectors
3. **Combined score** = average of the two

Agents above the `behavioral_similarity_threshold` are clustered together.

### Sybil Detection Governance Lever

The `SybilDetectionLever` integrates with the governance engine:

```python
from src.governance import GovernanceConfig

gov_config = GovernanceConfig(
    sybil_detection_enabled=True,
    sybil_similarity_threshold=0.8,
    sybil_penalty_multiplier=1.0,
    sybil_realtime_penalty=True,
    sybil_realtime_rate=0.1,
    sybil_max_cluster_size=3,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sybil_detection_enabled` | False | Enable Sybil detection |
| `sybil_similarity_threshold` | 0.8 | Behavioral similarity threshold |
| `sybil_penalty_multiplier` | 1.0 | Penalty severity multiplier |
| `sybil_realtime_penalty` | False | Penalize flagged pairs on interaction |
| `sybil_realtime_rate` | 0.1 | Per-interaction penalty amount |
| `sybil_max_cluster_size` | 3 | Block agents in clusters larger than this |

### Identity Event Types

| Event | Description |
|-------|-------------|
| `IDENTITY_CREATED` | New agent identity registered |
| `CREDENTIAL_ISSUED` | Credential issued to an agent |
| `SYBIL_DETECTED` | Sybil cluster detected |

---

## Further Reading

- Tomasev, N., Franklin, J., Leibo, J.Z., Jacobs, A.Z., Cunningham, T., Gabriel, I., & Osindero, S. (2025). *Virtual Agent Economies: A Framework for Multi-Agent System Governance*. arXiv:2509.10147. https://arxiv.org/abs/2509.10147
- Dworkin, R. (1981). *What is Equality? Part 2: Equality of Resources*. Philosophy & Public Affairs, 10(4), 283-345.
- Kyle, A.S., Obizhaeva, A.A., & Tuzun, T. (2017). *Flash Crashes and Market Microstructure*. Working Paper.
- Shapley, L.S. (1953). *A Value for n-Person Games*. Contributions to the Theory of Games, 2, 307-317.
