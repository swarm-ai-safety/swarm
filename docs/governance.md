# Governance

The governance module provides configurable levers that affect agent behavior and payoffs.

SWARM's governance design draws inspiration from behavioral ecology research on social insect colonies, which achieve coordination without central control through local interactions and threshold-based responses. See [Theoretical Foundations: Biological Foundations](research/theory.md#biological-foundations) for details on how ant colony task allocation maps to SWARM's distributed governance.

## Configuration

```python
from swarm.governance import GovernanceConfig, GovernanceEngine
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

gov_config = GovernanceConfig(
    # Transaction tax: 5% of interaction value, split 50/50
    # Tax base = max(expected_surplus, 0) + |tau|
    transaction_tax_rate=0.05,
    transaction_tax_split=0.5,

    # Reputation decays 10% each epoch (feeds back into observable quality)
    reputation_decay_rate=0.9,

    # Require minimum stake to participate
    staking_enabled=True,
    min_stake_to_participate=10.0,

    # Freeze agents with >70% toxicity after 3 violations
    circuit_breaker_enabled=True,
    freeze_threshold_toxicity=0.7,
    freeze_threshold_violations=3,
    freeze_duration_epochs=2,

    # Detect and penalize agents stuck in loops
    loop_detector_enabled=True,
    loop_detector_quality_threshold=0.4,
    loop_detector_penalty_multiplier=1.5,
    loop_detector_replan_threshold=2,
    loop_detector_freeze_threshold=4,

    # 10% chance of audit, penalty if p < 0.5
    audit_enabled=True,
    audit_probability=0.1,
    audit_threshold_p=0.5,
)

config = OrchestratorConfig(
    n_epochs=10,
    governance_config=gov_config,
)
orchestrator = Orchestrator(config=config)
```

## Governance Levers

| Lever | Effect | Hook |
|-------|--------|------|
| **Transaction Tax** | Tax on interaction value: `rate * (max(S_soft, 0) + \|tau\|)`, split between parties. Governance costs feed back into reputation, affecting future interaction quality. | on_interaction |
| **Reputation Decay** | `reputation *= decay_rate` each epoch. Reputation modulates observable signals (task_progress, engagement), creating a feedback loop to `p` and toxicity. | on_epoch_start |
| **Vote Normalization** | Diminishing influence as vote count increases | compute_vote_weight |
| **Staking** | Block actions if `resources < min_stake` | can_agent_act |
| **Circuit Breaker** | Freeze agent if `avg_toxicity > threshold` for N violations | on_interaction |
| **Loop Detector** | Detects repetitive behavior patterns (same counterparty, low quality, tool misuse, high rework). Triggers: (1) tool-budget penalties, (2) forced replan after threshold violations, (3) circuit breaker freeze after excessive violations. | on_interaction, on_epoch_start |
| **Random Audit** | Cost penalty `(threshold - p) * multiplier` plus reputation penalty if audited and `p < threshold`. Reputation penalty feeds back into observable quality. | on_interaction |
| **Collusion Detection** | Penalty based on collusion risk score from coordinated behavior | on_epoch_start, on_interaction |
| **Sybil Detection** | Detects and penalizes suspected Sybil agents via behavioral similarity | on_epoch_start, on_interaction, can_agent_act |

## Incoherence-Targeted Controls

In addition to baseline levers, the project includes replay-calibrated
incoherence controls focused on variance-dominated failures:

- self-ensemble
- incoherence breaker
- decomposition checkpoints
- dynamic friction

For deployment assumptions and caveats, see
`docs/transferability/incoherence_governance.md`.

## Integration Points

Governance hooks into the orchestrator at:
1. **Epoch start**: Reputation decay, unfreeze agents whose freeze duration expired
2. **Agent action**: Staking check blocks under-resourced agents
3. **Interaction completion**: Taxes, circuit breaker tracking, random audits

Costs are added to `interaction.c_a` and `interaction.c_b` before payoff computation. Governance costs also reduce the initiator's reputation delta, creating a feedback loop: higher costs -> slower reputation growth -> degraded observable signals -> lower `p` -> higher toxicity. This ensures governance levers (tax rate, audit probability, reputation decay) produce visible effects on both welfare and toxicity metrics in parameter sweeps.

## Collusion Detection

The collusion detection system identifies coordinated manipulation patterns among agents.

### Detection Signals

| Signal | Description | Threshold |
|--------|-------------|-----------|
| **Interaction Frequency** | Z-score of pair interaction count vs. population | > 2.0 |
| **Benefit Correlation** | Correlation between pair benefits across interactions | > 0.7 |
| **Acceptance Rate** | Fraction of mutually accepted interactions | > 0.8 |
| **Quality Asymmetry** | Difference in avg p (internal vs external) | > 0.2 |

### Collusion Quick Start

```python
from swarm.governance import GovernanceConfig, GovernanceEngine

config = GovernanceConfig(
    collusion_detection_enabled=True,
    collusion_frequency_threshold=2.0,
    collusion_score_threshold=0.5,
    collusion_penalty_multiplier=1.5,
    collusion_realtime_penalty=True,
)

# After simulation
report = orchestrator.get_collusion_report()
print(f"Ecosystem risk: {report.ecosystem_collusion_risk:.2f}")
print(f"Flagged pairs: {report.n_flagged_pairs}")
for pair in report.suspicious_pairs:
    print(f"  {pair.agent_a} <-> {pair.agent_b}: score={pair.collusion_score:.2f}")
```

### Detection Levels

**Pair-Level**: Analyzes interaction patterns between each agent pair:
- Frequency compared to population baseline
- Mutual benefit correlation
- Quality of interactions (avg p)
- Temporal clustering

**Group-Level**: Identifies clusters of suspicious pairs:
- Connected components of flagged pairs
- Internal vs external interaction rates
- Coordinated behavior patterns

### Collusion YAML Configuration

```yaml
governance:
  collusion_detection_enabled: true
  collusion_frequency_threshold: 2.0
  collusion_correlation_threshold: 0.7
  collusion_min_interactions: 3
  collusion_score_threshold: 0.5
  collusion_penalty_multiplier: 1.5
  collusion_realtime_penalty: true
  collusion_realtime_rate: 0.1
```

Run the collusion detection scenario:
```bash
python examples/run_scenario.py scenarios/collusion_detection.yaml
```

## Loop Detector

The loop detector identifies agents stuck in repetitive or unproductive behavior patterns and enforces three types of corrective actions: tool-budget penalties, forced replans, and circuit breaker freezes.

### Loop Types Detected

| Loop Type | Detection Criteria | Signal |
|-----------|-------------------|--------|
| **Repetition** | Same (counterparty, interaction_type) pattern repeats ≥60% of recent interactions | Stuck interacting with same agent |
| **Low Quality** | Average `p` falls below threshold over recent interactions | Persistently low success probability |
| **Tool Misuse** | Cumulative tool_misuse_flags exceed threshold | Repeated tool abuse |
| **Rework** | Cumulative rework_count exceeds threshold | Stuck redoing same work |

### Actions on Loop Detection

1. **Tool-Budget Penalty** (immediate): Cost multiplier applied to interaction
   ```
   penalty = loop_detector_penalty_multiplier * p (if accepted) or 0.5 (if not)
   ```

2. **Forced Replan** (after threshold violations): Signals agent to reset strategy
   - Triggered when `violations >= loop_detector_replan_threshold`
   - Only once per epoch per agent
   - Signaled via `force_replan` flag in LeverEffect details

3. **Circuit Breaker** (after excessive violations): Freeze agent temporarily
   - Triggered when `violations >= loop_detector_freeze_threshold`
   - Agent frozen for `loop_detector_freeze_duration` epochs
   - Automatically unfrozen at epoch start

### Loop Detector Quick Start

```python
from swarm.governance import GovernanceConfig, GovernanceEngine

config = GovernanceConfig(
    loop_detector_enabled=True,
    loop_detector_quality_threshold=0.4,  # avg p threshold for quality loops
    loop_detector_tool_misuse_threshold=5,  # cumulative misuse flags
    loop_detector_rework_threshold=10,  # cumulative rework count
    loop_detector_penalty_multiplier=1.5,  # cost multiplier
    loop_detector_replan_threshold=2,  # violations before forced replan
    loop_detector_freeze_threshold=4,  # violations before freeze
    loop_detector_freeze_duration=1,  # epochs to freeze
)

# Check loop status during simulation
engine = GovernanceEngine(config=config)
lever = engine.get_loop_detector_lever()
if lever:
    status = lever.get_loop_status("agent_1")
    print(f"Violations: {status['violations']}")
    print(f"Is frozen: {status['is_frozen']}")
    print(f"Replan count: {status['replan_count']}")
```

### Loop Detector YAML Configuration

```yaml
governance:
  loop_detector_enabled: true
  loop_detector_quality_threshold: 0.4
  loop_detector_tool_misuse_threshold: 5
  loop_detector_rework_threshold: 10
  loop_detector_penalty_multiplier: 1.5
  loop_detector_replan_threshold: 2
  loop_detector_freeze_threshold: 4
  loop_detector_freeze_duration: 1
```

### Use Cases

- **Autonomous agents**: Detect when agents get stuck in unproductive loops (e.g., Ralph loop agents repeatedly failing same task)
- **Tool-limited environments**: Penalize agents that repeatedly misuse restricted tools (e.g., Claude Code agents exceeding permissions)
- **Quality gates**: Catch agents that produce persistently low-quality outputs
- **Rework prevention**: Identify agents stuck in edit-review-edit cycles without progress

## Security Detection

Security detection flags injection, manipulation, laundering, and contagion
patterns in interaction chains. It can apply penalties, quarantine agents,
and optionally enforce real-time penalties on high-risk interactions.

### Security YAML Configuration

```yaml
governance:
  security_enabled: true
  security_injection_threshold: 0.35
  security_manipulation_threshold: 0.6
  security_laundering_trust_gap: 0.3
  security_contagion_velocity: 2.0
  security_min_chain_length: 3
  security_min_interactions: 5
  security_penalty_threshold: 0.35
  security_quarantine_threshold: 0.7
  security_penalty_multiplier: 1.2
  security_realtime_penalty: true
  security_realtime_threshold: 0.6
  security_realtime_rate: 0.2
  security_clear_history_on_epoch: false
```

## Sybil Detection

The Sybil detection system identifies agents that appear to be controlled by the same entity, using behavioral similarity analysis of interaction patterns. Inspired by the identity infrastructure proposed in [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147).

### How It Works

1. Interaction patterns are tracked: which agents interact with which counterparties, and how often
2. At each epoch, behavioral similarity is computed for all agent pairs using Jaccard similarity of counterparty sets combined with cosine similarity of normalized frequency vectors
3. Agents above the similarity threshold are clustered together as suspected Sybils
4. Flagged agents receive reputation and resource penalties

### Sybil Detection Quick Start

```python
from swarm.governance import GovernanceConfig, GovernanceEngine

config = GovernanceConfig(
    sybil_detection_enabled=True,
    sybil_similarity_threshold=0.8,
    sybil_penalty_multiplier=1.0,
    sybil_realtime_penalty=True,
    sybil_realtime_rate=0.1,
    sybil_max_cluster_size=3,
)

# After simulation
clusters = orchestrator.get_sybil_clusters()
flagged = orchestrator.get_flagged_sybil_agents()
print(f"Sybil clusters: {len(clusters)}")
print(f"Flagged agents: {len(flagged)}")
```

### Sybil YAML Configuration

```yaml
governance:
  sybil_detection_enabled: true
  sybil_similarity_threshold: 0.8
  sybil_penalty_multiplier: 1.0
  sybil_realtime_penalty: true
  sybil_realtime_rate: 0.1
  sybil_max_cluster_size: 3
```

### Enforcement Actions

| Action | Trigger | Effect |
|--------|---------|--------|
| **Reputation penalty** | Agent flagged in cluster | `-penalty_multiplier * 0.1` per epoch |
| **Resource penalty** | Agent flagged in cluster | `-penalty_multiplier * 1.0` per epoch |
| **Real-time penalty** | Both parties flagged, interact with each other | Extra cost of `realtime_rate` per interaction |
| **Action block** | Agent in cluster larger than `max_cluster_size` | Agent cannot act |

For full identity infrastructure documentation (verifiable credentials, Proof-of-Personhood, trust scores), see [Virtual Agent Economies](virtual-agent-economies.md#identity-and-trust-infrastructure).
