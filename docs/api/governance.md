---
description: "Governance mechanisms for controlling multi-agent system behavior."
---

# Governance API

[Governance mechanisms](../concepts/governance.md) for controlling multi-agent system behavior.

## GovernanceEngine

Main governance controller.

::: swarm.governance.engine.GovernanceEngine
    options:
      show_root_heading: true

### Usage

```python
from swarm.governance import GovernanceEngine, GovernanceConfig

config = GovernanceConfig(
    transaction_tax=0.02,
    reputation_decay=0.1,
    circuit_breaker_threshold=0.3,
)

engine = GovernanceEngine(config)

# Check if agent can act
if engine.can_agent_act(agent_id):
    # Process action
    pass

# Record interaction
engine.on_interaction(interaction)

# Get governance costs
costs = engine.compute_costs(interaction)
```

## GovernanceConfig

Configuration for all governance levers.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `transaction_tax` | float | 0.0 | Tax per interaction |
| `reputation_decay` | float | 0.0 | Decay rate per epoch |
| `initial_reputation` | float | 1.0 | Starting reputation |
| `circuit_breaker_threshold` | float | 1.0 | Toxicity threshold |
| `circuit_breaker_window` | int | 10 | Lookback window |
| `circuit_breaker_cooldown` | int | 5 | Freeze duration |
| `audit_probability` | float | 0.0 | Audit frequency |
| `audit_penalty` | float | 0.0 | Failed audit cost |
| `staking_requirement` | float | 0.0 | Minimum stake |
| `stake_slash_rate` | float | 0.0 | Slash fraction |
| `collusion_detection` | bool | False | Enable detection |
| `collusion_threshold` | float | 0.8 | Correlation threshold |

## Security YAML Keys

Security detection can be configured via scenario YAML using the `security_*`
governance keys below:

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

## Individual Levers

### TransactionTax

```python
from swarm.governance.levers import TransactionTax

tax = TransactionTax(rate=0.02)
cost = tax.compute(interaction)
```

### ReputationDecay

```python
from swarm.governance.levers import ReputationDecay

decay = ReputationDecay(rate=0.1)
new_rep = decay.apply(current_rep, epoch_delta=1)
```

### CircuitBreaker

```python
from swarm.governance.levers import CircuitBreaker

breaker = CircuitBreaker(
    threshold=0.3,
    window=10,
    cooldown=5,
)

# Check status
if breaker.is_frozen(agent_id):
    return  # Agent cannot act

# Record interaction
breaker.record(agent_id, toxicity=0.4)

# Check if triggered
if breaker.should_freeze(agent_id):
    breaker.freeze(agent_id)
```

### RandomAudit

```python
from swarm.governance.levers import RandomAudit

audit = RandomAudit(probability=0.05, penalty=0.5)

if audit.should_audit():
    result = audit.execute(interaction)
    if not result.passed:
        apply_penalty(interaction.initiator, audit.penalty)
```

### Staking

```python
from swarm.governance.levers import Staking

staking = Staking(
    requirement=10.0,
    slash_rate=0.1,
)

# Check eligibility
if not staking.can_participate(agent_id):
    return  # Insufficient stake

# Slash on bad behavior
staking.slash(agent_id, reason="failed_audit")
```

## Collusion Detection

```python
from swarm.governance.collusion import CollusionDetector

detector = CollusionDetector(
    threshold=0.8,
    window=20,
)

# Record interactions
detector.record(agent_a, agent_b, interaction)

# Check for collusion
colluding_pairs = detector.detect()
for pair, score in colluding_pairs:
    print(f"Potential collusion: {pair} (score: {score:.2f})")
```

## Hooks

Governance integrates with the orchestrator via hooks:

```python
orchestrator = Orchestrator(
    config=config,
    governance=governance_engine,
)

# Hooks are called automatically:
# - on_epoch_start
# - on_interaction
# - on_epoch_end
```

## Custom Levers

Create custom governance mechanisms:

```python
from swarm.governance.levers import GovernanceLever

class CustomLever(GovernanceLever):
    def __init__(self, param: float):
        self.param = param

    def compute_cost(self, interaction) -> float:
        # Custom cost computation
        return interaction.p * self.param

    def should_block(self, agent_id: str) -> bool:
        # Custom blocking logic
        return False
```
