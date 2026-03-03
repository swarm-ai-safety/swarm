---
description: "The core module provides the fundamental building blocks of SWARM."
---

# Core API

The core module provides the fundamental building blocks of SWARM.

## ProxyComputer

Converts observable signals to [soft probabilistic labels](../concepts/soft-labels.md).

::: swarm.core.proxy.ProxyComputer
    options:
      show_root_heading: true
      members:
        - compute_labels

### Usage

```python
from swarm.core.proxy import ProxyComputer, ProxyObservables

proxy = ProxyComputer()

obs = ProxyObservables(
    task_progress_delta=0.7,
    rework_count=1,
    verifier_rejections=0,
    counterparty_engagement_delta=0.4,
)

v_hat, p = proxy.compute_labels(obs)
```

### ProxyObservables

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `task_progress_delta` | float | [-1, 1] | Progress on task |
| `rework_count` | int | [0, ∞) | Number of rework cycles |
| `verifier_rejections` | int | [0, ∞) | Safety rejections |
| `counterparty_engagement_delta` | float | [-1, 1] | Engagement change |

### ProxyWeights

Default weights for combining signals:

| Signal | Weight |
|--------|--------|
| task_progress | 0.4 |
| rework_penalty | 0.2 |
| verifier_penalty | 0.2 |
| engagement | 0.2 |

## SoftPayoffEngine

Computes payoffs using soft probabilistic labels.

::: swarm.core.payoff.SoftPayoffEngine
    options:
      show_root_heading: true
      members:
        - payoff_initiator
        - payoff_counterparty

### Usage

```python
from swarm.core.payoff import SoftPayoffEngine, PayoffConfig
from swarm.models.interaction import SoftInteraction

config = PayoffConfig(
    s_plus=1.0,
    s_minus=0.5,
    h=0.3,
    theta=0.5,
)

engine = SoftPayoffEngine(config)

interaction = SoftInteraction(
    initiator="a",
    counterparty="b",
    accepted=True,
    v_hat=0.5,
    p=0.8,
)

payoff_a = engine.payoff_initiator(interaction)
payoff_b = engine.payoff_counterparty(interaction)
```

### PayoffConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `s_plus` | 1.0 | Surplus if beneficial |
| `s_minus` | 0.5 | Loss if harmful |
| `h` | 0.3 | External harm |
| `theta` | 0.5 | Initiator's share |
| `tau` | 0.0 | Transfer |
| `w_rep` | 0.1 | Reputation weight |
| `rho_a` | 0.1 | Initiator externality internalization |
| `rho_b` | 0.1 | Counterparty externality internalization |

## Orchestrator

Runs multi-agent simulations.

::: swarm.core.orchestrator.Orchestrator
    options:
      show_root_heading: true
      members:
        - register_agent
        - run
        - from_scenario

### Usage

```python
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    n_epochs=10,
    steps_per_epoch=10,
    seed=42,
)

orchestrator = Orchestrator(config=config)
orchestrator.register_agent(agent1)
orchestrator.register_agent(agent2)

metrics = orchestrator.run()
```

### OrchestratorConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 10 | Number of epochs |
| `steps_per_epoch` | 10 | Steps per epoch |
| `seed` | None | Random seed |
| `async_mode` | False | Async agent execution |
| `governance` | None | Governance configuration |
| `payoff` | None | Payoff configuration |

## Sigmoid Functions

::: swarm.core.sigmoid
    options:
      members:
        - calibrated_sigmoid
        - inverse_sigmoid

### Usage

```python
from swarm.core.sigmoid import calibrated_sigmoid, inverse_sigmoid

# v_hat to probability
p = calibrated_sigmoid(v_hat=0.5, k=3.0)

# probability to v_hat
v_hat = inverse_sigmoid(p=0.8, k=3.0)
```
