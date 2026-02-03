# Distributional AGI Safety Sandbox

## Soft Label Payoff & Metrics System

This project implements a simulation framework for studying distributional safety in multi-agent AI systems using soft (probabilistic) labels rather than hard binary classifications.

## Overview

The system provides:
1. **Core data models** for interactions with probabilistic labels
2. **Downstream proxy computation** (v_hat → p)
3. **Soft payoff engine** (reformulated A3b)
4. **Metrics system** (toxicity, conditional loss, spread, quality gap)
5. **Basic event logging infrastructure**

## Installation

```bash
# Install base dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with analysis tools (pandas, matplotlib)
pip install -e ".[analysis]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

```python
from src.models.interaction import SoftInteraction, InteractionType
from src.core.proxy import ProxyComputer, ProxyObservables
from src.core.payoff import SoftPayoffEngine
from src.metrics.reporters import MetricsReporter

# Create an interaction with observables
obs = ProxyObservables(
    task_progress_delta=0.7,
    rework_count=1,
    verifier_rejections=0,
    counterparty_engagement_delta=0.4,
)

# Compute soft labels
proxy = ProxyComputer()
v_hat, p = proxy.compute_labels(obs)

# Create interaction
interaction = SoftInteraction(
    initiator="agent_1",
    counterparty="agent_2",
    interaction_type=InteractionType.COLLABORATION,
    accepted=True,
    v_hat=v_hat,
    p=p,
)

# Compute payoffs
engine = SoftPayoffEngine()
payoff_a = engine.payoff_initiator(interaction)
payoff_b = engine.payoff_counterparty(interaction)

# Compute metrics on a batch
from tests.fixtures.interactions import generate_mixed_batch

interactions = generate_mixed_batch(count=100, toxic_fraction=0.3)
reporter = MetricsReporter()
summary = reporter.summary(interactions)

print(reporter.format_report(interactions, verbose=True))
```

## Directory Structure

```
distributional-agi-safety/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── interaction.py      # SoftInteraction, InteractionType
│   │   ├── agent.py            # AgentType, AgentState
│   │   └── events.py           # Event log schema
│   ├── core/
│   │   ├── __init__.py
│   │   ├── payoff.py           # SoftPayoffEngine
│   │   ├── proxy.py            # ProxyComputer (v_hat computation)
│   │   └── sigmoid.py          # Calibrated sigmoid utilities
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── soft_metrics.py     # SoftMetrics class
│   │   └── reporters.py        # Dual reporting (soft + hard)
│   └── logging/
│       ├── __init__.py
│       └── event_log.py        # Append-only event logger
├── tests/
│   ├── __init__.py
│   ├── test_payoff.py
│   ├── test_proxy.py
│   ├── test_metrics.py
│   └── fixtures/
│       └── interactions.py     # Test data generators
├── pyproject.toml
└── README.md
```

## Core Concepts

### Soft Labels

Instead of binary labels (good/bad), interactions carry a probability `p = P(v = +1)` representing the likelihood of a beneficial outcome. This is computed from downstream observables via:

1. **Proxy signals** are combined into a raw score `v_hat ∈ [-1, +1]`
2. **Calibrated sigmoid** converts to probability: `p = 1 / (1 + exp(-k * v_hat))`

### Proxy Computation

The `ProxyComputer` combines multiple observable signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| task_progress | 0.4 | Forward progress on task |
| rework_penalty | 0.2 | Quality signal from rework cycles |
| verifier_penalty | 0.2 | Safety signal from rejections |
| engagement_signal | 0.2 | Counterparty response |

### Soft Payoff Engine (Reformulated A3b)

Payoffs are computed using expected values:

```
π_a = θ * S_soft - τ - c_a - ρ_a * E_soft + w_rep * r_a
π_b = (1-θ) * S_soft + τ - c_b - ρ_b * E_soft + w_rep * r_b
```

Where:
- `S_soft = p * s_plus - (1-p) * s_minus` (expected surplus)
- `E_soft = (1-p) * h` (expected harm externality)
- `τ` = transfer from initiator to counterparty
- `c_a, c_b` = governance costs
- `r_a, r_b` = reputation changes
- `θ` = surplus split (initiator share)
- `ρ_a, ρ_b` = externality internalization

**Default Parameters:**
- s_plus = 2.0, s_minus = 1.0
- h = 2.0 (harm magnitude)
- θ = 0.5 (equal split)
- ρ_a = ρ_b = 0.0 (no externality internalization)
- w_rep = 1.0 (reputation weight)
- sigmoid_k = 2.0 (calibration sharpness)

### Metrics System

#### Soft Metrics (Probabilistic)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Toxicity rate | `E[1-p \| accepted]` | Expected harm among accepted |
| Conditional loss | `E[π_a \| accepted] - E[π_a]` | Selection effect on payoffs |
| Spread | `(s+ + s-) * (E[p] - E[p \| accepted])` | Quality filtering effectiveness |
| Quality gap | `E[p \| accepted] - E[p \| rejected]` | Adverse selection indicator |

#### Hard Metrics (Threshold-based)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Toxicity rate | `P(p < 0.5 \| accepted)` | Fraction classified as harmful |
| Acceptance rates | By quality tier | Participation patterns |

### Event Logging

Append-only JSONL logger for simulation replay:

```python
from src.logging.event_log import EventLog
from src.models.events import Event, EventType

log = EventLog(Path("simulation.jsonl"))
log.append(Event(
    event_type=EventType.INTERACTION_PROPOSED,
    interaction_id="abc123",
    initiator_id="agent_1",
    counterparty_id="agent_2",
    payload={"v_hat": 0.5, "p": 0.73},
))

# Replay events
for event in log.replay():
    print(event.event_type, event.timestamp)

# Reconstruct interactions
interactions = log.to_interactions()
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_payoff.py -v
```

## Test Fixtures

Generate test data with various patterns:

```python
from tests.fixtures.interactions import (
    generate_benign_batch,      # High quality, positive outcomes
    generate_toxic_batch,       # Low quality, exploitation patterns
    generate_mixed_batch,       # Realistic distribution
    generate_adversarial_scenario,  # Coordinated attack pattern
    generate_uncertain_batch,   # Labels near p=0.5
)

# Generate 100 interactions, 30% toxic
interactions = generate_mixed_batch(count=100, toxic_fraction=0.3, seed=42)
```

## Key Design Decisions

1. **Soft over hard labels**: Preserves uncertainty information and enables more nuanced analysis

2. **Downstream proxies**: Labels are computed from observable outcomes rather than claimed intentions

3. **Externality modeling**: Harm to the ecosystem is explicitly modeled and can be internalized via ρ parameters

4. **Dual reporting**: Both soft and hard metrics are computed for comparison

5. **Append-only logging**: Events are immutable for auditability and replay

## Future Extensions

- Governance lever integration (taxes, decay)
- Feed/orchestrator integration
- Scenario runner for batch simulations
- Dashboard visualization
- Multi-agent coordination protocols

## Inspired By

This project is inspired by research on multi-agent safety and market dynamics:

- [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.141)

## Dependencies

**Core:**
- numpy >= 1.24
- pydantic >= 2.0

**Development:**
- pytest >= 7.0
- pytest-cov
- mypy
- ruff

**Analysis:**
- pandas >= 2.0
- matplotlib >= 3.7
- seaborn >= 0.12
