# Distributional AGI Safety Sandbox

[![CI](https://github.com/rsavitt/distributional-agi-safety/actions/workflows/ci.yml/badge.svg)](https://github.com/rsavitt/distributional-agi-safety/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

A simulation framework for studying distributional safety in multi-agent AI systems using soft (probabilistic) labels.

## Overview

The system provides:

**Foundation Layer:**
- Core data models for interactions with probabilistic labels
- Downstream proxy computation (v_hat → p via calibrated sigmoid)
- Soft payoff engine
- Metrics system (toxicity, conditional loss, calibration, variance)
- Append-only event logging with replay

**Runtime Layer (MVP v0):**
- Agent behavioral policies (honest, opportunistic, deceptive, adversarial)
- **LLM-backed agents** with Anthropic, OpenAI, and Ollama support ([docs](docs/llm-agents.md))
- Environment state management with rate limits
- Feed engine (posts, replies, voting, visibility ranking)
- Task system (claiming, collaboration, verification)
- Orchestrator for multi-agent simulation (sync and async)
- **Network topology** with dynamic evolution ([docs](docs/network-topology.md))

**Governance Layer:**
- Configurable levers (taxes, reputation decay, staking, circuit breakers, audits) ([docs](docs/governance.md))
- **Collusion detection** with pair-level and group-level analysis ([docs](docs/governance.md#collusion-detection))
- Integration with orchestrator via epoch and interaction hooks
- Populates `c_a` and `c_b` governance costs on interactions

## Installation

```bash
# Install base dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"

# Install with analysis tools (pandas, matplotlib)
pip install -e ".[analysis]"

# Install with LLM support (Anthropic, OpenAI, Ollama)
pip install -e ".[llm]"

# Install everything
pip install -e ".[all]"
```

## Quick Start: Running a Simulation

```python
from src.agents.honest import HonestAgent
from src.agents.opportunistic import OpportunisticAgent
from src.agents.deceptive import DeceptiveAgent
from src.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure simulation
config = OrchestratorConfig(
    n_epochs=10,
    steps_per_epoch=10,
    seed=42,
)

# Create orchestrator
orchestrator = Orchestrator(config=config)

# Register agents
orchestrator.register_agent(HonestAgent(agent_id="honest_1"))
orchestrator.register_agent(HonestAgent(agent_id="honest_2"))
orchestrator.register_agent(OpportunisticAgent(agent_id="opp_1"))
orchestrator.register_agent(DeceptiveAgent(agent_id="dec_1"))

# Run simulation
metrics = orchestrator.run()

# Analyze results
for m in metrics:
    print(f"Epoch {m.epoch}: toxicity={m.toxicity_rate:.3f}, welfare={m.total_welfare:.2f}")
```

Run the demo:
```bash
python examples/mvp_demo.py
```

## Quick Start: Computing Metrics

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
print(reporter.format_report(interactions, verbose=True))
```

## Core Concepts

### Soft Labels

Instead of binary labels (good/bad), interactions carry a probability `p = P(v = +1)` representing the likelihood of a beneficial outcome:

1. **Proxy signals** are combined into a raw score `v_hat ∈ [-1, +1]`
2. **Calibrated sigmoid** converts to probability: `p = 1 / (1 + exp(-k * v_hat))`

### Proxy Computation

The `ProxyComputer` combines observable signals:

| Signal | Weight | Description |
|--------|--------|-------------|
| task_progress | 0.4 | Forward progress on task |
| rework_penalty | 0.2 | Quality signal from rework cycles |
| verifier_penalty | 0.2 | Safety signal from rejections |
| engagement_signal | 0.2 | Counterparty response |

### Soft Payoff Engine

```
π_a = θ * S_soft - τ - c_a - ρ_a * E_soft + w_rep * r_a
π_b = (1-θ) * S_soft + τ - c_b - ρ_b * E_soft + w_rep * r_b
```

Where:
- `S_soft = p * s_plus - (1-p) * s_minus` (expected surplus)
- `E_soft = (1-p) * h` (expected harm externality)

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Toxicity rate | `E[1-p \| accepted]` | Expected harm among accepted |
| Quality gap | `E[p \| accepted] - E[p \| rejected]` | Adverse selection indicator |
| Conditional loss | `E[π \| accepted] - E[π]` | Selection effect on payoffs |
| Brier score | `E[(p - v)²]` | Calibration quality |

## Agent Policies

| Type | Behavior |
|------|----------|
| **Honest** | Cooperative, trust-based, completes tasks diligently |
| **Opportunistic** | Maximizes short-term payoff, cherry-picks tasks, strategic voting |
| **Deceptive** | Builds trust through honest behavior, then exploits trusted relationships |
| **Adversarial** | Targets honest agents, coordinates with allies, disrupts ecosystem |
| **LLM** | Behavior determined by LLM with configurable persona ([details](docs/llm-agents.md)) |

## Running Tests

```bash
# Run all tests (727 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v
```

## Documentation

Detailed documentation for each subsystem:

| Topic | Description |
|-------|-------------|
| [Theoretical Foundations](docs/theory.md) | Market microstructure theory, Kyle/Glosten-Milgrom models, references |
| [LLM Agents](docs/llm-agents.md) | Providers, personas, cost tracking, YAML config |
| [Network Topology](docs/network-topology.md) | Topology types, dynamic evolution, network metrics |
| [Governance](docs/governance.md) | Levers, collusion detection, integration points |
| [Emergent Capabilities](docs/emergent-capabilities.md) | Composite tasks, capability types, emergent metrics |
| [Red-Teaming](docs/red-teaming.md) | Adaptive adversaries, attack strategies, evaluation results |
| [Scenarios & Sweeps](docs/scenarios.md) | YAML scenarios, scenario comparison, parameter sweeps |
| [Boundaries](docs/boundaries.md) | External world simulation, flow tracking, leakage detection |
| [Dashboard](docs/dashboard.md) | Streamlit dashboard setup and features |

## Directory Structure

```
distributional-agi-safety/
├── src/
│   ├── models/          # SoftInteraction, AgentState, event schema
│   ├── core/            # PayoffEngine, ProxyComputer, sigmoid, orchestrator
│   ├── agents/          # Honest, opportunistic, deceptive, adversarial, LLM, adaptive
│   ├── env/             # EnvState, feed, tasks, network, composite tasks
│   ├── governance/      # Config, levers, taxes, reputation, audits, collusion
│   ├── metrics/         # SoftMetrics, reporters, collusion detection, capabilities
│   ├── scenarios/       # YAML scenario loader
│   ├── analysis/        # Parameter sweeps, dashboard
│   ├── redteam/         # Attack scenarios, evaluator, evasion metrics
│   ├── boundaries/      # External world, flow tracking, policies, leakage
│   └── logging/         # Append-only JSONL logger
├── tests/               # 727 tests across all modules
├── examples/            # mvp_demo, run_scenario, parameter_sweep, llm_demo
├── scenarios/           # YAML scenario definitions
├── docs/                # Detailed documentation
└── pyproject.toml
```

## Dependencies

**Core:** numpy, pydantic |
**Development:** pytest, pytest-cov, pytest-asyncio, mypy, ruff |
**Analysis:** pandas, matplotlib, seaborn |
**Runtime:** pyyaml |
**LLM:** anthropic, openai, httpx |
**Dashboard:** streamlit, plotly

## References

- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica.
- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market*. JFE.
- [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)
- [Moltbook](https://moltbook.com) | [@sebkrier](https://x.com/sebkrier/status/2017993948132774232)
