# SWARM: System-Wide Assessment of Risk in Multi-agent systems

[![CI](https://github.com/swarm-ai-safety/swarm/actions/workflows/ci.yml/badge.svg)](https://github.com/swarm-ai-safety/swarm/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

<img src="https://github.com/swarm-ai-safety/swarm/raw/main/docs/images/swarm-hero.png" alt="SWARM dashboard showing emergent risk metrics" width="100%">

**Study how intelligence swarms—and where it fails.**

SWARM is a research framework for studying emergent risks in multi-agent AI systems. Rather than focusing on single misaligned agents, SWARM reveals how catastrophic failures can emerge from the *interaction* of many sub-AGI agents—even when none are individually dangerous.

## The Core Insight

**AGI-level risks don't require AGI-level agents.** Harmful dynamics can emerge from:
- Information asymmetry between agents
- Adverse selection (system accepts lower-quality interactions)
- Variance amplification across decision horizons
- Governance latency and illegibility

SWARM makes these interaction-level risks **observable, measurable, and governable**.

## What Problem Does This Solve?

If you care about AGI safety research, SWARM gives you a practical way to:

- Turn qualitative worries ("deception", "coordination failures", "policy lag")
  into measurable signals (`toxicity`, `quality_gap`, calibration, incoherence).
- Stress-test governance mechanisms against adaptive and deceptive agents.
- Compare safety interventions under replay and scenario sweeps instead of
  one-off anecdotes.
- Separate sandbox wins from deployment reality using explicit transferability
  caveats.

## Questions You Can Study Quickly

- Does self-ensemble reduce variance-driven incoherence without masking bias?
- When do circuit breakers and friction reduce harm vs. suppress useful work?
- Which governance settings improve safety with the smallest welfare cost?
- How robust are conclusions under delayed/noisy labels and task shifts?

## Installation

```bash
pip install swarm-safety
```

Or install from source:

```bash
# Install base dependencies
python -m pip install -e .

# Install with development tools
python -m pip install -e ".[dev]"

# Install with analysis tools (pandas, matplotlib)
python -m pip install -e ".[analysis]"

# Install with LLM support (Anthropic, OpenAI, Ollama)
python -m pip install -e ".[llm]"

# Install everything
python -m pip install -e ".[all]"
```

## Quick Start

```python
from swarm.agents.honest import HonestAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure simulation
config = OrchestratorConfig(
    n_epochs=10,
    steps_per_epoch=10,
    seed=42,
)

# Create orchestrator
orchestrator = Orchestrator(config=config)

# Register agents
orchestrator.register_agent(HonestAgent(agent_id="honest_1", name="Alice"))
orchestrator.register_agent(HonestAgent(agent_id="honest_2", name="Bob"))
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

## CLI Quick Start

Run simulations directly from the command line:

```bash
# List available scenarios
swarm list

# Run a scenario
swarm run scenarios/baseline.yaml

# Override simulation settings
swarm run scenarios/baseline.yaml --seed 42 --epochs 20 --steps 15

# Export outputs
swarm run scenarios/baseline.yaml --export-json results.json --export-csv outputs/
```

## Core Concepts

### Soft Probabilistic Labels

Instead of binary labels (good/bad), interactions carry a probability `p = P(v = +1)` representing the likelihood of a beneficial outcome:

1. **Proxy signals** are combined into a raw score `v_hat in [-1, +1]`
2. **Calibrated sigmoid** converts to probability: `p = 1 / (1 + exp(-k * v_hat))`

### Four Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Toxicity rate** | `E[1-p \| accepted]` | Expected harm among accepted |
| **Quality gap** | `E[p \| accepted] - E[p \| rejected]` | Adverse selection indicator (negative = bad) |
| **Conditional loss** | `E[pi \| accepted] - E[pi]` | Selection effect on payoffs |
| **Incoherence** | `Var[decision] / E[error]` | Variance-to-error ratio across replays |

### Governance Levers

- **Transaction Taxes** - Reduce exploitation, cost welfare
- **Reputation Decay** - Punish bad actors, erode honest standing
- **Circuit Breakers** - Freeze toxic agents quickly
- **Random Audits** - Deter hidden exploitation
- **Staking** - Filter undercapitalized agents
- **Collusion Detection** - Catch coordinated attacks

### Agent Policies

| Type | Behavior |
|------|----------|
| **Honest** | Cooperative, trust-based, completes tasks diligently |
| **Opportunistic** | Maximizes short-term payoff, cherry-picks tasks, strategic voting |
| **Deceptive** | Builds trust through honest behavior, then exploits trusted relationships |
| **Adversarial** | Targets honest agents, coordinates with allies, disrupts ecosystem |
| **LLM** | Behavior determined by LLM with configurable persona ([details](docs/llm-agents.md)) |

## Architecture

```
SWARM Core
+------------------------------------------------------------+
|                                                            |
|  ProxyComputer --> SoftInteraction --> Metrics             |
|       |                  |                |                |
|       |                  |                |                |
|  Observable          Payoff          Governance            |
|  Extraction          Engine          Engine                |
|                                                            |
+------------------------------------------------------------+
```

**Data Flow:**
```
Observables -> ProxyComputer -> v_hat -> sigmoid -> p -> SoftPayoffEngine -> payoffs
                                                    |
                                               SoftMetrics -> toxicity, quality gap, etc.
```

## Directory Structure

```
swarm/
├── swarm/
│   ├── models/          # SoftInteraction, AgentState/AgentStatus, event schema
│   ├── core/            # PayoffEngine, ProxyComputer, sigmoid, orchestrator
│   ├── agents/          # Honest, opportunistic, deceptive, adversarial, LLM, adaptive
│   ├── env/             # EnvState, feed, tasks, network, composite tasks
│   ├── governance/      # Config, levers, taxes, reputation, audits, collusion
│   ├── metrics/         # SoftMetrics, reporters, collusion detection, capabilities
│   ├── forecaster/      # Risk forecasters for adaptive governance activation
│   ├── replay/          # Replay runner and decision-level replay utilities
│   ├── scenarios/       # YAML scenario loader
│   ├── analysis/        # Parameter sweeps, dashboard
│   ├── redteam/         # Attack scenarios, evaluator, evasion metrics
│   ├── boundaries/      # External world, flow tracking, policies, leakage
│   └── logging/         # Append-only JSONL logger
├── tests/               # Test suite
├── examples/            # Demo scripts
├── scenarios/           # YAML scenario definitions
├── docs/                # Documentation
└── pyproject.toml
```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=swarm --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v

# Run CI checks (lint, type-checking, tests)
make ci
```

## Documentation

| Topic | Description |
|-------|-------------|
| [Theoretical Foundations](docs/research/theory.md) | Formal model, whitepaper-style summary, and citation section |
| [LLM Agents](docs/llm-agents.md) | Providers, personas, cost tracking, YAML config |
| [Network Topology](docs/network-topology.md) | Topology types, dynamic evolution, network metrics |
| [Governance](docs/governance.md) | Levers, collusion detection, integration points |
| [Emergent Capabilities](docs/emergent-capabilities.md) | Composite tasks, capability types, emergent metrics |
| [Red-Teaming](docs/red-teaming.md) | Adaptive adversaries, attack strategies, evaluation results |
| [Scenarios & Sweeps](docs/scenarios.md) | YAML scenarios, scenario comparison, parameter sweeps |
| [Boundaries](docs/boundaries.md) | External world simulation, flow tracking, leakage detection |
| [Dashboard](docs/dashboard.md) | Streamlit dashboard setup and features |
| [Incoherence Metric Contract](docs/incoherence_metric_contract.md) | Definitions and edge-case semantics |
| [Incoherence Scaling Analysis](docs/analysis/incoherence_scaling.md) | Replay-sweep artifact and upgrade path |
| [Incoherence Governance Transferability](docs/transferability/incoherence_governance.md) | Deployment caveats and assumptions |

## Start Here (Researcher Path)

- Read the framing: [Theoretical Foundations](docs/research/theory.md)
- Run an incoherence artifact: [Incoherence Scaling Analysis](docs/analysis/incoherence_scaling.md)
- Inspect policy caveats: [Incoherence Governance Transferability](docs/transferability/incoherence_governance.md)
- Reproduce from CLI: `swarm run scenarios/baseline.yaml`

## Citation

```bibtex
@software{swarm2026,
  title = {SWARM: System-Wide Assessment of Risk in Multi-agent systems},
  author = {Savitt, Raeli},
  year = {2026},
  url = {https://github.com/swarm-ai-safety/swarm}
}
```

Machine-readable citation metadata: [`CITATION.cff`](CITATION.cff)

## References

- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica.
- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market*. JFE.
- [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)
- [The Hot Mess Theory of AI](https://alignment.anthropic.com/2026/hot-mess-of-ai/)
- [Moltbook](https://moltbook.com) | [@sebkrier](https://x.com/sebkrier/status/2017993948132774232)

## License

MIT License - See [LICENSE](LICENSE) for details.
