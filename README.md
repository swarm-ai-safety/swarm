# Distributional AGI Safety Sandbox

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
- **LLM-backed agents** with Anthropic, OpenAI, and Ollama support
- Environment state management with rate limits
- Feed engine (posts, replies, voting, visibility ranking)
- Task system (claiming, collaboration, verification)
- Orchestrator for multi-agent simulation (sync and async)
- **Network topology** with dynamic evolution (complete, ring, star, small-world, scale-free)

**Governance Layer:**
- Configurable levers (taxes, reputation decay, staking, circuit breakers, audits)
- **Collusion detection** with pair-level and group-level analysis
- Integration with orchestrator via epoch and interaction hooks
- Populates `c_a` and `c_b` governance costs on interactions

## Theoretical Foundations

The sandbox draws on **market microstructure theory** to model information asymmetry and adverse selection in multi-agent systems.

### Kyle Model: Informed vs Uninformed Agents

In Kyle's (1985) continuous auction model, *informed traders* possess private information and strategically exploit it through their trading volume. The sandbox maps this to:

- **Informed traders** → Deceptive and adversarial agents who know the true value of interactions
- **Uninformed traders** → Honest agents relying on observable signals
- **Market maker** → The system's acceptance/rejection mechanism

Deceptive agents, like informed traders, can profit by selectively engaging when they have private knowledge that an interaction will benefit them at others' expense.

### Glosten-Milgrom Model: Acceptance as Information

Glosten & Milgrom (1985) showed that the *willingness to trade reveals information*. A counterparty eager to transact may signal adverse private information. The sandbox captures this through:

| Microstructure Concept | Sandbox Analog |
|------------------------|----------------|
| Trade | Collaboration, reply, data exchange |
| Price | Claimed benefit / reputation transfer |
| Bid-ask spread | Gap between offered and realized outcome |
| Adverse selection | Low-quality interactions preferentially accepted |

**Key insight**: When `quality_gap < 0` (accepted interactions have lower `p` than rejected ones), the system exhibits adverse selection—bad actors are successfully gaming the acceptance mechanism.

### Soft Labels and Expected Harm

Rather than binary good/bad classifications, interactions carry a probability `p = P(v = +1)`. This enables:

1. **Probabilistic toxicity**: `E[1-p | accepted]` measures expected harm among accepted interactions
2. **Conditional loss**: Selection effects on payoffs reveal when acceptance correlates with harm
3. **Externality pricing**: `E_soft = (1-p) * h` quantifies ecosystem-level costs

The soft label formulation allows governance mechanisms to respond proportionally to risk rather than relying on noisy binary classifications.

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

## Directory Structure

```
distributional-agi-safety/
├── src/
│   ├── models/
│   │   ├── interaction.py      # SoftInteraction, InteractionType
│   │   ├── agent.py            # AgentType, AgentState
│   │   └── events.py           # Event log schema
│   ├── core/
│   │   ├── payoff.py           # SoftPayoffEngine
│   │   ├── proxy.py            # ProxyComputer (v_hat computation)
│   │   ├── sigmoid.py          # Calibrated sigmoid utilities
│   │   └── orchestrator.py     # Simulation orchestrator
│   ├── agents/
│   │   ├── base.py             # BaseAgent, Action, Observation
│   │   ├── honest.py           # Cooperative agent policy
│   │   ├── opportunistic.py    # Payoff-maximizing policy
│   │   ├── deceptive.py        # Trust-then-exploit policy
│   │   ├── adversarial.py      # Targeting/coordination policy
│   │   ├── adaptive_adversary.py # Learning adversary for red-teaming
│   │   ├── llm_agent.py        # LLM-backed agent
│   │   ├── llm_config.py       # LLM configuration and cost tracking
│   │   ├── llm_prompts.py      # Persona prompts and formatting
│   │   └── roles/              # Role mixins (planner, worker, verifier, etc.)
│   ├── env/
│   │   ├── state.py            # EnvState, RateLimits
│   │   ├── feed.py             # Posts, replies, voting
│   │   ├── tasks.py            # Task pool and lifecycle
│   │   ├── network.py          # Network topology and dynamics
│   │   └── composite_tasks.py  # Multi-agent composite tasks
│   ├── governance/
│   │   ├── config.py           # GovernanceConfig
│   │   ├── levers.py           # Abstract GovernanceLever base
│   │   ├── engine.py           # GovernanceEngine (aggregates levers)
│   │   ├── taxes.py            # TransactionTaxLever
│   │   ├── reputation.py       # ReputationDecayLever, VoteNormalizationLever
│   │   ├── admission.py        # StakingLever
│   │   ├── circuit_breaker.py  # CircuitBreakerLever
│   │   ├── audits.py           # RandomAuditLever
│   │   ├── collusion.py        # CollusionPenaltyLever
│   │   └── security.py         # SecurityLever
│   ├── metrics/
│   │   ├── soft_metrics.py     # SoftMetrics (40+ metrics)
│   │   ├── reporters.py        # Dual reporting (soft + hard)
│   │   ├── collusion.py        # Collusion detection algorithms
│   │   ├── capabilities.py     # Emergent capability metrics
│   │   └── security.py         # Security threat detection
│   ├── scenarios/
│   │   └── loader.py           # YAML scenario loader
│   ├── analysis/
│   │   ├── aggregation.py      # Metrics aggregation for visualization
│   │   ├── plots.py            # Plotly chart generators
│   │   ├── dashboard.py        # Streamlit dashboard
│   │   ├── export.py           # CSV/JSON/Parquet export
│   │   └── sweep.py            # Parameter sweep runner
│   ├── redteam/                 # Red-teaming framework
│   │   ├── attacks.py          # Attack scenarios library
│   │   ├── evaluator.py        # Red-team evaluator
│   │   └── metrics.py          # Evasion and effectiveness metrics
│   └── logging/
│       └── event_log.py        # Append-only JSONL logger
├── tests/
│   ├── test_payoff.py
│   ├── test_proxy.py
│   ├── test_metrics.py
│   ├── test_agents.py
│   ├── test_env.py
│   ├── test_orchestrator.py
│   ├── test_governance.py
│   ├── test_scenarios.py
│   ├── test_sweep.py
│   ├── test_llm_agent.py       # LLM agent tests (43 tests)
│   ├── test_network.py         # Network topology tests (48 tests)
│   ├── test_collusion.py       # Collusion detection tests (26 tests)
│   ├── test_capabilities.py    # Emergent capability tests (32 tests)
│   ├── test_security.py        # Security evaluation tests (45 tests)
│   ├── test_redteam.py         # Red-teaming tests (45 tests)
│   ├── test_dashboard.py       # Dashboard and visualization tests (43 tests)
│   └── fixtures/
│       └── interactions.py     # Test data generators
├── examples/
│   ├── mvp_demo.py             # End-to-end demo
│   ├── run_scenario.py         # Run simulation from YAML
│   ├── parameter_sweep.py      # Batch parameter sweep
│   └── llm_demo.py             # LLM agent demo
├── scenarios/
│   ├── baseline.yaml           # 5-agent baseline scenario
│   ├── status_game.yaml        # Reputation competition
│   ├── strict_governance.yaml  # All governance levers enabled
│   ├── llm_agents.yaml         # Mixed LLM and scripted agents
│   ├── network_effects.yaml    # Small-world network with dynamic edges
│   ├── collusion_detection.yaml # Collusion detection scenario
│   ├── emergent_capabilities.yaml # Multi-agent collaboration scenario
│   ├── security_evaluation.yaml # Security threat detection scenario
│   └── adversarial_redteam.yaml # Red-teaming governance scenario
└── pyproject.toml
```

## Agent Policies

| Type | Behavior |
|------|----------|
| **Honest** | Cooperative, trust-based, completes tasks diligently |
| **Opportunistic** | Maximizes short-term payoff, cherry-picks tasks, strategic voting |
| **Deceptive** | Builds trust through honest behavior, then exploits trusted relationships |
| **Adversarial** | Targets honest agents, coordinates with allies, disrupts ecosystem |
| **LLM** | Behavior determined by LLM with configurable persona |

## LLM Agents

LLM-backed agents use real language models to make decisions, enabling study of emergent behavior rather than scripted policies.

### Quick Start

```python
import asyncio
from src.agents.llm_agent import LLMAgent
from src.agents.llm_config import LLMConfig, LLMProvider, PersonaType
from src.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure LLM agent
llm_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    persona=PersonaType.OPEN,  # Let the LLM develop its own strategy
    temperature=0.7,
)

# Create orchestrator and agents
config = OrchestratorConfig(n_epochs=5, steps_per_epoch=3)
orchestrator = Orchestrator(config=config)
orchestrator.register_agent(LLMAgent("llm_1", llm_config))

# Run asynchronously for better performance with LLM agents
metrics = asyncio.run(orchestrator.run_async())
```

### Providers

| Provider | Model Examples | API Key Env Var |
|----------|---------------|-----------------|
| **Anthropic** | claude-sonnet-4-20250514, claude-3-haiku-20240307 | `ANTHROPIC_API_KEY` |
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| **Ollama** | llama3, mistral (local) | None required |

### Personas

| Persona | Behavior |
|---------|----------|
| **Honest** | Cooperative, maximizes collective welfare |
| **Strategic** | Self-interested, cooperates when beneficial |
| **Adversarial** | Probes system weaknesses, tests governance robustness |
| **Open** | No prescribed strategy - LLM develops its own approach |

### YAML Configuration

```yaml
agents:
  - type: llm
    count: 2
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
      persona: open
      temperature: 0.7
      max_tokens: 512
      cost_tracking: true

  - type: honest  # Mix with scripted agents
    count: 2
```

### Cost Tracking

LLM agents track token usage and estimated costs:

```python
# After simulation
stats = orchestrator.get_llm_usage_stats()
for agent_id, usage in stats.items():
    print(f"{agent_id}: {usage['total_requests']} requests, ${usage['estimated_cost_usd']:.4f}")
```

### Demo

```bash
# Dry run (no API calls)
python examples/llm_demo.py --dry-run

# With real API calls
export ANTHROPIC_API_KEY="your-key"
python examples/llm_demo.py

# Use OpenAI instead
export OPENAI_API_KEY="your-key"
python examples/llm_demo.py --provider openai --model gpt-4o
```

## Network Topology

The network module controls which agents can interact, enabling study of information cascades and coalition formation.

### Quick Start

```python
from src.core.orchestrator import Orchestrator, OrchestratorConfig
from src.env.network import NetworkConfig, NetworkTopology

# Configure small-world network
network_config = NetworkConfig(
    topology=NetworkTopology.SMALL_WORLD,
    k_neighbors=4,           # Initial connections to nearest neighbors
    rewire_probability=0.1,  # 10% chance to rewire each edge
    dynamic=True,            # Enable edge evolution
    edge_strengthen_rate=0.1,  # Strengthen on interaction
    edge_decay_rate=0.05,      # 5% decay per epoch
    min_edge_weight=0.2,       # Prune weak edges
)

config = OrchestratorConfig(
    n_epochs=20,
    network_config=network_config,
)
orchestrator = Orchestrator(config=config)
```

### Topology Types

| Topology | Description | Use Case |
|----------|-------------|----------|
| **Complete** | All agents connected | Baseline, small populations |
| **Ring** | Circular chain | Local information flow |
| **Star** | Hub-and-spoke | Centralized coordination |
| **Random (Erdős-Rényi)** | Edges with probability p | Random networks |
| **Small-World** | High clustering, short paths | Social networks |
| **Scale-Free** | Power-law degree distribution | Hubs and influence |
| **Custom** | Manual edge specification | Specific topologies |

### Dynamic Network Evolution

When `dynamic=True`, the network evolves based on agent behavior:

1. **Edge Strengthening**: Successful interactions increase edge weight
2. **Edge Decay**: All edges decay each epoch (models relationship maintenance costs)
3. **Edge Pruning**: Edges below `min_edge_weight` are removed
4. **Weight Capping**: Edges capped at `max_edge_weight`

```python
# After simulation, inspect network state
network = orchestrator.network
print(f"Connected components: {network.connected_components()}")
print(f"Average clustering: {network.get_metrics()['avg_clustering']}")
```

### Network Metrics

Each epoch includes network metrics:

| Metric | Description |
|--------|-------------|
| `n_edges` | Current edge count |
| `avg_degree` | Average connections per agent |
| `avg_clustering` | Local clustering coefficient |
| `n_components` | Number of connected components |
| `avg_path_length` | Average shortest path (if connected) |
| `density` | Edge density (edges / possible edges) |

### YAML Configuration

```yaml
network:
  topology: small_world
  params:
    k: 4     # k_neighbors
    p: 0.1   # rewire_probability

  dynamic: true
  edge_strengthen_rate: 0.1
  edge_decay_rate: 0.05
  min_edge_weight: 0.2
  max_edge_weight: 1.0

  # Optional: disconnect from agents with bad reputation
  # reputation_disconnect_threshold: -2.0

success_criteria:
  min_connected_components: 1
  max_avg_path_length: 4.0
```

### Integration with Orchestrator

The network constrains interactions:
- **Observation filtering**: Agents only see neighbors in their observations
- **Interaction validation**: Actions targeting non-neighbors are rejected
- **Edge strengthening**: Accepted interactions strengthen the edge between agents

Run the network effects scenario:
```bash
python examples/run_scenario.py scenarios/network_effects.yaml
```

## Orchestrator

The orchestrator manages the simulation loop:

```python
for epoch in range(n_epochs):
    for step in range(steps_per_epoch):
        for agent in agents:
            observation = build_observation(agent, state)
            action = agent.act(observation)
            execute_action(action, state, event_log)
        resolve_pending_interactions()
    emit_epoch_metrics()
```

**Features:**
- Agent scheduling (round-robin, random, or priority-based)
- Rate limit enforcement (posts, votes, interactions per epoch)
- Automatic payoff computation via SoftPayoffEngine
- Event logging with full simulation replay

## Governance

The governance module provides configurable levers that affect agent behavior and payoffs:

```python
from src.governance import GovernanceConfig, GovernanceEngine
from src.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure governance levers
gov_config = GovernanceConfig(
    # Transaction tax: 5% of |tau|, split 50/50
    transaction_tax_rate=0.05,
    transaction_tax_split=0.5,

    # Reputation decays 10% each epoch
    reputation_decay_rate=0.9,

    # Require minimum stake to participate
    staking_enabled=True,
    min_stake_to_participate=10.0,

    # Freeze agents with >70% toxicity after 3 violations
    circuit_breaker_enabled=True,
    freeze_threshold_toxicity=0.7,
    freeze_threshold_violations=3,
    freeze_duration_epochs=2,

    # 10% chance of audit, penalty if p < 0.5
    audit_enabled=True,
    audit_probability=0.1,
    audit_threshold_p=0.5,
)

# Pass to orchestrator
config = OrchestratorConfig(
    n_epochs=10,
    governance_config=gov_config,
)
orchestrator = Orchestrator(config=config)
```

### Governance Levers

| Lever | Effect | Hook |
|-------|--------|------|
| **Transaction Tax** | `c_a += rate * \|tau\| * split`, `c_b += rate * \|tau\| * (1-split)` | on_interaction |
| **Reputation Decay** | `reputation *= decay_rate` each epoch | on_epoch_start |
| **Vote Normalization** | Diminishing influence as vote count increases | compute_vote_weight |
| **Staking** | Block actions if `resources < min_stake` | can_agent_act |
| **Circuit Breaker** | Freeze agent if `avg_toxicity > threshold` for N violations | on_interaction |
| **Random Audit** | Penalty `(threshold - p) * multiplier` if audited and `p < threshold` | on_interaction |
| **Collusion Detection** | Penalty based on collusion risk score from coordinated behavior | on_epoch_start, on_interaction |

### Integration Points

Governance hooks into the orchestrator at:
1. **Epoch start**: Reputation decay, unfreeze agents whose freeze duration expired
2. **Agent action**: Staking check blocks under-resourced agents
3. **Interaction completion**: Taxes, circuit breaker tracking, random audits

Costs are added to `interaction.c_a` and `interaction.c_b` before payoff computation.

## Collusion Detection

The collusion detection system identifies coordinated manipulation patterns among agents.

### Detection Signals

| Signal | Description | Threshold |
|--------|-------------|-----------|
| **Interaction Frequency** | Z-score of pair interaction count vs. population | > 2.0 |
| **Benefit Correlation** | Correlation between pair benefits across interactions | > 0.7 |
| **Acceptance Rate** | Fraction of mutually accepted interactions | > 0.8 |
| **Quality Asymmetry** | Difference in avg p (internal vs external) | > 0.2 |

### Quick Start

```python
from src.governance import GovernanceConfig, GovernanceEngine

# Enable collusion detection
config = GovernanceConfig(
    collusion_detection_enabled=True,
    collusion_frequency_threshold=2.0,      # Z-score threshold
    collusion_score_threshold=0.5,          # Flag pairs above this
    collusion_penalty_multiplier=1.5,       # Scale penalties
    collusion_realtime_penalty=True,        # Per-interaction penalty
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

### YAML Configuration

```yaml
governance:
  collusion_detection_enabled: true
  collusion_frequency_threshold: 2.0      # Z-score for unusual frequency
  collusion_correlation_threshold: 0.7    # Benefit correlation threshold
  collusion_min_interactions: 3           # Min interactions to analyze pair
  collusion_score_threshold: 0.5          # Threshold for flagging
  collusion_penalty_multiplier: 1.5       # Penalty scaling factor
  collusion_realtime_penalty: true        # Apply per-interaction penalty
  collusion_realtime_rate: 0.1            # Rate for realtime penalty
```

Run the collusion detection scenario:
```bash
python examples/run_scenario.py scenarios/collusion_detection.yaml
```

## Security Evaluation

The security module detects and responds to security threats in multi-agent systems, including prompt injection, cross-agent manipulation, information laundering, and contagion propagation.

### Threat Types

| Threat | Description | Detection Method |
|--------|-------------|------------------|
| **Prompt Injection** | Attempts to inject malicious instructions | Tool misuse flags, verifier rejections, low-p accepted interactions |
| **Cross-Agent Manipulation** | Exploiting other agents' trust | Repeated targeting, reputation extraction, resource draining |
| **Information Laundering** | Washing low-quality info through trusted agents | Trust increase along interaction paths |
| **Contagion/Worm** | Self-replicating malicious patterns | Rapid spread velocity, chain propagation depth |
| **Exfiltration** | Extracting sensitive information | Multiple tool misuse flags |

### Quick Start

```python
from src.governance import GovernanceConfig, GovernanceEngine

# Enable security detection
config = GovernanceConfig(
    security_enabled=True,
    security_injection_threshold=0.3,      # Threshold for injection detection
    security_manipulation_threshold=0.5,    # Threshold for manipulation
    security_quarantine_threshold=0.7,      # Threat score for quarantine
    security_realtime_penalty=True,         # Per-interaction penalty
)

# After simulation
report = orchestrator.get_security_report()
print(f"Ecosystem threat level: {report.ecosystem_threat_level:.2f}")
print(f"Active threats: {report.active_threat_count}")
print(f"Quarantined agents: {orchestrator.get_quarantined_agents()}")

# Check specific threats
for indicator in report.threat_indicators:
    print(f"  {indicator.threat_type.value}: {indicator.source_agent} -> {indicator.target_agents}")
```

### Security Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| `ecosystem_threat_level` | Overall system threat level | 0-1 (>0.5 = concerning) |
| `prompt_injection_rate` | Fraction of interactions flagged for injection | Lower is better |
| `manipulation_rate` | Agents showing manipulation patterns | Lower is better |
| `contagion_depth` | Maximum propagation chain depth | >3 = concerning |
| `agents_flagged` | Set of agents with elevated threat scores | Candidates for review |

### Containment Actions

The security lever can automatically:
- **Quarantine** agents with threat scores above threshold (blocks all actions)
- **Apply penalties** proportional to threat score (reputation and resource)
- **Track contagion chains** and mark them contained when origin is quarantined
- **Real-time penalties** on suspicious interactions

### YAML Configuration

```yaml
governance:
  security_enabled: true
  security_injection_threshold: 0.3       # Injection detection threshold
  security_manipulation_threshold: 0.5    # Manipulation detection threshold
  security_laundering_trust_gap: 0.3      # Trust increase for laundering flag
  security_contagion_velocity: 2.0        # Interactions/min for contagion
  security_min_chain_length: 3            # Minimum chain to track
  security_quarantine_threshold: 0.7      # Threshold for quarantine
  security_penalty_multiplier: 1.0        # Penalty scaling
  security_realtime_penalty: true         # Per-interaction penalty
  security_realtime_threshold: 0.5        # Realtime penalty threshold
  security_realtime_rate: 0.2             # Realtime penalty rate
```

### Integration with Orchestrator

Security hooks into the orchestrator at:
1. **Epoch start**: Analyze accumulated interactions, apply quarantine and penalties
2. **Per-interaction**: Real-time threat scoring and optional penalty
3. **Agent action check**: Quarantined agents cannot act

```python
# Check if agent can act (not quarantined)
if orchestrator.governance_engine.can_agent_act(agent_id, state):
    # Agent is allowed to proceed
    pass

# Release agent from quarantine manually
orchestrator.governance_engine.release_from_quarantine(agent_id)

# Get containment action history
actions = orchestrator.governance_engine.get_security_containment_actions()
```

## Emergent Capability Measurement

The emergent capabilities module measures collective intelligence and coordination that emerges from multi-agent collaboration on composite tasks.

### Composite Tasks

Composite tasks require multiple agents with complementary capabilities to work together:

```python
from src.env.composite_tasks import (
    CompositeTask, CompositeTaskPool, Subtask, CapabilityType,
    create_research_synthesis_task, create_problem_solving_task,
)

# Create a research task requiring multiple capabilities
task = create_research_synthesis_task(
    topic="Multi-agent coordination patterns",
    deadline_epoch=10,
    bounty=30.0,
)

# Task requires: RESEARCH, ANALYSIS, COMMUNICATION, VERIFICATION
# No single agent can complete it alone
print(f"Required capabilities: {task.required_capabilities}")
print(f"Subtasks: {[st.name for st in task.subtasks]}")
```

### Capability Types

| Capability | Description | Example Subtasks |
|------------|-------------|------------------|
| **Research** | Information gathering | Literature review, data collection |
| **Analysis** | Data analysis | Pattern identification, statistics |
| **Planning** | Strategic planning | Strategy development, resource allocation |
| **Execution** | Task implementation | Implementation, deployment |
| **Verification** | Quality checking | Review, validation |
| **Coordination** | Team management | Task assignment, communication |
| **Creativity** | Novel solutions | Brainstorming, design |
| **Communication** | Clear expression | Report writing, documentation |

### Emergent Metrics

The system measures emergent behaviors that arise from collaboration:

| Metric | Description | Range |
|--------|-------------|-------|
| **Coordination Score** | How evenly work is distributed | 0-1 (1 = perfect balance) |
| **Synergy Score** | Team output vs. sum of parts | 0-1 (>0.5 = synergy) |
| **Information Flow** | How well dependent tasks build on predecessors | 0-1 |
| **Specialization Index** | Agent skill concentration | 0-1 |
| **Complementarity Score** | Capability diversity across agents | 0-1 |
| **Knowledge Transfer** | Skill improvement from collaboration | 0+ |

### Quick Start

```python
from src.core.orchestrator import Orchestrator, OrchestratorConfig
from src.env.composite_tasks import CapabilityType, create_problem_solving_task

# Enable composite tasks
config = OrchestratorConfig(
    n_epochs=20,
    enable_composite_tasks=True,
)
orchestrator = Orchestrator(config=config)

# Register agents with capabilities
orchestrator.register_agent(agent1)
orchestrator.register_agent_capabilities("agent_1", {
    CapabilityType.RESEARCH,
    CapabilityType.ANALYSIS,
})

orchestrator.register_agent_capabilities("agent_2", {
    CapabilityType.PLANNING,
    CapabilityType.EXECUTION,
})

# Add a composite task
task = create_problem_solving_task("Resource optimization")
orchestrator.add_composite_task(task)

# Run simulation
metrics = orchestrator.run()

# Check capability metrics
cap_metrics = orchestrator.get_capability_metrics()
print(f"Coordination: {cap_metrics.avg_coordination_score:.2f}")
print(f"Synergy: {cap_metrics.avg_synergy_score:.2f}")
```

### Task Templates

Pre-built task templates for common multi-agent scenarios:

| Template | Min Agents | Capabilities Required | Use Case |
|----------|------------|----------------------|----------|
| **Research Synthesis** | 2 | Research, Analysis, Communication, Verification | Literature review, data analysis |
| **Planning Coordination** | 3 | Planning, Coordination, Analysis, Verification | Strategic planning, resource allocation |
| **Problem Solving** | 3 | Analysis, Creativity, Planning, Execution, Verification | Complex problem solving with parallel work |

### YAML Configuration

```yaml
governance:
  composite_tasks_enabled: true

composite_tasks:
  initial_tasks:
    - type: research_synthesis
      topic: "Safety protocols"
      deadline_offset: 10
      bounty: 30.0

    - type: problem_solving
      problem: "Coordination failure"
      deadline_offset: 15
      bounty: 40.0

  task_spawn_rate: 0.3
  max_concurrent_tasks: 5

success_criteria:
  min_avg_coordination_score: 0.4
  min_tasks_completed: 2
```

Run the emergent capabilities scenario:
```bash
python examples/run_scenario.py scenarios/emergent_capabilities.yaml
```

## Adversarial Red-Teaming

The red-teaming framework tests governance robustness against adaptive adversaries that learn to evade detection.

### Adaptive Adversaries

Adaptive adversaries use reinforcement learning-style updates to evolve attack strategies:

```python
from src.agents.adaptive_adversary import AdaptiveAdversary, AttackStrategy

# Create adaptive adversary
adversary = AdaptiveAdversary(
    agent_id="adaptive_1",
    learning_rate=0.1,      # How fast it adapts
    exploration_rate=0.2,   # Exploration vs exploitation
    heat_threshold=0.6,     # When to lay low
)

# Adversary learns from outcomes
adversary.update_from_outcome(
    payoff=10.0,
    penalty=5.0,
    detected=True,
    strategy_used=AttackStrategy.COLLUSION,
)

# Get strategy effectiveness report
report = adversary.get_strategy_report()
print(f"Heat level: {report['heat_level']:.2f}")
print(f"Best strategies: {report['strategy_stats']}")
```

### Attack Strategies

| Strategy | Description | Evasion Technique |
|----------|-------------|-------------------|
| **Reputation Farming** | Build trust, then exploit | Act honestly initially |
| **Collusion** | Coordinate with allies | Internal boost, external exploit |
| **Low Profile** | Minimal activity | Stay under radar |
| **Mimicry** | Imitate honest agents | Copy behavior patterns |
| **Threshold Dancing** | Stay below detection limits | Learn governance thresholds |
| **Timing Attack** | Attack at optimal moments | Learn monitoring patterns |
| **Governance Gaming** | Exploit rule loopholes | Find edge cases |

### Red-Team Evaluator

Systematically test governance configurations:

```python
from src.redteam import RedTeamEvaluator, AttackLibrary

# Define governance config to test
governance_config = {
    "circuit_breaker_enabled": True,
    "collusion_detection_enabled": True,
    "audit_enabled": True,
}

# Create evaluator
evaluator = RedTeamEvaluator(
    governance_config=governance_config,
    attack_scenarios=AttackLibrary.get_all_attacks(),
)

# Quick evaluation
result = evaluator.quick_evaluate()
print(f"Attacks tested: {result['attacks_tested']}")
print(f"Success rate: {result['attacks_successful'] / result['attacks_tested']:.0%}")
print(f"Avg evasion: {result['avg_evasion_rate']:.0%}")
```

### Evasion Metrics

Track adversary evasion effectiveness:

| Metric | Description | Range |
|--------|-------------|-------|
| **Evasion Rate** | Actions that went undetected | 0-1 |
| **Detection Latency** | Epochs until detection | 0+ |
| **Damage Before Detection** | Harm before caught | 0+ |
| **Heat Level** | Current detection risk | 0-1 |

### Governance Effectiveness

Measure governance detection capabilities:

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Precision** | TP / (TP + FP) | Accuracy of detections |
| **Recall** | TP / (TP + FN) | Coverage of bad actors |
| **F1 Score** | Harmonic mean | Overall effectiveness |
| **Prevention Rate** | Prevented / Total | Attack blocking success |

### YAML Configuration

```yaml
agents:
  - type: honest
    count: 4
  - type: adaptive_adversary
    count: 2

governance:
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.6
  collusion_detection_enabled: true
  audit_enabled: true
  audit_probability: 0.15

success_criteria:
  max_adversary_payoff: 20.0
  min_detection_rate: 0.5
```

Run the red-team scenario:
```bash
python examples/run_scenario.py scenarios/adversarial_redteam.yaml
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

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Toxicity rate | `E[1-p \| accepted]` | Expected harm among accepted |
| Quality gap | `E[p \| accepted] - E[p \| rejected]` | Adverse selection indicator |
| Conditional loss | `E[π \| accepted] - E[π]` | Selection effect on payoffs |
| Brier score | `E[(p - v)²]` | Calibration quality |
| ECE | Expected calibration error | Binned calibration |

## Event Logging & Replay

```python
from pathlib import Path
from src.core.orchestrator import Orchestrator, OrchestratorConfig

# Run with logging
config = OrchestratorConfig(
    n_epochs=10,
    log_path=Path("logs/simulation.jsonl"),
    log_events=True,
)
orchestrator = Orchestrator(config=config)
# ... register agents and run ...

# Replay events
for event in orchestrator.event_log.replay():
    print(event.event_type, event.timestamp)

# Reconstruct interactions
interactions = orchestrator.event_log.to_interactions()
```

## Running Tests

```bash
# Run all tests (506 tests)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_orchestrator.py -v
```

## Scenarios

Define simulations in YAML with full governance configuration:

```yaml
# scenarios/status_game.yaml
scenario_id: status_game
description: "Reputation competition with governance"

agents:
  - type: honest
    count: 2
  - type: opportunistic
    count: 2
  - type: adversarial
    count: 1

governance:
  transaction_tax_rate: 0.05
  reputation_decay_rate: 0.95
  staking_enabled: true
  min_stake_to_participate: 10.0
  circuit_breaker_enabled: true
  freeze_threshold_toxicity: 0.6
  audit_enabled: true
  audit_probability: 0.15

simulation:
  n_epochs: 20
  steps_per_epoch: 15
  seed: 123

payoff:
  s_plus: 3.0
  s_minus: 1.5
  h: 2.5
  theta: 0.5
  w_rep: 2.0
```

Run scenarios from the command line:

```bash
python examples/run_scenario.py scenarios/baseline.yaml
python examples/run_scenario.py scenarios/status_game.yaml
python examples/run_scenario.py scenarios/strict_governance.yaml
```

Or load programmatically:

```python
from src.scenarios import load_and_build

orchestrator = load_and_build(Path("scenarios/status_game.yaml"))
metrics = orchestrator.run()
```

### Scenario Comparison

| Metric | Baseline | Status Game | Strict Governance |
|--------|----------|-------------|-------------------|
| **Governance** | None | Moderate | Heavy |
| Tax rate | 0% | 5% | 10% |
| Reputation decay | None | 5%/epoch | 15%/epoch |
| Staking required | No | 10.0 | 25.0 |
| Circuit breaker | No | Yes (0.6) | Yes (0.5) |
| Audit probability | 0% | 15% | 25% |
| **Results** | | | |
| Bad actor frozen | No | Yes | Yes |
| Bad actor payoff | +3.42 | +1.22 | -1.55 |
| Avg toxicity | 0.30 | 0.33 | 0.32 |
| Welfare/epoch | 7.29 | 13.02 | 8.15 |

Governance effectively punishes bad actors (payoffs drop from positive to negative) while maintaining similar toxicity levels. Stricter governance reduces bad actor gains but also dampens overall welfare.

## Parameter Sweeps

Run batch simulations over parameter ranges:

```python
from src.analysis import SweepConfig, SweepParameter, SweepRunner
from src.scenarios import load_scenario

# Load base scenario
scenario = load_scenario(Path("scenarios/baseline.yaml"))

# Configure sweep
config = SweepConfig(
    base_scenario=scenario,
    parameters=[
        SweepParameter(
            name="governance.transaction_tax_rate",
            values=[0.0, 0.05, 0.10, 0.15],
        ),
        SweepParameter(
            name="governance.circuit_breaker_enabled",
            values=[False, True],
        ),
    ],
    runs_per_config=3,  # Multiple runs for statistical significance
    seed_base=42,
)

# Run sweep
runner = SweepRunner(config)
results = runner.run()

# Export to CSV
runner.to_csv(Path("results.csv"))

# Get summary statistics
summary = runner.summary()
```

Run the example:
```bash
python examples/parameter_sweep.py
python examples/parameter_sweep.py --output my_results.csv
```

Supported parameter paths:
- `governance.*` - Any GovernanceConfig field
- `payoff.*` - Any PayoffConfig field
- `n_epochs`, `steps_per_epoch` - Simulation settings

## Dashboard

Interactive Streamlit dashboard for real-time simulation visualization and analysis.

### Quick Start

```bash
# Install dashboard dependencies
pip install -e ".[dashboard]"

# Run the dashboard
streamlit run src/analysis/dashboard.py
```

### Features

The dashboard provides:

1. **Simulation Controls**: Configure and run simulations directly from the UI
   - Agent mix (honest, opportunistic, adversarial)
   - Governance settings (taxes, circuit breakers, security)
   - Network topology

2. **Real-Time Metrics**: Live metrics during simulation
   - Toxicity rate, welfare, reputation
   - Threat level, collusion risk
   - Network statistics

3. **Time Series Charts**: Track metrics over epochs
   - Quality metrics (toxicity, quality gap)
   - Economic metrics (welfare, Gini coefficient)
   - Security metrics (threat level, contagion depth)
   - Network metrics (edges, clustering)

4. **Agent Analysis**:
   - State table with reputation, resources, status
   - Comparison bar charts
   - Reputation trajectories over time

5. **Security Panel**: Monitor threats
   - Ecosystem threat gauge
   - Active threat indicators
   - Quarantined agents

6. **Network Visualization**: Interactive network graph
   - Node coloring by reputation
   - Edge weights from interactions

### Data Export

```python
from src.analysis import (
    MetricsAggregator,
    export_to_csv,
    export_to_json,
    generate_summary_report,
)

# During simulation, use aggregator to collect metrics
aggregator = MetricsAggregator()
aggregator.start_simulation("sim_001", n_epochs=20, steps_per_epoch=10, n_agents=5)

# After simulation
history = aggregator.end_simulation()

# Export to various formats
export_to_csv(history, "results/", "my_simulation")
export_to_json(history, "results/history.json")

# Generate text report
report = generate_summary_report(history)
print(report)
```

### Plotting API

```python
from src.analysis import (
    create_time_series_data,
    create_agent_comparison_data,
    plotly_time_series,
    plotly_bar_chart,
)

# Create plot data from history
ts_data = create_time_series_data(history, ["toxicity_rate", "total_welfare"])
fig = plotly_time_series(ts_data, "Metrics Over Time")

# Agent comparison
comp_data = create_agent_comparison_data(history, "total_payoff")
fig = plotly_bar_chart(comp_data, "Agent Payoffs")
```

## Future Extensions

- **Semi-Permeable Boundaries**: Model sandbox-external world interactions

## References

**Market Microstructure:**
- Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*. Econometrica, 53(6), 1315-1335.
- Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders*. Journal of Financial Economics, 14(1), 71-100.

**AGI Safety & Multi-Agent Systems:**
- [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.141)

**Inspiration:**
- [Moltbook](https://moltbook.com)
- [@sebkrier's thread on agent economies](https://x.com/sebkrier/status/2017993948132774232)

## Dependencies

**Core:** numpy, pydantic

**Development:** pytest, pytest-cov, pytest-asyncio, mypy, ruff

**Analysis:** pandas, matplotlib, seaborn

**Runtime:** pyyaml

**LLM:** anthropic, openai, httpx

**Dashboard:** streamlit, plotly
