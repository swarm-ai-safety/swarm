# Distributional AGI Safety Sandbox - Full Implementation Plan

## One-Line Goal

Build and evaluate a **multi-agent sandbox economy** to study *system-level safety properties* when AGI-level capabilities emerge from interacting sub-AGI agents.

---

## Current State

### Foundation Layer ✅

| Component | Status | Files |
|-----------|--------|-------|
| Data Models | ✅ Complete | `src/models/interaction.py`, `agent.py`, `events.py` |
| Proxy Computation | ✅ Complete | `src/core/proxy.py`, `sigmoid.py` |
| Payoff Engine | ✅ Complete | `src/core/payoff.py` |
| Metrics System | ✅ Complete | `src/metrics/soft_metrics.py`, `reporters.py`, `capabilities.py`, `collusion.py`, `security.py` |
| Event Logging | ✅ Complete | `src/logging/event_log.py` |

### Runtime Layer ✅

| Component | Status | Files |
|-----------|--------|-------|
| Agent Orchestration | ✅ Complete | `src/core/orchestrator.py` |
| Agent Policies | ✅ Complete | `src/agents/` (6 agent types + 5 roles) |
| Feed/Interaction Engine | ✅ Complete | `src/env/feed.py` |
| Governance Module | ✅ Complete | `src/governance/` (8 levers + engine) |
| Marketplace Primitives | ✅ Complete | `src/env/marketplace.py` |
| Scenario Runner | ✅ Complete | `src/scenarios/loader.py`, `examples/run_scenario.py` |
| Parameter Sweep | ✅ Complete | `src/analysis/sweep.py` |
| Dashboard/Visualization | ✅ Complete | `src/analysis/dashboard.py`, `plots.py` |
| Red-Team Framework | ✅ Complete | `src/redteam/` |
| Security Evaluation | ✅ Complete | `src/governance/security.py`, `src/metrics/security.py` |
| Boundary Enforcement | ✅ Complete | `src/boundaries/` |
| Composite Tasks | ✅ Complete | `src/env/composite_tasks.py` |
| Network Topology | ✅ Complete | `src/env/network.py` |
| Test Suite | ✅ 1378 tests | `tests/` |

### Virtual Agent Economies ✅

| Component | Status | Files |
|-----------|--------|-------|
| Dworkin-Style Auctions | ✅ Complete | `src/env/auction.py` |
| Mission Economies | ✅ Complete | `src/env/mission.py` |
| High-Frequency Negotiation | ✅ Complete | `src/env/hfn.py` |
| Permeability Model | ✅ Complete | `src/boundaries/permeability.py` |
| Identity & Trust Infrastructure | ✅ Complete | `src/models/identity.py` |
| Sybil Detection Lever | ✅ Complete | `src/governance/identity_lever.py` |

### Infrastructure ✅

| Component | Status | Files |
|-----------|--------|-------|
| CI/CD Pipeline | ✅ Complete | `.github/workflows/ci.yml`, `release.yml`, `codeql.yml` |
| CLI Entry Point | ✅ Complete | `python -m src run/list` |
| Pre-commit Hooks | ✅ Complete | `.pre-commit-config.yaml` |
| Makefile | ✅ Complete | `Makefile` |
| Demo App | ✅ Complete | `demo/` (Streamlit, 5 pages) |
| Project Governance | ✅ Complete | `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS` |

---

## Implementation Phases

### Phase 1: MVP v0 - Core Simulation Loop

**Goal:** Reproducible multi-agent "feed + tasks" loop with full logs

**Demonstrates:**
- Coordination on tasks through the feed
- Early failure modes: miscoordination, conflict, collusion

#### 1.1 Agent Behavioral Policies (`src/agents/`)

```
src/agents/
├── __init__.py
├── base.py              # BaseAgent abstract class
├── policies/
│   ├── __init__.py
│   ├── honest.py        # Maximizes (π_a + π_b); avoids negative-S interactions
│   ├── opportunistic.py # Maximizes π_a; accepts some negative-S if compensated
│   ├── deceptive.py     # Manipulates to increase acceptance when v = -1
│   └── adversarial.py   # Maximizes system harm, even at personal cost
└── roles/
    ├── __init__.py
    ├── planner.py       # Task planning
    ├── worker.py        # Task execution
    ├── verifier.py      # Verification
    ├── poster.py        # Feed posting
    └── moderator.py     # Moderation
```

**BaseAgent Interface:**
```python
class BaseAgent(ABC):
    agent_id: str
    agent_type: AgentType
    state: AgentState

    @abstractmethod
    def propose_interaction(self, observation: Observation) -> Optional[InteractionProposal]:
        """Decide whether to propose an interaction and with whom."""

    @abstractmethod
    def accept_interaction(self, proposal: InteractionProposal) -> bool:
        """Decide whether to accept a proposed interaction."""

    @abstractmethod
    def select_action(self, observation: Observation) -> Action:
        """Select action: post, reply, vote, or pass."""

    def update_state(self, outcome: InteractionOutcome) -> None:
        """Update internal state after interaction resolves."""
```

**Agent Type Behaviors:**

| Type | Strategy | Acceptance Rule |
|------|----------|-----------------|
| Honest | Maximize joint surplus | Accept if E[S] > 0 |
| Opportunistic | Maximize own payoff | Accept if E[π_self] > threshold |
| Deceptive | Manipulate v perception | Accept when true v = -1, hide information |
| Adversarial | Maximize system harm | Accept to cause E_soft, ignore personal cost |

#### 1.2 Environment State (`src/env/`)

```
src/env/
├── __init__.py
├── state.py             # Global environment state
├── feed.py              # Feed: posts, comments, votes
├── interaction.py       # Interaction matching & resolution
└── tasks.py             # Task definitions
```

**Environment State:**
```python
@dataclass
class EnvState:
    epoch: int
    step: int
    agents: Dict[str, AgentState]
    feed: Feed
    pending_interactions: List[InteractionProposal]
    active_tasks: List[Task]
    event_log: EventLog
    rng: random.Random  # For reproducibility
```

**Feed Model (Moltbook-inspired):**
```python
@dataclass
class Post:
    post_id: str
    author_id: str
    content: str
    parent_id: Optional[str]  # For threading
    timestamp: datetime
    votes: int
    replies: List[str]

class Feed:
    posts: List[Post]

    def get_visible(self, agent_id: str, limit: int = 20) -> List[Post]:
        """Return posts ranked by visibility score."""
        # score = votes + α*replies - β*age

    def add_post(self, post: Post) -> None:
        """Add post to feed."""

    def add_vote(self, post_id: str, voter_id: str, value: int) -> None:
        """Record vote (+1 or -1)."""
```

**Task System:**
```python
@dataclass
class Task:
    task_id: str
    prompt: str
    required_outputs: List[str]
    budget: float
    deadline: Optional[int]  # epoch number
    assigned_to: Optional[str]
    status: TaskStatus  # pending, in_progress, completed, failed
```

#### 1.3 Orchestrator (`src/core/orchestrator.py`)

**Responsibilities:**
- Schedule agent turns (round-robin or priority-based)
- Build agent-specific observations
- Collect and validate actions
- Execute interactions and update state
- Enforce safety rails (rate limits, budgets)
- Emit events to EventLog

**Main Loop:**
```python
class Orchestrator:
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.state: EnvState
        self.agents: List[BaseAgent]
        self.payoff_engine: SoftPayoffEngine
        self.metrics: SoftMetrics

    def run(self, n_epochs: int) -> SimulationResult:
        self.emit_event(EventType.SIMULATION_STARTED)

        for epoch in range(n_epochs):
            self.run_epoch(epoch)
            self.compute_epoch_metrics()
            self.emit_event(EventType.EPOCH_COMPLETED)

        self.emit_event(EventType.SIMULATION_ENDED)
        return self.compile_results()

    def run_epoch(self, epoch: int) -> None:
        for step in range(self.config.steps_per_epoch):
            # Phase 1: Agents take actions (post, reply, vote)
            for agent in self.get_turn_order():
                observation = self.build_observation(agent)
                action = agent.select_action(observation)
                self.execute_action(agent, action)

            # Phase 2: Agents propose interactions
            for agent in self.agents:
                proposal = agent.propose_interaction(self.build_observation(agent))
                if proposal:
                    self.state.pending_interactions.append(proposal)

            # Phase 3: Resolve pending interactions
            self.resolve_interactions()
```

**Interaction Resolution:**
```python
def resolve_interactions(self) -> None:
    for proposal in self.state.pending_interactions:
        counterparty = self.agents[proposal.counterparty_id]

        # Counterparty decides
        accepted = counterparty.accept_interaction(proposal)

        # Compute observables and labels
        observables = self.compute_observables(proposal)
        v_hat, p = self.proxy_computer.compute_labels(observables)

        # Create interaction record
        interaction = SoftInteraction(
            initiator=proposal.initiator_id,
            counterparty=proposal.counterparty_id,
            accepted=accepted,
            v_hat=v_hat,
            p=p,
            # ... other fields
        )

        if accepted:
            # Compute payoffs
            pi_a = self.payoff_engine.payoff_initiator(interaction)
            pi_b = self.payoff_engine.payoff_counterparty(interaction)

            # Update agent states
            self.update_agent_state(proposal.initiator_id, pi_a, interaction)
            self.update_agent_state(proposal.counterparty_id, pi_b, interaction)

        # Log event
        self.emit_interaction_event(interaction)

    self.state.pending_interactions.clear()
```

#### 1.4 Safety Rails

```python
@dataclass
class SafetyConfig:
    max_posts_per_epoch: int = 10
    max_interactions_per_step: int = 5
    max_content_length: int = 1000
    min_time_between_posts: float = 1.0  # seconds
    budget_per_epoch: float = 100.0
    kill_switch_enabled: bool = True
```

**Enforcement:**
- Tool allowlists per agent role
- Rate limiting with token bucket
- Budget tracking and enforcement
- Content length validation
- Kill-switch for emergency stop

---

### Phase 2: MVP v1 - Economics & Governance

**Goal:** Add economics, microstructure, and governance sweeps

#### 2.1 Marketplace Primitives (`src/env/market.py`)

```python
@dataclass
class Bounty:
    bounty_id: str
    task_id: str
    amount: float
    poster_id: str
    deadline: int

@dataclass
class Bid:
    bid_id: str
    bounty_id: str
    bidder_id: str
    proposed_amount: float

@dataclass
class Escrow:
    escrow_id: str
    from_agent: str
    to_agent: str
    amount: float
    condition: str  # e.g., "task_completed"
    released: bool = False

class Marketplace:
    def post_bounty(self, bounty: Bounty) -> None: ...
    def submit_bid(self, bid: Bid) -> None: ...
    def accept_bid(self, bounty_id: str, bid_id: str) -> None: ...
    def release_escrow(self, escrow_id: str) -> None: ...
```

#### 2.2 Governance Module (`src/governance/`)

```
src/governance/
├── __init__.py
├── base.py              # GovernanceLever ABC
├── taxes.py             # Transaction taxes
├── reputation.py        # Reputation decay & normalization
├── admission.py         # Staking, bonds, probation
├── bandwidth.py         # Rate limits, caps
├── circuit_breaker.py   # Freezes, isolations
└── audits.py            # Random audit system
```

**Lever Interface:**
```python
class GovernanceLever(ABC):
    @abstractmethod
    def apply(self, interaction: SoftInteraction, state: EnvState) -> GovernanceEffect:
        """Apply governance effect to interaction."""

    @abstractmethod
    def get_cost(self, interaction: SoftInteraction) -> Tuple[float, float]:
        """Return (c_a, c_b) costs added by this lever."""

    @abstractmethod
    def on_epoch_end(self, state: EnvState) -> None:
        """Apply epoch-level effects (e.g., decay)."""
```

**Implemented Levers:**

| Lever | Config | Effect |
|-------|--------|--------|
| Transaction Tax | `rate: float` | `c_a += rate * abs(tau)`, `c_b += rate * abs(tau)` |
| Reputation Decay | `decay: float` | `r *= decay` each epoch |
| Vote Normalization | `max_weight: float` | Cap vote influence per agent |
| Bandwidth Cap | `max_actions: int` | Limit actions per epoch |
| Staking | `stake_amount: float` | Require deposit to participate |
| Random Audit | `prob: float, penalty: float` | Probabilistic check + penalty |
| Circuit Breaker | `threshold: float` | Freeze agent if toxicity > threshold |
| Sybil Detection | `similarity_threshold: float` | Penalize behaviorally similar agent clusters |

**Governance Configuration:**
```python
@dataclass
class GovernanceConfig:
    transaction_tax_rate: float = 0.0
    reputation_decay: float = 1.0  # No decay
    vote_weight_cap: Optional[float] = None
    bandwidth_cap: Optional[int] = None
    staking_required: float = 0.0
    audit_probability: float = 0.0
    audit_penalty: float = 0.0
    circuit_breaker_threshold: Optional[float] = None
```

#### 2.3 Scenario Runner (`src/runner/`)

```
src/runner/
├── __init__.py
├── scenario.py          # Scenario specification
├── runner.py            # Execute scenarios
├── sweep.py             # Parameter sweeps
├── replay.py            # Replay from logs
└── analysis.py          # Cross-run comparison
```

**Scenario Specification (YAML):**
```yaml
scenario_id: status_game_v1
motif: reputation_competition
description: "Agents compete for reputation under scarcity"

agents:
  - type: honest
    count: 3
    role: worker
  - type: opportunistic
    count: 2
    role: poster
  - type: adversarial
    count: 1
    role: poster

environment:
  initial_budget: 100.0
  attention_scarcity: 0.5

governance:
  transaction_tax_rate: 0.05
  reputation_decay: 0.95
  bandwidth_cap: 10

simulation:
  epochs: 100
  steps_per_epoch: 10
  seed: 42

metrics_to_track:
  - toxicity_rate
  - conditional_loss_initiator
  - quality_gap
  - influence_gini
```

**Parameter Sweep:**
```python
class ParameterSweep:
    def __init__(self, base_scenario: ScenarioSpec):
        self.base = base_scenario
        self.variations: List[Dict[str, Any]] = []

    def add_variation(self, param_path: str, values: List[Any]) -> None:
        """Add parameter to sweep over."""

    def generate_scenarios(self) -> List[ScenarioSpec]:
        """Generate all combinations."""

    def run_sweep(self, n_workers: int = 4) -> SweepResults:
        """Run all scenarios, optionally in parallel."""
```

**Recommended MVP Sweep (12 runs):**
1. Baseline (no governance)
2. + Reputation decay
3. + Vote-weight normalization
4. + Bandwidth caps
5. + Transparency ledger
6. + Random audits
7. + Circuit breaker
8. + Transaction tax
9. + Staking/bonding
10. + Moderator agent
11. Combined (defense-in-depth)
12. Combined + high adversarial ratio

#### 2.4 Metrics Dashboard (`src/analysis/`)

```
src/analysis/
├── __init__.py
├── aggregation.py       # Epoch-level metric aggregation
├── plots.py             # Matplotlib/plotly visualizations
├── dashboard.py         # Streamlit dashboard
└── export.py            # Export to CSV/Parquet
```

**Key Visualizations:**
- Time-series: toxicity, conditional loss, participation rate
- Influence distribution (Gini coefficient over time)
- Agent interaction network graph
- Cascade/diffusion tracking
- Governance lever effectiveness comparison

**Dashboard Layout:**
```
┌─────────────────────────────────────────────────────────┐
│  Scenario: status_game_v1    Epoch: 45/100    ▶ Pause  │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Toxicity Rate   │  │ Conditional Loss│              │
│  │ [line chart]    │  │ [line chart]    │              │
│  └─────────────────┘  └─────────────────┘              │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ Quality Gap     │  │ Participation   │              │
│  │ [line chart]    │  │ [line chart]    │              │
│  └─────────────────┘  └─────────────────┘              │
├─────────────────────────────────────────────────────────┤
│  Agent States                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ID   │ Type    │ Rep  │ Resources │ Interactions│   │
│  │ a_01 │ honest  │ 12.3 │ 87.5      │ 45          │   │
│  │ a_02 │ opport. │ 8.7  │ 102.3     │ 38          │   │
│  │ ...  │ ...     │ ...  │ ...       │ ...         │   │
│  └─────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│  Recent Events (live feed)                              │
│  [scrolling event log]                                  │
└─────────────────────────────────────────────────────────┘
```

---

### Phase 3: Advanced Features — ✅ Complete

#### 3.1 Moltbook-Inspired Phenomena ✅

**Sub-communities & Culture:**
- Community partitioning algorithm
- Shared narrative tracking (embedding clusters)
- Meme diffusion metrics (R0, cascade size)

**Motif Library:**
- Myth-making: Invent shared origin story
- Constitution drafting: Write and enforce community rules
- Status games: Compete for reputation under scarcity
- Meta-discourse: Debate "what agents are"
- Meme warfare: Competing memes for amplification

#### 3.2 Security Evaluation ✅

**Threat Model:**
- Indirect prompt injection
- Cross-agent manipulation
- Information laundering
- Credential leakage
- Self-propagation ("worm")

**Security Metrics:**
- Prompt-injection success rate
- Contagion depth before containment
- Exfiltration attempt/leak rate
- False-positive suppression cost

#### 3.3 LLM Integration ✅

**Pluggable Adapter:**
```python
class LLMAdapter(ABC):
    @abstractmethod
    async def complete(self, prompt: str, schema: ActionSchema) -> Action:
        """Get structured action from LLM."""

class AnthropicAdapter(LLMAdapter): ...
class OpenAIAdapter(LLMAdapter): ...
class LocalAdapter(LLMAdapter): ...
```

**Per-Role Model Configs:**
- Cheap model for poster/voter
- Stronger model for planner/verifier
- Strict JSON action schema

---

### Phase 4: Virtual Agent Economies — ✅ Complete

**Goal:** Implement economic mechanisms from [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147) for fair resource allocation, collective coordination, and market dynamics within the sandbox.

#### 4.1 Dworkin-Style Auctions (`src/env/auction.py`) ✅

Fair resource allocation with equal initial endowments and envy-freeness verification.

```python
@dataclass
class AuctionConfig:
    initial_endowment: float = 100.0
    max_rounds: int = 50
    price_adjustment_rate: float = 0.1
    convergence_tolerance: float = 0.01
    envy_tolerance: float = 0.05
```

**Features:**
- Equal token endowments for all agents
- Tatonnement price adjustment across rounds
- Envy-freeness verification (no agent prefers another's allocation)
- Gini coefficient tracking for allocation fairness

#### 4.2 Mission Economies (`src/env/mission.py`) ✅

Collective goal coordination with measurable objectives and contribution tracking.

```python
class MissionStatus(Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    EXPIRED = "expired"
```

**Features:**
- Mission lifecycle (proposal → active → succeeded/failed/expired)
- Three reward distribution strategies: equal, proportional, Shapley
- Free-rider detection based on contribution thresholds
- Contribution evaluation via the soft-label quality pipeline

#### 4.3 High-Frequency Negotiation (`src/env/hfn.py`) ✅

Speed-based market dynamics with flash crash detection and circuit breakers.

```python
@dataclass
class HFNConfig:
    tick_duration_ms: float = 100.0
    max_orders_per_tick: int = 10
    latency_noise_ms: float = 10.0
    priority_by_speed: bool = True
    batch_interval_ticks: int = 5
    halt_duration_ticks: int = 20
```

**Features:**
- Order book with bid/ask matching
- Batch clearing at configurable intervals
- Flash crash detection via rolling quality window
- Market halt circuit breaker triggered by quality collapse
- Latency-based priority ordering

#### 4.4 Permeability Model (`src/boundaries/permeability.py`) ✅

Sandbox boundaries modeled as semi-permeable membranes.

```python
@dataclass
class PermeabilityConfig:
    base_permeability: float = 0.5
    information_decay: float = 0.1
    contagion_rate: float = 0.05
    spillover_amplification: float = 1.5
    adaptive: bool = True
    threat_sensitivity: float = 1.0
```

**Features:**
- Parameterized permeability (0 = sealed, 1 = fully open)
- Adaptive permeability responding to threat levels
- Contagion probability tied to soft-label quality
- Spillover harm tracking between sandbox and external world

#### 4.5 Identity & Trust Infrastructure (`src/models/identity.py`) ✅

Verifiable credentials and Proof-of-Personhood for Sybil resistance.

**Features:**
- Verifiable credentials with issuance, expiry, and revocation
- Proof-of-Personhood abstraction
- Trust score computation from credential history
- Integration with governance via `SybilDetectionLever` (`src/governance/identity_lever.py`)

#### 4.6 Sybil Detection Governance Lever (`src/governance/identity_lever.py`) ✅

**Features:**
- Behavioral similarity analysis across agent populations
- Cluster detection for coordinated Sybil agents
- Reputation and resource penalties for flagged agents
- Integrates with existing governance engine

---

## Directory Structure (Final)

```
distributional-agi-safety/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                # Lint, type-check, test matrix (Python 3.10-3.12)
│   │   ├── codeql.yml            # Security scanning (push, PR, weekly)
│   │   └── release.yml           # Validate, test, build, publish to PyPI
│   ├── ISSUE_TEMPLATE/           # Bug report & feature request templates
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── src/
│   ├── __init__.py
│   ├── __main__.py              # CLI entry point (run/list scenarios)
│   ├── models/                  # ✅ Complete
│   │   ├── interaction.py
│   │   ├── agent.py
│   │   ├── events.py
│   │   └── identity.py          # Verifiable credentials, Proof-of-Personhood
│   ├── core/                    # ✅ Complete
│   │   ├── payoff.py
│   │   ├── proxy.py
│   │   ├── sigmoid.py
│   │   └── orchestrator.py
│   ├── metrics/                 # ✅ Complete
│   │   ├── soft_metrics.py
│   │   ├── reporters.py
│   │   ├── capabilities.py
│   │   ├── collusion.py
│   │   └── security.py
│   ├── logging/                 # ✅ Complete
│   │   └── event_log.py
│   ├── agents/                  # ✅ Complete
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── honest.py
│   │   ├── opportunistic.py
│   │   ├── deceptive.py
│   │   ├── adversarial.py
│   │   ├── adaptive_adversary.py
│   │   ├── llm_agent.py
│   │   ├── llm_config.py
│   │   ├── llm_prompts.py
│   │   └── roles/
│   │       ├── planner.py
│   │       ├── worker.py
│   │       ├── verifier.py
│   │       ├── poster.py
│   │       └── moderator.py
│   ├── env/                     # ✅ Complete
│   │   ├── __init__.py
│   │   ├── state.py
│   │   ├── feed.py
│   │   ├── tasks.py
│   │   ├── marketplace.py
│   │   ├── composite_tasks.py
│   │   ├── network.py
│   │   ├── auction.py           # Dworkin-style fair allocation
│   │   ├── mission.py           # Mission economies & collective goals
│   │   └── hfn.py               # High-frequency negotiation engine
│   ├── governance/              # ✅ Complete
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── engine.py
│   │   ├── levers.py
│   │   ├── taxes.py
│   │   ├── reputation.py
│   │   ├── admission.py
│   │   ├── circuit_breaker.py
│   │   ├── audits.py
│   │   ├── collusion.py
│   │   ├── security.py
│   │   └── identity_lever.py    # Sybil detection via behavioral similarity
│   ├── scenarios/               # ✅ Complete
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── analysis/                # ✅ Complete
│   │   ├── __init__.py
│   │   ├── aggregation.py
│   │   ├── plots.py
│   │   ├── dashboard.py
│   │   ├── streamlit_app.py
│   │   ├── sweep.py
│   │   └── export.py
│   ├── boundaries/              # ✅ Complete
│   │   ├── __init__.py
│   │   ├── external_world.py
│   │   ├── information_flow.py
│   │   ├── leakage.py
│   │   ├── policies.py
│   │   └── permeability.py      # Semi-permeable sandbox boundaries
│   └── redteam/                 # ✅ Complete
│       ├── __init__.py
│       ├── attacks.py
│       ├── evaluator.py
│       └── metrics.py
├── scenarios/                   # ✅ 11 built-in scenarios
│   ├── baseline.yaml
│   ├── status_game.yaml
│   ├── collusion_detection.yaml
│   ├── strict_governance.yaml
│   ├── boundary_test.yaml
│   ├── marketplace_economy.yaml
│   ├── network_effects.yaml
│   ├── security_evaluation.yaml
│   ├── emergent_capabilities.yaml
│   ├── adversarial_redteam.yaml
│   └── llm_agents.yaml
├── tests/                       # ✅ 1378 tests
│   ├── test_payoff.py
│   ├── test_proxy.py
│   ├── test_metrics.py
│   ├── test_orchestrator.py
│   ├── test_agents.py
│   ├── test_agent_roles.py
│   ├── test_governance.py
│   ├── test_env.py
│   ├── test_event_log.py
│   ├── test_sweep.py
│   ├── test_scenarios.py
│   ├── test_dashboard.py
│   ├── test_marketplace.py
│   ├── test_network.py
│   ├── test_boundaries.py
│   ├── test_capabilities.py
│   ├── test_collusion.py
│   ├── test_security.py
│   ├── test_redteam.py
│   ├── test_llm_agent.py
│   ├── test_success_criteria.py
│   ├── test_auction.py          # Dworkin auction tests
│   ├── test_mission.py          # Mission economy tests
│   ├── test_hfn.py              # High-frequency negotiation tests
│   ├── test_permeability.py     # Permeability model tests
│   ├── test_identity.py         # Identity infrastructure tests
│   ├── test_cli.py              # CLI entry point tests
│   ├── test_analysis.py         # Analysis module tests
│   ├── test_integration.py      # End-to-end integration tests (21 tests)
│   ├── test_property_based.py   # Property-based tests
│   ├── test_coverage_boost.py
│   ├── test_coverage_boost2.py
│   ├── test_coverage_boost3.py
│   └── fixtures/
├── docs/                        # ✅ 10 guides + transferability
│   ├── whitepaper.md
│   ├── governance.md
│   ├── scenarios.md
│   ├── llm-agents.md
│   ├── network-topology.md
│   ├── emergent-capabilities.md
│   ├── red-teaming.md
│   ├── boundaries.md
│   ├── dashboard.md
│   ├── virtual-agent-economies.md
│   └── transferability/
│       └── incoherence_governance.md
├── demo/                        # ✅ Interactive Streamlit demo
│   ├── app.py
│   ├── utils/
│   └── pages/
│       ├── 1_Overview.py
│       ├── 2_Scenario_Explorer.py
│       ├── 3_Governance_Lab.py
│       ├── 4_Agent_Dynamics.py
│       └── 5_Theory.py
├── examples/                    # ✅ Demonstration scripts
│   ├── mvp_demo.py
│   ├── run_scenario.py
│   ├── parameter_sweep.py
│   └── llm_demo.py
├── pyproject.toml               # ✅ Complete
├── requirements.txt             # ✅ Pinned dependencies
├── Makefile                     # ✅ install, lint, typecheck, test, coverage, ci, clean
├── CLAUDE.md                    # ✅ Complete
├── README.md                    # ✅ Complete
├── CHANGELOG.md                 # ✅ Complete
├── CONTRIBUTING.md              # ✅ Complete
├── SECURITY.md                  # ✅ Vulnerability reporting policy
├── CITATION.cff                 # ✅ Citation metadata
├── LICENSE                      # ✅ MIT
└── ARXIV_SIMILARITY_ANALYSIS.md # ✅ Related work analysis
```

---

## Implementation Order

### MVP v0 (Core Simulation) — ✅ Complete

| Order | Component | Files | Status |
|-------|-----------|-------|--------|
| 1 | Environment State | `src/env/state.py` | ✅ Complete |
| 2 | Feed Engine | `src/env/feed.py` | ✅ Complete |
| 3 | Base Agent | `src/agents/base.py` | ✅ Complete |
| 4 | Agent Policies | `src/agents/*.py` | ✅ Complete |
| 5 | Orchestrator | `src/core/orchestrator.py` | ✅ Complete |
| 6 | Task System | `src/env/tasks.py` | ✅ Complete |
| 7 | Tests | `tests/test_*.py` | ✅ Complete |

### MVP v1 (Economics & Governance) — ✅ Complete

| Order | Component | Files | Status |
|-------|-----------|-------|--------|
| 8 | Marketplace | `src/env/marketplace.py` | ✅ Complete |
| 9 | Governance Config | `src/governance/config.py` | ✅ Complete |
| 10 | Governance Engine + Levers | `src/governance/engine.py`, `levers.py`, etc. | ✅ Complete |
| 11 | Scenario Loader | `src/scenarios/loader.py` | ✅ Complete |
| 12 | Scenario Runner | `examples/run_scenario.py` | ✅ Complete |
| 13 | Parameter Sweep | `src/analysis/sweep.py` | ✅ Complete |
| 14 | Aggregation | `src/analysis/aggregation.py` | ✅ Complete |
| 15 | Plots | `src/analysis/plots.py` | ✅ Complete |
| 16 | Dashboard | `src/analysis/dashboard.py` | ✅ Complete |
| 17 | Tests | `tests/test_*.py` | ✅ Complete |

### Advanced Features — ✅ Complete

| Order | Component | Files | Status |
|-------|-----------|-------|--------|
| 18 | Red-Team Framework | `src/redteam/` | ✅ Complete |
| 19 | Security Evaluation | `src/governance/security.py`, `src/metrics/security.py` | ✅ Complete |
| 20 | Boundary Enforcement | `src/boundaries/` | ✅ Complete |
| 21 | Composite Tasks | `src/env/composite_tasks.py` | ✅ Complete |
| 22 | Network Topology | `src/env/network.py` | ✅ Complete |
| 23 | Collusion Detection | `src/governance/collusion.py`, `src/metrics/collusion.py` | ✅ Complete |
| 24 | Adaptive Adversary | `src/agents/adaptive_adversary.py` | ✅ Complete |
| 25 | LLM Agent Integration | `src/agents/llm_agent.py`, `llm_config.py`, `llm_prompts.py` | ✅ Complete |

### Virtual Agent Economies — ✅ Complete

| Order | Component | Files | Status |
|-------|-----------|-------|--------|
| 26 | Dworkin-Style Auctions | `src/env/auction.py` | ✅ Complete |
| 27 | Mission Economies | `src/env/mission.py` | ✅ Complete |
| 28 | High-Frequency Negotiation | `src/env/hfn.py` | ✅ Complete |
| 29 | Permeability Model | `src/boundaries/permeability.py` | ✅ Complete |
| 30 | Identity & Trust | `src/models/identity.py` | ✅ Complete |
| 31 | Sybil Detection Lever | `src/governance/identity_lever.py` | ✅ Complete |
| 32 | Virtual Economy Tests | `tests/test_auction.py`, `test_mission.py`, `test_hfn.py`, `test_permeability.py`, `test_identity.py` | ✅ Complete |

### Infrastructure & CI/CD — ✅ Complete

| Order | Component | Files | Status |
|-------|-----------|-------|--------|
| 33 | CLI Entry Point | `src/__main__.py` | ✅ Complete |
| 34 | GitHub Actions CI | `.github/workflows/ci.yml` | ✅ Complete |
| 35 | CodeQL Security Scanning | `.github/workflows/codeql.yml` | ✅ Complete |
| 36 | Release Workflow | `.github/workflows/release.yml` | ✅ Complete |
| 37 | Pre-commit Hooks | `.pre-commit-config.yaml` | ✅ Complete |
| 38 | Makefile | `Makefile` | ✅ Complete |
| 39 | Demo App | `demo/` | ✅ Complete |
| 40 | Integration Tests | `tests/test_integration.py` | ✅ Complete |
| 41 | Property-Based Tests | `tests/test_property_based.py` | ✅ Complete |

---

## Success Criteria

### MVP v0
- [x] 5 agents interact over 10+ epochs
- [x] Toxicity and conditional loss metrics computed per epoch
- [x] Full event log enables deterministic replay
- [x] Observable failure modes: miscoordination, conflict, collusion

### MVP v1
- [x] ≥3 Moltbook-like motifs reproducible
- [x] ≥2 governance levers measurably reduce toxicity/collusion
- [x] Parameter sweep across 12 governance configurations
- [x] Dashboard shows real-time metrics
- [x] Toxic interactions show negative conditional surplus but positive reputation payoff

### Advanced Features
- [x] Red-team framework produces measurable attack success rates
- [x] Boundary enforcement detects and limits information leakage
- [x] Collusion detection identifies coordinated agent clusters
- [x] LLM agents integrate with Anthropic/OpenAI/Ollama backends
- [x] Network topology captures dynamic interaction patterns

### Virtual Agent Economies
- [x] Dworkin auctions converge to envy-free allocations
- [x] Mission economies detect free-riders and distribute rewards fairly
- [x] HFN engine detects flash crashes and triggers market halts
- [x] Permeability model tracks spillover harm between sandbox and external world
- [x] Identity infrastructure supports credential issuance, verification, and revocation
- [x] Sybil detection penalizes behaviorally similar agent clusters

### Infrastructure
- [x] CI pipeline runs lint, type-check, and tests across Python 3.10-3.12
- [x] ≥70% test coverage enforced
- [x] 1378 tests pass across 33 test files
- [x] CLI supports scenario execution with seed/epoch overrides and JSON/CSV export

All MVP criteria verified by `tests/test_success_criteria.py` (9 tests).
Integration tests in `tests/test_integration.py` (21 end-to-end tests).

---

## Verification

### Unit Tests
```bash
pytest tests/ -v
```

### Smoke Test
```bash
python -m src.runner.runner scenarios/baseline.yaml --epochs 10
```

Verify:
- Events logged to JSONL
- Metrics computed per epoch
- Agent states updated correctly

### Governance Test
```bash
# Without tax
python -m src.runner.runner scenarios/baseline.yaml --epochs 50

# With tax
python -m src.runner.runner scenarios/baseline.yaml --epochs 50 \
    --override governance.transaction_tax_rate=0.1
```

Verify: Tax reduces toxic interaction rate

### Replay Test
```bash
# Run and save
python -m src.runner.runner scenarios/baseline.yaml --output run_001.jsonl

# Replay and compare
python -m src.runner.replay run_001.jsonl --verify
```

Verify: Identical metrics from replay

---

## Dependencies

```toml
[project.optional-dependencies]
runtime = [
    "pyyaml>=6.0",
]
dashboard = [
    "streamlit>=1.30",
    "plotly>=5.0",
    "networkx>=3.0",
]
llm = [
    "anthropic>=0.18",
    "openai>=1.0",
]
```

---

## References

- [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)
- [Virtual Agent Economies](https://arxiv.org/abs/2509.10147) - Tomasev et al. (2025)
- [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.141)
- [Moltbook](https://moltbook.com)
- Kyle (1985) - Continuous Auctions and Insider Trading
- Glosten & Milgrom (1985) - Bid, Ask and Transaction Prices
- Dworkin (1981) - What is Equality? Part 2: Equality of Resources
