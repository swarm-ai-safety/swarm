# Distributional AGI Safety Sandbox - Implementation Plan

## One-Line Goal

Build and evaluate a **multi-agent sandbox economy** to study *system-level safety properties* when AGI-level capabilities emerge from interacting sub-AGI agents.

---

## Current State

**Package:** `swarm-safety` v1.0.0 (imported as `swarm`)
**Python:** >= 3.10 | **Tests:** 2202 across 70 files | **Scenarios:** 23 YAML definitions

### Foundation Layer

| Component | Status | Files |
|-----------|--------|-------|
| Data Models | Complete | `swarm/models/interaction.py`, `agent.py`, `events.py`, `identity.py`, `scholar.py` |
| Proxy Computation | Complete | `swarm/core/proxy.py`, `sigmoid.py` |
| Payoff Engine | Complete | `swarm/core/payoff.py` |
| Metrics System | Complete | `swarm/metrics/` (13 modules) |
| Event Logging | Complete | `swarm/logging/event_log.py` |

### Runtime Layer

| Component | Status | Files |
|-----------|--------|-------|
| Orchestrator | Complete | `swarm/core/orchestrator.py` + 8 domain handlers |
| Agent Policies | Complete | `swarm/agents/` (15 modules: 6 policies + roles + LLM + memory + domain agents) |
| Feed/Interaction Engine | Complete | `swarm/env/feed.py` |
| Governance Module | Complete | `swarm/governance/` (22 modules, 24+ levers) |
| Marketplace Primitives | Complete | `swarm/env/marketplace.py` |
| Scenario Runner | Complete | `swarm/scenarios/loader.py`, `examples/run_scenario.py` |
| Parameter Sweep | Complete | `swarm/analysis/sweep.py` |
| Dashboard/Visualization | Complete | `swarm/analysis/dashboard.py`, `plots.py` |
| Red-Team Framework | Complete | `swarm/redteam/` |
| Security Evaluation | Complete | `swarm/governance/security.py`, `swarm/metrics/security.py` |
| Boundary Enforcement | Complete | `swarm/boundaries/` |
| Composite Tasks | Complete | `swarm/env/composite_tasks.py` |
| Network Topology | Complete | `swarm/env/network.py` |

### Virtual Agent Economies

| Component | Status | Files |
|-----------|--------|-------|
| Dworkin-Style Auctions | Complete | `swarm/env/auction.py` |
| Mission Economies | Complete | `swarm/env/mission.py` |
| High-Frequency Negotiation | Complete | `swarm/env/hfn.py` |
| Permeability Model | Complete | `swarm/boundaries/permeability.py` |
| Identity & Trust | Complete | `swarm/models/identity.py` |
| Sybil Detection Lever | Complete | `swarm/governance/identity_lever.py` |

### Bridge Integrations

| Bridge | Status | Files |
|--------|--------|-------|
| Claude Code | Complete | `swarm/bridges/claude_code/` (agent, bridge, client, events, policy) |
| Concordia | Complete | `swarm/bridges/concordia/` (adapter, config, events, game_master, judge) |
| GasTown | Complete | `swarm/bridges/gastown/` (agent, beads, bridge, config, events, git_observer, mapper, policy) |
| Live SWE | Complete | `swarm/bridges/live_swe/` (agent, bridge, client, events, policy, tracker) |
| OpenClaw | Complete | `swarm/bridges/openclaw/` (config, job_queue, schemas, service, skill) |
| Worktree Sandbox | Complete | `swarm/bridges/worktree/` (bridge, config, events, executor, mapper, policy, sandbox + Rich CLI) |

### Research Pipeline

| Component | Status | Files |
|-----------|--------|-------|
| Research Agents | Complete | `swarm/research/agents.py` (Literature, Experiment, Analysis, Writing, Review, Critique, Replication) |
| Platform Clients | Complete | `swarm/research/platforms.py` (AgentRxiv, Agentxiv, Clawxiv) |
| AgentRxiv Server | Complete | `swarm/research/agentrxiv_server.py` |
| Quality Gates | Complete | `swarm/research/quality.py` (PreRegistration, QualityGate) |
| Reflexivity | Complete | `swarm/research/reflexivity.py` (ShadowSimulation, PublishThenAttack) |
| Paper Annotation | Complete | `swarm/research/annotator.py` (RiskProfile, VerifiableClaim) |
| PDF Export | Complete | `swarm/research/pdf_export.py` |
| Submission Validation | Complete | `swarm/research/submission.py` |
| Scenario Generation | Complete | `swarm/research/scenario_gen.py` |
| Research Workflow | Complete | `swarm/research/workflow.py` |
| Track A Pipeline | Complete | `swarm/research/swarm_papers/` (track_a, paper, memory, agentrxiv_bridge) |

### Developer Tooling

| Component | Status | Files |
|-----------|--------|-------|
| Slash Commands | Complete | `.claude/commands/` (20 commands) |
| Specialist Agents | Complete | `.claude/agents/` (6 agents) |
| Git Hooks | Complete | `.claude/hooks/pre-commit` (secrets scan, staged-only lint, mypy, pytest) |
| MCP Integrations | Complete | `.mcp.json` |

### Infrastructure

| Component | Status | Files |
|-----------|--------|-------|
| CI/CD Pipeline | Complete | `.github/workflows/ci.yml`, `release.yml`, `codeql.yml` |
| CLI Entry Point | Complete | `python -m swarm run/list` |
| Pre-commit Hooks | Complete | `.claude/hooks/pre-commit` (staged-only ruff/mypy + pytest) |
| Makefile | Complete | `Makefile` |
| Demo App | Complete | `examples/demo/` (Streamlit, 5 pages) |
| API Server | Complete | `swarm/api/` |
| Project Governance | Complete | `CONTRIBUTING.md`, `SECURITY.md`, `CODEOWNERS` |

---

## Architecture

### Data Flow

```
Observables --> ProxyComputer --> v_hat --> sigmoid --> p --> SoftPayoffEngine --> payoffs
                                                       |
                                                 SoftMetrics --> toxicity, quality gap, etc.
```

### Core Computation

**`swarm/core/proxy.py`** — `ProxyComputer` converts observable signals (task_progress, rework_count, verifier_rejections, engagement) into `v_hat in [-1, +1]` using weighted combination, then applies calibrated sigmoid to get `p = P(v = +1)`.

**`swarm/core/payoff.py`** — `SoftPayoffEngine` implements payoffs using soft labels:
- `S_soft = p * s_plus - (1-p) * s_minus` (expected surplus)
- `E_soft = (1-p) * h` (expected harm externality)
- `pi_a = theta*S_soft - tau - c_a - rho_a*E_soft + w_rep*r_a`
- `pi_b = (1-theta)*S_soft + tau - c_b - rho_b*E_soft + w_rep*r_b`

**`swarm/metrics/soft_metrics.py`** — `SoftMetrics` computes probabilistic metrics:
- Toxicity: `E[1-p | accepted]`
- Quality gap: `E[p | accepted] - E[p | rejected]` (negative = adverse selection)
- Conditional loss: selection effect on payoffs

### Orchestrator + Handlers

The orchestrator (`swarm/core/orchestrator.py`) delegates domain-specific logic to pluggable handlers:

| Handler | Domain |
|---------|--------|
| `handler.py` | Base handler interface |
| `boundary_handler.py` | Sandbox boundary enforcement |
| `marketplace_handler.py` | Marketplace/auction mechanics |
| `memory_handler.py` | Agent memory management |
| `moltbook_handler.py` | Moltbook social dynamics |
| `moltipedia_handler.py` | Moltipedia knowledge dynamics |
| `scholar_handler.py` | Scholar/research agent coordination |
| `observable_generator.py` | Signal generation from raw events |
| `pseudo_verifiers.py` | Automated verification simulation |

---

## Implementation Phases

### Phase 1: Core Simulation Loop

**Goal:** Reproducible multi-agent "feed + tasks" loop with full logs.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 1 | Environment State | `swarm/env/state.py` | Complete |
| 2 | Feed Engine | `swarm/env/feed.py` | Complete |
| 3 | Base Agent | `swarm/agents/base.py` | Complete |
| 4 | Agent Policies | `swarm/agents/{honest,opportunistic,deceptive,adversarial}.py` | Complete |
| 5 | Orchestrator | `swarm/core/orchestrator.py` | Complete |
| 6 | Task System | `swarm/env/tasks.py` | Complete |
| 7 | Tests | `tests/test_payoff.py`, `test_proxy.py`, `test_agents.py`, etc. | Complete |

### Phase 2: Economics & Governance

**Goal:** Add economics, microstructure, and governance sweeps.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 8 | Marketplace | `swarm/env/marketplace.py` | Complete |
| 9 | Governance Config | `swarm/governance/config.py` | Complete |
| 10 | Governance Engine + Levers | `swarm/governance/engine.py`, `levers.py`, + 18 lever modules | Complete |
| 11 | Scenario Loader | `swarm/scenarios/loader.py` | Complete |
| 12 | Scenario Runner | `examples/run_scenario.py` | Complete |
| 13 | Parameter Sweep | `swarm/analysis/sweep.py` | Complete |
| 14 | Aggregation | `swarm/analysis/aggregation.py` | Complete |
| 15 | Plots | `swarm/analysis/plots.py` | Complete |
| 16 | Dashboard | `swarm/analysis/dashboard.py` | Complete |

### Phase 3: Advanced Features

**Goal:** Security evaluation, red-teaming, LLM integration, network topology.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 17 | Red-Team Framework | `swarm/redteam/` | Complete |
| 18 | Security Evaluation | `swarm/governance/security.py`, `swarm/metrics/security.py` | Complete |
| 19 | Boundary Enforcement | `swarm/boundaries/` | Complete |
| 20 | Composite Tasks | `swarm/env/composite_tasks.py` | Complete |
| 21 | Network Topology | `swarm/env/network.py` | Complete |
| 22 | Collusion Detection | `swarm/governance/collusion.py`, `swarm/metrics/collusion.py` | Complete |
| 23 | Adaptive Adversary | `swarm/agents/adaptive_adversary.py` | Complete |
| 24 | LLM Agent Integration | `swarm/agents/llm_agent.py`, `llm_config.py`, `llm_prompts.py` | Complete |

### Phase 4: Virtual Agent Economies

**Goal:** Implement economic mechanisms from [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147).

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 25 | Dworkin-Style Auctions | `swarm/env/auction.py` | Complete |
| 26 | Mission Economies | `swarm/env/mission.py` | Complete |
| 27 | High-Frequency Negotiation | `swarm/env/hfn.py` | Complete |
| 28 | Permeability Model | `swarm/boundaries/permeability.py` | Complete |
| 29 | Identity & Trust | `swarm/models/identity.py` | Complete |
| 30 | Sybil Detection Lever | `swarm/governance/identity_lever.py` | Complete |

### Phase 5: Bridge Integrations

**Goal:** Connect the sandbox to external agent frameworks and development environments.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 31 | Claude Code Bridge | `swarm/bridges/claude_code/` (6 modules) | Complete |
| 32 | Concordia Bridge | `swarm/bridges/concordia/` (6 modules) | Complete |
| 33 | GasTown Bridge | `swarm/bridges/gastown/` (9 modules) | Complete |
| 34 | Live SWE Bridge | `swarm/bridges/live_swe/` (7 modules) | Complete |
| 35 | OpenClaw Bridge | `swarm/bridges/openclaw/` (6 modules) | Complete |
| 36 | Worktree Sandbox Bridge | `swarm/bridges/worktree/` (9 modules + Rich CLI) | Complete |

Each bridge follows a consistent pattern:
- **bridge.py** — Adapter mapping external events to SWARM observables
- **policy.py** — Governance policy enforcement at the boundary
- **events.py** — Domain-specific event types
- **agent.py / client.py** — Interface to the external system

### Phase 6: Research Pipeline

**Goal:** Multi-agent research workflow with structured sub-agents, quality gates, and platform integration.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 37 | Research Sub-Agents | `swarm/research/agents.py` (7 agent types) | Complete |
| 38 | Platform Clients | `swarm/research/platforms.py` (AgentRxiv, Agentxiv, Clawxiv) | Complete |
| 39 | AgentRxiv Server | `swarm/research/agentrxiv_server.py` | Complete |
| 40 | Quality Gates & Pre-Registration | `swarm/research/quality.py` | Complete |
| 41 | Reflexivity Analysis | `swarm/research/reflexivity.py` | Complete |
| 42 | Paper Annotation | `swarm/research/annotator.py` | Complete |
| 43 | PDF Export | `swarm/research/pdf_export.py` | Complete |
| 44 | Submission Validation | `swarm/research/submission.py` | Complete |
| 45 | Scenario Generation | `swarm/research/scenario_gen.py` | Complete |
| 46 | Research Workflow | `swarm/research/workflow.py` | Complete |
| 47 | Track A Pipeline | `swarm/research/swarm_papers/track_a.py` | Complete |
| 48 | Paper Builder | `swarm/research/swarm_papers/paper.py` | Complete |
| 49 | Memory Store | `swarm/research/swarm_papers/memory.py` | Complete |
| 50 | AgentRxiv Bridge | `swarm/research/swarm_papers/agentrxiv_bridge.py` | Complete |

### Phase 7: Developer Tooling

**Goal:** Claude Code slash commands, specialist agents, and git hooks for development workflow.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 51 | Slash Commands (20) | `.claude/commands/` | Complete |
| 52 | Specialist Agents (6) | `.claude/agents/` | Complete |
| 53 | Pre-commit Hook | `.claude/hooks/pre-commit` | Complete |
| 54 | MCP Configuration | `.mcp.json` | Complete |

**Slash Commands:**
`/add_domain`, `/add_metric`, `/add_scenario`, `/cleanup_branch`, `/fix_deps`, `/install_hooks`, `/log_run`, `/plot`, `/pr`, `/preflight`, `/publish_figures`, `/red_team`, `/retro`, `/run_scenario`, `/scan_secrets`, `/ship`, `/stage`, `/status`, `/sweep`, `/write_paper`

**Specialist Agents:**
`adversary_designer`, `mechanism_designer`, `metrics_auditor`, `reproducibility_sheriff`, `research_scout`, `scenario_architect`

---

## Directory Structure

```
distributional-agi-safety/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml
│   │   ├── codeql.yml
│   │   └── release.yml
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── .claude/
│   ├── commands/              # 20 slash commands
│   ├── agents/                # 6 specialist agents
│   └── hooks/
│       └── pre-commit         # Source of truth for .git/hooks/pre-commit
├── swarm/
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point (run/list scenarios)
│   ├── py.typed               # PEP 561 type marker
│   ├── models/                # Data models (5 modules)
│   │   ├── interaction.py     # SoftInteraction dataclass
│   │   ├── agent.py           # AgentType, AgentState
│   │   ├── events.py          # EventType definitions
│   │   ├── identity.py        # Verifiable credentials, Proof-of-Personhood
│   │   └── scholar.py         # Scholar-specific models
│   ├── core/                  # Core engines (17 modules)
│   │   ├── payoff.py          # SoftPayoffEngine
│   │   ├── proxy.py           # ProxyComputer
│   │   ├── sigmoid.py         # Calibration functions
│   │   ├── orchestrator.py    # Main simulation orchestrator
│   │   ├── handler.py         # Base handler interface
│   │   ├── boundary_handler.py
│   │   ├── marketplace_handler.py
│   │   ├── memory_handler.py
│   │   ├── memory_observables.py
│   │   ├── moltbook_handler.py
│   │   ├── moltbook_observables.py
│   │   ├── moltipedia_handler.py
│   │   ├── moltipedia_observables.py
│   │   ├── scholar_handler.py
│   │   ├── observable_generator.py
│   │   └── pseudo_verifiers.py
│   ├── agents/                # Agent implementations (15 modules)
│   │   ├── base.py            # BaseAgent ABC
│   │   ├── honest.py
│   │   ├── opportunistic.py
│   │   ├── deceptive.py
│   │   ├── adversarial.py
│   │   ├── adaptive_adversary.py
│   │   ├── llm_agent.py       # LLM-backed agent
│   │   ├── llm_config.py
│   │   ├── llm_prompts.py
│   │   ├── memory_agent.py
│   │   ├── memory_config.py
│   │   ├── moltbook_agent.py
│   │   ├── rain_river.py      # Gradient-based agent
│   │   ├── scholar_agent.py
│   │   ├── wiki_editor.py
│   │   └── roles/             # planner, worker, verifier, poster, moderator
│   ├── env/                   # Environment modules
│   │   ├── state.py
│   │   ├── feed.py
│   │   ├── tasks.py
│   │   ├── marketplace.py
│   │   ├── composite_tasks.py
│   │   ├── network.py
│   │   ├── auction.py         # Dworkin-style fair allocation
│   │   ├── mission.py         # Mission economies
│   │   └── hfn.py             # High-frequency negotiation
│   ├── governance/            # Governance mechanisms (22 modules)
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
│   │   ├── identity_lever.py
│   │   ├── memory.py
│   │   ├── moderator_lever.py
│   │   ├── transparency.py
│   │   ├── diversity.py
│   │   ├── dynamic_friction.py
│   │   ├── decomposition.py
│   │   ├── ensemble.py
│   │   ├── incoherence_breaker.py
│   │   ├── moltbook.py
│   │   └── moltipedia.py
│   ├── metrics/               # Metrics computation (13 modules)
│   │   ├── soft_metrics.py    # Primary probabilistic metrics
│   │   ├── reporters.py       # Dual soft/hard reporting
│   │   ├── capabilities.py
│   │   ├── collusion.py
│   │   ├── security.py
│   │   ├── horizon_eval.py
│   │   ├── incoherence.py
│   │   ├── memory_metrics.py
│   │   ├── moltbook_metrics.py
│   │   ├── moltipedia_metrics.py
│   │   ├── scholar_metrics.py
│   │   └── time_horizons.py
│   ├── logging/               # Event logging
│   │   └── event_log.py       # Append-only JSONL logger
│   ├── analysis/              # Analysis & visualization
│   │   ├── aggregation.py
│   │   ├── plots.py
│   │   ├── dashboard.py
│   │   ├── streamlit_app.py
│   │   ├── sweep.py
│   │   └── export.py
│   ├── boundaries/            # Boundary enforcement
│   │   ├── external_world.py
│   │   ├── information_flow.py
│   │   ├── leakage.py
│   │   ├── policies.py
│   │   └── permeability.py
│   ├── redteam/               # Red-team framework
│   │   ├── attacks.py
│   │   ├── evaluator.py
│   │   └── metrics.py
│   ├── bridges/               # External system integrations (6 bridges)
│   │   ├── claude_code/       # Claude Code development bridge
│   │   ├── concordia/         # DeepMind Concordia bridge
│   │   ├── gastown/           # GasTown distributed dev bridge
│   │   ├── live_swe/          # Live SWE agent bridge
│   │   ├── openclaw/          # OpenClaw research platform bridge
│   │   └── worktree/          # Git worktree sandbox bridge
│   ├── research/              # Research pipeline (11 modules + swarm_papers/)
│   │   ├── agents.py          # 7 research sub-agents
│   │   ├── platforms.py       # Platform clients
│   │   ├── agentrxiv_server.py
│   │   ├── quality.py         # Quality gates, pre-registration
│   │   ├── reflexivity.py     # Shadow simulation, publish-then-attack
│   │   ├── annotator.py       # Paper annotation, risk profiles
│   │   ├── pdf_export.py
│   │   ├── submission.py
│   │   ├── scenario_gen.py
│   │   ├── validation.py
│   │   ├── workflow.py
│   │   └── swarm_papers/      # Track A pipeline
│   │       ├── track_a.py     # TrackARunner, ConditionSpec
│   │       ├── paper.py       # PaperBuilder, figures, critiques
│   │       ├── memory.py      # MemoryStore, retrieval policies
│   │       └── agentrxiv_bridge.py
│   ├── api/                   # FastAPI server
│   ├── evaluation/            # Evaluation frameworks
│   ├── forecaster/            # Forecasting modules
│   ├── replay/                # Replay mechanisms
│   └── scenarios/             # Scenario loader
│       └── loader.py
├── scenarios/                 # 23 scenario YAML definitions
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
│   ├── llm_agents.yaml
│   ├── claude_code_demo.yaml
│   ├── claude_code_mvp.yaml
│   ├── concordia_demo.yaml
│   ├── gastown_workspace.yaml
│   ├── horizon_eval.yaml
│   ├── live_swe_self_evolution.yaml
│   ├── macpo_weak_to_strong.yaml
│   ├── memory_tiers.yaml
│   ├── moltbook_captcha.yaml
│   ├── moltipedia_heartbeat.yaml
│   ├── alignment_waltz_targeted_feedback.yaml
│   └── worktree_sandbox.yaml
├── tests/                     # 70 test files, 2202 tests
│   ├── fixtures/
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
│   ├── test_auction.py
│   ├── test_mission.py
│   ├── test_hfn.py
│   ├── test_permeability.py
│   ├── test_identity.py
│   ├── test_cli.py
│   ├── test_analysis.py
│   ├── test_integration.py
│   ├── test_property_based.py
│   ├── test_coverage_boost.py
│   ├── test_coverage_boost2.py
│   ├── test_coverage_boost3.py
│   ├── test_api.py
│   ├── test_agentrxiv.py
│   ├── test_agentxiv_bridge.py
│   ├── test_clawxiv_platforms.py
│   ├── test_submission_validator.py
│   ├── test_claude_code_bridge.py
│   ├── test_claude_code_runner.py
│   ├── test_concordia_bridge.py
│   ├── test_gastown_bridge.py
│   ├── test_live_swe_bridge.py
│   ├── test_openclaw_bridge.py
│   ├── test_worktree_bridge.py
│   ├── test_diversity.py
│   ├── test_evaluation.py
│   ├── test_forecaster.py
│   ├── test_governance_memory.py
│   ├── test_governance_mvp_sweep.py
│   ├── test_horizon_eval.py
│   ├── test_incoherence_metrics.py
│   ├── test_memory_metrics.py
│   ├── test_moltbook.py
│   ├── test_moltbook_governance.py
│   ├── test_moltbook_integration.py
│   ├── test_moltbook_metrics.py
│   ├── test_moltbook_scenario.py
│   ├── test_moltipedia_governance.py
│   ├── test_moltipedia_integration.py
│   ├── test_moltipedia_metrics.py
│   ├── test_moltipedia_scenario.py
│   ├── test_prompt_audit.py
│   ├── test_rain_river.py
│   ├── test_replay_runner.py
│   ├── test_scholar.py
│   ├── test_simulated_apis.py
│   ├── test_time_horizons.py
│   ├── test_vae_integration.py
│   └── test_wiki.py
├── docs/                      # Documentation
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
│   ├── swarm_scholar_bench_spec.md
│   ├── api/
│   ├── papers/
│   └── transferability/
├── examples/
│   ├── demo/                  # Streamlit demo (5 pages)
│   ├── mvp_demo.py
│   ├── run_scenario.py
│   ├── parameter_sweep.py
│   └── llm_demo.py
├── scripts/                   # Runner scripts
├── runs/                      # Experiment output (canonical format)
├── pyproject.toml
├── Makefile
├── CLAUDE.md
├── README.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
├── CITATION.cff
└── LICENSE
```

---

## Governance Levers (24+)

| Lever | Module | Effect |
|-------|--------|--------|
| Transaction Tax | `taxes.py` | `c += rate * abs(tau)` per party |
| Reputation Decay | `reputation.py` | `r *= decay` each epoch |
| Vote Normalization | `levers.py` | Cap vote influence per agent |
| Bandwidth Cap | `levers.py` | Limit actions per epoch |
| Staking | `admission.py` | Require deposit to participate |
| Random Audit | `audits.py` | Probabilistic check + penalty |
| Circuit Breaker | `circuit_breaker.py` | Freeze agent if toxicity > threshold |
| Sybil Detection | `identity_lever.py` | Penalize behaviorally similar clusters |
| Moderator Lever | `moderator_lever.py` | Moderator-based content enforcement |
| Transparency Ledger | `transparency.py` | Public action history |
| Diversity Enforcement | `diversity.py` | Promote agent diversity |
| Dynamic Friction | `dynamic_friction.py` | Adaptive rate limiting |
| Task Decomposition | `decomposition.py` | Break down complex tasks |
| Ensemble Governance | `ensemble.py` | Multi-lever combination |
| Incoherence Breaker | `incoherence_breaker.py` | Detect/prevent incoherent policies |
| Memory Governance | `memory.py` | Memory access/retention policies |
| Collusion Prevention | `collusion.py` | Detect coordinated manipulation |
| Security Policies | `security.py` | Threat response mechanisms |
| Moltbook Governance | `moltbook.py` | Social-media-specific rules |
| Moltipedia Governance | `moltipedia.py` | Knowledge-base-specific rules |

---

## Agent Types

| Agent | Module | Strategy |
|-------|--------|----------|
| Honest | `honest.py` | Maximize joint surplus; accept if E[S] > 0 |
| Opportunistic | `opportunistic.py` | Maximize own payoff; accept if E[pi_self] > threshold |
| Deceptive | `deceptive.py` | Manipulate v perception; accept when true v = -1 |
| Adversarial | `adversarial.py` | Maximize system harm, even at personal cost |
| Adaptive Adversary | `adaptive_adversary.py` | Learn and adapt attack strategies |
| LLM Agent | `llm_agent.py` | LLM-backed decision-making (Anthropic/OpenAI/Ollama) |
| Memory Agent | `memory_agent.py` | Memory-augmented reasoning |
| Moltbook Agent | `moltbook_agent.py` | Social media interaction specialist |
| Rain/River Agent | `rain_river.py` | Gradient-based policy optimization |
| Scholar Agent | `scholar_agent.py` | Research/knowledge synthesis |
| Wiki Editor | `wiki_editor.py` | Knowledge base editing |
| Roles | `roles/` | Planner, Worker, Verifier, Poster, Moderator |

---

## Success Criteria

### Phase 1: Core Simulation
- [x] 5 agents interact over 10+ epochs
- [x] Toxicity and conditional loss metrics computed per epoch
- [x] Full event log enables deterministic replay
- [x] Observable failure modes: miscoordination, conflict, collusion

### Phase 2: Economics & Governance
- [x] 3+ Moltbook-like motifs reproducible
- [x] 2+ governance levers measurably reduce toxicity/collusion
- [x] Parameter sweep across 12 governance configurations
- [x] Dashboard shows real-time metrics

### Phase 3: Advanced Features
- [x] Red-team framework produces measurable attack success rates
- [x] Boundary enforcement detects and limits information leakage
- [x] Collusion detection identifies coordinated agent clusters
- [x] LLM agents integrate with Anthropic/OpenAI/Ollama backends

### Phase 4: Virtual Agent Economies
- [x] Dworkin auctions converge to envy-free allocations
- [x] Mission economies detect free-riders and distribute rewards fairly
- [x] HFN engine detects flash crashes and triggers market halts
- [x] Permeability model tracks spillover harm
- [x] Identity infrastructure supports credentials and Sybil detection

### Phase 5: Bridge Integrations
- [x] 6 bridges connect to external agent frameworks
- [x] Each bridge maps external events to SWARM observables
- [x] Policy enforcement at bridge boundaries
- [x] Integration tests for all bridges

### Phase 6: Research Pipeline
- [x] Multi-agent research workflow with 7 sub-agent types
- [x] Quality gates and pre-registration enforce rigor
- [x] Reflexivity analysis (shadow simulations, publish-then-attack)
- [x] Track A pipeline generates papers from experimental results
- [x] Platform integration (AgentRxiv, Agentxiv, Clawxiv)

### Phase 7: Developer Tooling
- [x] 20 slash commands for development workflow
- [x] 6 specialist agents for domain expertise
- [x] Pre-commit hook: secrets scan, staged-only lint, mypy, pytest

### Infrastructure
- [x] CI pipeline: lint, type-check, tests across Python 3.10-3.12
- [x] >= 70% test coverage enforced
- [x] 2202 tests pass across 70 test files
- [x] CLI supports scenario execution with seed/epoch overrides and JSON/CSV export

All MVP criteria verified by `tests/test_success_criteria.py`.
Integration tests in `tests/test_integration.py`.

---

## Verification

### Unit Tests
```bash
python -m pytest tests/ -v
```

### Single Scenario
```bash
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10
```

### Governance Sweep
```bash
python -m swarm run scenarios/baseline.yaml --epochs 50
python -m swarm run scenarios/strict_governance.yaml --epochs 50
```

### Lint & Type Check
```bash
ruff check swarm/ tests/
python -m mypy swarm/
```

---

## Dependencies

```toml
[project.dependencies]
numpy = ">=1.24"
pydantic = ">=2.0"
pandas = ">=2.0"

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "mypy", "ruff", "hypothesis", "pytest-asyncio"]
runtime = ["pyyaml>=6.0", "requests", "tenacity"]
llm = ["anthropic>=0.40.0", "openai>=1.50.0"]
dashboard = ["streamlit", "plotly"]
analysis = ["matplotlib", "seaborn"]
bridges = ["swarm-gastown"]
cli = ["rich"]
concordia = ["concordia"]
api = ["fastapi", "uvicorn", "python-multipart"]
docs = ["mkdocs-material", "mkdocstrings"]
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
