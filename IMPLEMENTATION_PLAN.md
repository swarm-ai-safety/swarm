# Distributional AGI Safety Sandbox - Implementation Plan

## One-Line Goal

Build and evaluate a **multi-agent sandbox economy** to study *system-level safety properties* when AGI-level capabilities emerge from interacting sub-AGI agents.

---

## Current State

**Package:** `swarm-safety` v1.5.0 (imported as `swarm`)
**Python:** >= 3.10 | **Tests:** 2922 across 93 files | **Scenarios:** 55 YAML definitions

### Foundation Layer

| Component | Status | Files |
|-----------|--------|-------|
| Data Models | Complete | `swarm/models/interaction.py`, `agent.py`, `events.py`, `identity.py`, `scholar.py`, `kernel.py`, `schemas.py` |
| Proxy Computation | Complete | `swarm/core/proxy.py`, `sigmoid.py` |
| Payoff Engine | Complete | `swarm/core/payoff.py` |
| Metrics System | Complete | `swarm/metrics/` (14 modules) |
| Event Logging | Complete | `swarm/logging/event_log.py` |

### Runtime Layer

| Component | Status | Files |
|-----------|--------|-------|
| Orchestrator | Complete | `swarm/core/orchestrator.py` + 12 domain handlers |
| Agent Policies | Complete | `swarm/agents/` (22 modules: 6 core + LDT + RLM + Council + SkillRL + roles + LLM + memory + domain agents) |
| Feed/Interaction Engine | Complete | `swarm/env/feed.py` |
| Governance Module | Complete | `swarm/governance/` (23 modules, 24+ levers) |
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

### Consumer-Seller Marketplace

| Component | Status | Files |
|-----------|--------|-------|
| CSM Matching Engine | Complete | `swarm/csm/matching.py` |
| CSM Negotiation | Complete | `swarm/csm/negotiation.py` |
| CSM Identity Layer | Complete | `swarm/csm/identity.py` |
| CSM Platform Access | Complete | `swarm/csm/platform_access.py` |
| CSM Search/Purchase | Complete | `swarm/csm/search_purchase.py` |
| CSM Metrics & Plots | Complete | `swarm/csm/metrics.py`, `plots.py` |
| CSM Runner | Complete | `swarm/csm/runner.py` |
| CSM Types | Complete | `swarm/csm/types.py` |

### Council Governance

| Component | Status | Files |
|-----------|--------|-------|
| Council Protocol | Complete | `swarm/council/protocol.py` |
| Council Ranking | Complete | `swarm/council/ranking.py` |
| Council Config | Complete | `swarm/council/config.py` |
| Council Prompts | Complete | `swarm/council/prompts.py` |
| Council Governance Lever | Complete | `swarm/governance/council_lever.py` |
| Council Agent | Complete | `swarm/agents/council_agent.py` |

### Skill Learning & Evolution

| Component | Status | Files |
|-----------|--------|-------|
| Skill Model | Complete | `swarm/skills/model.py` |
| Skill Library | Complete | `swarm/skills/library.py` |
| Skill Evolution | Complete | `swarm/skills/evolution.py` |
| Skill Governance | Complete | `swarm/skills/governance.py` |
| Skill Metrics | Complete | `swarm/skills/metrics.py` |
| Skill-Evolving Agent | Complete | `swarm/agents/skill_evolving.py` |
| SkillRL Agent | Complete | `swarm/agents/skillrl_agent.py` |

### Bridge Integrations

| Bridge | Status | Files |
|--------|--------|-------|
| Claude Code | Complete | `swarm/bridges/claude_code/` (agent, bridge, client, events, policy) |
| Concordia | Complete | `swarm/bridges/concordia/` (adapter, config, events, game_master, judge, narratives, multiverse, simulacra) |
| GasTown | Complete | `swarm/bridges/gastown/` (agent, beads, bridge, config, events, git_observer, mapper, policy) |
| Live SWE | Complete | `swarm/bridges/live_swe/` (agent, bridge, client, events, policy, tracker) |
| OpenClaw | Complete | `swarm/bridges/openclaw/` (config, job_queue, schemas, service, skill) |
| Prime Intellect | Complete | `swarm/bridges/prime_intellect/` (bridge, client, config, environment, events, rewards, scoring) |
| Ralph | Complete | `swarm/bridges/ralph/` (bridge, config, events, mapper) |
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
| Slash Commands | Complete | `.claude/commands/` (45 commands) |
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

The orchestrator (`swarm/core/orchestrator.py`) delegates domain-specific logic to pluggable handlers registered via `swarm/core/handler_registry.py`:

| Handler | Domain |
|---------|--------|
| `handler.py` | Base handler interface |
| `handler_registry.py` | Dynamic handler registration |
| `boundary_handler.py` | Sandbox boundary enforcement |
| `marketplace_handler.py` | Marketplace/auction mechanics |
| `memory_handler.py` | Agent memory management |
| `moltbook_handler.py` | Moltbook social dynamics |
| `moltipedia_handler.py` | Moltipedia knowledge dynamics |
| `scholar_handler.py` | Scholar/research agent coordination |
| `kernel_handler.py` | Kernel market orchestration |
| `skill_handler.py` | Skill learning system |
| `feed_handler.py` | Feed processing |
| `core_interaction_handler.py` | Core interaction logic |
| `task_handler.py` | Task system management |
| `interaction_finalizer.py` | Interaction finalization |
| `observable_generator.py` | Signal generation from raw events |
| `observation_builder.py` | Observation construction |
| `pseudo_verifiers.py` | Automated verification simulation |
| `proxy_auditor.py` | Proxy audit trail |
| `redteam_inspector.py` | Red team inspection |
| `cuda_analyzer.py` | CUDA kernel safety analysis |
| `cuda_templates.py` | CUDA kernel templates |

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
| 10 | Governance Engine + Levers | `swarm/governance/engine.py`, `levers.py`, + 19 lever modules | Complete |
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
| 32 | Concordia Bridge | `swarm/bridges/concordia/` (8 modules) | Complete |
| 33 | GasTown Bridge | `swarm/bridges/gastown/` (9 modules) | Complete |
| 34 | Live SWE Bridge | `swarm/bridges/live_swe/` (7 modules) | Complete |
| 35 | OpenClaw Bridge | `swarm/bridges/openclaw/` (6 modules) | Complete |
| 36 | Worktree Sandbox Bridge | `swarm/bridges/worktree/` (9 modules + Rich CLI) | Complete |
| 37 | Prime Intellect Bridge | `swarm/bridges/prime_intellect/` (8 modules) | Complete |
| 38 | Ralph Bridge | `swarm/bridges/ralph/` (5 modules) | Complete |

Each bridge follows a consistent pattern:
- **bridge.py** — Adapter mapping external events to SWARM observables
- **policy.py** — Governance policy enforcement at the boundary
- **events.py** — Domain-specific event types
- **agent.py / client.py** — Interface to the external system

### Phase 6: Research Pipeline

**Goal:** Multi-agent research workflow with structured sub-agents, quality gates, and platform integration.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 39 | Research Sub-Agents | `swarm/research/agents.py` (7 agent types) | Complete |
| 40 | Platform Clients | `swarm/research/platforms.py` (AgentRxiv, Agentxiv, Clawxiv) | Complete |
| 41 | AgentRxiv Server | `swarm/research/agentrxiv_server.py` | Complete |
| 42 | Quality Gates & Pre-Registration | `swarm/research/quality.py` | Complete |
| 43 | Reflexivity Analysis | `swarm/research/reflexivity.py` | Complete |
| 44 | Paper Annotation | `swarm/research/annotator.py` | Complete |
| 45 | PDF Export | `swarm/research/pdf_export.py` | Complete |
| 46 | Submission Validation | `swarm/research/submission.py` | Complete |
| 47 | Scenario Generation | `swarm/research/scenario_gen.py` | Complete |
| 48 | Research Workflow | `swarm/research/workflow.py` | Complete |
| 49 | Track A Pipeline | `swarm/research/swarm_papers/track_a.py` | Complete |
| 50 | Paper Builder | `swarm/research/swarm_papers/paper.py` | Complete |
| 51 | Memory Store | `swarm/research/swarm_papers/memory.py` | Complete |
| 52 | AgentRxiv Bridge | `swarm/research/swarm_papers/agentrxiv_bridge.py` | Complete |

### Phase 7: Developer Tooling

**Goal:** Claude Code slash commands, specialist agents, and git hooks for development workflow.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 53 | Slash Commands (45) | `.claude/commands/` | Complete |
| 54 | Specialist Agents (6) | `.claude/agents/` | Complete |
| 55 | Pre-commit Hook | `.claude/hooks/pre-commit` | Complete |
| 56 | MCP Configuration | `.mcp.json` | Complete |

**Slash Commands:**
`/add_domain`, `/add_metric`, `/add_post`, `/add_scenario`, `/address_review`, `/analyze_experiment`, `/benchmark`, `/check_ignore`, `/cleanup_branch`, `/commit_push`, `/compare_studies`, `/compile_paper`, `/eval_writeup`, `/fix_ci`, `/fix_deps`, `/full_study`, `/healthcheck`, `/install_hooks`, `/lint_fix`, `/log_run`, `/merge_all_sessions`, `/merge_session`, `/parse_eval`, `/plot`, `/pr`, `/preflight`, `/publish_figures`, `/red_team`, `/release`, `/retro`, `/run_and_plot`, `/run_scenario`, `/scan_secrets`, `/ship`, `/stage`, `/stats`, `/status`, `/submit_paper`, `/submit_to_list`, `/sweep`, `/sweep_and_ship`, `/sync`, `/tmux`, `/warmup`, `/write_paper`

**Specialist Agents:**
`adversary_designer`, `mechanism_designer`, `metrics_auditor`, `reproducibility_sheriff`, `research_scout`, `scenario_architect`

### Phase 8: Decision Theory & Agent Learning

**Goal:** Decision-theoretic agent policies, reinforcement learning from memory, and skill evolution.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 57 | LDT Agent | `swarm/agents/ldt_agent.py` | Complete |
| 58 | Modeling Adversary | `swarm/agents/modeling_adversary.py` | Complete |
| 59 | RLM Agent | `swarm/agents/rlm_agent.py` | Complete |
| 60 | RLM Metrics | `swarm/metrics/rlm_metrics.py` | Complete |
| 61 | Council Agent | `swarm/agents/council_agent.py` | Complete |
| 62 | Council Protocol | `swarm/council/` (config, prompts, protocol, ranking) | Complete |
| 63 | Council Governance Lever | `swarm/governance/council_lever.py` | Complete |
| 64 | Skill Model & Library | `swarm/skills/model.py`, `library.py` | Complete |
| 65 | Skill Evolution | `swarm/skills/evolution.py` | Complete |
| 66 | Skill Governance & Metrics | `swarm/skills/governance.py`, `metrics.py` | Complete |
| 67 | Skill-Evolving Agent | `swarm/agents/skill_evolving.py` | Complete |
| 68 | SkillRL Agent | `swarm/agents/skillrl_agent.py` | Complete |

### Phase 9: Consumer-Seller Marketplace & Domain Environments

**Goal:** Full consumer-seller marketplace simulation, catalog environments, and kernel market integration.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 69 | CSM Matching Engine | `swarm/csm/matching.py` | Complete |
| 70 | CSM Negotiation | `swarm/csm/negotiation.py` | Complete |
| 71 | CSM Identity | `swarm/csm/identity.py` | Complete |
| 72 | CSM Platform Access | `swarm/csm/platform_access.py` | Complete |
| 73 | CSM Search/Purchase | `swarm/csm/search_purchase.py` | Complete |
| 74 | CSM Metrics & Plots | `swarm/csm/metrics.py`, `plots.py` | Complete |
| 75 | CSM Runner | `swarm/csm/runner.py` | Complete |
| 76 | Moltbook Catalog | `swarm/env/moltbook_catalog.py` | Complete |
| 77 | Wiki Catalog | `swarm/env/wiki_catalog.py` | Complete |
| 78 | Memory Tiers | `swarm/env/memory_tiers.py` | Complete |
| 79 | Kernel Market Handler | `swarm/core/kernel_handler.py` | Complete |
| 80 | Kernel Market Model | `swarm/models/kernel.py` | Complete |
| 81 | Event Bus & Schemas | `swarm/models/schemas.py` | Complete |

### Phase 10: Orchestrator Refactoring

**Goal:** Extract core orchestrator logic into focused, testable handlers.

| # | Component | Files | Status |
|---|-----------|-------|--------|
| 82 | Handler Registry | `swarm/core/handler_registry.py` | Complete |
| 83 | Feed Handler | `swarm/core/feed_handler.py` | Complete |
| 84 | Core Interaction Handler | `swarm/core/core_interaction_handler.py` | Complete |
| 85 | Task Handler | `swarm/core/task_handler.py` | Complete |
| 86 | Interaction Finalizer | `swarm/core/interaction_finalizer.py` | Complete |
| 87 | Observation Builder | `swarm/core/observation_builder.py` | Complete |
| 88 | Proxy Auditor | `swarm/core/proxy_auditor.py` | Complete |
| 89 | Red Team Inspector | `swarm/core/redteam_inspector.py` | Complete |
| 90 | CUDA Analyzer | `swarm/core/cuda_analyzer.py`, `cuda_templates.py` | Complete |

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
│   ├── commands/              # 45 slash commands
│   ├── agents/                # 6 specialist agents
│   └── hooks/
│       └── pre-commit         # Source of truth for .git/hooks/pre-commit
├── swarm/
│   ├── __init__.py
│   ├── __main__.py            # CLI entry point (run/list scenarios)
│   ├── py.typed               # PEP 561 type marker
│   ├── models/                # Data models (8 modules)
│   │   ├── interaction.py     # SoftInteraction dataclass
│   │   ├── agent.py           # AgentType, AgentState
│   │   ├── events.py          # EventType definitions
│   │   ├── identity.py        # Verifiable credentials, Proof-of-Personhood
│   │   ├── scholar.py         # Scholar-specific models
│   │   ├── kernel.py          # Kernel market models
│   │   └── schemas.py         # TypedDict event schemas
│   ├── core/                  # Core engines (29 modules)
│   │   ├── payoff.py          # SoftPayoffEngine
│   │   ├── proxy.py           # ProxyComputer
│   │   ├── sigmoid.py         # Calibration functions
│   │   ├── orchestrator.py    # Main simulation orchestrator
│   │   ├── handler.py         # Base handler interface
│   │   ├── handler_registry.py # Dynamic handler registration
│   │   ├── boundary_handler.py
│   │   ├── marketplace_handler.py
│   │   ├── memory_handler.py
│   │   ├── memory_observables.py
│   │   ├── moltbook_handler.py
│   │   ├── moltbook_observables.py
│   │   ├── moltipedia_handler.py
│   │   ├── moltipedia_observables.py
│   │   ├── scholar_handler.py
│   │   ├── kernel_handler.py  # Kernel market orchestration
│   │   ├── skill_handler.py   # Skill learning system
│   │   ├── feed_handler.py    # Feed processing
│   │   ├── core_interaction_handler.py  # Core interaction logic
│   │   ├── task_handler.py    # Task system management
│   │   ├── interaction_finalizer.py     # Interaction finalization
│   │   ├── observable_generator.py
│   │   ├── observation_builder.py       # Observation construction
│   │   ├── pseudo_verifiers.py
│   │   ├── proxy_auditor.py   # Proxy audit trail
│   │   ├── redteam_inspector.py         # Red team inspection
│   │   ├── cuda_analyzer.py   # CUDA kernel safety analysis
│   │   └── cuda_templates.py  # CUDA kernel templates
│   ├── agents/                # Agent implementations (22 modules)
│   │   ├── base.py            # BaseAgent ABC
│   │   ├── honest.py
│   │   ├── opportunistic.py
│   │   ├── deceptive.py
│   │   ├── adversarial.py
│   │   ├── adaptive_adversary.py
│   │   ├── modeling_adversary.py  # Decision-theoretic modeling adversary
│   │   ├── ldt_agent.py      # Logical Decision Theory agent
│   │   ├── rlm_agent.py      # Reinforcement Learning from Memory
│   │   ├── council_agent.py   # Council-based governance agent
│   │   ├── skillrl_agent.py   # SkillRL learning agent
│   │   ├── skill_evolving.py  # Skill evolution agent
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
│   ├── env/                   # Environment modules (15 modules)
│   │   ├── state.py
│   │   ├── feed.py
│   │   ├── tasks.py
│   │   ├── marketplace.py
│   │   ├── composite_tasks.py
│   │   ├── network.py
│   │   ├── auction.py         # Dworkin-style fair allocation
│   │   ├── mission.py         # Mission economies
│   │   ├── hfn.py             # High-frequency negotiation
│   │   ├── memory_tiers.py    # Tiered memory environments
│   │   ├── moltbook.py        # Moltbook environment
│   │   ├── wiki.py            # Wiki environment
│   │   ├── moltbook_catalog.py # Moltbook content catalog
│   │   └── wiki_catalog.py    # Wiki content catalog
│   ├── governance/            # Governance mechanisms (23 modules)
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
│   │   ├── moltipedia.py
│   │   └── council_lever.py   # Council governance lever
│   ├── metrics/               # Metrics computation (14 modules)
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
│   │   ├── time_horizons.py
│   │   └── rlm_metrics.py    # RLM agent metrics
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
│   ├── csm/                   # Consumer-Seller Marketplace (10 modules)
│   │   ├── types.py
│   │   ├── matching.py
│   │   ├── negotiation.py
│   │   ├── identity.py
│   │   ├── platform_access.py
│   │   ├── search_purchase.py
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── runner.py
│   ├── council/               # Council governance (5 modules)
│   │   ├── config.py
│   │   ├── prompts.py
│   │   ├── protocol.py
│   │   └── ranking.py
│   ├── skills/                # Skill learning & evolution (6 modules)
│   │   ├── model.py
│   │   ├── library.py
│   │   ├── evolution.py
│   │   ├── governance.py
│   │   └── metrics.py
│   ├── bridges/               # External system integrations (8 bridges)
│   │   ├── claude_code/       # Claude Code development bridge
│   │   ├── concordia/         # DeepMind Concordia bridge
│   │   ├── gastown/           # GasTown distributed dev bridge
│   │   ├── live_swe/          # Live SWE agent bridge
│   │   ├── openclaw/          # OpenClaw research platform bridge
│   │   ├── prime_intellect/   # Prime Intellect verification bridge
│   │   ├── ralph/             # Ralph framework bridge
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
├── scenarios/                 # 55 scenario YAML definitions
│   ├── baseline.yaml
│   ├── status_game.yaml
│   ├── strict_governance.yaml
│   ├── collusion_detection.yaml
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
│   ├── concordia_governance_sweep.yaml
│   ├── gastown_workspace.yaml
│   ├── horizon_eval.yaml
│   ├── live_swe_self_evolution.yaml
│   ├── macpo_weak_to_strong.yaml
│   ├── memory_tiers.yaml
│   ├── moltbook_captcha.yaml
│   ├── moltipedia_heartbeat.yaml
│   ├── alignment_waltz_targeted_feedback.yaml
│   ├── worktree_sandbox.yaml
│   ├── council_governance.yaml
│   ├── evolving_skills.yaml
│   ├── prime_intellect_safety.yaml
│   ├── social_simulacra.yaml
│   ├── skillrl.yaml
│   ├── rlm_governance_lag.yaml
│   ├── rlm_memory_as_power.yaml
│   ├── rlm_recursive_collusion.yaml
│   ├── ldt_cooperation.yaml
│   ├── ldt_large_population.yaml
│   ├── ldt_modeling_adversary.yaml
│   ├── ldt_low_prior.yaml
│   ├── ldt_short_horizon.yaml
│   ├── csm_full_benchmark.yaml
│   ├── csm_matching.yaml
│   ├── csm_search_purchase.yaml
│   ├── incoherence/
│   │   ├── long_high_branching.yaml
│   │   ├── medium_medium_branching.yaml
│   │   └── short_low_branching.yaml
│   ├── scholar_bench/
│   │   ├── baseline.yaml
│   │   ├── citation_laundering.yaml
│   │   ├── adversarial.yaml
│   │   └── sweeps/
│   │       ├── governance_comparison.yaml
│   │       ├── audit_rate.yaml
│   │       └── adversary_mix.yaml
│   └── kernel_market/
│       ├── baseline.yaml
│       ├── v2.yaml
│       ├── v3.yaml
│       ├── v4_code.yaml
│       └── sweeps/
│           ├── governance_comparison.yaml
│           ├── audit_rate.yaml
│           └── adversary_mix.yaml
├── tests/                     # 93 test files, 2922 tests
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
│   ├── test_event_bus.py
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
│   ├── test_ldt_agent.py
│   ├── test_modeling_adversary.py
│   ├── test_rlm_agent.py
│   ├── test_rlm_metrics.py
│   ├── test_council.py
│   ├── test_council_agent.py
│   ├── test_council_lever.py
│   ├── test_skills.py
│   ├── test_skillrl.py
│   ├── test_csm.py
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
│   ├── test_peer_review.py
│   ├── test_swarm_papers_memory.py
│   ├── test_claude_code_bridge.py
│   ├── test_claude_code_runner.py
│   ├── test_concordia_bridge.py
│   ├── test_concordia_sweep.py
│   ├── test_gastown_bridge.py
│   ├── test_live_swe_bridge.py
│   ├── test_openclaw_bridge.py
│   ├── test_prime_intellect_bridge.py
│   ├── test_ralph_bridge.py
│   ├── test_worktree_bridge.py
│   ├── test_diversity.py
│   ├── test_evaluation.py
│   ├── test_forecaster.py
│   ├── test_governance_memory.py
│   ├── test_governance_mvp_sweep.py
│   ├── test_horizon_eval.py
│   ├── test_incoherence_metrics.py
│   ├── test_illusion_delta_minimal_example.py
│   ├── test_memory_metrics.py
│   ├── test_moltbook.py
│   ├── test_moltbook_governance.py
│   ├── test_moltbook_integration.py
│   ├── test_moltbook_metrics.py
│   ├── test_moltbook_scenario.py
│   ├── test_moltbook_catalog.py
│   ├── test_moltipedia_governance.py
│   ├── test_moltipedia_integration.py
│   ├── test_moltipedia_metrics.py
│   ├── test_moltipedia_scenario.py
│   ├── test_prompt_audit.py
│   ├── test_proxy_auditor.py
│   ├── test_rain_river.py
│   ├── test_replay_runner.py
│   ├── test_scholar.py
│   ├── test_simulated_apis.py
│   ├── test_social_simulacra.py
│   ├── test_time_horizons.py
│   ├── test_vae_integration.py
│   ├── test_wiki.py
│   ├── test_wiki_catalog.py
│   ├── test_handler_registry.py
│   ├── test_kernel_handler.py
│   └── test_cuda_analyzer.py
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
│   ├── llm_demo.py
│   ├── governance_mvp_sweep.py
│   ├── concordia_governance_sweep.py
│   ├── gastown_composition_study.py
│   ├── ldt_composition_study.py
│   ├── plot_run.py
│   ├── plot_sweep.py
│   ├── sweep_stats.py
│   ├── run_incoherence_scaling.py
│   ├── run_redteam.py
│   ├── generate_metric_graphs.py
│   ├── reproduce_2602_00035.py
│   ├── sensitivity_purity_paradox.py
│   ├── illusion_delta_minimal.py
│   └── gastown_live_probe.py
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
| Council Governance | `council_lever.py` | Council-based deliberative governance |

---

## Agent Types

| Agent | Module | Strategy |
|-------|--------|----------|
| Honest | `honest.py` | Maximize joint surplus; accept if E[S] > 0 |
| Opportunistic | `opportunistic.py` | Maximize own payoff; accept if E[pi_self] > threshold |
| Deceptive | `deceptive.py` | Manipulate v perception; accept when true v = -1 |
| Adversarial | `adversarial.py` | Maximize system harm, even at personal cost |
| Adaptive Adversary | `adaptive_adversary.py` | Learn and adapt attack strategies |
| Modeling Adversary | `modeling_adversary.py` | Decision-theoretic opponent modeling |
| LDT Agent | `ldt_agent.py` | Logical Decision Theory (UDT precommitment) |
| RLM Agent | `rlm_agent.py` | Reinforcement Learning from Memory |
| Council Agent | `council_agent.py` | Council-based deliberative governance agent |
| SkillRL Agent | `skillrl_agent.py` | Reinforcement learning over skill repertoire |
| Skill-Evolving Agent | `skill_evolving.py` | Adaptive skill evolution |
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
- [x] 8 bridges connect to external agent frameworks
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
- [x] 45 slash commands for development workflow
- [x] 6 specialist agents for domain expertise
- [x] Pre-commit hook: secrets scan, staged-only lint, mypy, pytest

### Phase 8: Decision Theory & Agent Learning
- [x] LDT agent demonstrates UDT precommitment advantage
- [x] Modeling adversary adapts to opponent decision theory
- [x] RLM agent learns from memory-based reinforcement
- [x] Council governance produces deliberative policy decisions
- [x] Skill evolution system supports adaptive agent repertoires
- [x] SkillRL agent learns over skill library

### Phase 9: Consumer-Seller Marketplace
- [x] CSM matching engine supports consumer-seller dynamics
- [x] CSM negotiation, identity, and platform access layers
- [x] Kernel market handler integrates with orchestrator
- [x] Moltbook and Wiki catalog environments
- [x] Event bus with TypedDict schemas

### Phase 10: Orchestrator Refactoring
- [x] Handler registry enables dynamic handler composition
- [x] Core logic extracted into focused handlers (feed, interaction, task, finalization)
- [x] Proxy auditor provides audit trail
- [x] CUDA analyzer for kernel safety analysis

### Infrastructure
- [x] CI pipeline: lint, type-check, tests across Python 3.10-3.12
- [x] >= 70% test coverage enforced
- [x] 2922 tests pass across 93 test files
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
