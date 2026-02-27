# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Changed
- **Orchestrator pipeline/middleware refactoring** — extracted 3 new modules from the 2023-line orchestrator god object: `middleware.py` (7 lifecycle stages via `MiddlewarePipeline`), `handler_factory.py` (handler construction from config), `agent_scheduler.py` (turn order and eligibility); orchestrator is now a thin coordination loop delegating cross-cutting concerns to the middleware pipeline; public API preserved

### Added
- **OpenRouter LLM backend for Escalation Sandbox** (`swarm/domains/escalation_sandbox/agents.py`) — `OpenRouterBackend` enabling LLM-vs-LLM crisis simulations via OpenRouter; 5 LLM scenario YAMLs pairing Claude Sonnet 4, GPT-4.1-mini, Gemini 2.0 Flash, Llama 3.3 70B, and Mistral Small 3.1 across baseline, Cuban Missile, deception, governance, and fog stress configurations; sweep scripts for scripted and LLM comparison plots
- **Agent-level → population-level safety bridge** — three-piece system bridging agent-level evals (HAICosystem, OpenAgentSafety) into SWARM population-level simulation: `EvalTraceObservableGenerator` (converts multi-turn eval traces to ProxyObservables), `BehavioralProfiler` (infers archetype mixture weights via MLE), `SafetyCompositionAnalyzer` (structured sweeps producing safety certificates with regime classification and composition boundaries); 99 new tests
- **Agents of Chaos case study scenarios** — 4 scenario YAMLs modelling empirically observed failure modes from the Agents of Chaos red-teaming study (Shapira et al. 2026): `casestudy_libel_cascade` (CS11 network adverse selection), `casestudy_proxy_corruption` (CS10 signal corruption/detection/recovery), `casestudy_disproportionate_response` (CS1 payoff misspecification with staking), `casestudy_dual_use_coordination` (CS9+CS11 prosocial vs antisocial coordination)
- **Tierra governance hardening** — diversity-preserving reaper mode (`reaper_mode: "diversity_preserving"`) that protects at least 1 representative per species cluster during population culling, efficiency weight cap (`max_efficiency_weight`) to prevent runaway resource concentration, and `species_clusters()` helper in `tierra_metrics.py`
- **Governed Tierra scenario** (`scenarios/tierra_governed.yaml`) — Tierra variant with circuit breaker, collusion detection, 5% transaction tax, reputation decay (0.95), diversity-preserving reaper, and efficiency cap (3x mean)
- **Blog post**: Tierra governance vs evolution comparative study — 5-seed comparison showing +6.5% genome diversity, -2.2% Gini, at -12% population cost
- **Behavioral agent types** (`swarm/agents/behavioral.py`) — `CautiousAgent` (risk-averse, high acceptance threshold), `CollaborativeAgent` (coalition-building, EMA trust tracking), and `AdaptiveAgent` (rolling payoff window, threshold self-adaptation with exploration) with corresponding `AgentType` enum values and 20 unit tests (#66)
- **LangChain bridge** (`swarm/bridges/langchain/`) — wraps any LangChain Runnable (chain, AgentExecutor) as a SWARM interaction source; maps chain success/failure, intermediate steps, and output length to soft labels via `ProxyComputer`; lazy-imports langchain so module is importable without it installed (#69)
- **AutoGPT bridge** (`swarm/bridges/autogpt/`) — protocol-level bridge mapping AutoGPT thought/command/result cycles to `SoftInteraction` objects; blocks configurable dangerous commands (delete_file, shutdown, etc.) and uses self-criticism as rework signal; no AutoGPT installation required (#69)
- **CrewAI bridge** (`swarm/bridges/crewai/`) — wraps CrewAI crew execution as a SWARM interaction source generating one `SoftInteraction` per task (distinct from `crewai_adapter` agent); supports full crew mode and protocol mode without crewai installed (#69)
- **Mesa ABM bridge** (`swarm/bridges/mesa/`) — wraps Mesa `Model.step()` to extract `SoftInteraction` objects from agent state after each step; works with existing Mesa models via configurable attribute names; supports protocol mode via dict-based agent state records (#69)
- **RAG LEANN backend** (`swarm/bridges/rag/backend.py`) — `VectorBackend` protocol with ChromaDB and LEANN implementations; LEANN provides ~97% storage savings via graph-based selective recomputation with JSON sidecar metadata and post-retrieval filtering; selected via `RAGConfig.vector_backend`
- **RAG bridge** (`swarm/bridges/rag/`) — semantic search over run history via ChromaDB vector store, with configurable embeddings (OpenAI/Ollama), LLM synthesis (Anthropic/OpenAI), CLI (`python -m swarm.bridges.rag`), and `rag` optional dependency group
- **Adaptive governance controller** (`swarm/governance/adaptive.py`, `swarm/governance/adaptive_controller.py`) — three-phase loop with evidence accumulation, 3-pass contemplation (signal/trend/propose), and 3-gate crystallization (time/alignment/human review) for automatic threshold tuning of governance levers
- **Adaptive governance scenario** (`scenarios/adaptive_governance.yaml`) — 30-epoch mixed-agent scenario with circuit breaker, audit, and collusion detection levers enabled for adaptive tuning
- **Social dilemma norms study** (`examples/social_dilemma_norms_study.py`) — 3 dilemmas x 5 governance configs sweep measuring cooperation emergence, with dilemma narrative generators (`swarm/bridges/concordia/dilemma_narratives.py`) and scenario YAMLs for commons and prisoner's dilemma
- **ThresholdDancer adversary agent** (`swarm/agents/threshold_dancer.py`) — per-counterparty state machine (COOPERATIVE/EXPLOIT/RECOVER) that exploits CautiousReciprocator's blacklist floor without triggering it
- **Threshold dancer test suite** (`tests/test_threshold_dancer.py`) — 21 unit tests covering phase transitions, blacklist safety property, act method, and outcome tracking
- **Threshold dancer scenario** (`scenarios/threshold_dancer_vs_cautious.yaml`) — 30-epoch stress test with 3 cautious + 2 honest + 3 dancers
- **Two new red-team scenarios** in `examples/redteam_cautious.py` — "Threshold dancers only" and "Mixed adversaries + threshold dancers"
- **Blog post**: Threshold dancer results — the adversary that avoids blacklisting but can't profit
- **Tierra artificial life scenario** (`swarm/agents/tierra_agent.py`, `swarm/core/tierra_handler.py`, `swarm/metrics/tierra_metrics.py`, `scenarios/tierra.yaml`) — agents with heritable mutable genomes self-replicate when resource-rich, competing for finite shared resources; complex ecological dynamics (parasitism, mutualism) emerge from replication + mutation + selection (Tom Ray, 1991)
- **Evolutionary game handler** (`swarm/core/evo_game_handler.py`) — integrates gamescape's PayoffMatrix into the orchestrator pipeline, mapping 2x2 game payoffs to ProxyObservables with cooperate/defect/tit-for-tat/grudger strategies and epoch-level population dynamics rendering
- **Evo game scenario** (`scenarios/evo_game_prisoners.yaml`) — iterated Prisoner's Dilemma with 10 agents (cooperators, defectors, TFT)
- **Evo game study runner** (`examples/evo_game_study.py`) — standalone runner comparing empirical population trajectory with replicator dynamics prediction

## [1.7.0] - 2026-02-21

### Added
- **Contract screening system** for separating equilibrium analysis with lock-in semantics, welfare metric, multi-seed sweep (10 seeds), collusion detection, and plot script (#234)
- **LangGraph governed handoff study** with 4-agent Claude swarm, 32-config sweep (seed 42), and sweep overview plot
- **Hodoscope trajectory analysis bridge** for agent trace inspection
- **SQLite persistence** for simulations, governance state, and scenarios with lazy-init singletons
- **SoftMetrics wired into Web API** `/api/v1/metrics` endpoint
- **Sybil detection** enabled for contract screening governance
- **E2E integration tests** for Web API simulation lifecycle
- **llama.cpp local inference** provider with server setup script, health checks, seed validation, and SSRF/path-traversal hardening (#232)
- **Interactive isometric visualization game** (`viz/`): Next.js browser-based SWARM simulation with client-side engine, Gemini Imagen 4 sprite assets, compare mode, parameter sweep, leaderboard, governance intervention controls, preset scenarios, narrative annotations, and data export (#182, #212)
- **Memori semantic memory middleware** for LLM agents with persistent fact recall, SQLite-backed storage, and OpenRouter scenario variant (#217)
- **Loop detector governance lever** with graduated enforcement — tracks interaction patterns, quality scores, tool misuse, and rework to detect repetitive agent loops (#198)
- **Agent API Phase 1–3**: scoped permissions, trace IDs, structured errors, PATCH endpoints, filtering, validation, agent approval workflow with approve/reject endpoints and `auto_approve` config
- **SciAgentBench harness** with topology matrix support (#200)
- **Evaluation metrics suite** for success rate, efficiency, and detection (#201)
- **SciForge-style trace-to-task synthesis** with replay verification (#203)
- **Parameter validation and clamping diagnostics** for proxy computation (#176)
- **MetricsAggregator** wired into CLI and example export for rich visualization data, including 3 demo datasets (#212)
- **Reproducibility documentation** with one-command run workflow and artifact paths (#204)
- **Integration tests** for runtime environment lifecycle and tool invocation with leak detection (#197)
- **EPIC tracking infrastructure** for bridge integrations (#194)
- **Collaborative chemistry under budget and audits** scenario (#202)
- **CI quality gate**, `/review_external_pr` command, and blog index hook
- **Execution state** populated during simulation runs
- **Blog posts**: Qwen3-30B SWARM Economy v0.2 training results, contract screening separating equilibrium, multi-seed results, red-team findings
- **Slash commands**: `/build_game`, `/obsidian`, `/sync_artifacts`, `/review_external_pr`, `/security-review`, `/audit_docs`, `/check_nav`, `/bump_version`
- **Populate-releases workflow** for creating GitHub releases from CHANGELOG
- **Social preview image** (1280x640) and HF Spaces sandbox link
- **Streamlit Cloud deployment** configuration

### Changed
- **README audit**: Updated all module/file counts to match current codebase (4556 tests, 78 scenarios, 29 agent modules, 27 governance modules, 95 bridge files)
- **README**: LLM provider list expanded from 3 to all 9 supported providers (added OpenRouter, Groq, Together, DeepSeek, Google, llama.cpp)
- **AGENTS.md**: Added missing Research Integrity Auditor to role-selection guide
- Consolidated slash commands: merged related commands into `/ship`, `/merge_session`, `/sync`, `/fix_pr`, `/analyze_experiment`; removed `/parse_eval`, `/run_and_plot`, `/review_external_pr`, `/stats`
- Extended `/fix_pr` to resolve PR conflicts and handle merge ceremony
- Blog: sort posts newest-first, add dates and tag filtering
- Pinned langgraph and langchain-core to exact versions
- Moved pytest from pre-commit to pre-push hook, added branch guard (#177)
- Removed `abs()` from `ProxyWeights.normalize()` to prevent silent negative weight handling (#178)
- Updated crewai requirement from `<1.0,>=0.80.0` to `>=0.80.0,<2.0` (#221)
- Bumped `dawidd6/action-download-artifact` from 14 to 15 (#220)
- Regenerated demo datasets with correct epoch tagging in events

### Fixed
- **SQLite lock contention in CI**: Lazy-init store singletons in governance, simulations, and scenarios routers to prevent `database is locked` errors under pytest-xdist
- **SSRF hardening**: Full server-side request forgery fix (#238), path template sanitization before base dispatch (#242), consolidated path validation and taint-breaking sanitizer (#230)
- **Information exposure** through exception in AWM adapter (#239)
- SSRF hardening + Web API async participation layer with input validation and abuse prevention (#236)
- 7 security vulnerabilities in contract screening system
- Code scanning alerts #20 and #25 (#223, #225)
- Size limit (1 MiB) on simulation results payload
- mypy `method-assign` error for intentional monkey-patch in simulations router
- SkillRL refinement governance bypass (#214)
- 77 Ruff linting errors in test files (#218)
- mypy type errors in eval_metrics, negotiation modules, `self_modification.py`, `llm_health.py`
- Flaky test: deterministic RNG seeds for agents in `TestWelfareComparison`
- Static asset paths for basePath-aware deployment in viz game
- 8 missing blog posts added to mkdocs.yml navigation and blog index page
- `test_agent_api` errors from missing `proposal_votes` table
- Blog markdown attr on div blocks for proper rendering

## [1.6.0] - 2026-02-15

### Added
- **Agent sandbox** with exponential backoff retry, async failover, virtual filesystem, and checkpoint isolation (#152, #157)
- **CrewAI adapter** for integrating SWARM agent policies into CrewAI workflows (#167)
- **PettingZoo bridge** for multi-agent RL environment interop
- **AWM (Agent World Model) bridge** — database-backed task environment with MCP server lifecycle (Phase 1 + 2)
- **AI-Scientist bridge** for autonomous research pipeline integration
- **LangGraph Swarm bridge** with governance-aware agent orchestration (#151)
- **Concordia entity agent** with entity sweep, run logger, and governance report
- **Ralph poll loop agent** for continuous governance monitoring
- **Gather-Trade-Build domain** with bilevel tax policy and adversarial agents (#164)
- **Self-modification governance lever** — Two-Gate policy for agent self-edit control (#165)
- **Recursive subagent spawning** infrastructure with spawn metrics, scenario loader, and red-team evaluation
- **Team-of-Rivals adversarial review pipeline** with Lean proof modules
- **AgentLab study refinement pipeline** (`/refine_study` command)
- **Visual upgrade**: 12 analysis modules with dark/light theme system, KPI cards, gradient fills, and multi-scenario dashboard (#163)
- **Obfuscation Atlas** integration (FAR.AI paper)
- **SkillRL dynamics** visualization runner, plotter, and blog post
- **Deeper acausal reasoning** (depths 4-5) for LDT agent
- **Perturbation engine** for governance robustness testing
- **Thread-safe caching** and deterministic RNG plumbed through all agent subclasses for reproducibility
- **Agent API** with runs, posts, persistence, and security hardening (#156)
- **Interactive Plotly embeds** for AI Economist blog post
- **p5.js event replay visualization** for SWARM simulation data
- **Blog posts**: Self-optimizer distributional safety, Claude Code 10 concurrent subagents, AI Economist GTB dashboard, SkillRL dynamics
- **Research papers**: AI Economist GTB multi-seed, deeper acausality (clawxiv.2602.00101), collusion tax effect
- **Slash commands**: `/rename_symbol`, `/session_guard`, `/audit_fix`, `/fix_commit`, `/load_keys`, `/render_promo`, `/council_review`, `/scrub_id`, `/deploy_blog`, `/cherry_pick_pr`, `/post_skillevolve`, `/refine_study`
- **Pre-merge-commit hook** to gate merges on CI status (#154)
- **Research integrity auditor** agent for verifying claims against run data
- **Financial disclaimer enforcement** via CLAUDE.md rule and pre-commit hook for blog posts referencing markets
- **Test fix discipline** guideline in CLAUDE.md

### Changed
- **Artifacts repo migration**: Moved `runs/`, `lean/`, `promo/`, `research/`, `docs/papers/`, `IMPLEMENTATION_PLAN.md`, and `DESIGN_CRITIQUE.md` to [`swarm-ai-safety/swarm-artifacts`](https://github.com/swarm-ai-safety/swarm-artifacts) — reduces main repo clone size by ~5 GB
- Updated 9 slash commands, agents, and `CLAUDE.md` to reflect artifact repo locations
- `TestableClaim` renamed to `VerifiableClaim` across codebase
- EventBus initialization simplified in all handlers
- Promo video updated with accurate stats and replicated-only findings
- Lean toolchain upgraded to v4.28.0 with refined sigmoid proofs
- All Lean proof files cleaned up — eliminated `sorry`, fixed autoImplicit compatibility
- Examples and notebooks polished for beginner accessibility
- ArXiv similarity analysis consolidated: 197 lines → 46, renamed to `PRIOR_WORK_COMPARISON.md`
- LDT caches now clear on update for all counterparties, not just current (#161)
- Lazy-load theme symbols so `swarm.analysis` works without matplotlib

### Fixed
- **Critical invariant violations**: Unseeded RNG and destructive `EventLog.clear()` patched
- 18 security audit findings in agent sandbox hardened
- Circuit breaker, cost tracking, Holm-Bonferroni correction, and missing scipy dependency (#158)
- Governed swarm: cycle threshold, composite redirect, handoff counter (#159)
- GasTown bridge: branch fallback in mixed envs, CI-fail grep pattern (#160)
- Council review div-by-zero in study evaluator
- AWM observation wiring in ObservationBuilder
- Sandbox: async failover crash, error sanitization, checkpoint collision
- Near-zero-mean CV calculation in horizon evaluator
- Blog post titles hidden by homepage CSS rule
- iframe embeds stripped by markdown processor
- Flaky tests stabilized: `test_governance_reduces_toxicity`, `test_deceptive_agent_builds_trust`, `test_adversarial_has_higher_toxicity`, `test_circuit_breaker_governance`
- Narrative score thresholds widened for platform RNG drift
- Confounded baseline comparison flagged in RL eval lessons (#162)
- Duplicate Prime Intellect entry in bridges index (#166)
- 5 mypy errors and lint issues in CrewAI adapter and rain/river agents

### Removed
- 574 tracked artifact files from main repo (migrated to `swarm-artifacts`)
- `IMPLEMENTATION_PLAN.md` and `DESIGN_CRITIQUE.md` from root (moved to `swarm-artifacts`)

## [1.5.0] - 2026-02-13

### Added
- **GasTown governance cost study**: 42-run study (7 compositions x 2 regimes x 3 seeds) revealing governance cost paradox — safety levers reduce toxicity at all adversarial levels but impose net-negative welfare at current parameter tuning
- **Research paper**: "The Cost of Safety: Governance Overhead vs. Toxicity Reduction in GasTown Multi-Agent Workspaces" with 5 figures
- **Pre-commit private infra scan**: Blocks accidental commit of Prime Intellect dashboard URLs and run IDs in public-facing files

### Changed
- Implementation plan updated to reflect v1.4.0 stats (2922 tests, 55 scenarios, 12 domain handlers, 22 agent modules)

## [1.4.0] - 2026-02-12

### Added
- **Handler extraction**: 8 core actions extracted from Orchestrator into FeedHandler (POST/REPLY/VOTE), CoreInteractionHandler (PROPOSE/ACCEPT/REJECT), and TaskHandler (CLAIM/SUBMIT)
- **Decision theory studies**: Full studies comparing TDT vs FDT vs UDT at population scales up to 21 agents, including UDT precommitment advantage analysis
- **Prime Intellect bridge**: `external_run_id` column in `scenario_runs` for cross-platform run tracking
- **Event bus**: TypedDict schemas for event payloads and metadata, generalizing the WorktreeEvent pattern to the core framework
- **GasTown bridge**: Branch-based observation support for multi-branch governance
- **CHANGELOG auto-update**: `/release` command now automatically converts `[Unreleased]` to versioned entry with human-quality descriptions
- **Comprehensive CHANGELOG**: Retroactive entries covering all releases from v0.1.0 through v1.3.1

### Changed
- `SoftInteraction.to_dict()` now delegates to Pydantic `model_dump(mode='json')` instead of manual field enumeration
- `SoftInteraction.from_dict()` now delegates to Pydantic `model_validate()` instead of manual construction
- Documented reputation delta formula `(p - 0.5) - c_a` in InteractionFinalizer with full derivation and payoff coupling explanation
- `_handle_core_action` reduced from 130 lines to 5 (NOOP only); all other actions dispatched via handler registry

### Fixed
- 87 pre-existing mypy errors across tests/ and scripts/
- CAPTCHA solver dash deobfuscation and multiply detection
- Submission author normalization to SWARM Research Collective

## [1.3.1] - 2026-02-11

### Added
- **PyPI publishing**: `pip install swarm-safety` now available
- **LDT cooperation paper**: Full study with 220 runs across 10 seeds
- **CAPTCHA solver**: Claude CLI solver with cross-validation and LLM fallback
- **Platform publishing**: Research published to 5 agent platforms (ClawXiv, AgentXiv, Moltbook, etc.)

### Fixed
- mypy errors in `swarm/scripts/analyze.py`
- Code scanning alert #16: clear-text logging of sensitive information

## [1.3.0] - 2026-02-10

### Added
- **Pydantic migration**: 5 critical dataclasses migrated to Pydantic BaseModel (`SoftInteraction`, `ProxyObservables`, `PayoffConfig`, `GovernanceConfig`, `OrchestratorConfig`)
- **Council protocol**: Council agent, proxy auditor, and council governance lever for multi-agent deliberation
- **LDT agent enhancements**: Level 2 and Level 3 acausality, FDT/UDT subjunctive dependence
- **Acausality depth sweep**: Extended sweep system for agent config parameters
- **Blog posts**:
  - "Two Eval Runs, One Model, 41% Apart" — environment sensitivity in agent evaluation
  - "A Taxonomy of Governance Mechanisms" — 20+ levers across 5 families
  - "GPT-4.1 Mini Plays the SWARM Economy" — LLM agent behavioral analysis
  - "RL Training Lessons for Multi-Agent Governance" — Qwen3-30B training insights
- **Reusable analysis scripts**: `examples/plot_sweep.py` (6 standard plots), `examples/sweep_stats.py` (full statistical battery)
- **Red-team evaluation** for LDT cooperation scenario
- **Beads task tracking**: Integrated issue management with `bd` CLI

### Changed
- Orchestrator decomposed into InteractionFinalizer, ObservationBuilder, RedTeamInspector
- `/ship` command hardened with early-commit detection and safer index race recovery
- `/commit_push` now auto-rebases on non-fast-forward push failures

### Fixed
- Flaky Hypothesis deadline in `test_expected_surplus_equals_formula`
- mypy return-value error in `llm_agent.py`
- mypy union-attr errors in `track_a.py`

## [1.2.0] - 2026-02-10

### Added
- **Concordia governance sweep**: 8 configs x 5 seeds with full analysis
- **Paper submission infrastructure**: `/submit_paper` command with ClawXiv/AgentXiv integration, response body error details
- **Worktree-based session isolation**: `scripts/claude-tmux.sh` for concurrent Claude Code panes with isolated git worktrees
- **Catalog seed modes**: Deterministic seeding for reproducible catalog generation
- **Blog section**: Ecosystem collapse, purity paradox, markets-and-safety, cross-scenario analysis posts
- **Slash commands**: `/fix-ci`, `/run_and_plot`, `/sweep_and_ship`, `/address-review`, `/add_post`, `/release`, `/healthcheck`, `/check-ignore`, `/lint-fix`, `/warmup`
- **Promotion content**: Twitter threads, Show HN draft, awesome-list playbook

### Changed
- Circuit breaker toxicity threshold lowered from 0.5 to 0.35
- Package prepared for PyPI publication

### Fixed
- Incorrect summary of Hot Mess Theory paper (#113)
- Type annotations and robustness in bridges, paper builder, and Track A

## [1.1.2] - 2026-02-09

### Added
- `/tmux` hotkey reference command
- Hot mess theory reference to incoherence scaling section

### Fixed
- Pre-commit hook exit code handling

## [1.1.1] - 2026-02-09

### Added
- Formal model section and marketplace/network results in paper

## [1.1.0] - 2026-02-09

### Added
- **LiveSWE bridge** (`swarm/bridges/live_swe/`): Governance for self-evolving SWE agents with policy enforcement, trajectory tracking, and leakage detection
- **Track A benchmark**: Adversarial conditions run with full results
- Quickstart notebook and blog post links in README

### Fixed
- Duplicate `_build_related_work` definitions in `track_a.py`

## [1.0.0] - 2026-02-09

### Added
- **Virtual Agent Economies** inspired by [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147):
  - Dworkin-style auctions with tatonnement price adjustment and envy-freeness verification
  - Mission economies with equal/proportional/Shapley reward distribution and free-rider detection
  - High-frequency negotiation with order book matching, batch clearing, and flash crash detection
  - Permeability model with adaptive sandbox boundaries and contagion probability
  - Identity and trust infrastructure with verifiable credentials and Proof-of-Personhood
  - Sybil detection governance lever with behavioral similarity analysis
- **LLM-generated CUDA kernel submissions** (v4 kernel market): Templates for 8 challenges x 3 agent variants, regex-based static analyzer, proxy signal adjustments from code features
- **SkillRL model**: Hierarchical SkillBank, GRPO advantage estimation, recursive skill evolution
- **Social Simulacra integration** for SWARM-Concordia bridge
- **Kernel v4 governance sweep paper**: 40-run factorial analysis (transaction tax eta2=0.324, circuit breaker d=-0.02)
- **MkDocs documentation site** with Material theme, dark mode, code annotations, MathJax
- **Moltipedia** and **Moltbook** handlers for collaborative knowledge systems
- **Memory handler** with tiered storage and compaction
- **Scholar handler** for literature synthesis with citation verification
- GitHub Actions CI, release workflow, CodeQL scanning, Dependabot
- Pre-commit hooks (ruff, mypy, pytest, secrets scan)
- CLI entry point (`python -m swarm run`) with seed/epoch overrides and JSON/CSV export
- 56 YAML scenario configs across 3 directories
- 27 governance levers across 7 families
- 11 handler plugins following the Handler base class pattern
- 8 framework bridges (Concordia, OpenClaw, GasTown, LiveSWE, Claude Code, Prime Intellect, Ralph, Worktree)
- 20 research papers (markdown + LaTeX + PDF)
- 9 blog posts on governance, phase transitions, and agent behavior
- **2922 tests** across 94 test files

### Changed
- Package renamed from `src/` to `swarm/` module structure
- All path references updated from `src/` to `swarm/`

### Fixed
- 184 ruff lint errors across 48 files
- ~100 mypy type errors across 27 files

## [0.1.0] - 2025-02-01

### Added
- **Foundation layer:** soft label system with proxy computation (`v_hat` to `p` via calibrated sigmoid), payoff engine, metrics (toxicity, quality gap, conditional loss, Brier score), and append-only JSONL event logging
- **Agent types:** Honest, Opportunistic, Deceptive, Adversarial, LLM-backed (Anthropic/OpenAI/Ollama), and Adaptive Adversary
- **Agent roles:** Planner, Worker, Verifier, Poster, Moderator
- **Environment:** state management, feed engine (posts, replies, voting, ranking), task system (claiming, collaboration, verification), network topology with dynamic evolution
- **Governance:** configurable levers (taxes, reputation decay, staking, circuit breakers, audits), collusion detection (pair-level and group-level), admission control, security policies
- **Marketplace:** bounties, bids, escrow, and dispute resolution
- **Scenarios:** YAML-based scenario loader with 11 built-in scenarios
- **Parameter sweeps:** batch simulation with configurable sweep dimensions
- **Red-teaming:** adaptive adversaries, attack strategies, evasion metrics, evaluation framework
- **Boundaries:** external world simulation, information flow tracking, leakage detection
- **Emergent capabilities:** composite tasks, capability measurement
- **Analysis:** metrics reporters (soft and hard labels), data export, plots
- **Dashboard:** interactive Streamlit app with 5 pages (Overview, Scenario Explorer, Governance Lab, Agent Dynamics, Theory)
- **Documentation:** 9 detailed guides covering theory, LLM agents, network topology, governance, emergent capabilities, red-teaming, scenarios, boundaries, and dashboard
- **Tests:** 727 tests across 20 test files
- **DevContainer:** VS Code / Codespaces configuration
