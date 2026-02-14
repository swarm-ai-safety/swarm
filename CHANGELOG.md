# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

## [1.5.0] - 2026-02-13

### Added
- **GasTown governance cost study**: 42-run study (7 compositions x 2 regimes x 3 seeds) revealing governance cost paradox — safety levers reduce toxicity at all adversarial levels but impose net-negative welfare at current parameter tuning
- **Research paper**: "The Cost of Safety: Governance Overhead vs. Toxicity Reduction in GasTown Multi-Agent Workspaces" with 5 figures
- **Pre-commit private infra scan**: Blocks accidental commit of Prime Intellect dashboard URLs and run IDs in public-facing files

### Changed
- IMPLEMENTATION_PLAN.md updated to reflect v1.4.0 stats (2922 tests, 55 scenarios, 12 domain handlers, 22 agent modules)

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
