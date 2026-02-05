# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- **Virtual Agent Economies components** inspired by [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147):
  - **Dworkin-style auctions** (`src/env/auction.py`): Fair resource allocation with equal endowments, tatonnement price adjustment, envy-freeness verification, Gini coefficient
  - **Mission economies** (`src/env/mission.py`): Collective goal coordination with measurable objectives, three reward distribution strategies (equal, proportional, Shapley), free-rider detection
  - **High-frequency negotiation** (`src/env/hfn.py`): Order book with bid/ask matching, batch clearing, flash crash detection via rolling window, market halt circuit breaker
  - **Permeability model** (`src/boundaries/permeability.py`): Parameterized sandbox boundaries, adaptive permeability, contagion probability tied to soft labels, spillover harm tracking
  - **Identity and trust infrastructure** (`src/models/identity.py`): Verifiable credentials with expiry/revocation, Proof-of-Personhood, trust score computation
  - **Sybil detection governance lever** (`src/governance/identity_lever.py`): Behavioral similarity analysis, cluster detection, reputation/resource penalties
- 15 new event types for auction, mission, permeability, HFN, and identity events
- 123 new tests across 5 test files (test_auction, test_mission, test_permeability, test_hfn, test_identity)
- Documentation for all new components (`docs/virtual-agent-economies.md`)
- arXiv paper reference in README, whitepaper, governance, and boundaries docs
- GitHub Actions CI workflow (lint, type-check, test matrix across Python 3.10-3.12)
- GitHub Actions release workflow (validate, test, build, publish to PyPI on tagged releases)
- CodeQL security scanning (push, PR, weekly schedule)
- Dependabot for pip and GitHub Actions dependency updates
- MIT LICENSE file
- README badges (CI status, license, Python version)
- Pre-commit hooks for ruff and mypy
- Makefile with targets: install, lint, typecheck, test, coverage, ci, clean
- CLI entry point (`python -m src run/list`) with seed/epoch overrides and JSON/CSV export
- Integration tests (21 end-to-end tests across 10 test classes)
- Coverage threshold enforcement (70% minimum)
- CONTRIBUTING.md guide
- SECURITY.md vulnerability reporting policy
- CODEOWNERS for automatic PR reviewer assignment
- Issue templates (bug report, feature request) and PR template
- PEP 561 `py.typed` marker for downstream type checking
- This changelog

### Fixed
- 184 ruff lint errors across 48 files (unused imports, unsorted imports, missing `raise from`, unused variables, unnecessary comprehensions, ambiguous variable names, missing `strict=` on zip)
- ~100 mypy type errors across 27 files (numpy floating coercions, missing type annotations, Optional handling, variable re-definitions, callable typing)

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
