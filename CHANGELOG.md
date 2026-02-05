# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added
- GitHub Actions CI workflow (lint, type-check, test matrix)
- MIT LICENSE file
- README badges (CI status, license, Python version)
- Pre-commit hooks for ruff and mypy
- Makefile for common development tasks
- CONTRIBUTING.md guide
- This changelog

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
