# GitHub Copilot Instructions

This file provides project context and conventions for GitHub Copilot when working in this repository.

## Project Overview

**SWARM** (System-Wide Risk Evaluation for Multi-Agent AI Systems) is a research framework for measuring emergent failures that only appear when many AI agents interact — even when individual agents are safe. It studies distributional safety using **soft (probabilistic) labels** instead of binary good/bad classifications.

## Commands

```bash
# Install for development
python -m pip install -e ".[dev,runtime]"

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_payoff.py -v

# Run with coverage
python -m pytest tests/ --cov=swarm --cov-report=html

# Lint
ruff check swarm/ tests/

# Type check
python -m mypy swarm/

# Run a scenario (CLI)
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10
```

## Architecture

### Data Flow

```
Observables → ProxyComputer → v_hat → sigmoid → p → SoftPayoffEngine → payoffs
                                                  ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

### Key Components

- **`swarm/core/proxy.py`** — `ProxyComputer` converts observable signals into `v_hat ∈ [-1, +1]` using a weighted combination, then applies a calibrated sigmoid to get `p = P(v = +1)`.
- **`swarm/core/payoff.py`** — `SoftPayoffEngine` computes payoffs from soft labels: expected surplus `S_soft = p * s_plus - (1-p) * s_minus` and expected harm externality `E_soft = (1-p) * h`.
- **`swarm/metrics/soft_metrics.py`** — `SoftMetrics` computes toxicity (`E[1-p | accepted]`), quality gap (`E[p | accepted] - E[p | rejected]`), and conditional loss.
- **`swarm/metrics/reporters.py`** — `MetricsReporter` provides dual reporting of soft (probabilistic) and hard (threshold-based) metrics.
- **`swarm/logging/event_log.py`** — Append-only JSONL logger for simulation replay.

### Test Fixtures

`tests/fixtures/interactions.py` provides generators for test data:
- `generate_benign_batch()` — high p, positive outcomes
- `generate_toxic_batch()` — low p, exploitation patterns
- `generate_mixed_batch()` — realistic distribution
- `generate_adversarial_scenario()` — coordinated attack pattern

## Domain Concepts

- **`p`**: Probability that an interaction is beneficial, `P(v = +1)`, always in `[0, 1]`
- **`v_hat`**: Raw proxy score before sigmoid, in `[-1, +1]`
- **Adverse selection**: When low-quality interactions are preferentially accepted (`quality_gap < 0`)
- **Externality internalization**: `ρ` parameters control how much agents bear the cost of ecosystem harm
- **Illusion delta**: `Δ_illusion = C_perceived − C_distributed` — gap between how good the system looks and how consistent it actually is

## Safety Invariants (do not break)

- `p` must remain in `[0, 1]` everywhere it is surfaced or logged.
- Event logs (`*.jsonl`) are append-only and must remain replayable.
- Runs must be reproducible from: scenario YAML + seed + exported history/CSVs.

## Code Conventions

- **Minimal changes**: Make the smallest change that satisfies the requirement. Avoid refactoring unrelated code.
- **Tests**: New logic should have corresponding tests in `tests/`. Match the style of existing tests.
- **Flaky tests**: Fix by making them deterministic (set seeds, constrain inputs) rather than loosening assertions.
- **Linting**: Code must pass `ruff check swarm/ tests/` before merging.
- **Type annotations**: New functions should include type annotations consistent with existing code.

## Repository Layout

```
swarm/           # Core library (proxy, payoff, metrics, agents, logging)
tests/           # Pytest test suite
scenarios/       # YAML scenario configs
examples/        # Runnable example scripts
docs/            # MkDocs documentation site
.claude/         # Claude Code slash commands and agents (Claude-specific tooling)
```

## PR Guidelines

- PRs must pass CI: `pytest tests/ -v`, `ruff check swarm/ tests/`, and `mypy swarm/`.
- If the change affects scenarios, metrics, agents, or governance, include a reproducible run command and headline metrics in the PR description.
- Verify the invariants checklist: `p ∈ [0, 1]` everywhere, event logs remain replayable.
