# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install for development
python -m pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_payoff.py -v

# Run a specific test
pytest tests/test_payoff.py::TestPayoffInitiator::test_payoff_linear_in_p -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Architecture

This is a simulation framework for studying distributional safety in multi-agent AI systems using **soft (probabilistic) labels** instead of binary good/bad classifications.

### Data Flow

```
Observables → ProxyComputer → v_hat → sigmoid → p → SoftPayoffEngine → payoffs
                                                  ↓
                                            SoftMetrics → toxicity, quality gap, etc.
```

### Key Components

**`src/core/proxy.py`** - `ProxyComputer` converts observable signals (task_progress, rework_count, verifier_rejections, engagement) into `v_hat ∈ [-1, +1]` using weighted combination, then applies calibrated sigmoid to get `p = P(v = +1)`.

**`src/core/payoff.py`** - `SoftPayoffEngine` implements payoffs using soft labels:
- `S_soft = p * s_plus - (1-p) * s_minus` (expected surplus)
- `E_soft = (1-p) * h` (expected harm externality)
- Payoffs include surplus share, transfers, governance costs, externality costs, and reputation

**`src/metrics/soft_metrics.py`** - `SoftMetrics` computes probabilistic metrics:
- Toxicity: `E[1-p | accepted]`
- Quality gap: `E[p | accepted] - E[p | rejected]` (negative = adverse selection)
- Conditional loss: selection effect on payoffs

**`src/metrics/reporters.py`** - `MetricsReporter` provides dual reporting of soft (probabilistic) and hard (threshold-based) metrics for comparison.

**`src/logging/event_log.py`** - Append-only JSONL logger for simulation replay. Can reconstruct `SoftInteraction` objects from event stream.

### Test Fixtures

`tests/fixtures/interactions.py` provides generators for test data:
- `generate_benign_batch()` - high p, positive outcomes
- `generate_toxic_batch()` - low p, exploitation patterns
- `generate_mixed_batch()` - realistic distribution
- `generate_adversarial_scenario()` - coordinated attack pattern

## Domain Concepts

- **p**: Probability that interaction is beneficial, `P(v = +1)`, always in `[0, 1]`
- **v_hat**: Raw proxy score before sigmoid, in `[-1, +1]`
- **Adverse selection**: When low-quality interactions are preferentially accepted (quality_gap < 0)
- **Externality internalization**: ρ parameters control how much agents bear cost of ecosystem harm
