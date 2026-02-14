# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Recommended workflow (SWARM)

This repo is set up as a **Claude Code template** for SWARM-style research work:

- Custom slash commands live in `.claude/commands/` (e.g. `/run_scenario`, `/sweep`, `/plot`, `/red_team`).
- Research-role specialist agents live in `.claude/agents/`.
- Role-selection guidance lives in `AGENTS.md`; keep it synchronized with `.claude/agents/`.
- Optional git hygiene hooks live in `.claude/hooks/` (install via `/install_hooks`).
- MCP integrations are configured in `.mcp.json` (safe-by-default placeholders; no secrets committed).

### Run artifacts

Prefer writing experiment outputs to a self-contained run folder:

- `runs/<timestamp>_<scenario>_seed<seed>/history.json` (JSON export)
- `runs/<timestamp>_<scenario>_seed<seed>/csv/` (CSV exports)
- `runs/<...>/plots/` (generated plots)

The legacy `logs/` directory remains for scenario-declared outputs, but `runs/` is the canonical “reproduce from PR” format.

## Commands

```bash
# Install for development
python -m pip install -e ".[dev,runtime]"

# Run all tests (use python -m to ensure correct environment)
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_payoff.py -v

# Run a specific test
python -m pytest tests/test_payoff.py::TestPayoffInitiator::test_payoff_linear_in_p -v

# Run with coverage
python -m pytest tests/ --cov=swarm --cov-report=html

# Lint
ruff check swarm/ tests/

# Type check
python -m mypy swarm/

# Run a scenario (CLI)
python -m swarm run scenarios/baseline.yaml --seed 42 --epochs 10 --steps 10

# Run a scenario (example runner)
python examples/run_scenario.py scenarios/baseline.yaml
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

**`swarm/core/proxy.py`** - `ProxyComputer` converts observable signals (task_progress, rework_count, verifier_rejections, engagement) into `v_hat ∈ [-1, +1]` using weighted combination, then applies calibrated sigmoid to get `p = P(v = +1)`.

**`swarm/core/payoff.py`** - `SoftPayoffEngine` implements payoffs using soft labels:
- `S_soft = p * s_plus - (1-p) * s_minus` (expected surplus)
- `E_soft = (1-p) * h` (expected harm externality)
- Payoffs include surplus share, transfers, governance costs, externality costs, and reputation

**`swarm/metrics/soft_metrics.py`** - `SoftMetrics` computes probabilistic metrics:
- Toxicity: `E[1-p | accepted]`
- Quality gap: `E[p | accepted] - E[p | rejected]` (negative = adverse selection)
- Conditional loss: selection effect on payoffs

**`swarm/metrics/reporters.py`** - `MetricsReporter` provides dual reporting of soft (probabilistic) and hard (threshold-based) metrics for comparison.

**`swarm/logging/event_log.py`** - Append-only JSONL logger for simulation replay. Can reconstruct `SoftInteraction` objects from event stream.

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

## Multi-Session Worktree Workflow

When running 15+ concurrent Claude Code sessions, each session runs in its own git worktree to avoid index races and branch conflicts.

### Launch

```bash
./scripts/claude-tmux.sh 4    # Launch 4 isolated sessions
./scripts/claude-tmux.sh kill  # Kill tmux session (worktrees kept)
./scripts/claude-tmux.sh cleanup  # Remove all session worktrees + branches
```

### Environment Variables

Each session pane has these env vars set via `scripts/detect-session.sh`:

| Variable | Example | Description |
|---|---|---|
| `IS_SESSION_WORKTREE` | `true` | Whether this shell is inside a session worktree |
| `SESSION_ID` | `session-2` | Worktree directory name |
| `WORKTREE_ID` | `pane-2` | Pane identifier |
| `SESSION_BRANCH` | `session/pane-2` | Git branch for this session |
| `MAIN_REPO_ROOT` | `/path/to/repo` | Absolute path to the main repo |

### Session-Aware Commands

| Command | Worktree behavior |
|---|---|
| `/status` | Shows session identity and main repo path at top |
| `/pr` | Branches from `origin/main` instead of checking out `main` |
| `/commit_push` | Uses `bd --sandbox sync` to avoid beads daemon contention |
| `/merge_session` | Merges session branch into main (run from main repo) |
| `/merge_all_sessions` | Merges all session branches at once |

### Beads in Sessions

Use `bd --sandbox` in worktrees to avoid contention with the main repo's beads daemon. The `/commit_push` command does this automatically.

## Paper Author Resolution

When `/write_paper` or `/compile_paper` needs an author name, resolve in this order:

1. `$SWARM_AUTHOR` environment variable (if set)
2. `git config user.name` (if set)
3. Prompt the user

Never guess or infer from the OS username.

## Test fix discipline

- When fixing a flaky test, prefer making it deterministic (set seeds, constrain inputs) over loosening assertions.

## Safety / invariants (do not break)

- `p` must remain in `[0, 1]` everywhere it is surfaced or logged.
- Event logs (`*.jsonl`) are append-only and should remain replayable.
- Runs should be reproducible from: scenario YAML + seed + exported history/CSVs.
