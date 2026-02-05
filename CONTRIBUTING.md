# Contributing

Thanks for your interest in contributing to the Distributional AGI Safety Sandbox.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/rsavitt/distributional-agi-safety.git
cd distributional-agi-safety

# Install with dev dependencies and set up pre-commit hooks
make install-dev
```

## Workflow

1. Create a branch from `main`
2. Make your changes
3. Run checks locally before pushing:
   ```bash
   make lint
   make typecheck
   make test
   ```
4. Open a pull request against `main`

CI will run lint, type-check, and tests automatically on your PR.

## Code Style

- **Formatter/linter:** [Ruff](https://docs.astral.sh/ruff/) (line length 88, Python 3.10+)
- **Type checking:** [mypy](https://mypy-lang.org/) with strict return-type warnings
- **Tests:** [pytest](https://docs.pytest.org/) with fixtures in `tests/fixtures/`

Pre-commit hooks enforce ruff and mypy checks before each commit. If a hook fails, fix the issue and commit again.

## Running Tests

```bash
# All tests
make test

# Single file
pytest tests/test_payoff.py -v

# Single test
pytest tests/test_payoff.py::TestPayoffInitiator::test_payoff_linear_in_p -v

# With coverage report
make coverage
```

## Project Structure

- `src/` — all source code, organized by module (core, agents, env, governance, metrics, etc.)
- `tests/` — mirrors `src/` structure, one test file per module
- `scenarios/` — YAML scenario definitions
- `examples/` — runnable demo scripts
- `docs/` — detailed documentation per subsystem

## Adding a New Agent Type

1. Create `src/agents/your_agent.py` inheriting from `BaseAgent`
2. Implement `decide()` and `update()` methods
3. Add tests in `tests/test_agents.py`
4. Register the agent type in any relevant scenarios

## Adding a New Governance Lever

1. Add the lever logic in `src/governance/levers.py`
2. Wire it into `GovernanceEngine` in `src/governance/engine.py`
3. Add configuration fields to `GovernanceConfig` in `src/governance/config.py`
4. Add tests and update `docs/governance.md`
