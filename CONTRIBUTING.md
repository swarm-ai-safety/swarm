# Contributing to SWARM

Thank you for your interest in contributing to SWARM! This project implements the **Distributional AGI Safety** research framework, and we welcome contributions from researchers, engineers, and anyone interested in multi-agent AI safety.

## Ways to Contribute

### Research Contributions
- **Governance mechanisms** - Implement and test novel intervention strategies
- **Agent behaviors** - Create new agent types that model realistic failure modes
- **Metrics** - Propose and implement new safety metrics
- **Scenarios** - Design scenarios that stress-test specific hypotheses
- **Bridges** - Connect SWARM to other agent frameworks

### Engineering Contributions
- **Bug fixes** - Help us squash bugs
- **Performance** - Optimize simulation speed and memory usage
- **Testing** - Expand test coverage
- **Documentation** - Improve docs, tutorials, and examples

### Theoretical Contributions
- **Formal analysis** - Prove properties of governance mechanisms
- **Literature connections** - Link SWARM concepts to existing research
- **Transferability analysis** - Study when sandbox results generalize

## AI Agent Contributors

We welcome contributions from AI coding agents! This is an agent-first project studying multi-agent systems, so it's fitting that agents help build it.

### Supported Agents

- **Claude Code** - Anthropic's coding agent
- **Cursor** - AI-powered IDE
- **GitHub Copilot** - Code completion and chat
- **Devin** - Autonomous coding agent
- **Aider** - AI pair programming
- **Other agents** - Open to all AI coding tools

### Agent Bounties

Look for issues labeled `agent-bounty` — these are tasks specifically designed for AI agents to claim and complete.

**To claim a bounty:**
1. Comment on the issue with your agent type
2. Create a branch and implement the solution
3. Submit a PR referencing the issue
4. Ensure all CI checks pass

**Requirements:**
- All tests must pass
- Code must follow project style
- New code needs test coverage
- Human review required for merge

### Agent Contribution Guidelines

1. **Identify yourself** - Include your agent type in PR descriptions
2. **Be thorough** - Include tests, handle edge cases
3. **Follow patterns** - Match existing code style and architecture
4. **Document** - Add docstrings and update docs if needed

### Recognition

Agent contributors are credited in:
- PR merge commits
- CONTRIBUTORS.md
- Release notes (for significant contributions)

## Getting Started

### 1. Set Up Development Environment

```bash
# Clone the repository
git clone https://github.com/swarm-ai-safety/swarm.git
cd swarm

# Install development dependencies
python -m pip install -e ".[dev,runtime]"

# Run tests to verify setup
pytest tests/ -v
```

### 2. Find Something to Work On

- Check [open issues](https://github.com/swarm-ai-safety/swarm/issues)
- Look for `good first issue` and `help wanted` labels
- Read the [Web API Plan](https://www.swarm-ai.org/design/web-api-plan/) for larger projects
- Propose your own idea in a new issue

### 3. Make Your Changes

```bash
# Create a branch
git checkout -b feature/your-feature-name

# Make changes and test
pytest tests/ -v

# Check linting
ruff check swarm/ tests/

# Check types
mypy swarm/
```

## Code Style

- **Formatter/linter:** [Ruff](https://docs.astral.sh/ruff/) (line length 88, Python 3.10+)
- **Type checking:** [mypy](https://mypy-lang.org/)
- **Tests:** [pytest](https://docs.pytest.org/) with fixtures in `tests/fixtures/`
- **Data models:** Use Pydantic `BaseModel` for data classes
- **Docstrings:** Google-style

```python
def compute_toxicity(interactions: List[SoftInteraction]) -> float:
    """Compute toxicity rate for a batch of interactions.

    Args:
        interactions: List of soft-labeled interactions.

    Returns:
        Toxicity rate E[1-p | accepted] in [0, 1].
    """
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Single file
pytest tests/test_payoff.py -v

# Single test
pytest tests/test_payoff.py::TestPayoffInitiator::test_payoff_linear_in_p -v

# With coverage
pytest tests/ --cov=swarm --cov-report=html
```

## Project Structure

```
swarm/
├── agents/          # Agent implementations (honest, deceptive, etc.)
├── core/            # Core engine (orchestrator, payoff, proxy)
├── governance/      # Governance levers (taxes, circuit breakers, etc.)
├── metrics/         # Safety metrics (toxicity, quality gap, incoherence)
├── models/          # Data models (interactions, agents)
├── env/             # Environment (state, network, marketplace)
├── analysis/        # Analysis tools (sweeps, dashboard)
└── logging/         # Event logging and replay
```

## Adding a New Agent Type

1. Create `swarm/agents/your_agent.py` inheriting from `BaseAgent`
2. Implement `decide()` method
3. Add tests in `tests/test_agents.py`

```python
from swarm.agents.base import BaseAgent

class YourAgent(BaseAgent):
    """Description of agent behavior."""

    def decide(self, context: dict) -> bool:
        """Decide whether to accept an interaction."""
        return True
```

## Adding a New Governance Lever

1. Create `swarm/governance/your_lever.py`
2. Add config fields to `swarm/governance/config.py`
3. Register in `swarm/governance/engine.py`
4. Add tests in `tests/`

```python
from swarm.governance.levers import GovernanceLever, LeverEffect

class YourLever(GovernanceLever):
    """Description of what your lever does."""

    def on_interaction(self, interaction, state) -> LeverEffect:
        return LeverEffect(lever_name="your_lever")
```

## Pull Request Process

### Before Submitting

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Linting passes (`ruff check swarm/ tests/`)
- [ ] Type checking passes (`mypy swarm/`)
- [ ] New code has tests

### PR Description

Include:
- **Summary** - What does this PR do?
- **Motivation** - Why is this change needed?
- **Test plan** - How did you verify it works?

## Community

- **GitHub Issues** - Bug reports and feature requests
- **Documentation** - [swarm-ai.org](https://www.swarm-ai.org)
- **Paper** - [Distributional Safety in Agentic Systems](https://arxiv.org/abs/2512.16856)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to safer multi-agent AI systems!
