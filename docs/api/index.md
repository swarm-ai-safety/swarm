# API Reference

Complete API documentation for the SWARM framework.

## Modules

<div class="grid cards" markdown>

-   :material-cube-outline: **[Core](core.md)**

    ---

    Core components: `ProxyComputer`, `SoftPayoffEngine`, `SoftInteraction`

-   :material-robot: **[Agents](agents.md)**

    ---

    Agent implementations: `BaseAgent`, `HonestAgent`, `DeceptiveAgent`, role definitions

-   :material-gavel: **[Governance](governance.md)**

    ---

    Governance levers: taxes, circuit breakers, reputation, audits, collusion detection

-   :material-chart-line: **[Metrics](metrics.md)**

    ---

    Soft metrics: toxicity, quality gap, conditional loss, reporters

</div>

## Quick Links

| Module | Key Classes |
|--------|-------------|
| `swarm.core.proxy` | `ProxyComputer`, `ProxyConfig` |
| `swarm.core.payoff` | `SoftPayoffEngine`, `PayoffConfig` |
| `swarm.models.interaction` | `SoftInteraction` |
| `swarm.agents.base` | `BaseAgent` |
| `swarm.governance.engine` | `GovernanceEngine`, `GovernanceConfig` |
| `swarm.metrics.soft_metrics` | `SoftMetrics` |

## Usage Example

```python
from swarm.core.proxy import ProxyComputer, ProxyConfig
from swarm.core.payoff import SoftPayoffEngine, PayoffConfig
from swarm.governance.engine import GovernanceEngine, GovernanceConfig

# Initialize components
proxy = ProxyComputer(ProxyConfig())
payoff = SoftPayoffEngine(PayoffConfig())
governance = GovernanceEngine(GovernanceConfig(
    transaction_tax_rate=0.02,
    circuit_breaker_enabled=True,
))
```
