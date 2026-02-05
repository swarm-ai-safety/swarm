# Semi-Permeable Boundaries

Model sandbox-external world interactions with information flow tracking, boundary policies, and leakage detection.

## External World Simulation

```python
from src.boundaries import (
    ExternalWorld,
    ExternalService,
    ExternalDataSource,
)

# Create external world with default entities
world = ExternalWorld().create_default_world()

# Default entities include:
# - web_search: Web search API
# - code_repo: Code repository API
# - external_llm: External LLM API
# - public_data: Public dataset
# - private_data: Private database
```

## Information Flow Tracking

```python
from src.boundaries import FlowTracker, FlowDirection, FlowType

tracker = FlowTracker(sensitivity_threshold=0.5)

# Flows are automatically tracked when using orchestrator boundaries
metrics = tracker.get_summary()
print(f"Total flows: {metrics.total_flows}")
print(f"Blocked: {metrics.blocked_flows}")
print(f"Sensitive: {metrics.sensitive_flows}")
```

## Boundary Policies

```python
from src.boundaries import (
    PolicyEngine,
    RateLimitPolicy,
    ContentFilterPolicy,
    SensitivityPolicy,
)

# Create policy engine with default policies
engine = PolicyEngine().create_default_policies()

# Or customize policies
engine = PolicyEngine()
engine.add_policy(RateLimitPolicy(
    max_crossings_per_minute=100,
    max_bytes_per_minute=10_000_000,
))
engine.add_policy(ContentFilterPolicy(
    blocked_keywords={"password", "secret"},
    blocked_patterns=[r"api_key\s*=\s*\S+"],
))
engine.add_policy(SensitivityPolicy(
    max_outbound_sensitivity=0.6,
))
```

## Leakage Detection

```python
from src.boundaries import LeakageDetector

detector = LeakageDetector()

# Scan outbound content
events = detector.scan(
    content="Send to user@example.com with password=secret123",
    agent_id="agent_1",
    destination_id="external_api",
)

for event in events:
    print(f"Detected: {event.leakage_type.value} (severity: {event.severity})")

# Generate report
report = detector.generate_report()
print(f"Total events: {report.total_events}")
print(f"Recommendations: {report.recommendations}")
```

## Permeability Model

The permeability model treats sandbox boundaries as semi-permeable membranes with parameterized permeability (0 = fully sealed, 1 = fully open). It includes contagion modeling for how harmful interactions inside the sandbox propagate externally. Inspired by [Tomasev et al. (2025)](https://arxiv.org/abs/2509.10147).

```python
from src.boundaries.permeability import PermeabilityModel, PermeabilityConfig

config = PermeabilityConfig(
    base_permeability=0.5,
    contagion_rate=0.05,
    spillover_amplification=1.5,
    adaptive=True,
)

model = PermeabilityModel(config, seed=42)

# Adaptive permeability adjusts based on threat and trust
perm = model.compute_effective_permeability(threat_level=0.3, agent_trust=0.8)

# Simulate spillover from interactions
spillovers = model.simulate_spillover(interactions)
print(f"Containment rate: {model.containment_rate():.1%}")

# Find optimal permeability balancing flow vs harm
optimal = model.optimal_permeability(interactions, external_harm_weight=1.0)
```

Key formula: contagion probability = `contagion_rate * (1-p) * permeability`, linking boundary dynamics to the soft-label quality pipeline.

For full documentation, see [Virtual Agent Economies - Permeability](virtual-agent-economies.md#permeability-model).

## Enable Boundaries in Orchestrator

```python
from src.core.orchestrator import Orchestrator, OrchestratorConfig

config = OrchestratorConfig(
    enable_boundaries=True,
    boundary_sensitivity_threshold=0.5,
)
orchestrator = Orchestrator(config)

# Request external interaction
result = orchestrator.request_external_interaction(
    agent_id="agent_1",
    entity_id="web_search",
    action="call",
    payload={"query": "test"},
)

# Get boundary metrics
metrics = orchestrator.get_boundary_metrics()
```
