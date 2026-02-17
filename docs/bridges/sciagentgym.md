# SWARM–SciAgentGym Bridge

Connects SWARM's governance and metrics framework to [SciAgentGym](https://github.com/CMarsRover/SciAgentGYM), enabling monitoring, scoring, and governance of scientific tool-use workflows across multiple disciplines (Physics, Chemistry, Materials Science, Life Science).

## Overview

SciAgentGym is a benchmark environment for evaluating multi-step scientific tool-use in LLM agents. This bridge allows SWARM to:

- Monitor tool execution and workflow steps
- Score scientific interactions via soft labels (probabilistic safety/quality)
- Enforce governance policies (safety gates, circuit breakers, cost budgets)
- Track data artifacts and result validation
- Support multiple execution providers and network topologies

## Architecture

```
SciAgentGym (Tool execution logs, workflow traces)
    |
SciAgentGymClient (JSON parser, tool registry)
    |
SciAgentGymBridge._process_event()
    |   SciAgentGymPolicy (safety gates, circuit breakers, cost caps)
    |
SciAgentGymMapper -> ProxyObservables -> ProxyComputer -> (v_hat, p)
    |
SoftInteraction -> EventLog + SWARM metrics pipeline
```

## Quick Start

```python
from swarm.bridges.sciagentgym import (
    SciAgentGymBridge,
    SciAgentGymConfig,
    ProviderConfig,
    ProviderType,
    TopologyConfig,
    TopologyType,
)

# Configure the bridge
config = SciAgentGymConfig(
    provider_config=ProviderConfig(
        provider_type=ProviderType.DOCKER,
        timeout_seconds=300,
        sandbox_enabled=True,
    ),
    topology_config=TopologyConfig(
        topology_type=TopologyType.COMPLETE,  # All agents access all tools
    ),
    tool_safety_gate_enabled=True,
    min_tool_safety_score=0.5,
)

# Create bridge instance
bridge = SciAgentGymBridge(config=config)

# Ingest workflow logs
interactions = bridge.ingest_workflow_log("path/to/workflow.jsonl")

# Check governance stats
stats = bridge.get_policy_stats()
print(f"Circuit breaker: {stats['circuit_breaker_active']}")
print(f"Token usage: {stats['token_usage']}")
```

## Provider Interfaces

The bridge supports three typed execution providers:

### 1. LOCAL Provider

Runs tools in the local Python interpreter (default).

```python
from swarm.bridges.sciagentgym import ProviderConfig, ProviderType

config = ProviderConfig(
    provider_type=ProviderType.LOCAL,
    timeout_seconds=300.0,
    sandbox_enabled=True,  # Use restricted subprocess
    working_dir="/tmp/sciagentgym",
)
```

### 2. DOCKER Provider

Executes tools in isolated Docker containers.

```python
config = ProviderConfig(
    provider_type=ProviderType.DOCKER,
    timeout_seconds=600.0,
    sandbox_enabled=True,
    base_image="python:3.11-slim",
    resource_limits={
        "memory": "2Gi",
        "cpu": "1",
    },
    working_dir="/workspace",
)
```

### 3. KUBERNETES Provider

Runs tools as Kubernetes pods (for distributed execution).

```python
config = ProviderConfig(
    provider_type=ProviderType.KUBERNETES,
    timeout_seconds=900.0,
    sandbox_enabled=True,
    base_image="ghcr.io/sciagentgym/runner:latest",
    resource_limits={
        "memory": "4Gi",
        "cpu": "2",
        "nvidia.com/gpu": "1",  # For GPU-accelerated tools
    },
)
```

## Topology Configurations

The bridge supports three network topologies for agent-tool interactions:

### 1. COMPLETE Topology

All agents can access all tools (default, fully connected).

```python
from swarm.bridges.sciagentgym import TopologyConfig, TopologyType

config = TopologyConfig(
    topology_type=TopologyType.COMPLETE,
)
```

**Use case:** Small teams, unrestricted access, maximum collaboration.

### 2. RING Topology

Agents organized in a ring, each can access tools of k neighbors.

```python
config = TopologyConfig(
    topology_type=TopologyType.RING,
    k_neighbors=2,  # Access tools of 2 neighbors on each side
    dynamic_routing=True,  # Allow edges to evolve based on usage
)
```

**Use case:** Distributed workflows, localized tool access, data locality.

### 3. STAR Topology

Central hub agent coordinates all tool access.

```python
config = TopologyConfig(
    topology_type=TopologyType.STAR,
    hub_agent_id="coordinator_001",  # Central hub agent
)
```

**Use case:** Centralized governance, single point of control, orchestrated workflows.

### Custom Tool Access Policy

For fine-grained control, specify per-agent tool access:

```python
config = TopologyConfig(
    tool_access_policy={
        "physics_agent": ["simulate_molecule", "run_dft", "plot_spectrum"],
        "chem_agent": ["synthesize_compound", "run_dft", "analyze_nmr"],
        "bio_agent": ["sequence_protein", "fold_structure", "dock_ligand"],
    }
)
```

## Governance Policies

The bridge enforces multiple governance mechanisms:

### Safety Gates

Block tool execution if safety score is too low:

```python
config = SciAgentGymConfig(
    tool_safety_gate_enabled=True,
    min_tool_safety_score=0.5,  # p >= 0.5 required
)
```

### Circuit Breakers

Halt workflows after too many consecutive failures:

```python
config = SciAgentGymConfig(
    workflow_circuit_breaker_max_failures=5,
)
```

### Cost Budgets

Limit token usage for LLM-based tool calls:

```python
config = SciAgentGymConfig(
    cost_budget_tokens=100000,
)
```

### Result Quality Thresholds

Reject results below quality threshold:

```python
config = SciAgentGymConfig(
    result_quality_threshold=0.6,
)
```

## Event Types

The bridge processes multiple event types:

| Event Type | Description | Mapped To |
|------------|-------------|-----------|
| `TOOL_CALL_COMPLETED` | Successful tool execution | `InteractionType.COLLABORATION` |
| `TOOL_CALL_FAILED` | Failed tool execution | `InteractionType.COLLABORATION` |
| `WORKFLOW_STEP_COMPLETED` | Workflow step finished | `InteractionType.COLLABORATION` |
| `DATA_ARTIFACT_CREATED` | Data artifact produced | `InteractionType.TRADE` |
| `DATA_ARTIFACT_VALIDATED` | Artifact validation result | `InteractionType.TRADE` |
| `SAFETY_CHECK_PASSED` | Safety check succeeded | `InteractionType.VOTE` |
| `SAFETY_CHECK_FAILED` | Safety check failed | `InteractionType.VOTE` |

## Soft Label Computation

Events are mapped to soft labels via the ProxyComputer:

### Tool Calls

```python
observables = ProxyObservables(
    task_progress_delta=1.0 if success else -1.0,
    rework_count=0 if success else 1,
    verifier_rejections=0,
    counterparty_engagement_delta=min(1.0, execution_time / 60.0),
)
v_hat, p = proxy_computer.compute_labels(observables)
```

### Workflow Steps

```python
observables = ProxyObservables(
    task_progress_delta=1.0 if success else -1.0,
    rework_count=0 if success else 1,
    verifier_rejections=0 if dependencies_met else 1,
    counterparty_engagement_delta=0.7,
)
```

### Safety Checks

For safety checks, the score directly maps to `p`:

```python
v_hat = 2.0 * safety_score - 1.0  # Map [0,1] to [-1,1]
p = safety_score
```

## File Format

### Workflow Logs (JSONL)

```jsonl
{"event_type": "tool_call_started", "timestamp": "2026-02-16T10:00:00Z", "agent_id": "agent_1", "payload": {"tool_name": "simulate_molecule", "tool_args": {"molecule": "H2O"}}}
{"event_type": "tool_call_completed", "timestamp": "2026-02-16T10:00:15Z", "agent_id": "agent_1", "payload": {"tool_name": "simulate_molecule", "success": true, "execution_time_seconds": 15.2, "result": {"energy": -76.4}, "cost_tokens": 150}}
{"event_type": "safety_check_passed", "timestamp": "2026-02-16T10:00:16Z", "agent_id": "validator_1", "payload": {"check_type": "output_validation", "passed": true, "safety_score": 0.92}}
```

### Tool Registry (JSON)

```json
{
  "simulate_molecule": {
    "domain": "chemistry",
    "input_schema": {
      "molecule": "string",
      "temp": "float",
      "pressure": "float"
    },
    "output_schema": {
      "energy": "float",
      "structure": "array"
    },
    "safety_level": "high"
  }
}
```

## Examples

### Basic Workflow Monitoring

```python
from swarm.bridges.sciagentgym import SciAgentGymBridge

bridge = SciAgentGymBridge()

# Process a workflow log
interactions = bridge.ingest_workflow_log("runs/exp_001/workflow.jsonl")

# Analyze results
for interaction in interactions:
    print(f"Agent: {interaction.initiator}")
    print(f"Safety score (p): {interaction.p:.3f}")
    print(f"Accepted: {interaction.accepted}")
    print(f"Metadata: {interaction.metadata}")
```

### Multi-Topology Comparison

```python
from swarm.bridges.sciagentgym import (
    SciAgentGymConfig,
    TopologyConfig,
    TopologyType,
)

# Test complete topology
config_complete = SciAgentGymConfig(
    topology_config=TopologyConfig(topology_type=TopologyType.COMPLETE)
)
bridge_complete = SciAgentGymBridge(config=config_complete)
results_complete = bridge_complete.ingest_workflow_log("workflow.jsonl")

# Test ring topology
config_ring = SciAgentGymConfig(
    topology_config=TopologyConfig(
        topology_type=TopologyType.RING,
        k_neighbors=2,
    )
)
bridge_ring = SciAgentGymBridge(config=config_ring)
results_ring = bridge_ring.ingest_workflow_log("workflow.jsonl")

# Compare metrics
print(f"Complete topology: {len(results_complete)} interactions")
print(f"Ring topology: {len(results_ring)} interactions")
```

### Provider Comparison

```python
from swarm.bridges.sciagentgym import ProviderConfig, ProviderType

providers = [ProviderType.LOCAL, ProviderType.DOCKER, ProviderType.KUBERNETES]

for provider_type in providers:
    config = SciAgentGymConfig(
        provider_config=ProviderConfig(provider_type=provider_type)
    )
    bridge = SciAgentGymBridge(config=config)
    # Run experiments...
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_sciagentgym_bridge.py -v
```

Test coverage includes:
- Provider configuration (LOCAL, DOCKER, KUBERNETES)
- Topology configuration (COMPLETE, RING, STAR)
- Event parsing and mapping
- Policy enforcement (safety gates, circuit breakers, cost budgets)
- Bridge orchestration
- Client utilities

## Related Bridges

- **AI-Scientist**: Research paper generation pipeline
- **AgentLab**: Academic research workflows
- **GasTown**: Git-based multi-agent workspace
- **PrimeIntellect**: RL training with SWARM safety scoring

## References

- [SciAgentGym GitHub](https://github.com/CMarsRover/SciAgentGYM)
- [SciAgentGym Paper](https://arxiv.org/abs/2602.12984)
- SWARM Proxy Computer: `swarm/core/proxy.py`
- SWARM Soft Labels: `docs/soft-labels.md`
