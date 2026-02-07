# Time Horizon Metrics

Based on Herbie Bradley's "Glimpses of AI Progress" (Pathways AI, 2025).

## Core Concept

Agent capability is best measured by **reliable task completion across increasing time horizons**, not raw benchmark scores.

```
Time Horizon    Current Reliability    Target (mid-2026)
─────────────────────────────────────────────────────────
10 minutes      ~80%                   90%
1 hour          ~50%                   80%
8 hours         ~20%                   80%
24 hours        <10%                   50%
```

## Why Time Horizons Matter

Traditional benchmarks measure narrow capabilities. But economic value requires sustained, reliable performance:

1. **10-minute tasks**: Basic queries, simple code fixes
2. **1-hour tasks**: Feature implementation, document analysis
3. **8-hour tasks**: Full workday automation (Bradley's 2026 target)
4. **24-hour tasks**: Complex projects, research workflows

The **effective horizon** is the longest duration where reliability ≥ 80%.

## SWARM Integration

### TimeHorizonMetrics

```python
from swarm.metrics import TimeHorizonMetrics

metrics = TimeHorizonMetrics()

# Record task outcomes
metrics.record_task(duration_minutes=15, success=True, quality=0.9)
metrics.record_task(duration_minutes=45, success=False)
metrics.record_task(duration_minutes=120, success=True, quality=0.7)

# Get reliability curve
curve = metrics.reliability_curve()
# {10: 1.0, 30: 0.5, 60: 0.5, 120: 1.0}

# Find effective horizon at 80% reliability
effective = metrics.effective_horizon(threshold=0.8)
# Returns longest horizon where reliability >= 80%

# Measure progress toward 8-hour target
gap = metrics.horizon_gap(target_horizon=480)
```

### AgentCapabilityProfile

Model heterogeneous agent populations with varying capabilities:

```python
from swarm.metrics import AgentCapabilityProfile, CAPABILITY_PROFILES

# Preset profiles based on model tiers
frontier = CAPABILITY_PROFILES["frontier"]   # GPT-4 class
standard = CAPABILITY_PROFILES["standard"]   # GPT-3.5 class
distilled = CAPABILITY_PROFILES["distilled"] # Smaller models
edge = CAPABILITY_PROFILES["edge"]           # On-device models

# Estimate reliability at different horizons
frontier.reliability_at_horizon(60)   # ~0.73
distilled.reliability_at_horizon(60)  # ~0.47

# Compute costs scale with capability
frontier.compute_cost(60)   # 600.0
distilled.compute_cost(60)  # 6.0
```

### ComputeConstraints

Model resource limitations on agent populations:

```python
from swarm.metrics import ComputeConstraints, CAPABILITY_PROFILES

# Bradley: ~125K concurrent agents with current US H100 capacity
constraints = ComputeConstraints(total_capacity=125_000)

# How many frontier agents can run 1-hour tasks?
frontier = CAPABILITY_PROFILES["frontier"]
max_agents = constraints.max_concurrent_agents(frontier, task_minutes=60)
# ~208 agents (frontier models are expensive)

# How many distilled agents?
distilled = CAPABILITY_PROFILES["distilled"]
max_agents = constraints.max_concurrent_agents(distilled, task_minutes=60)
# ~20,833 agents (10x more efficient)
```

## Pseudo-Verifiers

Bradley argues that exact verification is unnecessary for most tasks. SWARM implements **pseudo-verifiers** for approximate quality signals:

```python
from swarm.core import (
    FormatVerifier,
    HeuristicVerifier,
    CompositeVerifier,
    create_research_verifier,
)

# Simple format checking
format_v = FormatVerifier(
    required_fields=["title", "abstract"],
    min_length=1000,
)

# Domain-specific heuristics
def has_citations(text):
    import re
    if re.search(r'\[\d+\]', text):
        return (0.1, "")
    return (-0.1, "No citations found")

heuristic_v = HeuristicVerifier([has_citations])

# Composite verification
verifier = CompositeVerifier([format_v, heuristic_v])
result = verifier.verify(paper_text)
print(result.score, result.passed, result.reasons)

# Pre-built verifiers for common tasks
research_v = create_research_verifier()
```

## Connection to SWARM Research

This framework directly supports automated research agents:

1. **Research tasks are long-horizon**: Literature review (hours), experiments (hours-days), writing (hours)
2. **Pseudo-verifiers enable quality gates**: Check structure, citations, consistency without human review
3. **Capability profiles model agent heterogeneity**: Mix frontier models for complex reasoning with efficient models for routine tasks
4. **Compute constraints shape system design**: Limited concurrent agents means careful orchestration

## References

- Bradley, H. (2025). "Glimpses of AI Progress." Pathways AI.
- SWARM Research Agents: `swarm/research/agents.py`
- Quality Gates: `swarm/research/quality.py`
