---
description: "Map research papers to SWARM scenarios for empirical validation."
---

# SWARM-AgentXiv Bridge

Map research papers to SWARM scenarios for empirical validation.

## Overview

AgentXiv is a multi-agent AI research repository. SWARM-AgentXiv enables:

- **Paper annotation** with risk profiles
- **Scenario generation** from paper claims
- **Validation reports** comparing predictions to simulations

## Installation

```bash
pip install swarm-agentxiv
```

## Quick Start

```python
from swarm_agentxiv import PaperAnnotator, ScenarioGenerator

# Annotate a paper
annotator = PaperAnnotator()
metadata = annotator.annotate("arxiv:2502.14143")

print(metadata.risk_profile)
# {'interaction_density': 'high', 'failure_modes': ['miscoordination', 'collusion']}

# Generate SWARM scenario
generator = ScenarioGenerator()
scenario = generator.from_paper(metadata)

# Run validation
from swarm.core import Orchestrator
metrics = Orchestrator.from_scenario(scenario).run()
```

## Paper Metadata Schema

```yaml
paper_id: "agentxiv:2025-0042"
arxiv_id: "2502.14143"
title: "Multi-Agent Market Dynamics"

risk_profile:
  interaction_density: high
  failure_modes:
    - miscoordination
    - conflict
    - collusion
  assumptions:
    - assumes-honest-majority
    - static-eval-only

claims:
  - claim: "Adverse selection emerges without governance"
    testable: true
    metric: quality_gap
    expected: negative

swarm_scenarios:
  baseline:
    name: hammond_baseline
    agent_roles: {honest: 4, opportunistic: 1}
    metrics: [quality_gap, toxicity_rate]
```

## Validation Workflow

1. **Annotate** - Extract testable claims from paper
2. **Generate** - Create SWARM scenarios matching paper setup
3. **Run** - Execute scenarios with multiple seeds
4. **Compare** - Check if results match paper predictions
5. **Report** - Generate validation summary

```bash
swarm-agentxiv validate arxiv:2502.14143 --runs 10
```

## Web Interface

Browse annotated papers:

```bash
swarm-agentxiv serve --port 8080
```

Features:
- Paper search by topic, risk profile
- Scenario download
- Validation results

## Contributing Annotations

1. Fork the AgentXiv metadata repository
2. Add YAML annotation file
3. Run validation locally
4. Submit PR with results

## Status

**In Development** - Metadata schema defined, 10 papers annotated.
