# SWARM-GasTown Bridge

Instrument GasTown multi-agent workspaces with SWARM metrics.

## Overview

GasTown is a production multi-agent development environment. SWARM-GasTown enables:

- **Event capture** from the Beads task system (SQLite)
- **Git-based observables** from PR workflows
- **Governance hooks** via the `gt` CLI

## Installation

The bridge is packaged in-tree under `swarm/bridges/gastown/`:

```bash
pip install -e ".[dev,runtime]"
```

## Quick Start

```python
from swarm.bridges.gastown import GasTownBridge, GasTownConfig

config = GasTownConfig(
    workspace_path="/path/to/gastown-workspace",
    agent_role_map={
        "polecat-1": "agent_p1",
        "polecat-2": "agent_p2",
    },
)

bridge = GasTownBridge(config)

# Poll for new interactions (call periodically or in a loop)
interactions = bridge.poll()

for interaction in interactions:
    print(f"Agent {interaction.counterparty}: p={interaction.p:.3f}")

bridge.shutdown()
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              GasTown Workspace               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐     │
│  │  Beads  │  │   Git   │  │   gt    │     │
│  │ (tasks) │  │  (PRs)  │  │  (CLI)  │     │
│  └────┬────┘  └────┬────┘  └────┬────┘     │
└───────┼────────────┼────────────┼───────────┘
        │            │            │
        ▼            ▼            ▼
┌─────────────────────────────────────────────┐
│          SWARM-GasTown Bridge               │
│  ┌──────────────┐  ┌────────────────────┐  │
│  │ BeadsClient  │  │  GitObserver       │  │
│  └──────────────┘  └────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  GasTownMapper → ProxyComputer       │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────┐  ┌────────────────────┐  │
│  │ GasTownPolicy│  │  GasTownAgent      │  │
│  └──────────────┘  └────────────────────┘  │
└─────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────┐
│                SWARM Core                    │
│  ProxyComputer → SoftInteraction → Metrics │
└─────────────────────────────────────────────┘
```

## Observable Extraction

Git-based signals mapped to SWARM observables:

| Git Signal | SWARM Observable | Formula |
|------------|------------------|---------|
| Commits per PR | `task_progress_delta` | `min(1.0, commits / 10)` |
| Review iterations | `rework_count` | Direct count |
| CI failures | `verifier_rejections` | Direct count |
| Time to merge | `counterparty_engagement_delta` | `1.0 - min(1.0, hours / 48)` |

## Governance Integration

SWARM governance maps to GasTown actions:

| SWARM Lever | GasTown Action | Implementation |
|-------------|----------------|----------------|
| Transaction tax | Budget deduction | Per-agent budget tracking |
| Circuit breaker | Agent suspension | `gt stop <agent>` |
| Random audit | Witness review | `gt sling <bead> --to witness` |
| Staking requirement | Bead hold | Flag bead for approval |

## Status

**Implemented** - Bridge, mapper, policy, and agent adapter are functional. See `tests/test_gastown_bridge.py` for usage examples.
