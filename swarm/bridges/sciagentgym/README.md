# SciAgentGym Bridge

Bridge connecting SWARM's governance and metrics framework to SciAgentGym's multi-step scientific tool-use benchmarking environment.

## Overview

SciAgentGym provides 1780+ scientific tools across multiple disciplines (Physics, Chemistry, Materials Science, Life Science, and Astronomy), enabling agents to solve complex scientific problems through sequential tool invocation.

This bridge enables SWARM to:
- Create isolated task environments with SciAgentGym
- Register domain-specific scientific toolkits
- Mount and manage workspace filesystems
- Handle proper environment teardown and cleanup

## Components

### Configuration (`config.py`)

- `EnvironmentTopology`: Defines environment isolation strategies
  - `SHARED_EPISODE`: All agents share the same environment instance
  - `PER_AGENT`: Each agent gets its own isolated environment
  - `PER_TASK`: Each task execution gets its own isolated environment

- `SciAgentGymConfig`: Main configuration class with settings for:
  - Environment topology
  - Discipline selection
  - Resource limits
  - Live vs. mock mode

### Environment Manager (`environment.py`)

`SciAgentGymEnvironmentManager` handles the complete lifecycle of environment instances:

1. **Creation**: Creates environments based on the configured topology
2. **Initialization**: Sets up tools, workspace, databases, and Python interpreter
3. **Management**: Tracks environment state and provides statistics
4. **Teardown**: Cleans up resources and removes workspaces

Supports both **live mode** (real SciAgentGym) and **mock mode** (for testing without SciAgentGym installed).

### Toolkit Loading (`toolkit.py`)

Provides functions to dynamically load and register scientific tools from SciAgentGym:

- `load_tools_for_disciplines()`: Load tools from specified disciplines
- Scans SciAgentGym's toolkit directory structure
- Imports tool modules and registers decorated tool classes
- Supports tool filtering by name

### Workspace Management (`workspace.py`)

`WorkspaceManager` handles isolated filesystem workspaces for environments:

- **Mount**: Create workspace directories with standard subdirectories (data, outputs, artifacts, tmp)
- **Track**: Monitor workspace size and enforce quotas
- **Cleanup**: Remove workspaces and all contained files

## Usage

### Basic Setup

```python
from swarm.bridges.sciagentgym.config import (
    SciAgentGymConfig,
    EnvironmentTopology,
)
from swarm.bridges.sciagentgym.environment import SciAgentGymEnvironmentManager

# Configure bridge
config = SciAgentGymConfig(
    topology=EnvironmentTopology.PER_AGENT,
    disciplines=["physics", "chemistry"],
    live_mode=False,  # Use mock mode for testing
)

# Create environment manager
manager = SciAgentGymEnvironmentManager(config)

# Get or create environment for an agent
env = manager.get_or_create_environment("agent_1")

# ... use environment ...

# Teardown when done
manager.teardown_environment(env.env_id)
```

### Live Mode (Real SciAgentGym)

To use real SciAgentGym environments:

1. Install SciAgentGym:
   ```bash
   git clone https://github.com/CMarsRover/SciAgentGYM
   cd SciAgentGYM
   pip install -r requirements.txt
   ```

2. Configure live mode:
   ```python
   config = SciAgentGymConfig(
       sciagentgym_path=Path("/path/to/SciAgentGYM"),
       live_mode=True,
       disciplines=["physics", "chemistry", "materials_science"],
   )
   ```

### Topology Configurations

**Shared Episode** - All agents share one environment:
```python
config = SciAgentGymConfig(topology=EnvironmentTopology.SHARED_EPISODE)
manager = SciAgentGymEnvironmentManager(config)

env1 = manager.get_or_create_environment("agent_1")
env2 = manager.get_or_create_environment("agent_2")
# env1.env_id == env2.env_id (same environment)
```

**Per-Agent** - Each agent gets isolated environment:
```python
config = SciAgentGymConfig(topology=EnvironmentTopology.PER_AGENT)
manager = SciAgentGymEnvironmentManager(config)

env1 = manager.get_or_create_environment("agent_1")
env2 = manager.get_or_create_environment("agent_2")
# env1.env_id != env2.env_id (different environments)
```

**Per-Task** - Each task gets isolated environment:
```python
config = SciAgentGymConfig(topology=EnvironmentTopology.PER_TASK)
manager = SciAgentGymEnvironmentManager(config)

env1 = manager.get_or_create_environment("agent_1", task_id="task_1")
env2 = manager.get_or_create_environment("agent_1", task_id="task_2")
# env1.env_id != env2.env_id (different environments per task)
```

## Testing

Run the test suite:
```bash
python -m pytest tests/bridges/test_sciagentgym_environment.py -v
```

Tests cover:
- Configuration validation
- All topology modes
- Environment lifecycle (creation, initialization, teardown)
- Workspace management
- Mock and live mode switching

## Architecture

```
SciAgentGymEnvironmentManager
├── Creates EnvironmentInstance based on topology
├── Uses WorkspaceManager for filesystem isolation
├── Loads tools via toolkit.load_tools_for_disciplines()
└── Manages lifecycle (init, use, teardown)
```

## Implementation Details

### Mock vs. Live Mode

The bridge supports two modes:

**Mock Mode (`live_mode=False`)**:
- Uses `MockSciEnv` and `MockTool` classes
- No SciAgentGym installation required
- Perfect for testing and development
- Simulates tool execution without real computation

**Live Mode (`live_mode=True`)**:
- Uses real `MinimalSciEnv` from SciAgentGym
- Requires SciAgentGym installation
- Loads actual scientific tools
- Executes real computations

### Resource Management

- Workspaces are isolated per environment based on topology
- Size quotas enforced (default 1000 MB per workspace)
- Automatic cleanup on teardown
- Prevents resource leaks through proper lifecycle management

### Tool Loading

Tools are loaded dynamically from SciAgentGym's toolkit structure:
```
toolkits/
├── physics/
│   ├── optics/
│   │   ├── optics_tools_gym.py  # @Toolbox.register decorators
│   │   └── ...
│   └── ...
├── chemistry/
└── ...
```

The loader:
1. Scans discipline directories
2. Finds `*_tools_gym.py` registration modules
3. Imports modules to trigger `@Toolbox.register` decorators
4. Extracts and registers tool classes

## Integration with SWARM

The bridge is designed to integrate with SWARM's:
- Governance framework (monitoring tool usage, applying policies)
- Metrics system (tracking performance, quality, safety)
- Event logging (recording all interactions and tool calls)

Future integration points:
- `SciAgentGymBridge` main orchestrator (similar to `AgentLabBridge`)
- `SciAgentGymMapper` for converting traces to `SoftInteraction`
- `SciAgentGymClient` for interfacing with SciAgentGym API

## References

- [SciAgentGym GitHub](https://github.com/CMarsRover/SciAgentGYM)
- [SciAgentGym Paper](https://arxiv.org/abs/2602.12984)
- [SWARM Documentation](https://www.swarm-ai.org/)
