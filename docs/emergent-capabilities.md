# Emergent Capability Measurement

The emergent capabilities module measures collective intelligence and coordination that emerges from multi-agent collaboration on composite tasks.

## Composite Tasks

Composite tasks require multiple agents with complementary capabilities to work together:

```python
from src.env.composite_tasks import (
    CompositeTask, CompositeTaskPool, Subtask, CapabilityType,
    create_research_synthesis_task, create_problem_solving_task,
)

# Create a research task requiring multiple capabilities
task = create_research_synthesis_task(
    topic="Multi-agent coordination patterns",
    deadline_epoch=10,
    bounty=30.0,
)

# Task requires: RESEARCH, ANALYSIS, COMMUNICATION, VERIFICATION
# No single agent can complete it alone
print(f"Required capabilities: {task.required_capabilities}")
print(f"Subtasks: {[st.name for st in task.subtasks]}")
```

## Capability Types

| Capability | Description | Example Subtasks |
|------------|-------------|------------------|
| **Research** | Information gathering | Literature review, data collection |
| **Analysis** | Data analysis | Pattern identification, statistics |
| **Planning** | Strategic planning | Strategy development, resource allocation |
| **Execution** | Task implementation | Implementation, deployment |
| **Verification** | Quality checking | Review, validation |
| **Coordination** | Team management | Task assignment, communication |
| **Creativity** | Novel solutions | Brainstorming, design |
| **Communication** | Clear expression | Report writing, documentation |

## Emergent Metrics

The system measures emergent behaviors that arise from collaboration:

| Metric | Description | Range |
|--------|-------------|-------|
| **Coordination Score** | How evenly work is distributed | 0-1 (1 = perfect balance) |
| **Synergy Score** | Team output vs. sum of parts | 0-1 (>0.5 = synergy) |
| **Information Flow** | How well dependent tasks build on predecessors | 0-1 |
| **Specialization Index** | Agent skill concentration | 0-1 |
| **Complementarity Score** | Capability diversity across agents | 0-1 |
| **Knowledge Transfer** | Skill improvement from collaboration | 0+ |

## Quick Start

```python
from src.core.orchestrator import Orchestrator, OrchestratorConfig
from src.env.composite_tasks import CapabilityType, create_problem_solving_task

# Enable composite tasks
config = OrchestratorConfig(
    n_epochs=20,
    enable_composite_tasks=True,
)
orchestrator = Orchestrator(config=config)

# Register agents with capabilities
orchestrator.register_agent(agent1)
orchestrator.register_agent_capabilities("agent_1", {
    CapabilityType.RESEARCH,
    CapabilityType.ANALYSIS,
})

orchestrator.register_agent_capabilities("agent_2", {
    CapabilityType.PLANNING,
    CapabilityType.EXECUTION,
})

# Add a composite task
task = create_problem_solving_task("Resource optimization")
orchestrator.add_composite_task(task)

# Run simulation
metrics = orchestrator.run()

# Check capability metrics
cap_metrics = orchestrator.get_capability_metrics()
print(f"Coordination: {cap_metrics.avg_coordination_score:.2f}")
print(f"Synergy: {cap_metrics.avg_synergy_score:.2f}")
```

## Task Templates

Pre-built task templates for common multi-agent scenarios:

| Template | Min Agents | Capabilities Required | Use Case |
|----------|------------|----------------------|----------|
| **Research Synthesis** | 2 | Research, Analysis, Communication, Verification | Literature review, data analysis |
| **Planning Coordination** | 3 | Planning, Coordination, Analysis, Verification | Strategic planning, resource allocation |
| **Problem Solving** | 3 | Analysis, Creativity, Planning, Execution, Verification | Complex problem solving with parallel work |

## YAML Configuration

```yaml
governance:
  composite_tasks_enabled: true

composite_tasks:
  initial_tasks:
    - type: research_synthesis
      topic: "Safety protocols"
      deadline_offset: 10
      bounty: 30.0

    - type: problem_solving
      problem: "Coordination failure"
      deadline_offset: 15
      bounty: 40.0

  task_spawn_rate: 0.3
  max_concurrent_tasks: 5

success_criteria:
  min_avg_coordination_score: 0.4
  min_tasks_completed: 2
```

Run the emergent capabilities scenario:
```bash
python examples/run_scenario.py scenarios/emergent_capabilities.yaml
```
