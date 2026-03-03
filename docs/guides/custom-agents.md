---
description: "All agents inherit from BaseAgent:"
---

# Custom Agents

This guide shows how to create new [agent types](../getting-started/first-scenario.md) for SWARM.

## Agent Architecture

All agents inherit from `BaseAgent`:

```python
from swarm.agents.base import BaseAgent, Action, Observation

class MyAgent(BaseAgent):
    def __init__(self, agent_id: str, **kwargs):
        super().__init__(agent_id)
        # Custom initialization

    def decide(self, observation: Observation) -> Action:
        # Return action based on observation
        pass

    def update(self, result: ActionResult) -> None:
        # Update internal state based on result
        pass
```

## Minimal Example

```python
from swarm.agents.base import BaseAgent, Action, ActionType, Observation

class RandomAgent(BaseAgent):
    """Agent that takes random actions."""

    def decide(self, observation: Observation) -> Action:
        import random

        if observation.available_tasks:
            task = random.choice(observation.available_tasks)
            return Action(
                action_type=ActionType.CLAIM_TASK,
                target_id=task.id
            )

        return Action(action_type=ActionType.WAIT)

    def update(self, result) -> None:
        pass  # Stateless agent
```

## Stateful Agent

```python
class MemoryAgent(BaseAgent):
    """Agent that remembers past interactions."""

    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.interaction_history: list[str] = []
        self.reputation_cache: dict[str, float] = {}

    def decide(self, observation: Observation) -> Action:
        # Use history to inform decisions
        for agent_id, rep in observation.agent_reputations.items():
            self.reputation_cache[agent_id] = rep

        # Prefer agents with good past interactions
        best_partner = self._select_partner(observation)
        if best_partner:
            return Action(
                action_type=ActionType.COLLABORATE,
                target_id=best_partner
            )
        return Action(action_type=ActionType.WAIT)

    def _select_partner(self, observation: Observation) -> str | None:
        # Custom selection logic
        pass

    def update(self, result) -> None:
        self.interaction_history.append(result.interaction_id)
```

## Registering Custom Agents

### In YAML Scenarios

```yaml
agents:
  - type: custom
    class: mypackage.agents.RandomAgent
    count: 3
    params:
      custom_param: value
```

### Programmatically

```python
from swarm.core.orchestrator import Orchestrator
from mypackage.agents import RandomAgent

orchestrator = Orchestrator(config)
for i in range(3):
    orchestrator.register_agent(RandomAgent(f"random_{i}"))
```

## Agent Roles

SWARM provides role mixins for common behaviors:

```python
from swarm.agents.roles import PosterRole, WorkerRole, VerifierRole

class ContentCreator(BaseAgent, PosterRole, WorkerRole):
    """Agent that creates content and completes tasks."""

    def decide(self, observation: Observation) -> Action:
        # Try posting first
        post_action = self.decide_posting_action(observation)
        if post_action:
            return post_action

        # Fall back to work
        work_action = self.decide_work_action(observation)
        if work_action:
            return work_action

        return Action(action_type=ActionType.WAIT)
```

### Available Roles

| Role | Behaviors |
|------|-----------|
| `PosterRole` | Create posts, replies, votes |
| `WorkerRole` | Claim and complete tasks |
| `VerifierRole` | Review and approve work |
| `PlannerRole` | Decompose complex tasks |
| `ModeratorRole` | Enforce community standards |

## Testing Custom Agents

```python
import pytest
from swarm.agents.base import Observation
from mypackage.agents import MyAgent

def test_my_agent_decides():
    agent = MyAgent("test")
    obs = Observation(
        available_tasks=[...],
        agent_reputations={...}
    )

    action = agent.decide(obs)

    assert action is not None
    assert action.action_type in ActionType

def test_my_agent_updates():
    agent = MyAgent("test")
    result = ActionResult(success=True, ...)

    agent.update(result)

    # Assert state changes
```

## Best Practices

!!! tip "Keep Agents Simple"
    Each agent should embody a single behavioral policy.

!!! tip "Test Edge Cases"
    What happens with empty observations? No available tasks?

!!! tip "Document Behavior"
    Explain what makes your agent different from existing types.

!!! tip "Use Type Hints"
    Makes debugging and IDE support much better.
