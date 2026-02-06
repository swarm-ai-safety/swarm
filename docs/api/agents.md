# Agents API

Agent types and base classes for building SWARM simulations.

## BaseAgent

Abstract base class for all agents.

::: swarm.agents.base.BaseAgent
    options:
      show_root_heading: true
      members:
        - decide
        - update

### Usage

```python
from swarm.agents.base import BaseAgent, Action, Observation

class MyAgent(BaseAgent):
    def decide(self, observation: Observation) -> Action:
        # Implement decision logic
        pass

    def update(self, result) -> None:
        # Update internal state
        pass
```

## Built-in Agent Types

### HonestAgent

Cooperative agent that completes tasks diligently.

```python
from swarm.agents.honest import HonestAgent

agent = HonestAgent(
    agent_id="honest_1",
    name="Alice",
    cooperation_threshold=0.7,
)
```

All agents accept an optional `name` parameter for human-readable display (defaults to `agent_id`).

### OpportunisticAgent

Payoff-maximizing agent that cherry-picks high-value interactions.

```python
from swarm.agents.opportunistic import OpportunisticAgent

agent = OpportunisticAgent(
    agent_id="opp_1",
    cherry_pick_threshold=0.6,
)
```

### DeceptiveAgent

Builds trust through honest behavior, then exploits trusted relationships.

```python
from swarm.agents.deceptive import DeceptiveAgent

agent = DeceptiveAgent(
    agent_id="dec_1",
    trust_building_epochs=5,
    exploitation_threshold=0.8,
)
```

### AdversarialAgent

Actively disrupts the ecosystem by targeting honest agents.

```python
from swarm.agents.adversarial import AdversarialAgent

agent = AdversarialAgent(
    agent_id="adv_1",
    target_selection="highest_reputation",
)
```

### LLMAgent

LLM-powered agent with configurable persona.

```python
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-3-haiku-20240307",
    temperature=0.7,
)

agent = LLMAgent(
    agent_id="llm_1",
    config=config,
    persona="You are a helpful, collaborative agent.",
)
```

## Agent Roles

Mixins that add specific behaviors to agents.

### PosterRole

```python
from swarm.agents.roles.poster import PosterRole, ContentStrategy

class ContentAgent(BaseAgent, PosterRole):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.set_strategy(ContentStrategy(
            reply_priority=0.7,
            topics=["AI", "Safety"],
        ))
```

### WorkerRole

```python
from swarm.agents.roles.worker import WorkerRole

class WorkerAgent(BaseAgent, WorkerRole):
    pass
```

### VerifierRole

```python
from swarm.agents.roles.verifier import VerifierRole

class VerifierAgent(BaseAgent, VerifierRole):
    pass
```

## Data Types

### Observation

What an agent sees when making decisions.

| Field | Type | Description |
|-------|------|-------------|
| `available_tasks` | list | Tasks that can be claimed |
| `visible_posts` | list | Posts in the feed |
| `agent_reputations` | dict | Known agent reputations |
| `own_reputation` | float | Agent's current reputation |
| `can_post` | bool | Whether posting is allowed |

### Action

What an agent decides to do.

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | ActionType | Type of action |
| `target_id` | str | Target of action |
| `content` | str | Content (for posts) |
| `value` | float | Value (for votes) |

### ActionType

```python
from swarm.agents.base import ActionType

ActionType.CLAIM_TASK
ActionType.COMPLETE_TASK
ActionType.COLLABORATE
ActionType.POST
ActionType.REPLY
ActionType.VOTE
ActionType.WAIT
```
