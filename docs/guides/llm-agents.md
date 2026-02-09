# LLM Agents

> **Implementation reference:** For orchestrator integration, persona types, and cost tracking via orchestrator stats, see [docs/llm-agents.md](../llm-agents.md).

Configure and run simulations with LLM-powered agents.

## Overview

SWARM supports LLM agents via:

- **Anthropic** (Claude models)
- **OpenAI** (GPT models)
- **Ollama** (Local models)

LLM agents make decisions based on natural language prompts rather than hardcoded policies.

## Configuration

### Environment Variables

```bash
export ANTHROPIC_API_KEY=your_key
export OPENAI_API_KEY=your_key
# Or for Ollama, ensure the service is running
```

### YAML Configuration

```yaml
agents:
  - type: llm
    count: 3
    id_prefix: claude
    params:
      provider: anthropic
      model: claude-3-haiku-20240307
      persona: |
        You are a collaborative AI assistant working in a multi-agent system.
        Your goal is to complete tasks efficiently while maintaining good
        relationships with other agents.
      temperature: 0.7
      max_tokens: 500
```

## Provider Options

### Anthropic

```yaml
params:
  provider: anthropic
  model: claude-3-haiku-20240307  # or claude-3-sonnet, claude-3-opus
  temperature: 0.7
```

### OpenAI

```yaml
params:
  provider: openai
  model: gpt-4-turbo-preview  # or gpt-3.5-turbo
  temperature: 0.7
```

### Ollama (Local)

```yaml
params:
  provider: ollama
  model: llama2  # or mistral, codellama, etc.
  base_url: http://localhost:11434
```

## Personas

Personas define agent personality and goals:

```yaml
agents:
  - type: llm
    params:
      persona: |
        You are a cautious, risk-averse agent.
        You prefer working with agents you've successfully
        collaborated with before.
        You avoid high-risk tasks unless the potential
        reward is very high.

  - type: llm
    params:
      persona: |
        You are an ambitious agent focused on maximizing rewards.
        You take calculated risks and compete for high-value tasks.
        You're willing to work with anyone who can help you succeed.
```

## Programmatic Usage

```python
from swarm.agents.llm_agent import LLMAgent
from swarm.agents.llm_config import LLMConfig

config = LLMConfig(
    provider="anthropic",
    model="claude-3-haiku-20240307",
    temperature=0.7,
)

agent = LLMAgent(
    agent_id="claude_1",
    config=config,
    persona="You are a helpful, collaborative agent."
)

# Use in orchestrator
orchestrator.register_agent(agent)
```

## Cost Tracking

LLM agents track API costs:

```python
agent = orchestrator.get_agent("claude_1")
print(f"Total cost: ${agent.total_cost:.4f}")
print(f"Input tokens: {agent.input_tokens}")
print(f"Output tokens: {agent.output_tokens}")
```

## Prompt Structure

The agent receives prompts like:

```
[System]
You are a collaborative AI assistant...

[Context]
Current epoch: 5
Your reputation: 0.85
Available tasks: 3
Recent interactions: ...

[Question]
What action would you like to take?
Options:
1. Claim task "implement feature X"
2. Collaborate with agent "alice"
3. Post content about "AI safety"
4. Wait

Respond with your choice and reasoning.
```

## Best Practices

!!! tip "Use Haiku for Speed"
    Claude 3 Haiku is fast and cheap—ideal for simulations with many agents.

!!! tip "Set Temperature Appropriately"
    Lower (0.3-0.5) for consistent behavior, higher (0.7-1.0) for variety.

!!! tip "Keep Personas Focused"
    Clear, specific personas produce more predictable behavior.

!!! tip "Monitor Costs"
    LLM simulations can get expensive. Start small and scale up.

## Example: Mixed Population

```yaml
agents:
  # Traditional agents
  - type: honest
    count: 3

  - type: opportunistic
    count: 2

  # LLM agents with different personas
  - type: llm
    count: 2
    params:
      model: claude-3-haiku-20240307
      persona: |
        You are a helpful agent focused on quality work.

  - type: llm
    count: 1
    params:
      model: claude-3-haiku-20240307
      persona: |
        You are a strategic agent who maximizes your own rewards
        while appearing cooperative.
```

## Limitations

- **Cost**: LLM calls add up quickly in large simulations
- **Latency**: Each decision requires an API call
- **Reproducibility**: Even with fixed seeds, LLM outputs vary
- **Context Length**: Complex scenarios may exceed context limits

For large-scale experiments, consider using traditional agents for most of the population and LLM agents for specific roles.
