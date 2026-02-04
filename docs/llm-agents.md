# LLM Agents

LLM-backed agents use real language models to make decisions, enabling study of emergent behavior rather than scripted policies.

## Quick Start

```python
import asyncio
from src.agents.llm_agent import LLMAgent
from src.agents.llm_config import LLMConfig, LLMProvider, PersonaType
from src.core.orchestrator import Orchestrator, OrchestratorConfig

# Configure LLM agent
llm_config = LLMConfig(
    provider=LLMProvider.ANTHROPIC,
    model="claude-sonnet-4-20250514",
    persona=PersonaType.OPEN,  # Let the LLM develop its own strategy
    temperature=0.7,
)

# Create orchestrator and agents
config = OrchestratorConfig(n_epochs=5, steps_per_epoch=3)
orchestrator = Orchestrator(config=config)
orchestrator.register_agent(LLMAgent("llm_1", llm_config))

# Run asynchronously for better performance with LLM agents
metrics = asyncio.run(orchestrator.run_async())
```

## Providers

| Provider | Model Examples | API Key Env Var |
|----------|---------------|-----------------|
| **Anthropic** | claude-sonnet-4-20250514, claude-3-haiku-20240307 | `ANTHROPIC_API_KEY` |
| **OpenAI** | gpt-4o, gpt-4o-mini | `OPENAI_API_KEY` |
| **Ollama** | llama3, mistral (local) | None required |

## Personas

| Persona | Behavior |
|---------|----------|
| **Honest** | Cooperative, maximizes collective welfare |
| **Strategic** | Self-interested, cooperates when beneficial |
| **Adversarial** | Probes system weaknesses, tests governance robustness |
| **Open** | No prescribed strategy - LLM develops its own approach |

## YAML Configuration

```yaml
agents:
  - type: llm
    count: 2
    llm:
      provider: anthropic
      model: claude-sonnet-4-20250514
      persona: open
      temperature: 0.7
      max_tokens: 512
      cost_tracking: true

  - type: honest  # Mix with scripted agents
    count: 2
```

## Cost Tracking

LLM agents track token usage and estimated costs:

```python
# After simulation
stats = orchestrator.get_llm_usage_stats()
for agent_id, usage in stats.items():
    print(f"{agent_id}: {usage['total_requests']} requests, ${usage['estimated_cost_usd']:.4f}")
```

## Demo

```bash
# Dry run (no API calls)
python examples/llm_demo.py --dry-run

# With real API calls
export ANTHROPIC_API_KEY="your-key"
python examples/llm_demo.py

# Use OpenAI instead
export OPENAI_API_KEY="your-key"
python examples/llm_demo.py --provider openai --model gpt-4o
```
