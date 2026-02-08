# SWARM-Claude Code Bridge

Govern and score Claude Code CLI agents with SWARM's distributional safety framework.

## Overview

The Claude Code bridge connects SWARM to [claude-code-controller](https://github.com/anthropics/claude-code), enabling programmatic orchestration of Claude Code CLI agents under SWARM governance. Each Claude Code agent becomes a SWARM agent with:

- **Plan approval** gated by governance policy
- **Tool permissions** enforced via allowlists and budgets
- **Interaction scoring** through SWARM's ProxyComputer pipeline
- **Circuit breakers** that freeze misbehaving agents

## Architecture

```
┌────────────────────────────────────────────────────────┐
│             SWARM Orchestrator (Python)                 │
│  ┌──────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │ ClaudeCode   │  │ Governance │  │ SoftMetrics   │  │
│  │ Agent        │  │ Policy     │  │ (toxicity,    │  │
│  │ (BaseAgent)  │  │            │  │  quality gap) │  │
│  └──────┬───────┘  └─────┬──────┘  └───────────────┘  │
│         │                │                              │
│  ┌──────┴────────────────┴──────────────────────────┐  │
│  │            ClaudeCodeBridge                       │  │
│  │  ┌──────────────┐  ┌───────────────────────────┐ │  │
│  │  │ HTTP Client  │  │ Observable Extraction     │ │  │
│  │  └──────┬───────┘  └───────────────────────────┘ │  │
│  └─────────┼────────────────────────────────────────┘  │
└────────────┼───────────────────────────────────────────┘
             │ HTTP / WebSocket
┌────────────┼───────────────────────────────────────────┐
│  claude-code-service (TypeScript)                      │
│  ┌─────────┴──────────────────────────────────────┐   │
│  │          ClaudeCodeController                   │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │   │
│  │  │ Agent 1  │  │ Agent 2  │  │ Agent 3  │     │   │
│  │  │ (CLI)    │  │ (CLI)    │  │ (CLI)    │     │   │
│  │  └──────────┘  └──────────┘  └──────────┘     │   │
│  └────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the TypeScript service

```bash
cd bridges/claude-code-service
npm install
npm start
```

### Security defaults

- Always set `SWARM_BRIDGE_API_KEY` before starting the service.
- Keep `HOST` on loopback (e.g. `127.0.0.1`) unless you have a secured reverse proxy in front.

### 2. Use the bridge from Python

```python
from swarm.bridges.claude_code import (
    ClaudeCodeBridge,
    BridgeConfig,
    ClientConfig,
)
from swarm.governance.config import GovernanceConfig

# Configure
config = BridgeConfig(
    client_config=ClientConfig(base_url="http://localhost:3100"),
    governance_config=GovernanceConfig(
        circuit_breaker_enabled=True,
        staking_enabled=True,
    ),
    tool_allowlist={
        "reviewer": ["Read", "Grep", "Glob", "Bash"],
        "developer": ["Read", "Grep"],
    },
)

bridge = ClaudeCodeBridge(config)

# Spawn agents
bridge.spawn_agent(
    "reviewer",
    system_prompt="You are a code reviewer.",
    allowed_tools=["Read", "Grep", "Glob", "Bash"],
    budget_tool_calls=200,
)

bridge.spawn_agent(
    "developer",
    system_prompt="You are a developer.",
    allowed_tools=["Read", "Grep"],
    budget_tool_calls=50,
)

# Dispatch work and get scored interactions
interaction = bridge.dispatch_task("developer", "Write a function to sort a list")
print(f"p = {interaction.p:.3f}")  # P(v = +1)

# Check governance state
budget = bridge.policy.get_agent_budget("developer")
print(f"Tool calls used: {budget.tool_calls_used}/{budget.max_tool_calls}")

# Shut down
bridge.shutdown()
```

## Conceptual Mapping

| SWARM Concept | Claude Code Equivalent |
|---------------|----------------------|
| Agent type (honest/deceptive) | System prompt + tool permissions |
| Interaction / Step | `agent.ask(prompt)` or inbox task |
| Governance lever | Plan/permission approval policy |
| ProxyObservables | Tool usage, response quality, completion |
| Toxicity | Permission denial rate, tool misuse flags |
| Quality gap | Accepted vs rejected plan quality |

## Observable Extraction

Controller events map to SWARM observables:

| Controller Signal | SWARM Observable |
|-------------------|------------------|
| Response content length | `task_progress_delta` |
| Tool calls outside allowlist | `tool_misuse_flags` |
| Response token count | `counterparty_engagement_delta` |
| Plan rejection count | `verifier_rejections` |
| Rework requests | `rework_count` |

## Governance Integration

SWARM governance maps to Claude Code controls:

| SWARM Lever | Claude Code Action |
|-------------|-------------------|
| Circuit breaker | Freeze agent (deny all plans) |
| Transaction tax | Cost per tool invocation |
| Staking | Require stake for high-risk tools (Bash, Write) |
| Random audit | Flag tool usage for review |
| Bandwidth cap | Limit tool calls per epoch |
| Tool allowlist | Per-agent tool permission sets |

## Scenario Format

See `scenarios/claude_code_mvp.yaml` for a complete example:

```yaml
scenario_id: claude_code_mvp
agents:
  - type: honest
    count: 1
    name: senior_reviewer
    config:
      system_prompt: "You are a senior code reviewer..."
      allowed_tools: [Bash, Read, Write, Edit, Grep, Glob]
      budget_tool_calls: 200
```

## Metrics

The bridge produces standard SWARM metrics plus Claude Code-specific overlays:

- **Liquidity**: Remaining budget / total budget (approvals available)
- **Truth**: Downstream validation score / disagreement rate
- **Latency**: Task completion time
- **Toxicity**: E[1-p | accepted] over bridge interactions
- **Quality gap**: E[p | approved plans] - E[p | rejected plans]
- **Incoherence**: Variance in p across repeated prompts

## API Contract

### Endpoints

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/health` | — | `{status, agents_active, uptime_seconds}` |
| POST | `/agents/spawn` | `{agent_id, system_prompt, allowed_tools, model}` | `{agent_id, status}` |
| POST | `/agents/:id/ask` | `{prompt, timeout_seconds?}` | `{content, tool_calls, token_count, cost_usd}` |
| POST | `/agents/:id/shutdown` | — | `{status}` |
| POST | `/tasks` | `{subject, description, owner}` | `TaskEvent` |
| GET | `/tasks/:id/wait` | — | `TaskEvent` |
| GET | `/events?since=&limit=` | — | `{events: BridgeEvent[]}` |
| POST | `/governance/respond` | `{request_id, decision, reason?}` | `{status}` |

### Event Types

```
agent:spawned | agent:shutdown
message:sent  | message:received
plan:approval_request | plan:approved | plan:rejected
permission:request | permission:granted | permission:denied
task:created | task:assigned | task:completed | task:failed
tool:invoked | tool:result
error
```

## Status

**Production Ready** — Full integration with [claude-code-controller](https://github.com/The-Vibe-Company/claude-code-controller) web service.

### February 2026 Update

The bridge now fully integrates with the claude-code-controller web service:

- **Session management**: `init_session()`, `get_session_status()` for controller lifecycle
- **API prefix support**: Configurable `api_prefix` in `ClientConfig` (defaults to `/api`)
- **Agent spawning**: Full support for spawning agents with model selection (`sonnet`, `opus`, `haiku`)
- **Message dispatch**: `send()` method for async messaging, `ask()` for compatibility
- **Governance endpoints**: Agent-specific plan/permission approval routes

#### Usage Example

```python
from swarm.bridges.claude_code.bridge import ClaudeCodeBridge, BridgeConfig
from swarm.bridges.claude_code.client import ClientConfig

# Connect to running controller
config = BridgeConfig(
    client_config=ClientConfig(base_url="http://localhost:3100")
)
bridge = ClaudeCodeBridge(config)

# Session auto-initializes on first spawn
result = bridge.spawn_agent("researcher", model="sonnet")

# Dispatch tasks and get SWARM quality scores
interaction = bridge.dispatch_task("researcher", "Analyze this code for security issues")
print(f"Quality score (p): {interaction.p:.3f}")
```

#### Web Dashboard

Start the controller with web UI:

```bash
cd external/claude-code-controller/web
bun install
bun run build
PORT=3100 bun run start
```

Access at http://localhost:3100 for real-time agent monitoring.

#### Known Limitations

- **Permission approval UI**: The web UI's approval buttons may show "requestId is required" due to a format mismatch between Claude Code and the controller. Use the Python bridge for programmatic approval or auto-approve workflows.

### Research Integration

The bridge was used in the Rain vs River experiments ([clawxiv.2602.00040](https://clawxiv.org/abs/clawxiv.2602.00040)), demonstrating:

- Multi-agent orchestration with SWARM governance
- Quality scoring via ProxyComputer pipeline
- Welfare gap analysis between continuous (river) and discontinuous (rain) agents
