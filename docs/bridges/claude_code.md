# SWARM-Claude Code Bridge

Govern and score Claude Code CLI agents with SWARM's distributional safety framework.

## Overview

The Claude Code bridge connects SWARM to [claude-code-controller](https://github.com/anthropics/claude-code), enabling programmatic orchestration of Claude Code CLI agents under SWARM governance. Each Claude Code agent becomes a SWARM agent with:

- **Plan approval** gated by governance policy
- **Tool permissions** enforced via allowlists and budgets
- **Interaction scoring** through SWARM's ProxyComputer pipeline
- **Circuit breakers** that freeze misbehaving agents

This bridge is designed for:
- Governance experiments on tool-using agents.
- Long-horizon inbox tasks (task creation + wait).
- Safety telemetry from tool usage and plan approval events.

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

### Requirements
- Node 18+ for the controller service.
- Python 3.10+ for SWARM.
- Claude Code installed and configured for the controller.
- `SWARM_BRIDGE_API_KEY` set for the controller (recommended).

### Security defaults

- Always set `SWARM_BRIDGE_API_KEY` before starting the service.
- Keep `HOST` on loopback (e.g. `127.0.0.1`) unless you have a secured reverse proxy in front.
- Auto-approval is off by default and only allowed for loopback controller URLs.

### Controller setup script (repeatable)
```bash
#!/usr/bin/env bash
set -euo pipefail

export HOST=127.0.0.1
export PORT=3100
export SWARM_BRIDGE_API_KEY="replace_me"

cd bridges/claude-code-service
npm install
npm start
```

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

### 3. Run the minimal demo (end-to-end)
```bash
python scripts/run_claude_code_scenario.py \
  --scenario scenarios/claude_code_demo.yaml \
  --base-url http://localhost:3100 \
  --api-prefix /api \
  --auto-approve
```

Shortcut wrapper:
```bash
bash scripts/run_claude_code_demo.sh
```

## Configuration

### ClientConfig
Key fields:
- `base_url`: controller URL (default `http://localhost:3100`).
- `api_prefix`: prefix for endpoints (default `/api`).
- `timeout_seconds`, `max_retries`, `retry_backoff_base`.
- `api_key`: bearer token, if configured in the service.

### BridgeConfig
Key fields:
- `governance_config`: SWARM governance controls.
- `auto_respond_governance`: auto-approve/deny plan and permission requests.
- `tool_allowlist`: per-agent tool permissions.
- `proxy_sigmoid_k`: calibration for ProxyComputer.
- `event_poll_interval`: event polling rate.

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

## Event Flow
1. `spawn_agent()` requests a controller agent.
2. Controller emits `agent:spawned`.
3. `dispatch_task()` sends a message or creates an inbox task.
4. Controller emits message/task events.
5. Bridge converts events to `SoftInteraction` and logs metrics.
6. If plan/permission requests appear, the governance policy responds.

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

Minimal demo: `scenarios/claude_code_demo.yaml`  
Full example: `scenarios/claude_code_mvp.yaml`

Note: These scenarios assume a Claude Code controller is running and a bridge-aware runner is used to spawn Claude Code agents from the agent config block. Use `scripts/run_claude_code_scenario.py` for the demo.

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

All endpoints are prefixed with `/api` by default (configurable via `ClientConfig.api_prefix`).

| Method | Path | Body | Response |
|--------|------|------|----------|
| GET | `/session/status` | — | `{initialized, agents_active, uptime_seconds}` |
| POST | `/session/init` | `{teamName, cwd?}` | `{initialized, ...}` |
| POST | `/agents/spawn` | `{name, model, type}` | `{agent_id, status}` |
| POST | `/agents/:id/send` | `{message, summary?}` | `{status}` |
| POST | `/agents/:id/shutdown` | — | `{status}` |
| POST | `/agents/:id/kill` | — | `{status}` |
| POST | `/agents/:id/approve-plan` | `{requestId, approve, feedback?}` | `{status}` |
| POST | `/agents/:id/approve-permission` | `{requestId, approve}` | `{status}` |
| POST | `/tasks` | `{subject, description, owner}` | `TaskEvent` |
| GET | `/tasks/:id/wait` | — | `TaskEvent` |
| GET | `/events?since=&limit=` | — | `{events: BridgeEvent[]}` |
| POST | `/governance/respond` | `{requestId, decision, reason?}` | `{status}` |

Notes:
- The Python bridge uses `/agents/:id/send` and treats `ask()` as fire-and-forget.
- Plan/permission approvals should be sent to the per-agent endpoints; `/governance/respond` is a fallback.

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

## Troubleshooting
- **401 Unauthorized**: Ensure `SWARM_BRIDGE_API_KEY` matches the bearer token used by the Python bridge `ClientConfig.api_key`.
- **404 on /events or /agents/**: Verify `ClientConfig.api_prefix` (default `/api`) matches the controller.
- **No responses after `ask()`**: The bridge uses `/send` as fire-and-forget; poll `/events` for responses.
- **Approval buttons fail in UI**: Use the Python bridge `respond_to_plan()` or enable `auto_respond_governance`.
- **Agent spawn succeeds but no output**: Confirm the controller session is initialized (`/session/status`) and agents are active.
- **Auto-approval disabled unexpectedly**: Auto-approval only works for loopback controller URLs; use `http://localhost:3100` or `http://127.0.0.1:3100`.

## Validation Against Python Bridge (2026-02-09)
Validated the API contract against `swarm/bridges/claude_code/client.py` and `swarm/bridges/claude_code/bridge.py`.

Verified:
- API prefix defaults to `/api` (`ClientConfig.api_prefix`).
- Session lifecycle endpoints: `/session/init`, `/session/status`.
- Agent lifecycle endpoints: `/agents/spawn`, `/agents/:id/shutdown`, `/agents/:id/kill`.
- Messaging endpoint: `/agents/:id/send` (not `/ask`).
- Inbox endpoints: `/tasks`, `/tasks/:id/wait`.
- Events polling: `/events`.
- Governance responses: per-agent `/agents/:id/approve-plan` and `/agents/:id/approve-permission`.

Notes on compatibility:
- `ask()` in Python is a wrapper over `/send` and returns a placeholder event. Actual responses arrive via `/events`.
- `system_prompt` and `allowed_tools` are accepted by the Python bridge but marked as unused by the web controller.
- `/governance/respond` is kept as a fallback but may not exist in all controller deployments.

### Research Integration

The bridge was used in the Rain vs River experiments ([clawxiv.2602.00040](https://clawxiv.org/abs/clawxiv.2602.00040)), demonstrating:

- Multi-agent orchestration with SWARM governance
- Quality scoring via ProxyComputer pipeline
- Welfare gap analysis between continuous (river) and discontinuous (rain) agents
