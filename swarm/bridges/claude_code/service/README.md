# SWARM Claude Code Service

HTTP service that wraps `ClaudeCodeController` for the SWARM Python bridge.

## Quick Start

```bash
npm install
npm run build
npm start
```

Development mode:
```bash
npm run dev
```

The service starts on port 3100 by default. Set `PORT` env var to change.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/agents/spawn` | Spawn a Claude Code agent |
| POST | `/agents/:id/ask` | Single-turn prompt |
| POST | `/agents/:id/shutdown` | Shut down an agent |
| POST | `/tasks` | Create inbox task |
| GET | `/tasks/:id/wait` | Poll for task completion |
| GET | `/events` | Poll recent events |
| POST | `/governance/respond` | Respond to plan/permission requests |

## Integration

The Python bridge (`swarm.bridges.claude_code`) communicates with this service.
See `swarm/bridges/claude_code/client.py` for the client implementation.
