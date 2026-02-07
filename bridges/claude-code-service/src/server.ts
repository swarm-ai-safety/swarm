/**
 * SWARM Claude Code Controller Service
 *
 * HTTP/WebSocket server that wraps ClaudeCodeController and exposes
 * endpoints for the Python SWARM bridge. This service owns the Claude
 * Code CLI agent processes and streams governance-relevant events back
 * to SWARM.
 *
 * Endpoints:
 *   GET  /health                  → service health check
 *   POST /agents/spawn            → spawn a new Claude Code agent
 *   POST /agents/:id/ask          → single-turn prompt
 *   POST /agents/:id/shutdown     → shut down an agent
 *   POST /tasks                   → create inbox task
 *   GET  /tasks/:id/wait          → poll for task completion
 *   GET  /events                  → poll recent events
 *   POST /governance/respond      → respond to plan/permission requests
 *
 * In production, this wraps the real ClaudeCodeController from
 * claude-code-controller. For development/testing, it runs with
 * a mock controller that simulates agent behavior.
 */

import express, { Request, Response, NextFunction } from "express";
import { v4 as uuidv4 } from "uuid";
import type {
  SpawnRequest,
  SpawnResponse,
  AskRequest,
  AskResponse,
  CreateTaskRequest,
  TaskResponse,
  BridgeEvent,
  GovernanceResponse,
  HealthResponse,
} from "./types";

const app = express();
app.use(express.json({ limit: "100kb" }));

// --- Security: Bearer token authentication ---

const API_KEY = process.env.SWARM_BRIDGE_API_KEY || "";

function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  // Health endpoint is unauthenticated for liveness probes
  if (req.path === "/health") {
    next();
    return;
  }
  if (!API_KEY) {
    // If no API key is configured, allow all requests (dev mode)
    next();
    return;
  }
  const authHeader = req.headers.authorization;
  if (!authHeader || authHeader !== `Bearer ${API_KEY}`) {
    res.status(401).json({ error: "Unauthorized: invalid or missing Bearer token" });
    return;
  }
  next();
}

app.use(authMiddleware);

// --- Security: Input validation helpers ---

const AGENT_ID_RE = /^[a-zA-Z0-9_-]{1,128}$/;
const MAX_AGENTS = 1000;
const MAX_TASKS = 10000;
const MAX_PROMPT_LENGTH = 100000;
const MAX_EVENTS_LIMIT = 1000;

function isValidAgentId(id: string): boolean {
  return AGENT_ID_RE.test(id);
}

// --- In-memory state ---

interface AgentState {
  agent_id: string;
  system_prompt: string;
  allowed_tools: string[];
  model: string;
  spawned_at: Date;
}

interface TaskState {
  task_id: string;
  subject: string;
  description: string;
  owner: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  result?: string;
  created_at: Date;
  completed_at?: Date;
  tools_used: string[];
}

const agents = new Map<string, AgentState>();
const tasks = new Map<string, TaskState>();
const events: BridgeEvent[] = [];
const startTime = Date.now();

function emitEvent(
  type: BridgeEvent["event_type"],
  agentId: string,
  payload: Record<string, unknown> = {}
): BridgeEvent {
  const event: BridgeEvent = {
    event_id: uuidv4(),
    event_type: type,
    timestamp: new Date().toISOString(),
    agent_id: agentId,
    payload,
  };
  events.push(event);
  // Keep bounded event buffer
  if (events.length > 10000) {
    events.splice(0, events.length - 10000);
  }
  return event;
}

// --- Health ---

app.get("/health", (_req: Request, res: Response) => {
  const resp: HealthResponse = {
    status: "ok",
    agents_active: agents.size,
    uptime_seconds: Math.floor((Date.now() - startTime) / 1000),
  };
  res.json(resp);
});

// --- Agent lifecycle ---

app.post("/agents/spawn", (req: Request, res: Response) => {
  const body = req.body as SpawnRequest;
  const agentId = body.agent_id || uuidv4();

  if (!isValidAgentId(agentId)) {
    res.status(400).json({ error: "Invalid agent_id: must match [a-zA-Z0-9_-]{1,128}" });
    return;
  }

  if (agents.size >= MAX_AGENTS) {
    res.status(429).json({ error: `Agent limit reached (max ${MAX_AGENTS})` });
    return;
  }

  if (agents.has(agentId)) {
    res.status(409).json({ status: "error", message: "Agent already exists" });
    return;
  }

  const agent: AgentState = {
    agent_id: agentId,
    system_prompt: body.system_prompt || "",
    allowed_tools: body.allowed_tools || [],
    model: body.model || "claude-sonnet-4-20250514",
    spawned_at: new Date(),
  };
  agents.set(agentId, agent);

  emitEvent("agent:spawned", agentId, {
    model: agent.model,
    tools: agent.allowed_tools,
  });

  // TODO: In production, this would call ClaudeCodeController.spawn()
  // For now, we register the agent in our state and return.

  const resp: SpawnResponse = { agent_id: agentId, status: "spawned" };
  res.status(201).json(resp);
});

app.post("/agents/:id/shutdown", (req: Request, res: Response) => {
  const agentId = req.params.id;
  if (!isValidAgentId(agentId)) {
    res.status(400).json({ error: "Invalid agent_id format" });
    return;
  }
  if (!agents.has(agentId)) {
    res.status(404).json({ error: "Agent not found" });
    return;
  }

  agents.delete(agentId);
  emitEvent("agent:shutdown", agentId);

  // TODO: In production, call ClaudeCodeController.shutdown(agentId)

  res.json({ status: "shutdown", agent_id: agentId });
});

// --- Ask (single-turn) ---

app.post("/agents/:id/ask", (req: Request, res: Response) => {
  const agentId = req.params.id;
  const body = req.body as AskRequest;

  if (!isValidAgentId(agentId)) {
    res.status(400).json({ error: "Invalid agent_id format" });
    return;
  }
  if (!agents.has(agentId)) {
    res.status(404).json({ error: "Agent not found" });
    return;
  }
  if (!body.prompt || typeof body.prompt !== "string") {
    res.status(400).json({ error: "Missing or invalid 'prompt' field" });
    return;
  }
  if (body.prompt.length > MAX_PROMPT_LENGTH) {
    res.status(400).json({ error: `Prompt too long (max ${MAX_PROMPT_LENGTH} chars)` });
    return;
  }

  // Log event with truncated prompt to avoid storing sensitive data in full
  emitEvent("message:sent", agentId, { prompt_length: body.prompt.length });

  // TODO: In production, forward to ClaudeCodeController.ask()
  // For now, return a mock response.

  const resp: AskResponse = {
    content: `[Mock response from agent] Acknowledged prompt (${body.prompt.length} chars)`,
    tool_calls: [],
    token_count: Math.floor(body.prompt.length * 1.5),
    cost_usd: 0.001,
  };

  emitEvent("message:received", agentId, {
    content_length: resp.content.length,
    token_count: resp.token_count,
  });

  res.json(resp);
});

// --- Tasks (inbox protocol) ---

app.post("/tasks", (req: Request, res: Response) => {
  const body = req.body as CreateTaskRequest;

  if (!body.subject || !body.owner) {
    res.status(400).json({ error: "Missing required fields: 'subject' and 'owner'" });
    return;
  }
  if (tasks.size >= MAX_TASKS) {
    res.status(429).json({ error: `Task limit reached (max ${MAX_TASKS})` });
    return;
  }

  const taskId = uuidv4();

  const task: TaskState = {
    task_id: taskId,
    subject: body.subject,
    description: body.description,
    owner: body.owner,
    status: "pending",
    created_at: new Date(),
    tools_used: [],
  };
  tasks.set(taskId, task);

  emitEvent("task:created", body.owner, {
    task_id: taskId,
    subject: body.subject,
  });

  // TODO: In production, write to the controller's inbox
  // and start monitoring for completion.

  // Simulate task starting immediately
  task.status = "in_progress";
  emitEvent("task:assigned", body.owner, { task_id: taskId });

  // For mock: auto-complete after a brief delay
  setTimeout(() => {
    const t = tasks.get(taskId);
    if (t && t.status === "in_progress") {
      t.status = "completed";
      t.result = `Completed: ${t.subject}`;
      t.completed_at = new Date();
      emitEvent("task:completed", t.owner, {
        task_id: taskId,
        result: t.result,
      });
    }
  }, 2000);

  const resp: TaskResponse = {
    task_id: taskId,
    agent_id: body.owner,
    subject: body.subject,
    description: body.description,
    owner: body.owner,
    status: task.status,
    duration_ms: 0,
    tools_used: [],
    timestamp: task.created_at.toISOString(),
  };
  res.status(201).json(resp);
});

app.get("/tasks/:id/wait", (req: Request, res: Response) => {
  const taskId = req.params.id;
  const task = tasks.get(taskId);

  if (!task) {
    res.status(404).json({ error: "Task not found" });
    return;
  }

  const durationMs = task.completed_at
    ? task.completed_at.getTime() - task.created_at.getTime()
    : Date.now() - task.created_at.getTime();

  const resp: TaskResponse = {
    task_id: task.task_id,
    agent_id: task.owner,
    subject: task.subject,
    description: task.description,
    owner: task.owner,
    status: task.status,
    result: task.result,
    duration_ms: durationMs,
    tools_used: task.tools_used,
    timestamp: task.created_at.toISOString(),
  };
  res.json(resp);
});

// --- Events ---

app.get("/events", (req: Request, res: Response) => {
  const since = req.query.since as string | undefined;
  const rawLimit = parseInt((req.query.limit as string) || "100", 10);
  const limit = Math.min(Math.max(1, isNaN(rawLimit) ? 100 : rawLimit), MAX_EVENTS_LIMIT);

  let filtered = events;
  if (since) {
    const idx = events.findIndex((e) => e.event_id === since);
    if (idx >= 0) {
      filtered = events.slice(idx + 1);
    }
  }

  res.json({ events: filtered.slice(-limit) });
});

// --- Governance ---

app.post("/governance/respond", (req: Request, res: Response) => {
  const body = req.body as GovernanceResponse;

  if (!body.request_id || !body.decision) {
    res.status(400).json({ error: "Missing required fields: 'request_id' and 'decision'" });
    return;
  }
  const validDecisions = ["approve", "reject", "grant", "deny"];
  if (!validDecisions.includes(body.decision)) {
    res.status(400).json({ error: `Invalid decision: must be one of ${validDecisions.join(", ")}` });
    return;
  }

  // TODO: In production, forward to ClaudeCodeController's
  // plan/permission response handler.

  const isApproval = body.decision === "approve" || body.decision === "grant";
  const eventType = isApproval ? "plan:approved" : "plan:rejected";

  emitEvent(eventType, "", {
    request_id: body.request_id,
    decision: body.decision,
    reason: body.reason,
  });

  res.json({ status: "acknowledged", request_id: body.request_id });
});

// --- Start ---

const PORT = parseInt(process.env.PORT || "3100", 10);
const HOST = process.env.HOST || "127.0.0.1";

app.listen(PORT, HOST, () => {
  console.log(`SWARM Claude Code service listening on ${HOST}:${PORT}`);
  if (!API_KEY) {
    console.warn("WARNING: No SWARM_BRIDGE_API_KEY set - running without authentication");
  }
});

export default app;
