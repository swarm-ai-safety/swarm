/**
 * Shared type definitions for the SWARM-Claude Code bridge protocol.
 *
 * These types match the Python event schemas in
 * swarm/bridges/claude_code/events.py.
 */

// --- Agent types ---

export interface SpawnRequest {
  agent_id: string;
  system_prompt?: string;
  allowed_tools?: string[];
  model?: string;
}

export interface SpawnResponse {
  agent_id: string;
  status: "spawned" | "error";
  message?: string;
}

// --- Ask (single-turn) ---

export interface AskRequest {
  prompt: string;
  timeout_seconds?: number;
}

export interface AskResponse {
  content: string;
  tool_calls: ToolCall[];
  token_count: number;
  cost_usd: number;
}

export interface ToolCall {
  tool: string;
  args: Record<string, unknown>;
  result?: string;
}

// --- Task (inbox protocol) ---

export interface CreateTaskRequest {
  subject: string;
  description: string;
  owner: string;
}

export interface TaskResponse {
  task_id: string;
  agent_id: string;
  subject: string;
  description: string;
  owner: string;
  status: "pending" | "in_progress" | "completed" | "failed";
  result?: string;
  duration_ms: number;
  tools_used: string[];
  timestamp: string;
}

// --- Events ---

export type BridgeEventType =
  | "agent:spawned"
  | "agent:shutdown"
  | "message:sent"
  | "message:received"
  | "plan:approval_request"
  | "plan:approved"
  | "plan:rejected"
  | "permission:request"
  | "permission:granted"
  | "permission:denied"
  | "task:created"
  | "task:assigned"
  | "task:completed"
  | "task:failed"
  | "tool:invoked"
  | "tool:result"
  | "error";

export interface BridgeEvent {
  event_id: string;
  event_type: BridgeEventType;
  timestamp: string;
  agent_id: string;
  payload: Record<string, unknown>;
}

// --- Governance ---

export interface GovernanceResponse {
  request_id: string;
  decision: "approve" | "reject" | "grant" | "deny";
  reason?: string;
}

// --- Health ---

export interface HealthResponse {
  status: "ok" | "degraded";
  agents_active: number;
  uptime_seconds: number;
}
