/** Mirrors swarm/models/agent.py AgentType enum */
export type AgentType =
  | "honest"
  | "opportunistic"
  | "deceptive"
  | "adversarial"
  | "rlm"
  | "crewai";

/** Mirrors swarm/models/interaction.py InteractionType enum */
export type InteractionType = "reply" | "vote" | "trade" | "collaboration";

/** Mirrors swarm/analysis/aggregation.py AgentSnapshot */
export interface AgentSnapshot {
  agent_id: string;
  epoch: number;
  name?: string;
  agent_type?: AgentType;
  reputation: number;
  resources: number;
  interactions_initiated: number;
  interactions_received: number;
  avg_p_initiated: number;
  avg_p_received: number;
  total_payoff: number;
  is_frozen: boolean;
  is_quarantined: boolean;
}

/** Mirrors swarm/analysis/aggregation.py EpochSnapshot */
export interface EpochSnapshot {
  simulation_id?: string;
  epoch: number;
  timestamp?: string;
  total_interactions: number;
  accepted_interactions: number;
  rejected_interactions: number;
  toxicity_rate: number;
  quality_gap: number;
  avg_p: number;
  incoherence_index?: number;
  total_welfare: number;
  avg_payoff: number;
  payoff_std: number;
  gini_coefficient: number;
  total_posts: number;
  total_votes: number;
  total_tasks_completed: number;
  n_agents: number;
  n_frozen: number;
  n_quarantined: number;
  avg_reputation: number;
  reputation_std: number;
  n_edges?: number;
  avg_degree?: number;
  avg_clustering?: number;
  n_components?: number;
  ecosystem_threat_level?: number;
  active_threats?: number;
  contagion_depth?: number;
  ecosystem_collusion_risk?: number;
  n_flagged_pairs?: number;
  avg_coordination_score?: number;
  avg_synergy_score?: number;
  tasks_completed?: number;
}

/** Interaction event from events.jsonl (derived from raw SwarmEvents) */
export interface InteractionEvent {
  event_type: string;
  timestamp: string;
  epoch: number;
  step: number;
  interaction_id: string;
  initiator: string;
  counterparty: string;
  interaction_type: InteractionType;
  accepted: boolean;
  p: number;
  v_hat: number;
}

/** All event types emitted by SWARM simulation */
export type SwarmEventType =
  | "simulation_started"
  | "simulation_ended"
  | "epoch_completed"
  | "agent_created"
  | "contract_signing"
  | "interaction_proposed"
  | "interaction_completed"
  | "governance_cost_applied"
  | "reputation_updated"
  | "payoff_computed"
  | "contract_metrics";

/** Raw event from events.jsonl — superset of all event types */
export interface SwarmEvent {
  event_id: string;
  timestamp: string;
  event_type: SwarmEventType;
  interaction_id: string | null;
  agent_id: string | null;
  initiator_id: string | null;
  counterparty_id: string | null;
  epoch: number | null;
  step: number | null;
  payload: Record<string, unknown>;
}

/** Mirrors swarm/analysis/export.py JSON structure */
export interface SimulationData {
  simulation_id: string;
  started_at: string | null;
  ended_at: string | null;
  n_epochs: number;
  steps_per_epoch: number;
  n_agents: number;
  seed: number | null;
  epoch_snapshots: EpochSnapshot[];
  agent_snapshots: AgentSnapshot[];
  events?: InteractionEvent[];
  rawEvents?: SwarmEvent[];
}
