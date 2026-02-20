/** Configuration and message types for the client-side SWARM simulation engine. */

import type { AgentType } from "@/data/types";

/** Configuration for a single agent group */
export interface AgentGroupConfig {
  type: AgentType;
  count: number;
}

/** Governance lever configuration */
export interface GovernanceConfig {
  /** Transaction tax rate [0, 0.5] */
  taxRate: number;
  /** Reputation decay factor per epoch [0.8, 1.0] */
  reputationDecay: number;
  /** Whether circuit breaker is enabled */
  circuitBreakerEnabled: boolean;
  /** Toxicity threshold to trip circuit breaker */
  circuitBreakerThreshold: number;
}

/** Payoff equation parameters */
export interface PayoffParams {
  s_plus: number;
  s_minus: number;
  h: number;
  theta: number;
  rho_a: number;
  rho_b: number;
  w_rep: number;
}

/** Full scenario configuration sent to the worker */
export interface ScenarioConfig {
  agents: AgentGroupConfig[];
  governance: GovernanceConfig;
  payoff: PayoffParams;
  epochs: number;
  stepsPerEpoch: number;
  seed: number;
  sigmoidK: number;
}

/** Runtime state for a simulated agent */
export interface SimAgentState {
  id: string;
  name: string;
  type: AgentType;
  reputation: number;
  resources: number;
  totalPayoff: number;
  interactionsInitiated: number;
  interactionsReceived: number;
  sumPInitiated: number;
  sumPReceived: number;
  isFrozen: boolean;
  isQuarantined: boolean;
}

/** Messages from main thread to worker */
export type WorkerRequest =
  | { type: "run"; config: ScenarioConfig }
  | { type: "cancel" };

/** Messages from worker to main thread */
export type WorkerResponse =
  | { type: "progress"; epoch: number; totalEpochs: number }
  | { type: "complete"; data: import("@/data/types").SimulationData }
  | { type: "error"; message: string };

/** Default scenario configuration */
export const DEFAULT_CONFIG: ScenarioConfig = {
  agents: [
    { type: "honest", count: 3 },
    { type: "opportunistic", count: 1 },
    { type: "deceptive", count: 1 },
    { type: "adversarial", count: 0 },
  ],
  governance: {
    taxRate: 0.05,
    reputationDecay: 0.95,
    circuitBreakerEnabled: false,
    circuitBreakerThreshold: 0.4,
  },
  payoff: {
    s_plus: 2.0,
    s_minus: 1.0,
    h: 2.0,
    theta: 0.5,
    rho_a: 0.0,
    rho_b: 0.0,
    w_rep: 1.0,
  },
  epochs: 10,
  stepsPerEpoch: 10,
  seed: 42,
  sigmoidK: 2.0,
};
