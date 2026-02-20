/** Metrics computation — port of swarm/metrics/soft_metrics.py (subset). */

import type { EpochSnapshot, AgentSnapshot } from "@/data/types";
import type { SimAgentState } from "./types";

export interface StepResult {
  p: number;
  accepted: boolean;
  initiatorPayoff: number;
  counterpartyPayoff: number;
}

/** Compute Gini coefficient from an array of values */
export function gini(values: number[]): number {
  const n = values.length;
  if (n === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  let numerator = 0;
  for (let i = 0; i < n; i++) {
    numerator += (2 * (i + 1) - n - 1) * sorted[i];
  }
  const mean = sorted.reduce((s, v) => s + v, 0) / n;
  if (mean === 0) return 0;
  return numerator / (n * n * mean);
}

/** Build an EpochSnapshot from step results and agent states */
export function buildEpochSnapshot(
  epoch: number,
  steps: StepResult[],
  agents: SimAgentState[],
  simId: string,
): EpochSnapshot {
  const accepted = steps.filter((s) => s.accepted);
  const rejected = steps.filter((s) => !s.accepted);

  const toxicityRate =
    accepted.length > 0
      ? accepted.reduce((s, r) => s + (1 - r.p), 0) / accepted.length
      : 0;

  const avgPAccepted = accepted.length > 0 ? accepted.reduce((s, r) => s + r.p, 0) / accepted.length : 0;
  const avgPRejected = rejected.length > 0 ? rejected.reduce((s, r) => s + r.p, 0) / rejected.length : 0;
  const qualityGap = accepted.length > 0 && rejected.length > 0 ? avgPAccepted - avgPRejected : 0;

  const avgP = steps.length > 0 ? steps.reduce((s, r) => s + r.p, 0) / steps.length : 0.5;

  const payoffs = agents.map((a) => a.totalPayoff);
  const totalWelfare = payoffs.reduce((s, v) => s + v, 0);
  const avgPayoff = agents.length > 0 ? totalWelfare / agents.length : 0;
  const payoffStd = agents.length > 0
    ? Math.sqrt(payoffs.reduce((s, v) => s + (v - avgPayoff) ** 2, 0) / agents.length)
    : 0;

  const reps = agents.map((a) => a.reputation);
  const avgRep = reps.length > 0 ? reps.reduce((s, v) => s + v, 0) / reps.length : 0.5;
  const repStd = reps.length > 0
    ? Math.sqrt(reps.reduce((s, v) => s + (v - avgRep) ** 2, 0) / reps.length)
    : 0;

  return {
    simulation_id: simId,
    epoch,
    timestamp: new Date().toISOString(),
    total_interactions: steps.length,
    accepted_interactions: accepted.length,
    rejected_interactions: rejected.length,
    toxicity_rate: toxicityRate,
    quality_gap: qualityGap,
    avg_p: avgP,
    total_welfare: totalWelfare,
    avg_payoff: avgPayoff,
    payoff_std: payoffStd,
    gini_coefficient: gini(payoffs.map((v) => Math.max(0, v))),
    total_posts: accepted.length,
    total_votes: Math.floor(steps.length * 0.3),
    total_tasks_completed: accepted.length,
    n_agents: agents.length,
    n_frozen: agents.filter((a) => a.isFrozen).length,
    n_quarantined: agents.filter((a) => a.isQuarantined).length,
    avg_reputation: avgRep,
    reputation_std: repStd,
  };
}

/** Convert SimAgentState to AgentSnapshot for a given epoch */
export function buildAgentSnapshot(agent: SimAgentState, epoch: number): AgentSnapshot {
  const avgPInit = agent.interactionsInitiated > 0
    ? agent.sumPInitiated / agent.interactionsInitiated
    : 0.5;
  const avgPRecv = agent.interactionsReceived > 0
    ? agent.sumPReceived / agent.interactionsReceived
    : 0.5;

  return {
    agent_id: agent.id,
    epoch,
    name: agent.name,
    agent_type: agent.type,
    reputation: agent.reputation,
    resources: agent.resources,
    interactions_initiated: agent.interactionsInitiated,
    interactions_received: agent.interactionsReceived,
    avg_p_initiated: avgPInit,
    avg_p_received: avgPRecv,
    total_payoff: agent.totalPayoff,
    is_frozen: agent.isFrozen,
    is_quarantined: agent.isQuarantined,
  };
}
