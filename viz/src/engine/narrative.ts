/** Detect key narrative moments in simulation data for auto-annotation. */

import type { EpochSnapshot, AgentSnapshot, SimulationData } from "@/data/types";

export interface NarrativeEvent {
  epoch: number;
  type: "danger" | "warning" | "info" | "success";
  message: string;
}

/** Detect all narrative events across an entire simulation */
export function detectNarrativeEvents(data: SimulationData): NarrativeEvent[] {
  const events: NarrativeEvent[] = [];
  const { epoch_snapshots: epochs, agent_snapshots: agents } = data;

  for (let i = 0; i < epochs.length; i++) {
    const e = epochs[i];
    const prev = i > 0 ? epochs[i - 1] : null;

    // Toxicity spike
    if (prev && e.toxicity_rate > prev.toxicity_rate + 0.15 && e.toxicity_rate > 0.2) {
      events.push({
        epoch: e.epoch,
        type: "danger",
        message: `Toxicity spike: ${fmt(prev.toxicity_rate)} \u2192 ${fmt(e.toxicity_rate)}`,
      });
    }

    // Toxicity crossed critical threshold
    if (prev && prev.toxicity_rate <= 0.4 && e.toxicity_rate > 0.4) {
      events.push({
        epoch: e.epoch,
        type: "danger",
        message: "Toxicity crossed critical threshold (40%)",
      });
    }

    // Circuit breaker tripped (agents frozen)
    if (prev && e.n_frozen > prev.n_frozen) {
      const newFrozen = e.n_frozen - prev.n_frozen;
      events.push({
        epoch: e.epoch,
        type: "warning",
        message: `Circuit breaker: ${newFrozen} agent${newFrozen > 1 ? "s" : ""} frozen`,
      });
    }

    // Agent quarantined
    if (prev && e.n_quarantined > prev.n_quarantined) {
      const newQ = e.n_quarantined - prev.n_quarantined;
      events.push({
        epoch: e.epoch,
        type: "warning",
        message: `${newQ} agent${newQ > 1 ? "s" : ""} quarantined`,
      });
    }

    // Adverse selection detected (quality gap goes negative)
    if (prev && prev.quality_gap >= 0 && e.quality_gap < -0.05) {
      events.push({
        epoch: e.epoch,
        type: "danger",
        message: `Adverse selection detected (quality gap: ${e.quality_gap.toFixed(3)})`,
      });
    }

    // Gini inequality spike
    if (prev && e.gini_coefficient > prev.gini_coefficient + 0.1 && e.gini_coefficient > 0.3) {
      events.push({
        epoch: e.epoch,
        type: "warning",
        message: `Inequality rising: Gini ${prev.gini_coefficient.toFixed(2)} \u2192 ${e.gini_coefficient.toFixed(2)}`,
      });
    }

    // Welfare collapse
    if (prev && prev.total_welfare > 0 && e.total_welfare < 0) {
      events.push({
        epoch: e.epoch,
        type: "danger",
        message: "Welfare collapsed below zero",
      });
    }

    // Recovery: toxicity dropped significantly
    if (prev && prev.toxicity_rate > 0.3 && e.toxicity_rate < prev.toxicity_rate - 0.1) {
      events.push({
        epoch: e.epoch,
        type: "success",
        message: `Recovery: toxicity ${fmt(prev.toxicity_rate)} \u2192 ${fmt(e.toxicity_rate)}`,
      });
    }

    // All agents accepting (high quality epoch)
    if (e.accepted_interactions === e.total_interactions && e.total_interactions > 0 && e.avg_p > 0.7) {
      events.push({
        epoch: e.epoch,
        type: "success",
        message: `Perfect cooperation: all ${e.total_interactions} interactions accepted`,
      });
    }

    // Mass rejection
    if (e.total_interactions > 0 && e.rejected_interactions / e.total_interactions > 0.7) {
      events.push({
        epoch: e.epoch,
        type: "warning",
        message: `Trust breakdown: ${Math.round(e.rejected_interactions / e.total_interactions * 100)}% rejected`,
      });
    }

    // Check agent-level events
    const epochAgents = agents.filter((a) => a.epoch === e.epoch);
    const prevEpochAgents = i > 0 ? agents.filter((a) => a.epoch === epochs[i - 1].epoch) : [];

    // Agent reputation collapse (any agent drops below 0.1 from above 0.3)
    for (const agent of epochAgents) {
      const prevAgent = prevEpochAgents.find((a) => a.agent_id === agent.agent_id);
      if (prevAgent && prevAgent.reputation > 0.3 && agent.reputation < 0.1) {
        events.push({
          epoch: e.epoch,
          type: "info",
          message: `${agent.name || agent.agent_id} reputation collapsed`,
        });
      }
    }
  }

  return events;
}

function fmt(v: number): string {
  return (v * 100).toFixed(0) + "%";
}
