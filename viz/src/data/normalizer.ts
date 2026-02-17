import type { SimulationData, AgentSnapshot, EpochSnapshot } from "./types";

export interface NormalizationRanges {
  reputation: { min: number; max: number };
  resources: { min: number; max: number };
  totalPayoff: { min: number; max: number };
  toxicity: { min: number; max: number };
  avgP: { min: number; max: number };
  gini: { min: number; max: number };
  threatLevel: { min: number; max: number };
}

/** Compute global min/max ranges across all epochs for consistent visual scaling */
export function computeRanges(data: SimulationData): NormalizationRanges {
  const ranges: NormalizationRanges = {
    reputation: { min: 0, max: 1 },
    resources: { min: 0, max: 100 },
    totalPayoff: { min: 0, max: 1 },
    toxicity: { min: 0, max: 1 },
    avgP: { min: 0, max: 1 },
    gini: { min: 0, max: 1 },
    threatLevel: { min: 0, max: 1 },
  };

  // Agent ranges
  for (const snap of data.agent_snapshots) {
    ranges.reputation.min = Math.min(ranges.reputation.min, snap.reputation);
    ranges.reputation.max = Math.max(ranges.reputation.max, snap.reputation);
    ranges.resources.min = Math.min(ranges.resources.min, snap.resources);
    ranges.resources.max = Math.max(ranges.resources.max, snap.resources);
    ranges.totalPayoff.min = Math.min(ranges.totalPayoff.min, snap.total_payoff);
    ranges.totalPayoff.max = Math.max(ranges.totalPayoff.max, snap.total_payoff);
  }

  // Epoch ranges
  for (const snap of data.epoch_snapshots) {
    ranges.toxicity.max = Math.max(ranges.toxicity.max, snap.toxicity_rate);
    ranges.gini.max = Math.max(ranges.gini.max, snap.gini_coefficient);
    ranges.threatLevel.max = Math.max(ranges.threatLevel.max, snap.ecosystem_threat_level ?? 0);
  }

  return ranges;
}

/** Group agent snapshots by epoch */
export function groupAgentsByEpoch(
  snapshots: AgentSnapshot[],
): Map<number, Map<string, AgentSnapshot>> {
  const grouped = new Map<number, Map<string, AgentSnapshot>>();
  for (const snap of snapshots) {
    if (!grouped.has(snap.epoch)) {
      grouped.set(snap.epoch, new Map());
    }
    grouped.get(snap.epoch)!.set(snap.agent_id, snap);
  }
  return grouped;
}

/** Get unique agent IDs from snapshots */
export function getUniqueAgentIds(snapshots: AgentSnapshot[]): string[] {
  return [...new Set(snapshots.map((s) => s.agent_id))];
}
