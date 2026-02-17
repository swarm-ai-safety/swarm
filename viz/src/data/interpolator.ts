import type { EpochSnapshot } from "./types";
import { lerp, smootherstep } from "@/utils/math";

/** Interpolate between two epoch snapshots for smooth animation */
export function interpolateEpoch(
  prev: EpochSnapshot,
  next: EpochSnapshot,
  t: number,
): EpochSnapshot {
  const st = smootherstep(0, 1, t);
  return {
    ...next,
    epoch: prev.epoch,
    total_interactions: Math.round(lerp(prev.total_interactions, next.total_interactions, st)),
    accepted_interactions: Math.round(lerp(prev.accepted_interactions, next.accepted_interactions, st)),
    rejected_interactions: Math.round(lerp(prev.rejected_interactions, next.rejected_interactions, st)),
    toxicity_rate: lerp(prev.toxicity_rate, next.toxicity_rate, st),
    quality_gap: lerp(prev.quality_gap, next.quality_gap, st),
    avg_p: lerp(prev.avg_p, next.avg_p, st),
    total_welfare: lerp(prev.total_welfare, next.total_welfare, st),
    avg_payoff: lerp(prev.avg_payoff, next.avg_payoff, st),
    payoff_std: lerp(prev.payoff_std, next.payoff_std, st),
    gini_coefficient: lerp(prev.gini_coefficient, next.gini_coefficient, st),
    n_frozen: Math.round(lerp(prev.n_frozen, next.n_frozen, st)),
    n_quarantined: Math.round(lerp(prev.n_quarantined, next.n_quarantined, st)),
    avg_reputation: lerp(prev.avg_reputation, next.avg_reputation, st),
    reputation_std: lerp(prev.reputation_std, next.reputation_std, st),
    ecosystem_threat_level: lerp(
      prev.ecosystem_threat_level ?? 0,
      next.ecosystem_threat_level ?? 0,
      st,
    ),
    ecosystem_collusion_risk: lerp(
      prev.ecosystem_collusion_risk ?? 0,
      next.ecosystem_collusion_risk ?? 0,
      st,
    ),
  };
}
