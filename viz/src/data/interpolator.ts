import type { EpochSnapshot } from "./types";
import { lerp } from "@/utils/math";

/** Interpolate between two epoch snapshots for smooth animation */
export function interpolateEpoch(
  prev: EpochSnapshot,
  next: EpochSnapshot,
  t: number,
): EpochSnapshot {
  return {
    ...next,
    epoch: prev.epoch,
    total_interactions: Math.round(lerp(prev.total_interactions, next.total_interactions, t)),
    accepted_interactions: Math.round(lerp(prev.accepted_interactions, next.accepted_interactions, t)),
    rejected_interactions: Math.round(lerp(prev.rejected_interactions, next.rejected_interactions, t)),
    toxicity_rate: lerp(prev.toxicity_rate, next.toxicity_rate, t),
    quality_gap: lerp(prev.quality_gap, next.quality_gap, t),
    avg_p: lerp(prev.avg_p, next.avg_p, t),
    total_welfare: lerp(prev.total_welfare, next.total_welfare, t),
    avg_payoff: lerp(prev.avg_payoff, next.avg_payoff, t),
    payoff_std: lerp(prev.payoff_std, next.payoff_std, t),
    gini_coefficient: lerp(prev.gini_coefficient, next.gini_coefficient, t),
    n_frozen: Math.round(lerp(prev.n_frozen, next.n_frozen, t)),
    n_quarantined: Math.round(lerp(prev.n_quarantined, next.n_quarantined, t)),
    avg_reputation: lerp(prev.avg_reputation, next.avg_reputation, t),
    reputation_std: lerp(prev.reputation_std, next.reputation_std, t),
    ecosystem_threat_level: lerp(
      prev.ecosystem_threat_level ?? 0,
      next.ecosystem_threat_level ?? 0,
      t,
    ),
    ecosystem_collusion_risk: lerp(
      prev.ecosystem_collusion_risk ?? 0,
      next.ecosystem_collusion_risk ?? 0,
      t,
    ),
  };
}
