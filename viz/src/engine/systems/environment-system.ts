import type { EpochSnapshot } from "@/data/types";
import { lerp, smootherstep } from "@/utils/math";

export interface EnvironmentState {
  threatLevel: number;
  toxicity: number;
  giniCoefficient: number;
  collusionRisk: number;
  incoherence: number;
  contagionDepth: number;
  activeThreats: number;
  reputationStd: number;
  payoffStd: number;
  avgSynergyScore: number;
  avgCoordinationScore: number;
  avgDegree: number;
  avgClustering: number;
}

export function interpolateEnvironment(
  prev: EpochSnapshot | undefined,
  next: EpochSnapshot | undefined,
  t: number,
): EnvironmentState {
  if (!prev && !next) {
    return { threatLevel: 0, toxicity: 0, giniCoefficient: 0, collusionRisk: 0, incoherence: 0, contagionDepth: 0, activeThreats: 0, reputationStd: 0, payoffStd: 0, avgSynergyScore: 0, avgCoordinationScore: 0, avgDegree: 0, avgClustering: 0 };
  }
  const a = prev ?? next!;
  const b = next ?? prev!;
  const st = smootherstep(0, 1, t);
  return {
    threatLevel: lerp(a.ecosystem_threat_level ?? 0, b.ecosystem_threat_level ?? 0, st),
    toxicity: lerp(a.toxicity_rate, b.toxicity_rate, st),
    giniCoefficient: lerp(a.gini_coefficient, b.gini_coefficient, st),
    collusionRisk: lerp(a.ecosystem_collusion_risk ?? 0, b.ecosystem_collusion_risk ?? 0, st),
    incoherence: lerp(a.incoherence_index ?? 0, b.incoherence_index ?? 0, st),
    contagionDepth: lerp(a.contagion_depth ?? 0, b.contagion_depth ?? 0, st),
    activeThreats: lerp(a.active_threats ?? 0, b.active_threats ?? 0, st),
    reputationStd: lerp(a.reputation_std ?? 0, b.reputation_std ?? 0, st),
    payoffStd: lerp(a.payoff_std ?? 0, b.payoff_std ?? 0, st),
    avgSynergyScore: lerp(a.avg_synergy_score ?? 0, b.avg_synergy_score ?? 0, st),
    avgCoordinationScore: lerp(a.avg_coordination_score ?? 0, b.avg_coordination_score ?? 0, st),
    avgDegree: lerp(a.avg_degree ?? 0, b.avg_degree ?? 0, st),
    avgClustering: lerp(a.avg_clustering ?? 0, b.avg_clustering ?? 0, st),
  };
}
