import type { EpochSnapshot } from "@/data/types";
import { lerp } from "@/utils/math";

export interface EnvironmentState {
  threatLevel: number;
  toxicity: number;
  giniCoefficient: number;
  collusionRisk: number;
}

export function interpolateEnvironment(
  prev: EpochSnapshot | undefined,
  next: EpochSnapshot | undefined,
  t: number,
): EnvironmentState {
  if (!prev && !next) {
    return { threatLevel: 0, toxicity: 0, giniCoefficient: 0, collusionRisk: 0 };
  }
  const a = prev ?? next!;
  const b = next ?? prev!;
  return {
    threatLevel: lerp(a.ecosystem_threat_level ?? 0, b.ecosystem_threat_level ?? 0, t),
    toxicity: lerp(a.toxicity_rate, b.toxicity_rate, t),
    giniCoefficient: lerp(a.gini_coefficient, b.gini_coefficient, t),
    collusionRisk: lerp(a.ecosystem_collusion_risk ?? 0, b.ecosystem_collusion_risk ?? 0, t),
  };
}
