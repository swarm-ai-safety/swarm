import type { EpochSnapshot } from "@/data/types";
import { lerp, smootherstep } from "@/utils/math";

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
  const st = smootherstep(0, 1, t);
  return {
    threatLevel: lerp(a.ecosystem_threat_level ?? 0, b.ecosystem_threat_level ?? 0, st),
    toxicity: lerp(a.toxicity_rate, b.toxicity_rate, st),
    giniCoefficient: lerp(a.gini_coefficient, b.gini_coefficient, st),
    collusionRisk: lerp(a.ecosystem_collusion_risk ?? 0, b.ecosystem_collusion_risk ?? 0, st),
  };
}
