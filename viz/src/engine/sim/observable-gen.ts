/** Generate observables from agent type + reputation + noise. */

import type { ProxyObservables } from "./proxy";
import type { AgentProfile } from "./agents";
import { gaussianRandom } from "./rng";

export function generateObservables(
  profile: AgentProfile,
  reputation: number,
  rng: () => number,
): ProxyObservables {
  const noise = gaussianRandom(rng);

  // Progress influenced by reputation (good rep → slightly better progress)
  const progressDelta = Math.max(
    -1,
    Math.min(1, profile.progressMean + profile.progressStd * noise + (reputation - 0.5) * 0.1),
  );

  // Rework: Poisson-ish draw based on rework probability
  const reworkCount = rng() < profile.reworkProb ? (rng() < profile.reworkProb ? 2 : 1) : 0;

  // Verifier rejections
  const verifierRejections = rng() < profile.rejectionProb ? (rng() < profile.rejectionProb ? 2 : 1) : 0;

  // Engagement
  const engagementDelta = Math.max(
    -1,
    Math.min(1, profile.engagementMean + 0.2 * gaussianRandom(rng) + (reputation - 0.5) * 0.15),
  );

  return {
    taskProgressDelta: progressDelta,
    reworkCount,
    verifierRejections,
    counterpartyEngagementDelta: engagementDelta,
  };
}
