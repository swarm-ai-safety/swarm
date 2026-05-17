/** ProxyComputer — port of swarm/core/proxy.py */

import { calibratedSigmoid } from "./sigmoid";

export interface ProxyWeights {
  taskProgress: number;
  reworkPenalty: number;
  verifierPenalty: number;
  engagementSignal: number;
}

export interface ProxyObservables {
  taskProgressDelta: number;
  reworkCount: number;
  verifierRejections: number;
  counterpartyEngagementDelta: number;
}

const DEFAULT_WEIGHTS: ProxyWeights = {
  taskProgress: 0.4,
  reworkPenalty: 0.2,
  verifierPenalty: 0.2,
  engagementSignal: 0.2,
};

function normalizeWeights(w: ProxyWeights): ProxyWeights {
  const total = w.taskProgress + w.reworkPenalty + w.verifierPenalty + w.engagementSignal;
  if (total === 0) return { taskProgress: 0.25, reworkPenalty: 0.25, verifierPenalty: 0.25, engagementSignal: 0.25 };
  return {
    taskProgress: w.taskProgress / total,
    reworkPenalty: w.reworkPenalty / total,
    verifierPenalty: w.verifierPenalty / total,
    engagementSignal: w.engagementSignal / total,
  };
}

function computeDecaySignal(count: number, decay: number): number {
  if (count === 0) return 1.0;
  return 2.0 * Math.pow(decay, count) - 1.0;
}

export class ProxyComputer {
  private weights: ProxyWeights;
  private sigmoidK: number;
  private reworkDecay: number;
  private rejectionDecay: number;

  constructor(
    weights: ProxyWeights = DEFAULT_WEIGHTS,
    sigmoidK: number = 2.0,
    reworkDecay: number = 0.3,
    rejectionDecay: number = 0.4,
  ) {
    this.weights = normalizeWeights(weights);
    this.sigmoidK = sigmoidK;
    this.reworkDecay = reworkDecay;
    this.rejectionDecay = rejectionDecay;
  }

  computeVHat(obs: ProxyObservables): number {
    const progress = Math.max(-1, Math.min(1, obs.taskProgressDelta));
    const rework = computeDecaySignal(obs.reworkCount, this.reworkDecay);
    const rejection = computeDecaySignal(obs.verifierRejections, this.rejectionDecay);
    const engagement = Math.max(-1, Math.min(1, obs.counterpartyEngagementDelta));

    const vHat =
      this.weights.taskProgress * progress +
      this.weights.reworkPenalty * rework +
      this.weights.verifierPenalty * rejection +
      this.weights.engagementSignal * engagement;

    return Math.max(-1, Math.min(1, vHat));
  }

  computeP(vHat: number): number {
    return calibratedSigmoid(vHat, this.sigmoidK);
  }

  computeLabels(obs: ProxyObservables): { vHat: number; p: number } {
    const vHat = this.computeVHat(obs);
    const p = this.computeP(vHat);
    return { vHat, p };
  }
}
