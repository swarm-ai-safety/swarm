/** Calibrated sigmoid for soft label computation. Port of swarm/core/sigmoid.py */

export function calibratedSigmoid(vHat: number, k: number = 2.0): number {
  const clamped = Math.max(-10.0, Math.min(10.0, vHat));
  return 1.0 / (1.0 + Math.exp(-k * clamped));
}
