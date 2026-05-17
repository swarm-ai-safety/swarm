/** Governance levers: tax, reputation decay, circuit breaker. */

import type { GovernanceConfig, SimAgentState } from "./types";

/** Apply transaction tax — returns the tax amount deducted */
export function applyTax(config: GovernanceConfig): number {
  return config.taxRate;
}

/** Decay all agent reputations toward 0.5 */
export function applyReputationDecay(agents: SimAgentState[], decay: number): void {
  for (const agent of agents) {
    if (agent.isFrozen) continue;
    // Decay toward 0.5: rep = 0.5 + decay * (rep - 0.5)
    agent.reputation = 0.5 + decay * (agent.reputation - 0.5);
  }
}

/** Check circuit breaker — returns true if tripped (should freeze low-rep agents) */
export function checkCircuitBreaker(
  config: GovernanceConfig,
  currentToxicity: number,
): boolean {
  if (!config.circuitBreakerEnabled) return false;
  return currentToxicity > config.circuitBreakerThreshold;
}

/** Freeze agents with reputation below threshold when circuit breaker trips */
export function applyCircuitBreaker(
  agents: SimAgentState[],
  repThreshold: number = 0.25,
): void {
  for (const agent of agents) {
    if (agent.reputation < repThreshold) {
      agent.isFrozen = true;
    }
  }
}
