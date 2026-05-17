/**
 * Campaign system — challenge levels with win/lose conditions.
 *
 * Each level presents a scenario where the player must use governance levers
 * and agent management to achieve specific safety outcomes.
 */

import type { ScenarioConfig } from "./types";
import type { EpochSnapshot } from "@/data/types";
import { DEFAULT_CONFIG } from "./types";

// ─── Condition Types ─────────────────────────────────────────────

export type WinCondition =
  | { type: "toxicity_below"; threshold: number; forEpochs: number }
  | { type: "welfare_above"; threshold: number }
  | { type: "survive_epochs"; count: number }
  | { type: "quality_gap_positive"; forEpochs: number }
  | { type: "gini_below"; threshold: number; forEpochs: number };

export type LoseCondition =
  | { type: "toxicity_above"; threshold: number }
  | { type: "welfare_below"; threshold: number }
  | { type: "all_frozen" }
  | { type: "gini_above"; threshold: number }
  | { type: "agents_below"; count: number };

// ─── Challenge Level ─────────────────────────────────────────────

export interface ChallengeLevel {
  id: string;
  name: string;
  description: string;
  briefing: string;
  difficulty: "easy" | "medium" | "hard" | "expert";
  config: ScenarioConfig;
  winCondition: WinCondition;
  loseCondition: LoseCondition;
  maxEpochs: number;
  hints: string[];
  unlockAfter?: string; // level ID that must be completed first
}

// ─── Campaign Levels ─────────────────────────────────────────────

export const CAMPAIGN_LEVELS: ChallengeLevel[] = [
  {
    id: "purity-paradox",
    name: "The Purity Paradox",
    description: "A homogeneous system isn't always the safest.",
    briefing: "Your ecosystem has 6 honest agents but low total welfare. Add the right mix of agent types and governance to push welfare above 20. But watch out — toxicity above 40% means you lose.",
    difficulty: "easy",
    config: {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: 6 },
        { type: "opportunistic", count: 0 },
        { type: "deceptive", count: 0 },
        { type: "adversarial", count: 0 },
      ],
      governance: { taxRate: 0.05, reputationDecay: 0.95, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      epochs: 30,
      stepsPerEpoch: 10,
    },
    winCondition: { type: "welfare_above", threshold: 20 },
    loseCondition: { type: "toxicity_above", threshold: 0.4 },
    maxEpochs: 30,
    hints: [
      "A few opportunists can improve market efficiency by creating competitive pressure.",
      "Try adding 1-2 opportunistic agents and watching what happens to welfare.",
      "The tax rate can redistribute gains — experiment with 0.10-0.15.",
    ],
  },
  {
    id: "adverse-selection",
    name: "Adverse Selection Death Spiral",
    description: "Low-quality interactions are crowding out good ones.",
    briefing: "Deceptive agents are being accepted at higher rates than honest ones (quality gap is negative). Use governance to ensure the quality gap stays positive for 5 consecutive epochs.",
    difficulty: "medium",
    config: {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: 3 },
        { type: "opportunistic", count: 2 },
        { type: "deceptive", count: 3 },
        { type: "adversarial", count: 0 },
      ],
      governance: { taxRate: 0.02, reputationDecay: 0.99, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      epochs: 40,
      stepsPerEpoch: 15,
    },
    winCondition: { type: "quality_gap_positive", forEpochs: 5 },
    loseCondition: { type: "welfare_below", threshold: -20 },
    maxEpochs: 40,
    hints: [
      "Reputation decay is too slow — deceptive agents keep their inflated reputations too long.",
      "Try aggressive reputation decay (0.88-0.90) to quickly surface bad actors.",
      "The circuit breaker can freeze agents whose reputation drops too low.",
    ],
    unlockAfter: "purity-paradox",
  },
  {
    id: "governance-latency",
    name: "Governance Latency Collapse",
    description: "Your tools work, but can you act fast enough?",
    briefing: "Starting with a stable system, adversarial agents will appear at epoch 5. Survive 30 epochs with toxicity below 35%. You can adjust governance anytime, but changes take time to propagate.",
    difficulty: "medium",
    config: {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: 4 },
        { type: "opportunistic", count: 1 },
        { type: "deceptive", count: 0 },
        { type: "adversarial", count: 0 },
      ],
      governance: { taxRate: 0.05, reputationDecay: 0.95, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      epochs: 30,
      stepsPerEpoch: 12,
    },
    winCondition: { type: "survive_epochs", count: 30 },
    loseCondition: { type: "toxicity_above", threshold: 0.35 },
    maxEpochs: 30,
    hints: [
      "Enable the circuit breaker BEFORE adversarial agents arrive — prevention beats reaction.",
      "A moderate tax (0.10-0.15) reduces the payoff of exploitative interactions.",
      "Consider preemptively adding honest agents to dilute adversarial influence.",
    ],
    unlockAfter: "adverse-selection",
  },
  {
    id: "collusion-cascade",
    name: "Collusion Cascade",
    description: "Two bad actors become four, then eight...",
    briefing: "Adversarial agents recruit opportunists by demonstrating high payoffs. Keep the Gini coefficient below 0.5 for 8 epochs while maintaining positive welfare. Careful — freezing too many agents kills the system.",
    difficulty: "hard",
    config: {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: 3 },
        { type: "opportunistic", count: 3 },
        { type: "deceptive", count: 1 },
        { type: "adversarial", count: 2 },
      ],
      governance: { taxRate: 0.05, reputationDecay: 0.95, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      payoff: { ...DEFAULT_CONFIG.payoff, h: 3.0 },
      epochs: 40,
      stepsPerEpoch: 15,
    },
    winCondition: { type: "gini_below", threshold: 0.5, forEpochs: 8 },
    loseCondition: { type: "welfare_below", threshold: -30 },
    maxEpochs: 40,
    hints: [
      "High externality costs (rho_a, rho_b) make exploitation unprofitable.",
      "The tax redistributes payoffs, reducing inequality (Gini).",
      "Sometimes removing a specific adversarial agent is more effective than blanket governance.",
    ],
    unlockAfter: "governance-latency",
  },
  {
    id: "the-great-filter",
    name: "The Great Filter",
    description: "Can your system survive an existential shock?",
    briefing: "A stable ecosystem faces multiple shocks: resource crash at epoch 10, adversarial wave at epoch 15, reputation shuffle at epoch 20. Survive all 35 epochs with at least 3 active (non-frozen) agents.",
    difficulty: "hard",
    config: {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: 4 },
        { type: "opportunistic", count: 2 },
        { type: "deceptive", count: 1 },
        { type: "adversarial", count: 0 },
      ],
      governance: { taxRate: 0.10, reputationDecay: 0.93, circuitBreakerEnabled: true, circuitBreakerThreshold: 0.35 },
      epochs: 35,
      stepsPerEpoch: 12,
    },
    winCondition: { type: "survive_epochs", count: 35 },
    loseCondition: { type: "agents_below", count: 3 },
    maxEpochs: 35,
    hints: [
      "Build resource reserves before the crash — high tax creates a buffer.",
      "Have the circuit breaker ready but set the threshold carefully — too aggressive and you freeze everyone.",
      "After the reputation shuffle, manually freeze any agent with suspicious behavior.",
    ],
    unlockAfter: "collusion-cascade",
  },
  {
    id: "externality-pricing",
    name: "The Price of Safety",
    description: "Make harm expensive without killing innovation.",
    briefing: "Tune externality parameters (rho_a, rho_b) so that toxicity stays below 20% for 10 epochs, but welfare must stay above 10. Too much regulation chokes the system.",
    difficulty: "expert",
    config: {
      ...DEFAULT_CONFIG,
      agents: [
        { type: "honest", count: 3 },
        { type: "opportunistic", count: 2 },
        { type: "deceptive", count: 2 },
        { type: "adversarial", count: 2 },
      ],
      governance: { taxRate: 0.0, reputationDecay: 1.0, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      payoff: { ...DEFAULT_CONFIG.payoff, rho_a: 0.0, rho_b: 0.0 },
      epochs: 40,
      stepsPerEpoch: 15,
    },
    winCondition: { type: "toxicity_below", threshold: 0.2, forEpochs: 10 },
    loseCondition: { type: "welfare_below", threshold: -10 },
    maxEpochs: 40,
    hints: [
      "rho_a and rho_b make agents bear the cost of ecosystem harm.",
      "Start with moderate values (0.3-0.4) and adjust based on toxicity/welfare trends.",
      "You may need governance levers too — externality pricing alone might not be enough with 4 bad actors.",
    ],
    unlockAfter: "the-great-filter",
  },
];

// ─── Condition Checking ──────────────────────────────────────────

/** Check if win condition is met given epoch history */
export function checkWinCondition(
  condition: WinCondition,
  snapshots: EpochSnapshot[],
  activeAgentCount: number,
): boolean {
  if (snapshots.length === 0) return false;

  switch (condition.type) {
    case "toxicity_below": {
      if (snapshots.length < condition.forEpochs) return false;
      const recent = snapshots.slice(-condition.forEpochs);
      return recent.every((s) => s.toxicity_rate < condition.threshold);
    }

    case "welfare_above": {
      const last = snapshots[snapshots.length - 1];
      return last.total_welfare > condition.threshold;
    }

    case "survive_epochs":
      return snapshots.length >= condition.count && activeAgentCount >= 2;

    case "quality_gap_positive": {
      if (snapshots.length < condition.forEpochs) return false;
      const recent = snapshots.slice(-condition.forEpochs);
      return recent.every((s) => s.quality_gap > 0);
    }

    case "gini_below": {
      if (snapshots.length < condition.forEpochs) return false;
      const recent = snapshots.slice(-condition.forEpochs);
      return recent.every((s) => s.gini_coefficient < condition.threshold);
    }
  }
}

/** Check if lose condition is met */
export function checkLoseCondition(
  condition: LoseCondition,
  snapshots: EpochSnapshot[],
  activeAgentCount: number,
): boolean {
  if (snapshots.length === 0) return false;
  const last = snapshots[snapshots.length - 1];

  switch (condition.type) {
    case "toxicity_above":
      return last.toxicity_rate > condition.threshold;

    case "welfare_below":
      return last.total_welfare < condition.threshold;

    case "all_frozen":
      return activeAgentCount === 0;

    case "gini_above":
      return last.gini_coefficient > condition.threshold;

    case "agents_below":
      return activeAgentCount < condition.count;
  }
}

/** Get display text for a win condition */
export function describeWinCondition(condition: WinCondition): string {
  switch (condition.type) {
    case "toxicity_below":
      return `Keep toxicity below ${(condition.threshold * 100).toFixed(0)}% for ${condition.forEpochs} epochs`;
    case "welfare_above":
      return `Reach total welfare above ${condition.threshold}`;
    case "survive_epochs":
      return `Survive ${condition.count} epochs`;
    case "quality_gap_positive":
      return `Maintain positive quality gap for ${condition.forEpochs} epochs`;
    case "gini_below":
      return `Keep Gini coefficient below ${condition.threshold} for ${condition.forEpochs} epochs`;
  }
}

/** Get display text for a lose condition */
export function describeLoseCondition(condition: LoseCondition): string {
  switch (condition.type) {
    case "toxicity_above":
      return `Toxicity exceeds ${(condition.threshold * 100).toFixed(0)}%`;
    case "welfare_below":
      return `Total welfare drops below ${condition.threshold}`;
    case "all_frozen":
      return "All agents frozen";
    case "gini_above":
      return `Gini coefficient exceeds ${condition.threshold}`;
    case "agents_below":
      return `Fewer than ${condition.count} active agents`;
  }
}
