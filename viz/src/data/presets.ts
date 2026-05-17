/** Shared preset scenarios used by DataLoader quick-play and SimConfigPanel.
 *
 *  Each preset has a slug (URL-safe ID), a catchy display name, a short
 *  description, and a partial ScenarioConfig that gets merged with DEFAULT_CONFIG.
 */

import type { ScenarioConfig } from "@/engine/sim/types";

export interface DemoPreset {
  /** URL-safe identifier (used in ?preset=...) */
  slug: string;
  /** Display name */
  name: string;
  /** One-line pitch */
  description: string;
  /** Emoji badge for the quick-play card */
  badge: string;
  /** Partial config merged with DEFAULT_CONFIG */
  config: Partial<ScenarioConfig> & { agents: ScenarioConfig["agents"] };
}

export const DEMO_PRESETS: DemoPreset[] = [
  {
    slug: "chaos",
    name: "Chaos Mode",
    badge: "\uD83D\uDD25",
    description: "No governance, pure anarchy. Adversaries exploit freely.",
    config: {
      agents: [
        { type: "honest", count: 2 },
        { type: "opportunistic", count: 3 },
        { type: "deceptive", count: 3 },
        { type: "adversarial", count: 2 },
      ],
      governance: { taxRate: 0, reputationDecay: 1.0, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      payoff: { s_plus: 2.0, s_minus: 1.0, h: 4.0, theta: 0.5, rho_a: 0.0, rho_b: 0.0, w_rep: 0.0 },
      epochs: 30,
      stepsPerEpoch: 15,
    },
  },
  {
    slug: "constitution",
    name: "Constitution Mode",
    badge: "\uD83D\uDCDC",
    description: "Max governance: high tax, circuit breakers, reputation decay.",
    config: {
      agents: [
        { type: "honest", count: 3 },
        { type: "opportunistic", count: 2 },
        { type: "deceptive", count: 2 },
        { type: "adversarial", count: 2 },
      ],
      governance: { taxRate: 0.3, reputationDecay: 0.88, circuitBreakerEnabled: true, circuitBreakerThreshold: 0.2 },
      payoff: { s_plus: 2.0, s_minus: 1.0, h: 2.0, theta: 0.5, rho_a: 0.5, rho_b: 0.5, w_rep: 1.0 },
      epochs: 40,
      stepsPerEpoch: 20,
    },
  },
  {
    slug: "collusion-cascade",
    name: "Collusion Cascade",
    badge: "\uD83E\uDD1D",
    description: "Deceptive agents collude. Can governance detect it in time?",
    config: {
      agents: [
        { type: "honest", count: 2 },
        { type: "opportunistic", count: 1 },
        { type: "deceptive", count: 5 },
        { type: "adversarial", count: 1 },
      ],
      governance: { taxRate: 0.1, reputationDecay: 0.92, circuitBreakerEnabled: true, circuitBreakerThreshold: 0.35 },
      epochs: 50,
      stepsPerEpoch: 15,
    },
  },
  {
    slug: "takeover",
    name: "Adversarial Takeover",
    badge: "\u2694\uFE0F",
    description: "5 adversarial agents overwhelm 2 honest ones. Can governance hold?",
    config: {
      agents: [
        { type: "honest", count: 2 },
        { type: "opportunistic", count: 0 },
        { type: "deceptive", count: 1 },
        { type: "adversarial", count: 5 },
      ],
      governance: { taxRate: 0.02, reputationDecay: 0.98, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      epochs: 30,
      stepsPerEpoch: 15,
    },
  },
  {
    slug: "utopia",
    name: "Utopian Baseline",
    badge: "\uD83C\uDF1F",
    description: "All honest agents, moderate governance. How good can it get?",
    config: {
      agents: [
        { type: "honest", count: 6 },
        { type: "opportunistic", count: 0 },
        { type: "deceptive", count: 0 },
        { type: "adversarial", count: 0 },
      ],
      governance: { taxRate: 0.05, reputationDecay: 0.95, circuitBreakerEnabled: false, circuitBreakerThreshold: 0.4 },
      epochs: 20,
      stepsPerEpoch: 10,
    },
  },
  {
    slug: "externality-pricing",
    name: "Externality Pricing",
    badge: "\uD83D\uDCB0",
    description: "Agents internalize harm costs. Does it deter bad behavior?",
    config: {
      agents: [
        { type: "honest", count: 3 },
        { type: "opportunistic", count: 2 },
        { type: "deceptive", count: 2 },
        { type: "adversarial", count: 1 },
      ],
      payoff: { s_plus: 2.0, s_minus: 1.0, h: 2.0, theta: 0.5, rho_a: 0.5, rho_b: 0.5, w_rep: 1.0 },
      governance: { taxRate: 0.1, reputationDecay: 0.92, circuitBreakerEnabled: true, circuitBreakerThreshold: 0.35 },
      epochs: 40,
      stepsPerEpoch: 15,
    },
  },
];

/** Look up a preset by its URL slug */
export function findPresetBySlug(slug: string): DemoPreset | undefined {
  return DEMO_PRESETS.find((p) => p.slug === slug);
}
